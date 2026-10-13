import itertools
import math
import os
import pytest
import statistics
import tempfile
import torch
import torch.nn.functional as F
import uuid
from contextlib import contextmanager
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from types import SimpleNamespace

from mcore_bridge.config import ModelConfig
from mcore_bridge.config.parser import hf_to_mcore_config
from mcore_bridge.model.mm_gpts.glm5_next import Glm5NextRMSNorm, _get_physical_cu_seqlens
from mcore_bridge.model.register import get_mcore_model
from mcore_bridge.utils import split_cp_inputs


def _glm_config():
    transformers = pytest.importorskip('transformers.models.glm5_next.configuration_glm5_next')
    text = {
        'vocab_size':
        128,
        'hidden_size':
        64,
        'intermediate_size':
        128,
        'moe_intermediate_size':
        32,
        'num_hidden_layers':
        8,
        'num_attention_heads':
        4,
        'num_key_value_heads':
        4,
        'n_routed_experts':
        8,
        'n_shared_experts':
        1,
        'num_experts_per_tok':
        2,
        'q_lora_rank':
        16,
        'kv_lora_rank':
        8,
        'qk_nope_head_dim':
        8,
        'qk_rope_head_dim':
        0,
        'v_head_dim':
        8,
        'first_k_dense_replace':
        3,
        'layer_types': [
            'linear_attention', 'linear_attention', 'linear_attention', 'deepseek_sparse_attention', 'linear_attention',
            'linear_attention', 'linear_attention', 'deepseek_sparse_attention'
        ],
        'mlp_layer_types': ['dense', 'dense', 'dense', 'sparse', 'sparse', 'sparse', 'sparse', 'sparse'],
        'linear_attn_config': {
            'num_heads': 4,
            'head_dim': 8,
            'short_conv_kernel_size': 4,
            'gate_lower_bound': -5.0,
            'kda_layers': [0, 1, 2, 4, 5, 6],
            'full_attn_layers': [3, 7],
        },
        'index_n_heads':
        2,
        'index_head_dim':
        4,
        'index_topk':
        4,
        'index_kpool':
        2,
        'index_kpool_compress':
        True,
        'index_kpool_always_select_tail':
        True,
        'indexer_types': ['full'] * 8,
        'hc_mult':
        4,
        'hc_eps':
        1e-6,
        'hc_sinkhorn_iters':
        4,
        'swiglu_limit':
        10.0,
        'scoring_func':
        'sigmoid',
        'topk_method':
        'noaux_tc',
        'norm_topk_prob':
        True,
        'routed_scaling_factor':
        2.5,
        'num_nextn_predict_layers':
        1,
    }
    return transformers.Glm5NextConfig(text_config=text)


def _tiny_glm_config(moe=False, optimized_dsa=False):
    config = _glm_config()
    text_config = config.text_config
    text_config.pad_token_id = 0
    text_config.num_hidden_layers = 2
    text_config.layer_types = ['linear_attention', 'deepseek_sparse_attention']
    text_config.mlp_layer_types = ['dense', 'sparse' if moe else 'dense']
    text_config.indexer_types = ['full', 'full']
    text_config.linear_attn_config['kda_layers'] = [0]
    text_config.linear_attn_config['full_attn_layers'] = [1]
    # The hybrid path reuses FLA's chunk KDA; head_dim=8 violates the K>=16 constraint of Triton dot.
    text_config.linear_attn_config['head_dim'] = 64
    text_config.linear_head_dim = 64
    if optimized_dsa:
        text_config.q_lora_rank = 64
        text_config.kv_lora_rank = 512
        text_config.qk_nope_head_dim = 256
        text_config.v_head_dim = 256
        text_config.index_n_heads = 4
        text_config.index_head_dim = 128
    return config


def _mcore_config(hf_config, tp=1, pp=1, ep=1, cp=1, sequence_parallel=False, dtype=torch.float32, mtp=0):
    values = hf_to_mcore_config(hf_config)
    # HF `glm5_next` resolves to the multimodal type; these are language-model-only fixtures.
    values['mcore_model_type'] = 'glm5_next'
    values.update(
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        perform_initialization=True,
        use_cpu_initialization=False,
        language_model_only=True,
        moe_grouped_gemm=True,
        overlap_p2p_comm=False,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=1,
        context_parallel_size=cp,
        sequence_parallel=sequence_parallel,
    )
    if mtp:
        values.update(mtp_num_layers=mtp, mtp_loss_scaling_factor=0.1)
    return ModelConfig(**values)


@pytest.fixture(scope='session', autouse=True)
def _distributed_session():
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@contextmanager
def _parallel_context(tp=1, pp=1, ep=1, cp=1):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required')
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    expected_world_size = tp * pp * cp * ep
    if world_size != expected_world_size:
        pytest.skip(f'requires world size {expected_world_size}')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    if world_size == 1:
        init_method = f'file:///tmp/glm5-next-{uuid.uuid4().hex}'
        torch.distributed.init_process_group('nccl', init_method=init_method, rank=0, world_size=1)
    elif not torch.distributed.is_initialized():
        # Distributed cases share the default group so a stale Gloo address is not reused from the
        # env:// rendezvous.
        torch.distributed.init_process_group('nccl')
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=1,
        context_parallel_size=cp,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    pg = ProcessGroupCollection.use_mpu_process_groups()
    assert pg.tp.size() == tp and pg.pp.size() == pp and pg.ep.size() == ep
    assert pg.cp.size() == parallel_state.get_context_parallel_world_size() == cp
    model_parallel_cuda_manual_seed(123)
    try:
        yield
    finally:
        if world_size > 1:
            torch.distributed.barrier()
        parallel_state.destroy_model_parallel()
        from mcore_bridge.bridge import gpt_bridge
        gpt_bridge.EP_PP_GROUP = gpt_bridge.EP_PP_RANK = gpt_bridge.EP_PP_SIZE = None
        if world_size == 1:
            torch.distributed.destroy_process_group()


class _LazyTensor:

    def __init__(self, tensor):
        self.tensor = tensor

    def load(self):
        return self.tensor


def _checkpoint_state(hf_model, lm_head):
    state = {}
    for key, value in hf_model.state_dict().items():
        if key.endswith('self_attn.conv1d.weight'):
            prefix = key.removesuffix('conv1d.weight')
            for name, weight in zip(('q_conv1d', 'k_conv1d', 'v_conv1d'), value.chunk(3, dim=0)):
                state[f'model.language_model.{prefix}{name}.weight'] = weight.detach()
            continue
        if key.endswith('mlp.experts.gate_up_proj'):
            prefix = key.removesuffix('gate_up_proj')
            for expert_idx, expert_weight in enumerate(value):
                gate, up = expert_weight.chunk(2, dim=0)
                state[f'model.language_model.{prefix}{expert_idx}.gate_proj.weight'] = gate.detach()
                state[f'model.language_model.{prefix}{expert_idx}.up_proj.weight'] = up.detach()
            continue
        if key.endswith('mlp.experts.down_proj'):
            prefix = key.removesuffix('down_proj')
            for expert_idx, expert_weight in enumerate(value):
                state[f'model.language_model.{prefix}{expert_idx}.down_proj.weight'] = expert_weight.detach()
            continue
        key = key.replace('.self_attn.forget_gate.', '.self_attn.')
        key = key.replace('.attn_hc.fn', '.hc_attn_fn')
        key = key.replace('.attn_hc.base', '.hc_attn_base')
        key = key.replace('.attn_hc.scale', '.hc_attn_scale')
        key = key.replace('.ffn_hc.fn', '.hc_ffn_fn')
        key = key.replace('.ffn_hc.base', '.hc_ffn_base')
        key = key.replace('.ffn_hc.scale', '.hc_ffn_scale')
        state[f'model.language_model.{key}'] = value.detach()
    state['lm_head.weight'] = lm_head.detach()
    return state


def _set_hf_model_dtype(model, dtype):
    model.to(dtype)
    if dtype == torch.float32:
        return
    fp32_names = ('conv1d.weight', 'A_log', 'dt_bias', 'attn_hc.base', 'attn_hc.scale', 'ffn_hc.base', 'ffn_hc.scale',
                  'e_score_correction_bias')
    for name, parameter in itertools.chain(model.named_parameters(), model.named_buffers()):
        if name.endswith(fp32_names):
            parameter.data = parameter.data.float()


def _build_parity_models(moe=False,
                         tp=1,
                         pp=1,
                         ep=1,
                         sequence_parallel=False,
                         dtype=torch.float32,
                         optimized_dsa=False):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

    from mcore_bridge.model.register import get_mcore_model

    torch.manual_seed(17)
    hf_config = _tiny_glm_config(moe=moe, optimized_dsa=optimized_dsa)
    hf_model = Glm5NextTextModel(hf_config.text_config).cuda().eval()
    _set_hf_model_dtype(hf_model, dtype)
    lm_head = torch.randn(
        hf_config.text_config.vocab_size, hf_config.text_config.hidden_size, device='cuda', dtype=dtype) * 0.02
    checkpoint = _checkpoint_state(hf_model, lm_head)
    config = _mcore_config(hf_config, tp=tp, pp=pp, ep=ep, sequence_parallel=sequence_parallel, dtype=dtype)
    mcore_model = get_mcore_model(config)[0].cuda().eval()
    lazy_checkpoint = {key: _LazyTensor(value) for key, value in checkpoint.items()}
    list(config.bridge._convert([mcore_model], lazy_checkpoint, '', True, 'Loading test: '))
    # `mcore_model` is the multimodal wrapper (the only registered type): the bridge needs it for
    # its `language_model.*` module_mapping, and with language_model_only its `visual` is None.
    return hf_model, lm_head, mcore_model, checkpoint, config


def _model_inputs(batch=2, sequence=6):
    # Seeded per call. Under PP each stage initialises different modules, so the CUDA generator
    # state -- and an unseeded draw -- differs per rank; the last stage would then compare
    # activations produced from another rank's tokens against its own reference logits.
    generator = torch.Generator(device='cuda').manual_seed(23)
    input_ids = torch.randint(1, 128, (batch, sequence), device='cuda', generator=generator)
    position_ids = torch.arange(sequence, device='cuda').unsqueeze(0).expand(batch, -1)
    attention_mask = torch.triu(torch.ones(batch, 1, sequence, sequence, device='cuda', dtype=torch.bool), diagonal=1)
    return input_ids, position_ids, attention_mask


def _hf_logits(hf_model, lm_head, input_ids):
    attention_mask = torch.ones_like(input_ids)
    hidden = hf_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    return F.linear(hidden, lm_head)


def test_glm5_parser_preserves_heterogeneous_schedules():
    parsed = hf_to_mcore_config(_glm_config())
    # Derived from text_config.model_type; no longer normalized to the registered name.
    assert parsed['llm_model_type'] == 'glm5_next_text'
    assert parsed['num_layers'] == 16
    assert parsed['hybrid_layer_pattern'] == 'K-K-K-DEKEKEKEDE'
    assert parsed['linear_attention_freq'] == '[1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0]'
    assert parsed['moe_layer_freq'] == '[0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1]'
    assert parsed['position_embedding_type'] == 'none'
    assert parsed['qk_pos_emb_head_dim'] == 0
    assert parsed['linear_lower_bound'] == -5.0
    assert parsed['index_kpool'] == 2
    assert parsed['moe_router_load_balancing_type'] == 'none'
    # The fp32 MoE router comes from ModelConfig's own default; restating it in the parser
    # would be a second source of truth that can drift.
    assert 'moe_router_dtype' not in parsed
    assert parsed.get('mtp_num_layers') is None


def test_glm5_registers_only_the_composite_model_type():
    """HF `glm5_next` is the composite type; its sub-config types are not registered separately.

    Same convention as qwen4_exp / qwen3_5 / glm4v: one multimodal mcore type, and training
    without the vision tower is `--language_model_only`, not another model type.
    """
    import mcore_bridge.model
    from mcore_bridge.model.register import get_mcore_model_type, get_model_meta

    assert get_mcore_model_type('glm5_next') == 'glm5_next'
    assert get_model_meta('glm5_next').is_multimodal
    # `glm5_next_text` is the text sub-config's own model_type; deliberately not a registry key.
    assert get_mcore_model_type('glm5_next_text') is None


def test_glm5_config_wires_the_patched_dev_guard(monkeypatch):
    """ModelConfig must actually reach `require_glm5_hybrid` for this family.

    The guard keys on `hf_model_type`: `llm_model_type` is derived from `text_config.model_type`,
    so it reads `glm5_next_text` even for the official composite checkpoint. Keying on it disables
    the guard silently, and a user on unpatched Megatron then gets an obscure failure deep in the
    hybrid stack instead of the "requires Megatron dev with the #7054 patch" ImportError.
    """
    import mcore_bridge.model.mm_gpts.glm5_next as glm5_next

    calls = []
    monkeypatch.setattr(glm5_next, 'require_glm5_hybrid', lambda: calls.append(True))

    with _parallel_context():
        config = _mcore_config(_tiny_glm_config(moe=True))

        assert calls == [True]
        assert config.hf_model_type == 'glm5_next'
        assert config.llm_model_type == 'glm5_next_text'
        # GLM's routed experts are neither HF-grouped nor gate_up concatenated; the branch is keyed
        # on the HF model type (`bridge.model_type`), not on `llm_model_type`.
        assert config.bridge._get_hf_experts_attr() == (False, False)


def glm5_kda_recurrent(query, key, value, decay, beta, initial_state=None):
    """Mirrors transformers `recurrent_kimi_delta_attention` (single-token decode path).

    Line-for-line: fp32 cast, then l2norm on q/k (+eps inside the sqrt, matching FLA), then
    the `head_dim**-0.5` query scale, then the delta-rule recurrence. Test-only oracle: the
    training path is Megatron dev's fused KDA, which forbids inference/cache entirely.
    """
    input_dtype = query.dtype
    query = query.float()
    key = key.float()
    query = query / torch.sqrt(query.square().sum(dim=-1, keepdim=True) + 1e-6)
    key = key / torch.sqrt(key.square().sum(dim=-1, keepdim=True) + 1e-6)
    query = query * query.shape[-1]**-0.5
    value = value.float()
    decay = decay.float()
    beta = beta.float()
    batch, _, num_heads, key_head_dim = query.shape
    value_head_dim = value.shape[-1]
    if initial_state is None:
        state = query.new_zeros(batch, num_heads, key_head_dim, value_head_dim)
    else:
        state = initial_state.float()
    outputs = []
    for token_idx in range(query.shape[1]):
        q_i = query[:, token_idx]
        k_i = key[:, token_idx]
        v_i = value[:, token_idx]
        state = state * decay[:, token_idx].exp().unsqueeze(-1)
        kv_mem = (state * k_i.unsqueeze(-1)).sum(dim=-2)
        delta = (v_i - kv_mem) * beta[:, token_idx].unsqueeze(-1)
        state = state + k_i.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * q_i.unsqueeze(-1)).sum(dim=-2))
    return torch.stack(outputs, dim=1).to(input_dtype), state


def glm5_kpool_indices(query, key, head_weights, gate, ape, topk, kpool):
    """Torch reference for the k-pool indexer selection.

    Mirrors transformers `Glm5NextTextIndexer.forward` + `get_pooled_states` +
    `append_visible_tail`, specialised to a single unpadded sequence: pooling starts at the
    first visible token (HF's `first_key`), only complete pools are scored (HF's "a pool is
    selectable only if its final token is visible"), and the incomplete causal tail is always
    appended. Test-only oracle: the hybrid path runs Megatron dev's fused kernel, and the
    former Triton kernel this once bit-compared was removed with the legacy GPTModel path.

    O(seq_len) python loop -- reference only.
    """
    seq_len, _, head_dim = query.shape
    width = min(seq_len, topk + kpool - 1)
    result = torch.full((seq_len, width), -1, dtype=torch.long, device=query.device)
    scale = head_dim**-0.5
    for token_idx in range(seq_len):
        visible = token_idx + 1
        if visible <= topk:
            result[token_idx, :visible] = torch.arange(visible, device=query.device)
            continue
        complete_pools = visible // kpool
        pool_key = key[:complete_pools * kpool].view(complete_pools, kpool, head_dim)
        pool_gate = gate[:complete_pools * kpool].view(complete_pools, kpool, head_dim)
        pool_weight = torch.softmax((pool_gate.float() + ape.float()).transpose(0, 1), dim=0).transpose(0, 1)
        # HF masks incomplete pools with -inf and `nan_to_num`s the softmax; only complete
        # pools reach this point, so no masking is needed. HF then casts the probabilities
        # back to the key dtype before the weighted sum, this stays in fp32.
        pooled = (pool_key.float() * pool_weight).sum(dim=1)
        scores = F.relu(torch.einsum('hd,pd->hp', query[token_idx].float(), pooled) * scale)
        scores = torch.einsum('h,hp->p', head_weights[token_idx].float(), scores)
        num_selected = min(topk // kpool, complete_pools)
        pools = torch.topk(scores, num_selected, sorted=False).indices
        selected = (pools[:, None] * kpool + torch.arange(kpool, device=query.device)).flatten()
        tail = torch.arange(complete_pools * kpool, visible, device=query.device)
        selected = torch.cat((selected, tail))
        result[token_idx, :selected.numel()] = selected
    return result


def test_glm5_kda_recurrent_matches_token_reference():
    torch.manual_seed(7)
    query = torch.randn(2, 6, 3, 4)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    decay = -torch.rand_like(query) * 4
    beta = torch.sigmoid(torch.randn(2, 6, 3))
    actual, actual_state = glm5_kda_recurrent(query, key, value, decay, beta)

    q = query.float() / torch.sqrt(query.float().square().sum(dim=-1, keepdim=True) + 1e-6)
    k = key.float() / torch.sqrt(key.float().square().sum(dim=-1, keepdim=True) + 1e-6)
    q = q * query.shape[-1]**-0.5
    state = torch.zeros(2, 3, 4, 4)
    expected = []
    for token_idx in range(query.shape[1]):
        state *= decay[:, token_idx].float().exp().unsqueeze(-1)
        memory = (state * k[:, token_idx].unsqueeze(-1)).sum(dim=-2)
        delta = (value[:, token_idx].float() - memory) * beta[:, token_idx].unsqueeze(-1)
        state += k[:, token_idx].unsqueeze(-1) * delta.unsqueeze(-2)
        expected.append((state * q[:, token_idx].unsqueeze(-1)).sum(dim=-2))
    expected = torch.stack(expected, dim=1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_state, state)


def test_glm5_kpool_keeps_causal_tail_and_complete_pools():
    torch.manual_seed(11)
    query = torch.randn(9, 2, 4)
    key = torch.randn(9, 4)
    weights = torch.randn(9, 2)
    gate = torch.randn(9, 4)
    ape = torch.randn(2, 4)
    selected = glm5_kpool_indices(query, key, weights, gate, ape, topk=4, kpool=2)
    for token_idx, row in enumerate(selected):
        valid = row[row >= 0]
        assert (valid <= token_idx).all()
        if token_idx < 4:
            torch.testing.assert_close(valid, torch.arange(token_idx + 1))
        if (token_idx + 1) % 2:
            assert token_idx in valid


def test_glm5_rejects_inference_contexts():
    from mcore_bridge.model.mm_gpts.glm5_next import Glm5NextHybridModel

    model = object.__new__(Glm5NextHybridModel)
    torch.nn.Module.__init__(model)
    # MegatronModule.__call__ reads self.config.cuda_graph_impl first to decide on TE cudagraphs, so a
    # bare instance needs a config, otherwise it never reaches the inference guard inside forward.
    model.config = SimpleNamespace(cuda_graph_impl='none')
    input_ids = torch.zeros(1, 1, dtype=torch.long)
    position_ids = torch.zeros(1, 1, dtype=torch.long)
    with pytest.raises(NotImplementedError, match='inference'):
        model(input_ids, position_ids, None, inference_context=object())
    with pytest.raises(NotImplementedError, match='inference'):
        model(input_ids, position_ids, None, inference_params=object())


def test_glm5_uses_padded_boundaries_for_physical_thd_slices():
    packed = PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=torch.tensor([0, 2, 4], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 3, 6], dtype=torch.int32),
    )
    cu_seqlens = _get_physical_cu_seqlens(packed)
    torch.testing.assert_close(cu_seqlens, packed.cu_seqlens_q_padded)
    plain = PackedSeqParams(qkv_format='thd', cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32))
    torch.testing.assert_close(_get_physical_cu_seqlens(plain), plain.cu_seqlens_q)
    assert _get_physical_cu_seqlens(None) is None


def test_glm5_mtp_layer_is_filtered_only_when_mtp_disabled():
    from mcore_bridge.model.mm_gpts.glm5_next import Glm5NextBridge

    state = {
        'model.language_model.layers.44.input_layernorm.weight': 1,
        'model.language_model.layers.45.input_layernorm.weight': 2,
        'language_model.layers.45.mlp.down_proj.weight': 3,
        'layers.45.self_attn.q_proj.weight': 4,
    }

    # MTP off: the extra decoder layer (index num_hidden_layers) has nowhere to go -> filtered.
    bridge = object.__new__(Glm5NextBridge)
    bridge.config = SimpleNamespace(num_layers=90, mtp_num_layers=0)
    converted = bridge._convert_hf_state_dict(dict(state), True)
    assert converted == {'model.language_model.layers.44.input_layernorm.weight': 1}

    # MTP on: the layer-45 tensors are kept for _convert_mtp_layer.
    bridge_mtp = object.__new__(Glm5NextBridge)
    bridge_mtp.config = SimpleNamespace(num_layers=90, mtp_num_layers=1)
    kept = bridge_mtp._convert_hf_state_dict(dict(state), True)
    assert 'model.language_model.layers.45.input_layernorm.weight' in kept


def _build_glm_mtp_model(mtp=1, moe=True, dtype=torch.float32, tp=1, pp=1, ep=1, cp=1, sequence_parallel=False, seed=5):
    torch.manual_seed(seed)
    hf_config = _tiny_glm_config(moe=moe, optimized_dsa=cp > 1 or sequence_parallel)
    config = _mcore_config(
        hf_config,
        tp=tp,
        pp=pp,
        ep=ep,
        cp=cp,
        sequence_parallel=sequence_parallel,
        dtype=dtype,
        mtp=mtp,
    )
    model = get_mcore_model(config)[0].cuda()
    return model, config


def _lm_of(model):
    return model.language_model if hasattr(model, 'language_model') else model


def test_glm5_mtp_builds_non_mhc_eh_proj_head():
    """GLM's MTP head is non-mHC despite the mHC backbone: fused eh_proj, no hyper-connected inner block."""
    with _parallel_context():
        model, config = _build_glm_mtp_model(mtp=1)
        lm = _lm_of(model)
        assert config.mtp_hybrid_override_pattern == 'DE'
        assert lm.mtp_process and len(lm.mtp.layers) == 1
        mtp_layer = lm.mtp.layers[0]
        assert type(mtp_layer).__name__ == 'Glm5NextMultiTokenPredictionLayer'
        assert mtp_layer.mhc_enabled is False
        assert mtp_layer.eh_proj is not None and mtp_layer.e_proj is None and mtp_layer.h_proj is None
        inner = mtp_layer.mtp_model_layer
        assert inner.is_mtp_layer and len(inner.layers) == 2
        # No hyper-connection wrapping on the MTP inner sublayers (the checkpoint carries no hc_* there).
        assert all(not hasattr(sub, 'hyper_connection') for sub in inner.layers)
        # The outer decoder must be the stack that suppresses the multi-stream tensor.
        assert type(lm.decoder).__name__ == 'Glm5NextHybridStack'


def test_glm5_mtp_bridge_roundtrip_matches_checkpoint_layout():
    """mcore -> hf export of the MTP head matches the real GLM-5.3-Flash key layout and re-imports bit-exactly."""
    with _parallel_context():
        model_a, config = _build_glm_mtp_model(mtp=1, seed=5)
        exported = _export_to_hf(config, model_a)
        hf_idx = config.num_layers // 2  # MTP head lives at decoder-layer index num_hidden_layers
        p = f'model.language_model.layers.{hf_idx}.'
        for key in ('enorm.weight', 'hnorm.weight', 'eh_proj.weight', 'shared_head.norm.weight',
                    'input_layernorm.weight', 'post_attention_layernorm.weight', 'mlp.gate.weight',
                    'mlp.gate.e_score_correction_bias', 'self_attn.q_a_proj.weight',
                    'self_attn.kv_a_proj_with_mqa.weight', 'self_attn.indexer.wq_b.weight'):
            assert p + key in exported, f'MTP export missing {p}{key}'

        model_b, _ = _build_glm_mtp_model(mtp=1, seed=999)
        lazy = {key: _LazyTensor(value) for key, value in exported.items()}
        list(config.bridge._convert([model_b], lazy, '', True, 'Reloading test: '))
        sd_a = _lm_of(model_a).mtp.state_dict()
        sd_b = _lm_of(model_b).mtp.state_dict()
        assert sd_a.keys() == sd_b.keys()
        for key in sd_a:
            torch.testing.assert_close(
                sd_b[key].cpu(), sd_a[key].cpu(), atol=0, rtol=0, msg=lambda m, k=key: f'{k}: {m}')


def test_glm5_mtp_forward_backward_flows_grads_to_head():
    """The MTP loss path runs and every trainable MTP parameter receives a gradient.

    BF16 (how GLM trains): the FP32 KDA backward routes through a TileLang kernel whose nvcc
    toolchain is unrelated to MTP; BF16 uses the FLA/Triton chunk-KDA path.
    """
    with _parallel_context():
        model, _ = _build_glm_mtp_model(mtp=1, dtype=torch.bfloat16)
        lm = _lm_of(model).train()
        input_ids, position_ids, attention_mask = _model_inputs(batch=2, sequence=8)
        labels = torch.randint(1, 128, (2, 8), device='cuda')
        loss_mask = torch.ones(2, 8, device='cuda')
        out = lm(input_ids, position_ids, attention_mask, labels=labels, loss_mask=loss_mask)
        loss = out if out.dim() == 0 else out.float().mean()
        assert torch.isfinite(loss).item()
        loss.backward()
        nograd = [n for n, p in lm.mtp.named_parameters() if p.requires_grad and p.grad is None]
        assert not nograd, f'MTP params without grad: {nograd}'
        head = dict(lm.mtp.named_parameters())
        for probe in ('layers.0.eh_proj.weight', 'layers.0.enorm.weight', 'layers.0.hnorm.weight',
                      'layers.0.final_layernorm.weight'):
            assert probe in head, f'MTP head missing {probe}'
            assert head[probe].grad is not None and torch.isfinite(head[probe].grad.float()).all()


def test_glm5_mtp_cp2_packed_forward_backward():
    """KPool gates must follow the same CP gather/reorder as their indexer keys."""
    if os.environ.get('GLM5_MTP_PARALLEL_TEST') != 'cp2':
        pytest.skip('run with GLM5_MTP_PARALLEL_TEST=cp2 torchrun --nproc-per-node=2')
    with _parallel_context(cp=2):
        model, _ = _build_glm_mtp_model(mtp=1, dtype=torch.bfloat16, cp=2)
        lm = _lm_of(model).train()
        input_ids, position_ids, packed_seq_params = _packed_inputs([16])
        labels = torch.randint(1, 128, input_ids.shape, device='cuda')
        loss_mask = torch.ones_like(labels, dtype=torch.bool)
        cu_seqlens = packed_seq_params.cu_seqlens_q
        position_ids = split_cp_inputs(position_ids, cu_seqlens, -1)
        labels = split_cp_inputs(labels, cu_seqlens, -1)
        loss_mask = split_cp_inputs(loss_mask, cu_seqlens, -1)

        out = model(
            input_ids,
            position_ids,
            None,
            labels=labels,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
        )
        loss = out if out.dim() == 0 else out.float().mean()
        assert torch.isfinite(loss).item()
        loss.backward()
        missing = [name for name, param in lm.mtp.named_parameters() if param.requires_grad and param.grad is None]
        assert not missing, f'MTP params without grad: {missing}'


def test_glm5_mtp_tp2_sequence_parallel_packed_forward_backward():
    """MTP rolls a TP-full padding mask, then restores the SP shard used by its MoE router."""
    if os.environ.get('GLM5_MTP_PARALLEL_TEST') != 'tp2_sp':
        pytest.skip('run with GLM5_MTP_PARALLEL_TEST=tp2_sp torchrun --nproc-per-node=2')
    pytest.importorskip('tilelang')
    with _parallel_context(tp=2):
        model, _ = _build_glm_mtp_model(
            mtp=1,
            dtype=torch.bfloat16,
            tp=2,
            sequence_parallel=True,
        )
        lm = _lm_of(model).train()
        input_ids, position_ids, packed_seq_params = _packed_inputs([16])
        packed_seq_params.seq_lens = torch.tensor([13], device='cuda', dtype=torch.int32)
        labels = torch.randint(1, 128, input_ids.shape, device='cuda')
        loss_mask = torch.ones_like(labels, dtype=torch.bool)
        loss_mask[:, 13:] = False

        out = model(
            input_ids,
            position_ids,
            None,
            labels=labels,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
        )
        loss = out if out.dim() == 0 else out.float().mean()
        assert torch.isfinite(loss).item()
        loss.backward()
        missing = [name for name, param in lm.mtp.named_parameters() if param.requires_grad and param.grad is None]
        assert not missing, f'MTP params without grad: {missing}'


def test_glm5_kda_head_sharding_matches_world8():
    if int(os.environ.get('WORLD_SIZE', '1')) != 8 or not torch.cuda.is_available():
        pytest.skip('run with torchrun --nproc-per-node=8 to validate eight-way KDA head sharding')
    torch.distributed.init_process_group('nccl')
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)
    generator = torch.Generator(device='cuda').manual_seed(19)
    query = torch.randn(1, 7, 8, 4, generator=generator, device='cuda')
    key = torch.randn(query.shape, generator=generator, device='cuda')
    value = torch.randn(query.shape, generator=generator, device='cuda')
    decay = -torch.rand(query.shape, generator=generator, device='cuda')
    beta = torch.sigmoid(torch.randn(1, 7, 8, generator=generator, device='cuda'))
    expected, _ = glm5_kda_recurrent(query, key, value, decay, beta)
    local, _ = glm5_kda_recurrent(query[:, :, rank:rank + 1], key[:, :, rank:rank + 1], value[:, :, rank:rank + 1],
                                  decay[:, :, rank:rank + 1], beta[:, :, rank:rank + 1])
    shards = [torch.empty_like(local) for _ in range(8)]
    torch.distributed.all_gather(shards, local)
    torch.testing.assert_close(torch.cat(shards, dim=2), expected)
    torch.distributed.destroy_process_group()


def test_glm5_cuda_forward_and_checkpoint_parity():
    if os.environ.get('GLM5_PARALLEL_TEST') is not None:
        pytest.skip('single-rank CUDA parity test')
    with _parallel_context():
        hf_model, lm_head, model, checkpoint, config = _build_parity_models(moe=True)
        input_ids, position_ids, attention_mask = _model_inputs()
        with torch.no_grad():
            expected = _hf_logits(hf_model, lm_head, input_ids)
            actual = model(input_ids, position_ids, attention_mask)
        torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)

        exported = dict(config.bridge.export_weights([model], target_device='cuda'))
        assert set(exported) == set(checkpoint)
        for key, expected_weight in checkpoint.items():
            torch.testing.assert_close(exported[key], expected_weight)
        assert exported['model.language_model.layers.0.self_attn.A_log'].dtype == torch.float32
        assert exported['model.language_model.layers.0.hc_attn_base'].dtype == torch.float32
        assert exported[
            'model.language_model.layers.1.self_attn.indexer.index_kpool_compress_ape'].dtype == torch.float32


def test_glm5_bf16_forward_and_gradient_parity():
    if os.environ.get('GLM5_PARALLEL_TEST') is not None:
        pytest.skip('single-rank CUDA parity test')
    with _parallel_context():
        hf_model, lm_head, model, _, _ = _build_parity_models(dtype=torch.bfloat16)
        input_ids, position_ids, attention_mask = _model_inputs()
        expected = _hf_logits(hf_model, lm_head, input_ids)
        actual = model(input_ids, position_ids, attention_mask)
        torch.testing.assert_close(actual, expected, atol=4e-3, rtol=2e-2)

        expected.float().square().mean().backward()
        actual.float().square().mean().backward()
        kda = model.language_model.decoder.layers[0].inner_layer.self_attention
        dsa = model.language_model.decoder.layers[2].inner_layer.self_attention
        hc = model.language_model.decoder.layers[0].hyper_connection
        gradient_pairs = (
            (hf_model.layers[0].self_attn.q_proj.weight.grad, kda.in_proj.weight.grad.chunk(3)[0]),
            (hf_model.layers[0].attn_hc.fn.grad, hc.mapping_proj.weight.grad),
            (hf_model.layers[1].self_attn.q_a_proj.weight.grad, dsa.linear_q_down_proj.weight.grad),
        )
        for hf_grad, mcore_grad in gradient_pairs:
            assert hf_grad is not None and mcore_grad is not None
            torch.testing.assert_close(hf_grad.float(), mcore_grad.float(), atol=2e-5, rtol=5e-2)

        assert kda.conv1d.weight.dtype == torch.float32
        assert kda.A_log.dtype == torch.float32
        assert kda.dt_bias.dtype == torch.float32
        assert kda.out_norm.weight.dtype == torch.bfloat16
        assert hc.mapping_proj.weight.dtype == torch.float32
        assert hc.bias.dtype == torch.float32
        assert dsa.core_attention.indexer.index_kpool_compress_ape.dtype == torch.bfloat16


def test_glm5_bf16_optimized_dsa_model_parity():
    pytest.importorskip('tilelang')
    if os.environ.get('GLM5_PARALLEL_TEST') is not None:
        pytest.skip('single-rank CUDA parity test')
    with _parallel_context():
        hf_model, lm_head, model, _, _ = _build_parity_models(dtype=torch.bfloat16, optimized_dsa=True)
        input_ids, position_ids, attention_mask = _model_inputs(batch=1)
        cu_seqlens = torch.tensor([0, input_ids.shape[1]], device='cuda', dtype=torch.int32)
        packed_seq_params = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=input_ids.shape[1],
            max_seqlen_kv=input_ids.shape[1],
        )
        with torch.no_grad():
            expected = _hf_logits(hf_model, lm_head, input_ids)
            actual = model(input_ids, position_ids, None, packed_seq_params=packed_seq_params)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=3e-2)


# -----------------------------------------------------------------------------------------
# padding_free / packed (thd) coverage.
#
# Production GLM-5.3 training runs with `--padding_free true`, and that is the only
# configuration that builds `packed_seq_params` -- hence the only one that reaches the fused
# Triton k-pool indexer and the TileLang SparseMLA kernels. Everything below pins both the
# numerics and the fact that the fused paths ran.
# -----------------------------------------------------------------------------------------


def _packed_inputs(lengths, vocab=128):
    """One thd row of `len(lengths)` packed sequences plus its PackedSeqParams."""
    total = sum(lengths)
    generator = torch.Generator(device='cuda').manual_seed(23)  # see _model_inputs: ranks must agree
    input_ids = torch.randint(1, vocab, (1, total), device='cuda', generator=generator)
    position_ids = torch.cat([torch.arange(length, device='cuda') for length in lengths]).unsqueeze(0)
    cu_seqlens = torch.tensor([0, *itertools.accumulate(lengths)], device='cuda', dtype=torch.int32)
    packed_seq_params = PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max(lengths),
        max_seqlen_kv=max(lengths),
    )
    return input_ids, position_ids, packed_seq_params


def _hf_logits_packed(hf_model, lm_head, input_ids, lengths):
    """HF has no packing, so run each sequence on its own and concatenate."""
    logits, offset = [], 0
    for length in lengths:
        logits.append(_hf_logits(hf_model, lm_head, input_ids[:, offset:offset + length]))
        offset += length
    return torch.cat(logits, dim=1)


@contextmanager
def _hybrid_path_probe():
    """Count which indexer / sparse-MLA implementation the DSA layers actually ran."""
    from megatron.core.transformer.experimental_attention_variant import dsa

    counts = {'kpool': 0, 'sparse_attention': 0}
    targets = {
        'kpool': (dsa, 'fused_qk_topk_kpool'),
        'sparse_attention': (dsa, '_run_sparse_attention'),
    }
    originals = {key: getattr(owner, name) for key, (owner, name) in targets.items()}
    for key, (owner, name) in targets.items():
        original = originals[key]

        def wrapper(*args, _original=original, _key=key, **kwargs):
            counts[_key] += 1
            return _original(*args, **kwargs)

        setattr(owner, name, wrapper)
    try:
        yield counts
    finally:
        for key, (owner, name) in targets.items():
            setattr(owner, name, originals[key])


def _assert_hybrid_paths_ran(counts):
    assert counts['kpool'] > 0, f'the hybrid KPool path did not run: {counts}'
    assert counts['sparse_attention'] > 0, f"dev's sparse attention did not run: {counts}"


# Lengths that are not multiples of index_kpool are the interesting case: a varlen fixture
# built from kpool-aligned lengths cannot catch a pool straddling a sequence boundary.
@pytest.mark.parametrize('lengths', [[6], [5, 7], [3, 8, 4, 6], [2, 2, 2, 2, 2, 2], [7, 1, 9]])
def test_glm5_packed_forward_parity(lengths):
    pytest.importorskip('tilelang')
    if os.environ.get('GLM5_PARALLEL_TEST') is not None:
        pytest.skip('single-rank CUDA parity test')
    with _parallel_context():
        hf_model, lm_head, model, _, _ = _build_parity_models(dtype=torch.bfloat16, optimized_dsa=True)
        input_ids, position_ids, packed_seq_params = _packed_inputs(lengths)
        with _hybrid_path_probe() as counts:
            with torch.no_grad():
                actual = model(input_ids, position_ids, None, packed_seq_params=packed_seq_params)
        expected = _hf_logits_packed(hf_model, lm_head, input_ids, lengths)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=3e-2)
        _assert_hybrid_paths_ran(counts)


def test_glm5_packed_backward_produces_finite_grads():
    pytest.importorskip('tilelang')
    if os.environ.get('GLM5_PARALLEL_TEST') is not None:
        pytest.skip('single-rank CUDA parity test')
    with _parallel_context():
        _, _, model, _, _ = _build_parity_models(dtype=torch.bfloat16, optimized_dsa=True)
        input_ids, position_ids, packed_seq_params = _packed_inputs([5, 7, 3])
        with _hybrid_path_probe() as counts:
            logits = model(input_ids, position_ids, None, packed_seq_params=packed_seq_params)
        _assert_hybrid_paths_ran(counts)
        logits.float().square().mean().backward()
        dsa = model.language_model.decoder.layers[2].inner_layer.self_attention
        for name, parameter in (('linear_q_down_proj', dsa.linear_q_down_proj.weight),
                                ('linear_kv_up_proj', dsa.linear_kv_up_proj.weight), ('linear_proj',
                                                                                      dsa.linear_proj.weight)):
            assert parameter.grad is not None, f'{name} got no gradient'
            assert torch.isfinite(parameter.grad).all(), f'{name} gradient is not finite'
            assert parameter.grad.abs().sum() > 0, f'{name} gradient is all zero'
        # HF runs the indexer under `@torch.no_grad()`, so every indexer parameter is frozen
        # and must stay out of the autograd graph entirely.
        for name, parameter in dsa.core_attention.indexer.named_parameters():
            assert not parameter.requires_grad, f'indexer.{name} should be frozen'
            assert parameter.grad is None, f'indexer.{name} unexpectedly got a gradient'
        kda = model.language_model.decoder.layers[0].inner_layer.self_attention
        for name, parameter in (('in_proj', kda.in_proj.weight), ('out_proj', kda.out_proj.weight),
                                ('out_norm', kda.out_norm.weight), ('conv1d', kda.conv1d.weight), ('A_log', kda.A_log),
                                ('dt_bias', kda.dt_bias)):
            assert parameter.grad is not None, f'kda.{name} got no gradient'
            assert torch.isfinite(parameter.grad).all(), f'kda.{name} gradient is not finite'


def test_glm5_tp2_sequence_parallel_forward_parity():
    if os.environ.get('GLM5_PARALLEL_TEST') != 'tp2':
        pytest.skip('run with GLM5_PARALLEL_TEST=tp2 torchrun --nproc-per-node=2')
    with _parallel_context(tp=2):
        hf_model, lm_head, model, _, _ = _build_parity_models(tp=2, sequence_parallel=True)
        input_ids, position_ids, attention_mask = _model_inputs()
        with torch.no_grad():
            expected = _hf_logits(hf_model, lm_head, input_ids)
            actual = model(input_ids, position_ids, attention_mask, runtime_gather_output=True)
        torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)

        # padding_free / thd phase: bf16 with the production DSA dims, so the fused Triton
        # indexer and TileLang SparseMLA run under TP+SP. Under sequence parallelism the packed
        # row is scattered over the TP group, so its total length must be a multiple of
        # tp_size -- ms-swift guarantees that by padding the row (`get_padding_to`), and the
        # fixture picks lengths that already satisfy it. 5 and 7 are still not multiples of
        # index_kpool, which is the case a kpool-aligned fixture cannot catch.
        pytest.importorskip('tilelang')
        hf_model, lm_head, model, _, _ = _build_parity_models(
            tp=2, sequence_parallel=True, dtype=torch.bfloat16, optimized_dsa=True)
        lengths = [5, 7, 4]
        input_ids, position_ids, packed_seq_params = _packed_inputs(lengths)
        with _hybrid_path_probe() as counts:
            with torch.no_grad():
                actual = model(
                    input_ids, position_ids, None, packed_seq_params=packed_seq_params, runtime_gather_output=True)
        expected = _hf_logits_packed(hf_model, lm_head, input_ids, lengths)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=3e-2)
        _assert_hybrid_paths_ran(counts)


def test_glm5_tp2_kda_sharded_state_dict_declares_global_shapes():
    """`A_log` / `dt_bias` / `conv1d.weight` must be declared TP-sharded, not replicated.

    Regression: `MegatronModule.sharded_state_dict`'s default records the *local* shard as the
    global shape for any tensor it is not told about. At TP=1 the two coincide, so a converted
    checkpoint loads and every forward/backward parity test above passes; at TP>=2 loading a
    checkpoint dies with `CheckpointingException: Global shape mismatch for loaded
    (torch.Size([4])) and expected ((2,)) tensor for key ...self_attention.A_log`. Compute never
    reads this metadata, so only a dist-checkpointing assertion catches it -- which is why the
    forward-parity tests did not.
    """
    if os.environ.get('GLM5_PARALLEL_TEST') != 'tp2':
        pytest.skip('run with GLM5_PARALLEL_TEST=tp2 torchrun --nproc-per-node=2')
    with _parallel_context(tp=2):
        _, _, model, _, _ = _build_parity_models(tp=2, sequence_parallel=True)
        kda = model.language_model.decoder.layers[0].inner_layer.self_attention
        tp_size = 2
        rank = torch.distributed.get_rank()
        sharded = kda.sharded_state_dict()
        projection_size = kda.num_key_heads * kda.key_head_dim
        expected = {
            'A_log': ((kda.num_key_heads, ), (kda.num_key_heads // tp_size, )),
            'dt_bias': ((projection_size, ), (projection_size // tp_size, )),
        }
        # dev uses three separate shards for the packed QKV/conv: check the factory output rather than
        # one contiguous split of the whole axis.
        for key in ('in_proj.weight', 'conv1d.weight'):
            parts = sharded[key].build()
            assert len(parts) == 3
            for part, name in zip(parts, ('query', 'key', 'value')):
                assert part.key == f'{key}.{name}'
                assert part.global_shape[0] == projection_size
                assert part.local_shape[0] == projection_size // tp_size
                assert part.global_offset[0] == rank * projection_size // tp_size
        for key, (global_shape, local_shape) in expected.items():
            assert key in sharded, f'{key} missing from sharded_state_dict: {sorted(sharded)}'
            assert tuple(sharded[key].global_shape) == global_shape, (
                f'{key} global_shape is {tuple(sharded[key].global_shape)}, expected {global_shape}: the TP shard '
                'axis is not declared, so dist-checkpointing records the local shard as the global shape')
            assert tuple(sharded[key].local_shape) == local_shape, (
                f'{key} local_shape is {tuple(sharded[key].local_shape)}, expected {local_shape}')
            # Each TP rank must claim its own slice; two ranks claiming offset 0 would make a
            # save silently drop half of the tensor. `global_offset` has one entry per dim and
            # only the TP axis is offset, so the trailing dims stay 0 (matters for conv1d's 3D
            # [channels, 1, kernel] weight).
            expected_offset = (rank * local_shape[0], ) + (0, ) * (len(global_shape) - 1)
            assert tuple(sharded[key].global_offset) == expected_offset, (
                f'{key} global_offset is {tuple(sharded[key].global_offset)} on rank {rank}, '
                f'expected {expected_offset}')
        # What TP does *not* split must stay replicated: `f_a_proj` / `g_a_proj` output head_dim
        # and `o_norm` normalises over head_dim, so declaring them sharded would be just as wrong.
        for key in ('f_a_proj.weight', 'g_a_proj.weight', 'out_norm.weight'):
            assert key in sharded, f'{key} missing from sharded_state_dict: {sorted(sharded)}'
            assert tuple(sharded[key].global_shape) == tuple(sharded[key].local_shape), (
                f'{key} is replicated over TP but was declared sharded: {tuple(sharded[key].global_shape)} vs '
                f'{tuple(sharded[key].local_shape)}')


def test_glm5_pp2_pipeline_schedule_forward_parity():
    if os.environ.get('GLM5_PARALLEL_TEST') != 'pp2':
        pytest.skip('run with GLM5_PARALLEL_TEST=pp2 torchrun --nproc-per-node=2')
    with _parallel_context(pp=2):
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

        hf_model, lm_head, model, _, _ = _build_parity_models(pp=2)
        input_ids, position_ids, attention_mask = _model_inputs()
        with torch.no_grad():
            expected = _hf_logits(hf_model, lm_head, input_ids)

            def forward_step(data_iterator, stage_model):
                ids, positions, mask = next(data_iterator)
                output = stage_model(ids, positions, mask)
                return output, lambda tensor, **kwargs: tensor

            outputs = get_forward_backward_func()(
                forward_step_func=forward_step,
                data_iterator=iter([(input_ids, position_ids, attention_mask)]),
                model=model,
                num_microbatches=1,
                seq_length=input_ids.shape[1],
                micro_batch_size=input_ids.shape[0],
                forward_only=True,
                collect_non_loss_data=True,
            )
        if parallel_state.is_pipeline_last_stage():
            assert len(outputs) == 1
            torch.testing.assert_close(outputs[0], expected, atol=3e-4, rtol=3e-4)
        else:
            assert outputs == []

        # padding_free / thd phase, bf16 with the production DSA dims.
        pytest.importorskip('tilelang')
        hf_model, lm_head, model, _, _ = _build_parity_models(pp=2, dtype=torch.bfloat16, optimized_dsa=True)
        lengths = [5, 7, 3]
        input_ids, position_ids, packed_seq_params = _packed_inputs(lengths)
        with _hybrid_path_probe() as counts:
            with torch.no_grad():
                expected = _hf_logits_packed(hf_model, lm_head, input_ids, lengths)

                def packed_forward_step(data_iterator, stage_model):
                    ids, positions, packed = next(data_iterator)
                    output = stage_model(ids, positions, None, packed_seq_params=packed)
                    return output, lambda tensor, **kwargs: tensor

                outputs = get_forward_backward_func()(
                    forward_step_func=packed_forward_step,
                    data_iterator=iter([(input_ids, position_ids, packed_seq_params)]),
                    model=model,
                    num_microbatches=1,
                    seq_length=input_ids.shape[1],
                    micro_batch_size=input_ids.shape[0],
                    forward_only=True,
                    collect_non_loss_data=True,
                )
        # The 2-layer fixture puts the KDA layer on stage 0 and the DSA layer on stage 1, so
        # only the last stage runs the indexer / SparseMLA.
        if parallel_state.is_pipeline_last_stage():
            assert len(outputs) == 1
            torch.testing.assert_close(outputs[0], expected, atol=2e-2, rtol=3e-2)
            _assert_hybrid_paths_ran(counts)
        else:
            assert outputs == []


def test_glm5_ep2_forward_parity():
    if os.environ.get('GLM5_PARALLEL_TEST') != 'ep2':
        pytest.skip('run with GLM5_PARALLEL_TEST=ep2 torchrun --nproc-per-node=2')
    with _parallel_context(ep=2):
        hf_model, lm_head, model, _, _ = _build_parity_models(moe=True, ep=2)
        input_ids, position_ids, attention_mask = _model_inputs()
        with torch.no_grad():
            expected = _hf_logits(hf_model, lm_head, input_ids)
            actual = model(input_ids, position_ids, attention_mask)
        torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)

        # padding_free / thd phase: packed MoE dispatch plus the fused DSA kernels.
        pytest.importorskip('tilelang')
        hf_model, lm_head, model, _, _ = _build_parity_models(moe=True, ep=2, dtype=torch.bfloat16, optimized_dsa=True)
        lengths = [5, 7, 3]
        input_ids, position_ids, packed_seq_params = _packed_inputs(lengths)
        with _hybrid_path_probe() as counts:
            with torch.no_grad():
                expected = _hf_logits_packed(hf_model, lm_head, input_ids, lengths)
                actual = model(input_ids, position_ids, None, packed_seq_params=packed_seq_params)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=3e-2)
        _assert_hybrid_paths_ran(counts)


# -----------------------------------------------------------------------------------------
# Line-by-line alignment with transformers `modeling_glm5_next.py`.
# -----------------------------------------------------------------------------------------


def test_glm5_rmsnorm_variants_match_hf_bit_exactly():
    """HF's RMSNorm casts back to the input dtype before the weight multiply; the mcore
    class must match bit-exactly."""
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextRMSNorm
    torch.manual_seed(5)
    for dtype in (torch.float32, torch.bfloat16):
        hidden = torch.randn(4, 7, dtype=dtype) * 3
        weight = torch.randn(7).to(dtype)

        plain, plain_hf = Glm5NextRMSNorm(7, 1e-5, dtype=dtype), Glm5NextTextRMSNorm(7, eps=1e-5).to(dtype)
        plain.weight.data.copy_(weight)
        plain_hf.weight.data.copy_(weight)
        torch.testing.assert_close(plain(hidden), plain_hf(hidden), atol=0, rtol=0)


def _export_to_hf(config, mcore_model):
    """Run the bridge in the mcore -> hf direction and collect the exported tensors."""
    exported = {}
    for key, value in config.bridge._convert([mcore_model], {}, '', False, 'Exporting test: '):
        exported[key] = value.load() if hasattr(value, 'load') else value
    return exported


@pytest.mark.parametrize('moe', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_glm5_bridge_hf_to_mcore_to_hf_roundtrip(moe, dtype):
    """hf -> mcore -> hf must return every original tensor bit-for-bit."""
    with _parallel_context():
        _, _, mcore_model, checkpoint, config = _build_parity_models(moe=moe, dtype=dtype)
        exported = _export_to_hf(config, mcore_model)

        missing = sorted(set(checkpoint) - set(exported))
        extra = sorted(set(exported) - set(checkpoint))
        assert not missing, f'bridge dropped {len(missing)} tensors on export, e.g. {missing[:5]}'
        assert not extra, f'bridge invented {len(extra)} tensors on export, e.g. {extra[:5]}'

        for key, original in checkpoint.items():
            actual = exported[key]
            assert actual.shape == original.shape, f'{key}: {actual.shape} != {original.shape}'
            assert actual.dtype == original.dtype, f'{key}: {actual.dtype} != {original.dtype}'
            torch.testing.assert_close(
                actual.to(original.dtype).cpu(), original.cpu(), atol=0, rtol=0, msg=lambda m, k=key: f'{k}: {m}')


def test_glm5_bridge_roundtrip_preserves_forward_logits():
    """Re-importing an exported checkpoint must reproduce identical logits."""
    with _parallel_context():
        from mcore_bridge.model.register import get_mcore_model

        _, _, mcore_model, _, config = _build_parity_models(moe=True)
        input_ids, position_ids, attention_mask = _model_inputs()
        with torch.no_grad():
            expected = mcore_model(input_ids, position_ids, attention_mask)

        exported = _export_to_hf(config, mcore_model)
        reimported = get_mcore_model(config)[0].cuda().eval()
        lazy = {key: _LazyTensor(value) for key, value in exported.items()}
        list(config.bridge._convert([reimported], lazy, '', True, 'Reloading test: '))
        with torch.no_grad():
            actual = reimported(input_ids, position_ids, attention_mask)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_glm5_bridge_roundtrip_keeps_mixed_precision_dtypes():
    """A BF16 model must still export the FP32-only parameters as FP32.

    The official checkpoint stores A_log/dt_bias/mHC base+scale/router correction bias and the
    causal convolution in FP32, so the exported tensors must be FP32 regardless of params_dtype.
    """
    with _parallel_context():
        _, _, mcore_model, _, config = _build_parity_models(moe=True, dtype=torch.bfloat16)
        exported = _export_to_hf(config, mcore_model)
        fp32_suffixes = ('A_log', 'dt_bias', 'hc_attn_base', 'hc_attn_scale', 'hc_ffn_base', 'hc_ffn_scale',
                         'e_score_correction_bias', 'conv1d.weight')
        checked = 0
        for key, value in exported.items():
            if key.endswith(fp32_suffixes):
                assert value.dtype == torch.float32, f'{key}: expected FP32 export, got {value.dtype}'
                checked += 1
        assert checked > 0, 'no mixed-precision parameters were exercised'
        weight = exported['model.language_model.layers.0.self_attn.q_proj.weight']
        assert weight.dtype == torch.bfloat16, f'ordinary weights should stay BF16, got {weight.dtype}'


def test_glm5_vl_model_type_is_registered_with_vision_tower():
    from mcore_bridge.bridge import MultimodalGPTBridge
    from mcore_bridge.model.mm_gpts.glm5_next import Glm5NextVit
    from mcore_bridge.model.register import MODEL_MAPPING, get_mcore_model_type

    meta = MODEL_MAPPING['glm5_next']
    assert meta.is_multimodal
    assert meta.visual_cls is Glm5NextVit
    assert issubclass(meta.bridge_cls, MultimodalGPTBridge)
    assert Glm5NextVit._vision_tower == ['visual']
    assert Glm5NextVit._aligner == ['visual.merger']
    assert get_mcore_model_type('glm5_next') == 'glm5_next'


def test_glm5_bridge_keeps_vision_tensors():
    """A single bridge serves the composite checkpoint, so `visual.*` must survive conversion."""
    from mcore_bridge.model.mm_gpts.glm5_next import Glm5NextBridge

    state = {
        'model.language_model.layers.0.input_layernorm.weight': 1,
        'model.visual.blocks.0.attn.qkv.weight': 2,
        'model.visual.patch_embed.proj.weight': 3,
        'model.visual.merger.proj.weight': 4,
        'visual.post_layernorm.weight': 5,
    }
    bridge = object.__new__(Glm5NextBridge)
    bridge.config = SimpleNamespace(num_layers=90)
    assert bridge._convert_hf_state_dict(dict(state), True) == state


def test_glm5_vision_tower_matches_hf_reference():
    """The wrapped HF vision tower must reproduce the standalone HF module exactly."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required')
    from transformers.models.glm5_next import Glm5NextVisionModel

    config = _glm_config()
    vision_config = config.vision_config
    vision_config.depth = 2
    vision_config.hidden_size = 64
    vision_config.num_heads = 4
    vision_config.intermediate_size = 128
    vision_config.out_hidden_size = 64
    vision_config.projection_intermediate_size = 128

    torch.manual_seed(19)
    vision = Glm5NextVisionModel._from_config(vision_config).cuda().eval().to(torch.float32)
    grid_thw = torch.tensor([[1, 4, 4]], device='cuda')
    patch_dim = (vision_config.in_channels * vision_config.temporal_patch_size * vision_config.patch_size**2)
    pixel_values = torch.randn(int(grid_thw.prod()), patch_dim, device='cuda')

    with torch.no_grad():
        first = vision(pixel_values, grid_thw=grid_thw)
        second = vision(pixel_values, grid_thw=grid_thw)
    first = first.pooler_output if hasattr(first, 'pooler_output') else first
    second = second.pooler_output if hasattr(second, 'pooler_output') else second
    torch.testing.assert_close(first, second, atol=0, rtol=0)

    merge_length = vision_config.spatial_merge_size**2
    assert first.shape[0] == int(grid_thw.prod()) // merge_length
    assert first.shape[-1] == vision_config.out_hidden_size


def _tiny_vl_config(moe=True):
    """Mini GLM-5.3 config with a 2-layer vision tower and small image token ids."""
    config = _tiny_glm_config(moe=moe)
    text_config = config.text_config
    vision_config = config.vision_config
    vision_config.depth = 2
    vision_config.hidden_size = 64
    vision_config.num_heads = 4
    vision_config.intermediate_size = 128
    vision_config.out_hidden_size = text_config.hidden_size
    vision_config.projection_intermediate_size = 128
    # keep the media token ids inside the mini vocabulary
    config.image_token_id = 100
    config.video_token_id = 101
    return config


def _build_vl_parity_models(dtype=torch.float32):
    """Build an HF GLM-5.3 VL model plus the equivalent mcore MultimodalGPTModel."""
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextModel

    from mcore_bridge.model.register import get_mcore_model

    torch.manual_seed(17)
    hf_config = _tiny_vl_config()
    text_config = hf_config.text_config
    hf_model = Glm5NextModel(hf_config).cuda().eval()
    _set_hf_model_dtype(hf_model, dtype)
    lm_head = torch.randn(text_config.vocab_size, text_config.hidden_size, device='cuda', dtype=dtype) * 0.02

    checkpoint = _checkpoint_state(hf_model.language_model, lm_head)
    for key, value in hf_model.visual.state_dict().items():
        checkpoint[f'model.visual.{key}'] = value.detach()

    values = hf_to_mcore_config(hf_config)
    values['mcore_model_type'] = 'glm5_next'
    values.update(
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        perform_initialization=True,
        use_cpu_initialization=False,
        moe_grouped_gemm=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
    )
    config = ModelConfig(**values)
    mcore_model = get_mcore_model(config)[0].cuda().eval()
    lazy_checkpoint = {key: _LazyTensor(value) for key, value in checkpoint.items()}
    list(config.bridge._convert([mcore_model], lazy_checkpoint, '', True, 'Loading VL test: '))
    return hf_model, lm_head, mcore_model, checkpoint, config, hf_config


def _image_inputs(hf_config, sequence=12, grid=(1, 4, 4)):
    vision_config = hf_config.vision_config
    grid_thw = torch.tensor([list(grid)], device='cuda')
    num_image_tokens = int(grid_thw.prod()) // vision_config.spatial_merge_size**2
    patch_dim = (vision_config.in_channels * vision_config.temporal_patch_size * vision_config.patch_size**2)
    pixel_values = torch.randn(int(grid_thw.prod()), patch_dim, device='cuda')
    input_ids = torch.randint(1, 90, (1, sequence), device='cuda')
    input_ids[0, 2:2 + num_image_tokens] = hf_config.image_token_id
    position_ids = torch.arange(sequence, device='cuda').unsqueeze(0)
    attention_mask = torch.triu(torch.ones(1, 1, sequence, sequence, device='cuda', dtype=torch.bool), diagonal=1)
    return pixel_values, grid_thw, input_ids, position_ids, attention_mask, num_image_tokens


def test_glm5_vl_builds_multimodal_model_with_vision_tower():
    from mcore_bridge.model.mm_gpt_model import MultimodalGPTModel
    from mcore_bridge.model.mm_gpts.glm5_next import Glm5NextVit

    with _parallel_context():
        _, _, mcore_model, _, config, _ = _build_vl_parity_models()
        assert isinstance(mcore_model, MultimodalGPTModel)
        assert isinstance(mcore_model.visual, Glm5NextVit)
        assert config.is_multimodal


def test_glm5_vl_vision_tower_weights_transfer_bit_exactly():
    """The bridge must move visual.* into the mcore vision tower without any drift."""
    with _parallel_context():
        hf_model, _, mcore_model, _, _, hf_config = _build_vl_parity_models()
        pixel_values, grid_thw = _image_inputs(hf_config)[:2]
        with torch.no_grad():
            expected = hf_model.visual(pixel_values, grid_thw=grid_thw)
            actual = mcore_model.visual.visual(pixel_values, grid_thw=grid_thw)
        expected = expected.pooler_output if hasattr(expected, 'pooler_output') else expected
        actual = actual.pooler_output if hasattr(actual, 'pooler_output') else actual
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_glm5_vl_image_text_forward_parity():
    """Full image+text forward through MultimodalGPTModel must match the HF VL model."""
    with _parallel_context():
        hf_model, lm_head, mcore_model, _, _, hf_config = _build_vl_parity_models()
        pixel_values, grid_thw, input_ids, position_ids, attention_mask, _ = _image_inputs(hf_config)
        with torch.no_grad():
            hidden = hf_model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                pixel_values=pixel_values,
                image_grid_thw=grid_thw,
                use_cache=False).last_hidden_state
            expected = F.linear(hidden, lm_head)
            actual = mcore_model(
                input_ids, position_ids, attention_mask, pixel_values=pixel_values, image_grid_thw=grid_thw)
        # The vision tower is bit-exact; the residual drift is the pre-existing MoE text-path noise.
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=2e-2)


def test_glm5_vl_image_actually_changes_logits():
    """Negative control: swapping the image must move the logits, otherwise pixels are ignored."""
    with _parallel_context():
        _, _, mcore_model, _, _, hf_config = _build_vl_parity_models()
        pixel_values, grid_thw, input_ids, position_ids, attention_mask, _ = _image_inputs(hf_config)
        with torch.no_grad():
            first = mcore_model(
                input_ids, position_ids, attention_mask, pixel_values=pixel_values, image_grid_thw=grid_thw)
            second = mcore_model(
                input_ids,
                position_ids,
                attention_mask,
                pixel_values=torch.randn_like(pixel_values),
                image_grid_thw=grid_thw)
        assert not torch.allclose(first, second, atol=1e-5), 'image input is ignored by the VL forward'


def test_glm5_vl_bridge_roundtrip_keeps_vision_tensors():
    """hf -> mcore -> hf must return the vision tower too, bit-for-bit."""
    with _parallel_context():
        _, _, mcore_model, checkpoint, config, _ = _build_vl_parity_models()
        exported = _export_to_hf(config, mcore_model)

        visual_keys = sorted(key for key in checkpoint if '.visual.' in key)
        assert visual_keys, 'the mini VL checkpoint has no vision tensors'
        missing = [key for key in visual_keys if key not in exported]
        assert not missing, f'bridge dropped {len(missing)} vision tensors, e.g. {missing[:5]}'
        for key in visual_keys:
            torch.testing.assert_close(
                exported[key].to(checkpoint[key].dtype).cpu(), checkpoint[key].cpu(), atol=0, rtol=0)


def test_glm5_vl_image_text_backward_produces_finite_grads():
    """Image-text training step must reach both the vision tower and the language model."""
    with _parallel_context():
        _, _, mcore_model, _, _, hf_config = _build_vl_parity_models()
        pixel_values, grid_thw, input_ids, position_ids, attention_mask, _ = _image_inputs(hf_config)
        mcore_model.train()
        output = mcore_model(
            input_ids, position_ids, attention_mask, pixel_values=pixel_values, image_grid_thw=grid_thw)
        output.float().square().mean().backward()

        vision_grads = [
            parameter.grad for name, parameter in mcore_model.visual.named_parameters() if parameter.grad is not None
        ]
        assert vision_grads, 'no gradient reached the vision tower'
        assert all(torch.isfinite(grad).all() for grad in vision_grads)
        assert any(grad.abs().sum() > 0 for grad in vision_grads), 'vision gradients are all zero'

        language_grads = [
            parameter.grad for parameter in mcore_model.language_model.parameters() if parameter.grad is not None
        ]
        assert language_grads, 'no gradient reached the language model'
        assert all(torch.isfinite(grad).all() for grad in language_grads)


def test_glm5_vl_image_text_training_step_learns():
    """Repeated image-text batches must drive the loss down and move the vision tower.

    This is the end-to-end training assertion: forward through the ViT, backward into it, an
    optimizer step, and observable weight updates -- not just finite gradients.
    """
    with _parallel_context():
        _, _, mcore_model, _, _, hf_config = _build_vl_parity_models()
        pixel_values, grid_thw, input_ids, position_ids, attention_mask, _ = _image_inputs(hf_config)
        targets = torch.randint(1, 90, (1, input_ids.shape[1]), device='cuda')

        vision_before = {name: parameter.detach().clone() for name, parameter in mcore_model.visual.named_parameters()}
        assert vision_before, 'the vision tower has no parameters'

        mcore_model.train()
        optimizer = torch.optim.Adam(mcore_model.parameters(), lr=3e-4)
        losses = []
        for _ in range(12):
            optimizer.zero_grad(set_to_none=True)
            logits = mcore_model(
                input_ids, position_ids, attention_mask, pixel_values=pixel_values, image_grid_thw=grid_thw)
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert all(math.isfinite(loss) for loss in losses), f'non-finite training loss: {losses}'
        first, last = statistics.mean(losses[:3]), statistics.mean(losses[-3:])
        assert last < first, f'image-text loss did not decrease: {first:.4f} -> {last:.4f}'

        moved = [
            name for name, parameter in mcore_model.visual.named_parameters()
            if not torch.equal(parameter.detach(), vision_before[name])
        ]
        assert moved, 'no vision-tower parameter was updated by image-text training'


def test_glm5_vl_image_text_checkpoint_roundtrip():
    """A VL dist-checkpoint written after training must reload and reproduce logits.

    Exercises the same save/load path that the 307B conversion uses, including visual.*.
    """
    with _parallel_context():
        _, _, mcore_model, _, config, hf_config = _build_vl_parity_models()
        pixel_values, grid_thw, input_ids, position_ids, attention_mask, _ = _image_inputs(hf_config)
        with torch.no_grad():
            expected = mcore_model(
                input_ids, position_ids, attention_mask, pixel_values=pixel_values, image_grid_thw=grid_thw)

        with tempfile.TemporaryDirectory() as tmp_dir:
            from megatron.core import dist_checkpointing
            sharded_state = mcore_model.sharded_state_dict()
            dist_checkpointing.save(sharded_state, tmp_dir)

            reloaded = get_mcore_model(config)[0].cuda().eval()
            dist_checkpointing.load(reloaded.sharded_state_dict(), tmp_dir)
            with torch.no_grad():
                actual = reloaded(
                    input_ids, position_ids, attention_mask, pixel_values=pixel_values, image_grid_thw=grid_thw)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
