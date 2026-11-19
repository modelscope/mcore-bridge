# Copyright (c) ModelScope Contributors. All rights reserved.
"""MiMo-V2 parity / roundtrip / LoRA / sharding tests.

The HF oracle is the official remote code (modeling_mimo_v2.py / configuration_mimo_v2.py).
Point MIMO_V2_REMOTE_DIR at a directory holding those two files (default: the local dequantized
tiny checkpoint). Two HF quirks shape the harness:
* ``MiMoV2MoEGate.forward`` raises in train mode (noaux_tc is inference-only) -> the HF model
  stays in ``eval()`` for every parity check; gradients still flow (dropout is 0).
* ``MiMoV2ForCausalLM`` ignores ``model.mtp.*`` on load -> the MTP reference is hand-built from
  HF ``MiMoV2DecoderLayer`` (SWA + dense) plus enorm/hnorm/eh_proj/final_layernorm.
"""
import copy
import json
import os
import pytest
import shutil
import subprocess
import sys
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from types import SimpleNamespace as NS

from mcore_bridge.config import ModelConfig
from mcore_bridge.config.parser import hf_to_mcore_config
from mcore_bridge.model.mm_gpts import mimo_v2
from mcore_bridge.model.register import get_mcore_model
from mcore_bridge.utils.safetensors import LazyTensor

_REMOTE_DIR = os.environ.get('MIMO_V2_REMOTE_DIR', '/mnt/workspace/hjh/models/MiMo-V2.6-Flash-RL-bf16-tiny')
_PATTERN = [0, 1, 1, 0, 1, 1]


def _remote_code_available():
    return all(
        os.path.exists(os.path.join(_REMOTE_DIR, f)) for f in ('modeling_mimo_v2.py', 'configuration_mimo_v2.py'))


requires_remote_code = pytest.mark.skipif(not _remote_code_available(), reason='MiMo-V2 HF remote code not found')

# Real-weight regression needs the dequantized checkpoint itself, not just the remote code.
_REAL_WEIGHTS_DIR = os.environ.get('MIMO_V2_REAL_WEIGHTS_DIR', _REMOTE_DIR)


def _real_weights_available():
    return _remote_code_available() and all(
        os.path.exists(os.path.join(_REAL_WEIGHTS_DIR, f)) for f in ('config.json', 'model.safetensors.index.json'))


requires_real_weights = pytest.mark.skipif(
    not _real_weights_available(), reason='MiMo-V2 dequantized real weights not found')


def _tiny_config_dict(layers=6, mtp=1):
    return dict(
        model_type='mimo_v2',
        architectures=['MiMoV2ForCausalLM'],
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=layers,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        v_head_dim=16,
        swa_num_attention_heads=8,
        swa_num_key_value_heads=4,
        swa_head_dim=32,
        swa_v_head_dim=16,
        swa_rope_theta=10000.0,
        rope_theta=10000000.0,
        partial_rotary_factor=0.5,
        hybrid_layer_pattern=_PATTERN[:layers],
        moe_layer_freq=[0, 1, 1, 1, 1, 1][:layers],
        sliding_window=8,
        sliding_window_size=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=None,
        scoring_func='sigmoid',
        topk_method='noaux_tc',
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=None,
        layernorm_epsilon=1e-6,
        hidden_act='silu',
        attention_value_scale=0.707,
        attention_bias=False,
        attention_dropout=0.0,
        add_swa_attention_sink_bias=True,
        add_full_attention_sink_bias=False,
        num_nextn_predict_layers=mtp,
        max_position_embeddings=512,
        tie_word_embeddings=False,
        use_cache=False,
        auto_map={
            'AutoConfig': 'configuration_mimo_v2.MiMoV2Config',
            'AutoModelForCausalLM': 'modeling_mimo_v2.MiMoV2ForCausalLM',
        },
    )


def _copy_remote_code(src, dst):
    """Copy the official remote code, shimming the transformers 5.x masking API.

    The shipped modeling file predates the rename ``input_embeds`` -> ``inputs_embeds`` and
    still passes ``cache_position`` (dropped from create_causal_mask's signature). Patch only
    the mask_kwargs dict; the oracle math is untouched.
    """
    for name in ('modeling_mimo_v2.py', 'configuration_mimo_v2.py'):
        shutil.copy(os.path.join(src, name), dst)
    path = os.path.join(dst, 'modeling_mimo_v2.py')
    with open(path) as f:
        text = f.read()
    text = text.replace('"input_embeds": inputs_embeds,', '"inputs_embeds": inputs_embeds,')
    text = text.replace('                "cache_position": cache_position,\n', '')
    with open(path, 'w') as f:
        f.write(text)


@pytest.fixture(scope='session')
def remote_dir(tmp_path_factory):
    if not _remote_code_available():
        pytest.skip('MiMo-V2 HF remote code not found')
    directory = str(tmp_path_factory.mktemp('mimo_v2_remote'))
    _copy_remote_code(_REMOTE_DIR, directory)
    with open(os.path.join(directory, 'config.json'), 'w') as f:
        json.dump(_tiny_config_dict(), f)
    return directory


def _hf_config(remote_dir, **overrides):
    # Rebuild from the canonical dict: rejection tests write overrides into the shared session
    # config.json, which would leak into every later test if read back.
    payload = _tiny_config_dict()
    payload.update(overrides)
    with open(os.path.join(remote_dir, 'config.json'), 'w') as f:
        json.dump(payload, f)
    config = AutoConfig.from_pretrained(remote_dir, trust_remote_code=True)
    config._attn_implementation = 'eager'  # sink bias needs the eager path
    return config


def _randomize(model, seed=17):
    """Re-randomize every parameter: default init leaves sink/router-bias at empty/zeros,
    which would make the sink and noaux_tc parity checks vacuous."""
    generator = torch.Generator(device='cuda').manual_seed(seed)

    def draw(shape, scale):
        return torch.randn(shape, generator=generator, device='cuda', dtype=torch.float32) * scale

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith('attention_sink_bias'):
                param.copy_(draw(param.shape, 0.5))
            elif name.endswith('e_score_correction_bias'):
                param.copy_(draw(param.shape, 0.1))
            elif name.endswith('norm.weight'):
                param.copy_(1.0 + draw(param.shape, 0.1))
            elif name.endswith('gate.weight'):
                param.copy_(draw(param.shape, 0.05))
            else:
                param.copy_(draw(param.shape, 0.02))


def _build_hf(remote_dir, dtype=torch.float32, seed=17, **overrides):
    config = _hf_config(remote_dir, **overrides)
    cls = get_class_from_dynamic_module('modeling_mimo_v2.MiMoV2ForCausalLM', remote_dir)
    model = cls(config).cuda().eval()
    _randomize(model, seed=seed)
    return model.to(dtype)


def _mtp_state(hf_model, remote_dir, seed=29):
    """Random MTP weights in HF checkpoint layout, plus the reference modules that produced them.

    HF ignores model.mtp.* on load, so the draft block is built by hand: a SWA + dense
    MiMoV2DecoderLayer (moe_layer_freq zeroed, layer_idx=1 -> SWA) wrapped with the
    enorm/hnorm/eh_proj/final_layernorm the checkpoint carries.
    """
    cls_layer = get_class_from_dynamic_module('modeling_mimo_v2.MiMoV2DecoderLayer', remote_dir)
    cls_norm = get_class_from_dynamic_module('modeling_mimo_v2.MiMoV2RMSNorm', remote_dir)
    config = copy.deepcopy(hf_model.config)
    config.moe_layer_freq = [0] * config.num_hidden_layers
    layer = cls_layer(config, layer_idx=1).cuda()
    enorm = cls_norm(config.hidden_size, eps=config.layernorm_epsilon).cuda()
    hnorm = cls_norm(config.hidden_size, eps=config.layernorm_epsilon).cuda()
    eh_proj = torch.nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False).cuda()
    final_layernorm = cls_norm(config.hidden_size, eps=config.layernorm_epsilon).cuda()
    reference = NS(layer=layer, enorm=enorm, hnorm=hnorm, eh_proj=eh_proj, final_layernorm=final_layernorm)
    for module in (layer, enorm, hnorm, eh_proj, final_layernorm):
        _randomize(module, seed=seed)
        seed += 1
    prefix = 'model.mtp.layers.0.'
    state = {}
    for key, value in layer.state_dict().items():
        key = key.replace('post_attention_layernorm.', 'pre_mlp_layernorm.')
        state[prefix + key] = value.detach().clone()
    state[prefix + 'enorm.weight'] = enorm.weight.detach().clone()
    state[prefix + 'hnorm.weight'] = hnorm.weight.detach().clone()
    state[prefix + 'eh_proj.weight'] = eh_proj.weight.detach().clone()
    state[prefix + 'final_layernorm.weight'] = final_layernorm.weight.detach().clone()
    # The hand-built modules default to FP32; match the checkpoint dtype (no-op for FP32 runs).
    state = {key: value.to(hf_model.dtype) for key, value in state.items()}
    return state, reference


def _checkpoint_state(hf_model, mtp_state=None):
    state = {key: value.detach().clone() for key, value in hf_model.state_dict().items()}
    if mtp_state:
        state.update(mtp_state)
    return state


def _mcore_config(hf_config, tp=1, pp=1, ep=1, sequence_parallel=False, dtype=torch.float32, mtp=0, grouped=True):
    values = hf_to_mcore_config(hf_config)
    values['mcore_model_type'] = 'mimo_v2'
    values.update(
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        perform_initialization=True,
        use_cpu_initialization=False,
        language_model_only=True,
        moe_grouped_gemm=grouped,
        overlap_p2p_comm=False,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=1,
        sequence_parallel=sequence_parallel,
        mtp_num_layers=mtp,
    )
    if mtp:
        values['mtp_loss_scaling_factor'] = 0.1
    return ModelConfig(**values)


def _build_parity_models(remote_dir, tp=1, pp=1, ep=1, sequence_parallel=False, dtype=torch.float32, mtp=0, seed=17):
    hf_model = _build_hf(remote_dir, dtype=dtype, seed=seed)
    mtp_state, reference = _mtp_state(hf_model, remote_dir) if mtp else (None, None)
    checkpoint = _checkpoint_state(hf_model, mtp_state)
    config = _mcore_config(
        hf_model.config, tp=tp, pp=pp, ep=ep, sequence_parallel=sequence_parallel, dtype=dtype, mtp=mtp)
    model = get_mcore_model(config)[0].cuda().eval()
    lazy = {key: LazyTensor(tensor=value) for key, value in checkpoint.items()}
    list(config.bridge._convert([model], lazy, '', True, 'Loading test: '))
    return hf_model, model, checkpoint, config, reference


def _model_inputs(batch=2, sequence=24):
    # sequence > sliding_window (8) so the SWA mask actually truncates attention.
    generator = torch.Generator(device='cuda').manual_seed(23)
    input_ids = torch.randint(1, 128, (batch, sequence), device='cuda', generator=generator)
    position_ids = torch.arange(sequence, device='cuda').unsqueeze(0).expand(batch, -1)
    attention_mask = torch.triu(torch.ones(batch, 1, sequence, sequence, device='cuda', dtype=torch.bool), diagonal=1)
    return input_ids, position_ids, attention_mask


def _hf_logits(hf_model, input_ids):
    attention_mask = torch.ones_like(input_ids)
    return hf_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits


def _mcore_logits(model, config, input_ids, position_ids, attention_mask):
    logits = model(input_ids, position_ids, attention_mask, runtime_gather_output=True)
    return logits[..., :config.actual_vocab_size]


def _export(config, model):
    result = {}
    for key, value in config.bridge._convert([model], {}, '', False, 'Exporting test: '):
        result[key] = value.load() if hasattr(value, 'load') else value
    return result


def _lm(model):
    return model.language_model


@requires_remote_code
def test_parser_maps_mimo_fields(remote_dir):
    config = _hf_config(remote_dir)
    values = hf_to_mcore_config(config)
    assert values['layernorm_epsilon'] == 1e-6
    assert values['window_size'] == '7,0'
    assert values['window_attn_skip_freq'] == '[0,1,1,0,1,1]'
    assert values['moe_layer_freq'] == '[0,1,1,1,1,1]'
    assert values['moe_router_load_balancing_type'] == 'none'
    assert values['moe_router_enable_expert_bias'] is True
    assert values['moe_router_bias_update_rate'] == 0.0
    assert values['moe_router_dtype'] == 'fp32'
    assert values['apply_rope_fusion'] is False
    assert values['hidden_dropout'] == 0.0
    assert values['attention_value_scale'] == 0.707
    assert values['swa_rope_theta'] == 10000.0
    assert values['swa_num_key_value_heads'] == 4
    assert values['kv_channels'] == 32
    assert values['v_head_dim'] == 16
    assert 'moe_router_num_groups' not in values
    assert 'moe_router_group_topk' not in values


@requires_remote_code
@pytest.mark.parametrize('field', ['swa_num_attention_heads', 'swa_head_dim', 'swa_v_head_dim'])
def test_parser_rejects_heterogeneous_swa_dims(remote_dir, field):
    config = _hf_config(remote_dir)
    setattr(config, field, getattr(config, field) + 1)
    with pytest.raises(ValueError, match=field):
        hf_to_mcore_config(config)


@requires_remote_code
def test_parser_handles_pro_scale_variant(remote_dir):
    # MiMo-V2.6-Pro-RL scale: 8 KV heads on both attention families, 384 experts, 1-in-7 pattern.
    pattern = [0, 1, 1, 1, 1, 1, 1] * 10
    config = _hf_config(
        remote_dir,
        hidden_size=6144,
        num_hidden_layers=70,
        num_attention_heads=128,
        num_key_value_heads=8,
        swa_num_attention_heads=128,
        swa_num_key_value_heads=8,
        n_routed_experts=384,
        attention_value_scale=0.612,
        hybrid_layer_pattern=pattern,
        moe_layer_freq=[0] + [1] * 69)
    values = hf_to_mcore_config(config)
    assert values['window_attn_skip_freq'] == '[' + ','.join(map(str, pattern)) + ']'
    assert values['moe_layer_freq'] == '[0' + ',1' * 69 + ']'
    assert values['attention_value_scale'] == 0.612
    assert values['swa_num_key_value_heads'] == 8
    assert values['num_query_groups'] == 8
    assert values['num_moe_experts'] == 384
    assert 'moe_router_num_groups' not in values


@requires_remote_code
def test_parser_handles_v25_rope_scaling_variant(remote_dir):
    # MiMo-V2.5 ships the legacy transformers-4.x rope_scaling layout (already used by the tiny
    # config); after 5.x normalization the parser must still recover theta / partial factor.
    payload = _tiny_config_dict()
    payload['rope_scaling'] = {'rope_type': 'default', 'type': 'default'}
    with open(os.path.join(remote_dir, 'config.json'), 'w') as f:
        json.dump(payload, f)
    config = AutoConfig.from_pretrained(remote_dir, trust_remote_code=True)
    values = hf_to_mcore_config(config)
    assert values['rotary_base'] == 10000000.0
    assert values['partial_rotary_factor'] == 0.5
    assert values['swa_rope_theta'] == 10000.0


@requires_remote_code
@pytest.mark.parametrize('change,match', [({'add_full_attention_sink_bias': True}, 'full-attention sink')])
def test_bridge_rejects_unsupported_checkpoints(remote_dir, change, match):
    config = _hf_config(remote_dir, **change)
    values = hf_to_mcore_config(config)
    values['mcore_model_type'] = 'mimo_v2'
    values['language_model_only'] = True
    # The bridge is built eagerly in ModelConfig.__post_init__, so the guard fires there.
    from test_glm5_next import _parallel_context

    with _parallel_context():
        with pytest.raises((ValueError, NotImplementedError), match=match):
            ModelConfig(**values)


_OFFICIAL_QUANT_CONFIG = dict(
    quant_method='fp8',
    fmt='e4m3',
    activation_scheme='dynamic',
    weight_block_size=[128, 128],
    mxfp4_block_size=32,
    store_dtype='mxfp4',
)

# V2.5: same FP8 block scheme, but the routed experts are block-FP8 too (no MXFP4 anywhere).
_V25_QUANT_CONFIG = dict(
    quant_method='fp8',
    fmt='e4m3',
    activation_scheme='dynamic',
    weight_block_size=[128, 128],
    store_dtype='fp8',
)


@requires_remote_code
@pytest.mark.parametrize('change', [
    {
        'attention_projection_layout': 'fused_qkv',
        'quantization_config': dict(_OFFICIAL_QUANT_CONFIG)
    },
    {
        'quantization_config': dict(_OFFICIAL_QUANT_CONFIG)
    },
])
def test_bridge_accepts_quantized_input(remote_dir, change):
    config = _hf_config(remote_dir, **change)
    values = hf_to_mcore_config(config)
    assert values.get('fp8') is None and values.get('fp4') is None  # no auto FP8 training mode
    values['mcore_model_type'] = 'mimo_v2'
    values['language_model_only'] = True
    from test_glm5_next import _parallel_context

    with _parallel_context():
        mcore_config = ModelConfig(**values)
    assert mcore_config.bridge._quantized_input
    # Markers stay on the in-memory config (other consumers may read them); the export-side
    # config writeback strips them so saved BF16 checkpoints are not re-read as quantized.
    assert hasattr(mcore_config.hf_config, 'quantization_config')


@requires_remote_code
def test_bridge_stale_layout_marker_stays_plain(remote_dir):
    # A split BF16 checkpoint with a leftover 'fused_qkv' marker must not enter the dequantizing
    # view (which would demand FP8 scales); quantization_config alone decides.
    config = _hf_config(remote_dir, attention_projection_layout='fused_qkv')
    values = hf_to_mcore_config(config)
    values['mcore_model_type'] = 'mimo_v2'
    values['language_model_only'] = True
    from test_glm5_next import _parallel_context

    with _parallel_context():
        mcore_config = ModelConfig(**values)
    assert not mcore_config.bridge._quantized_input


@pytest.mark.parametrize('change,match', [
    ({
        'context_parallel_size': 2
    }, 'context_parallel_size=1'),
    ({
        'fp8': 'hybrid'
    }, 'FP8/FP4'),
    ({
        'fp4': 'mxfp4'
    }, 'FP8/FP4'),
    ({
        'tensor_model_parallel_size': 8
    }, 'TP <='),
    ({
        'flash_decode': True
    }, 'flash_decode/cuda-graph'),
    ({
        'enable_cuda_graph': True
    }, 'flash_decode/cuda-graph'),
])
def test_loader_guards(change, match):
    config = NS(
        context_parallel_size=1,
        fp8=None,
        fp4=None,
        num_query_groups=2,
        swa_num_key_value_heads=4,
        tensor_model_parallel_size=1)
    for key, value in change.items():
        setattr(config, key, value)
    with pytest.raises(ValueError, match=match):
        mimo_v2.MiMoV2Loader(config)


def test_loader_rejects_flash_backend_with_swa_sink():
    # TE disables every flash backend for the learnable SWA sink; reject swift's default up front.
    config = NS(
        context_parallel_size=1,
        fp8=None,
        fp4=None,
        num_query_groups=2,
        swa_num_key_value_heads=4,
        tensor_model_parallel_size=1,
        attention_backend=NS(name='flash'),
        hf_config=NS(add_swa_attention_sink_bias=True))
    with pytest.raises(ValueError, match='attention_backend=fused'):
        mimo_v2.MiMoV2Loader(config)
    for backend in ('fused', 'auto'):
        config.attention_backend = NS(name=backend)
        mimo_v2.MiMoV2Loader(config)
    # Without the sink (or without an hf_config, as in the guard tests above) flash stays allowed.
    config.attention_backend = NS(name='flash')
    config.hf_config = NS(add_swa_attention_sink_bias=False)
    mimo_v2.MiMoV2Loader(config)


@pytest.mark.parametrize('audio_key', ['input_features', 'audio_codes', 'audio_embeds'])
def test_vit_rejects_audio_inputs(audio_key, monkeypatch):
    # The shared vision helper only reads vision keys: audio must fail loudly at the guard,
    # never slip through and be silently dropped. audio_codes is what the swift template emits.
    vit = object.__new__(mimo_v2.MiMoV2Vit)  # the guard runs before any instance attribute is read
    embeds = torch.zeros(1, 4, 8)
    with pytest.raises(NotImplementedError, match='audio'):
        vit.get_inputs_embeds(embeds, **{audio_key: torch.zeros(2, 3)})
    reached = []
    monkeypatch.setattr(mimo_v2.MiMoV2Vit, '_hf_get_inputs_embeds', staticmethod(lambda *a, **k: reached.append(1)))
    vit.hf_config = NS(vision_config={})
    vit.visual = None  # only passed through to the stubbed helper
    vit.get_inputs_embeds(embeds)  # plain text passes the guard
    vit.get_inputs_embeds(embeds, pixel_values=torch.zeros(4, 8))  # vision passes the guard
    assert len(reached) == 2


def test_registration_exposes_mtp_prefix():
    from mcore_bridge.model.register import MODEL_MAPPING
    meta = MODEL_MAPPING['mimo_v2']
    assert meta.bridge_cls is mimo_v2.MiMoV2Bridge
    assert meta.loader is mimo_v2.MiMoV2Loader
    assert meta.mtp_prefix == 'model.mtp.layers'


def test_split_qkv_asymmetric_layout():
    # One KV group = [hpg*qk, qk, v]; a symmetric split would misplace V rows.
    groups, hpg, qk, v, hidden = 2, 4, 32, 16, 8
    q = torch.arange(groups * hpg * qk * hidden, dtype=torch.float32).reshape(groups * hpg * qk, hidden)
    k = torch.arange(groups * qk * hidden, dtype=torch.float32).reshape(groups * qk, hidden) + 1e6
    v_t = torch.arange(groups * v * hidden, dtype=torch.float32).reshape(groups * v, hidden) + 2e6
    fused = torch.cat(
        [torch.cat([q.view(groups, hpg * qk, hidden),
                    k.view(groups, qk, hidden),
                    v_t.view(groups, v, hidden)], dim=1)],
        dim=0).reshape(-1, hidden)
    got_q, got_k, got_v = mimo_v2._split_qkv(fused, dim=0, groups=groups, heads_per_group=hpg, qk_dim=qk, v_dim=v)
    torch.testing.assert_close(got_q.reshape(-1, hidden), q, atol=0, rtol=0)
    torch.testing.assert_close(got_k.reshape(-1, hidden), k, atol=0, rtol=0)
    torch.testing.assert_close(got_v.reshape(-1, hidden), v_t, atol=0, rtol=0)
    # Negative control: the symmetric (qk==v) split must NOT recover the same tensors.
    wrong = fused.view(groups, -1, hidden).chunk(3, dim=1)
    assert not torch.equal(wrong[2].reshape(-1, hidden), v_t)


def _fp8_block_quantize(weight):
    """[rows, cols] -> FP8 e4m3 bytes + scale_inv (128x128 blocks, edge blocks partial)."""
    rows, cols = weight.shape
    scale_rows, scale_cols = -(-rows // 128), -(-cols // 128)
    packed = torch.zeros(rows, cols, dtype=torch.float8_e4m3fn, device=weight.device)
    scale = torch.zeros(scale_rows, scale_cols, dtype=torch.float32, device=weight.device)
    weight = weight.float()
    for i in range(scale_rows):
        for j in range(scale_cols):
            block = weight[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128]
            s = block.abs().amax().clamp(min=1e-12) / 448.0
            packed[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = (block / s).to(torch.float8_e4m3fn)
            scale[i, j] = s
    return packed, scale


def _mxfp4_quantize(weight):
    """[out, in] -> packed e2m1 uint8 + E8M0 scale (one per 32 inputs), official MXFP4 layout."""
    out_dim, in_dim = weight.shape
    blocks = weight.float().view(out_dim, in_dim // 32, 32)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    exponent = (torch.floor(torch.log2(amax)) - 2).clamp(-127, 127)
    scaled = blocks / torch.exp2(exponent)
    # Independent OCP e2m1 code table (deliberately not the production LUT, so the fixture
    # quantizer does not close a self-consistency loop with the dequantizer under test).
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                         dtype=torch.float32,
                         device=weight.device)
    codes = (scaled.unsqueeze(-1) - table).abs().argmin(dim=-1).view(out_dim, in_dim).to(torch.uint8)
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    return packed, (exponent.view(out_dim, in_dim // 32) + 127).to(torch.uint8)


def _fuse_qkv_official(q, k, v, hf_config, is_swa):
    """Split q/k/v -> per-KV-shard [q_i|k_i|v_i], FP8-blocked with exact rows/cols; only the
    scale grid covers whole 128 blocks (edge blocks partial), as in the official checkpoint."""
    shards = hf_config.num_key_value_heads
    prefix = 'swa_' if is_swa else ''
    heads = getattr(hf_config, f'{prefix}num_attention_heads')
    kv_heads = getattr(hf_config, f'{prefix}num_key_value_heads')
    qk_dim, v_dim = getattr(hf_config, f'{prefix}head_dim'), getattr(hf_config, f'{prefix}v_head_dim')
    sizes = [heads // shards * qk_dim, kv_heads // shards * qk_dim, kv_heads // shards * v_dim]
    hidden = q.shape[1]
    parts, scales = [], []
    for i in range(shards):
        shard = torch.cat([t.view(shards, size, hidden)[i] for t, size in zip((q, k, v), sizes)], dim=0)
        fp8, scale = _fp8_block_quantize(shard)
        parts.append(fp8)
        scales.append(scale)
    return torch.cat(parts, dim=0), torch.cat(scales, dim=0)


def _quantize_to_official_layout(state, hf_config, experts_mxfp4=True, router_fp32=False):
    """Canonical BF16 split state -> official quantized layout (the inverse of the convert tool)."""
    if router_fp32:
        # V2.5 stores the router gate in FP32: redraw it at full precision (the BF16 HF model
        # rounded it) so a loader that casts the gate to BF16 fails the bit-exact comparison.
        generator = torch.Generator(device='cuda').manual_seed(41)
        state = dict(state)
        for key in state:
            if key.endswith('mlp.gate.weight'):
                state[key] = torch.randn(state[key].shape, generator=generator, device='cuda') * 0.05
    quantized, qkv_parts = {}, {}
    for key, value in state.items():
        if router_fp32 and key.endswith('mlp.gate.weight'):
            quantized[key] = value
            continue
        value = value.to(torch.bfloat16)
        if key.endswith(('self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight')):
            prefix, proj = key.rsplit('.', 2)[0], key.split('.')[-2].removesuffix('_proj')
            qkv_parts.setdefault(prefix, {})[proj] = value
        elif '.mlp.experts.' in key:
            if experts_mxfp4:
                packed, scale = _mxfp4_quantize(value)
                quantized[key] = packed
                quantized[key.removesuffix('weight') + 'weight_scale'] = scale
            else:
                fp8, scale = _fp8_block_quantize(value)
                quantized[key] = fp8
                quantized[key + '_scale_inv'] = scale
        elif key.endswith(('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight')):
            is_mtp = key.startswith('model.mtp.layers.')
            if is_mtp or not hf_config.moe_layer_freq[int(key.split('.')[2])]:
                fp8, scale = _fp8_block_quantize(value)
                quantized[key] = fp8
                quantized[key + '_scale_inv'] = scale
            else:
                quantized[key] = value
        else:
            quantized[key] = value
    for prefix, parts in qkv_parts.items():
        is_swa = prefix.startswith('model.mtp.') or bool(hf_config.hybrid_layer_pattern[int(prefix.split('.')[2])])
        fp8, scale = _fuse_qkv_official(parts['q'], parts['k'], parts['v'], hf_config, is_swa)
        quantized[prefix + '.qkv_proj.weight'] = fp8
        quantized[prefix + '.qkv_proj.weight_scale_inv'] = scale
    return quantized


def _write_quantized_fixture(directory, state, hf_config, quant_config=None):
    from safetensors.torch import save_file

    quant_config = _OFFICIAL_QUANT_CONFIG if quant_config is None else quant_config
    os.makedirs(directory, exist_ok=True)
    _copy_remote_code(_REMOTE_DIR, directory)
    quantized = _quantize_to_official_layout(
        state,
        hf_config,
        experts_mxfp4='mxfp4_block_size' in quant_config,
        router_fp32=quant_config.get('store_dtype') == 'fp8')
    shard = 'model-00001.safetensors'
    save_file({
        k: v.contiguous()
        for k, v in quantized.items()
    },
              os.path.join(directory, shard),
              metadata={'format': 'pt'})
    payload = _tiny_config_dict()
    payload['attention_projection_layout'] = 'fused_qkv'
    ignored = [f'model.layers.{i}.self_attn.o_proj' for i in range(payload['num_hidden_layers'])]
    ignored += [f'model.mtp.layers.{i}.self_attn.o_proj' for i in range(payload['num_nextn_predict_layers'])]
    payload['quantization_config'] = dict(quant_config, ignored_layers=ignored)
    with open(os.path.join(directory, 'config.json'), 'w') as f:
        json.dump(payload, f)
    total = sum(v.numel() * v.element_size() for v in quantized.values())
    with open(os.path.join(directory, 'model.safetensors.index.json'), 'w') as f:
        json.dump({'metadata': {'total_size': total}, 'weight_map': dict.fromkeys(quantized, shard)}, f)


def _reference_dequant(state, hf_config):
    """Test-local dequant of the quantized fixture (the offline-loading oracle).

    Deliberately independent formulas (not the production helpers), so T2 checks the online
    adapter's wiring against a separate implementation; the math itself is verified bit-exact
    against the old formulas and official weights by the A/B probes. Quantization is lossy, so
    the pre-quantization canonical state is NOT a valid oracle here.
    """
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                         dtype=torch.float32)

    def fp8(w, s):
        rows, cols = w.shape
        scale = s.float().repeat_interleave(128, dim=0)[:rows].repeat_interleave(128, dim=1)
        return (w.float() * scale[:, :cols]).to(torch.bfloat16)

    def mxfp4(packed, scale):
        u8 = packed.view(torch.uint8)
        lut = table.to(u8.device)
        vals = torch.stack([lut[(u8 & 0x0F).long()], lut[(u8 >> 4).long()]],
                           dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)
        return (vals * torch.exp2(scale.float() - 127.0).repeat_interleave(32, dim=1)).to(torch.bfloat16)

    out = {}
    for key, value in state.items():
        if key.endswith('_scale_inv') or key.endswith('weight_scale'):
            continue
        if key.endswith('self_attn.qkv_proj.weight'):
            shards = hf_config.num_key_value_heads
            rows = value.shape[0] // shards
            scale_rows = -(-rows // 128)
            scales = state[key + '_scale_inv']
            qkv = torch.cat([fp8(w, s) for w, s in zip(value.chunk(shards), scales.split(scale_rows), strict=True)])
            prefix = key.rsplit('.', 3)[0]  # strip '.self_attn.qkv_proj.weight' -> layer prefix
            is_swa = prefix.startswith('model.mtp.') or bool(hf_config.hybrid_layer_pattern[int(prefix.split('.')[2])])
            pfx = 'swa_' if is_swa else ''
            heads = getattr(hf_config, f'{pfx}num_attention_heads')
            kv_heads = getattr(hf_config, f'{pfx}num_key_value_heads')
            qk_dim, v_dim = getattr(hf_config, f'{pfx}head_dim'), getattr(hf_config, f'{pfx}v_head_dim')
            sizes = [heads // shards * qk_dim, kv_heads // shards * qk_dim, kv_heads // shards * v_dim]
            for proj, part in zip('qkv', qkv.view(shards, sum(sizes), -1).split(sizes, dim=1), strict=True):
                out[f'{prefix}.self_attn.{proj}_proj.weight'] = part.reshape(-1, qkv.shape[1])
        elif key + '_scale_inv' in state:
            out[key] = fp8(value, state[key + '_scale_inv'])
        elif key.removesuffix('weight') + 'weight_scale' in state:
            out[key] = mxfp4(value, state[key.removesuffix('weight') + 'weight_scale'])
        else:
            out[key] = value
    return out


def _write_bf16_reference(directory, quantized_state, hf_config):
    """Dump the fixture's direct dequantization as a plain BF16 split-q/k/v checkpoint."""
    from safetensors.torch import save_file

    os.makedirs(directory, exist_ok=True)
    _copy_remote_code(_REMOTE_DIR, directory)
    state = _reference_dequant(quantized_state, hf_config)
    shard = 'model-00001.safetensors'
    save_file({k: v.contiguous() for k, v in state.items()}, os.path.join(directory, shard), metadata={'format': 'pt'})
    with open(os.path.join(directory, 'config.json'), 'w') as f:
        json.dump(_tiny_config_dict(), f)
    total = sum(v.numel() * v.element_size() for v in state.values())
    with open(os.path.join(directory, 'model.safetensors.index.json'), 'w') as f:
        json.dump({'metadata': {'total_size': total}, 'weight_map': dict.fromkeys(state, shard)}, f)


def _check_online_matches_offline(remote_dir, tmp_path, tp=1, mtp=1, quant_config=None):
    """T2: online dequant loading vs a direct dequant of the same quantized fixture."""
    from mcore_bridge.utils.safetensors import SafetensorLazyLoader

    hf_model = _build_hf(remote_dir, dtype=torch.bfloat16)
    mtp_state, _ = _mtp_state(hf_model, remote_dir) if mtp else (None, None)
    canonical = _checkpoint_state(hf_model, mtp_state)
    qdir = str(tmp_path / 'quantized')
    _write_quantized_fixture(qdir, canonical, hf_model.config, quant_config)
    odir = str(tmp_path / 'offline')
    quant_config = _OFFICIAL_QUANT_CONFIG if quant_config is None else quant_config
    quantized = _quantize_to_official_layout(
        canonical,
        hf_model.config,
        experts_mxfp4='mxfp4_block_size' in quant_config,
        router_fp32=quant_config.get('store_dtype') == 'fp8')
    _write_bf16_reference(odir, quantized, hf_model.config)

    models, configs = [], []
    for directory in (qdir, odir):
        hf_config = AutoConfig.from_pretrained(directory, trust_remote_code=True)
        config = _mcore_config(hf_config, tp=tp, mtp=mtp)
        model = get_mcore_model(config)[0].cuda().eval()
        with SafetensorLazyLoader(directory) as loader:
            list(config.bridge._convert([model], loader.get_state_dict(), '', True, 'Loading: '))
        models.append(model)
        configs.append(config)
    online, offline = models
    exported_online, exported_offline = _export(configs[0], online), _export(configs[1], offline)
    assert set(exported_online) == set(exported_offline)
    assert not any('_scale' in key for key in exported_online)
    for key in exported_online:
        assert exported_online[key].dtype == exported_offline[key].dtype, f'{key}: dtype'
        torch.testing.assert_close(
            exported_online[key], exported_offline[key], atol=0, rtol=0, msg=lambda m: f'{key}: {m}')
    input_ids, position_ids, attention_mask = _model_inputs()
    actual = _mcore_logits(online, configs[0], input_ids, position_ids, attention_mask)
    expected = _mcore_logits(offline, configs[1], input_ids, position_ids, attention_mask)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@requires_remote_code
def test_online_dequant_matches_offline_tool(remote_dir, tmp_path):
    from test_glm5_next import _parallel_context

    with _parallel_context():
        _check_online_matches_offline(remote_dir, tmp_path, tp=1, mtp=1)


@requires_remote_code
def test_online_dequant_v25_fp8_experts_matches_offline_tool(remote_dir, tmp_path):
    """T2-V25: the all-FP8 layout (block-FP8 experts + weight_scale_inv) wires identically."""
    from test_glm5_next import _parallel_context

    with _parallel_context():
        _check_online_matches_offline(remote_dir, tmp_path, tp=1, mtp=1, quant_config=_V25_QUANT_CONFIG)


@requires_remote_code
def test_online_dequant_v25_router_stays_fp32(remote_dir, tmp_path):
    """V2.5 stores the router gate in FP32; a BF16 router would round it on load (T3 failure).

    The fixture gate is redrawn at full FP32 precision (not BF16-representable), so a router
    cast to params_dtype fails the dtype and bit-exact checks. Runs in BF16 to also exercise
    the gating upcast (TE's cuBLASLt GEMM rejects FP32 weight x BF16 input).
    """
    from test_glm5_next import _parallel_context

    from mcore_bridge.utils.safetensors import SafetensorLazyLoader

    hf_model = _build_hf(remote_dir, dtype=torch.bfloat16)
    canonical = _checkpoint_state(hf_model)
    qdir = str(tmp_path / 'quantized')
    _write_quantized_fixture(qdir, canonical, hf_model.config, _V25_QUANT_CONFIG)
    with SafetensorLazyLoader(qdir) as loader:
        state = loader.get_state_dict()
    gate_key = 'model.layers.1.mlp.gate.weight'
    gate = state[gate_key].load()
    assert gate.dtype == torch.float32
    # Non-vacuity: the fixture gate must carry precision a BF16 router would destroy.
    assert not torch.equal(gate, gate.to(torch.bfloat16).float())
    with _parallel_context():
        hf_config = AutoConfig.from_pretrained(qdir, trust_remote_code=True)
        config = _mcore_config(hf_config, dtype=torch.bfloat16)
        model = get_mcore_model(config)[0].cuda().eval()
        router = model.language_model.decoder.layers[1].mlp.router
        assert isinstance(router, mimo_v2.MiMoV2TopKRouter)
        assert router.weight.dtype == torch.bfloat16  # builds in params_dtype; load upcasts
        list(config.bridge._convert([model], state, '', True, 'Loading: '))
        assert router.weight.dtype == torch.float32
        assert getattr(router.weight, 'keep_in_fp32', False)
        torch.testing.assert_close(router.weight.data, gate.to(router.weight.device), atol=0, rtol=0)
        exported = _export(config, model)
        assert exported[gate_key].dtype == torch.float32
        torch.testing.assert_close(exported[gate_key].cpu(), gate.cpu(), atol=0, rtol=0)
        input_ids, position_ids, attention_mask = _model_inputs()
        logits = _mcore_logits(model, config, input_ids, position_ids, attention_mask)
        assert torch.isfinite(logits).all()
        # Flash-layout control: a BF16 gate keeps the router BF16 (no regression).
        fdir = str(tmp_path / 'flash')
        _write_quantized_fixture(fdir, canonical, hf_model.config)
        fconfig = _mcore_config(AutoConfig.from_pretrained(fdir, trust_remote_code=True), dtype=torch.bfloat16)
        fmodel = get_mcore_model(fconfig)[0].cuda().eval()
        with SafetensorLazyLoader(fdir) as floader:
            fstate = floader.get_state_dict()
        list(fconfig.bridge._convert([fmodel], fstate, '', True, 'Loading: '))
        assert fmodel.language_model.decoder.layers[1].mlp.router.weight.dtype == torch.bfloat16


@requires_remote_code
def test_router_fp32_survives_nonlocal_config(remote_dir, tmp_path, monkeypatch):
    """The gate dtype is read from the weights at load time, not probed from name_or_path:
    a repo-id (non-local) config must not silently round an FP32 gate to BF16."""
    from test_glm5_next import _parallel_context

    from mcore_bridge.utils.safetensors import SafetensorLazyLoader

    hf_model = _build_hf(remote_dir, dtype=torch.bfloat16)
    canonical = _checkpoint_state(hf_model)
    qdir = str(tmp_path / 'quantized')
    _write_quantized_fixture(qdir, canonical, hf_model.config, _V25_QUANT_CONFIG)
    with SafetensorLazyLoader(qdir) as loader:
        state = loader.get_state_dict()
    gate_key = 'model.layers.1.mlp.gate.weight'
    gate = state[gate_key].load()
    assert gate.dtype == torch.float32
    with _parallel_context():
        hf_config = AutoConfig.from_pretrained(qdir, trust_remote_code=True)
        config = _mcore_config(hf_config, dtype=torch.bfloat16)
        # Repo id AFTER parsing: the parser re-reads config.json via name_or_path (pre-existing
        # behavior; swift resolves hub configs to a local cache first). The load path must not
        # probe name_or_path for the gate dtype.
        hf_config.name_or_path = 'XiaomiMiMo/MiMo-V2.5'
        model = get_mcore_model(config)[0].cuda().eval()
        list(config.bridge._convert([model], state, '', True, 'Loading: '))
        router = model.language_model.decoder.layers[1].mlp.router
        assert router.weight.dtype == torch.float32
        torch.testing.assert_close(router.weight.data, gate.to(router.weight.device), atol=0, rtol=0)


@requires_remote_code
def test_router_fp32_fails_loudly_without_keep_marker(remote_dir, tmp_path, monkeypatch):
    """On megatron-core<0.20 Float16Module casts unconditionally; loading an FP32 gate there
    must fail loudly instead of silently rounding it."""
    from test_glm5_next import _parallel_context

    from mcore_bridge.utils.safetensors import SafetensorLazyLoader

    hf_model = _build_hf(remote_dir, dtype=torch.bfloat16)
    canonical = _checkpoint_state(hf_model)
    qdir = str(tmp_path / 'quantized')
    _write_quantized_fixture(qdir, canonical, hf_model.config, _V25_QUANT_CONFIG)
    with SafetensorLazyLoader(qdir) as loader:
        state = loader.get_state_dict()
    with _parallel_context():
        hf_config = AutoConfig.from_pretrained(qdir, trust_remote_code=True)
        config = _mcore_config(hf_config, dtype=torch.bfloat16)
        model = get_mcore_model(config)[0].cuda().eval()
        monkeypatch.setattr(mimo_v2, '_HAS_KEEP_IN_FP32', False)
        with pytest.raises(ValueError, match='mark_keep_in_fp32'):
            list(config.bridge._convert([model], state, '', True, 'Loading: '))


@requires_remote_code
def test_swa_without_sink_flag_has_no_learnable_offset(remote_dir):
    """A sink-less variant must not build a learnable softmax offset: TE would append an
    unloaded sink column to the logits (softmax denominator corruption)."""
    from test_glm5_next import _parallel_context

    hf_config = _hf_config(remote_dir, add_swa_attention_sink_bias=False)
    with _parallel_context():
        config = _mcore_config(hf_config)
        model = get_mcore_model(config)[0].cuda().eval()
        lm = _lm(model)
        for idx, layer in enumerate(lm.decoder.layers):
            core = layer.self_attention.core_attention
            assert getattr(core, 'softmax_offset', None) is None, f'layer{idx}: sink built without the flag'
        input_ids, position_ids, attention_mask = _model_inputs()
        logits = _mcore_logits(model, config, input_ids, position_ids, attention_mask)
        assert torch.isfinite(logits).all()


@requires_remote_code
def test_online_dequant_expert_subset_truncates_router(remote_dir, tmp_path):
    """Expert-subset models read experts.0..N-1; the router gate/bias must be truncated to the
    same prefix (mirrors the convert tool's --num-experts). Pro e64 T3 failed without it."""
    from test_glm5_next import _parallel_context

    from mcore_bridge.utils.safetensors import SafetensorLazyLoader

    hf_model = _build_hf(remote_dir, dtype=torch.bfloat16)
    canonical = _checkpoint_state(hf_model)
    qdir = str(tmp_path / 'quantized')
    _write_quantized_fixture(qdir, canonical, hf_model.config)
    with SafetensorLazyLoader(qdir) as loader:
        state = loader.get_state_dict()
    gate = state['model.layers.1.mlp.gate.weight'].load()
    bias = state['model.layers.1.mlp.gate.e_score_correction_bias'].load()
    n_sub = gate.shape[0] // 2
    assert gate.shape[0] > n_sub  # non-vacuous: the checkpoint is wider than the model
    with _parallel_context():
        hf_config = AutoConfig.from_pretrained(qdir, trust_remote_code=True)
        hf_config.n_routed_experts = n_sub
        config = _mcore_config(hf_config)
        model = get_mcore_model(config)[0].cuda().eval()
        list(config.bridge._convert([model], state, '', True, 'Loading: '))
        router = model.language_model.decoder.layers[1].mlp.router
        torch.testing.assert_close(
            router.weight.data, gate[:n_sub].to(device=router.weight.device, dtype=router.weight.dtype), atol=0, rtol=0)
        torch.testing.assert_close(
            router.expert_bias,
            bias[:n_sub].to(device=router.expert_bias.device, dtype=router.expert_bias.dtype),
            atol=0,
            rtol=0)
        exported = _export(config, model)
        assert exported['model.layers.1.mlp.gate.weight'].shape[0] == n_sub
        assert exported['model.layers.1.mlp.gate.e_score_correction_bias'].shape[0] == n_sub


@requires_remote_code
def test_parallel_tp2_online_dequant_matches_offline_tool(remote_dir, tmp_path):
    if os.environ.get('MIMO_PARALLEL_TEST') != 'tp2':
        pytest.skip('run with MIMO_PARALLEL_TEST=tp2 torchrun --nproc-per-node=2')
    from test_glm5_next import _parallel_context

    with _parallel_context(tp=2):
        _check_online_matches_offline(remote_dir, tmp_path, tp=2, mtp=1)


@requires_remote_code
def test_online_dequant_rejects_bad_layout_and_missing_scales(remote_dir, tmp_path):
    """T4: a forged quantized checkpoint must fail loudly, never be read as plain BF16."""
    from test_glm5_next import _parallel_context

    from mcore_bridge.utils.safetensors import SafetensorLazyLoader

    hf_model = _build_hf(remote_dir, dtype=torch.bfloat16)
    mtp_state, _ = _mtp_state(hf_model, remote_dir)
    qdir = str(tmp_path / 'quantized')
    _write_quantized_fixture(qdir, _checkpoint_state(hf_model, mtp_state), hf_model.config)
    with SafetensorLazyLoader(qdir) as loader:
        state = loader.get_state_dict()
    hf_config = AutoConfig.from_pretrained(qdir, trust_remote_code=True)
    with pytest.raises(ValueError, match='only supports the official'):
        mimo_v2._QuantizedStateDict(state, hf_config, dict(_OFFICIAL_QUANT_CONFIG, quant_method='awq'))
    # Layout signature negative controls: unknown MXFP4 block size, and neither MXFP4 nor the
    # all-FP8 marker (store_dtype=fp8) declared.
    with pytest.raises(ValueError, match='only supports the official'):
        mimo_v2._QuantizedStateDict(state, hf_config, dict(_OFFICIAL_QUANT_CONFIG, mxfp4_block_size=16))
    with pytest.raises(ValueError, match='only supports the official'):
        mimo_v2._QuantizedStateDict(state, hf_config, {
            k: v
            for k, v in _OFFICIAL_QUANT_CONFIG.items() if k != 'mxfp4_block_size'
        })
    for dropped_key, match in [
        ('model.layers.0.self_attn.qkv_proj.weight_scale_inv', 'missing FP8 scale'),
        ('model.layers.0.mlp.gate_proj.weight_scale_inv', 'missing FP8 scale'),
        ('model.mtp.layers.0.mlp.down_proj.weight_scale_inv', 'missing FP8 scale'),
        ('model.layers.1.mlp.experts.0.gate_proj.weight_scale', 'missing MXFP4 scale'),
    ]:
        dropped = {k: v for k, v in state.items() if k != dropped_key}
        with pytest.raises(AssertionError, match=match):
            mimo_v2._QuantizedStateDict(dropped, hf_config, dict(_OFFICIAL_QUANT_CONFIG))
    # End-to-end: the load path itself raises, not just the adapter constructor.
    dropped = {k: v for k, v in state.items() if k != 'model.layers.0.self_attn.qkv_proj.weight_scale_inv'}
    with _parallel_context():
        config = _mcore_config(hf_config, mtp=0)
        model = get_mcore_model(config)[0].cuda().eval()
        with pytest.raises(AssertionError, match='missing FP8 scale'):
            list(config.bridge._convert([model], dropped, '', True, 'Loading: '))


@requires_remote_code
def test_online_dequant_v25_scale_audit(remote_dir, tmp_path):
    """T4-V25: the all-FP8 layout audits its own scale flavor; mixed layouts fail loudly."""
    from mcore_bridge.utils.safetensors import SafetensorLazyLoader

    hf_model = _build_hf(remote_dir, dtype=torch.bfloat16)
    mtp_state, _ = _mtp_state(hf_model, remote_dir)
    canonical = _checkpoint_state(hf_model, mtp_state)
    v25dir = str(tmp_path / 'v25')
    _write_quantized_fixture(v25dir, canonical, hf_model.config, _V25_QUANT_CONFIG)
    with SafetensorLazyLoader(v25dir) as loader:
        v25_state = loader.get_state_dict()
    hf_config = AutoConfig.from_pretrained(v25dir, trust_remote_code=True)
    expert = 'model.layers.1.mlp.experts.0.gate_proj.weight'
    # Missing FP8 scale on an expert.
    dropped = {k: v for k, v in v25_state.items() if k != expert + '_scale_inv'}
    with pytest.raises(AssertionError, match='missing FP8 scale'):
        mimo_v2._QuantizedStateDict(dropped, hf_config, dict(_V25_QUANT_CONFIG))
    # Declared MXFP4 but the tensors carry FP8 scales (and vice versa): never dequantize
    # with the wrong rule, even though the loader dispatch alone would pick a flavor.
    with pytest.raises(AssertionError, match='missing MXFP4 scale'):
        mimo_v2._QuantizedStateDict(v25_state, hf_config, dict(_OFFICIAL_QUANT_CONFIG))
    fp8dir = str(tmp_path / 'flash')
    _write_quantized_fixture(fp8dir, canonical, hf_model.config)
    with SafetensorLazyLoader(fp8dir) as loader:
        fp8_state = loader.get_state_dict()
    with pytest.raises(AssertionError, match='missing FP8 scale'):
        mimo_v2._QuantizedStateDict(fp8_state, hf_config, dict(_V25_QUANT_CONFIG))
    # Both scale flavors present on one expert -> mixed layout, rejected in either declaration.
    mixed = dict(v25_state)
    mixed[expert.removesuffix('weight') + 'weight_scale'] = v25_state[expert + '_scale_inv']
    with pytest.raises(AssertionError, match='unexpected MXFP4 scale'):
        mimo_v2._QuantizedStateDict(mixed, hf_config, dict(_V25_QUANT_CONFIG))
    mixed = dict(fp8_state)
    mixed[expert + '_scale_inv'] = fp8_state[expert.removesuffix('weight') + 'weight_scale']
    with pytest.raises(AssertionError, match='unexpected FP8 scale'):
        mimo_v2._QuantizedStateDict(mixed, hf_config, dict(_OFFICIAL_QUANT_CONFIG))


@requires_remote_code
def test_online_dequant_passthrough_rejects_quantized_dtype(remote_dir, tmp_path):
    """T4-dtype: a quantized-dtype tensor without a scale key must fail loudly, not be cast."""
    from mcore_bridge.utils.safetensors import LazyTensor, SafetensorLazyLoader

    hf_model = _build_hf(remote_dir, dtype=torch.bfloat16)
    mtp_state, _ = _mtp_state(hf_model, remote_dir)
    qdir = str(tmp_path / 'quantized')
    _write_quantized_fixture(qdir, _checkpoint_state(hf_model, mtp_state), hf_model.config)
    with SafetensorLazyLoader(qdir) as loader:
        state = loader.get_state_dict()
    hf_config = AutoConfig.from_pretrained(qdir, trust_remote_code=True)
    # o_proj is on the ignored list (plain BF16 passthrough). Forge FP8 bytes / packed uint8
    # in its place: param.copy_ would silently cast them to BF16 without any scale.
    key = 'model.layers.0.self_attn.o_proj.weight'
    shape = state[key].load().shape
    for bad in (torch.zeros(shape, dtype=torch.float8_e4m3fn), torch.zeros(shape, dtype=torch.uint8)):
        forged = dict(state)
        forged[key] = LazyTensor(tensor=bad)
        adapter = mimo_v2._QuantizedStateDict(forged, hf_config, dict(_OFFICIAL_QUANT_CONFIG))
        with pytest.raises(AssertionError, match='quantized dtype'):
            adapter[key].load()
    # Positive control: the untouched BF16 tensor passes through unchanged.
    adapter = mimo_v2._QuantizedStateDict(state, hf_config, dict(_OFFICIAL_QUANT_CONFIG))
    assert adapter[key].load().dtype == torch.bfloat16


@requires_remote_code
def test_quantized_input_still_rejects_fp8_training(remote_dir):
    from test_glm5_next import _parallel_context

    config = _hf_config(
        remote_dir, attention_projection_layout='fused_qkv', quantization_config=dict(_OFFICIAL_QUANT_CONFIG))
    values = hf_to_mcore_config(config)
    values.update(mcore_model_type='mimo_v2', language_model_only=True, fp8='hybrid')
    with _parallel_context():
        mcore_config = ModelConfig(**values)
        with pytest.raises(ValueError, match='FP8/FP4'):
            mimo_v2.MiMoV2Loader(mcore_config)


@requires_remote_code
def test_export_rejects_fp8_weights(remote_dir):
    from test_glm5_next import _parallel_context

    with _parallel_context():
        config = _mcore_config(_hf_config(remote_dir))
    bridge = config.bridge
    attn = NS(linear_qkv=NS(weight=NS(data=torch.zeros(352, 64, device='cuda'))))
    bridge._get_weight = lambda *args, **kwargs: (torch.zeros(352, 64, device='cuda'), torch.ones(3, 1, device='cuda'))
    with pytest.raises(ValueError, match='FP8 export is unsupported'):
        bridge._set_qkv(attn, {}, False, layer_idx=0)


@requires_remote_code
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_forward_parity(remote_dir, dtype):
    from test_glm5_next import _parallel_context

    with _parallel_context():
        hf_model, model, _, config, _ = _build_parity_models(remote_dir, dtype=dtype)
        input_ids, position_ids, attention_mask = _model_inputs()
        expected = _hf_logits(hf_model, input_ids).float()
        actual = _mcore_logits(model, config, input_ids, position_ids, attention_mask).float()
        atol = 3e-4 if dtype == torch.float32 else 2e-2
        torch.testing.assert_close(actual, expected, atol=atol, rtol=atol)


@requires_remote_code
def test_gradient_parity_fp32(remote_dir):
    from test_glm5_next import _parallel_context

    with _parallel_context():
        # mtp=0 isolates the backbone; the HF gate raises in train mode, so HF stays eval().
        hf_model, model, _, config, _ = _build_parity_models(remote_dir, mtp=0)
        lm = _lm(model)
        input_ids, position_ids, attention_mask = _model_inputs()
        labels = torch.randint(
            1, 128, input_ids.shape, device='cuda', generator=torch.Generator(device='cuda').manual_seed(5))

        hf_logits = _hf_logits(hf_model, input_ids)
        hf_loss = F.cross_entropy(hf_logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten())
        hf_loss.backward()

        mg_logits = _mcore_logits(model, config, input_ids, position_ids, attention_mask)
        mg_loss = F.cross_entropy(mg_logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten())
        mg_loss.backward()
        torch.testing.assert_close(mg_loss, hf_loss, atol=3e-4, rtol=3e-4)

        def fused_qkv_grads(layer_idx):
            layer = lm.decoder.layers[layer_idx]
            groups = 4 if _PATTERN[layer_idx] else 2
            grad = layer.self_attention.linear_qkv.weight.grad
            q, k, v = mimo_v2._split_qkv(grad, dim=0, groups=groups, heads_per_group=8 // groups, qk_dim=32, v_dim=16)
            return q.reshape(-1, 64), k.reshape(-1, 64), v.reshape(-1, 64)

        pairs = [('embed_tokens', hf_model.model.embed_tokens.weight.grad, lm.embedding.word_embeddings.weight.grad),
                 ('lm_head', hf_model.lm_head.weight.grad, lm.output_layer.weight.grad),
                 ('final_norm', hf_model.model.norm.weight.grad, lm.decoder.final_layernorm.weight.grad)]
        for i in (0, 1):
            hf_layer = hf_model.model.layers[i]
            mg_layer = lm.decoder.layers[i]
            q, k, v = fused_qkv_grads(i)
            pairs += [
                (f'l{i}.q', hf_layer.self_attn.q_proj.weight.grad, q),
                (f'l{i}.k', hf_layer.self_attn.k_proj.weight.grad, k),
                (f'l{i}.v', hf_layer.self_attn.v_proj.weight.grad, v),
                (f'l{i}.o', hf_layer.self_attn.o_proj.weight.grad, mg_layer.self_attention.linear_proj.weight.grad),
                (f'l{i}.input_ln', hf_layer.input_layernorm.weight.grad,
                 mg_layer.self_attention.linear_qkv.layer_norm_weight.grad),
            ]
            if i == 0:  # dense
                fc1 = mg_layer.mlp.linear_fc1.weight.grad
                pairs += [
                    ('l0.gate', hf_layer.mlp.gate_proj.weight.grad, fc1[:64]),
                    ('l0.up', hf_layer.mlp.up_proj.weight.grad, fc1[64:]),
                    ('l0.down', hf_layer.mlp.down_proj.weight.grad, mg_layer.mlp.linear_fc2.weight.grad),
                    ('l0.post_ln', hf_layer.post_attention_layernorm.weight.grad,
                     mg_layer.mlp.linear_fc1.layer_norm_weight.grad),
                ]
            else:  # MoE (grouped): router + expert 0 + pre-MLP norm
                pairs += [
                    ('l1.router', hf_layer.mlp.gate.weight.grad, mg_layer.mlp.router.weight.grad),
                    ('l1.pre_ln', hf_layer.post_attention_layernorm.weight.grad,
                     mg_layer.pre_mlp_layernorm.weight.grad),
                ]
                experts = mg_layer.mlp.experts
                fc1_mod = getattr(experts, 'linear_fc1', None)
                if fc1_mod is not None and hasattr(fc1_mod, 'weight0'):  # TE grouped: per-expert weight{i}
                    fc1, fc2 = fc1_mod.weight0.grad, experts.linear_fc2.weight0.grad
                elif fc1_mod is not None:
                    fc1, fc2 = fc1_mod.weight.grad, experts.linear_fc2.weight.grad
                    if fc1.dim() == 3:  # grouped: [num_experts, out, in]
                        fc1, fc2 = fc1[0], fc2[0]
                else:  # sequential fallback
                    fc1 = experts[0].linear_fc1.weight.grad
                    fc2 = experts[0].linear_fc2.weight.grad
                pairs += [
                    ('l1.e0.gate', hf_layer.mlp.experts[0].gate_proj.weight.grad, fc1[:32]),
                    ('l1.e0.up', hf_layer.mlp.experts[0].up_proj.weight.grad, fc1[32:]),
                    ('l1.e0.down', hf_layer.mlp.experts[0].down_proj.weight.grad, fc2),
                ]
        for name, expected, actual in pairs:
            assert actual is not None, f'{name}: mcore grad missing'
            torch.testing.assert_close(
                actual.float(), expected.float(), atol=3e-4, rtol=3e-4, msg=lambda m: f'{name}: {m}')
        # Sinks are frozen on both sides (HF requires_grad=False; mcore frozen at build).
        for i in (1, 2, 4, 5):
            assert hf_model.model.layers[i].self_attn.attention_sink_bias.grad is None
            offset = lm.decoder.layers[i].self_attention.core_attention.softmax_offset
            assert not offset.requires_grad and offset.grad is None


@requires_remote_code
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_roundtrip_bit_exact_and_reimport(remote_dir, dtype):
    from test_glm5_next import _parallel_context

    with _parallel_context():
        hf_model, model, checkpoint, config, _ = _build_parity_models(remote_dir, mtp=1, dtype=dtype)
        exported = _export(config, model)
        assert set(exported) == set(checkpoint), (
            f'missing={set(checkpoint) - set(exported)} extra={set(exported) - set(checkpoint)}')
        for key in checkpoint:
            # dtype included: a BF16 export must not mix in FP32 tensors (e.g. TE's FP32 sink).
            # Exception: expert_bias is deliberately F32-stored (as in the official checkpoint).
            if key.endswith('e_score_correction_bias'):
                assert exported[key].dtype == torch.float32, f'{key}: expert_bias must stay F32'
                torch.testing.assert_close(
                    exported[key], checkpoint[key].float(), atol=0, rtol=0, msg=lambda m: f'{key}: {m}')
                continue
            assert exported[key].dtype == checkpoint[key].dtype, f'{key}: dtype {exported[key].dtype}'
            torch.testing.assert_close(exported[key], checkpoint[key], atol=0, rtol=0, msg=lambda m: f'{key}: {m}')
        # SWA-only sinks: full layers must not carry the key.
        for i in (0, 3):
            assert f'model.layers.{i}.self_attn.attention_sink_bias' not in exported
        for i in (1, 2, 4, 5):
            assert f'model.layers.{i}.self_attn.attention_sink_bias' in exported
        assert 'model.mtp.layers.0.eh_proj.weight' in exported
        assert exported['model.mtp.layers.0.eh_proj.weight'].shape == (64, 128)

        input_ids, position_ids, attention_mask = _model_inputs()
        expected = _mcore_logits(model, config, input_ids, position_ids, attention_mask)
        reloaded = get_mcore_model(config)[0].cuda().eval()
        lazy = {key: LazyTensor(tensor=value) for key, value in exported.items()}
        list(config.bridge._convert([reloaded], lazy, '', True, 'Reloading test: '))
        actual = _mcore_logits(reloaded, config, input_ids, position_ids, attention_mask)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@requires_remote_code
def test_sink_negative_control(remote_dir):
    """A perturbed sink must change the logits: proves the parity test is not vacuous."""
    from test_glm5_next import _parallel_context

    with _parallel_context():
        hf_model, model, _, config, _ = _build_parity_models(remote_dir)
        input_ids, position_ids, attention_mask = _model_inputs()
        baseline = _mcore_logits(model, config, input_ids, position_ids, attention_mask).float()
        with torch.no_grad():
            hf_model.model.layers[1].self_attn.attention_sink_bias += 1.0
        perturbed = _hf_logits(hf_model, input_ids).float()
        assert (perturbed - baseline).abs().max().item() > 1e-3


def _hf_hidden_before_norm(hf_model, input_ids):
    """Run the HF decoder manually to expose the pre-final-norm hidden states."""
    model = hf_model.model
    attention_mask = torch.ones_like(input_ids)
    inputs_embeds = model.embed_tokens(input_ids)
    position_ids = torch.arange(input_ids.shape[1], device='cuda').unsqueeze(0).expand(input_ids.shape[0], -1)
    cache_position = torch.arange(input_ids.shape[1], device='cuda')
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    mask_kwargs = dict(
        config=model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=position_ids)
    masks = {
        'full_attention': create_causal_mask(**mask_kwargs),
        'sliding_window_attention': create_sliding_window_causal_mask(**mask_kwargs),
    }
    rope = model.rotary_emb(inputs_embeds, position_ids)
    swa_rope = model.swa_rotary_emb(inputs_embeds, position_ids)
    hidden = inputs_embeds
    for layer in model.layers:
        hidden = layer(
            hidden,
            attention_mask=masks[layer.attention_type],
            position_ids=position_ids,
            position_embeddings=rope if layer.attention_type == 'full_attention' else swa_rope,
            use_cache=False)
    return hidden, position_ids, masks['sliding_window_attention'], swa_rope


# Inferred from the mcore implementation, NOT from an upstream oracle: the HF remote code lists
# model.mtp.* in _keys_to_ignore_on_load_unexpected, so no HF MTP forward exists to compare against.
_MTP_USES_POST_NORM = [True]  # resolved empirically by test_mtp_logits_parity


def _mtp_reference_logits(hf_model, reference, input_ids, h_before_norm, h_after_norm, position_ids, swa_mask):
    """enorm(embed(x_{t+1})) + hnorm(h_t) -> eh_proj -> SWA dense layer -> final_ln -> lm_head.

    Megatron 0.20 applies the decoder final norm inside the last decoder layer when MTP is on,
    so the MTP block conditions on the POST-norm hidden; the caller picks the matching tensor.
    """
    next_ids = torch.roll(input_ids, -1, dims=-1)
    embed_next = hf_model.model.embed_tokens(next_ids)
    h = h_after_norm if _MTP_USES_POST_NORM[0] else h_before_norm
    x = reference.eh_proj(torch.cat([reference.enorm(embed_next), reference.hnorm(h)], dim=-1))
    shifted_positions = position_ids + 1
    rope = hf_model.model.swa_rotary_emb(x, shifted_positions)
    x = reference.layer(
        x, attention_mask=swa_mask, position_ids=shifted_positions, position_embeddings=rope, use_cache=False)
    x = reference.final_layernorm(x)
    return F.linear(x, hf_model.lm_head.weight)


@requires_remote_code
def test_mtp_logits_parity(remote_dir):
    """Self-consistency + formula check, NOT an upstream parity: the HF remote code ignores
    model.mtp.* on load, so the reference is hand-built from HF MiMoV2DecoderLayer (SWA + dense)
    plus enorm/hnorm/eh_proj/final_layernorm, and the post-norm choice is implementation-inferred.
    """
    from test_glm5_next import _parallel_context

    with _parallel_context():
        hf_model, model, _, config, reference = _build_parity_models(remote_dir, mtp=1)
        lm = _lm(model)
        # Where the decoder final norm lives decides what hnorm conditions on.
        _MTP_USES_POST_NORM[0] = lm.decoder.final_layernorm is not None
        input_ids, position_ids, attention_mask = _model_inputs()
        h_before, pos, swa_mask, _ = _hf_hidden_before_norm(hf_model, input_ids)
        h_after = hf_model.model.norm(h_before)
        expected = _mtp_reference_logits(hf_model, reference, input_ids, h_before, h_after, pos, swa_mask)

        # Direct MTP call, mirroring GPTModel._postprocess (no loss-alignment ambiguity).
        embed = lm.embedding(input_ids, position_ids)
        rope_dict, _, _ = lm._get_rotary_pos_emb(embed, position_ids, None)
        decoder_out = lm.decoder(hidden_states=embed, attention_mask=attention_mask, rotary_pos_emb=rope_dict)
        mtp_out = lm.mtp(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=decoder_out,
            attention_mask=attention_mask,
            rotary_pos_emb=rope_dict,
            embedding=lm.embedding)
        # Upstream cats decoder_hidden with per-depth MTP outputs; draft logits are depth-1 only.
        seq = input_ids.shape[1]
        mtp_hidden = mtp_out[seq:]
        actual = F.linear(mtp_hidden.transpose(0, 1).contiguous(), lm.output_layer.weight)
        # The last position wraps around (roll) and is masked out of the MTP loss upstream.
        torch.testing.assert_close(actual[:, :-1], expected[:, :-1], atol=3e-4, rtol=3e-4)


@requires_remote_code
def test_mtp_loss_backward_flows_grads(remote_dir):
    from test_glm5_next import _parallel_context

    with _parallel_context():
        hf_model, model, _, config, _ = _build_parity_models(remote_dir, mtp=1)
        lm = _lm(model).train()
        input_ids, position_ids, attention_mask = _model_inputs()
        labels = input_ids.clone()
        loss_mask = torch.ones_like(labels, dtype=torch.bool)
        out = model(input_ids, position_ids, attention_mask, labels=labels, loss_mask=loss_mask)
        loss = out if out.dim() == 0 else out.float().mean()
        assert torch.isfinite(loss).item()
        loss.backward()
        missing = [n for n, p in lm.mtp.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, f'MTP params without grad: {missing}'
        head = dict(lm.mtp.named_parameters())
        for probe in ('layers.0.eh_proj.weight', 'layers.0.enorm.weight', 'layers.0.hnorm.weight'):
            assert head[probe].grad is not None and head[probe].grad.abs().sum() > 0
        assert config.mtp_loss_scaling_factor == 0.1


_LORA_TARGETS = ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2', 'eh_proj']
# Decoder-only regex: excludes the MTP subtree, for the frozen-base-MTP reload check.
_LORA_DECODER_REGEX = (r'.*decoder\.layers\.\d+\.(self_attention\.linear_(qkv|proj)'
                       r'|mlp\.(linear_fc[12]|experts\.linear_fc[12]))')


def _wrap_lora(model, decoder_only=False):
    from peft import LoraConfig, get_peft_model

    target = _LORA_DECODER_REGEX if decoder_only else _LORA_TARGETS
    return get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=target, lora_dropout=0.))


def _lora_logits(model, config):
    model.eval()
    input_ids, position_ids, attention_mask = _model_inputs()
    with torch.no_grad():
        logits = model(input_ids, position_ids, attention_mask, runtime_gather_output=True)
    return logits[..., :config.actual_vocab_size].cpu()


def _train_steps(peft_model, config, steps=3):
    """Real optimizer steps so adapter weights (and only they) move; returns before/after state."""
    lm = _lm(peft_model)
    bias_before = {
        i: layer.mlp.router.expert_bias.detach().clone()
        for i, layer in enumerate(lm.decoder.layers) if hasattr(layer.mlp, 'router')
    }
    sink_before = {
        i: layer.self_attention.core_attention.softmax_offset.detach().clone()
        for i, layer in enumerate(lm.decoder.layers) if _PATTERN[i]
    }
    lora_params = [p for n, p in peft_model.named_parameters() if 'lora_' in n and p.requires_grad]
    assert lora_params, 'no trainable LoRA parameters'
    lora_before = [p.detach().clone() for p in lora_params]
    optimizer = torch.optim.SGD(lora_params, lr=0.5)
    peft_model.train()
    input_ids, position_ids, attention_mask = _model_inputs()
    labels = input_ids.clone()
    loss_mask = torch.ones_like(labels, dtype=torch.bool)
    for _ in range(steps):
        optimizer.zero_grad()
        out = peft_model(input_ids, position_ids, attention_mask, labels=labels, loss_mask=loss_mask)
        loss = out if out.dim() == 0 else out.float().mean()
        loss.backward()
        optimizer.step()
    # expert_bias (update rate 0) and the frozen sinks must not move during training.
    for i, bias in bias_before.items():
        torch.testing.assert_close(lm.decoder.layers[i].mlp.router.expert_bias, bias, atol=0, rtol=0)
    for i, sink in sink_before.items():
        torch.testing.assert_close(
            lm.decoder.layers[i].self_attention.core_attention.softmax_offset, sink, atol=0, rtol=0)
    assert any(not torch.equal(p, before) for p, before in zip(lora_params, lora_before)), 'LoRA did not train'


def _save_lora_fixture(output_dir, decoder_only=False, steps=0):
    from safetensors.torch import save_file
    from test_glm5_next import _parallel_context

    with _parallel_context():
        _, model, _, config, _ = _build_parity_models(remote_dir=_REMOTE_FIXTURE_DIR[0], mtp=1)
        base = {key: value.detach().cpu() for key, value in _export(config, model).items()}
        save_file(base, os.path.join(output_dir, 'base.safetensors'))
        peft_model = _wrap_lora(model, decoder_only=decoder_only)
        for name, parameter in peft_model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                torch.nn.init.normal_(parameter, std=0.1)
        if steps:
            _train_steps(peft_model, config, steps=steps)
        torch.save(_lora_logits(peft_model, config), os.path.join(output_dir, 'expected.pt'))
        adapter_dir = os.path.join(output_dir, 'adapter')
        config.bridge.save_weights([peft_model], adapter_dir, peft_format=True)
        peft_model.peft_config['default'].save_pretrained(adapter_dir)
        torch.save(decoder_only, os.path.join(output_dir, 'decoder_only.pt'))


def _check_lora_fixture(output_dir, drop_v_lora_b=False):
    from peft import PeftConfig, get_peft_model
    from safetensors.torch import load_file, save_file
    from test_glm5_next import _parallel_context

    with _parallel_context():
        _, model, _, config, _ = _build_parity_models(remote_dir=_REMOTE_FIXTURE_DIR[0], mtp=1)
        base = {
            key: LazyTensor(tensor=value)
            for key, value in load_file(os.path.join(output_dir, 'base.safetensors')).items()
        }
        list(config.bridge._convert([model], base, '', True, 'Loading base: '))
        adapter_dir = os.path.join(output_dir, 'adapter')
        if drop_v_lora_b:
            # Negative control: zero the asymmetric V LoRA factor (the loader requires every
            # adapter key present, so zero instead of delete).
            path = os.path.join(adapter_dir, 'adapter_model.safetensors')
            state = load_file(path)
            dropped = [key for key in state if key.endswith('v_proj.lora_B.weight')]
            assert dropped, 'adapter carries no v_proj LoRA B to zero'
            for key in dropped:
                state[key] = torch.zeros_like(state[key])
            save_file(state, path)
        peft_model = get_peft_model(model, PeftConfig.from_pretrained(adapter_dir))
        config.bridge.load_weights([peft_model], adapter_dir, peft_format=True)
        actual = _lora_logits(peft_model, config)
        expected = torch.load(os.path.join(output_dir, 'expected.pt'), weights_only=True)
        if drop_v_lora_b:
            assert not torch.allclose(actual, expected, atol=1e-4), 'zeroing v_proj LoRA B had no effect'
            return
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        # An adapter without the MTP subtree must leave the frozen base MTP intact.
        decoder_only = torch.load(os.path.join(output_dir, 'decoder_only.pt'), weights_only=True)
        if decoder_only:
            base_state = load_file(os.path.join(output_dir, 'base.safetensors'))
            # Reset peft mode or the full-base export skips non-LoRA modules (the frozen MTP).
            config.bridge._peft_format = False
            try:
                exported = _export(config, peft_model.base_model.model)
            finally:
                config.bridge._peft_format = True
            for key, value in base_state.items():
                if key.startswith('model.mtp.'):
                    torch.testing.assert_close(exported[key].cpu(), value, atol=0, rtol=0)


_REMOTE_FIXTURE_DIR = [None]  # set by the fresh-process entrypoint below


def test_lora_adapter_save_reload_fresh_process(tmp_path):
    if not torch.cuda.is_available() or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        pytest.skip('requires one CUDA device')
    if not _remote_code_available():
        pytest.skip('MiMo-V2 HF remote code not found')
    root = os.path.dirname(os.path.dirname(__file__))
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([
        os.path.join(root, 'src'),
        os.path.join(root, 'tests'), '/mnt/workspace/hjh/.pkgs/pytest_env', '/mnt/workspace/hjh/.pkgs/antlr49',
        env.get('PYTHONPATH', '')
    ])
    env['NCCL_DEBUG'] = 'WARN'
    env['MIMO_V2_REMOTE_DIR'] = _REMOTE_DIR
    for mode in ('save', 'check'):
        result = subprocess.run([sys.executable, __file__, mode, str(tmp_path)],
                                env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                timeout=600)
        assert result.returncode == 0, result.stdout


def test_lora_trained_and_decoder_only_fresh_process(tmp_path):
    """3 real optimizer steps, plus a decoder-only adapter whose reload must keep the base MTP
    byte-identical; a third pass drops v_proj LoRA B as a negative control."""
    if not torch.cuda.is_available() or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        pytest.skip('requires one CUDA device')
    if not _remote_code_available():
        pytest.skip('MiMo-V2 HF remote code not found')
    root = os.path.dirname(os.path.dirname(__file__))
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([
        os.path.join(root, 'src'),
        os.path.join(root, 'tests'), '/mnt/workspace/hjh/.pkgs/pytest_env', '/mnt/workspace/hjh/.pkgs/antlr49',
        env.get('PYTHONPATH', '')
    ])
    env['NCCL_DEBUG'] = 'WARN'
    env['MIMO_V2_REMOTE_DIR'] = _REMOTE_DIR
    for mode, subdir in (('save_trained', 'trained'), ('check', 'trained'), ('save_decoder_only', 'dec'),
                         ('check', 'dec'), ('check_drop_v', 'dec')):
        directory = tmp_path / subdir
        directory.mkdir(exist_ok=True)
        result = subprocess.run([sys.executable, __file__, mode, str(directory)],
                                env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                timeout=600)
        assert result.returncode == 0, result.stdout


@requires_remote_code
def test_lora_export_key_layout(remote_dir):
    from test_glm5_next import _parallel_context

    with _parallel_context():
        _, model, _, config, _ = _build_parity_models(remote_dir, mtp=1)
        peft_model = _wrap_lora(model)
        for name, parameter in peft_model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                torch.nn.init.normal_(parameter, std=0.1)
        adapter = {}
        config.bridge._peft_format = True
        try:
            for key, value in config.bridge._convert([peft_model], {}, '', False, 'Exporting LoRA: '):
                adapter[key] = value.load() if hasattr(value, 'load') else value
        finally:
            config.bridge._peft_format = False
        # Fused-QKV LoRA must export all three targets with a shared A and asymmetric B rows.
        for prefix in ('model.layers.0.self_attn.', 'model.layers.1.self_attn.', 'model.mtp.layers.0.self_attn.'):
            for proj in ('q_proj', 'k_proj', 'v_proj'):
                assert f'{prefix}{proj}.lora_A.weight' in adapter, f'missing {prefix}{proj}.lora_A'
                assert f'{prefix}{proj}.lora_B.weight' in adapter, f'missing {prefix}{proj}.lora_B'
            torch.testing.assert_close(
                adapter[f'{prefix}q_proj.lora_A.weight'], adapter[f'{prefix}k_proj.lora_A.weight'], atol=0, rtol=0)
        assert adapter['model.layers.0.self_attn.v_proj.lora_B.weight'].shape[0] == 2 * 16  # full kv=2, v=16
        assert adapter['model.layers.1.self_attn.v_proj.lora_B.weight'].shape[0] == 4 * 16  # swa kv=4
        assert adapter['model.layers.0.self_attn.q_proj.lora_B.weight'].shape[0] == 8 * 32
        assert 'model.mtp.layers.0.eh_proj.lora_B.weight' in adapter


@requires_remote_code
def test_sharded_state_dict_sink_and_qkv_layout(remote_dir):
    from megatron.core import parallel_state
    from test_glm5_next import _parallel_context

    tp = int(os.environ.get('WORLD_SIZE', '1'))
    if tp not in (1, 2):
        pytest.skip('run at world size 1 or 2')
    with _parallel_context(tp=tp):
        _, model, checkpoint, config, _ = _build_parity_models(remote_dir, tp=tp, mtp=1)
        lm = _lm(model)
        rank = parallel_state.get_tensor_model_parallel_rank()
        # TE's sharded_state_dict reads metadata['dp_cp_group'], normally set by the dist-ckpt walk.
        metadata = {'dp_cp_group': parallel_state.get_data_parallel_group(with_context_parallel=True)}
        attn_modules = [(f'decoder.layers.{i}', layer.self_attention) for i, layer in enumerate(lm.decoder.layers)]
        attn_modules += [('mtp.layers.0', layer.mtp_model_layer.self_attention) for layer in lm.mtp.layers]
        for name, attn in attn_modules:
            is_swa = 'mtp' in name or _PATTERN[int(name.split('.')[-1])]
            core_sd = attn.core_attention.sharded_state_dict(prefix='', metadata=metadata)
            if is_swa:
                assert 'softmax_offset' in core_sd, f'{name}: sink missing from sharded_state_dict'
                shard = core_sd['softmax_offset']
                assert tuple(shard.global_shape) == (8, )
                assert tuple(shard.local_shape) == (8 // tp, )
                assert tuple(shard.global_offset) == (rank * (8 // tp), )
                gathered = [torch.empty_like(shard.data) for _ in range(tp)]
                torch.distributed.all_gather(
                    gathered, shard.data.contiguous(), group=parallel_state.get_tensor_model_parallel_group())
                full = torch.cat(gathered)
                hf_key = ('model.mtp.layers.0.self_attn.attention_sink_bias'
                          if 'mtp' in name else f'model.layers.{name.split(".")[-1]}.self_attn.attention_sink_bias')
                torch.testing.assert_close(full.cpu(), checkpoint[hf_key].cpu(), atol=0, rtol=0)
            else:
                assert 'softmax_offset' not in core_sd, f'{name}: full layer must not declare a sink'
            # Asymmetric fused QKV: global row count is groups*(hpg*qk + qk + v), not 3*heads*qk.
            qkv_sd = attn.sharded_state_dict(prefix='', metadata=metadata)
            shard = qkv_sd['linear_qkv.weight']
            groups = 4 if is_swa else 2
            expected_rows = groups * ((8 // groups) * 32 + 32 + 16)
            assert tuple(shard.global_shape) == (expected_rows, 64), f'{name}: {shard.global_shape}'
            assert tuple(shard.local_shape) == (expected_rows // tp, 64)
            gathered = [torch.empty_like(shard.data) for _ in range(tp)]
            torch.distributed.all_gather(
                gathered, shard.data.contiguous(), group=parallel_state.get_tensor_model_parallel_group())
            full = torch.cat(gathered)
            idx = name.split('.')[-1]
            prefix = 'model.mtp.layers.0.self_attn.' if 'mtp' in name else f'model.layers.{idx}.self_attn.'
            q, k, v = (checkpoint[prefix + f'{p}_proj.weight'] for p in ('q', 'k', 'v'))
            hpg = 8 // groups
            expected = torch.cat(
                [torch.cat([q.view(groups, hpg * 32, 64),
                            k.view(groups, 32, 64),
                            v.view(groups, 16, 64)], dim=1)],
                dim=0).reshape(-1, 64)
            torch.testing.assert_close(full.cpu(), expected.cpu(), atol=0, rtol=0)


def _packed_inputs(lengths, vocab=128):
    import itertools
    from megatron.core.packed_seq_params import PackedSeqParams

    total = sum(lengths)
    generator = torch.Generator(device='cuda').manual_seed(23)  # ranks must agree (see _model_inputs)
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


def _hf_logits_packed(hf_model, input_ids, lengths):
    outs = []
    offset = 0
    for length in lengths:
        segment = input_ids[:, offset:offset + length]
        outs.append(_hf_logits(hf_model, segment))
        offset += length
    return torch.cat(outs, dim=1)


_TOPOLOGIES = {
    'tp2': dict(tp=2),
    'tp2_sp': dict(tp=2, sequence_parallel=True),
    'ep2': dict(ep=2),
    'tp2_ep2': dict(tp=2, ep=2),
    'pp2': dict(pp=2),
    'pp2_tp2': dict(tp=2, pp=2),
}


@requires_remote_code
@pytest.mark.parametrize('topo', list(_TOPOLOGIES))
@pytest.mark.parametrize('padding_free', ['off', 'on', 'pack'])
def test_parallel_forward_parity(remote_dir, topo, padding_free):
    if os.environ.get('MIMO_PARALLEL_TEST') != topo:
        pytest.skip(f'run with MIMO_PARALLEL_TEST={topo} torchrun --nproc-per-node=<world>')
    from megatron.core import parallel_state
    from test_glm5_next import _parallel_context

    kwargs = _TOPOLOGIES[topo]
    # sequence_parallel is a model-config flag, not an initialize_model_parallel argument.
    ctx_kwargs = {k: v for k, v in kwargs.items() if k != 'sequence_parallel'}
    with _parallel_context(**ctx_kwargs):
        hf_model, model, _, config, _ = _build_parity_models(remote_dir, mtp=1, **kwargs)
        if padding_free == 'off':
            input_ids, position_ids, attention_mask = _model_inputs()
            expected = _hf_logits(hf_model, input_ids).float()
            third = attention_mask
        else:
            # SP scatters the packed row over TP: total length must be a multiple of tp.
            if padding_free == 'pack':
                lengths = [3, 5, 8] if kwargs.get('tp', 1) == 1 else [4, 4, 8]
            else:
                lengths = [5, 7, 4] if kwargs.get('tp', 1) == 1 else [6, 10]
            input_ids, position_ids, packed_seq_params = _packed_inputs(lengths)
            expected = _hf_logits_packed(hf_model, input_ids, lengths).float()
            third = packed_seq_params
        # fp32 reduction-order noise across topologies reaches ~4.7e-4 on <1% of logits.
        atol = rtol = 5e-4
        if kwargs.get('pp', 1) > 1:
            # MTP+PP transports cat([hidden, embedding]) between stages, so parity must run
            # through the pipeline schedule (a direct call leaves get_input_tensor() unset).
            from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

            def forward_step(data_iterator, stage_model):
                ids, positions, extra = next(data_iterator)
                if padding_free == 'off':
                    output = stage_model(ids, positions, extra, runtime_gather_output=True)
                else:
                    output = stage_model(ids, positions, None, packed_seq_params=extra, runtime_gather_output=True)
                return output, lambda tensor, **kw: tensor

            with torch.no_grad():
                outputs = get_forward_backward_func()(
                    forward_step_func=forward_step,
                    data_iterator=iter([(input_ids, position_ids, third)]),
                    model=model,
                    num_microbatches=1,
                    seq_length=input_ids.shape[1],
                    micro_batch_size=input_ids.shape[0],
                    forward_only=True,
                    collect_non_loss_data=True,
                )
            if parallel_state.is_pipeline_last_stage():
                assert len(outputs) == 1
                actual = outputs[0][..., :config.actual_vocab_size].float()
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            else:
                assert outputs == []
        else:
            if padding_free == 'off':
                actual = _mcore_logits(model, config, input_ids, position_ids, attention_mask).float()
            else:
                logits = model(
                    input_ids, position_ids, None, packed_seq_params=packed_seq_params, runtime_gather_output=True)
                actual = logits[..., :config.actual_vocab_size].float()
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@requires_remote_code
def test_parallel_tp2_roundtrip_bit_exact(remote_dir):
    if os.environ.get('MIMO_PARALLEL_TEST') != 'tp2':
        pytest.skip('run with MIMO_PARALLEL_TEST=tp2 torchrun --nproc-per-node=2')
    from test_glm5_next import _parallel_context

    with _parallel_context(tp=2):
        _, model, checkpoint, config, _ = _build_parity_models(remote_dir, tp=2, mtp=1)
        exported = _export(config, model)
        assert set(exported) == set(checkpoint)
        for key in checkpoint:
            torch.testing.assert_close(exported[key], checkpoint[key], atol=0, rtol=0, msg=lambda m: f'{key}: {m}')
        reloaded = get_mcore_model(config)[0].cuda().eval()
        lazy = {key: LazyTensor(tensor=value) for key, value in exported.items()}
        list(config.bridge._convert([reloaded], lazy, '', True, 'Reloading tp2: '))
        input_ids, position_ids, attention_mask = _model_inputs()
        expected = _mcore_logits(model, config, input_ids, position_ids, attention_mask)
        actual = _mcore_logits(reloaded, config, input_ids, position_ids, attention_mask)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@requires_remote_code
def test_parallel_tp2_sp_gradient_parity(remote_dir):
    """Replicated params (LN/router) must accumulate the full token set under SP (param.sequence_parallel)."""
    if os.environ.get('MIMO_PARALLEL_TEST') != 'tp2_sp':
        pytest.skip('run with MIMO_PARALLEL_TEST=tp2_sp torchrun --nproc-per-node=2')
    from megatron.core import parallel_state
    from test_glm5_next import _parallel_context

    with _parallel_context(tp=2):
        hf_model, model, _, config, _ = _build_parity_models(remote_dir, tp=2, sequence_parallel=True, mtp=0)
        lm = _lm(model)
        input_ids, position_ids, attention_mask = _model_inputs()
        labels = torch.randint(
            1, 128, input_ids.shape, device='cuda', generator=torch.Generator(device='cuda').manual_seed(5))
        hf_logits = _hf_logits(hf_model, input_ids)
        F.cross_entropy(hf_logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten()).backward()
        mg_logits = _mcore_logits(model, config, input_ids, position_ids, attention_mask)
        F.cross_entropy(mg_logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten()).backward()
        # Raw backward leaves partial grads on replicated params under SP; mirror
        # finalize_model_grads' SUM all-reduce over sequence_parallel-marked params.
        tp_group = parallel_state.get_tensor_model_parallel_group()
        for param in model.parameters():
            if param.grad is not None and getattr(param, 'sequence_parallel', False):
                torch.distributed.all_reduce(param.grad, group=tp_group)
        pairs = [
            # Vocab-parallel embedding grads cover the FULL token set per vocab shard (TP
            # all-reduce), so compare against the matching HF shard, not the full table.
            ('embed', hf_model.model.embed_tokens.weight.grad.chunk(
                config.tensor_model_parallel_size)[parallel_state.get_tensor_model_parallel_rank()],
             lm.embedding.word_embeddings.weight.grad),
            ('final_norm', hf_model.model.norm.weight.grad, lm.decoder.final_layernorm.weight.grad),
            ('l1.router', hf_model.model.layers[1].mlp.gate.weight.grad, lm.decoder.layers[1].mlp.router.weight.grad),
            ('l1.pre_ln', hf_model.model.layers[1].post_attention_layernorm.weight.grad,
             lm.decoder.layers[1].pre_mlp_layernorm.weight.grad),
            ('l0.input_ln', hf_model.model.layers[0].input_layernorm.weight.grad,
             lm.decoder.layers[0].self_attention.linear_qkv.layer_norm_weight.grad),
        ]
        for name, expected, actual in pairs:
            assert actual is not None, f'{name}: grad missing'
            torch.testing.assert_close(
                actual.float(), expected.float(), atol=3e-4, rtol=3e-4, msg=lambda m: f'{name}: {m}')


# Real-weight regression: TE fused LN+GEMM fp32 kernels carry ~2e-3 rel error on SM100, so
# parity runs with them bypassed (pure torch fp32). The last layer carries massive activations
# (|h| up to ~45 vs <1 elsewhere), where fp32 GEMM rounding surfaces as ~3e-3 abs; the
# assertions below are massive-activation-aware instead of a flat atol.


@pytest.fixture(scope='session')
def real_weights_dir(tmp_path_factory):
    directory = str(tmp_path_factory.mktemp('mimo_v2_real'))
    _copy_remote_code(_REAL_WEIGHTS_DIR, directory)
    for name in ('config.json', 'model.safetensors.index.json'):
        shutil.copy(os.path.join(_REAL_WEIGHTS_DIR, name), directory)
    with open(os.path.join(directory, 'model.safetensors.index.json')) as f:
        shards = sorted(set(json.load(f)['weight_map'].values()))
    for shard in shards:
        os.symlink(os.path.join(_REAL_WEIGHTS_DIR, shard), os.path.join(directory, shard))
    return directory


def _build_real_weight_models(real_weights_dir, dtype, language_model_only=True):
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    hf_config = AutoConfig.from_pretrained(real_weights_dir, trust_remote_code=True)
    hf_config._attn_implementation = 'eager'  # sink bias needs the eager path
    # Cast per-parameter to checkpoint storage dtype: a blanket bf16 load would round the
    # fp32-stored e_score_correction_bias, which megatron keeps fp32 (TopKRouter.expert_bias).
    hf = AutoModelForCausalLM.from_pretrained(
        real_weights_dir, config=hf_config, trust_remote_code=True, torch_dtype=torch.float32).cuda().eval()
    with open(os.path.join(_REAL_WEIGHTS_DIR, 'model.safetensors.index.json')) as f:
        weight_map = json.load(f)['weight_map']
    ckpt_dtypes = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(os.path.join(_REAL_WEIGHTS_DIR, shard), framework='pt') as f:
            for key in f.keys():
                ckpt_dtypes[key] = f.get_slice(key).get_dtype()
    with torch.no_grad():
        for name, param in hf.named_parameters():
            storage = ckpt_dtypes.get(name)
            if storage == 'BF16' and dtype == torch.bfloat16:
                param.data = param.data.to(torch.bfloat16)
    # transformers meta-init leaves non-persistent buffers absent from the checkpoint as
    # UNINITIALIZED memory: the vision rope's inv_freq is garbage after load (the text rope's
    # buffers measure correct), and the vision forward reads its buffer directly -> rebuild it
    # deterministically (same formula as the remote code).
    if getattr(hf, 'visual', None) is not None:
        rope = hf.visual.rotary_pos_emb
        dim = rope.inv_freq.numel() * 2
        device = rope.inv_freq.device
        rope.inv_freq = (1.0 / (10000.0**(torch.arange(0, dim, 2, dtype=torch.float32) / dim))).to(device)
    values = hf_to_mcore_config(hf_config)
    values.update(
        mcore_model_type='mimo_v2',
        language_model_only=language_model_only,
        moe_grouped_gemm=True,
        expert_tensor_parallel_size=1,
        overlap_p2p_comm=False,
        mtp_num_layers=0,
        mtp_loss_scaling_factor=0.1,
        params_dtype=dtype)
    config = ModelConfig(**values)
    model = get_mcore_model(config)[0].cuda().eval()
    config.bridge.load_weights([model], _REAL_WEIGHTS_DIR)
    return hf, model, config


def _real_inputs(vocab, seq=32):
    generator = torch.Generator(device='cuda').manual_seed(7)
    input_ids = torch.randint(1, vocab, (1, seq), device='cuda', generator=generator)
    position_ids = torch.arange(seq, device='cuda').unsqueeze(0)
    mask4d = torch.triu(torch.ones(1, 1, seq, seq, device='cuda', dtype=torch.bool), diagonal=1)
    return input_ids, position_ids, mask4d


def _real_rmsnorm(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def _patch_unfused(model, eps):
    """Replace TE fused LN+Linear / grouped / plain TE linears with pure torch matmul.

    Returns (n_fused, n_plain) so callers can assert the patch was not a silent no-op.
    """
    n_fused = 0
    for name, mod in model.named_modules():
        is_grouped = 'Grouped' in type(mod).__name__ and hasattr(mod, 'weight0')
        if not (is_grouped or (hasattr(mod, 'layer_norm_weight') and hasattr(mod, 'weight') and 'linear' in name)):
            continue
        if is_grouped:

            def make_g(m):

                def fwd(inp, m_splits, *a, **kw):
                    x = inp
                    if getattr(m, 'layer_norm_weight', None) is not None:
                        x = _real_rmsnorm(x, m.layer_norm_weight, eps)
                    outs = []
                    off = 0
                    for e, n in enumerate(m_splits):
                        n = int(n)
                        outs.append(F.linear(x[off:off + n], getattr(m, f'weight{e}')))
                        off += n
                    out = torch.cat(outs, dim=0) if outs else x.new_zeros((0, m.weight0.shape[0]))
                    return out, None

                return fwd

            mod.forward = make_g(mod)
        else:

            def make(m):

                def fwd(inp, *a, **kw):
                    normed = _real_rmsnorm(inp, m.layer_norm_weight, eps)
                    out = F.linear(normed, m.weight)
                    bias = getattr(m, 'bias', None)
                    if bias is not None and bias.numel() > 0:
                        out = out + bias
                    return out, None

                return fwd

            mod.forward = make(mod)
        n_fused += 1
    n_plain = 0
    for name, mod in model.named_modules():
        if type(mod).__name__.startswith('TE') and hasattr(mod, 'weight') and 'Linear' in type(mod).__name__ \
                and not hasattr(mod, 'layer_norm_weight') and 'Grouped' not in type(mod).__name__:

            def make_p(m):

                def fwd(inp, *a, **kw):
                    out = F.linear(inp, m.weight)
                    bias = getattr(m, 'bias', None)
                    if bias is not None and bias.numel() > 0:
                        return out, bias
                    return out, None

                return fwd

            mod.forward = make_p(mod)
            n_plain += 1
    return n_fused, n_plain


def _patch_eager_attn(model):
    """Replace MiMoV2 TEDotProductAttention with a pure torch eager attention (fp32 softmax,
    sink column, v_scale), mirroring the HF eager reference. Returns the patch count."""
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, mimo_v2.MiMoV2TEDotProductAttention):

            def make(m):

                def fwd(query, key, value, attention_mask, *a, **kw):
                    # query [s,b,nh,qk], key [s,b,nkv,qk], value [s,b,nkv,v]
                    if getattr(m, 'attention_value_scale', None) is not None:
                        value = value * m.attention_value_scale
                    s, b, nh, qk = query.shape
                    rep = nh // key.shape[2]
                    q = query.permute(1, 2, 0, 3).float()
                    k = key.permute(1, 2, 0, 3).float().repeat_interleave(rep, dim=1)
                    v = value.permute(1, 2, 0, 3).float().repeat_interleave(rep, dim=1)
                    scale = m.softmax_scale if getattr(m, 'softmax_scale', None) else qk**-0.5
                    logits = torch.matmul(q, k.transpose(-1, -2)) * scale
                    if attention_mask is not None:
                        logits = logits.masked_fill(attention_mask[:, :, :s, :s], float('-inf'))
                    sink = getattr(m, 'softmax_offset', None)
                    if sink is not None:
                        logits = torch.cat([logits, sink.float().view(1, nh, 1, 1).expand(b, nh, s, 1)], dim=-1)
                    probs = torch.softmax(logits, dim=-1)
                    if sink is not None:
                        probs = probs[..., :-1]
                    out = torch.matmul(probs, v)  # [b,nh,s,v]
                    out = out.permute(2, 0, 1, 3).reshape(s, b, nh * v.shape[-1])
                    return out.contiguous().to(query.dtype)

                return fwd

            mod.forward = make(mod)
            n += 1
    return n


@requires_real_weights
def test_real_weight_loading_bit_exact(real_weights_dir):
    """Every bridge-loaded parameter must be bit-exact vs the HF checkpoint (bf16 storage)."""
    from test_glm5_next import _parallel_context

    with _parallel_context():
        hf, model, config = _build_real_weight_models(real_weights_dir, torch.bfloat16)
        try:
            lm, hfm = _lm(model), hf.model
            pattern = hf.config.hybrid_layer_pattern
            pairs = [
                ('embed', lm.embedding.word_embeddings.weight, hfm.embed_tokens.weight),
                ('final_norm', lm.decoder.final_layernorm.weight, hfm.norm.weight),
                ('lm_head', lm.output_layer.weight, hf.lm_head.weight),
            ]
            for idx, (hl, ml) in enumerate(zip(hfm.layers, lm.decoder.layers)):
                sa, msa = hl.self_attn, ml.self_attention
                nh, nkv = sa.num_attention_heads, sa.num_key_value_heads
                qk_dim, v_dim = sa.head_dim, sa.v_head_dim
                hpg = nh // nkv
                group_dim = hpg * qk_dim + qk_dim + v_dim
                qkv = msa.linear_qkv.weight.view(nkv, group_dim, -1)
                # dense layers fuse ln2 into linear_fc1; MoE layers keep a real pre_mlp_layernorm
                ln2_w = ml.pre_mlp_layernorm.weight if hasattr(ml.pre_mlp_layernorm, 'weight') \
                    else ml.mlp.linear_fc1.layer_norm_weight
                pairs += [
                    (f'l{idx}.q', qkv[:, :hpg * qk_dim].reshape(nh, qk_dim, -1), sa.q_proj.weight.view(nh, qk_dim, -1)),
                    (f'l{idx}.k', qkv[:, hpg * qk_dim:hpg * qk_dim + qk_dim], sa.k_proj.weight.view(nkv, qk_dim, -1)),
                    (f'l{idx}.v', qkv[:, hpg * qk_dim + qk_dim:], sa.v_proj.weight.view(nkv, v_dim, -1)),
                    (f'l{idx}.o', msa.linear_proj.weight, sa.o_proj.weight),
                    (f'l{idx}.ln1', msa.linear_qkv.layer_norm_weight, hl.input_layernorm.weight),
                    (f'l{idx}.ln2', ln2_w, hl.post_attention_layernorm.weight),
                ]
                if pattern[idx]:
                    pairs.append((f'l{idx}.sink', msa.core_attention.softmax_offset, sa.attention_sink_bias))
                if hasattr(ml.mlp, 'experts'):
                    gate = hl.mlp.gate
                    pairs.append((f'l{idx}.router', ml.mlp.router.weight, gate.weight))
                    pairs.append((f'l{idx}.router_bias', ml.mlp.router.expert_bias, gate.e_score_correction_bias))
                    fc1, fc2 = ml.mlp.experts.linear_fc1, ml.mlp.experts.linear_fc2
                    for e, he in enumerate(hl.mlp.experts):
                        # swiglu fc1 layout: [gate; up] stacked (megatron convention)
                        pairs.append((f'l{idx}.e{e}.fc1', getattr(fc1, f'weight{e}'),
                                      torch.cat([he.gate_proj.weight, he.up_proj.weight], dim=0)))
                        pairs.append((f'l{idx}.e{e}.fc2', getattr(fc2, f'weight{e}'), he.down_proj.weight))
                else:
                    pairs.append((f'l{idx}.fc1', ml.mlp.linear_fc1.weight,
                                  torch.cat([hl.mlp.gate_proj.weight, hl.mlp.up_proj.weight], dim=0)))
                    pairs.append((f'l{idx}.fc2', ml.mlp.linear_fc2.weight, hl.mlp.down_proj.weight))
            for name, actual, expected in pairs:
                torch.testing.assert_close(
                    actual.detach().float(), expected.detach().float(), atol=0, rtol=0, msg=lambda m: f'{name}: {m}')
        finally:
            del hf, model
            torch.cuda.empty_cache()


@requires_real_weights
def test_real_weight_forward_parity_fp32_unfused(real_weights_dir):
    """fp32 parity with TE fused kernels bypassed: layers 0..N-2 tight; the last layer carries
    massive activations, so its tolerance is relative there and absolute elsewhere."""
    from test_glm5_next import _parallel_context

    with _parallel_context():
        hf, model, config = _build_real_weight_models(real_weights_dir, torch.float32)
        try:
            lm = _lm(model)
            n_fused, n_plain = _patch_unfused(model, config.layernorm_epsilon)
            n_eager = _patch_eager_attn(model)
            n_layers = hf.config.num_hidden_layers
            assert n_fused > 0 and n_plain > 0 and n_eager == n_layers, 'patches must not be silent no-ops'
            input_ids, position_ids, mask4d = _real_inputs(hf.config.vocab_size)
            hf_outs, mg_outs = {}, {}

            def hook(store, i):

                def fn(m, a, o):
                    store[i] = (o[0] if isinstance(o, tuple) else o).detach().float().cpu()

                return fn

            for i in range(n_layers):
                hf.model.layers[i].register_forward_hook(hook(hf_outs, i))
                lm.decoder.layers[i].register_forward_hook(hook(mg_outs, i))
            with torch.no_grad():
                hf(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
                model(input_ids, position_ids, mask4d, runtime_gather_output=True)
            for i in range(n_layers - 1):
                torch.testing.assert_close(
                    mg_outs[i].transpose(0, 1), hf_outs[i], atol=5e-5, rtol=5e-5, msg=lambda m: f'layer{i}: {m}')
            hf_last, mg_last = hf_outs[n_layers - 1], mg_outs[n_layers - 1].transpose(0, 1)
            d = (hf_last - mg_last).abs()
            big = hf_last.abs() > 5.0
            assert bool(big.any()), 'expected massive activations in the last layer of this checkpoint'
            rel = d[big] / hf_last.abs()[big]
            assert rel.max().item() <= 1e-3, f'massive-feature rel noise {rel.max().item():.3e} exceeds fp32 GEMM level'
            assert d[~big].max().item() <= 1e-2, f'non-massive abs diff {d[~big].max().item():.3e}'
            assert d.max().item() <= 1e-2, f'last-layer abs diff {d.max().item():.3e}'
        finally:
            del hf, model
            torch.cuda.empty_cache()


@requires_real_weights
def test_real_weight_bf16_routing_agreement(real_weights_dir):
    """bf16 on the real TE path: mcore's top-k selection must equal an independent fp32 noaux_tc
    recomputation, and any HF-vs-mcore flip must sit inside the bf16 noise band -- a blanket
    bf16 oracle rounds the fp32-stored e_score_correction_bias that megatron keeps fp32."""
    from test_glm5_next import _parallel_context

    with _parallel_context():
        hf, model, config = _build_real_weight_models(real_weights_dir, torch.bfloat16)
        try:
            lm = _lm(model)
            last = hf.config.num_hidden_layers - 1
            hl, ml = hf.model.layers[last], lm.decoder.layers[last]
            rec = {}

            def gate_hook(m, a, o):
                rec['hf_idx'] = o[0].detach().cpu()  # gate returns (topk_idx, topk_weight)

            def ln2_hook(m, a, o):
                rec['hf_ln2'] = (o[0] if isinstance(o, tuple) else o).detach().float().cpu()

            def router_hook(m, a, o):
                rec['mg_in'] = a[0].detach().float().cpu()
                rec['rmap'] = o[1].detach().float().cpu()  # TopKRouter returns (probs, routing_map)

            hl.mlp.gate.register_forward_hook(gate_hook)
            hl.post_attention_layernorm.register_forward_hook(ln2_hook)
            ml.mlp.router.register_forward_hook(router_hook)
            input_ids, position_ids, mask4d = _real_inputs(hf.config.vocab_size)
            with torch.no_grad():
                hf(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
                model(input_ids, position_ids, mask4d, runtime_gather_output=True)
            seq = input_ids.shape[1]
            top_k, n_experts = hf.config.num_experts_per_tok, hf.config.n_routed_experts
            hf_idx = rec['hf_idx'].reshape(seq, top_k)
            rmap = rec['rmap'].reshape(seq, n_experts)
            assert bool(((rmap != 0).sum(-1) == top_k).all()), 'routing map must select exactly top-k experts'
            mg_idx = (rmap != 0).nonzero()[:, 1].reshape(seq, top_k)
            # Independent fp32 noaux_tc recomputation; n_group == 1 makes group selection a no-op.
            assert hf.config.n_group == 1
            gate = hl.mlp.gate
            bias = gate.e_score_correction_bias.detach().float().cpu()  # fp32 storage, unrounded
            mg_logits = F.linear(rec['mg_in'].reshape(seq, -1), ml.mlp.router.weight.detach().float().cpu())
            ref_idx = torch.topk(mg_logits.sigmoid() + bias, k=top_k, dim=-1, sorted=False)[1]
            assert bool((ref_idx.sort(-1).values == mg_idx.sort(-1).values).all()), \
                'mcore routing deviates from noaux_tc on its own input'
            # HF-vs-mcore flips must be marginal-token noise, not a systematic offset.
            hf_logits = F.linear(rec['hf_ln2'].reshape(seq, -1), gate.weight.detach().float().cpu())
            logit_gap = (hf_logits - mg_logits).abs().max().item()
            assert logit_gap < 0.15, f'router logit gap {logit_gap:.3e} exceeds the bf16 hidden-noise band'
            scores = hf_logits.sigmoid() + bias
            margins = scores.sort(dim=-1, descending=True).values
            margin = margins[:, top_k - 1] - margins[:, top_k]
            same = (hf_idx.sort(-1).values == mg_idx.sort(-1).values).all(dim=-1)
            flipped = (~same).nonzero().flatten().tolist()
            for t in flipped:
                assert margin[t].item() < 0.15, \
                    f'token {t} flips with margin {margin[t].item():.3e} outside the bf16 noise band'
        finally:
            del hf, model
            torch.cuda.empty_cache()


def test_vision_namespace_maps_in_chans():
    # The shared embedding helper reads in_channels; HF's vision dict only carries in_chans.
    ns = mimo_v2.MiMoV2Vit._vision_namespace(NS(vision_config={'in_chans': 3, 'patch_size': 16}))
    assert ns.in_channels == 3 and ns.patch_size == 16
    explicit = mimo_v2.MiMoV2Vit._vision_namespace(NS(vision_config={'in_chans': 3, 'in_channels': 1}))
    assert explicit.in_channels == 1


@requires_real_weights
def test_real_weight_vision_loading_and_forward(real_weights_dir):
    """visual.* must load bit-exact into the dynamically loaded HF vision tower, and one image
    forward must agree with the HF side. Both sides run the same remote-code class, so any
    divergence means broken weight loading or a wrong vision namespace (in_chans -> in_channels).
    Also exercises the get_inputs_embeds placeholder-scatter glue with real pixel_values."""
    from test_glm5_next import _parallel_context

    with _parallel_context():
        hf, model, config = _build_real_weight_models(real_weights_dir, torch.float32, language_model_only=False)
        try:
            mg_visual = model.visual.visual
            # Upstream inconsistency: the official ckpt ships no merger biases but the remote code
            # declares them, so HF's own load leaves them RANDOMLY initialized (not a meaningful
            # reference). MiMoV2Vit drops them (bias=None == zero bias); mirror that on the oracle.
            hf.visual.merger.ln_q.bias = None
            for module in hf.visual.merger.mlp:
                if isinstance(module, torch.nn.Linear):
                    module.bias = None
            hf_params = dict(hf.visual.named_parameters())
            mg_params = dict(mg_visual.named_parameters())
            assert len(hf_params) > 0 and hf_params.keys() == mg_params.keys()
            for name, param in hf_params.items():
                assert torch.equal(mg_params[name].detach(), param.detach()), f'visual.{name} differs'
            hf_buffers = dict(hf.visual.named_buffers())
            mg_buffers = dict(mg_visual.named_buffers())
            assert hf_buffers.keys() == mg_buffers.keys()
            for name, buf in hf_buffers.items():
                assert torch.equal(mg_buffers[name], buf), f'visual buffer {name} differs'

            vc = hf.config.vision_config
            patch_dim = vc['in_chans'] * vc['temporal_patch_size'] * vc['patch_size']**2
            generator = torch.Generator(device='cuda').manual_seed(29)
            pixel_values = torch.randn(16 * 16, patch_dim, generator=generator, device='cuda')
            grid_thw = torch.tensor([[1, 16, 16]], device='cuda')
            with torch.no_grad():
                ref_embeds = hf.visual(pixel_values, grid_thw=grid_thw)
                mg_embeds = mg_visual(pixel_values, grid_thw=grid_thw)
            n_img = 16 * 16 // vc['spatial_merge_size']**2
            assert ref_embeds.shape == (n_img, vc['out_hidden_size'])
            # Same class + bit-exact weights, but the two module instances hit different cuBLAS
            # split-k tilings: fp32 reduction-order noise only (measured max abs ~3e-6).
            torch.testing.assert_close(mg_embeds, ref_embeds, atol=1e-5, rtol=1e-5)

            # Placeholder-scatter glue: the image tokens receive exactly the tower output.
            input_ids = torch.full((1, n_img + 2), hf.config.image_token_id, device='cuda', dtype=torch.long)
            input_ids[0, 0], input_ids[0, -1] = 100, 101
            embeds = hf.model.embed_tokens(input_ids)
            expected = embeds.clone()
            expected[0, 1:1 + n_img] = ref_embeds.to(embeds.dtype)
            with torch.no_grad():
                actual = model.visual.get_inputs_embeds(
                    embeds.clone(), input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=grid_thw)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        finally:
            del hf, model
            torch.cuda.empty_cache()


if __name__ == '__main__':
    mode, output_dir = sys.argv[1:]
    _REMOTE_FIXTURE_DIR[0] = os.environ.get('MIMO_V2_REMOTE_DIR', _REMOTE_DIR)
    # The fixture dir must carry the session config.json; rebuild it from the canonical dict.
    os.makedirs(output_dir, exist_ok=True)
    if mode in ('save', 'save_trained', 'save_decoder_only'):
        _copy_remote_code(_REMOTE_FIXTURE_DIR[0], output_dir)
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(_tiny_config_dict(), f)
        _REMOTE_FIXTURE_DIR[0] = output_dir
    else:
        # check modes reuse the dir written by the matching save mode (config.json + remote code).
        assert os.path.exists(os.path.join(output_dir, 'config.json'))
        _REMOTE_FIXTURE_DIR[0] = output_dir
    if mode == 'save':
        _save_lora_fixture(output_dir)
    elif mode == 'save_trained':
        _save_lora_fixture(output_dir, steps=3)
    elif mode == 'save_decoder_only':
        _save_lora_fixture(output_dir, decoder_only=True)
    elif mode == 'check':
        _check_lora_fixture(output_dir)
    elif mode == 'check_drop_v':
        _check_lora_fixture(output_dir, drop_v_lora_b=True)
    else:
        raise ValueError(mode)
