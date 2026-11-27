# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import os
import pytest
import subprocess
import sys
import torch
from copy import deepcopy
from megatron.core.packed_seq_params import PackedSeqParams
from pathlib import Path
from peft import LoraConfig
from peft.tuners.lora.layer import Linear as PeftLinear
from torch import nn
from types import SimpleNamespace

from mcore_bridge.model.mm_gpts.inkling2_loop import Inkling2LoopBridge
from mcore_bridge.model.modules.inkling2_loop import (ATTENTION_PROJECTIONS, Inkling2Router, Inkling2TokenDispatcher,
                                                      _packed_document_layout)


@pytest.mark.parametrize('cu,physical,length,docs,valid', [
    ([0, 3, 5], None, 5, [0, 0, 0, 1, 1], [True] * 5),
    ([0, 3, 5], None, 7, [0, 0, 0, 1, 1, 2, 2], [True] * 5 + [False] * 2),
    ([0, 2, 3], [0, 4, 6], 7, [0, 0, 0, 0, 1, 1, 2], [True, True, False, False, True, False, False]),
    ([0, 0, 2, 2, 3], None, 3, [1, 1, 3], [True] * 3),
    ([0, 1, 2, 3], None, 3, [0, 1, 2], [True] * 3),
    ([0, 0], None, 0, [], []),
])
def test_document_layout_uses_cumulative_lengths(cu, physical, length, docs, valid):
    packed = PackedSeqParams(cu_seqlens_q=torch.tensor(cu, dtype=torch.int32), qkv_format='thd')
    if physical is not None:
        packed.cu_seqlens_q_padded = torch.tensor(physical, dtype=torch.int64)
    actual_docs, actual_valid = _packed_document_layout(packed, 1, length, 'cpu')
    assert actual_docs.dtype == torch.int32 and actual_docs.tolist() == [docs]
    assert actual_valid.dtype == torch.bool and actual_valid.tolist() == [valid]


@pytest.mark.parametrize('cu,physical,length,batch', [
    ([1, 3], None, 3, 1),
    ([0, 3, 2], None, 3, 1),
    ([0], None, 0, 1),
    ([0, 4], None, 3, 1),
    ([0., 3.], None, 3, 1),
    ([0, 2], [0, 1], 2, 1),
    ([0, 2, 3], [0, 3], 3, 1),
    ([0, 2], None, 2, 2),
])
def test_invalid_packed_layout_fails_before_computation(cu, physical, length, batch):
    packed = PackedSeqParams(cu_seqlens_q=torch.tensor(cu), qkv_format='thd')
    if physical is not None:
        packed.cu_seqlens_q_padded = torch.tensor(physical)
    with pytest.raises(ValueError):
        _packed_document_layout(packed, batch, length, 'cpu')


def test_unpacked_batch_rows_are_independent_documents():
    docs, valid = _packed_document_layout(None, 2, 5, 'cpu')
    assert docs.tolist() == [[0] * 5] * 2
    assert valid.all()


def test_router_selects_with_bias_but_weights_without_bias():
    router = object.__new__(Inkling2Router)
    nn.Module.__init__(router)
    router.config = SimpleNamespace(num_moe_experts=4, moe_router_topk=2, moe_router_topk_scaling_factor=8.)
    bias = [.2, -.1, .5, 0.]
    router.register_buffer('e_score_correction_bias', torch.tensor(bias))
    logits = torch.tensor([[1., 0., -.5, 2.], [2., 1., 0., -.5]])
    mask = torch.tensor([False, True])
    probabilities, routes = router.routing(logits, mask)
    expected = torch.zeros_like(logits)
    expected_routes = torch.zeros_like(routes)
    scores = [1 / (1 + math.exp(-x)) for x in logits[0].tolist()]
    selected = sorted(range(4), key=lambda index: scores[index] + bias[index], reverse=True)[:2]
    for index in selected:
        expected[0, index] = 8 * scores[index] / sum(scores[j] for j in selected)
        expected_routes[0, index] = True
    torch.testing.assert_close(probabilities, expected, rtol=1e-6, atol=1e-6)
    assert torch.equal(routes, expected_routes)


def test_combine_accumulates_in_expert_order_not_permutation_order():
    dispatcher = object.__new__(Inkling2TokenDispatcher)
    dispatcher.config = SimpleNamespace(moe_router_topk=3)
    dispatcher.routing_map = torch.tensor([[True, True, True, False], [False, True, True, True]])
    dispatcher.reversed_local_input_permutation_mapping = torch.tensor([0, 0, 1, 0, 1, 1])
    dispatcher.hidden_shape_before_permute = (2, 1)
    dispatcher.hidden_shape = (2, 1)
    tokens = torch.tensor([[1e8], [-1e8], [2.], [1.], [3.], [4.]], requires_grad=True)
    actual = dispatcher.combine_postprocess(tokens)
    expected = torch.stack([(tokens[0] + tokens[1]) + tokens[3], (tokens[2] + tokens[4]) + tokens[5]])
    assert torch.equal(actual, expected) and actual[:, 0].tolist() == [1., 9.]
    actual.sum().backward()
    assert torch.equal(tokens.grad, torch.ones_like(tokens))


@pytest.mark.parametrize('target', ['q_message_gate.down_proj', 'mlp.native.experts.linear_fc1', 'visual.merger'])
def test_actual_unmapped_adapter_is_rejected_before_export(target, tmp_path):
    model = nn.Module()
    parent = model
    for name in target.split('.')[:-1]:
        module = nn.Module()
        parent.add_module(name, module)
        parent = module
    parent.add_module(
        target.split('.')[-1],
        PeftLinear(nn.Linear(8, 8), 'default', config=LoraConfig(r=2, lora_alpha=4), r=2, lora_alpha=4))
    bridge = object.__new__(Inkling2LoopBridge)
    bridge.config = SimpleNamespace(language_model_only=False)
    with pytest.raises(ValueError, match='cannot preserve'):
        bridge.save_weights([model], str(tmp_path / 'export'), peft_format=True)
    assert not (tmp_path / 'export').exists()


def test_adapter_checkpoint_keys_are_checked():
    model = nn.Module()
    model.language_model = nn.Module()
    model.language_model.decoder = nn.Module()
    model.language_model.decoder.layers = nn.ModuleList([nn.Module() for _ in range(3)])
    attention = nn.Module()
    attention.q_proj = nn.Module()
    attention.q_proj.linear = nn.Module()
    attention.q_proj.linear.linear = PeftLinear(
        nn.Linear(8, 8), 'default', config=LoraConfig(r=2, lora_alpha=4), r=2, lora_alpha=4)
    model.language_model.decoder.layers[2].self_attn = attention
    assert Inkling2LoopBridge._check_adapter_modules([model], peft_format=True) == {('2', 'q_proj.linear')}
    supported = {
        f'base_model.model.model.language_model.layers.2.self_attn.{path}.lora_{factor}.weight': torch.ones(2, 2)
        for path in ATTENTION_PROJECTIONS
        for factor in ('A', 'B')
    }
    Inkling2LoopBridge._check_adapter_keys(supported, {('2', path) for path in ATTENTION_PROJECTIONS})
    supported['base_model.model.model.language_model.layers.2.q_message_gate.down_proj.lora_A.weight'] = torch.ones(
        2, 2)
    with pytest.raises(ValueError, match='unmapped adapter tensors'):
        Inkling2LoopBridge._check_adapter_keys(supported, {('2', path) for path in ATTENTION_PROJECTIONS})
    out_of_range = {
        key.replace('.layers.2.', '.layers.8.'): value
        for key, value in supported.items() if '.self_attn.' in key
    }
    with pytest.raises(ValueError, match='unmapped adapter tensors'):
        Inkling2LoopBridge._check_adapter_keys(out_of_range, {('2', path) for path in ATTENTION_PROJECTIONS})
    only_q = {key: value for key, value in supported.items() if '.self_attn.q_proj.' in key}
    with pytest.raises(ValueError, match='unmapped adapter tensors'):
        Inkling2LoopBridge._check_adapter_keys(only_q, {('2', 'k_proj.linear')})
    aliased = {key.replace('.layers.2.', '.layers.02.'): value for key, value in only_q.items()}
    with pytest.raises(ValueError, match='unmapped adapter tensors'):
        Inkling2LoopBridge._check_adapter_keys(aliased, {('2', 'q_proj.linear')})


def test_language_only_complete_export_rejects_disabled_vision_copy(tmp_path):
    bridge = object.__new__(Inkling2LoopBridge)
    bridge.config = SimpleNamespace(language_model_only=True)
    with pytest.raises(ValueError, match='save_missing_weights=True'):
        bridge.save_weights([nn.Linear(2, 2)], str(tmp_path / 'export'), save_missing_weights=False)
    assert not (tmp_path / 'export').exists()


def test_missing_causal_convolution_dependency_uses_upstream_error():
    pytest.importorskip('transformers.models.inkling2_loop')
    code = ('import sys,torch; sys.modules["causal_conv1d"]=None; '
            'from transformers.models.inkling2_loop.modeling_inkling2_loop import causal_conv1d_fn; '
            'x=torch.tensor([[[1.,2.,100.,200.]]]); w=torch.ones(1,2); '
            'assert torch.equal(causal_conv1d_fn(x,w),torch.tensor([[[1.,3.,102.,300.]]])); '
            'causal_conv1d_fn(x,w,seq_idx=torch.tensor([[0,0,1,1]],dtype=torch.int32))')
    env = dict(os.environ, USE_HUB_KERNELS='NO')
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode != 0
    assert "can't keep packed sequences apart without" in result.stderr


def _config():
    from transformers import Inkling2LoopConfig, Inkling2LoopTextConfig

    text = Inkling2LoopTextConfig(
        vocab_size=128,
        unpadded_vocab_size=120,
        hidden_size=32,
        num_hidden_layers=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        swa_num_attention_heads=2,
        swa_num_key_value_heads=1,
        swa_head_dim=8,
        d_rel=4,
        rel_extent=16,
        sliding_window_size=16,
        conv_kernel_size=4,
        max_position_embeddings=64,
        n_layers_per_attn_res_block=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        loop_layer_index_start=2,
        loop_layer_index_end=6,
        loop_num_passes=3,
        loop_gate_rank=8,
        layer_types=['hybrid_sliding', 'hybrid'] * 4,
        pad_token_id=0,
        eos_token_id=1)
    return Inkling2LoopConfig(
        text_config=text,
        vision_config={
            'model_type': 'inkling2_vision',
            'text_hidden_size': 32,
            'hidden_size': 32,
            'intermediate_size': 64,
            'num_hidden_layers': 1,
            'num_attention_heads': 2,
            'projection_hidden_dim': 64,
            'posemb_grid_size': 4
        },
        image_token_id=125)


def _shards(value):
    from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory

    if isinstance(value, ShardedTensor):
        yield value
    elif isinstance(value, ShardedTensorFactory):
        yield from _shards(value.build())
    elif isinstance(value, dict):
        for child in value.values():
            yield from _shards(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _shards(child)


def _worker(root, ep, tuner, reload_only=False):
    import torch.distributed as dist
    from megatron.core import mpu
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnBackend
    from peft import get_peft_model
    from safetensors.torch import load_file
    from transformers import Inkling2LoopForConditionalGeneration

    from mcore_bridge import get_mcore_model, hf_to_mcore_config
    from mcore_bridge.model.mm_gpts.inkling2_loop import Inkling2LoopConfig
    from mcore_bridge.model.modules.inkling2_loop import Inkling2LoopLoraLinear, Inkling2TensorLinear

    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=1)
    assert mpu.get_expert_model_parallel_world_size() == ep
    assert mpu.get_data_parallel_world_size() == ep
    assert mpu.get_expert_data_parallel_world_size() == 1
    assert mpu.get_tensor_model_parallel_world_size() == mpu.get_context_parallel_world_size() == 1
    model_parallel_cuda_manual_seed(1234)
    torch.manual_seed(42)
    config = _config()
    config._attn_implementation = config.text_config._attn_implementation = 'eager'
    hf = Inkling2LoopForConditionalGeneration(config).cuda()
    with torch.no_grad():
        for name, parameter in hf.named_parameters():
            value = .1 * torch.randn_like(parameter.float())
            if name.endswith('.scale') or 'layernorm' in name or name.endswith('norm.weight'):
                value += 1
            parameter.copy_(value)
        for name, buffer in hf.named_buffers():
            if name.endswith('e_score_correction_bias'):
                buffer.copy_(torch.linspace(-.13, .17, buffer.numel(), device='cuda').reshape_as(buffer))
    source = root / 'source'
    if dist.get_rank() == 0 and not reload_only:
        hf.save_pretrained(source, save_original_format=False)
    dist.barrier()
    config._name_or_path = str(source)
    kwargs = hf_to_mcore_config(config)
    kwargs.update(
        params_dtype=torch.float32,
        bf16=False,
        perform_initialization=True,
        use_cpu_initialization=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        attention_backend=AttnBackend.local,
        mtp_num_layers=None,
        recompute_granularity=None,
        gradient_accumulation_fusion=False,
        language_model_only=False)
    core_config = Inkling2LoopConfig(**kwargs)
    model = get_mcore_model(core_config)[0].cuda()
    bridge = core_config.bridge
    bridge.load_weights([model], str(source))
    hf_targets = ['self_attn.' + path for path in ATTENTION_PROJECTIONS]
    if tuner == 'lora':
        for layer in model.language_model.decoder.layers:
            for path in ATTENTION_PROJECTIONS:
                container = layer.self_attn.get_submodule(path)
                container.linear = Inkling2LoopLoraLinear(container.linear, 'default', r=4, lora_alpha=8)
        model.requires_grad_(False)
        for name, parameter in model.named_parameters():
            if 'lora_' in name:
                parameter.requires_grad_(True)
        hf = get_peft_model(hf, LoraConfig(r=4, lora_alpha=8, target_modules=hf_targets))
        with torch.no_grad():
            for index, layer in enumerate(model.language_model.decoder.layers):
                for path in ATTENTION_PROJECTIONS:
                    left = hf.get_base_model().model.language_model.layers[index].self_attn.get_submodule(path)
                    right = layer.self_attn.get_submodule(path).linear
                    for factor in ('lora_A', 'lora_B'):
                        destination = getattr(right, factor)['default'].weight
                        destination.normal_(0., .03)
                        getattr(left, factor)['default'].weight.copy_(destination)
    ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70]], device='cuda')
    positions = torch.arange(ids.shape[1], device='cuda').unsqueeze(0)
    if reload_only:
        if tuner == 'full':
            bridge.load_weights([model], str(root / 'export'))
        else:
            bridge.load_weights([model], str(root / 'export'), peft_format=True)
        expected = torch.load(root / f'expected_{dist.get_rank()}.pt', weights_only=True)
        actual = model(ids, positions, None)
        torch.testing.assert_close(actual[..., :120].cpu(), expected, rtol=0, atol=0)
        if tuner == 'full':
            independent = Inkling2LoopForConditionalGeneration.from_pretrained(
                root / 'export', dtype=torch.float32).cuda()
        else:
            independent = Inkling2LoopForConditionalGeneration.from_pretrained(source, dtype=torch.float32).cuda()
            independent = get_peft_model(independent, LoraConfig.from_pretrained(root / 'export'))
            factors = load_file(str(root / 'export/adapter_model.safetensors'))
            assert len(factors) == 2 * len(ATTENTION_PROJECTIONS) * config.text_config.num_hidden_layers
            with torch.no_grad():
                for name, value in factors.items():
                    key = name.removeprefix('base_model.model.')
                    key = key.replace('.lora_A.weight', '.lora_A.default.weight')
                    key = key.replace('.lora_B.weight', '.lora_B.default.weight')
                    target = independent.get_base_model().get_parameter(key)
                    assert target.shape == value.shape
                    target.copy_(value.to(target.device))
        torch.testing.assert_close(actual[..., :120], independent(input_ids=ids).logits, rtol=1e-4, atol=1e-5)
        print('FRESH_CORE_ADAPTER_OR_FULL_RELOAD_EXACT', ep, tuner, dist.get_rank(), flush=True)
    else:
        model.eval()
        hf.eval()
        actual = model(ids, positions, None)
        expected = hf(input_ids=ids).logits
        torch.testing.assert_close(actual[..., :120], expected, rtol=1e-4, atol=1e-5)
        assert torch.isneginf(actual[..., 120:]).all()
        actual[..., :120].square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        original = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        optimizer.step()
        assert any(not torch.equal(original[name], p) for name, p in model.named_parameters() if p.requires_grad)
        snapshots = list(_shards(model.sharded_state_dict()))
        assert snapshots and any('visual.vision_tower' in shard.key for shard in snapshots)
        expert_shards = [shard for shard in snapshots if '.experts.' in shard.key and 'weight' in shard.key]
        if tuner == 'full':
            assert expert_shards
            assert all(shard.global_shape[0] == 4 for shard in expert_shards)
        torch.save(model(ids, positions, None)[..., :120].detach().cpu(), root / f'expected_{dist.get_rank()}.pt')
        bridge.save_weights([model], str(root / 'export'), peft_format=tuner == 'lora')
        if dist.get_rank() == 0:
            config.save_pretrained(root / 'export')
            if tuner == 'lora':
                hf.peft_config['default'].save_pretrained(root / 'export')
        _test_packed_forward(model, ids, core_config)
        _test_bf16_unmerge(core_config, Inkling2TensorLinear, Inkling2LoopLoraLinear)
        _test_export_guards(model, bridge, root)
        print('CORE_FORWARD_UPDATE_EXPORT_LAYOUT_OK', ep, tuner, dist.get_rank(), flush=True)
    dist.barrier()
    mpu.destroy_model_parallel()
    dist.destroy_process_group()


def _test_packed_forward(model, ids, config):
    packed = PackedSeqParams(cu_seqlens_q=torch.tensor([0, 3, 7], device='cuda', dtype=torch.int32), qkv_format='thd')
    positions = torch.arange(7, device='cuda').unsqueeze(0)
    actual = model(ids, positions, None, packed_seq_params=packed)[..., :120]
    expected = torch.cat(
        [model(ids[:, :3], positions[:, :3], None)[..., :120],
         model(ids[:, 3:], positions[:, 3:], None)[..., :120]],
        dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    changed = ids.clone()
    changed[:, :3] = torch.tensor([90, 91, 92], device='cuda')
    after = model(changed, positions, None, packed_seq_params=packed)[..., :120]
    torch.testing.assert_close(actual[:, 3:], after[:, 3:], rtol=1e-4, atol=1e-5)
    embeddings = []

    def capture(module, arguments, output):
        output.requires_grad_(True)
        output.retain_grad()
        embeddings.append(output)

    handle = model.language_model.embedding.register_forward_hook(capture)
    try:
        model(ids, positions, None, packed_seq_params=packed)[:, 3:, :120].square().mean().backward()
    finally:
        handle.remove()
    assert embeddings[0].shape[:2] == (7, 1)
    assert torch.equal(embeddings[0].grad[:3], torch.zeros_like(embeddings[0].grad[:3]))


def _test_export_guards(model, bridge, root):
    controller = model.language_model.decoder.layers[2].q_message_gate
    original = controller.down_proj
    controller.down_proj = PeftLinear(original, 'default', config=LoraConfig(r=2, lora_alpha=4), r=2, lora_alpha=4)
    try:
        with pytest.raises(ValueError, match='cannot preserve'):
            bridge.save_weights([model], str(root / 'unmapped_export'), peft_format=True)
        assert not (root / 'unmapped_export').exists()
    finally:
        controller.down_proj = original
    previous = bridge.config.language_model_only
    bridge.config.language_model_only = True
    try:
        with pytest.raises(ValueError, match='save_missing_weights=True'):
            bridge.save_weights([model], str(root / 'incomplete_export'), save_missing_weights=False)
        assert not (root / 'incomplete_export').exists()
    finally:
        bridge.config.language_model_only = previous


def _test_bf16_unmerge(config, linear_class, lora_class):
    config = deepcopy(config)
    config.params_dtype = torch.bfloat16
    layer = linear_class(16, 16, config).cuda()
    layer.linear = lora_class(layer.linear, 'default', r=4, lora_alpha=8)
    with torch.no_grad():
        layer.linear.lora_A['default'].weight.normal_(0., .1)
        layer.linear.lora_B['default'].weight.normal_(0., .1)
    x = torch.randn(7, 1, 16, dtype=torch.bfloat16, device='cuda')
    weight = layer.linear.get_base_layer().weight.detach().clone()
    before = layer(x).detach()
    for _ in range(2):
        layer.linear.merge()
        assert layer.linear.merged
        layer.linear.unmerge()
        assert not layer.linear.merged and layer.linear._unmerged_weight is None
        assert torch.equal(layer.linear.get_base_layer().weight, weight)
        assert torch.equal(layer(x), before)
    layer.linear.enable_adapters(False)
    assert not torch.equal(layer(x), before)


@pytest.mark.parametrize('ep', [1, 2])
@pytest.mark.parametrize('tuner', ['full', 'lora'])
def test_cuda_full_and_lora_fresh_roundtrip(ep, tuner, tmp_path):
    pytest.importorskip('transformers.models.inkling2_loop')
    if torch.cuda.device_count() < ep:
        pytest.skip(f'Requires {ep} CUDA devices.')
    env = dict(os.environ, NVIDIA_TF32_OVERRIDE='0', OMP_NUM_THREADS='4')
    args = [
        sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={ep}',
        str(Path(__file__)), '--worker',
        str(tmp_path),
        str(ep), tuner
    ]
    for extra in ([], ['reload']):
        result = subprocess.run(args + extra, env=env, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stdout + result.stderr


if __name__ == '__main__':
    _worker(Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4], len(sys.argv) > 5)
