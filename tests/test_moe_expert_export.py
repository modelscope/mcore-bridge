"""Parity tests for the MoE expert-export fast paths in ``gpt_bridge``.

``_store_ep_shard`` replaces the O(ep_size^2) accumulate-then-concat of the grouped HF layout, and
the one-shot gathers in ``_export_experts_ungrouped`` / ``_export_experts_grouped`` replace the
per-EP-rank broadcast plus per-expert clone. All of them must stay bit-identical to the paths they
replace, including under pipeline / expert-tensor parallelism.

Each topology needs its own world size, so the parallel case is selected by env and run once per
size:

    MOE_EXPORT_TOPOLOGY=pp2     torchrun --nproc_per_node=2 -m pytest tests/test_moe_expert_export.py -k parallel
    MOE_EXPORT_TOPOLOGY=ep2     torchrun --nproc_per_node=2 -m pytest tests/test_moe_expert_export.py -k parallel
    MOE_EXPORT_TOPOLOGY=etp2    torchrun --nproc_per_node=2 -m pytest tests/test_moe_expert_export.py -k parallel
    MOE_EXPORT_TOPOLOGY=pp2ep2  torchrun --nproc_per_node=4 -m pytest tests/test_moe_expert_export.py -k parallel
    MOE_EXPORT_TOPOLOGY=etp2ep2 torchrun --nproc_per_node=4 -m pytest tests/test_moe_expert_export.py -k parallel
"""

import os
import pytest
import torch
import uuid

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')

# topologies selected with MOE_EXPORT_TOPOLOGY=<name>; world size = tp * pp * ep
PARALLEL_CASES = {
    'pp2': (1, 2, 1, 1),
    'ep2': (1, 1, 2, 1),
    'etp2': (2, 1, 1, 2),
    'pp2ep2': (1, 2, 2, 1),
    'etp2ep2': (2, 1, 2, 2),
}


def _world_size():
    return int(os.environ.get('WORLD_SIZE', '1'))


_INITIALIZED = None


def _init_parallel(tp=1, pp=1, ep=1, etp=1):
    global _INITIALIZED
    expected = tp * pp * ep
    if _world_size() != expected:
        pytest.skip(f'requires world size {expected}')
    if _INITIALIZED == (tp, pp, ep, etp):
        return
    assert _INITIALIZED is None, 'Megatron parallel state is initialised once per process'
    _INITIALIZED = (tp, pp, ep, etp)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    if not torch.distributed.is_initialized():
        if expected == 1:
            torch.distributed.init_process_group(
                'nccl', init_method=f'file:///tmp/moe-expert-export-{uuid.uuid4().hex}', rank=0, world_size=1)
        else:
            torch.distributed.init_process_group('nccl')
    from megatron.core import parallel_state
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
        context_parallel_size=1,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(123)


def _build_tiny_moe(grouped=False, bias=True, tp=1, pp=1, ep=1, etp=1):
    from transformers import GptOssConfig, Qwen3MoeConfig

    from mcore_bridge.config import ModelConfig
    from mcore_bridge.config.parser import hf_to_mcore_config
    from mcore_bridge.model.register import get_mcore_model

    if grouped:
        hf_config = GptOssConfig(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            num_local_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            sliding_window=8,
            max_position_embeddings=32,
        )
    else:
        hf_config = Qwen3MoeConfig(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=64,
            moe_intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            max_position_embeddings=64,
        )
    values = hf_to_mcore_config(hf_config)
    values.update(
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        perform_initialization=True,
        use_cpu_initialization=False,
        moe_grouped_gemm=True,
        overlap_p2p_comm=False,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
        context_parallel_size=1,
        sequence_parallel=False,
    )
    if not bias:
        values['add_bias_linear'] = False
    config = ModelConfig(**values)
    return config, get_mcore_model(config)[0].cuda().eval()


def _assert_fast_matches_legacy(grouped, mode, bias=True, **parallel):
    config, model = _build_tiny_moe(grouped=grouped, bias=bias, **parallel)
    bridge = config.bridge
    fast = dict(bridge.export_weights([model], disable_tqdm=True))
    assert bridge._expert_export_mode(False, False) == mode, 'fixture must exercise the fast path'

    original = type(bridge)._expert_export_mode
    type(bridge)._expert_export_mode = lambda self, to_mcore, is_mtp: None
    try:
        legacy = dict(bridge.export_weights([model], disable_tqdm=True))
    finally:
        type(bridge)._expert_export_mode = original

    assert set(fast) == set(legacy)
    for key, value in fast.items():
        assert value.shape == legacy[key].shape, key
        assert value.dtype == legacy[key].dtype, key
        assert torch.equal(value, legacy[key]), key


def test_store_ep_shard_writes_slots_in_ep_order():
    """The slot write must reproduce ``torch.concat(shards, dim=0)`` exactly."""
    from mcore_bridge.bridge.gpt_bridge import GPTBridge

    bridge = GPTBridge.__new__(GPTBridge)  # the helper only needs ep_size
    bridge.ep_size = 3
    state = {}
    shards = [torch.full((2, 3), float(rank)) for rank in range(3)]

    for rank, shard in enumerate(shards):
        bridge._store_ep_shard(state, 'w', shard, rank)

    assert torch.equal(state['w'], torch.concat(shards, dim=0))


def test_ungrouped_expert_export_matches_legacy_path():
    """One all-gather plus per-expert views must equal the broadcast plus per-expert clone."""
    _init_parallel()
    _assert_fast_matches_legacy(False, 'ungrouped')


def test_grouped_expert_export_matches_legacy_path():
    """The grouped layout (transpose + gpt_oss interleave before the gather) must match too."""
    _init_parallel()
    _assert_fast_matches_legacy(True, 'grouped')


def test_packed_param_formats_take_the_legacy_path():
    """NVFP4 experts must stay legacy: only ``_get_weight`` dequantizes each shard first."""
    _init_parallel()
    config, model = _build_tiny_moe()
    bridge = config.bridge
    if not hasattr(config, 'fp4_param'):
        pytest.skip('this megatron build has no fp4_param')
    assert bridge._expert_export_mode(False, False) == 'ungrouped'

    config.fp4_param = True
    try:
        assert bridge._expert_export_mode(False, False) is None
    finally:
        config.fp4_param = False
    assert bridge._expert_export_mode(False, False) == 'ungrouped'


def test_parallel_expert_export_matches_legacy_path():
    """Same parity under pipeline / expert / expert-tensor parallelism (world size 2–4)."""
    topology = os.environ.get('MOE_EXPORT_TOPOLOGY', 'pp2')
    tp, pp, ep, etp = PARALLEL_CASES[topology]
    _init_parallel(tp, pp, ep, etp)
    _assert_fast_matches_legacy(False, 'ungrouped', tp=tp, pp=pp, ep=ep, etp=etp)
    # grouped covers gpt_oss (bias + interleave) and, with etp>1, the llama4-style layout
    # (grouped without bias) that expert-TP sharding keeps on the fast path.
    _assert_fast_matches_legacy(True, 'grouped', bias=etp == 1, tp=tp, pp=pp, ep=ep, etp=etp)


def test_save_weights_grouped_matches_export(tmp_path):
    """The checkpoint path gathers to CPU on the master rank; the slot writes must survive it."""
    from safetensors.torch import load_file

    _init_parallel()
    config, model = _build_tiny_moe(grouped=True)
    bridge = config.bridge
    exported = dict(bridge.export_weights([model], disable_tqdm=True))

    bridge.save_weights([model], str(tmp_path))

    saved = {}
    for name in sorted(os.listdir(tmp_path)):
        if name.endswith('.safetensors'):
            saved.update(load_file(tmp_path / name))
    assert set(saved) == set(exported)
    for key, value in exported.items():
        assert saved[key].dtype == value.dtype, key
        assert torch.equal(saved[key], value.cpu()), key
