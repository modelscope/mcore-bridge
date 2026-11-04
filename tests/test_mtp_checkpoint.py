# Copyright (c) ModelScope Contributors. All rights reserved.
"""Config and persistence contracts for the upstream-MTP adaptation layer.

Locks the two properties that changed around persistence:

* ``mtp_num_layers`` is the *logical* prediction depth and is never rewritten
  to 1 for shared weights; ``mtp_shared_weights`` aliases
  ``mtp_use_repeated_layer``.
* Repeated (shared-weight) mode persists exactly ONE physical MTP weight set,
  matching the HF checkpoint layout the bridge must keep reading.

Building a ``ModelConfig`` instantiates its bridge, which requires a
model-parallel group, so every case runs inside a single-GPU parallel context.
"""
import json
import os
import pytest
import torch
import uuid
from contextlib import contextmanager
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from mcore_bridge.config import ModelConfig
from mcore_bridge.config.parser import hf_to_mcore_config
from mcore_bridge.model.mtp import get_physical_mtp_count
from mcore_bridge.model.mtp_detect import resolve_mtp_config
from mcore_bridge.model.register import get_mcore_model, get_model_meta

# A model build/forward makes Megatron's attention-backend selection export
# these globally; snapshot/restore them so this file does not leak backend
# state into other test files run in the same process.
_NVTE_ENV = ('NVTE_FLASH_ATTN', 'NVTE_FUSED_ATTN', 'NVTE_UNFUSED_ATTN')


@pytest.fixture(scope='session', autouse=True)
def _distributed_session():
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@contextmanager
def _parallel_context():
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        pytest.skip('requires world size 1')
    saved_env = {k: os.environ.get(k) for k in _NVTE_ENV}
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    torch.distributed.init_process_group(
        'nccl', init_method=f'file:///tmp/mtp-ckpt-{uuid.uuid4().hex}', rank=0, world_size=1)
    parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()
        from mcore_bridge.bridge import gpt_bridge
        gpt_bridge.EP_PP_GROUP = gpt_bridge.EP_PP_RANK = gpt_bridge.EP_PP_SIZE = None
        torch.distributed.destroy_process_group()
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _llama_values(**overrides):
    transformers = pytest.importorskip('transformers')
    hf = overrides.pop('hf', None)
    if hf is None:
        hf = transformers.LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            tie_word_embeddings=False,
        )
    values = hf_to_mcore_config(hf)
    values.update(
        params_dtype=torch.float32,
        pipeline_dtype=torch.float32,
        perform_initialization=True,
        use_cpu_initialization=False)
    values.update(overrides)
    return values


def _mtp_layer_indices(state_dict):
    return sorted({k.split('.')[2] for k in state_dict if k.startswith('mtp.layers.')})


def test_model_config_accepts_resolved_mtp(tmp_path):
    values = _llama_values()
    hf = values['hf_config']
    with open(tmp_path / 'model.safetensors.index.json', 'w') as f:
        json.dump({'weight_map': {'model.layers.2.eh_proj.weight': 'x'}}, f)
    with _parallel_context():
        for requested_depth, depth, shared in (('auto', 1, False), (3, 3, True)):
            values['mtp_num_layers'], values['mtp_shared_weights'] = resolve_mtp_config(
                get_model_meta('gpt'), hf, str(tmp_path), requested_depth, None)
            cfg = ModelConfig(**values)
            assert cfg.mtp_num_layers == depth
            assert cfg.mtp_use_repeated_layer is shared


def test_model_config_rejects_unresolved_auto():
    with pytest.raises(TypeError, match='resolve_mtp_config'):
        ModelConfig(**_llama_values(mtp_num_layers='auto'))


def test_shared_weights_alias_maps_to_repeated_layer():
    with _parallel_context():
        cfg = ModelConfig(**_llama_values(mtp_num_layers=3, mtp_shared_weights=True))
        assert cfg.mtp_use_repeated_layer is True
        # logical depth is preserved, NOT collapsed to 1
        assert cfg.mtp_num_layers == 3
        # legacy alias tracks the logical depth
        assert cfg.mtp_unroll_steps == 3


def test_independent_mode_persists_one_weight_set_per_depth():
    with _parallel_context():
        model = get_mcore_model(ModelConfig(**_llama_values(mtp_num_layers=2)))[0].cuda()
        assert get_physical_mtp_count(model) == 2
        assert _mtp_layer_indices(model.state_dict()) == ['0', '1']


def test_repeated_mode_persists_single_weight_set():
    with _parallel_context():
        model = get_mcore_model(ModelConfig(**_llama_values(mtp_num_layers=3, mtp_shared_weights=True)))[0].cuda()
        # logical depth 3, but exactly one physical layer / weight set
        assert model.config.mtp_num_layers == 3
        assert model.config.mtp_use_repeated_layer is True
        assert get_physical_mtp_count(model) == 1
        assert _mtp_layer_indices(model.state_dict()) == ['0']


def _mtp_named_params(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith('mtp.')}


def _roundtrip(values, tmpdir):
    """Build model_a, save HF weights, build a DIFFERENTLY-SEEDED model_b,
    load the saved weights into it and compare parameter values exactly."""
    model_parallel_cuda_manual_seed(123)
    cfg_a = ModelConfig(**values)
    model_a = get_mcore_model(cfg_a)[0].cuda()
    params_a = _mtp_named_params(model_a)
    assert params_a, 'model_a has no MTP parameters'
    cfg_a.bridge.save_weights([model_a], tmpdir)

    model_parallel_cuda_manual_seed(999)
    cfg_b = ModelConfig(**values)
    model_b = get_mcore_model(cfg_b)[0].cuda()
    params_b_before = _mtp_named_params(model_b)
    # the fresh build must actually differ, otherwise the comparison is vacuous
    assert any(not torch.equal(params_a[n], params_b_before[n]) for n in params_a), \
        'seed change did not change MTP init -- round trip would be vacuous'

    cfg_b.bridge.load_weights([model_b], tmpdir)
    params_b = _mtp_named_params(model_b)
    assert params_a.keys() == params_b.keys()
    mismatched = [n for n in params_a if not torch.equal(params_a[n], params_b[n])]
    assert not mismatched, f'MTP parameters changed across the HF round trip: {mismatched}'
    return model_a, model_b


def test_independent_mode_hf_roundtrip(tmp_path):
    with _parallel_context():
        _roundtrip(_llama_values(mtp_num_layers=2), str(tmp_path / 'rt-independent'))


def test_repeated_mode_hf_roundtrip_single_set(tmp_path):
    with _parallel_context():
        values = _llama_values(mtp_num_layers=3, mtp_shared_weights=True)
        model_a, model_b = _roundtrip(values, str(tmp_path / 'rt-repeated'))
        # repeated mode: both models persist exactly one physical weight set
        assert get_physical_mtp_count(model_a) == 1
        assert get_physical_mtp_count(model_b) == 1
        assert _mtp_layer_indices(model_b.state_dict()) == ['0']


def _save_model_without_mtp(path):
    model_parallel_cuda_manual_seed(123)
    cfg = ModelConfig(**_llama_values())
    model = get_mcore_model(cfg)[0].cuda()
    cfg.bridge.save_weights([model], str(path))


def test_missing_mtp_weights_fail_loud(tmp_path):
    """A full-finetune checkpoint without MTP keys must not silently random-init."""
    with _parallel_context():
        ckpt = tmp_path / 'no-mtp'
        _save_model_without_mtp(ckpt)

        cfg = ModelConfig(**_llama_values(mtp_num_layers=1))
        model = get_mcore_model(cfg)[0].cuda()
        with pytest.raises(ValueError, match='MTP weights not found'):
            cfg.bridge.load_weights([model], str(ckpt))


def test_missing_mtp_weights_allowed_with_flag(tmp_path):
    with _parallel_context():
        ckpt = tmp_path / 'no-mtp-flag'
        _save_model_without_mtp(ckpt)

        cfg = ModelConfig(**_llama_values(mtp_num_layers=1, mtp_allow_random_init=True))
        model = get_mcore_model(cfg)[0].cuda()
        cfg.bridge.load_weights([model], str(ckpt))


def test_roundtrip_decoder_params_match(tmp_path):
    """The round trip must also preserve non-MTP weights (guards against the
    MTP branch corrupting the shared export stream)."""
    with _parallel_context():
        model_parallel_cuda_manual_seed(123)
        cfg_a = ModelConfig(**_llama_values(mtp_num_layers=1))
        model_a = get_mcore_model(cfg_a)[0].cuda()
        decoder_a = {
            n: p.detach().clone()
            for n, p in model_a.named_parameters() if n.startswith('decoder.') and p.ndim == 2
        }
        cfg_a.bridge.save_weights([model_a], str(tmp_path / 'rt-decoder'))
        model_parallel_cuda_manual_seed(999)
        cfg_b = ModelConfig(**_llama_values(mtp_num_layers=1))
        model_b = get_mcore_model(cfg_b)[0].cuda()
        cfg_b.bridge.load_weights([model_b], str(tmp_path / 'rt-decoder'))
        for n, p_a in decoder_a.items():
            p_b = dict(model_b.named_parameters())[n].detach()
            assert torch.equal(p_a, p_b), f'decoder param {n} changed across round trip'
