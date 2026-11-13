# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for the fp8-resident Engram tables against a stub of the upstream contract (the real
``megatron.core.models.engram`` package is absent from most installs).
"""
import importlib
import importlib.util
import os
import pytest
import sys
import torch
import types
from types import SimpleNamespace

import mcore_bridge.model.modules.engram_fp8 as engram_fp8


def _get_contiguous_row_range(global_rows, rank, world_size):
    base, remainder = divmod(global_rows, world_size)
    start = rank * base + min(rank, remainder)
    return start, start + base + int(rank < remainder)


def _install_stub_upstream(monkeypatch):
    from megatron.core.transformer.module import MegatronModule

    package = types.ModuleType('megatron.core.models.engram')
    package.__path__ = []
    module = types.ModuleType('megatron.core.models.engram.distributed_embedding')

    class EPShardedEmbeddingTable(MegatronModule):

        def __init__(self,
                     config,
                     global_num_embeddings,
                     embedding_dim,
                     init_method=None,
                     ep_group=None,
                     tp_group=None,
                     expt_dp_group=None):
            super().__init__(config=config)

        @property
        def local_num_embeddings(self) -> int:
            return self.row_end - self.row_start

    class EPShardedMultiTableEmbedding(MegatronModule):

        def forward(self, hash_ids):
            if hash_ids.shape[-1] != len(self.tables):
                raise ValueError(f'Expected {len(self.tables)} Engram hash heads, got {hash_ids.shape[-1]}.')
            shape = hash_ids.shape
            values = [table(hash_ids[..., index].reshape(-1)) for index, table in enumerate(self.tables)]
            return torch.stack(values, dim=-2).view(*shape, self.embedding_dim)

    module.EPShardedEmbeddingTable = EPShardedEmbeddingTable
    module.EPShardedMultiTableEmbedding = EPShardedMultiTableEmbedding
    module.get_contiguous_row_range = _get_contiguous_row_range
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.fixture(scope='module')
def single_rank_group():
    import torch.distributed as dist

    if not dist.is_initialized():
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29591')
        try:
            dist.init_process_group('gloo', world_size=1, rank=0)
        except Exception as exc:  # pragma: no cover - environment without a usable gloo backend
            pytest.skip(f'gloo is unavailable: {exc}')
    yield


@pytest.fixture()
def fp8_module(monkeypatch):
    """Load an isolated copy of the module against the stub, leaving the real module untouched."""
    _install_stub_upstream(monkeypatch)
    spec = importlib.util.spec_from_file_location('engram_fp8_under_stub', engram_fp8.__file__)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = engram_fp8.__package__
    spec.loader.exec_module(module)
    assert module.Fp8EngramTable is not None, 'stub upstream was not picked up'
    return module


def _config(**overrides):
    base = dict(
        use_cpu_initialization=True, params_dtype=torch.bfloat16, sequence_parallel=False, deterministic_mode=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def _bits(rows, width, block, seed=0):
    generator = torch.Generator().manual_seed(seed)
    payload = (torch.randn(rows, width, generator=generator) * 4).to(torch.float8_e4m3fn)
    # e8m0 stores only the biased exponent byte
    scales = torch.randint(120, 130, (rows, -(-width // block)), generator=generator, dtype=torch.uint8)
    return payload, scales


def test_decode_fp8_rows_matches_manual_reference(fp8_module):
    payload, scales = _bits(6, 64, 32)
    decoded = fp8_module.decode_fp8_rows(payload.view(torch.uint8), scales, 32, torch.bfloat16)
    reference = (payload.float().unflatten(-1, (-1, 32)) * scales.view(torch.float8_e8m0fnu).float().unsqueeze(-1))
    reference = reference.flatten(-2).to(torch.bfloat16)
    assert torch.equal(decoded, reference)


def test_decode_fp8_rows_pads_width_not_divisible_by_block(fp8_module):
    payload, scales = _bits(3, 40, 32)
    decoded = fp8_module.decode_fp8_rows(payload.view(torch.uint8), scales, 32, torch.bfloat16)
    assert decoded.shape == (3, 40)
    padded = torch.nn.functional.pad(payload.float(), (0, 24)).unflatten(-1, (-1, 32))
    reference = (padded * scales.view(torch.float8_e8m0fnu).float().unsqueeze(-1)).flatten(-2)[:, :40]
    assert torch.equal(decoded, reference.to(torch.bfloat16))


def test_table_keeps_fp8_payload_and_no_bf16_shard(fp8_module):
    table = fp8_module.Fp8EngramTable(config=_config(), global_num_embeddings=96, embedding_dim=64, ep_group=None)
    assert table.row_start == 0 and table.row_end == 96
    assert table.weight.numel() == 0, 'the bf16 shard must stay a zero-row placeholder'
    assert table.fp8_rows.shape == (96, 64) and table.fp8_rows.dtype == torch.uint8
    assert table.fp8_scales.shape == (96, 2) and table.fp8_scales.dtype == torch.uint8
    assert table.local_parameter_count == 96 * 64
    assert table.local_storage_bytes == 96 * 64 + 96 * 2


def test_fill_rows_then_forward_reproduces_checkpoint_bits(fp8_module):
    table = fp8_module.Fp8EngramTable(config=_config(), global_num_embeddings=8, embedding_dim=64, ep_group=None)
    payload, scales = _bits(8, 64, 32, seed=1)
    table.fill_rows(0, 8, payload, scales)

    ids = torch.tensor([[3, 1], [7, 0]])
    out = table(ids)
    assert out.shape == (2, 2, 64) and out.dtype == torch.bfloat16
    reference = fp8_module.decode_fp8_rows(payload.view(torch.uint8), scales, 32, torch.bfloat16)
    assert torch.equal(out, reference[ids])


def test_fill_rows_accepts_uint8_payload(fp8_module):
    table = fp8_module.Fp8EngramTable(config=_config(), global_num_embeddings=4, embedding_dim=32, ep_group=None)
    payload, scales = _bits(4, 32, 32, seed=2)
    table.fill_rows(0, 4, payload.view(torch.uint8), scales)
    assert torch.equal(table.fp8_rows, payload.view(torch.uint8))


def test_unfilled_table_decodes_to_zero_without_raising(fp8_module):
    table = fp8_module.Fp8EngramTable(config=_config(), global_num_embeddings=4, embedding_dim=32, ep_group=None)
    assert table(torch.tensor([0, 1])).abs().sum() == 0
    payload, scales = _bits(2, 32, 32, seed=4)
    table.fill_rows(0, 2, payload, scales)
    assert table._filled_rows == 2
    assert table(torch.tensor([0, 1])).abs().sum() > 0
    assert table(torch.tensor([2, 3])).abs().sum() == 0


def test_multitable_forward_shape_and_table_order(fp8_module):
    embedding = fp8_module.Fp8EngramEmbedding(config=_config(), table_sizes=(5, 7), embedding_dim=32, ep_group=None)
    assert [tuple(t.fp8_rows.shape) for t in embedding.tables] == [(5, 32), (7, 32)]
    payloads, scales = zip(*[_bits(rows, 32, 32, seed=10 + i) for i, rows in enumerate((5, 7))])
    for table, payload, scale in zip(embedding.tables, payloads, scales):
        table.fill_rows(0, payload.shape[0], payload, scale)

    hash_ids = torch.tensor([[[0, 0], [4, 6]], [[2, 3], [1, 5]]])
    out = embedding(hash_ids)
    assert out.shape == (2, 2, 2, 32)
    for table_index, payload in enumerate(payloads):
        expected = fp8_module.decode_fp8_rows(payload.view(torch.uint8), scales[table_index], 32, torch.bfloat16)
        assert torch.equal(out[..., table_index, :], expected[hash_ids[..., table_index]])


def test_multitable_requires_one_head_per_table(fp8_module):
    embedding = fp8_module.Fp8EngramEmbedding(config=_config(), table_sizes=(5, 7), embedding_dim=32, ep_group=None)
    # the inherited forward keeps upstream's head/table guard
    with pytest.raises(ValueError, match='hash heads'):
        embedding(torch.zeros(2, 2, 3, dtype=torch.int64))


def test_fp8_tables_follow_the_checkpoint_declaration(fp8_module):
    assert not fp8_module.fp8_quant_declared(SimpleNamespace())
    assert not fp8_module.fp8_quant_declared(SimpleNamespace(hf_config=None))
    ue8m0 = SimpleNamespace(hf_config=SimpleNamespace(quantization_config={'scale_fmt': 'ue8m0'}))
    assert fp8_module.fp8_quant_declared(ue8m0)
    block32 = SimpleNamespace(hf_config=SimpleNamespace(quantization_config={'weight_block_size': (32, 32)}))
    assert fp8_module.fp8_quant_declared(block32)
    block128 = SimpleNamespace(hf_config=SimpleNamespace(quantization_config={'weight_block_size': [128, 128]}))
    assert not fp8_module.fp8_quant_declared(block128)
    as_object = SimpleNamespace(hf_config=SimpleNamespace(quantization_config=SimpleNamespace(scale_fmt='ue8m0')))
    assert fp8_module.fp8_quant_declared(as_object)


def test_state_dict_round_trip_carries_the_fp8_payload(fp8_module):
    table = fp8_module.Fp8EngramTable(config=_config(), global_num_embeddings=8, embedding_dim=64, ep_group=None)
    payload, scales = _bits(8, 64, 32, seed=6)
    table.fill_rows(0, 8, payload, scales)

    state = {key: value.clone() for key, value in table.state_dict().items()}
    assert set(state) == {'weight', 'weight_scale_inv'}
    assert state['weight'].dtype == torch.float8_e4m3fn and tuple(state['weight'].shape) == (8, 64)
    assert state['weight_scale_inv'].dtype == torch.float8_e8m0fnu

    restored = fp8_module.Fp8EngramTable(config=_config(), global_num_embeddings=8, embedding_dim=64, ep_group=None)
    restored.load_state_dict(state)
    assert torch.equal(restored.fp8_rows, table.fp8_rows)
    assert torch.equal(restored.fp8_scales, table.fp8_scales)
    assert restored._filled_rows == 8
    ids = torch.tensor([[0, 7], [3, 5]])
    assert torch.equal(restored(ids), table(ids))


def test_state_dict_rejects_pre_fp8_and_partial_checkpoints(fp8_module):
    table = fp8_module.Fp8EngramTable(config=_config(), global_num_embeddings=8, embedding_dim=64, ep_group=None)
    with pytest.raises(RuntimeError, match='pre-fp8 layout'):
        table.load_state_dict({'weight': torch.zeros(8, 64, dtype=torch.bfloat16)})
    with pytest.raises(RuntimeError, match='needs both weight and weight_scale_inv'):
        table.load_state_dict({'weight': torch.zeros(8, 64, dtype=torch.float8_e4m3fn)})
    with pytest.raises(RuntimeError, match='not e8m0'):
        table.load_state_dict({
            'weight': torch.zeros(8, 64, dtype=torch.float8_e4m3fn),
            'weight_scale_inv': torch.zeros(8, 2, dtype=torch.float32)
        })
    assert table._filled_rows == 0


def test_torch_dist_round_trip_restores_the_table(fp8_module, single_rank_group, tmp_path):
    from megatron.core.dist_checkpointing import load, save

    table = fp8_module.Fp8EngramTable(config=_config(), global_num_embeddings=8, embedding_dim=64, ep_group=None)
    payload, scales = _bits(8, 64, 32, seed=7)
    table.fill_rows(0, 8, payload, scales)
    ids = torch.tensor([[0, 1, 7]])
    expected = table(ids)

    save(table.sharded_state_dict(), str(tmp_path))
    restored = fp8_module.Fp8EngramTable(config=_config(), global_num_embeddings=8, embedding_dim=64, ep_group=None)
    restored.load_state_dict(load(restored.sharded_state_dict(), str(tmp_path)))
    assert torch.equal(restored.fp8_rows, table.fp8_rows)
    assert torch.equal(restored.fp8_scales, table.fp8_scales)
    assert restored._filled_rows == 8
    assert torch.equal(restored(ids), expected)


def test_sharded_state_dict_exposes_loadable_planes(fp8_module):
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    table = fp8_module.Fp8EngramTable(config=_config(), global_num_embeddings=8, embedding_dim=64, ep_group=None)
    sharded = table.sharded_state_dict(prefix='tables.0.')
    assert set(sharded) == {'tables.0.weight', 'tables.0.weight_scale_inv'}
    assert all(isinstance(value, ShardedTensor) for value in sharded.values())
    assert tuple(sharded['tables.0.weight'].global_shape) == (8, 64)
    assert tuple(sharded['tables.0.weight_scale_inv'].global_shape) == (8, 2)


def test_reinterpret_does_not_convert_values(fp8_module):
    payload, _ = _bits(2, 32, 32, seed=3)
    raw = fp8_module.reinterpret_u8(payload)
    assert raw.dtype == torch.uint8
    assert torch.equal(raw, payload.view(torch.uint8))
    same = fp8_module.reinterpret_u8(raw)
    assert same.data_ptr() == raw.data_ptr()
