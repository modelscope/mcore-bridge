# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

native = pytest.importorskip('megatron.core.models.engram.distributed_embedding')
from mcore_bridge.model.modules.engram_cpu import retain_tables_on_cpu  # noqa: E402


@pytest.fixture
def embedding():
    config = SimpleNamespace(
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        perform_initialization=False,
        sequence_parallel=False,
        deterministic_mode=False)
    module = native.EPShardedMultiTableEmbedding(config, (5, 7, 11), 4, lambda value: value)
    for table_id, table in enumerate(module.tables):
        table.weight.data.copy_(torch.arange(table.weight.numel()).reshape_as(table.weight) / 16 + table_id)
    retain_tables_on_cpu(module)
    return module


@pytest.mark.parametrize('shape', [(2, 6, 3), (1, 0, 3)])
def test_lookup_and_consumer_gradient(embedding, shape):
    ids = torch.zeros(shape, dtype=torch.long)
    for table_id, size in enumerate(embedding.table_sizes):
        ids[..., table_id] = torch.arange(ids[..., table_id].numel()).reshape(ids[..., table_id].shape) % size
    expected = torch.stack([F.embedding(ids[..., i], table.weight) for i, table in enumerate(embedding.tables)], dim=-2)
    actual = embedding(ids)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not list(embedding.parameters())
    assert all(not table.weight.requires_grad for table in embedding.tables)
    if actual.numel():
        projection = torch.ones(4, 2, requires_grad=True)
        (actual @ projection).sum().backward()
        reference = expected.reshape(-1, 4).sum(0)[:, None].expand(4, 2)
        torch.testing.assert_close(projection.grad, reference, rtol=0, atol=0)


def test_owner_lookup_restores_request_order(embedding):
    requests = torch.tensor([[2, 3], [0, 2], [2, 1], [0, 0], [2, 1]])
    expected = torch.stack([embedding.tables[table].weight[row] for table, row in requests])
    actual = embedding._lookup_received_requests(requests, [2, 0, 3])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_state_dict_and_shard_metadata(embedding):
    state = {name: value.clone() for name, value in embedding.state_dict().items()}
    assert set(state) == {f'tables.{i}.weight' for i in range(3)}
    for table in embedding.tables:
        table.weight.zero_()
    embedding.load_state_dict(state, strict=True)
    for i, table in enumerate(embedding.tables):
        torch.testing.assert_close(table.weight, state[f'tables.{i}.weight'], rtol=0, atol=0)
        shard = embedding.sharded_state_dict()[f'tables.{i}.weight']
        assert shard.data.device.type == 'cpu'
        assert shard.global_shape == (embedding.table_sizes[i], 4)
        assert shard.global_offset == (0, 0)


@pytest.mark.parametrize('problem', ['missing', 'unexpected', 'shape'])
def test_strict_load_reports_incompatible_state(embedding, problem):
    state = embedding.state_dict()
    if problem == 'missing':
        state.pop('tables.0.weight')
    elif problem == 'unexpected':
        state['tables.0.other'] = torch.zeros(1)
    else:
        state['tables.0.weight'] = torch.zeros(2, 4)
    with pytest.raises(RuntimeError):
        embedding.load_state_dict(state, strict=True)


def test_keep_vars_load_preserves_frozen_host_tables(embedding):
    source = torch.nn.Module()
    source.tables = torch.nn.ModuleList(
        [torch.nn.Embedding(table.weight.shape[0], table.weight.shape[1]) for table in embedding.tables])
    state = source.state_dict(keep_vars=True)
    assert all(value.requires_grad for value in state.values())
    with torch.enable_grad():
        embedding.load_state_dict(state, strict=True)
    for table in embedding.tables:
        assert not table.weight.requires_grad
        assert table.weight.grad_fn is None
    ids = torch.zeros((1, 2, len(embedding.tables)), dtype=torch.long)
    assert not embedding(ids).requires_grad


@pytest.mark.parametrize('strict', [True, False])
@pytest.mark.parametrize('format', ['fp8_with_scales', 'fp8_without_scales', 'bf16_with_scales', 'int8'])
def test_quantized_state_is_rejected_without_overwriting_table(embedding, strict, format):
    state = {k: v.clone() for k, v in embedding.state_dict().items()}
    before = embedding.tables[0].weight.clone()
    dtype = torch.bfloat16 if format == 'bf16_with_scales' else (
        torch.int8 if format == 'int8' else torch.float8_e4m3fn)
    state['tables.0.weight'] = torch.ones_like(before).to(dtype)
    if format.endswith('with_scales'):
        state['tables.0.weight_scale_inv'] = torch.full((before.shape[0], 1), 4.)
    with pytest.raises(RuntimeError, match='requires dequantized floating-point weights'):
        embedding.load_state_dict(state, strict=strict)
    torch.testing.assert_close(embedding.tables[0].weight, before, rtol=0, atol=0)
    assert not embedding.tables[0].weight.requires_grad


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_module_dtype_conversion_keeps_lookup_and_projection_compatible(embedding, dtype):
    model = torch.nn.Module()
    model.embedding = embedding
    model.projection = torch.nn.Linear(12, 2, bias=False)
    ids = torch.zeros((2, 2, 3), dtype=torch.long)
    reference = embedding(ids).flatten(start_dim=-2).clone()
    # Include the bfloat16 -> float32 transition that previously failed.
    model.bfloat16().to(dtype=dtype)
    memory = model.embedding(ids).flatten(start_dim=-2)
    assert memory.dtype == model.projection.weight.dtype == dtype
    expected = F.linear(reference.to(dtype), model.projection.weight)
    torch.testing.assert_close(model.projection(memory), expected)
    assert all(t.weight.device.type == 'cpu' and not t.weight.requires_grad for t in embedding.tables)
    assert not list(embedding.parameters())
    requests = torch.tensor([[0, 0], [1, 0], [2, 0]])
    owner_rows = embedding._lookup_received_requests(requests, [1, 1, 1])
    assert owner_rows.dtype == dtype
    for i, table in enumerate(embedding.tables):
        assert embedding.sharded_state_dict()[f'tables.{i}.weight'].dtype == dtype


def test_device_conversion_does_not_move_or_copy_host_shards(embedding):
    pointers = [t.weight.data_ptr() for t in embedding.tables]
    embedding.to(device='meta')
    assert all(t.weight.device.type == 'cpu' for t in embedding.tables)
    assert pointers == [t.weight.data_ptr() for t in embedding.tables]
