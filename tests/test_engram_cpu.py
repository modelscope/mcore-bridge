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
