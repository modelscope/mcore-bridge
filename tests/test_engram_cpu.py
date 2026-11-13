# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from mcore_bridge.utils import use_ngram_embedding_cpu_offload

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
    for i, table in enumerate(module.tables):
        table.weight.data.copy_(torch.arange(table.weight.numel()).reshape_as(table.weight) / 16 + i)
    retain_tables_on_cpu(module)
    return module


@pytest.mark.parametrize('shared,legacy,expected', [
    (None, None, False),
    (None, '1', True),
    ('0', '1', False),
    ('1', '0', True),
])
def test_offload_switch_precedence(monkeypatch, shared, legacy, expected):
    for name, value in [('NGRAM_EMBEDDING_CPU_OFFLOAD', shared), ('PLE_CPU_OFFLOAD', legacy)]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert use_ngram_embedding_cpu_offload() is expected


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_lookup_dtype_and_projection_gradient(embedding, dtype):
    model = torch.nn.Module()
    model.embedding = embedding
    model.projection = torch.nn.Linear(12, 2, bias=False)
    ids = torch.arange(12).reshape(2, 2, 3) % 5
    reference = embedding(ids).flatten(start_dim=-2).clone()
    model.bfloat16().to(dtype=dtype)
    memory = model.embedding(ids).flatten(start_dim=-2)
    torch.testing.assert_close(memory, reference.to(dtype))
    torch.testing.assert_close(model.projection(memory), F.linear(reference.to(dtype), model.projection.weight))
    model.projection(memory).sum().backward()
    expected = reference.to(dtype).reshape(-1, 12).sum(0).expand(2, 12)
    torch.testing.assert_close(model.projection.weight.grad, expected)
    requests = torch.tensor([[2, 3], [0, 2], [2, 1], [0, 0], [2, 1]])
    expected_rows = torch.stack([embedding.tables[t].weight[r] for t, r in requests])
    torch.testing.assert_close(embedding._lookup_received_requests(requests, [2, 0, 3]), expected_rows)
    assert embedding(torch.zeros((1, 0, 3), dtype=torch.long)).shape == (1, 0, 3, 4)
    assert not list(embedding.parameters())
    assert all(t.weight.device.type == 'cpu' and not t.weight.requires_grad for t in embedding.tables)
    pointers = [t.weight.data_ptr() for t in embedding.tables]
    model.to(device='meta')
    assert pointers == [t.weight.data_ptr() for t in embedding.tables]


def test_state_dict_loading_keeps_tables_frozen_and_sharded(embedding):
    state = {k: v.clone().requires_grad_() for k, v in embedding.state_dict().items()}
    for table in embedding.tables:
        table.weight.zero_()
    embedding.load_state_dict(state, strict=True)
    for i, table in enumerate(embedding.tables):
        torch.testing.assert_close(table.weight, state[f'tables.{i}.weight'], rtol=0, atol=0)
        assert not table.weight.requires_grad and table.weight.grad_fn is None
        shard = embedding.sharded_state_dict()[f'tables.{i}.weight']
        assert shard.data.device.type == 'cpu'
        assert shard.global_shape == (embedding.table_sizes[i], 4) and shard.global_offset == (0, 0)
    state['tables.0.weight'] = torch.zeros(2, 4)
    with pytest.raises(RuntimeError, match='expected'):
        embedding.load_state_dict(state)


@pytest.mark.parametrize('dtype,scaled', [(torch.float8_e4m3fn, True), (torch.float8_e4m3fn, False),
                                          (torch.bfloat16, True)])
def test_quantized_state_rejected_even_when_not_strict(embedding, dtype, scaled):
    state = {k: v.clone() for k, v in embedding.state_dict().items()}
    before = embedding.tables[0].weight.clone()
    state['tables.0.weight'] = torch.ones_like(before).to(dtype)
    if scaled:
        state['tables.0.weight_scale_inv'] = torch.full((before.shape[0], 1), 4.)
    with pytest.raises(RuntimeError, match='requires dequantized floating-point weights'):
        embedding.load_state_dict(state, strict=False)
    torch.testing.assert_close(embedding.tables[0].weight, before, rtol=0, atol=0)
