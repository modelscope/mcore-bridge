# Copyright (c) ModelScope Contributors. All rights reserved.
"""PLE HF export and distributed checkpoint metadata with synthetic weights.

Process groups and collectives are simulated; these are not multi-GPU or
checkpoint save/load integration tests.
"""
import pytest
import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.tensor_parallel import VocabParallelEmbedding
from unittest.mock import Mock

from mcore_bridge.model.modules.ple import Qwen4ExpTextNGramEmbedding, Qwen4ExpTextPLELayer


@pytest.fixture
def tp_group(monkeypatch):
    group = Mock()
    group.rank.return_value = 0
    group.size.return_value = 1
    dp_group = Mock()
    dp_group.rank.return_value = 0
    dp_group.size.return_value = 1
    # Megatron's metadata helpers read group.rank()/size() only when initialized.
    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(mpu, 'get_tensor_model_parallel_group', lambda: group)
    monkeypatch.setattr(mpu, 'get_tensor_model_parallel_rank', lambda: group.rank())
    monkeypatch.setattr(mpu, 'get_tensor_model_parallel_world_size', lambda: group.size())
    monkeypatch.setattr(mpu, 'get_data_parallel_group', lambda **kwargs: dp_group)
    return group


class _Embedding(VocabParallelEmbedding):
    """Small local partition using the real embedding checkpoint method."""

    def __init__(self, weight, tp_group):
        torch.nn.Module.__init__(self)
        self.tp_group = tp_group
        self.weight = torch.nn.Parameter(weight.clone())
        self.num_embeddings_per_partition = weight.shape[0]


class _Table(Qwen4ExpTextNGramEmbedding):
    """Build only the state needed by export and checkpoint metadata methods."""

    def __init__(self, full_weight, tp_group, parts=3):
        torch.nn.Module.__init__(self)
        self.cpu_offload = False
        self.padded_vocab_size = full_weight.shape[0]
        self.split_ngram_parts = parts
        self.ngram_embedding = _Embedding(full_weight.chunk(tp_group.size())[tp_group.rank()], tp_group)
        self.register_buffer('layer_multipliers', torch.tensor([1, 3], dtype=torch.long))


class _PLELayer(Qwen4ExpTextPLELayer):
    """Exercise recursion into the table and a replicated convolution."""

    def __init__(self, table, dim):
        torch.nn.Module.__init__(self)
        self.ple_embedding = table
        self.conv1d = torch.nn.Conv1d(dim, dim, 3, groups=dim, bias=False)


@pytest.mark.parametrize('tp_size', [1, 2, 8])
@pytest.mark.parametrize('with_scale', [False, True])
def test_export_updated_table(monkeypatch, tp_group, tp_size, with_scale):
    """Combine actual contributions from every simulated rank into HF shards."""
    total, dim, parts = 32, 4, 3
    full_weight = (torch.arange(total * dim).reshape(total, dim) % 13 - 6).to(torch.bfloat16)
    expected = full_weight + 1
    tp_group.size.return_value = tp_size
    contributions = [[] for _ in range(parts)]
    shard_index = 0

    def all_reduce(tensor, group):
        nonlocal shard_index
        assert group is tp_group
        contributions[shard_index].append(tensor.clone())
        # Run rank 0 last, so its output uses contributions produced by all ranks.
        if group.rank() == 0:
            assert len(contributions[shard_index]) == tp_size
            tensor.copy_(torch.stack(contributions[shard_index]).sum(dim=0))
        shard_index += 1

    monkeypatch.setattr(dist, 'all_reduce', all_reduce)
    prefix = 'model.layers.3.'
    for rank in reversed(range(tp_size)):
        tp_group.rank.return_value = rank
        table = _Table(full_weight, tp_group, parts)
        if with_scale:
            table._ngram_weight_scale = torch.tensor(0.5)
        # Stand in for a training update; values remain exactly representable in FP8.
        with torch.no_grad():
            table.ngram_embedding.weight.add_(1)
        shard_index = 0
        exported = {}
        table.export_table_to_hf(exported, prefix=prefix)
        assert shard_index == (parts if tp_size > 1 else 0)
        if rank != 0:
            assert exported == {}

    weights = [exported[f'{prefix}ple.ple_embedding.ngram_embedding.shard_{i}.weight'] for i in range(parts)]
    assert len(exported) == parts + int(with_scale)
    assert [w.shape[0] for w in weights] == [11, 11, 10]
    assert all(w.dtype == (torch.float8_e4m3fn if with_scale else torch.bfloat16) for w in weights)
    restored = torch.cat([w.float() for w in weights])
    if with_scale:
        scale = exported[f'{prefix}{table._NGRAM_SCALE_KEY}']
        assert scale.shape == ()
        assert scale.item() == 0.5
        restored *= scale
    torch.testing.assert_close(restored, expected.float(), rtol=0, atol=0)


@pytest.mark.parametrize('tp_size', [8, 2])
def test_ple_sharded_state_dict_tp_layouts(tp_group, tp_size):
    """Both layouts describe the full table; buffers and convolution stay replicated."""
    total, dim = 32, 4
    full_weight = torch.arange(total * dim, dtype=torch.float32).reshape(total, dim)
    prefix = 'decoder.layers.ple.'
    tp_group.size.return_value = tp_size
    restored = torch.empty_like(full_weight)
    for rank in range(tp_size):
        tp_group.rank.return_value = rank
        layer = _PLELayer(_Table(full_weight, tp_group), dim)
        state = layer.sharded_state_dict(prefix, ((0, 1, 2), ))
        weight = state[f'{prefix}ple_embedding.ngram_embedding.weight']
        assert weight.global_shape == (2, total, dim)
        assert weight.local_shape == (total // tp_size, dim)
        assert weight.global_offset == (1, rank * (total // tp_size), 0)
        start = weight.global_offset[1]
        restored[start:start + weight.local_shape[0]] = weight.data
    torch.testing.assert_close(restored, full_weight, rtol=0, atol=0)
