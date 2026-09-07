"""Verify the PLE ngram embedding FP8 <-> bf16 `weight_scale` handling.

FP8 checkpoints store the PLE ngram embedding table as F8_E4M3 shards plus a
single scalar `ple.ple_embedding.ngram_embedding.weight_scale` (true value =
weight * scale). This only matters when training starts from an FP8-format
checkpoint; bf16 checkpoints have no such key.

Ported from upstream PR#183, which patched `Qwen4ExpBridge._set_ple_ngram_embedding`.
On this branch that method is gone: the shard walk lives in
`Qwen4ExpTextNGramEmbedding.fill_table_from_hf` / `export_table_to_hf`, so the
scale handling is exercised there instead. No model is downloaded -- the table is
synthetic, on a single gloo rank (the shard walk reads the TP rank).
"""
import os

os.environ.setdefault('RANK', '0')
os.environ.setdefault('LOCAL_RANK', '0')
os.environ.setdefault('WORLD_SIZE', '1')
os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '29519')

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from megatron.core import mpu  # noqa: E402

from mcore_bridge.model.modules.ple import Qwen4ExpTextNGramEmbedding  # noqa: E402

SCALE_KEY = Qwen4ExpTextNGramEmbedding._NGRAM_SCALE_KEY


@pytest.fixture(scope='module', autouse=True)
def _single_rank():
    """fill_table_from_hf reads the TP rank to find its slice of the table."""
    if not dist.is_initialized():
        dist.init_process_group('gloo')
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(1)
    yield


class _LazyTensor:
    """Stand-in for the lazy checkpoint tensors: only `.load()` is used."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def load(self) -> torch.Tensor:
        return self.tensor


class _FakeTable(Qwen4ExpTextNGramEmbedding):
    """Only the shard-walk state that fill_table_from_hf touches.

    Bypasses __init__ (which needs a full TransformerConfig and would allocate a
    VocabParallelEmbedding); the load path only reads padded_vocab_size,
    split_ngram_parts, cpu_offload and ngram_embedding.
    """

    def __init__(self, total: int, dim: int, parts: int):
        torch.nn.Module.__init__(self)
        self.padded_vocab_size = total
        self.split_ngram_parts = parts
        self.cpu_offload = False
        self.ngram_embedding = torch.nn.Embedding(total, dim, dtype=torch.bfloat16)
        torch.nn.init.zeros_(self.ngram_embedding.weight)
        # VocabParallelEmbedding attribute the TP-shard math reads.
        self.ngram_embedding.num_embeddings_per_partition = total


def _fp8_checkpoint(total, dim, parts, scale_value):
    """fp8 shards + scalar scale, plus the bf16 values they decode to."""
    shard_size = (total + parts - 1) // parts
    scale = torch.tensor(scale_value, dtype=torch.float32)
    state, expected = {SCALE_KEY: _LazyTensor(scale)}, torch.zeros(total, dim, dtype=torch.float32)
    for i in range(parts):
        cs, ce = i * shard_size, min((i + 1) * shard_size, total)
        raw = torch.arange(cs * dim, ce * dim, dtype=torch.float32).reshape(ce - cs, dim) % 13 - 6
        fp8 = raw.to(torch.float8_e4m3fn)
        state[f'ple.ple_embedding.ngram_embedding.shard_{i}.weight'] = _LazyTensor(fp8)
        expected[cs:ce] = fp8.to(torch.float32) * scale
    return state, expected


def test_fill_table_applies_weight_scale():
    total, dim, parts, scale = 16, 4, 2, 0.5
    table = _FakeTable(total, dim, parts)
    state, expected = _fp8_checkpoint(total, dim, parts, scale)

    table.fill_table_from_hf(state)

    torch.testing.assert_close(
        table.ngram_embedding.weight.data.float(), expected.to(torch.bfloat16).float(), rtol=0, atol=0)
    # Stashed for the export path to re-quantize with.
    assert float(table._ngram_weight_scale) == scale


def test_fill_table_without_scale_loads_raw():
    """bf16 checkpoints have no scale key: values must be taken as-is."""
    total, dim, parts = 16, 4, 2
    table = _FakeTable(total, dim, parts)
    state, expected = _fp8_checkpoint(total, dim, parts, 0.5)
    del state[SCALE_KEY]

    table.fill_table_from_hf(state)

    # Same shards, but decoded without the 0.5 factor.
    raw = expected / 0.5
    torch.testing.assert_close(
        table.ngram_embedding.weight.data.float(), raw.to(torch.bfloat16).float(), rtol=0, atol=0)
    assert table._ngram_weight_scale is None


def test_scale_is_a_lossless_roundtrip_for_representable_values():
    """weight * scale / scale must land back on the original fp8 codes."""
    scale = torch.tensor(0.5, dtype=torch.float32)
    fp8 = (torch.arange(-6, 7, dtype=torch.float32) % 13 - 6).to(torch.float8_e4m3fn)

    dequantized = fp8.to(torch.float32) * scale
    requantized = (dequantized / scale).to(torch.float8_e4m3fn)

    torch.testing.assert_close(requantized.to(torch.float32), fp8.to(torch.float32), rtol=0, atol=0)
