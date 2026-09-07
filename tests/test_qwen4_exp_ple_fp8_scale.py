"""Verify the PLE ngram embedding FP8 <-> bf16 `weight_scale` handling.

FP8 checkpoints store the PLE ngram embedding table as F8_E4M3 shards plus a
single scalar `ple.ple_embedding.ngram_embedding.weight_scale` (true value =
weight * scale). This only matters when training starts from an FP8-format
checkpoint; bf16 checkpoints have no such key.

No model is downloaded: the checkpoint and the Megatron-side embedding are
synthetic tensors, exercised through `_set_ple_ngram_embedding` on a single
gloo rank.
"""
import os

os.environ.setdefault('RANK', '0')
os.environ.setdefault('LOCAL_RANK', '0')
os.environ.setdefault('WORLD_SIZE', '1')
os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '29517')

from types import SimpleNamespace  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from megatron.core import mpu  # noqa: E402

from mcore_bridge.model.gpts.qwen4_exp import Qwen4ExpBridge  # noqa: E402

SCALE_KEY = Qwen4ExpBridge._PLE_NGRAM_SCALE_KEY


class _LazyTensor:
    """Stand-in for the lazy checkpoint tensors: only `.load()` is used."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def load(self) -> torch.Tensor:
        return self.tensor


def _init_single_rank():
    if not dist.is_initialized():
        dist.init_process_group('gloo')
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(1)


def _shard_keys(parts: int):
    return [f'ple.ple_embedding.ngram_embedding.shard_{i}.weight' for i in range(parts)]


def _make_fake_ple(total: int, dim: int, parts: int):
    ngram_embedding = SimpleNamespace(
        weight=torch.zeros(total, dim, dtype=torch.bfloat16),
        num_embeddings=total,
        num_embeddings_per_partition=total)
    return SimpleNamespace(
        ple_embedding=SimpleNamespace(ngram_embedding=ngram_embedding, split_ngram_parts=parts, head_dim=dim))


def _make_fake_bridge():
    return SimpleNamespace(
        pp_size=1,
        tp_group=dist.group.WORLD,
        pp_group=dist.group.WORLD,
        _target_device=None,
        config=SimpleNamespace(params_dtype=torch.bfloat16),
        _PLE_NGRAM_SCALE_KEY=SCALE_KEY)


def test_ple_ngram_fp8_scale_roundtrip():
    _init_single_rank()
    torch.manual_seed(0)
    total, dim, parts = 16, 8, 4
    rows_per_shard = total // parts
    scale = torch.tensor(0.05, dtype=torch.float32)
    fp8_table = torch.randn(total, dim).to(torch.float8_e4m3fn)
    hf_state_dict = {
        key: _LazyTensor(fp8_table[i * rows_per_shard:(i + 1) * rows_per_shard].clone())
        for i, key in enumerate(_shard_keys(parts))
    }
    hf_state_dict[SCALE_KEY] = _LazyTensor(scale.clone())

    # to_mcore: fp8 shards must be dequantized with the scalar weight_scale.
    ple = _make_fake_ple(total, dim, parts)
    bridge = _make_fake_bridge()
    Qwen4ExpBridge._set_ple_ngram_embedding(bridge, ple, hf_state_dict, True, 0)
    expected = (fp8_table.float() * scale).to(torch.bfloat16)
    assert torch.equal(ple.ple_embedding.ngram_embedding.weight.data, expected)

    # to_hf: symmetric re-quantization, the scale key is written back.
    exported = {}
    Qwen4ExpBridge._set_ple_ngram_embedding(bridge, ple, exported, False, 0)
    assert torch.equal(exported[SCALE_KEY], scale)
    # bf16(fp8 * scale) / scale stays within half an FP8 ulp of the original
    # grid point, so re-quantization must recover the exact FP8 values.
    for i, key in enumerate(_shard_keys(parts)):
        shard = exported[key]
        assert shard.dtype == torch.float8_e4m3fn
        assert torch.equal(shard, fp8_table[i * rows_per_shard:(i + 1) * rows_per_shard])


def test_ple_ngram_bf16_checkpoint_loads_as_is():
    _init_single_rank()
    torch.manual_seed(0)
    total, dim, parts = 8, 4, 2
    table = torch.randn(total, dim, dtype=torch.bfloat16)
    hf_state_dict = {
        key: _LazyTensor(table[i * (total // parts):(i + 1) * (total // parts)].clone())
        for i, key in enumerate(_shard_keys(parts))
    }  # no weight_scale key: a plain bf16 checkpoint

    ple = _make_fake_ple(total, dim, parts)
    bridge = _make_fake_bridge()
    Qwen4ExpBridge._set_ple_ngram_embedding(bridge, ple, hf_state_dict, True, 0)
    assert torch.equal(ple.ple_embedding.ngram_embedding.weight.data, table)
    assert bridge._ple_ngram_weight_scale is None

    # Without a known scale the export keeps the values as-is (no fp8 cast).
    exported = {}
    Qwen4ExpBridge._set_ple_ngram_embedding(bridge, ple, exported, False, 0)
    assert SCALE_KEY not in exported
    for key in _shard_keys(parts):
        assert exported[key].dtype == torch.bfloat16
