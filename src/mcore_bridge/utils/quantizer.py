# Copyright (c) ModelScope Contributors. All rights reserved.
"""Re-quantization primitives: the inverse direction of ``dequantizer.py``.

DeepSeek-V4.1 loads the official quantized source into BF16 training parameters (block32 FP8 dequantization,
MXFP4 experts via ``fp4_to_fp8``), but vLLM only serves the official quantized layout, so the HF export has to
restore it:

* dense/attention linears -> fp8-e4m3 weights + 32x32-block e8m0 scales (``<name>.weight`` + ``<name>.scale``);
* routed experts -> MXFP4 (two e2m1 nibbles per int8, even element in the low nibble) + one e8m0 scale per
  32-element group along the last (K) dimension.

Numeric semantics mirror the DeepGEMM reference implementation (``deep_gemm/utils/math.py``): amax clamped to
1e-4, ``scale = ceil_to_ue8m0(amax / 448)`` for fp8 and ``ceil_to_ue8m0(amax / 6)`` for fp4, e2m1
round-to-nearest via the midpoints {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}. Pure torch and device-agnostic.
"""
import re
import torch
from typing import Optional, Tuple

_UE8M0_EXP_MIN = 1
_UE8M0_EXP_MAX = 254
_FP8_E4M3_MAX = 448.0
_FP4_E2M1_MAX = 6.0
_AMAX_FLOOR = 1e-4


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Round positive fp32 values up to the nearest power of two (biased exponent clamped to [1, 254])."""
    bits = x.abs().float().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exponent.clamp(_UE8M0_EXP_MIN, _UE8M0_EXP_MAX) << 23).view(torch.float32)


def _to_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Reinterpret power-of-two fp32 scales as e8m0fnu bytes (the official ``F8_E8M0`` safetensors dtype)."""
    exponent = scale.view(torch.int32) >> 23
    return exponent.to(torch.uint8).view(torch.float8_e8m0fnu)


def per_block_cast_to_fp8(weight: torch.Tensor, block: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D weight to fp8-e4m3 with ``block x block`` e8m0 scales.

    Shapes follow the official checkpoint: weight ``[R, C]`` -> scale ``[ceil(R/block), ceil(C/block)]``;
    partial edge blocks are quantized against their zero-padded amax (DeepGEMM semantics).
    """
    if weight.ndim != 2:
        raise ValueError(f'block fp8 quantization expects a 2-D weight, got shape {tuple(weight.shape)}')
    rows, cols = weight.shape
    padded = weight.new_zeros(((rows + block - 1) // block * block, (cols + block - 1) // block * block))
    padded[:rows, :cols] = weight
    blocks = padded.view(-1, block, padded.shape[1] // block, block)
    amax = blocks.abs().float().amax(dim=(1, 3), keepdim=True).clamp(_AMAX_FLOOR)
    scale = ceil_to_ue8m0(amax / _FP8_E4M3_MAX)
    quantized = (blocks * (1.0 / scale)).to(torch.float8_e4m3fn)
    return (quantized.view_as(padded)[:rows, :cols].contiguous(),
            _to_e8m0(scale.reshape(blocks.shape[0], blocks.shape[2])))


def _quantize_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest e2m1 codes {0..7} with the sign in bit 3 (zero never carries a sign)."""
    magnitude = x.abs()
    code = torch.zeros_like(x, dtype=torch.uint8)
    for boundary in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        code += (magnitude > boundary).to(torch.uint8)
    sign = (x < 0) & (code != 0)
    code = code | (sign.to(torch.uint8) << 3)
    return code.view(torch.int8)


def per_token_cast_to_mxfp4(weight: torch.Tensor, group: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D weight to packed MXFP4: one e8m0 scale per ``group`` elements along the last dimension.

    Returns ``(packed, scale)`` with ``packed`` int8 ``[R, C/2]`` (even element in the low nibble) and ``scale``
    e8m0fnu ``[R, ceil(C/group)]``, matching the official ``experts.*.w{1,2,3}`` layout.
    """
    if weight.ndim != 2:
        raise ValueError(f'MXFP4 quantization expects a 2-D weight, got shape {tuple(weight.shape)}')
    rows, cols = weight.shape
    if cols % 2 != 0:
        raise ValueError(f'MXFP4 packs element pairs along the last dimension; got odd width {cols}')
    padded_cols = (cols + group - 1) // group * group
    padded = weight.new_zeros((rows, padded_cols))
    padded[:, :cols] = weight
    groups = padded.view(rows, -1, group)
    amax = groups.abs().float().amax(dim=2).clamp_min(_AMAX_FLOOR)
    scale = ceil_to_ue8m0(amax / _FP4_E2M1_MAX)
    codes = _quantize_to_fp4_e2m1(groups * (1.0 / scale.unsqueeze(2))).view(rows, padded_cols)
    pairs = codes.view(rows, padded_cols // 2, 2)
    packed = (pairs[:, :, 0] & 0x0F) | ((pairs[:, :, 1] & 0x0F) << 4)
    return packed[:, :cols // 2].contiguous(), _to_e8m0(scale)


# Official DeepSeek-V4.1 checkpoint keys carrying a ``.scale`` companion, normalized to regexes over the final
# HF export names (derived from model.safetensors.index.json; ``mtp.*`` reuses the same suffixes). Deliberately
# absent because the source keeps them in bf16: compressor.wkv/wgate, indexer.wk, gate, norms, hc_*, engram
# q_weight/k_weight, markov_head, vision/aligner, embed/head.
_DSV41_FP8_BLOCK32_RE = re.compile(
    r'(?:^|\.)(?:attn\.(?:wq_a|wq_b|wo_a|wo_b|wkv|indexer\.wq_b)'
    r'|ffn\.shared_experts\.w[123]|engram\.wkv|main_proj)\.weight$')
_DSV41_MXFP4_RE = re.compile(r'(?:^|\.)ffn\.experts\.\d+\.w[123]\.weight$')

_REQUANTIZABLE_DTYPES = (torch.bfloat16, torch.float32)


def quantize_dsv41_export_value(key: str, tensor) -> Optional[dict]:
    """Re-quantize one exported HF tensor to its official source format, keyed by the final checkpoint names.

    Returns ``{key: quantized, <base>.scale: e8m0}`` when ``key`` is an official quantized weight carrying a
    trainable-dtype payload, otherwise ``None`` (already-quantized fp8/int8 stores such as the Engram tables,
    bf16-only keys, lazy/None values on non-master ranks, and non-2-D tensors all pass through untouched).
    """
    if not isinstance(tensor, torch.Tensor) or tensor.dtype not in _REQUANTIZABLE_DTYPES:
        return None
    if tensor.ndim != 2:
        return None
    if _DSV41_MXFP4_RE.search(key):
        weight, scale = per_token_cast_to_mxfp4(tensor)
    elif _DSV41_FP8_BLOCK32_RE.search(key):
        weight, scale = per_block_cast_to_fp8(tensor)
    else:
        return None
    return {key: weight, key[:-len('.weight')] + '.scale': scale}
