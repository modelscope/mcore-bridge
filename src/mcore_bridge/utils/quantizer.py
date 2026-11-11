# Copyright (c) ModelScope Contributors. All rights reserved.
"""Re-quantization primitives (the inverse of ``dequantizer.py``) that restore the official DeepSeek-V4.1
checkpoint layout on export: fp8-e4m3 with 32x32-block e8m0 scales for linears, packed MXFP4 with per-32
scales for routed experts. Numerics mirror the DeepGEMM reference (``deep_gemm/utils/math.py``).
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
    """Quantize a 2-D weight ``[R, C]`` to fp8-e4m3 + e8m0 scales ``[ceil(R/block), ceil(C/block)]``."""
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
    """Quantize ``[R, C]`` to packed MXFP4 ``[R, C/2]`` (even element in the low nibble) + e8m0 ``[R, C/group]``."""
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


# Official keys carrying a ``.scale`` companion (``mtp.*`` reuses the suffixes). compressor/indexer.wk/gate/
# norms/hc_*/engram q,k_weight/markov_head stay bf16 in the source and are deliberately not matched.
_DSV41_FP8_BLOCK32_RE = re.compile(r'(?:^|\.)(?:attn\.(?:wq_a|wq_b|wo_a|wo_b|wkv|indexer\.wq_b)'
                                   r'|ffn\.shared_experts\.w[123]|engram\.wkv|main_proj)\.weight$')
_DSV41_MXFP4_RE = re.compile(r'(?:^|\.)ffn\.experts\.\d+\.w[123]\.weight$')


def quantize_dsv41_export_value(key: str, tensor) -> Optional[dict]:
    """Return ``{key: quantized, <base>.scale: e8m0}`` for an official quantized key, else ``None``."""
    if not isinstance(tensor, torch.Tensor) or tensor.dtype not in (torch.bfloat16, torch.float32):
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
