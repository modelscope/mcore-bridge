# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from typing import Optional, Sequence, Tuple, Union


class Fp8Dequantizer:

    def __init__(self, block_size: Tuple[int, int] = (None, None)):
        # Set to None to enable automatic selection.
        self.block_size = block_size

    def convert(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(quantized, torch.Tensor) or not isinstance(scales, torch.Tensor):
            raise TypeError('Fp8Dequantize expects tensors as inputs.')
        if quantized.dtype == torch.uint8:
            quantized = quantized.view(torch.float8_e4m3fn)
        quantized_fp32 = quantized.to(torch.float32)
        rows, cols = quantized_fp32.shape[-2:]
        scale_rows, scale_cols = scales.shape[-2:]
        block_size = self.block_size
        block_m, block_n = block_size
        if block_m is None:
            block_m = rows // scale_rows
        if block_n is None:
            block_n = cols // scale_cols
        needs_padding = rows % block_m != 0 or cols % block_n != 0

        input_tensor = quantized_fp32
        if needs_padding:
            pad_rows = (block_m - rows % block_m) % block_m
            pad_cols = (block_n - cols % block_n) % block_n
            input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_cols, 0, pad_rows))

        p_rows, p_cols = input_tensor.shape[-2:]

        reshaped = input_tensor.reshape(-1, p_rows // block_m, block_m, p_cols // block_n, block_n)
        expanded_scales = scales.to(torch.float32).reshape(-1, p_rows // block_m, p_cols // block_n)
        expanded_scales = expanded_scales.unsqueeze(-1).unsqueeze(2)

        dequantized = reshaped * expanded_scales
        dequantized = dequantized.reshape(input_tensor.shape)

        if needs_padding:
            dequantized = dequantized[..., :rows, :cols].contiguous()

        return dequantized


class MxFp4Dequantizer:

    def convert(
        self,
        blocks: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        from transformers.integrations import convert_moe_packed_tensors
        return convert_moe_packed_tensors(blocks, scales)


_FP4_E2M1_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def fp4_to_fp8(packed: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(_FP4_E2M1_LUT, dtype=torch.float32, device=packed.device)
    u8 = packed.contiguous().view(torch.uint8)
    low = (u8 & 0x0F).long()
    high = ((u8 >> 4) & 0x0F).long()

    low_f32 = lut[low]
    high_f32 = lut[high]

    unpacked = torch.stack([low_f32, high_f32], dim=-1)
    unpacked = unpacked.reshape(*packed.shape[:-1], 2 * packed.shape[-1])

    return unpacked.to(torch.float8_e4m3fn)


class PackedDequantizer:
    """Dequantize INT4/INT8 weights packed into int32 (compressed-tensors `pack_quantized` format).

    Mirrors ``compressed_tensors.compressors.pack_quantized.PackedDequantizer.decompress``
    but exposes a simple ``convert(...)`` API consistent with the other dequantizers in this module.

    Quantization parameters (num_bits, symmetric, strategy) are extracted from
    ``quantization_config`` at init time (i.e. ``hf_config.quantization_config``).
    """

    # Strategies that store the zero-point in a packed int32 layout.
    _PACK_ZP_STRATEGIES = ('group', 'channel')

    def __init__(self, quantization_config: dict):
        # Extract settings from the first (and usually only) config_groups entry.
        config_groups = quantization_config.get('config_groups', {})
        if config_groups:
            group_cfg = next(iter(config_groups.values()))
            weights_cfg = group_cfg.get('weights', {})
        else:
            weights_cfg = {}

        self.num_bits: int = weights_cfg.get('num_bits', 4)
        self.symmetric: bool = weights_cfg.get('symmetric', True)
        self.strategy: str = weights_cfg.get('strategy', 'group')

    def convert(
        self,
        packed: torch.Tensor,
        scale: torch.Tensor,
        original_shape: Union[torch.Size, torch.Tensor, Sequence[int]],
        zero_point: Optional[torch.Tensor] = None,
        g_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Unpack ``weight_packed`` and dequantize it back to a float weight tensor.

        :param packed: int32 packed weight tensor (``weight_packed``).
        :param scale: per-channel / per-group scale tensor (``weight_scale``).
        :param original_shape: original (unpacked) weight shape (``weight_shape``).
        :param zero_point: optional zero-point. For asymmetric GROUP/CHANNEL strategies it is
            still packed in int32 and will be unpacked here; for symmetric quantization it is
            ignored.
        :param g_idx: optional group index mapping (``weight_g_idx``).
        :return: dequantized float weight tensor with shape ``original_shape``.
        """
        from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
        from compressed_tensors.quantization.lifecycle.forward import dequantize

        if isinstance(original_shape, torch.Tensor):
            original_shape = tuple(int(x) for x in original_shape.tolist())
        else:
            original_shape = tuple(int(x) for x in original_shape)

        num_bits = self.num_bits
        symmetric = self.symmetric
        strategy = self.strategy

        # Unpack zero_point before dequantization if it was stored in packed int32 form.
        if (not symmetric) and strategy in self._PACK_ZP_STRATEGIES:
            assert zero_point is not None, 'Asymmetric quant requires zero-point values'
            original_zp_shape = (*original_shape[:-1], scale.shape[-1])
            zero_point = unpack_from_int32(zero_point, num_bits, original_zp_shape, packed_dim=0)

        unpacked = unpack_from_int32(packed, num_bits, original_shape)
        weight = dequantize(
            x_q=unpacked,
            scale=scale,
            zero_point=zero_point,
            g_idx=g_idx,
        )
        return weight
