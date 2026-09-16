# Copyright (c) ModelScope Contributors. All rights reserved.
"""DeepSeek-V4.1-Flash (text backbone) bridge for megatron-core.

Wires the V4.1 *language* model (composite HF ``model_type='deepseek_v41'`` with
text sub-config ``deepseek_v41_text``) into mcore-bridge. V4.1 reuses DeepSeek-V4's
DSv4 hybrid MLA + hyper-connection stack but swaps the sparse-attention core to
CSA2 (selected by ``config.dsv4_version == 'v4.1'``). Differences vs V4/CSA:

  * Compressor (``CSA2Compressor``): no absolute-position embedding (``ape``); the
    gate projection (``linear_wgate``) exists only on ratio-2 layers.
  * Indexer (``CSA2Indexer``): flattened -- kv-source (``owns_k``) layers own
    ``linear_wk`` + ``k_norm`` instead of a nested compressor.
  * mHC is single-pass (``mhc_single_pass=True``): there are no learned final
    ``hc_head_*`` params; per-layer ``hc_attn_*``/``hc_ffn_*`` are mapped by the
    base ``GPTBridge``.

The text integration also attaches the native trainable Engram modules and
loads their EP-local table rows directly from the official flat FP8 tensors.
Vision, MTP and DSpark are integrated separately.

The V4.1 loader deliberately selects megatron-core's native TransformerBlock,
whose forward owns the per-call CSA2State and SinglePassMHCState lifecycle. Other
mcore-bridge models keep the custom TransformerBlock path.
"""
import copy
import os
from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import transformer_engine
from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import VocabParallelEmbedding
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_block import TransformerBlock as McoreTransformerBlock
from torch import nn
from typing import Optional

from mcore_bridge.config import MLAModelConfig
from mcore_bridge.model.modules.dspark import DeepseekV41DSparkStack
from mcore_bridge.model.modules.engram import (
    adapt_deepseek_v41_layer_specs,
    allow_engram_inference,
    build_deepseek_v41_engram_config,
    has_native_engram,
)

from ..constant import ModelType
from ..mm_gpt_model import MultimodalGPTModel
from ..register import ModelMeta, register_model
from .deepseek_v4 import (
    _apply_mla_rope,
    DeepseekV4Bridge,
    DeepseekV4GPTModel,
    DeepseekV4Loader,
    DSv4HybridSelfAttention,
)

try:
    from megatron.core.transformer.experimental_attention_variant.csa2 import CSA2Compressor as McoreCSA2Compressor
    from megatron.core.transformer.experimental_attention_variant.csa2 import CSA2Indexer as McoreCSA2Indexer
except ImportError:
    McoreCSA2Compressor = object
    McoreCSA2Indexer = object


def _duplicated_linear_kwargs(config):
    return dict(
        config=config,
        init_method=config.init_method,
        bias=False,
        skip_bias_add=False,
        skip_weight_param_allocation=False,
        parallel_mode='duplicated',
    )


class CSA2Compressor(McoreCSA2Compressor):
    """CSA2 compressor keeping its bf16 projections out of fp8 under fp8_param.

    The V4.1 checkpoint stores ``compressor.wkv``/``compressor.wgate`` in bf16, so
    rebuild them with fp8 disabled to match (mirrors the V4 ``Compressor`` wrapper).
    """

    def __init__(self, config, submodules, *args, **kwargs):
        super().__init__(config, submodules, *args, **kwargs)
        if getattr(config, 'fp8_param', False):
            linear_kwargs = _duplicated_linear_kwargs(config)
            with transformer_engine.pytorch.fp8_model_init(enabled=False):
                self.linear_wkv = build_module(submodules.linear_wkv, config.hidden_size, config.v_head_dim,
                                               **linear_kwargs)
                if self.compress_ratio == 2:
                    self.linear_wgate = build_module(submodules.linear_wgate, config.hidden_size, config.v_head_dim,
                                                     **linear_kwargs)


class DeepseekV41DSparkCoreAttention(MegatronModule):
    """Parameter holder for DSpark's latent attention sink.

    The actual attention computation belongs to ``DeepseekV41DSparkAttention``;
    this module intentionally has no CSA compressor or indexer parameters.
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config=config)
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        if config.num_attention_heads % world_size:
            raise ValueError('DSpark attention heads must be divisible by tensor parallel size.')
        device = 'cpu' if config.use_cpu_initialization else torch.cuda.current_device()
        self.attn_sink = mark_keep_in_fp32(nn.Parameter(
            torch.zeros(config.num_attention_heads // world_size, dtype=torch.float32, device=device)))


class CSA2Indexer(McoreCSA2Indexer):
    """CSA2 indexer keeping its bf16 projections out of fp8 under fp8_param.

    ``linear_weights_proj`` and (on ``owns_k`` layers) ``linear_wk`` are bf16 in the
    V4.1 checkpoint; ``linear_wq_b`` stays fp8. Mirrors the V4 ``CSAIndexer`` wrapper.
    """

    def __init__(self, config, submodules, *args, **kwargs):
        super().__init__(config, submodules, *args, **kwargs)
        if getattr(config, 'fp8_param', False):
            linear_kwargs = _duplicated_linear_kwargs(config)
            with transformer_engine.pytorch.fp8_model_init(enabled=False):
                self.linear_weights_proj = build_module(submodules.linear_weights_proj, config.hidden_size,
                                                         self.n_heads, **linear_kwargs)
                if self.owns_k:
                    self.linear_wk = build_module(submodules.linear_wk, config.v_head_dim, self.head_dim,
                                                  **linear_kwargs)


class DeepseekV41DSparkAttention(DSv4HybridSelfAttention):
    """DSpark latent attention over a main-token ring window and draft block."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.window_size = config.csa_window_size
        self._dspark_window_kv_cache = None

    @staticmethod
    def _select_rotary(rotary_pos_emb):
        if isinstance(rotary_pos_emb, dict):
            return rotary_pos_emb['main']
        return rotary_pos_emb

    def _project_kv(self, hidden_states, rotary_pos_emb):
        kv, _ = self.linear_kv_proj(hidden_states)
        kv = self.kv_layernorm(kv)
        pos_dim = self.config.qk_pos_emb_head_dim
        kv_no_pe, kv_pos_emb = torch.split(kv, [kv.shape[-1] - pos_dim, pos_dim], dim=-1)
        kv_pos_emb = _apply_mla_rope(
            kv_pos_emb,
            rotary_pos_emb,
            config=self.config,
            cu_seqlens=None,
            cp_group=self.pg_collection.cp,
        )
        return torch.cat((kv_no_pe, kv_pos_emb), dim=-1).unsqueeze(-2).contiguous()

    def _project_query(self, hidden_states, rotary_pos_emb):
        query_compressed, _ = self.linear_q_down_proj(hidden_states)
        query_compressed = self.q_layernorm(query_compressed)
        query, _ = self.linear_q_up_proj(query_compressed)
        query = query.view(
            *query.shape[:-1],
            self.num_attention_heads_per_partition,
            self.q_head_dim,
        )
        pos_dim = self.config.qk_pos_emb_head_dim
        query_no_pe, query_pos_emb = torch.split(
            query, [query.shape[-1] - pos_dim, pos_dim], dim=-1)
        query_pos_emb = _apply_mla_rope(
            query_pos_emb,
            rotary_pos_emb,
            config=self.config,
            cu_seqlens=None,
            cp_group=self.pg_collection.cp,
        )
        return torch.cat((query_no_pe, query_pos_emb), dim=-1).contiguous()

    def _ensure_cache(self, slot_count, hidden_size, dtype, device):
        expected = (slot_count, self.window_size, hidden_size)
        cache = self._dspark_window_kv_cache
        if cache is None or cache.device != device or cache.dtype != dtype or cache.shape[-1] != hidden_size:
            cache = torch.zeros(expected, dtype=dtype, device=device)
        elif cache.shape[0] < slot_count:
            expanded = torch.zeros(expected, dtype=dtype, device=device)
            expanded[:cache.shape[0]].copy_(cache)
            cache = expanded
        self._dspark_window_kv_cache = cache
        return cache

    @staticmethod
    def _normalize_cache_inputs(main_kv, start_pos, cache_slots):
        batch_size = main_kv.shape[1]
        device = main_kv.device
        if cache_slots is None:
            cache_slots = torch.arange(batch_size, dtype=torch.long, device=device)
        else:
            cache_slots = cache_slots.to(device=device, dtype=torch.long)
        if cache_slots.shape != (batch_size,):
            raise ValueError(
                f'DSpark cache slots must be [b={batch_size}], got {tuple(cache_slots.shape)}.')
        start_positions = torch.as_tensor(start_pos, dtype=torch.long, device=device)
        if start_positions.ndim == 0:
            start_positions = start_positions.expand(batch_size)
        if start_positions.shape != (batch_size,):
            raise ValueError(
                f'DSpark start positions must be scalar or [b={batch_size}], got '
                f'{tuple(start_positions.shape)}.')
        return start_positions, cache_slots

    def _write_main_cache(self, main_kv, start_pos, cache_slots=None):
        main_kv = main_kv.squeeze(-2)
        start_positions, cache_slots = self._normalize_cache_inputs(
            main_kv, start_pos, cache_slots)
        cache = self._ensure_cache(
            int(cache_slots.max().item()) + 1,
            main_kv.shape[-1],
            main_kv.dtype,
            main_kv.device,
        )
        if main_kv.shape[0] > self.window_size:
            offset = main_kv.shape[0] - self.window_size
            main_kv = main_kv[offset:]
            start_positions = start_positions + offset
        sequence_offsets = torch.arange(main_kv.shape[0], device=main_kv.device)
        positions = (start_positions.unsqueeze(0) + sequence_offsets.unsqueeze(1)) % self.window_size
        slots = cache_slots.unsqueeze(0).expand_as(positions)
        cache[slots, positions] = main_kv.detach()
        active_cache = cache.index_select(0, cache_slots).transpose(0, 1).contiguous()
        valid_lengths = (start_positions + main_kv.shape[0]).clamp(max=self.window_size)
        return active_cache, valid_lengths

    def prefill_dspark(
        self,
        main_hidden,
        rotary_pos_emb,
        inference_context=None,
        start_pos=0,
        cache_slots=None,
    ):
        del inference_context
        rotary_pos_emb = self._select_rotary(rotary_pos_emb)
        if rotary_pos_emb is None:
            raise ValueError('DSpark prefill requires main-token rotary embeddings.')
        main_kv = self._project_kv(main_hidden, rotary_pos_emb)
        self._write_main_cache(main_kv, start_pos, cache_slots)

    def reset_dspark_cache(self):
        self._dspark_window_kv_cache = None

    def _latent_attention(self, query, key_value, valid_main_lengths=None):
        key_value = key_value.expand(-1, -1, query.shape[-2], -1)
        scores = torch.einsum('sbhd,tbhd->bhst', query.float(), key_value.float())
        scores.mul_(self.config.v_head_dim**-0.5)
        if valid_main_lengths is not None:
            main_width = self.window_size
            main_indices = torch.arange(main_width, device=scores.device)
            invalid_main = main_indices.unsqueeze(0) >= valid_main_lengths.unsqueeze(1)
            invalid = F.pad(invalid_main, (0, key_value.shape[0] - main_width), value=False)
            scores = scores.masked_fill(invalid[:, None, None, :], float('-inf'))
        sink = self.core_attention.attn_sink.view(1, -1, 1, 1)
        probabilities = torch.softmax(
            torch.cat((scores, sink.expand(scores.shape[:-1] + (1,))), dim=-1),
            dim=-1,
            dtype=torch.float32,
        )[..., :-1]
        return torch.einsum('bhst,tbhd->sbhd', probabilities.to(key_value.dtype), key_value)

    def _project_output(self, output):
        seq_len, batch_size = output.shape[:2]
        output = output.view(seq_len, batch_size, self.o_local_groups, -1)
        if self._o_group_proj_is_grouped_linear:
            output = output.permute(2, 0, 1, 3).contiguous().reshape(-1, output.shape[-1])
            output = self.linear_o_group_proj(output, [seq_len * batch_size] * self.o_local_groups)
            output = output.view(self.o_local_groups, seq_len, batch_size, -1)
            output = output.permute(1, 2, 0, 3).contiguous().reshape(seq_len, batch_size, -1)
        else:
            weight = self.linear_o_group_proj.view(self.o_local_groups, self.config.o_lora_rank, -1)
            output = torch.einsum('...gd,grd->...gr', output, weight).flatten(-2)
        return self.linear_proj(output)

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        dspark_main_hidden=None,
        dspark_main_rotary_pos_emb=None,
        dspark_cache_slots=None,
    ):
        del attention_mask, key_value_states, rotary_pos_cos, rotary_pos_sin
        del rotary_pos_cos_sin, attention_bias, packed_seq_params, position_ids, inference_params
        if dspark_main_hidden is None:
            raise ValueError('DSpark attention requires target-layer main hidden states.')
        if sequence_len_offset is None:
            raise ValueError('DSpark attention requires the main-token sequence offset.')
        start_pos = sequence_len_offset
        draft_rotary = self._select_rotary(rotary_pos_emb)
        main_rotary = self._select_rotary(dspark_main_rotary_pos_emb)
        if draft_rotary is None or main_rotary is None:
            raise ValueError('DSpark attention requires main and draft rotary embeddings.')

        main_kv = self._project_kv(dspark_main_hidden, main_rotary)
        main_window, valid_main = self._write_main_cache(
            main_kv, start_pos, dspark_cache_slots)
        main_window = main_window.unsqueeze(-2)
        query = self._project_query(hidden_states, draft_rotary)
        draft_kv = self._project_kv(hidden_states, draft_rotary)
        key_value = torch.cat((main_window, draft_kv), dim=0)
        output = self._latent_attention(query, key_value, valid_main)

        pos_dim = self.config.qk_pos_emb_head_dim
        output_no_pe, output_pos_emb = torch.split(
            output, [output.shape[-1] - pos_dim, pos_dim], dim=-1)
        output_pos_emb = _apply_mla_rope(
            output_pos_emb,
            draft_rotary,
            config=self.config,
            cu_seqlens=None,
            cp_group=self.pg_collection.cp,
            inverse=True,
        )
        return self._project_output(torch.cat((output_no_pe, output_pos_emb), dim=-1))


def _vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float, device: torch.device):
    """Build the official row-major 2D RoPE table for one image."""
    inv_freq = 1.0 / (theta**(torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    hpos = torch.arange(n_h, device=device).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w, device=device).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack((hpos, wpos), dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def _apply_vision_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(dtype)


class DeepseekV41VisionRMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class DeepseekV41PatchEmbed(nn.Module):

    def __init__(self, patch_size: int, hidden_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(3 * patch_size**2, hidden_size)

    def forward(self, patches: torch.Tensor):
        if patches.ndim not in (2, 4):
            raise ValueError(
                'DeepSeek-V4.1 pixel_values must be [num_patches, 3, patch, patch] '
                f'or flattened [num_patches, 3 * patch ** 2], got {tuple(patches.shape)}.')
        return self.proj(patches.flatten(1))


class DeepseekV41VisionAttention(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(f'vision hidden_size {hidden_size} must be divisible by num_heads {num_heads}.')
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.wqkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.wo = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        num_tokens = x.shape[0]
        q, k, v = (
            tensor.view(num_tokens, self.num_heads, self.head_dim)
            for tensor in self.wqkv(x).chunk(3, dim=-1)
        )
        q = _apply_vision_rotary(q, cos, sin)
        k = _apply_vision_rotary(k, cos, sin)
        output = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        return self.wo(output.transpose(0, 1).reshape(num_tokens, -1))


class DeepseekV41VisionMLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor):
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class DeepseekV41VisionBlock(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.norm1 = DeepseekV41VisionRMSNorm(hidden_size)
        self.attn = DeepseekV41VisionAttention(hidden_size, num_heads)
        self.norm2 = DeepseekV41VisionRMSNorm(hidden_size)
        self.mlp = DeepseekV41VisionMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class DeepseekV41VisionTransformer(nn.Module):

    def __init__(self, vision_config):
        super().__init__()
        hidden_size = vision_config.hidden_size
        num_heads = vision_config.num_attention_heads
        self.rope_dim = hidden_size // num_heads // 2
        self.rope_theta = vision_config.rope_theta
        self.patch_embed = DeepseekV41PatchEmbed(vision_config.patch_size, hidden_size)
        self.blocks = nn.ModuleList([
            DeepseekV41VisionBlock(hidden_size, num_heads, vision_config.intermediate_size)
            for _ in range(vision_config.num_hidden_layers)
        ])
        self.norm = DeepseekV41VisionRMSNorm(hidden_size)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int):
        if patches.shape[0] != n_h * n_w:
            raise ValueError(
                f'Image grid {n_h}x{n_w} requires {n_h * n_w} patches, got {patches.shape[0]}.')
        x = self.patch_embed(patches)
        cos, sin = _vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta, x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class DeepseekV41Aligner(nn.Module):

    def __init__(self, vision_config, text_hidden_size: int):
        super().__init__()
        self.downsample_ratio = vision_config.downsample_ratio
        in_dim = vision_config.hidden_size * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, text_hidden_size)
        self.w2 = nn.Linear(text_hidden_size, text_hidden_size)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int):
        ratio = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % ratio, 0, -n_h % ratio))
        x = F.unfold(x.unsqueeze(0), ratio, stride=ratio).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))


class DeepseekV41Vision(nn.Module):
    """Trainable V4.1 vision tower, aligner and image-span embedding merger."""

    # DeepseekV4Bridge normalizes root-level official keys under an internal
    # ``model.`` prefix before conversion and strips it again on export.
    module_mapping = {'model.vision': 'vision', 'model.aligner': 'aligner'}
    _vision_tower = ['vision']
    _aligner = ['aligner']
    test_mm_type = 'image'

    IMAGE_START = 0
    IMAGE = 1
    IMAGE_NEW_LINE = 2
    IMAGE_END = 3

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_token_id = config.hf_config.image_token_id
        if config.language_model_only:
            self.vision = None
            self.aligner = None
            self.register_parameter('image_start', None)
            self.register_parameter('image_end', None)
            self.register_parameter('image_newline', None)
            return
        vision_config = config.hf_config.vision_config
        self.vision = DeepseekV41VisionTransformer(vision_config)
        self.aligner = DeepseekV41Aligner(vision_config, config.hidden_size)
        self.image_start = nn.Parameter(torch.empty(config.hidden_size))
        self.image_end = nn.Parameter(torch.empty(config.hidden_size))
        self.image_newline = nn.Parameter(torch.empty(config.hidden_size))
        target_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self.to(device=target_device, dtype=config.params_dtype)
        # Official RMSNorm scales remain fp32 even when the remaining ViT is bf16.
        for module in self.modules():
            if isinstance(module, DeepseekV41VisionRMSNorm):
                module.weight.data = module.weight.data.float()

    def get_inputs_embeds_language_model(self, inputs_embeds, **kwargs):
        return inputs_embeds

    @staticmethod
    def _grid_hw(image_grid_thw: torch.Tensor):
        if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] not in (2, 3):
            raise ValueError(f'image_grid_thw must have shape [num_images, 2 or 3], got {tuple(image_grid_thw.shape)}.')
        if image_grid_thw.shape[1] == 3:
            if not torch.all(image_grid_thw[:, 0] == 1):
                raise ValueError('DeepSeek-V4.1 supports still images only; every temporal grid size must be 1.')
            image_grid_thw = image_grid_thw[:, 1:]
        return image_grid_thw.to(dtype=torch.long, device='cpu')

    def encode_images(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
        grids = self._grid_hw(image_grid_thw)
        outputs = []
        patch_offset = 0
        for n_h, n_w in grids.tolist():
            patch_count = n_h * n_w
            patches = pixel_values[patch_offset:patch_offset + patch_count]
            outputs.append(self.aligner(self.vision(patches, n_h, n_w), n_h, n_w))
            patch_offset += patch_count
        if patch_offset != pixel_values.shape[0]:
            raise ValueError(f'Image grids describe {patch_offset} patches, but pixel_values has {pixel_values.shape[0]}.')
        if not outputs:
            return pixel_values.new_empty((0, self.image_start.numel()))
        return torch.cat(outputs, dim=0)

    def _zero_parameter_dependency(self, inputs_embeds: torch.Tensor):
        zero = inputs_embeds.new_zeros(())
        for parameter in self.parameters():
            zero = zero + parameter.reshape(-1)[0].to(inputs_embeds.dtype) * 0
        return inputs_embeds + zero

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        pixel_values = kwargs.get('pixel_values')
        image_grid_thw = kwargs.get('image_grid_thw')
        token_types = kwargs.get('image_token_types', kwargs.get('token_types'))
        if pixel_values is None:
            return self._zero_parameter_dependency(inputs_embeds)
        if image_grid_thw is None or token_types is None:
            raise ValueError('DeepSeek-V4.1 vision requires image_grid_thw and image_token_types/token_types.')
        if token_types.shape != kwargs['input_ids'].shape:
            raise ValueError(
                f'image token types shape {tuple(token_types.shape)} must match input_ids '
                f'{tuple(kwargs["input_ids"].shape)}.')

        image_mask = token_types >= 0
        input_image_mask = kwargs['input_ids'] == self.image_token_id
        if not torch.equal(image_mask.to(input_image_mask.device), input_image_mask):
            raise ValueError('Every DeepSeek-V4.1 image-span position must carry image_token_id, and no text position may use it.')
        image_features = self.encode_images(pixel_values.to(self.vision.patch_embed.proj.weight), image_grid_thw)
        flat_types = token_types[image_mask].to(device=inputs_embeds.device)
        if int((flat_types == self.IMAGE).sum()) != image_features.shape[0]:
            raise ValueError(
                f'Image spans contain {int((flat_types == self.IMAGE).sum())} patch slots, '
                f'but the aligner produced {image_features.shape[0]} rows.')
        replacements = inputs_embeds.new_empty((flat_types.numel(), inputs_embeds.shape[-1]))
        replacements[flat_types == self.IMAGE_START] = self.image_start.to(inputs_embeds.dtype)
        replacements[flat_types == self.IMAGE_END] = self.image_end.to(inputs_embeds.dtype)
        replacements[flat_types == self.IMAGE_NEW_LINE] = self.image_newline.to(inputs_embeds.dtype)
        replacements[flat_types == self.IMAGE] = image_features.to(inputs_embeds.dtype)
        if not torch.all((flat_types >= self.IMAGE_START) & (flat_types <= self.IMAGE_END)):
            raise ValueError('image_token_types values must be TEXT=-1 or one of START=0, IMAGE=1, NEW_LINE=2, END=3.')
        expanded_mask = image_mask.to(inputs_embeds.device).unsqueeze(-1).expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(expanded_mask, replacements)


class DeepseekV41GPTModel(DeepseekV4GPTModel):
    """V4.1 language model with opt-in DSpark target-layer capture.

    DSpark consumes the attention inputs of its target layers. The official target
    IDs are zero-based; Megatron layer numbers are one-based. Capturing is opt-in
    so regular training does not retain three large activation graphs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._capture_dspark_hidden = False
        self._dspark_hidden_states = {}
        self._dspark_hook_handles = []
        target_ids = tuple(self.config.dspark_target_layer_ids or ())
        for layer in self.decoder.layers:
            layer_id = layer.layer_number - 1
            if layer_id not in target_ids:
                continue
            self._dspark_hook_handles.append(
                layer.register_forward_pre_hook(self._make_dspark_capture_hook(layer_id), with_kwargs=True))

    @staticmethod
    def _contract_dspark_target_hidden(hidden_states: torch.Tensor, num_streams: int):
        if hidden_states.ndim != 3:
            raise ValueError(f'DSpark target hidden states must be [s, b, n*h], got {tuple(hidden_states.shape)}.')
        if hidden_states.shape[-1] % num_streams:
            raise ValueError(
                f'DSpark target hidden width {hidden_states.shape[-1]} is not divisible by {num_streams} streams.')
        return hidden_states.unflatten(-1, (num_streams, -1)).mean(dim=-2)

    def _make_dspark_capture_hook(self, layer_id):

        def _capture(_module, args, kwargs):
            if not self._capture_dspark_hidden:
                return
            hidden_states = kwargs.get('hidden_states')
            if hidden_states is None and args:
                hidden_states = args[0]
            if hidden_states is None:
                raise ValueError(f'Could not capture the attention input for DSpark target layer {layer_id}.')
            self._dspark_hidden_states[layer_id] = self._contract_dspark_target_hidden(
                hidden_states, self.config.num_residual_streams)

        return _capture

    @contextmanager
    def capture_dspark_hidden_states(self):
        if not self.config.dspark_target_layer_ids:
            raise ValueError('DSpark target-layer capture requested, but DSpark is not configured.')
        self._dspark_hidden_states = {}
        self._capture_dspark_hidden = True
        try:
            yield
        finally:
            self._capture_dspark_hidden = False

    def get_dspark_main_hidden(self, clear: bool = True):
        target_ids = tuple(self.config.dspark_target_layer_ids or ())
        missing = [layer_id for layer_id in target_ids if layer_id not in self._dspark_hidden_states]
        if missing:
            raise RuntimeError(
                f'DSpark target layers {missing} were not captured on this pipeline rank. '
                'The DSpark draft stack must be colocated with all target layers.')
        result = torch.cat([self._dspark_hidden_states[layer_id] for layer_id in target_ids], dim=-1)
        if clear:
            self._dspark_hidden_states = {}
        return result

    def forward(self, *args, **kwargs):
        input_ids = kwargs.get('input_ids', args[0] if args else None)
        inference_context = kwargs.get('inference_context') or kwargs.get('inference_params')
        if inference_context is None and len(args) > 5:
            inference_context = args[5]
        extra_block_kwargs = kwargs.get('extra_block_kwargs')
        if extra_block_kwargs is None and len(args) > 7:
            extra_block_kwargs = args[7]

        with allow_engram_inference(self.config, input_ids, extra_block_kwargs) as block_kwargs:
            if len(args) > 7:
                args = (*args[:7], block_kwargs, *args[8:])
            else:
                kwargs['extra_block_kwargs'] = block_kwargs
            capture_dspark = (
                hasattr(self, 'dspark')
                and not self.training
                and inference_context is not None
                and inference_context.is_dynamic_batching()
                and inference_context.num_speculative_tokens > 0
            )
            if not capture_dspark:
                return super().forward(*args, **kwargs)
            if inference_context.using_cuda_graph_this_step():
                raise RuntimeError('DSpark speculative decoding does not support CUDA graph replay yet.')
            with self.capture_dspark_hidden_states():
                return super().forward(*args, **kwargs)

    def _dspark_rotary_for_positions(self, position_ids: torch.Tensor):
        if self.position_embedding_type != 'rope' or self.rotary_pos_emb is None:
            raise RuntimeError('DSpark requires RoPE position embeddings.')
        position_ids = position_ids.to(dtype=torch.long)
        max_position = int(position_ids.max().item()) + 1
        rotary_table = self.rotary_pos_emb(max_position)
        if isinstance(rotary_table, dict):
            rotary_table = rotary_table['main']
        selected = rotary_table.index_select(0, position_ids.reshape(-1))
        # Drop the table's singleton batch dimension. The requested position tensor
        # supplies the batch axes for packed main tokens or the parallel draft block.
        return selected.reshape(*position_ids.shape, *rotary_table.shape[2:])

    def _dspark_word_embeddings(self):
        """Return the token embedding DSpark uses to embed its draft seed.

        On a single stage (or a standard-MTP stage) the base input embedding is
        colocated and reused. On a PP>1 last stage with untied embeddings the base
        model has no ``embedding``; a dedicated replicated DSpark embedding built
        in ``build_model`` and loaded from ``model.embed_tokens.weight`` is used.
        """
        if hasattr(self, 'embedding'):
            return self.embedding.word_embeddings
        dspark_embedding = getattr(self, 'dspark_word_embeddings', None)
        if dspark_embedding is None:
            raise RuntimeError(
                'DSpark requires an input word embedding on its pipeline stage, but neither '
                'the base embedding nor a dedicated DSpark embedding is present.')
        return dspark_embedding

    def forward_dspark(
        self,
        main_hidden,
        input_ids,
        *,
        start_pos,
        rotary_pos_emb,
        main_rotary_pos_emb,
        inference_context=None,
        temperature=0.0,
        sample_fn=None,
        cache_slots=None,
        prefill_only=None,
    ):
        if not hasattr(self, 'dspark'):
            raise RuntimeError('DSpark is not available on this pipeline stage.')
        if not hasattr(self, 'output_layer'):
            raise RuntimeError('DSpark requires the output head on its pipeline stage.')
        return self.dspark(
            main_hidden,
            input_ids,
            self._dspark_word_embeddings(),
            self.output_layer,
            start_pos=start_pos,
            rotary_pos_emb=rotary_pos_emb,
            main_rotary_pos_emb=main_rotary_pos_emb,
            inference_context=inference_context,
            temperature=temperature,
            sample_fn=sample_fn,
            cache_slots=cache_slots,
            prefill_only=prefill_only,
        )

    def compute_dspark_speculative_tokens(
        self,
        next_token_ids,
        accepted_token_counts,
        last_accepted_seq_indices,
        num_speculative_tokens,
        inference_context,
        sample_fn,
    ):
        """Commit verified target states and produce one parallel DSpark draft block."""
        if inference_context.using_cuda_graph_this_step():
            raise RuntimeError('DSpark speculative decoding does not support CUDA graph replay yet.')
        if num_speculative_tokens > self.config.dspark_block_size:
            raise ValueError(
                f'Requested {num_speculative_tokens} speculative tokens, but DSpark block size is '
                f'{self.config.dspark_block_size}.')

        main_hidden = self.get_dspark_main_hidden()
        if self.config.sequence_parallel and parallel_state.get_tensor_model_parallel_world_size() > 1:
            main_hidden = gather_from_sequence_parallel_region(main_hidden, group=self.tp_group)
        if main_hidden.ndim != 3 or main_hidden.shape[1] != 1:
            raise RuntimeError(
                'DSpark dynamic inference expects packed target states [tokens, 1, targets*h], '
                f'got {tuple(main_hidden.shape)}.')

        active_count = inference_context.total_request_count - inference_context.paused_request_count
        active_slice = slice(inference_context.paused_request_count, inference_context.total_request_count)
        query_lengths = inference_context.request_query_lengths[active_slice].to(dtype=torch.long)
        active_token_count = int(query_lengths.sum().item())
        if main_hidden.shape[0] != active_token_count:
            raise RuntimeError(
                f'DSpark captured {main_hidden.shape[0]} target rows for {active_token_count} active tokens.')

        device = main_hidden.device
        request_ids = inference_context.request_ids[active_slice].to(device=device, dtype=torch.long)
        live_request_ids = inference_context.request_ids[:inference_context.total_request_count].to(
            device=device, dtype=torch.long)
        cache_slots = self.dspark.resolve_cache_slots(request_ids, live_request_ids)
        token_positions = inference_context.token_to_position_in_request[:active_token_count].to(
            device=device, dtype=torch.long)
        accepted_token_counts = accepted_token_counts[:active_count].to(device='cpu', dtype=torch.long)

        offset = 0
        for request_index, query_length in enumerate(query_lengths.tolist()):
            if request_index < inference_context.num_decode_requests:
                accepted_length = min(query_length, int(accepted_token_counts[request_index].item()) + 1)
            else:
                accepted_length = query_length
            if accepted_length:
                token_slice = slice(offset, offset + accepted_length)
                positions = token_positions[token_slice]
                if positions.numel() > 1 and not torch.all(positions[1:] == positions[:-1] + 1):
                    raise RuntimeError('DSpark cache updates require contiguous per-request token positions.')
                self.dspark.update_main_cache(
                    main_hidden[token_slice],
                    self._dspark_rotary_for_positions(positions),
                    start_pos=positions[0],
                    cache_slots=cache_slots[request_index:request_index + 1],
                    inference_context=inference_context,
                )
            offset += query_length

        last_indices = last_accepted_seq_indices[:active_count].to(device=device, dtype=torch.long)
        last_hidden = main_hidden.index_select(0, last_indices).transpose(0, 1).contiguous()
        main_positions = token_positions.index_select(0, last_indices)
        draft_positions = main_positions.unsqueeze(0) + 1 + torch.arange(
            self.config.dspark_block_size, device=device).unsqueeze(1)
        output_ids, _, _ = self.forward_dspark(
            last_hidden,
            next_token_ids[:active_count],
            start_pos=main_positions,
            rotary_pos_emb=self._dspark_rotary_for_positions(draft_positions),
            main_rotary_pos_emb=self._dspark_rotary_for_positions(main_positions),
            inference_context=inference_context,
            sample_fn=sample_fn,
            cache_slots=cache_slots,
            prefill_only=False,
        )
        return output_ids[:, 1:num_speculative_tokens + 1].transpose(0, 1).contiguous()


class DeepseekV41MultimodalGPTModel(MultimodalGPTModel):
    language_model_cls = DeepseekV41GPTModel

    @property
    def vocab_size(self):
        return self.language_model.vocab_size

    def forward_with_dspark_hidden(self, *args, **kwargs):
        with self.language_model.capture_dspark_hidden_states():
            output = self.forward(*args, **kwargs)
        return output, self.language_model.get_dspark_main_hidden()

    def forward_dspark(self, *args, **kwargs):
        return self.language_model.forward_dspark(*args, **kwargs)

    def compute_dspark_speculative_tokens(self, *args, **kwargs):
        return self.language_model.compute_dspark_speculative_tokens(*args, **kwargs)


class DeepseekV41Loader(DeepseekV4Loader):
    model_cls = DeepseekV41MultimodalGPTModel
    # Native V4.1 forward owns CSA2State + SinglePassMHCState. Using it only for
    # this loader avoids changing the custom bridge block used by V4/DSpark/MTP.
    transformer_block = McoreTransformerBlock

    def _engram_placement_layer_ids(self, hf_layer_ids):
        """Map 0-based HF Engram layer IDs to 1-based ``TransformerLayer`` placement numbers.

        On the GPT stack HF layer ``e`` is one ``TransformerLayer`` numbered ``e + 1``. The
        HybridStack loader overrides this because there each HF layer becomes two hybrid layers.
        """
        return tuple(layer_id + 1 for layer_id in hf_layer_ids)

    def _get_engram_config(self):
        hf_layer_ids = tuple(self.config.engram_layer_ids or ())
        if not hf_layer_ids:
            return None
        if not has_native_engram():
            raise RuntimeError(
                'DeepSeek-V4.1 Engram requires NVIDIA Megatron-LM Engram support. '
                'The PR #7224 text-backbone baseline intentionally does not provide it; '
                'install the official Engram extension or disable Engram explicitly.')
        required = (
            'engram_num_embeddings', 'engram_max_ngram_size', 'engram_vocab_size',
            'engram_n_heads', 'engram_head_dim', 'engram_pad_token_id',
        )
        missing = [name for name in required if getattr(self.config, name, None) is None]
        if missing:
            raise ValueError(f'DeepSeek-V4.1 Engram config is missing required fields: {missing}.')

        tokenizer_map = self.config.engram_tokenizer_map
        if tokenizer_map is None:
            model_dir = getattr(self.config.hf_config, 'name_or_path', '')
            candidate = os.path.join(model_dir, 'engram_tokenizer_map.json') if model_dir else ''
            if candidate and os.path.isfile(candidate):
                tokenizer_map = candidate
        if not tokenizer_map:
            raise ValueError(
                'DeepSeek-V4.1 Engram requires engram_tokenizer_map. Generate it with '
                'Megatron-LM/tools/engram/generate_tokenizer_map.py using the HF 0-based '
                f'layer IDs {list(hf_layer_ids)}.'
            )

        max_ngram_order = self.config.engram_max_ngram_size
        image_token_id = getattr(self.config.hf_config, 'image_token_id', None)
        engram_config = build_deepseek_v41_engram_config(
            global_vocab_sizes=(self.config.engram_vocab_size,) * (max_ngram_order - 1),
            # TransformerLayer numbers are 1-based, while the official checkpoint and
            # tokenizer artifact use the original 0-based HF layer IDs.
            placement_layer_ids=self._engram_placement_layer_ids(hf_layer_ids),
            hash_layer_ids=hf_layer_ids,
            max_ngram_order=max_ngram_order,
            num_hash_heads=self.config.engram_n_heads,
            memory_dim=self.config.engram_n_heads * self.config.engram_head_dim,
            kernel_size=1,
            hash_seed=0,
            boundary_token_id=self.config.engram_pad_token_id,
            tokenizer_map_path=tokenizer_map,
            excluded_token_ids=(() if image_token_id is None else (image_token_id,)),
        )
        actual_rows = tuple(sum(engram_config.table_sizes(layer_id)) for layer_id in engram_config.layer_ids)
        expected_rows = tuple(self.config.engram_num_embeddings)
        if actual_rows != expected_rows:
            raise ValueError(
                'DeepSeek-V4.1 Engram table layout does not match engram_num_embeddings: '
                f'computed {actual_rows}, checkpoint declares {expected_rows}.'
            )
        if (self.config.engram_compressed_vocab_size is not None
                and engram_config.tokenizer_remap.max().item() + 1 != self.config.engram_compressed_vocab_size):
            raise ValueError(
                'DeepSeek-V4.1 compressed tokenizer vocabulary mismatch: artifact has '
                f'{engram_config.tokenizer_remap.max().item() + 1}, config declares '
                f'{self.config.engram_compressed_vocab_size}.'
            )
        engram_config.validate_startup(
            self.config, expected_tokenizer_vocab_size=self.config.padded_vocab_size)
        return engram_config

    def get_dspark_layer_spec(self):
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            _get_backend_spec_provider,
            get_transformer_layer_with_experimental_attention_variant_spec,
        )

        dspark_config = copy.copy(self.config)
        dspark_config.hf_config = getattr(self.config.hf_config, 'text_config', self.config.hf_config)
        dspark_config.num_layers = self.config.dspark_num_layers
        dspark_config.num_moe_experts = self.config.dspark_num_experts
        dspark_config.moe_router_topk = self.config.dspark_router_topk
        dspark_config.moe_layer_freq = [1] * self.config.dspark_num_layers
        dspark_config.first_pipeline_num_layers = None
        dspark_config.last_pipeline_num_layers = None
        dspark_config.num_layers_in_first_pipeline_stage = None
        dspark_config.num_layers_in_last_pipeline_stage = None
        dspark_config.sequence_parallel = False
        dspark_config.csa_compress_ratios = [0] * self.config.dspark_num_layers
        dspark_config.csa2_kv_source_layers = []
        dspark_config.csa2_index_source_layers = []
        dspark_config.csa2_candidate_source_layer = None
        backend = _get_backend_spec_provider(config=dspark_config)
        layer_specs = get_transformer_layer_with_experimental_attention_variant_spec(
            config=dspark_config, backend=backend)
        for layer_spec in layer_specs:
            attention_spec = layer_spec.submodules.self_attention
            attention_spec.module = DeepseekV41DSparkAttention
            attention_spec.submodules.core_attention.module = DeepseekV41DSparkCoreAttention
        # DSpark specs bypass ModelLoader.build_model, so apply the same router
        # override the main layers get: swap the vendored McoreTopKRouter for the
        # custom TopKRouter, otherwise the draft MoE has no ``expert_bias_vl`` and
        # loading the checkpoint's ``mtp.*.ffn.gate.bias_vl`` asserts.
        self._replace_router(SimpleNamespace(layer_specs=layer_specs))
        return dspark_config, layer_specs

    def build_model(self, pre_process=True, post_process=True, vp_stage: Optional[int] = None):
        model = super().build_model(pre_process, post_process, vp_stage)
        if not self.config.dspark_num_layers or not post_process:
            return model
        language_model = model.language_model
        dspark_config, dspark_layer_specs = self.get_dspark_layer_spec()
        layers = [
            build_module(
                layer_spec,
                config=dspark_config,
                layer_number=index + 1,
                pg_collection=language_model.pg_collection,
            )
            for index, layer_spec in enumerate(dspark_layer_specs)
        ]
        language_model.dspark = DeepseekV41DSparkStack(dspark_config, layers)
        self._set_linear_is_expert(language_model.dspark)
        # DSpark embeds its draft seed with the base input embedding. On a PP>1 last
        # stage with untied embeddings the base model has no ``embedding`` here, so
        # build a dedicated replicated DSpark embedding; the bridge loads it from
        # ``model.embed_tokens.weight`` (same source as the first-stage embedding).
        if not hasattr(language_model, 'embedding'):
            language_model.dspark_word_embeddings = VocabParallelEmbedding(
                language_model.vocab_size,
                language_model.config.hidden_size,
                init_method=language_model.config.init_method,
                config=language_model.config,
                tp_group=language_model.pg_collection.tp,
            )
        return model

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import \
            get_transformer_block_with_experimental_attention_variant_spec
        transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(self.config, vp_stage)
        for layer_spec in transformer_layer_spec.layer_specs:
            layer_spec.submodules.self_attention.module = DSv4HybridSelfAttention
            core_attention_submodules = layer_spec.submodules.self_attention.submodules.core_attention.submodules
            if getattr(core_attention_submodules, 'compressor', None) is not None:
                core_attention_submodules.compressor.module = CSA2Compressor
            if getattr(core_attention_submodules, 'indexer', None) is not None:
                # CSA2 indexer is flat (no nested compressor).
                core_attention_submodules.indexer.module = CSA2Indexer
        engram_config = self._get_engram_config()
        if engram_config is not None:
            transformer_layer_spec = adapt_deepseek_v41_layer_specs(transformer_layer_spec, engram_config)
        return transformer_layer_spec


class DeepseekV41Bridge(DeepseekV4Bridge):
    _ENGRAM_LOAD_CHUNK_ROWS = 65536
    additional_dim0_keys = DeepseekV4Bridge.additional_dim0_keys | {'embed', 'head'}
    additional_dim1_keys = DeepseekV4Bridge.additional_dim1_keys | {'main_proj'}

    def _convert_pre_process(self, mg_model, hf_state_dict, hf_prefix: str, to_mcore: bool):
        result = super()._convert_pre_process(mg_model, hf_state_dict, hf_prefix, to_mcore)
        target = hf_state_dict if to_mcore else result
        for name in ('image_start', 'image_end', 'image_newline'):
            self._set_state_dict(mg_model, f'visual.{name}', target, f'model.{name}', to_mcore)
        return result

    def _set_router(self, mg_mlp, hf_state_dict, to_mcore, **kwargs):
        super()._set_router(mg_mlp, hf_state_dict, to_mcore, **kwargs)
        if self.config.moe_router_enable_vl_bias:
            self._set_state_dict(mg_mlp, 'router.expert_bias_vl', hf_state_dict, 'gate.bias_vl', to_mcore)

    @staticmethod
    def _get_layer_engram(mg_layer):
        if mg_layer is None:
            return None
        engram = getattr(mg_layer, 'engram', None)
        if engram is None:
            engram = getattr(getattr(mg_layer, 'inner_layer', None), 'engram', None)
        return engram

    @staticmethod
    def _load_lazy_slice(lazy_tensor, row_start, row_end):
        if hasattr(lazy_tensor, 'load_slice'):
            return lazy_tensor.load_slice(slice(row_start, row_end))
        return lazy_tensor.load()[row_start:row_end]

    @staticmethod
    def _dequantize_engram_rows(weight, scale):
        if scale is None:
            return weight
        if weight.ndim != 2 or scale.ndim != 2 or weight.shape[0] != scale.shape[0]:
            raise ValueError(
                f'Invalid Engram FP8 weight/scale shapes: {tuple(weight.shape)} and {tuple(scale.shape)}.')
        if weight.shape[1] % scale.shape[1] != 0:
            raise ValueError(
                f'Engram weight width {weight.shape[1]} is not divisible by scale width {scale.shape[1]}.')
        block_size = weight.shape[1] // scale.shape[1]
        return (weight.float().unflatten(-1, (-1, block_size)) * scale.float().unsqueeze(-1)).flatten(-2)

    def _load_engram_embedding(self, engram, hf_state_dict):
        weight = hf_state_dict['engram.embed.weight']
        scale = hf_state_dict.get('engram.embed.weight_scale_inv')
        flat_offset = 0
        for table in engram.embedding.tables:
            table_offset = flat_offset
            flat_offset += table.global_num_embeddings
            for local_start in range(0, table.local_num_embeddings, self._ENGRAM_LOAD_CHUNK_ROWS):
                local_end = min(local_start + self._ENGRAM_LOAD_CHUNK_ROWS, table.local_num_embeddings)
                source_start = table_offset + table.row_start + local_start
                source_end = table_offset + table.row_start + local_end
                rows = self._load_lazy_slice(weight, source_start, source_end)
                row_scales = None if scale is None else self._load_lazy_slice(scale, source_start, source_end)
                rows = self._dequantize_engram_rows(rows, row_scales)
                table.weight.data[local_start:local_end].copy_(
                    rows.to(device=table.weight.device, dtype=table.weight.dtype))
        hf_layer_id = self._engram_hf_layer_id(engram)
        expected_rows = self.config.engram_num_embeddings[
            self.config.engram_layer_ids.index(hf_layer_id)]
        if flat_offset != expected_rows:
            raise ValueError(
                f'Engram layer {hf_layer_id} expected {expected_rows} flat rows, '
                f'but its prime tables contain {flat_offset}.')

    def _engram_hf_layer_id(self, engram):
        """Recover the 0-based HF layer ID from a built Engram's 1-based ``layer_number``.

        On the GPT stack ``layer_number == hf_id + 1``. The HybridStack bridge overrides this
        because its Engram sits on the doubled-space attention layer ``2 * hf_id + 1``.
        """
        return engram.layer_number - 1

    def _set_layer_engram(self, mg_layer, hf_state_dict, to_mcore):
        engram = self._get_layer_engram(mg_layer)
        if to_mcore:
            if engram is None:
                return
            self._load_engram_embedding(engram, hf_state_dict)
            wkv = hf_state_dict['engram.wkv.weight'].load()
            wkv_scale = hf_state_dict.get('engram.wkv.weight_scale_inv')
            if wkv_scale is not None:
                wkv_scale = wkv_scale.load()
            wkv = self._dequantize_engram_rows(wkv, wkv_scale)
            key_rows = engram.num_streams * engram.hidden_size
            if tuple(wkv.shape) != (key_rows + engram.hidden_size, engram.engram_config.total_memory_dim):
                raise ValueError(f'Unexpected DeepSeek-V4.1 Engram wkv shape: {tuple(wkv.shape)}.')
            engram.key_projection.weight.data.copy_(
                wkv[:key_rows].to(engram.key_projection.weight))
            engram.value_projection.weight.data.copy_(
                wkv[key_rows:].to(engram.value_projection.weight))
            engram.query_norm.weight.data.copy_(
                hf_state_dict['engram.q_weight'].load().reshape(-1).to(engram.query_norm.weight))
            engram.key_norm.weight.data.copy_(
                hf_state_dict['engram.k_weight'].load().reshape(-1).to(engram.key_norm.weight))
        elif not self._peft_format:
            if getattr(self, '_skip_unsupported_export', False):
                # On-policy RL weight sync: Engram tables are frozen and already resident in the
                # rollout engine from the base checkpoint, so skip re-exporting them (a full 183 GiB
                # resync per step is infeasible) instead of raising.
                return
            if engram is None:
                return
            # --- export embedding tables to a single flat tensor (bf16, no FP8) ---
            all_rows = []
            for table in engram.embedding.tables:
                all_rows.append(table.weight.data.cpu())
            hf_state_dict['engram.embed.weight'] = torch.cat(all_rows, dim=0)
            # --- export key+value projections as combined wkv ---
            key_w = engram.key_projection.weight.data.cpu()    # [stream_width, total_memory_dim]
            val_w = engram.value_projection.weight.data.cpu()   # [hidden, total_memory_dim]
            hf_state_dict['engram.wkv.weight'] = torch.cat([key_w, val_w], dim=0)
            # --- export norm weights as q_weight / k_weight ---
            num_streams = engram.num_streams
            hf_state_dict['engram.q_weight'] = engram.query_norm.weight.data.cpu().reshape(num_streams, -1)
            hf_state_dict['engram.k_weight'] = engram.key_norm.weight.data.cpu().reshape(num_streams, -1)

    def _set_layer_state(self, mg_layer, hf_state_dict, hf_prefix: str, layer_idx: int, to_mcore: bool):
        layer_prefix = f'{hf_prefix}{layer_idx}.'
        local_state = self._remove_prefix(hf_state_dict, layer_prefix) if to_mcore else {}
        result = super()._set_layer_state(mg_layer, hf_state_dict, hf_prefix, layer_idx, to_mcore)
        if layer_idx in (self.config.engram_layer_ids or []):
            self._set_layer_engram(mg_layer, local_state, to_mcore)
        if not to_mcore and local_state:
            result.update(self._add_prefix(local_state, layer_prefix))
        return result

    def _set_dspark_layer_state(self, mg_layer, hf_state_dict, layer_idx, to_mcore):
        stage_prefix = f'{self.hf_mtp_prefix}.{layer_idx}.'
        local_state = self._remove_prefix(hf_state_dict, stage_prefix) if to_mcore else {}
        local_state.update(self._set_layer_attn(mg_layer, local_state, layer_idx, to_mcore))
        local_state.update(self._set_layer_mlp(mg_layer, local_state, layer_idx, to_mcore, is_mtp=True))
        self._set_hyper_connection(mg_layer, local_state, layer_idx, to_mcore)
        if to_mcore:
            return {}
        return self._add_prefix(local_state, stage_prefix)

    def _set_dspark_endpoints(self, dspark, hf_state_dict, to_mcore):
        first_prefix = f'{self.hf_mtp_prefix}.0.'
        last_prefix = f'{self.hf_mtp_prefix}.{self.config.dspark_num_layers - 1}.'
        first_state = self._remove_prefix(hf_state_dict, first_prefix) if to_mcore else {}
        last_state = self._remove_prefix(hf_state_dict, last_prefix) if to_mcore else {}
        self._set_state_dict(dspark, 'input.main_proj.weight', first_state, 'main_proj.weight', to_mcore)
        self._set_state_dict(dspark, 'input.main_norm.weight', first_state, 'main_norm.weight', to_mcore)
        self._set_state_dict(dspark, 'output.norm.weight', last_state, 'norm.weight', to_mcore)
        self._set_state_dict(
            dspark, 'output.markov_head.embed.weight', last_state, 'markov_head.embed.weight', to_mcore)
        self._set_state_dict(
            dspark, 'output.markov_head.head.weight', last_state, 'markov_head.head.weight', to_mcore)
        self._set_state_dict(
            dspark, 'output.confidence_head.proj.weight', last_state,
            'confidence_head.proj.weight', to_mcore)
        if to_mcore:
            return {}
        result = self._add_prefix(first_state, first_prefix)
        result.update(self._add_prefix(last_state, last_prefix))
        return result

    def _load_dspark_word_embeddings(self, embedding, hf_state_dict):
        """Load the dedicated DSpark input embedding from ``model.embed_tokens.weight``.

        Mirrors the base embedding load: pad the HF rows up to ``padded_vocab_size``
        (already a multiple of TP) and take this rank's vocab-parallel shard.
        """
        weight = hf_state_dict[self.hf_embed_key].load()
        padded = self.config.padded_vocab_size
        if weight.shape[0] < padded:
            weight = F.pad(weight, (0, 0, 0, padded - weight.shape[0]))
        if self.tp_size > 1:
            weight = weight.chunk(self.tp_size, dim=0)[self.tp_rank]
        embedding.weight.data.copy_(weight.to(device=embedding.weight.device, dtype=embedding.weight.dtype))

    def _convert_additional_layers(self, mg_model, hf_state_dict, hf_prefix, to_mcore, is_pp_last_stage):
        if not self.config.dspark_num_layers or (to_mcore and not is_pp_last_stage):
            return
        language_model = mg_model.language_model if self.is_multimodal else mg_model
        dspark = getattr(language_model, 'dspark', None)
        if dspark is None:
            raise RuntimeError('DSpark weights require the draft stack on the final pipeline stage.')

        # On a PP>1 last stage with untied embeddings, DSpark owns a dedicated input
        # embedding (see build_model). Load it from the same HF source as the base
        # first-stage embedding. On export the first stage already emits this tensor,
        # so the redundant DSpark copy is not written back.
        if to_mcore and getattr(language_model, 'dspark_word_embeddings', None) is not None:
            self._load_dspark_word_embeddings(language_model.dspark_word_embeddings, hf_state_dict)
            yield

        original_num_experts = self.config.num_moe_experts
        self.config.num_moe_experts = self.config.dspark_num_experts
        try:
            for layer_idx, layer in enumerate(dspark.layers):
                result = self._set_dspark_layer_state(layer, hf_state_dict, layer_idx, to_mcore)
                if to_mcore:
                    yield
                else:
                    result = self._convert_hf_state_dict(result, to_mcore)
                    yield from self._add_prefix(result, hf_prefix).items()
            result = self._set_dspark_endpoints(dspark, hf_state_dict, to_mcore)
            if to_mcore:
                yield
            else:
                result = self._convert_hf_state_dict(result, to_mcore)
                yield from self._add_prefix(result, hf_prefix).items()
        finally:
            self.config.num_moe_experts = original_num_experts

    def _set_mla_attn_state(self, mg_attn, hf_state_dict, hf_prefix, layer_idx, to_mcore):
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        # --- shared MLA projections (identical to V4) ---
        self._set_state_dict(mg_attn, 'linear_proj.weight', hf_state_dict, 'wo_b.weight', to_mcore)
        if self.config.fp8_param:
            self._set_o_group_proj_grouped(mg_attn, hf_state_dict, to_mcore)
        else:
            self._set_state_dict(mg_attn, 'linear_o_group_proj', hf_state_dict, 'wo_a.weight', to_mcore)
        self._set_state_dict(mg_attn, 'linear_q_down_proj.weight', hf_state_dict, 'wq_a.weight', to_mcore)
        self._set_state_dict(mg_attn, 'linear_q_up_proj.weight', hf_state_dict, 'wq_b.weight', to_mcore)
        self._set_state_dict(mg_attn, 'linear_kv_proj.weight', hf_state_dict, 'wkv.weight', to_mcore)
        self._set_state_dict(mg_attn, 'core_attention.attn_sink', hf_state_dict, 'attn_sink', to_mcore)
        if self.config.qk_layernorm:
            self._set_state_dict(mg_attn, 'q_layernorm.weight', hf_state_dict, 'q_norm.weight', to_mcore)
            self._set_state_dict(mg_attn, 'kv_layernorm.weight', hf_state_dict, 'kv_norm.weight', to_mcore)
        # --- CSA2 compressor / indexer (no `ape`; indexer owns wk/k_norm on owns_k) ---
        core_attn = None if mg_attn is None else mg_attn.core_attention
        compressor = None if core_attn is None else getattr(core_attn, 'compressor', None)
        indexer = None if core_attn is None else getattr(core_attn, 'indexer', None)
        has_compressor = self._reduce_tensor_pp_group(compressor is not None, to_mcore)
        has_indexer = self._reduce_tensor_pp_group(indexer is not None, to_mcore)
        # ratio-2 compressor layers additionally own a gate projection.
        has_wgate = self._reduce_tensor_pp_group(
            compressor is not None and getattr(compressor, 'linear_wgate', None) is not None, to_mcore)
        # kv-source (owns_k) indexer layers additionally own wk + k_norm.
        owns_k = self._reduce_tensor_pp_group(
            indexer is not None and getattr(indexer, 'linear_wk', None) is not None, to_mcore)
        if has_compressor:
            self._set_state_dict(mg_attn, 'core_attention.compressor.linear_wkv.weight', hf_state_dict,
                                 'compressor.wkv.weight', to_mcore)
            self._set_state_dict(mg_attn, 'core_attention.compressor.norm.weight', hf_state_dict,
                                 'compressor.norm.weight', to_mcore)
            if has_wgate:
                self._set_state_dict(mg_attn, 'core_attention.compressor.linear_wgate.weight', hf_state_dict,
                                     'compressor.wgate.weight', to_mcore)
        if has_indexer:
            self._set_state_dict(mg_attn, 'core_attention.indexer.linear_wq_b.weight', hf_state_dict,
                                 'indexer.wq_b.weight', to_mcore)
            self._set_state_dict(mg_attn, 'core_attention.indexer.linear_weights_proj.weight', hf_state_dict,
                                 'indexer.weights_proj.weight', to_mcore)
            if owns_k:
                self._set_state_dict(mg_attn, 'core_attention.indexer.linear_wk.weight', hf_state_dict,
                                     'indexer.wk.weight', to_mcore)
                self._set_state_dict(mg_attn, 'core_attention.indexer.k_norm.weight', hf_state_dict,
                                     'indexer.k_norm.weight', to_mcore)
        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _set_final_layernorm(self, lm_model, hf_state_dict, to_mcore):
        # V4.1 single-pass mHC has no learned hc_head_*; skip the V4 hc_head mapping
        # and only handle the plain final layernorm (base GPTBridge behaviour).
        super(DeepseekV4Bridge, self)._set_final_layernorm(lm_model, hf_state_dict, to_mcore)


register_model(
    ModelMeta(
        ModelType.deepseek_v41,
        ['deepseek_v41'],
        bridge_cls=DeepseekV41Bridge,
        visual_cls=DeepseekV41Vision,
        loader=DeepseekV41Loader,
        config_cls=MLAModelConfig,
    ))
