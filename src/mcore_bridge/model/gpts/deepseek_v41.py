# Copyright (c) ModelScope Contributors. All rights reserved.
"""DeepSeek-V4.1-Flash bridge for megatron-core.

Wires the V4.1 model (composite HF ``model_type='deepseek_v41'`` with text sub-config
``deepseek_v41_text``) into mcore-bridge. V4.1 reuses DeepSeek-V4's DSv4 hybrid MLA +
hyper-connection stack but swaps the sparse-attention core to CSA2 (selected by
``config.dsv4_version == 'v4.1'``). Differences vs V4/CSA:

  * Compressor (``CSA2Compressor``): no absolute-position embedding (``ape``); the
    gate projection (``linear_wgate``) exists only on ratio-2 layers.
  * Indexer (``CSA2Indexer``): flattened -- kv-source (``owns_k``) layers own
    ``linear_wk`` + ``k_norm`` instead of a nested compressor.
  * mHC is single-pass (``mhc_single_pass=True``): there are no learned final
    ``hc_head_*`` params; per-layer ``hc_attn_*``/``hc_ffn_*`` are mapped by the
    base ``GPTBridge``.

The backbone is megatron-core's native ``HybridModel``, which splits every HF layer into an
attention-only and an MLP-only hybrid layer. That split is what makes pipeline parallelism
work: a plain ``TransformerBlock`` cannot carry the hyper-connection payload across PP
stages. :func:`derive_hybrid_layer_config` re-expands the HF-space layer config into the
doubled hybrid space, and :class:`DeepseekV41Bridge` fans each HF layer out onto its two
hybrid layers.

The text integration also attaches the native trainable Engram modules -- loading their
EP-local table rows directly from the official flat FP8 tensors -- and the DSpark (``mtp.*``)
draft stack. Vision is wired in :class:`DeepseekV41MultimodalModel`.

Requires a megatron-core that ships ``megatron.core.models.hybrid``; without it V4.1 is not
registered at all (see ``_HYBRID_MODEL_AVAILABLE``).
"""
import copy
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformer_engine
from dataclasses import dataclass
from megatron.core import mpu, parallel_state
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.tensor_parallel.layers import VocabParallelEmbedding
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_block import TransformerBlock as McoreTransformerBlock
from torch import nn
from tqdm import tqdm
from types import SimpleNamespace
from typing import List, Optional, Sequence, Union

from mcore_bridge.config import MLAModelConfig
from mcore_bridge.model.modules.dspark import DeepseekV41DSparkStack
from mcore_bridge.model.modules.engram import (DeepseekV41Engram, DeepseekV41TransformerLayer,
                                               build_deepseek_v41_engram_config, has_native_engram)
from mcore_bridge.utils import is_master

from ..constant import ModelType
from ..mm_gpt_model import MultimodalGPTModel
from ..register import ModelMeta, register_model
from ..rope import get_rope_inv_freq
from .deepseek_v4 import DeepseekV4Bridge, DeepseekV4Loader, DSv4HybridSelfAttention, _apply_mla_rope

try:
    from megatron.core.transformer.experimental_attention_variant.csa2 import CSA2Compressor as McoreCSA2Compressor
    from megatron.core.transformer.experimental_attention_variant.csa2 import CSA2Indexer as McoreCSA2Indexer
except ImportError:
    McoreCSA2Compressor = object
    McoreCSA2Indexer = object

try:
    from megatron.core.models.hybrid.hybrid_block import HyperConnectionHybridLayer
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_dsv4_stack_spec

    from ..hybrid_model import HybridModel

    _HYBRID_MODEL_AVAILABLE = True
except ImportError as error:
    if not (error.name or '').startswith('megatron.core.models.hybrid'):
        raise
    HybridModel = hybrid_dsv4_stack_spec = HyperConnectionHybridLayer = None
    _HYBRID_MODEL_AVAILABLE = False


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
        self.attn_sink = mark_keep_in_fp32(
            nn.Parameter(torch.zeros(config.num_attention_heads // world_size, dtype=torch.float32, device=device)))


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
        query_no_pe, query_pos_emb = torch.split(query, [query.shape[-1] - pos_dim, pos_dim], dim=-1)
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
        if cache_slots.shape != (batch_size, ):
            raise ValueError(f'DSpark cache slots must be [b={batch_size}], got {tuple(cache_slots.shape)}.')
        start_positions = torch.as_tensor(start_pos, dtype=torch.long, device=device)
        if start_positions.ndim == 0:
            start_positions = start_positions.expand(batch_size)
        if start_positions.shape != (batch_size, ):
            raise ValueError(f'DSpark start positions must be scalar or [b={batch_size}], got '
                             f'{tuple(start_positions.shape)}.')
        return start_positions, cache_slots

    def _write_main_cache(self, main_kv, start_pos, cache_slots=None):
        main_kv = main_kv.squeeze(-2)
        start_positions, cache_slots = self._normalize_cache_inputs(main_kv, start_pos, cache_slots)
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
            torch.cat((scores, sink.expand(scores.shape[:-1] + (1, ))), dim=-1),
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
        main_window, valid_main = self._write_main_cache(main_kv, start_pos, dspark_cache_slots)
        main_window = main_window.unsqueeze(-2)
        query = self._project_query(hidden_states, draft_rotary)
        draft_kv = self._project_kv(hidden_states, draft_rotary)
        key_value = torch.cat((main_window, draft_kv), dim=0)
        output = self._latent_attention(query, key_value, valid_main)

        pos_dim = self.config.qk_pos_emb_head_dim
        output_no_pe, output_pos_emb = torch.split(output, [output.shape[-1] - pos_dim, pos_dim], dim=-1)
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
            raise ValueError('DeepSeek-V4.1 pixel_values must be [num_patches, 3, patch, patch] '
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
        q, k, v = (tensor.view(num_tokens, self.num_heads, self.head_dim) for tensor in self.wqkv(x).chunk(3, dim=-1))
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
            raise ValueError(f'Image grid {n_h}x{n_w} requires {n_h * n_w} patches, got {patches.shape[0]}.')
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
            raise ValueError(
                f'Image grids describe {patch_offset} patches, but pixel_values has {pixel_values.shape[0]}.')
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
            raise ValueError(f'image token types shape {tuple(token_types.shape)} must match input_ids '
                             f'{tuple(kwargs["input_ids"].shape)}.')

        image_mask = token_types >= 0
        input_image_mask = kwargs['input_ids'] == self.image_token_id
        if not torch.equal(image_mask.to(input_image_mask.device), input_image_mask):
            raise ValueError(
                'Every DeepSeek-V4.1 image-span position must carry image_token_id, and no text position may use it.')
        image_features = self.encode_images(pixel_values.to(self.vision.patch_embed.proj.weight), image_grid_thw)
        flat_types = token_types[image_mask].to(device=inputs_embeds.device)
        if int((flat_types == self.IMAGE).sum()) != image_features.shape[0]:
            raise ValueError(f'Image spans contain {int((flat_types == self.IMAGE).sum())} patch slots, '
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


@dataclass
class HybridLayerConfig:
    """Per-layer config re-expanded from HF layer space into hybrid (2x) layer space.

    All source-layer / candidate fields are 0-based indices in the doubled hybrid space
    (i.e. ``layer_number - 1`` as CSA2 reads them), where GPT layer ``i`` maps to hybrid
    attention layer ``2 * i``.
    """

    hybrid_layer_pattern: str
    num_layers: int
    csa_compress_ratios: List[int]
    csa2_kv_source_layers: List[int]
    csa2_index_source_layers: List[int]
    csa2_candidate_source_layer: Optional[int]


def _normalize_moe_layer_freq(moe_layer_freq: Union[int, Sequence[int], None], num_layers: int) -> List[int]:
    """Return a length-``num_layers`` 0/1 list marking MoE layers.

    Mirrors megatron-core's own interpretation (moe_logging.py:660-664): an ``int`` N means
    layer ``i`` is MoE iff ``i % N == 0``; a list is used verbatim. ``None`` (no experts)
    means every layer is dense.
    """
    if moe_layer_freq is None:
        return [0] * num_layers
    if isinstance(moe_layer_freq, int):
        return [1 if i % moe_layer_freq == 0 else 0 for i in range(num_layers)]
    freq = list(moe_layer_freq)
    if len(freq) != num_layers:
        raise ValueError(f'moe_layer_freq length {len(freq)} does not match num_layers {num_layers}.')
    return [1 if x else 0 for x in freq]


def derive_hybrid_layer_config(
    num_layers: int,
    csa_compress_ratios: Sequence[int],
    moe_layer_freq: Union[int, Sequence[int], None],
    csa2_kv_source_layers: Sequence[int] = (),
    csa2_index_source_layers: Sequence[int] = (),
    csa2_candidate_source_layer: Optional[int] = None,
) -> HybridLayerConfig:
    """Translate an HF-space V4.1 config into the doubled hybrid layer space.

    Each HF transformer layer ``i`` (0-based) becomes two hybrid layers:

    * hybrid index ``2*i`` -- attention-only, always the array-driven ``D`` symbol so it
      reads its ratio from ``csa_compress_ratios[2*i]`` and stays numerically identical to
      the GPT ``dsv4_hybrid`` attention layer (baking a fixed ratio via ``C``/``H``/``W``
      would instead trip the ``compress_ratio != ratio`` guard in csa2.py:1280).
    * hybrid index ``2*i + 1`` -- MLP-only, ``E`` if the layer is MoE else ``-``.

    Because CSA2 indexes every per-layer array by ``layer_number - 1``, the compress ratios
    and the kv/index/candidate source layers are re-expanded so a GPT source layer ``j``
    lands on hybrid attention index ``2*j`` (MLP slots get ratio 0 and are never read).
    """
    if len(csa_compress_ratios) != num_layers:
        raise ValueError(
            f'csa_compress_ratios length {len(csa_compress_ratios)} does not match num_layers {num_layers}.')
    moe_mask = _normalize_moe_layer_freq(moe_layer_freq, num_layers)

    pattern_chars: List[str] = []
    hybrid_ratios: List[int] = []
    for i in range(num_layers):
        # attention-only layer (array-driven DSv4 attention)
        pattern_chars.append('D')
        hybrid_ratios.append(int(csa_compress_ratios[i]))
        # MLP-only layer
        pattern_chars.append('E' if moe_mask[i] else '-')
        hybrid_ratios.append(0)

    def _remap(layers: Sequence[int]) -> List[int]:
        return [2 * int(j) for j in layers]

    return HybridLayerConfig(
        hybrid_layer_pattern=''.join(pattern_chars),
        num_layers=2 * num_layers,
        csa_compress_ratios=hybrid_ratios,
        csa2_kv_source_layers=_remap(csa2_kv_source_layers),
        csa2_index_source_layers=_remap(csa2_index_source_layers),
        csa2_candidate_source_layer=(None if csa2_candidate_source_layer is None else 2
                                     * int(csa2_candidate_source_layer)),
    )


if _HYBRID_MODEL_AVAILABLE:

    class DeepseekV41HyperConnectionHybridLayer(HyperConnectionHybridLayer):
        """Hyper-connection wrapper that applies Engram on the n-stream residual, matching GPT.

        The GPT single-pass path (``HyperConnectionTransformerLayer._forward_attention``, upstream
        transformer_layer.py) applies Engram to the *n-stream* residual (width
        ``num_residual_streams * hidden_size``) **before** the self-attention hyper-connection
        aggregates it to a single stream::

            hidden_states = self._maybe_apply_engram(hidden_states, input_ids)   # n-stream, 20480
            hidden_states, ... = self.self_attention_hyper_connection(hidden_states, ...)  # -> 1 stream

        HybridStack inverts that order: :meth:`HyperConnectionHybridLayer.forward` aggregates first
        (``self.hyper_connection(hidden_states)``) and runs the inner layer on the single aggregated
        stream, and its eager fast path (``_call_inner_transformer_layer_without_local_bda``, taken
        for the attention-only 'D' layer) calls ``_forward_self_attention_output_with_bias``
        directly, which skips ``_maybe_apply_engram`` entirely. So the base wrapper either drops
        Engram (fast path) or -- if the fast path is declined -- applies it on the aggregated
        *single*-stream tensor, which is both the wrong width (``hidden_size`` vs.
        ``num_streams * hidden_size``) and the wrong point in the residual.

        We therefore apply Engram here, on the incoming n-stream ``hidden_states``, before
        delegating to the base wrapper forward (aggregation + fast-path attention), so the Engram
        contribution lands on the pre-aggregation streams. The inner ``DeepseekV41TransformerLayer`` keeps its
        ``engram`` module only so the bridge can load/export its weights; the base fast path never
        calls it, so there is no double add. Non-Engram layers keep the base wrapper untouched (this
        subclass is only swapped onto Engram-carrying wrappers, see
        :meth:`DeepseekV41Loader._rewrap_engram_hyper_connection_layers`).

        The fast path is also invoked by the CUDA-graph capture body, which is out of scope for this
        change (plan: no CUDA Graph).
        """

        def forward(self,
                    hidden_states,
                    attention_mask=None,
                    inference_context=None,
                    rotary_pos_emb=None,
                    sequence_len_offset=None,
                    packed_seq_params=None,
                    padding_mask=None,
                    input_ids=None,
                    mhc_recompute_manager=None,
                    mhc_state=None,
                    **layer_kwargs):
            engram = getattr(self.inner_layer, 'engram', None)
            if engram is not None:
                if input_ids is None:
                    raise ValueError('DeepSeek-V4.1 hybrid Engram requires input token IDs on the layer forward.')
                # ``Engram.forward`` reads the THD / inference context off the module itself
                # (mirrors ``_DeepseekV41EngramLayerMixin._forward_attention``), so stash it for
                # the duration of this call and add the n-stream Engram delta like
                # ``TransformerLayer._maybe_apply_engram``.
                previous_ctx = getattr(engram, '_bridge_inference_context', None)
                previous_pack = getattr(engram, '_bridge_packed_seq_params', None)
                engram._bridge_inference_context = inference_context
                engram._bridge_packed_seq_params = packed_seq_params
                try:
                    hidden_states = hidden_states + engram(hidden_states, input_ids, inference_context)
                finally:
                    engram._bridge_inference_context = previous_ctx
                    engram._bridge_packed_seq_params = previous_pack
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                sequence_len_offset=sequence_len_offset,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
                input_ids=input_ids,
                mhc_recompute_manager=mhc_recompute_manager,
                **({
                    'mhc_state': mhc_state
                } if mhc_state is not None else {}),
                **layer_kwargs,
            )

    class DeepseekV41HybridStackModel(HybridModel):
        """``HybridModel`` that splits PP / VPP stages on complete attention+FFN blocks.

        Upstream ``select_pipeline_segment`` (called inside ``HybridModel.__init__``,
        hybrid_model.py:265) handles the split, but for a pattern *without* ``|`` separators it
        (a) refuses VPP outright and (b) slices the ``2 * num_layers`` sublayers evenly, which
        cuts a ``D``/``E`` block across a stage boundary whenever ``2N // stages`` is odd. Either
        breaks :class:`DeepseekV41Bridge`, whose 1-HF-layer -> 2-hybrid-layer fan-out
        assumes the attention half (``2 * i``) and its MLP half (``2 * i + 1``) are co-resident.

        Mirroring GLM-5.3 (``Glm5NextHybridModel``), we pre-segment the *main* pattern on block
        boundaries into ``|``-delimited, PP*VPP-ordered stages before it reaches upstream, and
        assert after build that this rank holds whole blocks -- failing loudly instead of
        mis-mapping weights. The MTP suffix is still appended by the base resolver, so a
        segmented main becomes ``seg0|seg1|.../mtp``.
        """

        # This backbone is text-only, but ``deepseek_v41`` is a multimodal model_type, so the
        # trainer's ``is_multimodal`` path reads ``model.visual`` (expecting ``None`` for text).
        # Expose it so that guard short-circuits; the real vision tower arrives with
        # :class:`DeepseekV41MultimodalModel`.
        visual = None

        # ``MultimodalGPTModel.forward`` (the multimodal wrapper) reads ``language_model.extra_forward_keys``
        # to forward a whitelist of extra kwargs into the decoder. ``McoreHybridModel`` has no such
        # attribute (it lives on the mcore-bridge ``GPTModel``, default ``[]``); expose the same
        # empty default so the wrapper can treat this backbone like any mcore-bridge ``GPTModel``.
        extra_forward_keys: List[str] = []

        @staticmethod
        def _segment_main_pattern(config) -> Optional[str]:
            pattern = config.hybrid_layer_pattern
            if getattr(config, 'pipeline_model_parallel_layout', None) is not None:
                raise ValueError('DeepSeek-V4.1 hybrid splits pipeline stages by hybrid_layer_pattern, so '
                                 'pipeline_model_parallel_layout does not apply; use '
                                 'num_layers_in_first_pipeline_stage / num_layers_in_last_pipeline_stage for an '
                                 'uneven split.')
            # An explicit layout is respected as-is; upstream + the post-build guard validate it.
            if (not pattern or '|' in pattern or config.num_layers_in_first_pipeline_stage is not None
                    or config.num_layers_in_last_pipeline_stage is not None):
                return pattern
            stages = config.pipeline_model_parallel_size
            if config.virtual_pipeline_model_parallel_size:
                stages *= config.virtual_pipeline_model_parallel_size
            if stages <= 1:
                return pattern
            blocks, extra = divmod(len(pattern) // 2, stages)
            if blocks == 0:
                raise ValueError('DeepSeek-V4.1 hybrid needs at least one attention+FFN block per pipeline stage, '
                                 f'but {len(pattern) // 2} blocks cannot cover {stages} stages; lower '
                                 'pipeline_model_parallel_size / virtual_pipeline_model_parallel_size.')
            # Consecutive segments map to (vp0,pp0),(vp0,pp1),... matching upstream's
            # segment_index = vp_stage * pp_size + pp_rank (hybrid_layer_allocation.py:478).
            segments, offset = [], 0
            for stage in range(stages):
                count = 2 * (blocks + int(stage < extra))
                segments.append(pattern[offset:offset + count])
                offset += count
            return '|'.join(segments)

        @staticmethod
        def _resolve_hybrid_layer_pattern(config) -> Optional[str]:
            segmented = DeepseekV41HybridStackModel._segment_main_pattern(config)
            if segmented == config.hybrid_layer_pattern:
                return HybridModel._resolve_hybrid_layer_pattern(config)
            seg_config = copy.copy(config)
            seg_config.hybrid_layer_pattern = segmented
            return HybridModel._resolve_hybrid_layer_pattern(seg_config)

        def __init__(self, config, transformer_layer_spec, pre_process=True, post_process=True, vp_stage=None):
            super().__init__(config, transformer_layer_spec, pre_process, post_process, vp_stage)
            # A stage holding a partial block would break the HF-layer fan-out in the bridge.
            layers = getattr(self.decoder, 'layers', None) or []
            if layers:
                offset = layers[0].layer_number - 1
                count = len(layers)
                if offset % 2 or count % 2:
                    raise ValueError('DeepSeek-V4.1 hybrid pipeline stage boundaries must fall on complete '
                                     f'attention+FFN blocks, but this stage starts at sublayer {offset} and holds '
                                     f'{count} sublayers (both must be even, since one block is two sublayers). '
                                     'Leave num_layers_in_first_pipeline_stage / num_layers_in_last_pipeline_stage '
                                     'unset for an even block-aligned split, or pass even values.')
            # ``HybridModel.forward`` builds no model-level RoPE for ``multi_latent_attention`` and
            # hard-sets ``rotary_pos_emb=None`` when calling the decoder. The reused DSv4 attention
            # (shared with V4) instead expects the decoupled ``{'main', 'compress'}`` dict
            # that ``DeepseekV4GPTModel`` builds. Build the same two RoPE tables here and inject the
            # dict into the decoder via a forward pre-hook, keeping the attention numerically
            # identical to the GPTModel baseline. ``get_rotary_seq_len`` reads ``decoder.input_tensor``
            # when the local ``hidden_states`` is ``None``, so this also covers PP intermediate/last
            # stages.
            self._dsv4_position_ids = None
            self._build_dsv4_rotary_tables()
            self.decoder.register_forward_pre_hook(self._inject_dsv4_rotary_pos_emb, with_kwargs=True)

        def _build_dsv4_rotary_tables(self):
            """Build the MLA decoupled-RoPE ``main``/``compress`` tables (mirrors
            ``mcore_bridge.model.gpt_model.GPTModel`` MLA setup + ``DeepseekV4GPTModel._set_inv_freq``)."""
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.qk_pos_emb_head_dim,
                rotary_percent=1,
                rotary_interleaved=self.config.rotary_interleaved,
                rotary_base=self.config.rotary_base,
                use_cpu_initialization=self.config.use_cpu_initialization,
            )
            rope_scaling = self.config.rope_scaling
            self.config.rope_scaling = rope_scaling['main']
            new_inv_freq, attention_scaling = get_rope_inv_freq(self.config)
            self.rotary_pos_emb.inv_freq = new_inv_freq.to(self.rotary_pos_emb.inv_freq.device)
            self.config.attention_scaling = attention_scaling
            # compress
            self.compress_rotary_pos_emb = copy.copy(self.rotary_pos_emb)
            self.config.rope_scaling = rope_scaling['compress']
            new_inv_freq, attention_scaling = get_rope_inv_freq(self.config)
            self.compress_rotary_pos_emb.inv_freq = new_inv_freq
            self.config.compress_attention_scaling = attention_scaling
            self.config.rope_scaling = rope_scaling

        def _dsv4_rotary_pos_emb(self, transformer_input, packed_seq_params, inference_context=None):
            """Return the ``{'main', 'compress'}`` RoPE dict the DSv4 attention indexes by
            ``rope_layer_type`` (mirrors ``DeepseekV4GPTModel._get_rotary_pos_emb`` plus the
            packed pre-indexing ``GPTModel.forward`` does).

            The DSv4 attention consumes *per-token* frequencies row-aligned with the hidden states
            (see ``_apply_mla_rope``), not a position->frequency table. For one sequence per row the
            table is already row-aligned, but under ``thd`` packing a row holds several sequences
            whose positions restart, so the table (sized by the longest sequence) must be indexed by
            ``position_ids`` here -- exactly what the GPT path does in ``GPTModel.forward``. Under CP
            ``position_ids`` arrives already split with the hidden states' partition mode, so the
            indexed frequencies come out rank-local while keeping absolute positions.
            """
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(inference_context, self.decoder, transformer_input,
                                                                    self.config, packed_seq_params)
            packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
            rotary_pos_emb = {
                'main': self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq),
                'compress': self.compress_rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq),
            }
            if packed_seq and not self.config.apply_rope_fusion:
                position_ids = self._dsv4_position_ids
                if position_ids is None:
                    raise ValueError('DeepSeek-V4.1 hybrid needs position_ids on every pipeline '
                                     'stage to pre-index the MLA rotary table under sequence '
                                     'packing.')
                assert position_ids.shape[0] == 1, f'position_ids.shape: {position_ids.shape}'
                rotary_pos_emb = {k: v[position_ids[0]] for k, v in rotary_pos_emb.items()}
            return rotary_pos_emb

        def _inject_dsv4_rotary_pos_emb(self, module, args, kwargs):
            if kwargs.get('rotary_pos_emb') is not None:
                return None
            transformer_input = kwargs.get('hidden_states')
            if transformer_input is None and args:
                transformer_input = args[0]
            kwargs['rotary_pos_emb'] = self._dsv4_rotary_pos_emb(transformer_input, kwargs.get('packed_seq_params'),
                                                                 kwargs.get('inference_context'))
            return args, kwargs

        # Visual kwargs are injected into the embeddings by the multimodal wrapper and then
        # cleared before the language model runs; the base HybridModel.forward never accepts them.
        # This backbone is text-only, so strip them here. For a text batch
        # ``DeepseekV41Vision.get_inputs_embeds`` is a numeric no-op (``_zero_parameter_dependency``
        # adds ``0 * vision_params``), so dropping them changes nothing numerically.
        _visual_forward_keys = ('pixel_values', 'image_grid_thw', 'image_token_types', 'token_types')

        def forward(self, *args, **kwargs):
            # The multimodal wrapper (``MultimodalGPTModel.forward``) always funnels the
            # decoder's extra kwargs through ``extra_block_kwargs`` -- the mcore-bridge ``GPTModel``
            # calling convention. Upstream ``HybridModel.forward`` has no such parameter (it threads
            # ``input_ids`` into the decoder itself, hybrid_model.py), so unpack the container here
            # and let the visual-key strip below drop anything the text backbone does not consume.
            extra_block_kwargs = kwargs.pop('extra_block_kwargs', None)
            if extra_block_kwargs:
                kwargs.update(extra_block_kwargs)
            if kwargs.get('pixel_values') is not None:
                raise NotImplementedError('The DeepSeek-V4.1 hybrid backbone is text-only; multimodal inputs must go '
                                          'through DeepseekV41MultimodalModel.')
            for key in self._visual_forward_keys:
                kwargs.pop(key, None)
            # Upstream ``HybridModel.forward`` never threads position_ids into the decoder, so stash
            # it for the rotary pre-hook (see :meth:`_dsv4_rotary_pos_emb`).
            position_ids = kwargs.get('position_ids')
            if position_ids is None and len(args) > 1:
                position_ids = args[1]
            self._dsv4_position_ids = position_ids
            try:
                return super().forward(*args, **kwargs)
            finally:
                self._dsv4_position_ids = None

    class DeepseekV41MultimodalModel(MultimodalGPTModel):
        """Multimodal wrapper hosting the ``HybridModel`` backbone.

        ``MultimodalGPTModel`` consumes its ``language_model`` through a backbone-agnostic
        interface -- ``embedding(input_ids, position_ids)`` / ``vp_stage`` /
        ``share_embeddings_and_output_weights`` / ``extra_forward_keys`` / ``set_input_tensor`` /
        ``get_input_tensor`` / ``shared_embedding_or_output_weight`` plus the standard forward
        signature -- all of which :class:`DeepseekV41HybridStackModel` provides
        (``extra_forward_keys`` is added on it for exactly this). The vision tower, image-embed
        injection (``_patch_word_embeddings``) and the vision/aligner weight bridging
        (``MultimodalGPTBridge._convert_pre_process``) are inherited unchanged.

        The wrapper injects image embeddings into the embedding output and clears the visual
        kwargs before the language model runs, so the hybrid backbone only ever sees a text batch
        (its ``forward`` strips ``_visual_forward_keys`` as a defensive backstop).
        """

        language_model_cls = DeepseekV41HybridStackModel

        @property
        def vocab_size(self):
            return self.language_model.vocab_size
else:
    DeepseekV41HyperConnectionHybridLayer = None
    DeepseekV41HybridStackModel = None
    DeepseekV41MultimodalModel = None


class DeepseekV41Loader(DeepseekV4Loader):
    """Build DeepSeek-V4.1 on megatron-core's native ``HybridModel``.

    Extends the V4 loader's MLA/CSA2 knowledge with the V4.1 Engram config resolution and
    rewrites the layer config into the doubled hybrid layer space (see
    :func:`derive_hybrid_layer_config`); the derivation works on a config copy so the caller's
    config is never mutated.

    On top of the text backbone it attaches the DSpark (``mtp.*``) draft stack (in
    :meth:`build_model`, mapped in :meth:`DeepseekV41Bridge._convert_additional_layers`).
    Autoregressive MTP (``mtp_num_layers`` / ``MultiTokenPredictionBlock``) does not apply to
    V4.1 -- its ``mtp.*`` checkpoint keys *are* DSpark -- so it stays disabled here. The backbone
    is wrapped in :class:`DeepseekV41MultimodalModel` for the vision tower + image-embed
    injection.
    """

    # ``HybridModel`` builds its own stack, so leave megatron-core's TransformerBlock
    # unpatched (``register.py`` would otherwise swap in mcore-bridge's variant).
    transformer_block = McoreTransformerBlock

    model_cls = DeepseekV41MultimodalModel

    def _engram_placement_layer_ids(self, hf_layer_ids):
        """On HybridStack, HF layer ``e`` becomes the attention-only 'D' layer at hybrid index
        ``2 * e`` (0-based) -- 1-based ``layer_number`` ``2 * e + 1``. Engram is placed there
        (never on the MLP-only 'E'/'-' layer), so ``TransformerLayer``'s
        ``layer_number in engram_config.layer_ids`` gate builds it on the right hybrid layers.
        ``hash_layer_ids`` stays 0-based HF so the tokenizer artifact / hash multipliers are
        looked up unchanged."""
        return tuple(2 * layer_id + 1 for layer_id in hf_layer_ids)

    def _build_hybrid_config(self):
        # Shallow copy + per-field reassignment (mirrors ``get_dspark_layer_spec``); every field
        # written below is replaced by a fresh object, so the original config is never mutated.
        cfg = copy.copy(self.config)
        derived = derive_hybrid_layer_config(
            self.config.num_layers,
            list(self.config.csa_compress_ratios),
            self.config.moe_layer_freq,
            csa2_kv_source_layers=self.config.csa2_kv_source_layers or [],
            csa2_index_source_layers=self.config.csa2_index_source_layers or [],
            csa2_candidate_source_layer=self.config.csa2_candidate_source_layer,
        )
        cfg.num_layers = derived.num_layers
        cfg.hybrid_layer_pattern = derived.hybrid_layer_pattern
        cfg.csa_compress_ratios = derived.csa_compress_ratios
        cfg.csa2_kv_source_layers = derived.csa2_kv_source_layers
        cfg.csa2_index_source_layers = derived.csa2_index_source_layers
        cfg.csa2_candidate_source_layer = derived.csa2_candidate_source_layer
        cfg.is_hybrid_model = True
        # HybridStack picks E/- from the pattern; keep moe_layer_freq consistent with the doubled
        # space so any layer-count validation that reads it still agrees with num_layers.
        cfg.moe_layer_freq = [1 if symbol == 'E' else 0 for symbol in derived.hybrid_layer_pattern]
        # Autoregressive MTP does not apply to V4.1: the parser never sets ``mtp_num_layers``
        # (it maps ``num_nextn_predict_layers`` to ``dspark_num_layers`` instead), and the
        # ``mtp.*`` checkpoint keys are the DSpark draft stack (attached in ``build_model``).
        # Keep it disabled so no ``MultiTokenPredictionBlock`` is built.
        cfg.mtp_num_layers = None
        return cfg

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        # Build the spec from the *hybrid* config so the CSA2 attention sees the doubled-space
        # csa arrays. ``build_model`` caches it on ``self._hybrid_config`` first.
        spec = hybrid_dsv4_stack_spec(self._hybrid_config)
        # Apply the fp8-parity module swaps on the array-driven 'D'
        # attention layer (the only attention symbol V4.1 emits).
        attn = spec.submodules.dsa_layer.submodules.self_attention
        attn.module = DSv4HybridSelfAttention
        core = attn.submodules.core_attention.submodules
        if getattr(core, 'compressor', None) is not None:
            core.compressor.module = CSA2Compressor
        if getattr(core, 'indexer', None) is not None:
            # CSA2 indexer is flat (no nested compressor).
            core.indexer.module = CSA2Indexer
        # Attach Engram to the 'D' (attention-only) layer spec. Because HybridStack shares one
        # ``dsa_layer`` spec across every 'D' layer, per-layer placement is handled by
        # ``TransformerLayer.__init__`` (only ``layer_number in engram_config.layer_ids`` builds
        # it) rather than by editing per-layer specs.
        engram_config = self._get_engram_config()
        if engram_config is not None:
            from megatron.core.transformer.spec_utils import ModuleSpec
            dsa = spec.submodules.dsa_layer
            # The inference-aware subclass adds the ``_forward_attention`` Engram hook.
            dsa.module = DeepseekV41TransformerLayer
            dsa.submodules.engram = ModuleSpec(module=DeepseekV41Engram, params={'engram_config': engram_config})
        # HybridStack exposes MoE via ``moe_layer`` (symbol 'E') instead of GPT's ``layer_specs``,
        # so ``ModelLoader._replace_router`` never sees it. Swap the stock ``McoreTopKRouter`` for
        # the project ``TopKRouter`` here too, otherwise the MoE ``router`` has no ``expert_bias_vl``
        # buffer and the V4.1 bridge fails to load ``gate.bias_vl`` (mirrors ``_replace_router``).
        self._replace_hybrid_router(spec)
        return spec

    @staticmethod
    def _replace_hybrid_router(spec):
        from functools import partial
        from megatron.core.transformer.moe.router import TopKRouter as McoreTopKRouter

        from ..modules import TopKRouter
        moe_layer = getattr(spec.submodules, 'moe_layer', None)
        mlp_spec = getattr(getattr(moe_layer, 'submodules', None), 'mlp', None)
        # ``get_moe_module_spec_for_backend`` hands back a ``functools.partial(MoELayer, ...)``
        # here (not a plain ``ModuleSpec``), so read its ``submodules`` from ``keywords`` -- same
        # dual handling as ``ModelLoader._replace_router``.
        if isinstance(mlp_spec, partial):
            mlp_submodules = mlp_spec.keywords.get('submodules')
        else:
            mlp_submodules = getattr(mlp_spec, 'submodules', None)
        if getattr(mlp_submodules, 'router', None) is McoreTopKRouter:
            mlp_submodules.router = TopKRouter

    def _rewrap_engram_hyper_connection_layers(self, model):
        """Retrofit Engram-carrying ``HyperConnectionHybridLayer`` wrappers with the V4.1
        subclass that declines the fast path (see
        :class:`DeepseekV41HyperConnectionHybridLayer`).

        HybridStack hard-codes ``HyperConnectionHybridLayer`` (hybrid_block.py:1120-1121) with no
        spec hook, so the swap is done in place after build. ``DeepseekV41HyperConnectionHybridLayer``
        only overrides one method and adds no state, making the ``__class__`` reassignment safe.
        Only wrappers whose inner layer actually built an Engram module (``layer_number in
        engram_config.layer_ids``) are touched; every other layer keeps the base fast path and
        stays numerically identical to a plain hybrid stack.
        """
        if not self.config.enable_hyper_connections or DeepseekV41HyperConnectionHybridLayer is None:
            return
        decoder = getattr(model, 'decoder', None)
        for layer in getattr(decoder, 'layers', []) or []:
            inner = getattr(layer, 'inner_layer', None)
            if (isinstance(layer, HyperConnectionHybridLayer) and inner is not None
                    and getattr(inner, 'engram', None) is not None):
                layer.__class__ = DeepseekV41HyperConnectionHybridLayer

    def build_model(self, pre_process=True, post_process=True, vp_stage: Optional[int] = None):
        """Build the multimodal wrapper around ``HybridModel``, skipping ``ModelLoader.build_model``'s
        GPT layer-spec post-processing (MLA / router / TransformerLayer substitution): a
        ``HybridStack`` spec exposes per-symbol submodules instead, and the DSv4 attention swap is
        done in ``get_transformer_layer_spec`` above.

        ``model`` is :class:`DeepseekV41MultimodalModel` (vision tower + wrapper); the hybrid
        text backbone -- which owns the decoder / MoE / Engram layers the fix-ups below touch --
        is nested under ``model.language_model``, so they target that."""
        self._hybrid_config = self._build_hybrid_config()
        model = self.model_cls(
            config=self._hybrid_config,
            transformer_layer_spec=self.get_transformer_layer_spec(vp_stage=vp_stage),
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )
        language_model = getattr(model, 'language_model', model)
        self._rewrap_engram_hyper_connection_layers(language_model)
        self._set_linear_is_expert(language_model)
        # DSpark: the ``mtp.*`` draft stack is backbone-agnostic (plain experimental-attention
        # layers), so :meth:`_attach_dspark` builds it unchanged and attaches it to the hybrid text
        # backbone (``language_model.dspark``). Inference-time target-layer capture on HybridStack is
        # not implemented (it is not exercised by training / weight round-trip); the stack only needs
        # to exist so its parameters are loaded / saved via ``mtp.*``.
        self._attach_dspark(language_model, post_process, vp_stage=vp_stage)
        return model

    def _get_engram_config(self):
        hf_layer_ids = tuple(self.config.engram_layer_ids or ())
        if not hf_layer_ids:
            return None
        if not has_native_engram():
            raise RuntimeError('DeepSeek-V4.1 Engram requires NVIDIA Megatron-LM Engram support. '
                               'The PR #7224 text-backbone baseline intentionally does not provide it; '
                               'install the official Engram extension or disable Engram explicitly.')
        required = (
            'engram_num_embeddings',
            'engram_max_ngram_size',
            'engram_vocab_size',
            'engram_n_heads',
            'engram_head_dim',
            'engram_pad_token_id',
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
            raise ValueError('DeepSeek-V4.1 Engram requires engram_tokenizer_map. Generate it with '
                             'Megatron-LM/tools/engram/generate_tokenizer_map.py using the HF 0-based '
                             f'layer IDs {list(hf_layer_ids)}.')

        max_ngram_order = self.config.engram_max_ngram_size
        image_token_id = getattr(self.config.hf_config, 'image_token_id', None)
        engram_config = build_deepseek_v41_engram_config(
            global_vocab_sizes=(self.config.engram_vocab_size, ) * (max_ngram_order - 1),
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
            excluded_token_ids=(() if image_token_id is None else (image_token_id, )),
        )
        actual_rows = tuple(sum(engram_config.table_sizes(layer_id)) for layer_id in engram_config.layer_ids)
        expected_rows = tuple(self.config.engram_num_embeddings)
        if actual_rows != expected_rows:
            raise ValueError('DeepSeek-V4.1 Engram table layout does not match engram_num_embeddings: '
                             f'computed {actual_rows}, checkpoint declares {expected_rows}.')
        if (self.config.engram_compressed_vocab_size is not None
                and engram_config.tokenizer_remap.max().item() + 1 != self.config.engram_compressed_vocab_size):
            raise ValueError('DeepSeek-V4.1 compressed tokenizer vocabulary mismatch: artifact has '
                             f'{engram_config.tokenizer_remap.max().item() + 1}, config declares '
                             f'{self.config.engram_compressed_vocab_size}.')
        engram_config.validate_startup(self.config, expected_tokenizer_vocab_size=self.config.padded_vocab_size)
        return engram_config

    def get_dspark_layer_spec(self):
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            _get_backend_spec_provider, get_transformer_layer_with_experimental_attention_variant_spec)

        dspark_config = copy.copy(self.config)
        dspark_config.hf_config = getattr(self.config.hf_config, 'text_config', self.config.hf_config)
        dspark_config.num_layers = self.config.dspark_num_layers
        # A checkpoint may leave the draft stack's expert counts out, which means its draft layers are
        # shaped like the backbone's MoE rather than carrying their own shape.
        if self.config.dspark_num_experts is not None:
            dspark_config.num_moe_experts = self.config.dspark_num_experts
        if self.config.dspark_router_topk is not None:
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

    def _attach_dspark(self, language_model, post_process, vp_stage: Optional[int] = None):
        """Build the DSpark (``mtp.*``) draft stack and attach it to ``language_model`` on the
        final pipeline stage.

        Backbone-agnostic: the draft layers are plain experimental-attention-variant
        ``TransformerLayer`` instances (see :meth:`get_dspark_layer_spec`), independent of whether
        the main model is a ``GPTModel`` or ``HybridModel``; the caller passes whichever object
        owns the stack -- here the ``HybridModel`` backbone, which exposes ``pg_collection`` /
        ``vocab_size`` / ``config`` all the same.
        The stack is never part of the training forward (capture is inference-only), so it only
        needs to exist here so its parameters are loaded / saved through the ``mtp.*`` bridge -- which
        is also why it is frozen at the end of this method.

        ``vp_stage`` must be threaded into ``build_module`` because the draft layers reuse the
        experimental-attention ``TransformerLayer``, whose ``__init__`` calls
        ``get_transformer_layer_offset`` -- and that helper asserts ``vp_stage is not None`` under
        VPP. The draft stack keeps its own local 1-based numbering; the pipeline offset the helper
        adds is the same value the last stage already applied under plain PP (0 for the tiny draft
        stack), so this only satisfies the VPP assertion without changing placement.
        """
        if not self.config.dspark_num_layers or not post_process:
            return
        dspark_config, dspark_layer_specs = self.get_dspark_layer_spec()
        layers = [
            build_module(
                layer_spec,
                config=dspark_config,
                layer_number=index + 1,
                pg_collection=language_model.pg_collection,
                vp_stage=vp_stage,
            ) for index, layer_spec in enumerate(dspark_layer_specs)
        ]
        language_model.dspark = DeepseekV41DSparkStack(dspark_config, layers)
        self._set_linear_is_expert(language_model.dspark)
        # ``Float16Module`` casts every unmarked float buffer, and mcore's ``TopKRouter`` only
        # restores the aux-loss-free bias to fp32 lazily -- from ``forward`` and from
        # ``_save_to_state_dict``. The draft stack never runs in the training forward, and the
        # bridge copies ``param.data`` directly instead of going through the state-dict hooks, so
        # without this marker the checkpoint's fp32 ``mtp.*.ffn.gate.bias`` would round-trip
        # through bf16. The main layers escape it only because their routers do run.
        for layer in layers:
            expert_bias = getattr(getattr(layer.mlp, 'router', None), 'expert_bias', None)
            if expert_bias is not None:
                mark_keep_in_fp32(expert_bias)
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
        # Nothing here runs in the training forward, so none of these parameters can ever receive a
        # gradient -- and leaving them trainable does more than waste the optimizer state and the
        # gradient buffers. Adam's weight decay is decoupled from the gradient, so a parameter whose
        # gradient stays zero is still multiplied by ``1 - lr * wd`` on every step with nothing to
        # balance it, and the ``mtp.*`` weights loaded from the checkpoint would decay away over a
        # long run (bf16 rounding only hides this until the drift crosses one ulp of the parameter;
        # the fp32 main weights shrink from the first step). Freezing keeps them out of the optimizer
        # altogether. The bridge reads and writes ``param.data`` directly, so the checkpoint still
        # round-trips the draft stack unchanged -- which is the whole reason it is built here.
        for module in (language_model.dspark, getattr(language_model, 'dspark_word_embeddings', None)):
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = False


class DeepseekV41Bridge(DeepseekV4Bridge):
    """Weight bridge for the ``HybridModel`` backbone.

    An HF layer owns both attention and MLP; on ``HybridModel`` it is split in two (see
    :func:`derive_hybrid_layer_config`), so this bridge fans a single HF layer ``i`` out onto
    two hybrid layers:

    * hybrid layer ``2*i`` -- attention half: MLA / CSA2 state + ``attn_norm`` (+ Engram when
      ``i in engram_layer_ids``) + the ``hc_attn_*`` hyper-connection channel.
    * hybrid layer ``2*i + 1`` -- MLP half: MoE / dense state + ``ffn_norm`` + the ``hc_ffn_*``
      hyper-connection channel.

    When ``enable_hyper_connections`` is set each hybrid layer is wrapped in a
    ``HyperConnectionHybridLayer`` whose real payload lives under ``inner_layer`` and which owns
    a *single* ``hyper_connection`` module, so the HF ``hc_{attn,ffn}_*`` keys split across the
    two wrappers.

    ``self.config`` is seen in two layer spaces depending on direction: on load it is the
    original HF-space config (``num_layers == N``, no ``hybrid_layer_pattern``); on export it is
    the doubled hybrid megatron config used to build the model (``num_layers == 2 * N``,
    ``hybrid_layer_pattern`` populated). :meth:`_convert` normalizes this so it always iterates
    the decoder's ``2 * N`` hybrid layers.
    """

    _ENGRAM_LOAD_CHUNK_ROWS = 65536
    additional_dim0_keys = DeepseekV4Bridge.additional_dim0_keys | {'embed', 'head'}
    additional_dim1_keys = DeepseekV4Bridge.additional_dim1_keys | {'main_proj'}

    @staticmethod
    def _lm(mg_model):
        """Resolve the language model. A bare HybridModel is text-only (no ``language_model``
        wrapper); :class:`DeepseekV41MultimodalModel` nests it under a multimodal container."""
        language_model = getattr(mg_model, 'language_model', None)
        return mg_model if language_model is None else language_model

    @staticmethod
    def _num_hybrid_layers(config) -> int:
        """Decoder layer count in the doubled hybrid space, regardless of which layer space
        ``config`` is currently in.

        The two conversion entrypoints hand :meth:`_convert` a config in *different* spaces:
        load (``to_mcore=True``) passes the original HF-space config (``num_layers == N``, no
        ``hybrid_layer_pattern``) whose built decoder holds ``2 * N`` layers; export
        (``to_mcore=False``) passes the doubled hybrid megatron config used to build the model
        (``num_layers == 2 * N`` with ``hybrid_layer_pattern`` populated). Discriminating by the
        pattern makes both directions iterate exactly the decoder's layer count -- using the raw
        ``2 * num_layers`` on export would over-count and dereference ``None`` layers past the
        decoder end (see PP-availability window in :meth:`_convert`)."""
        if getattr(config, 'hybrid_layer_pattern', None):
            return config.num_layers
        return 2 * config.num_layers

    def _engram_hf_layer_id(self, engram):
        # Engram lives on the doubled-space attention layer ``2 * hf_id + 1`` (see
        # ``DeepseekV41Loader._engram_placement_layer_ids``), so map it back to HF space.
        return (engram.layer_number - 1) // 2

    def _set_word_embeddings(self, mg_model, hf_state_dict, to_mcore):
        # The base ``MultimodalGPTBridge`` resolves the language model with a raw
        # ``getattr(mg_model, 'language_model')``; route it through :meth:`_lm` so both the
        # multimodal wrapper and a bare backbone resolve correctly.
        self._set_state_dict(
            self._lm(mg_model), 'embedding.word_embeddings.weight', hf_state_dict, self.hf_embed_key, to_mcore)

    def _convert_pre_process(self, mg_model, hf_state_dict, hf_prefix: str, to_mcore: bool):
        # Runs the vision/aligner + image_* block unconditionally instead of branching on *this*
        # rank's ``mg_model.visual``. On export every pipeline stage runs ``_convert`` ->
        # ``_convert_pre_process``, and this path issues the *same* pp-group collective sequence on
        # all ranks: word-embeddings (routed through ``_lm`` by :meth:`_set_word_embeddings`), then
        # the config-guarded vision/aligner block and the image_* markers, all driven via
        # ``_set_module``/``_set_state_dict`` which stay in lockstep even where the submodule is
        # ``None`` (see ``_set_module``'s ``src_rank`` all-reduce / ``_set_state_dict``'s ``state``
        # all-reduce). A per-rank ``visual is not None`` guard would skip that whole block on
        # non-first stages (where the wrapper built ``visual=None``), desynchronizing the
        # collectives so the last stage's later per-layer ``has_model`` all-reduce reads a stale
        # value -> ``next(mg_models)`` -> ``StopIteration``. On load only the first stage reaches
        # this method (see ``_convert``'s ``is_pp_first_stage`` guard).
        result = super()._convert_pre_process(mg_model, hf_state_dict, hf_prefix, to_mcore)
        target = hf_state_dict if to_mcore else result
        for name in ('image_start', 'image_end', 'image_newline'):
            self._set_state_dict(mg_model, f'visual.{name}', target, f'model.{name}', to_mcore)
        return result

    def _convert_post_process(self, mg_model, hf_state_dict, hf_prefix: str, to_mcore: bool):
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        lm_model = self._lm(mg_model)
        if self.config.task_type != 'embedding':
            if self.config.untie_embeddings_and_output_weights:
                hf_lm_head_key = self.hf_lm_head_key
                if self.config.task_type == 'seq_cls':
                    hf_lm_head_key = self.hf_score_key
                if not to_mcore or hf_lm_head_key in hf_state_dict:
                    self._set_state_dict(lm_model, 'output_layer.weight', hf_state_dict, hf_lm_head_key, to_mcore)
            elif to_mcore and lm_model.output_layer.weight is not None:
                self._set_state_dict(lm_model, 'output_layer.weight', hf_state_dict, self.hf_embed_key, to_mcore)
        self._set_final_layernorm(lm_model, hf_state_dict, to_mcore)
        if to_mcore:
            return {}
        return self._add_prefix(hf_state_dict, hf_prefix)

    def _set_final_layernorm(self, lm_model, hf_state_dict, to_mcore):
        # HybridStack names its trailing norm ``final_norm`` (vs the GPT block's
        # ``final_layernorm``). Like the GPT V4.1 bridge, single-pass mHC has no learned
        # ``hc_head_*`` output head (only built when ``not mhc_single_pass``), so nothing else
        # is mapped here.
        self._set_state_dict(lm_model, 'decoder.final_norm.weight', hf_state_dict, self.hf_final_layernorm_key,
                             to_mcore)

    def _set_one_hyper_connection(self, hyper_connection, hf_state_dict, hf_key, to_mcore):
        """Bridge a single ``HyperConnectionModule`` (one wrapper == one channel).

        Same parameter layout as the GPT ``_set_hyper_connection`` per-channel body, but keyed
        by an explicit ``hf_key`` ('attn' or 'ffn') because each hybrid wrapper owns exactly one
        connection instead of the GPT layer's attention + FFN pair.
        """
        self._set_state_dict(hyper_connection, 'mapping_proj.weight', hf_state_dict, f'hc_{hf_key}_fn', to_mcore)
        self._set_state_dict(hyper_connection, 'bias', hf_state_dict, f'hc_{hf_key}_base', to_mcore)
        has_hyper_connection = hyper_connection is not None
        has_hyper_connection = self._reduce_tensor_pp_group(has_hyper_connection, to_mcore)
        # ``alpha_*`` bypass ``_set_state_dict``, so mirror the peft guard the GPT
        # ``_set_hyper_connection`` applies -- these are frozen base weights and must stay out of
        # ``adapter_model.safetensors``.
        if has_hyper_connection and not self._peft_format:
            if to_mcore:
                alpha = hf_state_dict[f'hc_{hf_key}_scale'].load()
                for i, alpha_suffix in enumerate(['pre', 'post', 'res']):
                    getattr(hyper_connection, f'alpha_{alpha_suffix}').data[:] = alpha[i]
            else:
                alpha = None
                if hyper_connection is not None:
                    alpha = torch.concat(
                        [getattr(hyper_connection, f'alpha_{suffix}') for suffix in ['pre', 'post', 'res']], dim=0)
                hf_state_dict[f'hc_{hf_key}_scale'] = self._get_weight(alpha, 'alpha')[0]

    def _set_hybrid_layer_state(self, mg_layer, hf_state_dict, hf_prefix: str, hybrid_idx: int, to_mcore: bool):
        """Map HF layer ``hybrid_idx // 2`` onto one half of the hybrid pair.

        Even ``hybrid_idx`` is the attention half, odd is the MLP half; both read/write the
        same ``model.layers.{hf_idx}.`` prefix so the HF checkpoint stays single-layer-per-index.
        """
        hf_idx = hybrid_idx // 2
        is_attn = (hybrid_idx % 2 == 0)
        layer_prefix = f'{hf_prefix}{hf_idx}.'
        local_state = self._remove_prefix(hf_state_dict, layer_prefix) if to_mcore else {}
        # The wrapper carries the payload under ``inner_layer`` and the single hyper-connection
        # under ``hyper_connection``; without mHC the layer is the payload itself.
        inner = None if mg_layer is None else getattr(mg_layer, 'inner_layer', mg_layer)
        hyper_connection = None if mg_layer is None else getattr(mg_layer, 'hyper_connection', None)
        if is_attn:
            local_state.update(self._set_layer_attn(inner, local_state, hf_idx, to_mcore))
            if hf_idx in (self.config.engram_layer_ids or []):
                # ``_get_layer_engram`` already unwraps ``inner_layer.engram``.
                self._set_layer_engram(mg_layer, local_state, to_mcore)
            if self.config.enable_hyper_connections:
                self._set_one_hyper_connection(hyper_connection, local_state, 'attn', to_mcore)
        else:
            local_state.update(self._set_layer_mlp(inner, local_state, hf_idx, to_mcore))
            if self.config.enable_hyper_connections:
                self._set_one_hyper_connection(hyper_connection, local_state, 'ffn', to_mcore)
        if to_mcore:
            return {}
        return self._add_prefix(local_state, layer_prefix)

    def _convert_additional_layers(self, mg_model, hf_state_dict, hf_prefix, to_mcore, is_pp_last_stage):
        """Map the DSpark (``mtp.*``) draft stack.

        The draft layers are plain experimental-attention ``TransformerLayer`` instances --
        backbone-agnostic -- so :meth:`_convert_dspark_stack` owns the mapping and this method
        only locates the stack. It is attached to the text backbone (no ``language_model`` wrapper on
        a bare model, so use :meth:`_lm`) and only
        on the final pipeline stage. During export non-last stages use an empty structural proxy so
        every PP rank executes the same collective sequence. MTP (``mtp_num_layers``) is skipped in
        :meth:`_convert` and does not apply to V4.1."""
        if not self.config.dspark_num_layers or (to_mcore and not is_pp_last_stage):
            return
        language_model = self._lm(mg_model)
        dspark = getattr(language_model, 'dspark', None)
        if dspark is None:
            if to_mcore or is_pp_last_stage:
                raise RuntimeError('DSpark weights require the draft stack on the final pipeline stage.')
            dspark = SimpleNamespace(layers=[None] * self.config.dspark_num_layers)
        yield from self._convert_dspark_stack(language_model, dspark, hf_state_dict, hf_prefix, to_mcore)

    def _convert(self, mg_models, hf_state_dict, hf_prefix: str, to_mcore: bool, tqdm_desc: str = 'Converting: '):
        """Backbone conversion with a 1->2 layer fan-out.

        Mirrors :meth:`GPTBridge._convert` but iterates the doubled hybrid layer space
        (``2 * num_layers``) and dispatches each hybrid layer to :meth:`_set_hybrid_layer_state`.
        MTP is intentionally skipped: V4.1's ``mtp.*`` keys are DSpark, mapped separately.
        """
        self._pending_export_iter = None
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
            hf_state_dict = self._convert_hf_state_dict(hf_state_dict, to_mcore)
        else:
            hf_state_dict = {}
        mg_models = iter(mg_models)
        mg_model = next(mg_models)
        is_pp_first_stage = mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=mg_model.vp_stage)
        is_pp_last_stage = mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=mg_model.vp_stage)
        if not to_mcore or is_pp_first_stage:
            hf_state_dict.update(self._convert_pre_process(mg_model, hf_state_dict, '', to_mcore))
        if to_mcore:
            yield
        else:
            hf_state_dict = self._convert_hf_state_dict(hf_state_dict, to_mcore)
            yield from list(self._add_prefix(hf_state_dict, hf_prefix).items())
            hf_state_dict = {}
        # Total hybrid (attention + MLP) layer count in the doubled space; ``_num_hybrid_layers``
        # normalizes the two layer spaces ``self.config`` may be in (see its docstring) so both
        # load and export iterate exactly the decoder's layer count. HybridStack layer_number
        # spans this same space (i + 1 + pp_offset), matching this loop's hybrid index so the
        # PP-availability window below stays correct.
        num_hybrid_layers = self._num_hybrid_layers(self.config)
        layer_idx = 0
        disable_tqdm = self._disable_tqdm or not is_master()
        prog_bar = tqdm(range(num_hybrid_layers), dynamic_ncols=True, desc=tqdm_desc, disable=disable_tqdm)
        while layer_idx < num_hybrid_layers:
            lm_model = self._lm(mg_model)
            if len(lm_model.decoder.layers) > 0:
                start_idx = lm_model.decoder.layers[0].layer_number - 1
                mg_layer_available = (start_idx <= layer_idx < lm_model.decoder.layers[-1].layer_number)
            else:
                mg_layer_available = False
            if mg_layer_available:
                mg_layer = lm_model.decoder.layers[layer_idx - start_idx]
            else:
                if to_mcore:
                    layer_idx += 1
                    prog_bar.update()
                    continue
                else:
                    mg_layer = None
            if not to_mcore and self.pp_size > 1:
                has_model = torch.tensor([mg_layer is not None], dtype=torch.bool, device='cuda')
                dist.all_reduce(has_model, group=self.pp_group)
                if not has_model:
                    mg_model = next(mg_models)  # compat vpp
                    continue
            res = self._set_hybrid_layer_state(mg_layer, hf_state_dict, f'{self.hf_layers_prefix}.', layer_idx,
                                               to_mcore)
            layer_idx += 1
            prog_bar.update()
            if to_mcore:
                yield
            else:
                res = self._convert_hf_state_dict(res, to_mcore)
                yield from self._drain_pending_export(hf_prefix)
                yield from self._add_prefix(res, hf_prefix).items()
                hf_state_dict = {}
        prog_bar.close()
        yield from self._convert_additional_layers(mg_model, hf_state_dict, hf_prefix, to_mcore, is_pp_last_stage)
        if not to_mcore or is_pp_last_stage:
            hf_state_dict.update(self._convert_post_process(mg_model, hf_state_dict, '', to_mcore))
        if to_mcore:
            yield
        else:
            hf_state_dict = self._convert_hf_state_dict(hf_state_dict, to_mcore)
            yield from list(self._add_prefix(hf_state_dict, hf_prefix).items())

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
            raise ValueError(f'Invalid Engram FP8 weight/scale shapes: {tuple(weight.shape)} and {tuple(scale.shape)}.')
        if weight.shape[1] % scale.shape[1] != 0:
            raise ValueError(f'Engram weight width {weight.shape[1]} is not divisible by scale width {scale.shape[1]}.')
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
        expected_rows = self.config.engram_num_embeddings[self.config.engram_layer_ids.index(hf_layer_id)]
        if flat_offset != expected_rows:
            raise ValueError(f'Engram layer {hf_layer_id} expected {expected_rows} flat rows, '
                             f'but its prime tables contain {flat_offset}.')

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
            engram.key_projection.weight.data.copy_(wkv[:key_rows].to(engram.key_projection.weight))
            engram.value_projection.weight.data.copy_(wkv[key_rows:].to(engram.value_projection.weight))
            engram.query_norm.weight.data.copy_(hf_state_dict['engram.q_weight'].load().reshape(-1).to(
                engram.query_norm.weight))
            engram.key_norm.weight.data.copy_(hf_state_dict['engram.k_weight'].load().reshape(-1).to(
                engram.key_norm.weight))
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
            key_w = engram.key_projection.weight.data.cpu()  # [stream_width, total_memory_dim]
            val_w = engram.value_projection.weight.data.cpu()  # [hidden, total_memory_dim]
            hf_state_dict['engram.wkv.weight'] = torch.cat([key_w, val_w], dim=0)
            # --- export norm weights as q_weight / k_weight ---
            num_streams = engram.num_streams
            hf_state_dict['engram.q_weight'] = engram.query_norm.weight.data.cpu().reshape(num_streams, -1)
            hf_state_dict['engram.k_weight'] = engram.key_norm.weight.data.cpu().reshape(num_streams, -1)

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
        self._set_state_dict(dspark, 'output.markov_head.embed.weight', last_state, 'markov_head.embed.weight',
                             to_mcore)
        self._set_state_dict(dspark, 'output.markov_head.head.weight', last_state, 'markov_head.head.weight', to_mcore)
        self._set_state_dict(dspark, 'output.confidence_head.proj.weight', last_state, 'confidence_head.proj.weight',
                             to_mcore)
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

    def _convert_dspark_stack(self, language_model, dspark, hf_state_dict, hf_prefix, to_mcore):
        """Map the DSpark draft layers + endpoints between HF ``mtp.*`` keys and the megatron
        stack. Backbone-agnostic (the draft layers are plain ``TransformerLayer`` instances); the
        caller locates ``dspark`` and guards the pipeline stage."""
        # On a PP>1 last stage with untied embeddings, DSpark owns a dedicated input
        # embedding (see build_model). Load it from the same HF source as the base
        # first-stage embedding. On export the first stage already emits this tensor,
        # so the redundant DSpark copy is not written back.
        if to_mcore and getattr(language_model, 'dspark_word_embeddings', None) is not None:
            self._load_dspark_word_embeddings(language_model.dspark_word_embeddings, hf_state_dict)
            yield

        original_num_experts = self.config.num_moe_experts
        # Mirrors get_dspark_layer_spec: without its own expert count the draft stack was built with the
        # backbone's, so the weight mapping has to agree with what was built.
        if self.config.dspark_num_experts is not None:
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


if _HYBRID_MODEL_AVAILABLE:
    register_model(
        ModelMeta(
            ModelType.deepseek_v41,
            ['deepseek_v41'],
            bridge_cls=DeepseekV41Bridge,
            visual_cls=DeepseekV41Vision,
            loader=DeepseekV41Loader,
            config_cls=MLAModelConfig,
        ))
