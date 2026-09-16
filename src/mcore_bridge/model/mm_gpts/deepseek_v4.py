# Copyright (c) ModelScope Contributors. All rights reserved.
"""DeepSeek-V4-Flash-Vision (deepseek_v4_vl) model.

Implements the ViT vision encoder, aligner, image-token embedding parameters,
DSpark MTP per-layer weight conversion, and ``bias_vl`` MoE router bias support.
The reference implementation lives in the model card's ``inference/`` directory
(``vision.py`` for the ViT+Aligner, ``model.py`` for DSpark blocks).
"""
import math
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from torch import nn
from functools import lru_cache
from typing import Optional

from mcore_bridge.bridge import GPTBridge
from mcore_bridge.config import ModelConfig
from mcore_bridge.utils import deep_getattr, get_logger, reconstruct_tensor_cp, split_cp_inputs
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import VocabParallelEmbedding, scatter_to_sequence_parallel_region
from mcore_bridge.model.modules.topk_router import TopKRouter
from megatron.core.transformer.moe.router import TopKRouter as McoreTopKRouter

from ..constant import ModelType
from ..gpts.deepseek_v4 import DeepseekV4Bridge, DeepseekV4GPTModel, DeepseekV4Loader
from ..mm_gpt_model import MultimodalGPTModel
from ..register import ModelMeta, register_model
from .utils import HuggingFaceVit

logger = get_logger()

# ---------------------------------------------------------------------------
# Vision 2D RoPE helpers
# ---------------------------------------------------------------------------


@lru_cache(8)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    """Precompute 2D rotary cos/sin for the ViT.

    Positions are laid out as (h_pos, w_pos) interleaved so that the first
    half of the frequency table is driven by the row index and the second half
    by the column index.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_vision_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding in the interleaved (rotate_half) convention."""
    dtype = x
    x = x.float()
    x1, x2 = x.chunk(2, dim=-1)
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out.to(dtype.dtype)


# ---------------------------------------------------------------------------
# ViT building blocks
# ---------------------------------------------------------------------------


class _VisionRMSNorm(nn.Module):
    """RMSNorm with float32 weight, matching the reference ``inference/vision.py``."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class _PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(3 * patch_size ** 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class _VisionAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wqkv = nn.Linear(dim, 3 * dim)
        self.wo = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in self.wqkv(x).chunk(3, dim=-1))
        q = apply_vision_rotary(q, cos, sin)
        k = apply_vision_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        return self.wo(o.transpose(0, 1).reshape(n, -1))


class _VisionMLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, 2 * inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class _VisionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, inter_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm1 = _VisionRMSNorm(dim, eps)
        self.attn = _VisionAttention(dim, n_heads)
        self.norm2 = _VisionRMSNorm(dim, eps)
        self.mlp = _VisionMLP(dim, inter_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class _ViT(nn.Module):
    """DeepSeek ViT: full bidirectional attention over one image with 2D RoPE."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        c = config
        self.rope_dim = c.vision_dim // c.vision_n_heads // 2
        self.rope_theta = c.vision_rope_theta
        self.patch_embed = _PatchEmbed(c.vision_patch_size, c.vision_dim)
        self.blocks = nn.ModuleList([
            _VisionBlock(c.vision_dim, c.vision_n_heads, c.vision_inter_dim, eps=c.layernorm_epsilon)
            for _ in range(c.vision_n_layers)
        ])
        self.norm = _VisionRMSNorm(c.vision_dim, c.layernorm_epsilon)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        cos = cos.to(device=x.device, dtype=x.dtype)
        sin = sin.to(device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class _Aligner(nn.Module):
    """Spatial downsample + 2-layer MLP that maps ViT features to LLM hidden size."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        c = config
        self.downsample_ratio = c.vision_downsample_ratio
        in_dim = c.vision_dim * self.downsample_ratio ** 2
        self.w1 = nn.Linear(in_dim, c.hidden_size)
        self.w2 = nn.Linear(c.hidden_size, c.hidden_size)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))


# ---------------------------------------------------------------------------
# Full vision module (HuggingFaceVit subclass)
# ---------------------------------------------------------------------------

# Sentinel token offsets above vocab_size, matching inference/image_processor.py:
#   IMAGE_START=0, IMAGE_PAD=1, IMAGE=2, IMAGE_NEW_LINE=3, IMAGE_END=4
IMAGE_START = 0
IMAGE_PAD = 1
IMAGE = 2
IMAGE_NEW_LINE = 3
IMAGE_END = 4


class DeepSeekV4Vit(HuggingFaceVit):
    """Vision encoder + aligner + image-token embeddings for DeepSeek-V4-VL.

    Since transformers does not ship a DSv4 VL model, the ViT and Aligner are
    built from scratch using plain ``nn.Module`` layers.  Weight conversion is
    driven by ``module_mapping`` (HF ``vision.*`` / ``aligner.*``) and explicit
    handling of the four image-token embedding parameters.
    """

    # HF prefix -> mcore attribute path under ``visual.``
    module_mapping = {
        'model.vision': 'vit',
        'model.aligner': 'aligner',
    }
    _vision_tower = ['vit']
    _aligner = ['aligner']

    def prepare_model(self, hf_config):
        c = self.config
        self.vit = _ViT(c)
        self.aligner = _Aligner(c)
        # Learnable embeddings for the sentinel tokens.
        self.image_start = nn.Parameter(torch.empty(c.hidden_size, dtype=c.params_dtype))
        self.image_end = nn.Parameter(torch.empty(c.hidden_size, dtype=c.params_dtype))
        self.image_newline = nn.Parameter(torch.empty(c.hidden_size, dtype=c.params_dtype))
        self.image_pad = nn.Parameter(torch.empty(c.hidden_size, dtype=c.params_dtype))
        self.vocab_size = c.padded_vocab_size
        self._image_params = [self.image_start, self.image_pad, self.image_pad,
                              self.image_newline, self.image_end]

    # -- image encoding ---------------------------------------------------

    def _encode_image(self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        """Run ViT + Aligner and return aligned embeddings [n_llm_tokens, hidden_size]."""
        features = self.vit(patches, n_vit_h, n_vit_w)
        return self.aligner(features, n_vit_h, n_vit_w)

    # -- inputs_embeds ----------------------------------------------------

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        """Replace sentinel-token positions in ``inputs_embeds`` with vision features.

        Sentinel tokens are at ``vocab_size + [0..4]``.  ``IMAGE`` (offset 2)
        positions receive the corresponding aligned vision features; the other
        four sentinel types receive their learnable embedding parameter.
        """
        input_ids = kwargs.get('input_ids')
        pixel_values_list = kwargs.get('pixel_values')  # may be a list per sample

        if pixel_values_list is None or (isinstance(pixel_values_list, torch.Tensor)
                                         and pixel_values_list.numel() == 0):
            # No image in this micro-batch; zero-touch.
            return {'inputs_embeds': inputs_embeds}

        # Handle both single-tensor and list-of-tensors conventions.
        if isinstance(pixel_values_list, list):
            images = pixel_values_list
        else:
            images = [pixel_values_list]

        # Gather per-sentinel masks.
        vs = self.vocab_size
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        # Build the parameter stack [image_start, image_pad, image_pad, image_newline, image_end]
        # on the correct device/dtype.
        param_stack = torch.stack([
            self.image_start.to(device, dtype),
            self.image_pad.to(device, dtype),
            self.image_pad.to(device, dtype),
            self.image_newline.to(device, dtype),
            self.image_end.to(device, dtype),
        ])  # [5, hidden_size]

        # For each image, encode and scatter into inputs_embeds.
        # The caller is responsible for providing patches + grid info via kwargs.
        # We support the simple case: a list of (patches, n_vit_h, n_vit_w, perm, types, start)
        # tuples passed as ``image_inputs``.
        image_inputs = kwargs.get('image_inputs')
        if image_inputs is None:
            return {'inputs_embeds': inputs_embeds}

        bsz = inputs_embeds.shape[1] if inputs_embeds.ndim == 3 else inputs_embeds.shape[0]
        if inputs_embeds.ndim == 3:
            # [s, b, h] layout
            for sample_idx, sample_images in enumerate(image_inputs):
                if sample_images is None:
                    continue
                for img in sample_images:
                    embeds = self._encode_image(
                        img['patches'].to(device=device, dtype=dtype),
                        img['n_vit_h'], img['n_vit_w'],
                    )
                    if 'perm' in img:
                        embeds = embeds[img['perm'].to(device)]
                    types = img['types'].to(device)
                    block = param_stack[types]
                    mask_image = types == IMAGE
                    block[mask_image] = embeds
                    start = img['start']
                    end = start + block.size(0)
                    inputs_embeds[start:end, sample_idx] = block
        else:
            # [b, s, h] layout
            for sample_idx, sample_images in enumerate(image_inputs):
                if sample_images is None:
                    continue
                for img in sample_images:
                    embeds = self._encode_image(
                        img['patches'].to(device=device, dtype=dtype),
                        img['n_vit_h'], img['n_vit_w'],
                    )
                    if 'perm' in img:
                        embeds = embeds[img['perm'].to(device)]
                    types = img['types'].to(device)
                    block = param_stack[types]
                    mask_image = types == IMAGE
                    block[mask_image] = embeds
                    start = img['start']
                    end = start + block.size(0)
                    inputs_embeds[sample_idx, start:end] = block

        return {'inputs_embeds': inputs_embeds}

    def get_inputs_embeds_language_model(self, inputs_embeds, **kwargs):
        return inputs_embeds

    # -- mask_input_ids ---------------------------------------------------

    def mask_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Clamp sentinel tokens (>= vocab_size) to 0 so the LLM embedding
        lookup does not hit out-of-range indices.  The actual image features
        are already scattered into ``inputs_embeds`` by ``get_inputs_embeds``.
        """
        return torch.where(input_ids >= self.vocab_size, torch.zeros_like(input_ids), input_ids)


# ---------------------------------------------------------------------------
# Bridge subclass with VL-specific weight conversion
# ---------------------------------------------------------------------------

class DeepseekV4VLBridge(DeepseekV4Bridge):
    """Bridge for ``deepseek_v4_vl``: extends the text-only DeepseekV4Bridge
    with vision module conversion, image-token embedding parameters, and
    ``gate.bias_vl`` MoE router bias.
    """

    hf_mtp_prefix = 'model.mtp'
    hf_embed_key = 'model.embed.weight'
    hf_lm_head_key = 'model.head.weight'

    # _convert_mtp_extra and _set_router are inherited from DeepseekV4Bridge,
    # which handles per-layer DSpark keys and gate.bias_vl conversion.

    def _convert_pre_process(self, mg_model, hf_state_dict, hf_prefix: str, to_mcore):
        """Convert vision module weights + image-token embedding parameters."""
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}

        # Word embeddings (same as parent but we need to also handle visual).
        self._set_word_embeddings(mg_model, hf_state_dict, to_mcore)

        if self.is_multimodal and not self.config.language_model_only:
            # Vision tower + aligner via module_mapping.
            for prefix, mg_prefix in self.module_mapping.items():
                mg_module = deep_getattr(mg_model, f'visual.{mg_prefix}')
                hf_state_dict.update(
                    self._set_module(mg_module, hf_state_dict, f'{hf_prefix}{prefix}.', to_mcore))
            # Image-token embedding parameters (image_start, image_end, image_newline, image_pad).
            # Keys in the checkpoint are prefixed with ``model.`` by _convert_hf_state_dict.
            visual = getattr(mg_model, 'visual', None)
            if visual is not None:
                for key in ['image_start', 'image_end', 'image_newline', 'image_pad']:
                    self._set_state_dict(visual, key, hf_state_dict, f'model.{key}', to_mcore)

        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict


class DeepseekV4VLTopKRouter(TopKRouter):
    """TopKRouter with vision-language bias (``vl_bias``) for DeepSeek-V4-VL.

    The VL checkpoint ships a ``gate.bias_vl`` parameter on every MoE router.
    We register it as a persistent buffer so it round-trips through checkpoint
    save/load; the actual gating logic (adding ``vl_bias`` to logits only for
    image positions) is left to a future training hook — weight loading works
    without it.
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.enable_vl_bias = getattr(config, 'moe_router_enable_vl_bias', False)
        if not self.is_hash_layer:
            self.register_buffer(
                'vl_bias',
                torch.zeros(
                    config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
            )
        else:
            self.vl_bias = None


class DeepseekV4VLGPTModel(DeepseekV4GPTModel):
    """VL-specific language model that handles HC + MTP tensor shape mismatch.

    On multimodal+MTP+HC pipeline stages, the ``mtp_decoder_input`` carries
    HC-expanded layout ``[s, b, num_residual_streams * hidden_size]`` but the
    MTP layers expect single-stream ``[s, b, hidden_size]``.  These overrides
    slice/expand at the pipeline boundary.
    """

    def _preprocess(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        decoder_input: torch.Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        decoder_input, mtp_decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset = (
            super()._preprocess(
                input_ids=input_ids,
                position_ids=position_ids,
                decoder_input=decoder_input,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
            )
        )
        if (self.config.mtp_num_layers
                and self.config.enable_hyper_connections
                and mtp_decoder_input is not None
                and mtp_decoder_input.shape[-1] > self.config.hidden_size):
            mtp_decoder_input = mtp_decoder_input[..., :self.config.hidden_size]
        return decoder_input, mtp_decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset

    def _postprocess(self,
                     hidden_states,
                     input_ids,
                     position_ids,
                     labels,
                     rotary_pos_emb,
                     rotary_pos_cos,
                     rotary_pos_sin,
                     loss_mask=None,
                     decoder_input=None,
                     attention_mask=None,
                     inference_params=None,
                     packed_seq_params=None,
                     sequence_len_offset=None,
                     runtime_gather_output=None,
                     extra_block_kwargs=None,
                     inference_context=None,
                     **kwargs):
        if not self.post_process:
            if self.config.mtp_num_layers:
                if self.config.enable_hyper_connections:
                    n = self.config.num_residual_streams
                    s, b, c = decoder_input.shape
                    decoder_input = decoder_input.unsqueeze(2).expand(s, b, n, c).reshape(s, b, n * c)
                return torch.concat([hidden_states, decoder_input], dim=0)
            else:
                return hidden_states
        return super()._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
            **kwargs,
        )


class DeepseekV4VLMultimodalGPTModel(MultimodalGPTModel):
    """MultimodalGPTModel that uses DeepseekV4VLGPTModel as the language model,
    preserving DSpark hooks and other DSv4-specific behaviour.

    Overrides ``_patch_word_embeddings`` and ``forward`` to handle sentinel
    tokens (>= ``padded_vocab_size``) that DeepSeek-V4-VL uses for image
    placeholders.  These tokens are clamped to 0 before the embedding lookup
    so the ``VocabParallelEmbedding`` does not hit out-of-range indices.
    """
    language_model_cls = DeepseekV4VLGPTModel

    @contextmanager
    def _patch_word_embeddings(self, kwargs):
        origin_forward = VocabParallelEmbedding.forward

        def forward(_self, input_):
            reduce_scatter_embeddings = _self.reduce_scatter_embeddings
            _self.reduce_scatter_embeddings = False
            # DeepSeek-V4-VL uses sentinel tokens >= vocab_size for image
            # placeholders; clamp them to 0 before the lookup.
            input_ = torch.masked_fill(input_, (input_ < 0) | (input_ >= self.config.padded_vocab_size), 0)
            res = origin_forward(_self, input_)
            _self.reduce_scatter_embeddings = reduce_scatter_embeddings
            packed_seq_params = kwargs.get('packed_seq_params')
            if self.visual is not None:
                if self.config.language_model_only:
                    res = self.visual.get_inputs_embeds_language_model(res, **kwargs)
                else:
                    res = self.visual.get_inputs_embeds(res, **kwargs)
                kwargs.clear()
                if isinstance(res, dict):
                    inputs_embeds = res.pop('inputs_embeds')
                    kwargs.update(res)
                    res = inputs_embeds
            if self.config.context_parallel_size > 1:
                res = split_cp_inputs(res, getattr(packed_seq_params, 'cu_seqlens_q', None), 1)
            if reduce_scatter_embeddings:
                res = res.transpose(0, 1).contiguous()
                res = scatter_to_sequence_parallel_region(res, group=_self.tp_group)
            return res

        VocabParallelEmbedding.forward = forward
        try:
            yield
        finally:
            VocabParallelEmbedding.forward = origin_forward

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        decoder_input: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_params=None,
        packed_seq_params=None,
        runtime_gather_output: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Mask sentinel tokens before passing to language model.
        if self.visual is not None and not self.config.language_model_only and input_ids is not None:
            input_ids = self.visual.mask_input_ids(input_ids)
        return super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            runtime_gather_output=runtime_gather_output,
            **kwargs,
        )


class DeepseekV4LLoader(DeepseekV4Loader):
    """Loader for the VL variant; inherits all transformer/MTP spec logic
    from the text-only loader.  Overrides ``_replace_router`` to swap in
    ``DeepseekV4VLTopKRouter`` (which carries the ``vl_bias`` buffer).
    """
    model_cls = DeepseekV4VLMultimodalGPTModel

    def _replace_router(self, transformer_layer_spec, mlp_key='mlp'):
        from functools import partial
        for layer_spec in transformer_layer_spec.layer_specs:
            mlp_spec = getattr(layer_spec.submodules, mlp_key, None)
            if mlp_spec is not None:
                if isinstance(mlp_spec, partial):
                    mlp_submodules = mlp_spec.keywords.get('submodules')
                else:
                    mlp_submodules = getattr(mlp_spec, 'submodules', None)
                if getattr(mlp_submodules, 'router', None) in (McoreTopKRouter, TopKRouter):
                    mlp_submodules.router = DeepseekV4VLTopKRouter


register_model(
    ModelMeta(
        ModelType.deepseek_v4_vl,
        ['deepseek_v4_vl'],
        bridge_cls=DeepseekV4VLBridge,
        visual_cls=DeepSeekV4Vit,
        loader=DeepseekV4LLoader,
    ))
