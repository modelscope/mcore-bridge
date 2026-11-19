# Copyright (c) ModelScope Contributors. All rights reserved.
"""MiMo-V2 (XiaomiMiMo MiMo-V2.5 / V2.6-Flash / V2.6-Pro) support.

Hybrid full/SWA attention MoE with four quirks stock Megatron does not express: asymmetric
Q/K(192) vs V(128) head dims, per-layer KV-head counts, SWA-only learnable attention sinks,
and V scaled by ``attention_value_scale`` with a separate RoPE base per attention family.
The first three are per-layer config overrides in ``MiMoV2SelfAttention`` /
``MiMoV2TEDotProductAttention``; the dual RoPE lives in ``MiMoV2TextGPTModel``. MTP draft
layers are SWA + dense, so the loader swaps the decoder-last MoE spec for a dense one.

The official FP8/MXFP4 fused-qkv checkpoint is dequantized online while loading (shared math
from ``utils/dequantizer.py``; the MiMo-specific fused-qkv layout helpers live below). Plain
BF16 split-q/k/v checkpoints are also accepted. Vision uses the HF remote-code
``MiMoVisionTransformer``; audio is not supported.
"""
import copy
import dataclasses
import re
import torch
import torch.distributed as dist
from collections.abc import Mapping
from copy import deepcopy
from functools import partial
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_layer_specs
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.moe.router import TopKRouter as McoreTopKRouter
from transformers import PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from types import SimpleNamespace

try:
    from megatron.core.transformer.module import mark_keep_in_fp32
    _HAS_KEEP_IN_FP32 = True
except ImportError:
    # Megatron >= 0.20 helper; 0.18/0.19 Float16Module casts unconditionally and never reads the
    # raw attribute, so an FP32 router gate cannot be preserved there (the loader fails loudly).
    _HAS_KEEP_IN_FP32 = False

    def mark_keep_in_fp32(tensor):
        tensor.keep_in_fp32 = True
        return tensor


from mcore_bridge.bridge import MultimodalGPTBridge
from mcore_bridge.config import ModelConfig
from mcore_bridge.tuners import LoraParallelLinear
from mcore_bridge.utils import Fp8Dequantizer, dequant_mxfp4, get_logger
from mcore_bridge.utils.safetensors import LazyTensor

from ..constant import ModelType
from ..gpt_model import GPTModel
from ..mm_gpt_model import MultimodalGPTModel
from ..modules import TopKRouter, TransformerLayer
from ..register import ModelLoader, ModelMeta, register_model
from ..rope import get_rope_inv_freq
from .utils import HuggingFaceVit

logger = get_logger()

# MiMo checkpoint LAYOUT facts (the dequant math itself is shared in utils/dequantizer.py):
# qkv_proj is fused and interleaved across num_key_value_heads shards (shard i = [q_i | k_i | v_i]),
# blockwise-FP8 quantized per shard with exact rows/cols (the scale grid covers whole 128 blocks,
# edge blocks partial). The shard count is always num_key_value_heads, even for SWA/MTP layers
# whose KV heads differ. Splitting back into [q, k, v] needs the per-family head dims.
_FP8_BLOCK = 128


def _dequant_fused_qkv(weight: torch.Tensor, scale_inv: torch.Tensor, shards: int) -> torch.Tensor:
    """Each kv-head shard of a fused qkv_proj is block-quantized on its own, padded to whole blocks."""
    rows = weight.shape[0] // shards
    scale_rows = -(-rows // _FP8_BLOCK)
    assert scale_inv.shape[0] == shards * scale_rows, f'{weight.shape=} {scale_inv.shape=} {shards=}'
    dequantizer = Fp8Dequantizer((_FP8_BLOCK, _FP8_BLOCK))
    return torch.cat([
        dequantizer.convert(w, s).to(torch.bfloat16)
        for w, s in zip(weight.chunk(shards), scale_inv.split(scale_rows), strict=True)
    ])


def _split_fused_qkv(qkv: torch.Tensor, config: dict, is_swa: bool) -> list[torch.Tensor]:
    """Undo the kv-head interleave of a fused qkv_proj and return [q, k, v] in head order."""
    shards = config['num_key_value_heads']
    prefix = 'swa_' if is_swa else ''
    num_heads, kv_heads = config[f'{prefix}num_attention_heads'], config[f'{prefix}num_key_value_heads']
    assert num_heads % shards == 0 and kv_heads % shards == 0, f'{num_heads=} {kv_heads=} {shards=}'
    sizes = [
        num_heads // shards * config[f'{prefix}head_dim'],
        kv_heads // shards * config[f'{prefix}head_dim'],
        kv_heads // shards * config[f'{prefix}v_head_dim'],
    ]
    assert qkv.shape[0] == shards * sum(sizes), f'{qkv.shape=} does not match {shards=} x {sizes=}'
    return [part.reshape(-1, qkv.shape[1]) for part in qkv.view(shards, sum(sizes), -1).split(sizes, dim=1)]


class MiMoV2TopKRouter(TopKRouter):
    """Router whose gate keeps the checkpoint's storage dtype (FP32 on V2.5, BF16 on Flash/Pro).

    The bridge upcasts the gate to the source dtype at load time (``_set_router``), so no
    build-time checkpoint probe is needed and repo-id configs work too. TE's cuBLASLt GEMM
    rejects an FP32 weight with a BF16 input (``moe_router_dtype`` only sets the output dtype),
    so the gating input is upcast to match the weight.
    """

    def gating(self, input: torch.Tensor):
        if self.weight.dtype == torch.float32 and input.dtype != torch.float32:
            input = input.to(torch.float32)
        return super().gating(input)


def _qkv_group_sizes(*, heads_per_group: int, qk_dim: int, v_dim: int):
    """Rows of one KV group in the fused ``linear_qkv`` output: its query heads, one key, one value."""
    return [heads_per_group * qk_dim, qk_dim, v_dim]


def _split_qkv(qkv: torch.Tensor, *, dim: int, groups: int, heads_per_group: int, qk_dim: int, v_dim: int):
    """Split ``dim`` of a fused qkv into (query, key, value), each keeping a KV-group axis at ``dim``."""
    sizes = _qkv_group_sizes(heads_per_group=heads_per_group, qk_dim=qk_dim, v_dim=v_dim)
    dim %= qkv.dim()
    grouped = qkv.view(*qkv.shape[:dim], groups, sum(sizes), *qkv.shape[dim + 1:])
    return torch.split(grouped, sizes, dim=dim + 1)


def _sized_linear(build, *, input_size=None, output_size=None):
    """``build`` with the given input/output size in place of the one SelfAttention passes."""

    def build_sized(base_input_size, base_output_size, **kwargs):
        return build(
            base_input_size if input_size is None else input_size,
            base_output_size if output_size is None else output_size,
            **kwargs,
        )

    return build_sized


class MiMoV2Vit(HuggingFaceVit):
    # HF checkpoint keys are top-level ``visual.*`` (the merger lives inside the vision tower).
    module_mapping = {'visual': 'visual'}
    _vision_tower = ['visual']
    _aligner = ['visual.merger']

    @staticmethod
    def _vision_namespace(hf_config):
        # The shared embedding helper reads in_channels; HF's vision dict uses in_chans.
        vision_dict = dict(hf_config.vision_config)
        vision_dict.setdefault('in_channels', vision_dict.get('in_chans', 3))
        return SimpleNamespace(**vision_dict)

    def prepare_model(self, hf_config: PretrainedConfig):
        vision_cls = get_class_from_dynamic_module('modeling_mimo_v2.MiMoVisionTransformer', hf_config.name_or_path)
        self.visual = vision_cls(self._vision_namespace(hf_config)).to(self.config.params_dtype)
        # Official checkpoints (V2.5/Flash/Pro) ship no merger biases, but the remote code declares
        # them (LayerNorm/Linear default bias=True) and HF's own load leaves them randomly
        # initialized. Drop them so the tower matches the checkpoint key set exactly; bias=None
        # is numerically identical to zero bias.
        merger = self.visual.merger
        merger.ln_q.bias = None
        for module in merger.mlp:
            if isinstance(module, torch.nn.Linear):
                module.bias = None

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        # MiMo's audio entry points: mel features (tokenizer input) and the audio_codes/audio_embeds
        # the HF forward actually consumes (the swift template emits audio_codes). The shared vision
        # helper below only reads vision keys, so any audio key must fail loudly, not be dropped.
        if any(kwargs.get(key) is not None for key in ('input_features', 'audio_codes', 'audio_embeds')):
            raise NotImplementedError('MiMo-V2 audio input is not supported by mcore-bridge; '
                                      'use text/image/video only.')
        hf_config = copy.copy(self.hf_config)
        hf_config.vision_config = self._vision_namespace(self.hf_config)
        return self._hf_get_inputs_embeds(inputs_embeds, kwargs, self.visual, hf_config)


class MiMoV2SelfAttention(SelfAttention):
    """SelfAttention with V heads narrower than Q/K, per-layer KV-head counts and SWA sinks."""

    def __init__(self, config: ModelConfig, submodules, layer_number: int, *args, **kwargs):
        assert not config.attention_output_gate, 'MiMo-V2 has no attention output gate'
        # qk_clip reads linear_qkv as q + 2*kv rows of kv_channels each, which fails for a narrower V.
        assert not getattr(config, 'qk_clip', False), 'qk_clip does not support a V head narrower than Q/K'
        is_swa = getattr(config, '_mimo_force_swa', False) or bool(
            config.hf_config.hybrid_layer_pattern[layer_number - 1])
        self.is_swa = is_swa
        heads = config.num_attention_heads
        groups = config.swa_num_key_value_heads if is_swa else config.num_query_groups
        qk_dim, v_dim = config.kv_channels, config.v_head_dim
        group_sizes = _qkv_group_sizes(heads_per_group=heads // groups, qk_dim=qk_dim, v_dim=v_dim)
        qkv_out_dim = groups * sum(group_sizes)
        submodules = dataclasses.replace(
            submodules,
            linear_qkv=_sized_linear(submodules.linear_qkv, output_size=qkv_out_dim),
            linear_proj=_sized_linear(submodules.linear_proj, input_size=heads * v_dim),
        )
        # Per-layer copy, not a restored build-time override: TE re-reads config.softmax_type at
        # save/run time, so a temporary override would silently drop the SWA sink. Gate on the
        # checkpoint's sink flag too: 'learnable' makes TE append a sink column to the logits,
        # which a sink-less variant would never load (uninitialized bias in the softmax denominator).
        layer_config = copy.copy(config)
        layer_config.num_query_groups = groups
        has_sink = bool(getattr(config.hf_config, 'add_swa_attention_sink_bias', False))
        layer_config.softmax_type = 'learnable' if is_swa and has_sink else 'vanilla'
        super().__init__(layer_config, submodules, layer_number, *args, **kwargs)
        assert groups >= self.world_size, "TP must not exceed the layer's KV heads"
        self.val_hidden_size = v_dim
        if is_swa and getattr(self.core_attention, 'softmax_offset', None) is not None:
            # HF declares attention_sink_bias frozen; keep it frozen so reloads agree.
            self.core_attention.softmax_offset.requires_grad_(False)

    def get_query_key_value_tensors(self,
                                    hidden_states,
                                    key_value_states=None,
                                    output_gate=False,
                                    head_wise_gate=False,
                                    split_qkv=True):
        assert not (output_gate or head_wise_gate) and split_qkv, 'MiMo-V2 attention needs the split q/k/v path'
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        groups = self.num_query_groups_per_partition
        heads_per_group = self.num_attention_heads_per_partition // groups
        qk_dim, v_dim = self.hidden_size_per_attention_head, self.config.v_head_dim
        # [sq, b, ng*(np/ng*hn + hn + hv)] -> [sq, b, ng, np/ng*hn], [sq, b, ng, hn], [sq, b, ng, hv]
        query, key, value = _split_qkv(
            mixed_qkv, dim=-1, groups=groups, heads_per_group=heads_per_group, qk_dim=qk_dim, v_dim=v_dim)
        query = query.reshape(query.size(0), query.size(1), -1, qk_dim)
        return query, key, value


class MiMoV2TEDotProductAttention(TEDotProductAttention):
    """TE core attention with separate Q/K and V channels and V scaled by ``attention_value_scale``."""

    def __init__(self, config, layer_number, attn_mask_type, attention_type, attention_dropout=None, **kwargs):
        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
            k_channels=config.kv_channels,
            v_channels=config.v_head_dim,
            **kwargs,
        )
        self.attention_value_scale = config.attention_value_scale

    def forward(self, query, key, value, attention_mask, attn_mask_type, **kwargs):
        if self.attention_value_scale is not None:
            value = value * self.attention_value_scale
        return super().forward(query, key, value, attention_mask, attn_mask_type, **kwargs)


class MiMoV2TextGPTModel(GPTModel):
    """GPTModel with a second RoPE for the SWA layers (``swa_rope_theta``)."""

    def _set_inv_freq(self):
        super()._set_inv_freq()
        if getattr(self, 'rotary_pos_emb', None) is None:
            return
        # copy.copy gives the SWA rope its own lru_cache key (RotaryEmbedding.forward caches on self).
        self.swa_rotary_pos_emb = copy.copy(self.rotary_pos_emb)
        swa_config = copy.copy(self.config)
        swa_config.rotary_base = int(self.config.swa_rope_theta)
        new_inv_freq, _ = get_rope_inv_freq(swa_config)
        self.swa_rotary_pos_emb.inv_freq = new_inv_freq.to(self.rotary_pos_emb.inv_freq.device)

    def _get_rotary_pos_emb(self, decoder_input, position_ids, packed_seq_params, inference_context=None):
        if inference_context is not None and getattr(inference_context, 'use_flashinfer_fused_rope', False):
            raise ValueError('MiMo-V2 dual RoPE does not support flashinfer fused rope inference '
                             '(it needs precomputed cos/sin, not the per-family dict).')
        rotary_seq_len = RotaryEmbedding.get_rotary_seq_len(self, inference_context, self.decoder, decoder_input,
                                                            self.config, packed_seq_params)
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        full_rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        swa_rotary_pos_emb = self.swa_rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        rotary_pos_emb = {'full_attention': full_rotary_pos_emb, 'sliding_attention': swa_rotary_pos_emb}
        return rotary_pos_emb, None, None


class MiMoV2TransformerLayer(TransformerLayer):
    """Dispatches the dual RoPE dict to this layer's attention family."""

    def __init__(self, config, submodules, *args, **kwargs):
        super().__init__(config, submodules, *args, **kwargs)
        if self.is_mtp_layer:
            # MTP draft layers are SWA; their internal 1-based numbering must not index the decoder pattern.
            self.layer_type = 'sliding_attention'
        else:
            pattern = config.hf_config.hybrid_layer_pattern
            self.layer_type = 'sliding_attention' if pattern[self.layer_number - 1] else 'full_attention'

    def _forward_attention(self, hidden_states, **kwargs):
        rotary_pos_emb = kwargs.get('rotary_pos_emb')
        if isinstance(rotary_pos_emb, dict):
            kwargs['rotary_pos_emb'] = rotary_pos_emb[self.layer_type]
        return super()._forward_attention(hidden_states, **kwargs)


class MiMoV2MTPTransformerLayer(MiMoV2TransformerLayer):
    """MTP inner layer: force SWA (sink + KV heads) via a build-time config override, restored
    afterwards; window_attn_skip_freq=None makes every layer window attention."""

    def __init__(self, config, submodules, *args, **kwargs):
        orig_skip_freq = config.window_attn_skip_freq
        config.window_attn_skip_freq = None
        config._mimo_force_swa = True
        try:
            super().__init__(config, submodules, *args, **kwargs)
        finally:
            config.window_attn_skip_freq = orig_skip_freq
            del config._mimo_force_swa


class MiMoV2GPTModel(MultimodalGPTModel):
    language_model_cls = MiMoV2TextGPTModel


class _QuantizedStateDict(Mapping):
    """Lazy view of the official quantized checkpoint as the canonical BF16 split-q/k/v keys.

    Only canonical weight keys are exposed (no scale keys), so the base bridge takes its plain
    BF16 path; each tensor is read and dequantized on first access with the shared primitives
    from ``utils/dequantizer.py``.
    """

    _LAYER_RE = re.compile(r'^model\.layers\.(\d+)\.')
    _SCALE_SUFFIXES = ('weight_scale_inv', 'weight_scale')

    def __init__(self, state_dict: dict, hf_config, quant_config: dict):
        quant = quant_config or {}
        mxfp4_block = quant.get('mxfp4_block_size')
        # Two official layouts: FP8 dense + MXFP4 experts (Flash/Pro), or all-FP8 (V2.5).
        if (quant.get('quant_method') != 'fp8' or quant.get('fmt') != 'e4m3'
                or quant.get('weight_block_size') != [128, 128] or mxfp4_block not in (None, 32)
                or (mxfp4_block is None and quant.get('store_dtype') != 'fp8')):
            raise ValueError(f'MiMo-V2 online dequantization only supports the official layouts: '
                             f'FP8(e4m3, block [128,128]) + MXFP4(block 32) experts (Flash/Pro), '
                             f'or all-FP8(block [128,128]) (V2.5); got quantization_config={quant}')
        self._experts_mxfp4 = mxfp4_block == 32
        self._source = state_dict
        self._hf_config = hf_config
        self._qkv_cache = {}
        self._loaders = {}
        for key in state_dict:
            if key.endswith(self._SCALE_SUFFIXES):
                continue
            if key.endswith('self_attn.qkv_proj.weight'):
                match = self._LAYER_RE.match(key)
                # MTP layers use the SWA attention variant.
                is_swa = match is None or bool(hf_config.hybrid_layer_pattern[int(match.group(1))])
                for proj in 'qkv':
                    canonical = key.replace('qkv_proj', f'{proj}_proj')
                    self._loaders[canonical] = self._make_qkv_loader(key, is_swa, proj)
            elif key + '_scale_inv' in state_dict:
                self._loaders[key] = self._make_fp8_loader(key)
            elif key.endswith('.weight') and key.removesuffix('weight') + 'weight_scale' in state_dict:
                self._loaders[key] = self._make_mxfp4_loader(key)
            else:
                self._loaders[key] = self._make_passthrough_loader(key)
        # Fail loudly on a quantized-layout tensor whose scale key is absent, instead of
        # silently reading packed/FP8 bytes as BF16.
        for key in state_dict:
            if key.endswith(self._SCALE_SUFFIXES):
                continue
            if key.endswith('self_attn.qkv_proj.weight'):
                assert key + '_scale_inv' in state_dict, f'missing FP8 scale for {key}'
            elif re.search(r'\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$', key):
                # Experts follow the declared layout; a scale key of the other flavor means a
                # mixed/unknown layout and must fail loudly rather than dequantize with the wrong rule.
                base = key.removesuffix('weight')
                if self._experts_mxfp4:
                    assert base + 'weight_scale' in state_dict, f'missing MXFP4 scale for {key}'
                    assert base + 'weight_scale_inv' not in state_dict, f'unexpected FP8 scale for {key}'
                else:
                    assert base + 'weight_scale_inv' in state_dict, f'missing FP8 scale for {key}'
                    assert base + 'weight_scale' not in state_dict, f'unexpected MXFP4 scale for {key}'
            elif re.search(r'\.mlp\.(gate|up|down)_proj\.weight$', key):
                match = self._LAYER_RE.match(key)
                is_decoder_dense = match is not None and not hf_config.moe_layer_freq[int(match.group(1))]
                if is_decoder_dense or key.startswith('model.mtp.layers.'):
                    assert key + '_scale_inv' in state_dict, f'missing FP8 scale for {key}'

    def _make_passthrough_loader(self, key: str):

        def load():
            tensor = self._source[key].load()
            # A quantized-dtype tensor without a scale key would be silently cast to BF16 by
            # param.copy_ (raw FP8/packed bytes as values). Official passthrough tensors are
            # BF16/F32 only; anything else means an undeclared quantized tensor -> fail loudly.
            assert tensor.dtype in (torch.bfloat16, torch.float32, torch.float16), \
                f'{key}: quantized dtype {tensor.dtype} without a scale key'
            return tensor

        return load

    def _make_fp8_loader(self, key: str):

        def load():
            weight = self._source[key].load()
            scale_inv = self._source[key + '_scale_inv'].load()
            return Fp8Dequantizer((128, 128)).convert(weight, scale_inv).to(torch.bfloat16)

        return load

    def _make_mxfp4_loader(self, key: str):
        scale_key = key.removesuffix('weight') + 'weight_scale'

        def load():
            return dequant_mxfp4(self._source[key].load(), self._source[scale_key].load())

        return load

    def _make_qkv_loader(self, key: str, is_swa: bool, proj: str):
        fetched = set()

        def load():
            parts = self._qkv_cache.get(key)
            if parts is None:
                qkv = _dequant_fused_qkv(self._source[key].load(), self._source[key + '_scale_inv'].load(),
                                         self._hf_config.num_key_value_heads)
                parts = _split_fused_qkv(qkv, vars(self._hf_config), is_swa)
                self._qkv_cache[key] = parts
            fetched.add(proj)
            if len(fetched) == 3:
                # q/k/v are consumed consecutively per layer; drop the fused copy once all three
                # are read so the load peak stays at one layer instead of the whole decoder.
                del self._qkv_cache[key]
            return parts['qkv'.index(proj)]

        return load

    def __getitem__(self, key):
        return LazyTensor(loader=self._loaders[key])

    def __contains__(self, key):
        return key in self._loaders

    def __iter__(self):
        return iter(self._loaders)

    def __len__(self):
        return len(self._loaders)


class MiMoV2Bridge(MultimodalGPTBridge):
    # Text backbone at model.*, vision at top-level visual.*, MTP under model.mtp.layers.
    hf_layers_prefix = 'model.layers'
    hf_embed_key = 'model.embed_tokens.weight'
    hf_final_layernorm_key = 'model.norm.weight'
    hf_mtp_prefix = 'model.mtp.layers'
    hf_mtp_final_layernorm_key = 'final_layernorm.weight'

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        hf_config = config.hf_config
        # Quantization is decided by quantization_config alone: a stale 'fused_qkv' layout marker
        # on a split BF16 checkpoint must not route it through the dequantizing view. The markers
        # stay on the in-memory config (other consumers may read them); patch_export_hf_config
        # strips them on the exported copy so saved BF16 checkpoints are not re-read as quantized.
        self._quant_config = getattr(hf_config, 'quantization_config', None)
        self._quantized_input = bool(self._quant_config)
        self.hybrid_layer_pattern = [int(p) for p in hf_config.hybrid_layer_pattern]
        self.swa_attention_sink = bool(getattr(hf_config, 'add_swa_attention_sink_bias', False))
        if getattr(hf_config, 'add_full_attention_sink_bias', False):
            raise NotImplementedError('MiMo-V2 full-attention sink bias is not supported.')

    def patch_export_hf_config(self, hf_config):
        # Exported weights use the standard HF split q/k/v layout; the fused-layout marker
        # describes only the quantized source checkpoint and must not leak into the export.
        for cfg in (hf_config, getattr(hf_config, 'text_config', None)):
            if cfg is not None and hasattr(cfg, 'attention_projection_layout'):
                delattr(cfg, 'attention_projection_layout')

    def _convert(self, mg_models, hf_state_dict, hf_prefix: str, to_mcore: bool, tqdm_desc: str = 'Converting: '):
        if to_mcore and self._quantized_input and not self._peft_format:
            hf_state_dict = _QuantizedStateDict(hf_state_dict, self.config.hf_config, self._quant_config)
        yield from super()._convert(mg_models, hf_state_dict, hf_prefix, to_mcore, tqdm_desc)

    def _is_swa_layer(self, layer_idx: int) -> bool:
        if layer_idx == -1:  # MTP inner layer
            return True
        return bool(self.hybrid_layer_pattern[layer_idx])

    def _set_router(self, mg_mlp, hf_state_dict, to_mcore, **kwargs):
        if to_mcore:
            gate = hf_state_dict[self.hf_gate_key].load() if self.hf_gate_key in hf_state_dict else None
            if gate is not None and gate.dtype == torch.float32:
                # V2.5 stores the gate in FP32; the router builds in params_dtype, so upcast to
                # the source dtype instead of letting copy_ round it. Marked so Float16Module
                # keeps it FP32 whenever the wrap happens.
                if not _HAS_KEEP_IN_FP32:
                    raise ValueError('MiMo-V2 FP32 router gates need megatron-core>=0.20 (mark_keep_in_fp32); '
                                     '0.18/0.19 Float16Module would silently round the gate to BF16.')
                mg_mlp.router.weight.data = mg_mlp.router.weight.data.to(torch.float32)
                mark_keep_in_fp32(mg_mlp.router.weight)
            # Expert-subset models (fewer experts than the checkpoint) read experts.0..N-1; the
            # router must be truncated to the same prefix. Full-width checkpoints are a no-op.
            num_experts = self.config.num_moe_experts
            truncated = False
            for key, tensor in ((self.hf_gate_key, gate), (self.hf_expert_bias_key, None)):
                if key not in hf_state_dict:
                    continue
                if tensor is None:
                    tensor = hf_state_dict[key].load()
                if tensor.shape[0] > num_experts:
                    hf_state_dict = dict(hf_state_dict) if not truncated else hf_state_dict
                    hf_state_dict[key] = LazyTensor(tensor=tensor[:num_experts])
                    truncated = True
        return super()._set_router(mg_mlp, hf_state_dict, to_mcore, **kwargs)

    def _get_num_query_groups(self, layer_idx: int) -> int:
        if self._is_swa_layer(layer_idx):
            return self.config.swa_num_key_value_heads
        return self.config.num_query_groups

    def _set_qkv(self, mg_attn, hf_state_dict, to_mcore: bool, **kwargs):
        layer_idx = kwargs['layer_idx']
        groups = self._get_num_query_groups(layer_idx)
        if to_mcore:
            # The base path reshapes q/k/v by (groups, -1, hidden) before concat, so the
            # asymmetric V head dim needs no change here.
            return super()._set_qkv(mg_attn, hf_state_dict, to_mcore, num_query_groups=groups, **kwargs)
        # Export: slice the rank-local tensor by the per-group [hpg*qk_dim, qk_dim, v_dim] sizes
        # BEFORE any TP gather (mirrors the base export protocol, including the is_lora PP all_reduce).
        config = self.config
        qk_dim = config.kv_channels
        q_dim = (config.num_attention_heads // groups) * qk_dim
        kv_proj_list = ['k_proj', 'v_proj']
        is_lora = False if mg_attn is None else isinstance(mg_attn.linear_qkv, LoraParallelLinear) and self._peft_format
        is_lora = torch.tensor([is_lora], dtype=torch.bool, device='cuda')
        if self.pp_size > 1:
            dist.all_reduce(is_lora, group=self.pp_group)
        if is_lora:
            lora_A, _ = self._get_weight(
                None if mg_attn is None else mg_attn.linear_qkv.lora_A[self._adapter_name].weight.data,
                f'linear_qkv.lora_A.{self._adapter_name}.weight')
            lora_B, _ = self._get_weight(
                None if mg_attn is None else mg_attn.linear_qkv.lora_B[self._adapter_name].weight.data,
                f'linear_qkv.lora_B.{self._adapter_name}.weight')
            if lora_A is not None:
                self._peft_target_modules.update({'q_proj'} | set(kv_proj_list))
                for key in ['q_proj'] + kv_proj_list:
                    hf_state_dict[f'{key}.lora_A.weight'] = lora_A.clone()
                rank = lora_B.shape[-1]
                lora_B = lora_B.reshape((groups, -1, rank))
                hf_state_dict['q_proj.lora_B.weight'] = lora_B[:, :q_dim, :].reshape(-1, rank).clone()
                hf_state_dict['k_proj.lora_B.weight'] = lora_B[:, q_dim:q_dim + qk_dim, :].reshape(-1, rank).clone()
                hf_state_dict['v_proj.lora_B.weight'] = lora_B[:, q_dim + qk_dim:, :].reshape(-1, rank).clone()
        elif not self._peft_format:
            mg_attn_weight, scale_inv = self._get_weight(None if mg_attn is None else mg_attn.linear_qkv.weight.data,
                                                         'linear_qkv.weight')
            if mg_attn_weight is not None:
                mg_attn_weight = mg_attn_weight.reshape((groups, -1, config.hidden_size))
                hf_state_dict['q_proj.weight'] = mg_attn_weight[:, :q_dim, :].reshape(-1, config.hidden_size).clone()
                hf_state_dict['k_proj.weight'] = mg_attn_weight[:, q_dim:q_dim + qk_dim, :].reshape(
                    -1, config.hidden_size).clone()
                hf_state_dict['v_proj.weight'] = mg_attn_weight[:,
                                                                q_dim + qk_dim:, :].reshape(-1,
                                                                                            config.hidden_size).clone()
            if scale_inv is not None:
                raise ValueError('MiMo-V2 FP8 export is unsupported: the QK head dim (192) is not divisible '
                                 'by the FP8 block size (128). Export BF16.')
            del mg_attn_weight
        # Bias (MiMo sets add_qkv_bias/add_bias_linear False; mirrored for branch completeness).
        if (config.add_bias_linear or config.add_qkv_bias) and not self._peft_format:
            mg_attn_bias, _ = self._get_weight(None if mg_attn is None else mg_attn.linear_qkv.bias.data,
                                               'linear_qkv.bias')
            if mg_attn_bias is not None:
                mg_attn_bias = mg_attn_bias.reshape((groups, -1))
                hf_state_dict['q_proj.bias'] = mg_attn_bias[:, :q_dim].reshape(-1).clone()
                hf_state_dict['k_proj.bias'] = mg_attn_bias[:, q_dim:q_dim + qk_dim].reshape(-1).clone()
                hf_state_dict['v_proj.bias'] = mg_attn_bias[:, q_dim + qk_dim:].reshape(-1).clone()
        return hf_state_dict

    def _set_attn_state(self, mg_attn, hf_state_dict, hf_prefix: str, layer_idx: int, to_mcore: bool):
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        hf_state_dict.update(self._set_qkv(mg_attn, hf_state_dict, to_mcore, layer_idx=layer_idx))
        self._set_state_dict(mg_attn, 'linear_proj.weight', hf_state_dict, f'{self.hf_o_proj_key}.weight', to_mcore)
        # No qk_layernorm / linear_proj.bias; the SWA-only sink is HF's frozen attention_sink_bias.
        if self._is_swa_layer(layer_idx) and self.swa_attention_sink:
            self._set_state_dict(mg_attn, 'core_attention.softmax_offset', hf_state_dict, 'attention_sink_bias',
                                 to_mcore)
            if not to_mcore and 'attention_sink_bias' in hf_state_dict:
                # TE keeps softmax_offset in FP32 regardless of params_dtype; export in the model
                # dtype so a BF16 checkpoint does not mix in an FP32 tensor (load casts back exactly).
                hf_state_dict['attention_sink_bias'] = hf_state_dict['attention_sink_bias'].to(self.config.params_dtype)
        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _set_layer_mlp(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool, is_mtp: bool = False):
        if is_mtp:
            # The dense MTP draft names its pre-MLP norm pre_mlp_layernorm (decoder: post_attention).
            orig_key = self.hf_post_attention_layernorm_key
            self.hf_post_attention_layernorm_key = 'pre_mlp_layernorm.weight'
            try:
                return super()._set_layer_mlp(mg_layer, hf_state_dict, layer_idx, to_mcore, is_mtp=is_mtp)
            finally:
                self.hf_post_attention_layernorm_key = orig_key
        return super()._set_layer_mlp(mg_layer, hf_state_dict, layer_idx, to_mcore, is_mtp=is_mtp)


class MiMoV2Loader(ModelLoader):
    model_cls = MiMoV2GPTModel

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.context_parallel_size > 1:
            raise ValueError('MiMo-V2 does not support context_parallel_size>1; use context_parallel_size=1.')
        if config.fp8 or config.fp4:
            raise ValueError('MiMo-V2 does not support FP8/FP4: the QK head dim (192) is not divisible by the '
                             '128 block size. Use BF16.')
        min_kv_heads = min(config.num_query_groups, config.swa_num_key_value_heads)
        if config.tensor_model_parallel_size > min_kv_heads:
            raise ValueError(f'MiMo-V2 needs TP <= the smallest per-layer KV head count ({min_kv_heads}); '
                             f'got tensor_model_parallel_size={config.tensor_model_parallel_size}.')
        # The SWA sink is a per-layer learnable softmax_type override (global stays vanilla), so
        # swift's global flash check misses it and TE fails deep with no available backend.
        hf_config = getattr(config, 'hf_config', None)
        if getattr(hf_config, 'add_swa_attention_sink_bias', False):
            backend = getattr(config, 'attention_backend', None)
            if getattr(backend, 'name', backend) == 'flash':
                raise ValueError('MiMo-V2 SWA attention sinks need a learnable softmax_type, which the flash '
                                 'attention backend does not support; use attention_backend=fused (or auto).')
        # The dual-RoPE override returns a per-family dict and no precomputed cos/sin, which the
        # flash-decode / cuda-graph / flashinfer-fused-rope inference paths require.
        if getattr(config, 'flash_decode', False) or getattr(config, 'enable_cuda_graph', False):
            raise ValueError('MiMo-V2 does not support flash_decode/cuda-graph inference: the dual RoPE '
                             'provides no precomputed cos/sin. Use the standard inference path.')

    def _build_decoder_layer_specs(self):
        layer_specs = super()._build_decoder_layer_specs()
        for layer_spec in layer_specs:
            layer_spec.submodules.self_attention.module = MiMoV2SelfAttention
            layer_spec.submodules.self_attention.submodules.core_attention = MiMoV2TEDotProductAttention
            # Swap before the base _replace_router pass (it only replaces the stock class).
            # Always installed: the gating upcast is dtype-driven, a no-op for BF16 gates.
            mlp_spec = getattr(layer_spec.submodules, 'mlp', None)
            if isinstance(mlp_spec, partial):
                mlp_submodules = mlp_spec.keywords.get('submodules')
            else:
                mlp_submodules = getattr(mlp_spec, 'submodules', None)
            if getattr(mlp_submodules, 'router', None) is McoreTopKRouter:
                mlp_submodules.router = MiMoV2TopKRouter
        return layer_specs

    def _set_transformer_layer(self, transformer_layer_spec):
        super()._set_transformer_layer(transformer_layer_spec)
        for layer_spec in transformer_layer_spec.layer_specs:
            if layer_spec.module is TransformerLayer:
                layer_spec.module = MiMoV2TransformerLayer

    def _dense_mlp_submodules(self):
        # MTP drafts are dense even though the decoder's last layer is MoE.
        num_moe_experts = self.config.num_moe_experts
        self.config.num_moe_experts = None
        try:
            specs = get_gpt_decoder_layer_specs(
                self.config, use_transformer_engine=True, normalization=self.config.normalization)
            submodules = specs[0].submodules
            return submodules.mlp, submodules.pre_mlp_layernorm
        finally:
            self.config.num_moe_experts = num_moe_experts

    def _patch_mtp_block_spec(self, mtp_block_spec):
        if mtp_block_spec is None:
            return None
        # Upstream shares one spec object across MTP depths and with the decoder's last layer.
        mtp_block_spec = deepcopy(mtp_block_spec)
        dense_mlp, dense_pre_mlp_layernorm = self._dense_mlp_submodules()
        for layer_spec in mtp_block_spec.layer_specs:
            inner = layer_spec.submodules.mtp_model_layer
            inner.module = MiMoV2MTPTransformerLayer
            inner.submodules.mlp = dense_mlp
            inner.submodules.pre_mlp_layernorm = dense_pre_mlp_layernorm
        return mtp_block_spec


register_model(
    ModelMeta(
        ModelType.mimo_v2,
        ['mimo_v2'],
        bridge_cls=MiMoV2Bridge,
        visual_cls=MiMoV2Vit,
        loader=MiMoV2Loader,
        mtp_prefix='model.mtp.layers',
    ))
