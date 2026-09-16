# Copyright (c) ModelScope Contributors. All rights reserved.
"""DeepSeek-V4.1 on megatron-core's ``HybridModel`` (pipeline-parallel path).

The legacy :class:`DeepseekV41Loader` builds a ``GPTModel`` whose custom
``TransformerBlock`` owns the CSA2 / single-pass-mHC forward. Upstream refuses to
run that block under pipeline parallelism::

    # transformer_block.py:304-311
    if (config.pipeline_model_parallel_size > 1
            and config.experimental_attention_variant == "dsv4_hybrid"
            and config.dsv4_version == "v4.1"):
        raise ValueError("V4.1 pipeline parallelism requires HybridModel and its payload adapter")

so PP>1 requires the native ``HybridModel`` + ``CSA2HybridAdapter`` typed-payload path.

On ``HybridModel`` one *pattern symbol is one layer*: a GPT ``attn+mlp`` layer becomes
two hybrid layers -- an attention-only layer (symbol ``D``) followed by an MLP-only
layer (``E`` for MoE, ``-`` for dense). So a hybrid stack has ``2 * num_layers`` layers
and every per-layer config array that CSA2 indexes by ``layer_number - 1`` must be
re-expanded into this doubled index space (see :func:`derive_hybrid_layer_config`).

This module keeps the GPT loader available as a force-off regression baseline; the
hybrid path is now the default for every layout (plan step B5) after both were
validated to agree at iter-1 loss/grad and on the weight key ledger.
"""
import copy
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from tqdm import tqdm

from mcore_bridge.utils import is_master

from ..modules.engram import DeepseekV41Engram, DeepseekV41TransformerLayer
from ..rope import get_rope_inv_freq
from .deepseek_v41 import (CSA2Compressor, CSA2Indexer, DeepseekV41Bridge, DeepseekV41Loader,
                           DeepseekV41MultimodalGPTModel, DSv4HybridSelfAttention)

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
        delegating to the base wrapper forward (aggregation + fast-path attention). This reproduces
        the GPTModel golden path exactly. The inner ``DeepseekV41TransformerLayer`` keeps its
        ``engram`` module only so the bridge can load/export its weights; the base fast path never
        calls it, so there is no double add. Non-Engram layers keep the base wrapper untouched (this
        subclass is only swapped onto Engram-carrying wrappers, see
        :meth:`DeepseekV41HybridLoader._rewrap_engram_hyper_connection_layers`).

        The fast path is also invoked by the CUDA-graph capture body, which is out of scope for this
        change (plan: no CUDA Graph).
        """

        def forward(self, hidden_states, attention_mask=None, inference_context=None,
                    rotary_pos_emb=None, sequence_len_offset=None, packed_seq_params=None,
                    padding_mask=None, input_ids=None, mhc_recompute_manager=None,
                    mhc_state=None, **layer_kwargs):
            engram = getattr(self.inner_layer, 'engram', None)
            if engram is not None:
                if input_ids is None:
                    raise ValueError(
                        'DeepSeek-V4.1 hybrid Engram requires input token IDs on the layer forward.')
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
                **({'mhc_state': mhc_state} if mhc_state is not None else {}),
                **layer_kwargs,
            )

    class DeepseekV41HybridStackModel(HybridModel):
        """``HybridModel`` that splits PP / VPP stages on complete attention+FFN blocks.

        Upstream ``select_pipeline_segment`` (called inside ``HybridModel.__init__``,
        hybrid_model.py:265) handles the split, but for a pattern *without* ``|`` separators it
        (a) refuses VPP outright and (b) slices the ``2 * num_layers`` sublayers evenly, which
        cuts a ``D``/``E`` block across a stage boundary whenever ``2N // stages`` is odd. Either
        breaks :class:`DeepseekV41HybridBridge`, whose 1-HF-layer -> 2-hybrid-layer fan-out
        assumes the attention half (``2 * i``) and its MLP half (``2 * i + 1``) are co-resident.

        Mirroring GLM-5.3 (``Glm5NextHybridModel``), we pre-segment the *main* pattern on block
        boundaries into ``|``-delimited, PP*VPP-ordered stages before it reaches upstream, and
        assert after build that this rank holds whole blocks -- failing loudly instead of
        mis-mapping weights. The MTP suffix (B2) is still appended by the base resolver, so a
        segmented main becomes ``seg0|seg1|.../mtp``.
        """

        # B1 backbone is text-only, but ``deepseek_v41`` is a multimodal model_type, so the
        # trainer's ``is_multimodal`` path reads ``model.visual`` (expecting ``None`` for text,
        # like the GPT ``DeepseekV41MultimodalGPTModel``). Expose it so that guard short-circuits;
        # the real vision tower arrives with the multimodal wrapper in B4.
        visual = None

        # ``MultimodalGPTModel.forward`` (B4 wrapper) reads ``language_model.extra_forward_keys``
        # to forward a whitelist of extra kwargs into the decoder. ``McoreHybridModel`` has no such
        # attribute (it lives on the mcore-bridge ``GPTModel``, default ``[]``); expose the same
        # empty default so the wrapper treats the hybrid backbone exactly like the GPT one.
        extra_forward_keys: List[str] = []

        @staticmethod
        def _segment_main_pattern(config) -> Optional[str]:
            pattern = config.hybrid_layer_pattern
            if getattr(config, 'pipeline_model_parallel_layout', None) is not None:
                raise ValueError(
                    'DeepSeek-V4.1 hybrid splits pipeline stages by hybrid_layer_pattern, so '
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
                raise ValueError(
                    'DeepSeek-V4.1 hybrid needs at least one attention+FFN block per pipeline stage, '
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
                    raise ValueError(
                        'DeepSeek-V4.1 hybrid pipeline stage boundaries must fall on complete '
                        f'attention+FFN blocks, but this stage starts at sublayer {offset} and holds '
                        f'{count} sublayers (both must be even, since one block is two sublayers). '
                        'Leave num_layers_in_first_pipeline_stage / num_layers_in_last_pipeline_stage '
                        'unset for an even block-aligned split, or pass even values.')
            # ``HybridModel.forward`` builds no model-level RoPE for ``multi_latent_attention`` and
            # hard-sets ``rotary_pos_emb=None`` when calling the decoder. The reused DSv4 attention
            # (shared with the GPT path) instead expects the decoupled ``{'main', 'compress'}`` dict
            # that ``DeepseekV4GPTModel`` builds. Build the same two RoPE tables here and inject the
            # dict into the decoder via a forward pre-hook, keeping the attention numerically
            # identical to the GPTModel baseline. ``get_rotary_seq_len`` reads ``decoder.input_tensor``
            # when the local ``hidden_states`` is ``None``, so this also covers PP intermediate/last
            # stages.
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
            ``rope_layer_type`` (mirrors ``DeepseekV4GPTModel._get_rotary_pos_emb``)."""
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_context, self.decoder, transformer_input, self.config, packed_seq_params)
            packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
            return {
                'main': self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq),
                'compress': self.compress_rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq),
            }

        def _inject_dsv4_rotary_pos_emb(self, module, args, kwargs):
            if kwargs.get('rotary_pos_emb') is not None:
                return None
            transformer_input = kwargs.get('hidden_states')
            if transformer_input is None and args:
                transformer_input = args[0]
            kwargs['rotary_pos_emb'] = self._dsv4_rotary_pos_emb(
                transformer_input, kwargs.get('packed_seq_params'), kwargs.get('inference_context'))
            return args, kwargs

        # Visual kwargs are injected into the embeddings by the multimodal wrapper (B4) and then
        # cleared before the language model runs; the base HybridModel.forward never accepts them.
        # The B1 backbone is text-only, so strip them here. For a text batch the GPT wrapper's
        # ``get_inputs_embeds`` is a numeric no-op (``_zero_parameter_dependency`` adds ``0 *
        # vision_params``), so dropping them keeps parity with the GPTModel baseline.
        _visual_forward_keys = ('pixel_values', 'image_grid_thw', 'image_token_types', 'token_types')

        def forward(self, *args, **kwargs):
            # The B4 multimodal wrapper (``MultimodalGPTModel.forward``) always funnels the
            # decoder's extra kwargs through ``extra_block_kwargs`` -- the mcore-bridge ``GPTModel``
            # calling convention. Upstream ``HybridModel.forward`` has no such parameter (it threads
            # ``input_ids`` into the decoder itself, hybrid_model.py), so unpack the container here
            # and let the visual-key strip below drop anything the text backbone does not consume.
            extra_block_kwargs = kwargs.pop('extra_block_kwargs', None)
            if extra_block_kwargs:
                kwargs.update(extra_block_kwargs)
            if kwargs.get('pixel_values') is not None:
                raise NotImplementedError(
                    'DeepSeek-V4.1 hybrid (pipeline-parallel) path is text-only in B1; multimodal '
                    'inputs require the B4 multimodal wrapper.')
            for key in self._visual_forward_keys:
                kwargs.pop(key, None)
            return super().forward(*args, **kwargs)

    class DeepseekV41MultimodalHybridModel(DeepseekV41MultimodalGPTModel):
        """Multimodal wrapper (B4) hosting the PP-capable ``HybridModel`` backbone.

        ``MultimodalGPTModel`` consumes its ``language_model`` through a backbone-agnostic
        interface -- ``embedding(input_ids, position_ids)`` / ``vp_stage`` /
        ``share_embeddings_and_output_weights`` / ``extra_forward_keys`` /
        ``set_input_tensor`` / ``get_input_tensor`` / ``shared_embedding_or_output_weight`` plus
        the standard forward signature -- all of which :class:`DeepseekV41HybridStackModel`
        provides (``extra_forward_keys`` is added on it for exactly this). So the only change from
        the GPT :class:`DeepseekV41MultimodalGPTModel` is swapping the language-model class; the
        vision tower, image-embed injection (``_patch_word_embeddings``) and the vision/aligner
        weight bridging (``MultimodalGPTBridge._convert_pre_process``) are inherited unchanged.

        The wrapper injects image embeddings into the embedding output and clears the visual
        kwargs before the language model runs, so the hybrid backbone only ever sees a text batch
        (its ``forward`` strips ``_visual_forward_keys`` as a defensive backstop). The DSpark
        speculative-decoding helpers inherited from the GPT wrapper are inference-only and stay
        deferred on the hybrid path (see :meth:`DeepseekV41HybridLoader.build_model`).
        """

        language_model_cls = DeepseekV41HybridStackModel
else:
    DeepseekV41HyperConnectionHybridLayer = None
    DeepseekV41HybridStackModel = None
    DeepseekV41MultimodalHybridModel = None


@dataclass
class HybridLayerConfig:
    """Per-layer config re-expanded from GPT layer space into hybrid (2x) layer space.

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
    """Translate a GPT-space V4.1 config into the doubled hybrid layer space.

    Each GPT transformer layer ``i`` (0-based) becomes two hybrid layers:

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
        csa2_candidate_source_layer=(None if csa2_candidate_source_layer is None else
                                     2 * int(csa2_candidate_source_layer)),
    )


class DeepseekV41HybridLoader(DeepseekV41Loader):
    """Build DeepSeek-V4.1 on ``HybridModel`` (the PP-capable path).

    Reuses :class:`DeepseekV41Loader`'s Engram config resolution and MLA/CSA2 knowledge, but
    swaps the model class to the native ``HybridModel`` and rewrites the layer config into the
    doubled hybrid layer space (see :func:`derive_hybrid_layer_config`). The golden GPT
    ``DeepseekV41Loader`` is left untouched; this loader derives its own config copy so both
    paths can coexist in one process.

    B1 covers the text backbone; B3 adds the DSpark (``mtp.*``) draft stack on top (attached in
    :meth:`build_model`, mapped in :meth:`DeepseekV41HybridBridge._convert_additional_layers`).
    Autoregressive MTP (``mtp_num_layers`` / ``MultiTokenPredictionBlock``) does not apply to
    V4.1 -- its ``mtp.*`` checkpoint keys *are* DSpark -- so it stays disabled here. B4 wraps the
    backbone in :class:`DeepseekV41MultimodalHybridModel` (the vision tower + image-embed
    injection), mirroring the GPT :class:`DeepseekV41MultimodalGPTModel`.
    """

    model_cls = DeepseekV41MultimodalHybridModel

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
        # ``mtp.*`` checkpoint keys are the DSpark draft stack (attached in ``build_model``, B3).
        # Keep it disabled so no ``MultiTokenPredictionBlock`` is built.
        cfg.mtp_num_layers = None
        return cfg

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        # Build the spec from the *hybrid* config so the CSA2 attention sees the doubled-space
        # csa arrays. ``build_model`` caches it on ``self._hybrid_config`` first.
        spec = hybrid_dsv4_stack_spec(self._hybrid_config)
        # Apply the same fp8-parity module swaps the GPT loader uses, on the array-driven 'D'
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
        # it) rather than by editing per-layer specs like the GPT ``adapt_deepseek_v41_layer_specs``.
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
        # buffer and the V4.1 bridge fails to load ``gate.bias_vl`` (mirrors the GPT router swap).
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
            if (isinstance(layer, HyperConnectionHybridLayer)
                    and inner is not None and getattr(inner, 'engram', None) is not None):
                layer.__class__ = DeepseekV41HyperConnectionHybridLayer

    def build_model(self, pre_process=True, post_process=True, vp_stage: Optional[int] = None):
        """Build the multimodal wrapper around ``HybridModel``, skipping ``ModelLoader.build_model``'s
        GPT layer-spec post-processing (MLA / router / TransformerLayer substitution): a
        ``HybridStack`` spec exposes per-symbol submodules instead, and the DSv4 attention swap is
        done in ``get_transformer_layer_spec`` above.

        ``model`` is :class:`DeepseekV41MultimodalHybridModel` (vision tower + wrapper); the hybrid
        text backbone -- which owns the decoder / MoE / Engram layers the fix-ups below touch --
        is nested under ``model.language_model``, so they target that (matching the GPT wrapper,
        where the same fix-ups and DSpark live under ``language_model``)."""
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
        # DSpark (B3): the ``mtp.*`` draft stack is backbone-agnostic (plain
        # experimental-attention layers), so reuse the GPT loader's builder. It attaches to the
        # hybrid text backbone (``language_model.dspark``), mirroring the GPT wrapper. Inference-time
        # target-layer capture on HybridStack is deferred (it is not exercised by training / weight
        # round-trip, mirroring B1's deferral of ``allow_engram_inference``); the stack only needs
        # to exist so its parameters are loaded / saved via ``mtp.*``.
        self._attach_dspark(language_model, post_process, vp_stage=vp_stage)
        return model


class DeepseekV41HybridBridge(DeepseekV41Bridge):
    """Weight bridge for the ``HybridModel`` backbone (PP-capable path).

    The GPT bridge maps one HF layer onto one ``TransformerLayer`` that owns both attention
    and MLP. On ``HybridModel`` that layer is split in two (see
    :func:`derive_hybrid_layer_config`), so this bridge fans a single HF layer ``i`` out onto
    two hybrid layers:

    * hybrid layer ``2*i`` -- attention half: MLA / CSA2 state + ``attn_norm`` (+ Engram when
      ``i in engram_layer_ids``) + the ``hc_attn_*`` hyper-connection channel.
    * hybrid layer ``2*i + 1`` -- MLP half: MoE / dense state + ``ffn_norm`` + the ``hc_ffn_*``
      hyper-connection channel.

    When ``enable_hyper_connections`` is set each hybrid layer is wrapped in a
    ``HyperConnectionHybridLayer`` whose real payload lives under ``inner_layer`` and which owns
    a *single* ``hyper_connection`` module (the GPT layer instead carried two:
    ``self_attention_hyper_connection`` + ``mlp_hyper_connection``). The GPT
    ``hc_{attn,ffn}_*`` HF keys therefore split across the two wrappers.

    B1 handles the text backbone only. It treats the model as its own language model (the
    multimodal wrapper is B4) and skips MTP (B2). ``self.config`` is seen in two layer spaces
    depending on direction: on load it is the original GPT-space config (``num_layers == N``, no
    ``hybrid_layer_pattern``); on export it is the doubled hybrid megatron config used to build
    the model (``num_layers == 2 * N``, ``hybrid_layer_pattern`` populated). :meth:`_convert`
    normalizes this so it always iterates the decoder's ``2 * N`` hybrid layers.
    """

    @staticmethod
    def _lm(mg_model):
        """Resolve the language model. B1's HybridModel is text-only (no ``language_model``
        wrapper); B4 will nest it under a multimodal container."""
        language_model = getattr(mg_model, 'language_model', None)
        return mg_model if language_model is None else language_model

    @staticmethod
    def _num_hybrid_layers(config) -> int:
        """Decoder layer count in the doubled hybrid space, regardless of which layer space
        ``config`` is currently in.

        The two conversion entrypoints hand :meth:`_convert` a config in *different* spaces:
        load (``to_mcore=True``) passes the original GPT-space config (``num_layers == N``, no
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
        # ``DeepseekV41HybridLoader._engram_placement_layer_ids``), so map it back to HF space.
        return (engram.layer_number - 1) // 2

    def _set_word_embeddings(self, mg_model, hf_state_dict, to_mcore):
        # The base ``MultimodalGPTBridge`` resolves the language model with a raw
        # ``getattr(mg_model, 'language_model')``; route it through :meth:`_lm` so both the
        # multimodal wrapper (B4) and a bare backbone resolve correctly.
        self._set_state_dict(self._lm(mg_model), 'embedding.word_embeddings.weight', hf_state_dict, self.hf_embed_key,
                             to_mcore)

    def _convert_pre_process(self, mg_model, hf_state_dict, hf_prefix: str, to_mcore: bool):
        # First pipeline stage of a multimodal model: the vision tower + aligner + image_* markers
        # live on the wrapper (``mg_model.visual``). Reuse the GPT ``DeepseekV41Bridge`` pre-process
        # verbatim (``MultimodalGPTBridge`` word-embeddings + vision/aligner block, then the
        # image_start/end/newline markers); ``_set_word_embeddings`` above resolves the LM via
        # ``_lm``. ``super()`` here is ``DeepseekV41Bridge`` (MRO), matching the GPT path exactly.
        if getattr(mg_model, 'visual', None) is not None:
            return super()._convert_pre_process(mg_model, hf_state_dict, hf_prefix, to_mcore)
        # No vision tower on this rank (text-only backbone, or a non-first PP stage where the
        # wrapper built ``visual=None``): map only the word embeddings.
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        self._set_word_embeddings(mg_model, hf_state_dict, to_mcore)
        if to_mcore:
            return {}
        return self._add_prefix(hf_state_dict, hf_prefix)

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
        if has_hyper_connection:
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
        """Map the DSpark (``mtp.*``) draft stack (B3).

        The draft layers are plain experimental-attention ``TransformerLayer`` instances --
        identical in both paths -- so the base :meth:`DeepseekV41Bridge._convert_dspark_stack`
        mapping is reused verbatim; only where the stack lives differs. On the hybrid path it is
        attached to the model itself (no ``language_model`` wrapper, so use :meth:`_lm`) and only
        on the final pipeline stage, so non-last stages have nothing to convert (on load the base
        guard skips them; on export ``dspark`` is simply absent). MTP (``mtp_num_layers``) is
        skipped in :meth:`_convert` and does not apply to V4.1."""
        if not self.config.dspark_num_layers or (to_mcore and not is_pp_last_stage):
            return
        language_model = self._lm(mg_model)
        dspark = getattr(language_model, 'dspark', None)
        if dspark is None:
            if not is_pp_last_stage:
                # Export from a non-last PP stage: the draft stack lives on the final stage only.
                return
            raise RuntimeError('DSpark weights require the draft stack on the final pipeline stage.')
        yield from self._convert_dspark_stack(language_model, dspark, hf_state_dict, hf_prefix, to_mcore)

    def _convert(self, mg_models, hf_state_dict, hf_prefix: str, to_mcore: bool, tqdm_desc: str = 'Converting: '):
        """Backbone conversion with a 1->2 layer fan-out.

        Mirrors :meth:`GPTBridge._convert` but iterates the doubled hybrid layer space
        (``2 * num_layers``) and dispatches each hybrid layer to :meth:`_set_hybrid_layer_state`.
        MTP is intentionally skipped (B2); the multimodal-wrapper indirection is B4.
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
