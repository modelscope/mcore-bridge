# Copyright (c) ModelScope Contributors. All rights reserved.
"""DeepSeek-V4.1 on megatron-core's ``HybridModel`` (pipeline-parallel path).

The default :class:`DeepseekV41Loader` builds a ``GPTModel`` whose custom
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

This module keeps the GPT loader untouched (golden baseline) and adds the hybrid path
alongside it; both are validated to agree before the default is switched.
"""
import copy
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch
import torch.distributed as dist
from megatron.core import mpu
from tqdm import tqdm

from mcore_bridge.utils import is_master

from ..modules.engram import DeepseekV41Engram, DeepseekV41TransformerLayer
from .deepseek_v41 import (CSA2Compressor, CSA2Indexer, DeepseekV41Bridge, DeepseekV41Loader,
                           DSv4HybridSelfAttention)

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
        """Hyper-connection wrapper that keeps Engram inside the mHC layer delta.

        With ``enable_hyper_connections=True`` (always set for V4.1, see parser.py) HybridStack
        wraps every layer in :class:`HyperConnectionHybridLayer`. Its eager forward takes a
        *fast path* (:meth:`_call_inner_transformer_layer_without_local_bda`) that calls the
        inner layer's ``_forward_self_attention_output_with_bias`` directly. That method skips
        ``_forward_attention`` -> ``_maybe_apply_engram`` entirely, so it (a) never adds the
        Engram residual to the n-stream layer delta and (b) never forwards ``input_ids`` to the
        attention branch. Both silently drop Engram on the PP path.

        For the (few) layers that actually carry an Engram module we therefore decline the fast
        path by returning ``None``. :meth:`HyperConnectionHybridLayer.forward` then falls back to
        ``_call_inner_layer``, which runs the inner ``DeepseekV41TransformerLayer``'s full
        ``forward`` (the ``_DeepseekV41EngramLayerMixin`` stashes the inference context there) and
        computes ``layer_output - aggregated`` -- Engram delta included -- reproducing the
        GPTModel golden path exactly. Non-Engram layers keep the fast path untouched.

        This subclass adds no state and overrides one method, so it is applied by an in-place
        ``__class__`` swap on the already-built wrappers (see
        :meth:`DeepseekV41HybridLoader._rewrap_engram_hyper_connection_layers`) -- HybridStack
        hard-codes the wrapper class with no spec hook. The fast path is also invoked by the
        CUDA-graph capture body, which is out of scope for this change (plan: no CUDA Graph);
        returning ``None`` there would raise rather than miscompute.
        """

        def _call_inner_transformer_layer_without_local_bda(self, *args, **kwargs):
            if getattr(self.inner_layer, 'engram', None) is not None:
                return None
            return super()._call_inner_transformer_layer_without_local_bda(*args, **kwargs)
else:
    DeepseekV41HyperConnectionHybridLayer = None


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

    B1 covers the text backbone only. MTP (B2), DSpark capture (B3) and the multimodal wrapper
    (B4) are added on top; here MTP is disabled so the backbone can be aligned in isolation.
    """

    model_cls = HybridModel

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
        # MTP on HybridModel is B2 (its inner attention cannot use the CSA2 'D' symbol, which
        # rejects is_mtp_layer). Disable it for the B1 backbone-only alignment.
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
        return spec

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
        """Build via ``HybridModel``, skipping ``ModelLoader.build_model``'s GPT layer-spec
        post-processing (MLA / router / TransformerLayer substitution): a ``HybridStack`` spec
        exposes per-symbol submodules instead, and the DSv4 attention swap is done in
        ``get_transformer_layer_spec`` above."""
        self._hybrid_config = self._build_hybrid_config()
        model = self.model_cls(
            config=self._hybrid_config,
            transformer_layer_spec=self.get_transformer_layer_spec(vp_stage=vp_stage),
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )
        self._rewrap_engram_hyper_connection_layers(model)
        self._set_linear_is_expert(model)
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
    multimodal wrapper is B4) and skips MTP (B2). ``self.config`` stays in GPT layer space
    (``num_layers == N``); the model's decoder holds ``2 * N`` layers.
    """

    @staticmethod
    def _lm(mg_model):
        """Resolve the language model. B1's HybridModel is text-only (no ``language_model``
        wrapper); B4 will nest it under a multimodal container."""
        language_model = getattr(mg_model, 'language_model', None)
        return mg_model if language_model is None else language_model

    def _engram_hf_layer_id(self, engram):
        # Engram lives on the doubled-space attention layer ``2 * hf_id + 1`` (see
        # ``DeepseekV41HybridLoader._engram_placement_layer_ids``), so map it back to HF space.
        return (engram.layer_number - 1) // 2

    def _convert_pre_process(self, mg_model, hf_state_dict, hf_prefix: str, to_mcore: bool):
        # Text-only word embeddings; visual embeds (image_start/end/newline) are B4.
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        lm_model = self._lm(mg_model)
        self._set_state_dict(lm_model, 'embedding.word_embeddings.weight', hf_state_dict, self.hf_embed_key, to_mcore)
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
        # ``final_layernorm``); the block-level output hyper-connection head is unchanged.
        self._set_state_dict(lm_model, 'decoder.final_norm.weight', hf_state_dict, self.hf_final_layernorm_key,
                             to_mcore)
        for key in ['hc_head_base', 'hc_head_fn', 'hc_head_scale']:
            self._set_state_dict(lm_model, f'decoder.{key}', hf_state_dict, f'model.{key}', to_mcore)

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
        # HybridStack layer_number spans the doubled space (i + 1 + pp_offset), matching this
        # loop's hybrid index so the PP-availability window below stays correct.
        num_hybrid_layers = 2 * self.config.num_layers
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
