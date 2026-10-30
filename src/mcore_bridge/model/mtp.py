# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import copy
import inspect
import torch
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from megatron.core.transformer import multi_token_prediction as _mtp_mod
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionBlock as McoreMultiTokenPredictionBlock
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer as McoreMultiTokenPredictionLayer
from megatron.core.transformer.multi_token_prediction import get_mtp_layer_offset
from megatron.core.transformer.multi_token_prediction import process_mtp_loss as _upstream_process_mtp_loss
from megatron.core.transformer.multi_token_prediction import roll_tensor as _upstream_roll_tensor
from typing import Any, Callable, Dict, List, Optional, Tuple

from mcore_bridge.utils import get_logger

logger = get_logger()

# MCore divergence markers:
#   COMPAT(dev)  only needed for the unreleased Megatron-LM dev branch; the release
#                path is unconditional.
#   SEMANTICS    deliberate bridge behavior kept for legacy parity; not a bug.
#   UPSTREAM     workaround for an upstream defect; drop once upstream fixes it.

# `process_mtp_loss` gained `tp_group`/`input_ids` in 0.19 and more on dev. The bridge's
# single call site passes them unconditionally, so the accepted set is probed once here.
_PROCESS_MTP_LOSS_PARAMS = tuple(inspect.signature(_upstream_process_mtp_loss).parameters)


@dataclass(frozen=True)
class MtpCapabilities:
    """Signatures of the installed upstream MTP layer/block.

    Releases (0.18/0.19) and dev disagree on what the layer forward accepts and
    returns, so the bridge probes the installed code instead of branching on a
    version number.
    """

    has_detach_heads: bool = False

    layer_forward_params: Tuple[str, ...] = ()
    layer_returns_padding_mask: bool = False

    checkpointed_forward_accepts_padding_mask: bool = False

    block_forward_params: Tuple[str, ...] = ()

    has_roll_context: bool = False
    has_mhc_multistream: bool = False

    fingerprint: str = ''

    @property
    def needs_padding_recompute_fix(self) -> bool:
        # 0.18.x _checkpointed_forward has no padding_mask parameter.
        return (self.layer_returns_padding_mask and not self.checkpointed_forward_accepts_padding_mask)


def _params(fn) -> Tuple[str, ...]:
    if fn is None:
        return ()
    try:
        return tuple(inspect.signature(fn).parameters)
    except (ValueError, TypeError):
        return ()


def _accepts_var_keywords(fn) -> bool:
    try:
        return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(fn).parameters.values())
    except (ValueError, TypeError):
        return False


@lru_cache(maxsize=1)
def get_mtp_capabilities() -> MtpCapabilities:
    from megatron.core.transformer.transformer_config import TransformerConfig
    tc = TransformerConfig.__dataclass_fields__

    layer_fwd = _params(McoreMultiTokenPredictionLayer.forward)
    ckpt = _params(getattr(McoreMultiTokenPredictionLayer, '_checkpointed_forward', None))
    block_fwd = _params(McoreMultiTokenPredictionBlock.forward)

    caps = MtpCapabilities(
        has_detach_heads='mtp_detach_heads' in tc,
        layer_forward_params=layer_fwd,
        layer_returns_padding_mask='padding_mask' in layer_fwd,
        checkpointed_forward_accepts_padding_mask='padding_mask' in ckpt,
        block_forward_params=block_fwd,
        has_roll_context=hasattr(_mtp_mod, 'MTPSequenceRollContext'),
        has_mhc_multistream='mhc_multistream' in block_fwd,
        fingerprint=(f'pm={"padding_mask" in layer_fwd} ckpt_pm={"padding_mask" in ckpt} '
                     f'roll_ctx={hasattr(_mtp_mod, "MTPSequenceRollContext")} '
                     f'mhc={"mhc_multistream" in block_fwd}'),
    )
    logger.debug(f'MTP capabilities: {caps.fingerprint}')
    return caps


def roll_single(tensor: torch.Tensor,
                shifts: int = -1,
                dims: int = -1,
                cp_group=None,
                packed_seq_params=None) -> torch.Tensor:
    """Roll one tensor via whichever upstream API is installed.

    COMPAT(dev): MTPSequenceRollContext returns a list; releases return
    ``(tensor, ntok)``.
    """
    if get_mtp_capabilities().has_roll_context:
        out = _upstream_roll_tensor([tensor],
                                    shifts=shifts,
                                    dims=dims,
                                    cp_group=cp_group,
                                    packed_seq_params=packed_seq_params)
        return out[0] if isinstance(out, (list, tuple)) else out
    out, _ = _upstream_roll_tensor(
        tensor, shifts=shifts, dims=dims, cp_group=cp_group, packed_seq_params=packed_seq_params)
    return out


def get_physical_mtp_count(model) -> int:
    """Physical MTP layers: 1 under repeated/shared weights, else the logical depth."""
    mtp = getattr(model, 'mtp', None)
    return len(mtp.layers) if mtp is not None else 0


def get_physical_mtp_depth(config) -> int:
    """Physical MTP layers implied by a config, before the model is built."""
    depth = getattr(config, 'mtp_num_layers', None) or 0
    repeated = getattr(config, 'mtp_use_repeated_layer', False) or getattr(config, 'mtp_shared_weights', False)
    return 1 if depth and repeated else depth


class _PrecomputedEmbeddingStub:
    """Stand-in embedding when the bridge already holds a precomputed (multimodal)
    decoder input: skips the vocab lookup (placeholder ids may be negative) while
    keeping the module protocol upstream relies on -- callable returning a shaped
    Tensor, plus attribute delegation (dev probes ``add_position_embedding``).
    """

    def __init__(self, embedding, placeholder):
        self._embedding = embedding
        self._placeholder = placeholder

    def __call__(self, *args, **kwargs):
        return self._placeholder

    def __getattr__(self, name):
        return getattr(self._embedding, name)


class _TensorIdx:
    """Schema marker: position of a tensor in the flattened positional list."""

    __slots__ = ('idx', )

    def __init__(self, idx):
        self.idx = idx


def _flatten(obj, tensors):
    """Replace Tensor leaves with :class:`_TensorIdx` markers, appending the
    tensors to *tensors*.  Returns a new schema (the input is never mutated)."""
    if torch.is_tensor(obj):
        tensors.append(obj)
        return _TensorIdx(len(tensors) - 1)
    if isinstance(obj, dict):
        return {k: _flatten(v, tensors) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return type(obj)(_flatten(v, tensors) for v in obj)
    return obj


def _unflatten(schema, tensors):
    """Rebuild the structure from a schema and the staged tensor list (fresh
    containers; the tensors differ between forward and the recompute replay)."""
    if isinstance(schema, _TensorIdx):
        return tensors[schema.idx]
    if isinstance(schema, dict):
        return {k: _unflatten(v, tensors) for k, v in schema.items()}
    if isinstance(schema, (tuple, list)):
        return type(schema)(_unflatten(v, tensors) for v in schema)
    return schema


class MultiTokenPredictionLayer(McoreMultiTokenPredictionLayer):
    """Upstream MTP layer + bridge input semantics.

    Overrides ``__init__`` (eh_proj FP8 exclusion, fused-RoPE disable, extras
    wrapper), ``_get_embeddings`` (precomputed decoder input), ``forward``
    (depth-shifted RoPE, kwargs filtering) and ``_checkpointed_forward``
    (bridge-owned only where the upstream contract is broken).
    """

    # Per-forward bridge state; _pending_extras is re-set from the checkpoint
    # closure during recompute.
    _pending_decoder_input: Optional[torch.Tensor] = None
    _pending_depth: int = 0
    _pending_extras: Optional[Dict[str, Any]] = None
    _bridge_block: Optional['MultiTokenPredictionBlock'] = None

    def __init__(self, config, submodules, *args, **kwargs):
        # SEMANTICS: keep eh_proj out of both fp8 parameter init and fp8 autocast,
        # matching the legacy engine. The module is rebuilt outside fp8_model_init
        # so its weights stay BF16; FP8 numeric validation is still pending.
        need_eh_fix = (getattr(submodules, 'eh_proj', None) is not None and getattr(config, 'fp8_param', False))
        if need_eh_fix:
            saved_eh = submodules.eh_proj
            from megatron.core.transformer.identity_op import IdentityOp
            submodules.eh_proj = IdentityOp
        try:
            super().__init__(config, submodules, *args, **kwargs)
        finally:
            if need_eh_fix:
                submodules.eh_proj = saved_eh

        if need_eh_fix:
            import transformer_engine
            from megatron.core.transformer.spec_utils import build_module
            tp_grp = getattr(self, 'tp_group', None)
            extra = {'tp_group': tp_grp} if tp_grp is not None else {}
            with transformer_engine.pytorch.fp8_model_init(enabled=False):
                self.eh_proj = build_module(
                    saved_eh,
                    self.config.hidden_size * 2,
                    self.config.hidden_size,
                    config=self.config,
                    init_method=self.config.init_method,
                    gather_output=False,
                    bias=False,
                    skip_bias_add=False,
                    is_expert=False,
                    tp_comm_buffer_name='mtp_eh_proj',
                    **extra)
            # SEMANTICS: operation-precision half of the exception. Wrapping the
            # forward (not just init) also covers the recompute replay; quantizing
            # here would contradict the BF16 parameter copy built above.
            eh_orig_forward = self.eh_proj.forward

            def _eh_forward_no_fp8(*a, **kw):
                with transformer_engine.pytorch.fp8_autocast(enabled=False):
                    return eh_orig_forward(*a, **kw)

            self.eh_proj.forward = _eh_forward_no_fp8

        # Upstream does not shift RoPE per logical depth, so the bridge replaces
        # the rotary table (_shift_rope). Fused paths derive cos/sin themselves
        # from the sequence length (e.g. MLA get_cached_cos_sin) and would ignore
        # the shifted table; disabling fusion forces the table-consuming path.
        # Copy the config so the backbone keeps fusion enabled.
        inner = self.mtp_model_layer
        attn = getattr(inner, 'self_attention', None) if inner else None
        if attn is not None and hasattr(attn, 'config'):
            attn.config = copy.copy(attn.config)
            attn.config.apply_rope_fusion = False

        # Bridge extras (gemma4 shared_kv_states / per_layer_inputs, ...) are
        # injected here instead of being dropped at the upstream signature
        # boundary; _pending_extras is read at call time so the checkpoint
        # closure restores the per-microbatch value under recompute.
        if inner is not None:
            inner_orig_forward = inner.forward
            layer_ref = self

            def _inner_forward_with_extras(*a, **kw):
                extras = layer_ref._pending_extras
                if extras:
                    # Model-declared inputs win over values the upstream forward passes
                    # positionally by default (e.g. qwen4_exp's per-depth input_ids).
                    kw = {**kw, **extras}
                return inner_orig_forward(*a, **kw)

            inner.forward = _inner_forward_with_extras

    def _get_inner_layer_kwargs(self, input_ids, position_ids):
        """Model-specific rolled inputs the inner transformer layer needs per depth."""
        return {}

    def forward(self, *args, **kwargs):
        caps = get_mtp_capabilities()

        # Two producers, one 1-based convention:
        #   _bridge_depth -- bridge-owned loop (_forward_tracked), already 1-based;
        #   roll_depth    -- upstream's native loop (dev), 0-based look-ahead.
        # Both mean "how many tokens this depth looks ahead", i.e. exactly the
        # number of single-step rolls _shift_rope/_get_embeddings must apply.
        depth = kwargs.pop('_bridge_depth', None)
        if depth is None:
            roll_depth = kwargs.get('roll_depth')
            depth = (roll_depth + 1) if roll_depth is not None else 0

        block = self._bridge_block
        self._pending_decoder_input = getattr(block, '_bridge_decoder_input', None)
        self._pending_extras = getattr(block, '_bridge_extras', None)
        self._pending_depth = depth

        # Depth-shifted RoPE, computed from the ORIGINAL (block-entry)
        # position_ids: composing `depth` single-step rolls is correct whether or
        # not the installed upstream pre-rolls position_ids per layer.
        rope = kwargs.get('rotary_pos_emb')
        if rope is not None and depth > 0:
            orig_pos = getattr(block, '_bridge_position_ids', None)
            if orig_pos is None:
                orig_pos = kwargs.get('position_ids')
            kwargs['rotary_pos_emb'] = self._shift_rope(rope, depth, orig_pos, kwargs.get('packed_seq_params'))

        # Filter to upstream-accepted params; the rest become bridge extras for
        # the inner TransformerLayer -- never silently dropped.
        allowed = set(caps.layer_forward_params)
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        unexpected = {k: v for k, v in kwargs.items() if k not in allowed}
        if unexpected:
            self._pending_extras = {**(self._pending_extras or {}), **unexpected}
        # Model-specific per-depth inputs for the inner layer go through the extras
        # channel: it is staged for the recompute replay and set per microbatch.
        inner_kwargs = self._get_inner_layer_kwargs(kwargs.get('input_ids'), kwargs.get('position_ids'))
        if inner_kwargs:
            self._pending_extras = {**(self._pending_extras or {}), **inner_kwargs}

        try:
            return McoreMultiTokenPredictionLayer.forward(self, *args, **filtered)
        finally:
            self._pending_decoder_input = None
            self._pending_extras = None
            self._pending_depth = 0

    def _get_embeddings(self, *args, **kwargs):
        caps = get_mtp_capabilities()
        pre = self._pending_decoder_input
        if pre is not None and 'embedding' in kwargs:
            # Skip the vocab lookup (placeholder ids may be negative) while keeping
            # the upstream embedding protocol: shaped placeholder + attribute probes.
            hs = kwargs.get('hidden_states')
            placeholder = hs.detach() if torch.is_tensor(hs) else None
            kwargs['embedding'] = _PrecomputedEmbeddingStub(kwargs['embedding'], placeholder)
        result = McoreMultiTokenPredictionLayer._get_embeddings(self, *args, **kwargs)

        # Upstream returns 5 values on releases and 6 on dev (mtp_input_mask at
        # index 3). Keep the tuple shape intact so the caller's unpacking matches.
        values = list(result)
        if len(values) not in (5, 6):
            raise ValueError(f'Unexpected _get_embeddings result length: {len(values)}')
        mtp_input_mask = values[3] if len(values) == 6 else None
        dec = values[-2]

        if pre is not None:
            from megatron.core.tensor_parallel.mappings import (gather_from_sequence_parallel_region,
                                                                scatter_to_sequence_parallel_region)
            sp = (self.config.sequence_parallel and self.config.tensor_model_parallel_size > 1)
            rolled = pre
            if sp:
                rolled = gather_from_sequence_parallel_region(rolled, tensor_parallel_output_grad=False)
            packed = kwargs.get('packed_seq_params')
            depth = max(self._pending_depth, 1)
            # Composed single-step rolls: upstream packed/CP helpers only support
            # shifts == -1; composition reproduces the legacy per-layer semantics.
            for _ in range(depth):
                rolled = roll_single(
                    rolled.transpose(0, 2), shifts=-1, dims=-1, cp_group=self.cp_group,
                    packed_seq_params=packed).transpose(0, 2)
            rolled = rolled.contiguous()
            if sp:
                rolled = scatter_to_sequence_parallel_region(rolled)
            dec = rolled
            if mtp_input_mask is not None:
                valid_decoder_input = mtp_input_mask.transpose(0, 1).unsqueeze(-1)
                dec = torch.where(valid_decoder_input, dec, dec.detach())
            # Upstream detaches the embedding-derived decoder input before this
            # replacement; mirror it so the flag is not bypassed on this path.
            if caps.has_detach_heads and getattr(self.config, 'mtp_detach_heads', False):
                dec = dec.detach()

        # Selective detach
        if getattr(self.config, 'mtp_decoder_input_detach', False):
            dec = dec.detach()

        values[-2] = dec
        return tuple(values)

    def _shift_rope(self, rope, depth: int, position_ids=None, packed_seq_params=None):
        """Shift RoPE by cumulative logical depth (matching the legacy per-layer
        semantics, independent of the upstream position-rolling protocol):

        * packed + rope: compose ``depth`` single-step rolls of the ORIGINAL
          block-entry position_ids -- the only shift upstream packed/CP helpers
          support -- and index the ORIGINAL rope table;
        * mrope / non-packed: roll the table cyclically along the sequence dim.
        """
        packed = (packed_seq_params is not None and getattr(packed_seq_params, 'qkv_format', None) == 'thd')
        is_rope = getattr(self.config, 'position_embedding_type', 'rope') == 'rope'

        if packed and is_rope and position_ids is not None:
            rolled = position_ids
            for _ in range(depth):
                rolled = roll_single(
                    rolled, shifts=-1, dims=-1, cp_group=self.cp_group, packed_seq_params=packed_seq_params)
            assert rolled.shape[0] == 1, (f'packed rope expects batch-1 position_ids, got {tuple(position_ids.shape)}')
            if isinstance(rope, dict):
                return {k: v[rolled[0]] for k, v in rope.items()}
            return rope[rolled[0]]

        if isinstance(rope, dict):
            return {k: torch.roll(v, shifts=-depth, dims=0) for k, v in rope.items()}
        return torch.roll(rope, shifts=-depth, dims=0)

    def _checkpointed_forward(self, *args, **kwargs):
        caps = get_mtp_capabilities()
        has_containers = any(isinstance(v, (dict, list, tuple)) for v in kwargs.values())
        if self._pending_extras or caps.needs_padding_recompute_fix or has_containers:
            # Bridge-owned when (a) extras must keep their graph boundary instead of
            # being closure-captured through recompute, (b) 0.18.x lacks the
            # padding_mask parameter, or (c) a container (dict RoPE) would otherwise
            # be passed positionally into save_for_backward, which takes Tensor/None.
            return self._bridge_checkpoint(self._proj_and_transformer_layer, kwargs, self._pending_extras)
        return McoreMultiTokenPredictionLayer._checkpointed_forward(self, *args, **kwargs)

    def _bridge_checkpoint(self, forward_func, kwargs: dict, extras: Optional[dict]):
        """Bridge-owned activation checkpoint (mirrors ``transformer_block``):

        * Tensor leaves -- including those nested in dict/list kwargs (dict RoPE)
          and in extras -- are flattened into the POSITIONAL checkpoint inputs,
          the only channel that stages values, giving a differentiable extra a
          proper graph boundary instead of dragging its producer through recompute;
        * schemas and non-tensor objects (``PackedSeqParams``, ``InferenceParams``,
          dev roll context, ``None``) stay closure-captured, bound to THIS
          microbatch when the replay runs during backward;
        * fp8/fp4 route through ``te_checkpoint``.
        """
        from megatron.core import parallel_state, tensor_parallel
        from megatron.core.enums import Fp8Recipe
        from megatron.core.fp8_utils import get_fp8_context

        flat_tensors: List[torch.Tensor] = []
        kwargs_schema = _flatten(kwargs, flat_tensors)
        has_extras = bool(extras)
        extras_schema = _flatten(dict(extras), flat_tensors) if has_extras else None
        accepts_var_kw = _accepts_var_keywords(forward_func)
        accepted = set(_params(forward_func))

        def custom_forward(*staged):
            # Rebuild from `staged`: live tensors on forward, detached copies on
            # replay -- that is what keeps producer graphs out of the recompute.
            call_kw = _unflatten(kwargs_schema, staged)
            if not accepts_var_kw:
                call_kw = {k: v for k, v in call_kw.items() if k in accepted}
            saved_extras = self._pending_extras
            if has_extras:
                self._pending_extras = _unflatten(extras_schema, staged)
            try:
                return forward_func(**call_kw)
            finally:
                self._pending_extras = saved_extras

        # Only fp8 + delayed scaling needs the context active at te_checkpoint
        # entry (TE samples is_fp8_enabled() to gate the phase-1 amax stash).
        if self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed:
            outer_quantization_context = get_fp8_context(self.config)
        else:
            outer_quantization_context = nullcontext()

        def checkpoint_handler():
            # fp4 rides TE's fp8_autocast, so fp8/fp4 recompute both use te_checkpoint.
            if self.config.fp8 or getattr(self.config, 'fp4', False):
                from megatron.core.extensions.transformer_engine import te_checkpoint
                return te_checkpoint(
                    custom_forward,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    *flat_tensors,
                )
            return tensor_parallel.checkpoint(custom_forward, self.config.distribute_saved_activations, *flat_tensors)

        if self.config.recompute_method == 'uniform':
            assert self.config.recompute_num_layers == 1, 'recompute_num_layers must be 1 for MTP recompute'
            with outer_quantization_context:
                return checkpoint_handler()
        elif self.config.recompute_method == 'block':
            # TODO: implement block-based recompute for MTP
            warnings.warn("recompute_method == 'block' is not supported for MTP yet. Skipping recompute.")
            return custom_forward(*flat_tensors)
        raise ValueError('Invalid activation recompute method.')


class MultiTokenPredictionBlock(McoreMultiTokenPredictionBlock):
    """Upstream MTP block + per-depth bridge parameter injection.

    Per-forward bridge state lives on block attributes (no upstream block/layer
    forward takes ``**kwargs``) and is read by :class:`MultiTokenPredictionLayer` through a
    back-reference. Releases (0.18/0.19) get the manual per-depth loop; on dev
    (native ``roll_depth`` / ``mhc_multistream``) the loop is delegated to upstream.
    """

    # Per-forward bridge state, consumed by MultiTokenPredictionLayer via _bridge_block.
    _bridge_decoder_input: Optional[torch.Tensor] = None
    _bridge_extras: Optional[Dict[str, Any]] = None
    _bridge_position_ids: Optional[torch.Tensor] = None

    def __init__(self, config, spec, *args, **kwargs):
        super().__init__(config, spec, *args, **kwargs)
        for layer in self.layers:
            # Bypass nn.Module.__setattr__: a child registration would create a
            # block<->layer cycle that breaks children()/_apply/state_dict.
            layer.__dict__['_bridge_block'] = self

    def forward(self, input_ids, position_ids, hidden_states, attention_mask, **kwargs):
        if kwargs.get('inference_params') is not None:
            raise NotImplementedError('MTP inference is not supported by mcore-bridge')
        caps = get_mtp_capabilities()

        decoder_input = kwargs.pop('decoder_input', None)
        mhc = kwargs.pop('mhc_multistream', None)
        self._bridge_decoder_input = decoder_input
        # Block-entry position_ids for the RoPE shift (see MultiTokenPredictionLayer.forward).
        self._bridge_position_ids = position_ids
        try:
            # dev with native mhc + roll_depth: upstream drives the loop.
            if caps.has_roll_context and caps.has_mhc_multistream:
                return super().forward(input_ids, position_ids, hidden_states, attention_mask,
                                       **self._split_dev_kwargs(kwargs, mhc))

            # All other versions: manual depth tracking
            return self._forward_tracked(
                input_ids, position_ids, hidden_states, attention_mask, mhc_multistream=mhc, **kwargs)
        finally:
            self._bridge_decoder_input = None
            self._bridge_extras = None
            self._bridge_position_ids = None

    def _split_dev_kwargs(self, kwargs: dict, mhc) -> dict:
        """COMPAT(dev): route kwargs for the dev block forward. Keys the dev layer does not
        declare (bridge extras such as ``shared_kv_states``) are moved to the
        extras channel instead of being splatted, which would raise TypeError.
        """
        caps = get_mtp_capabilities()
        block_ok = set(caps.block_forward_params)
        layer_ok = set(caps.layer_forward_params)
        passed = {}
        extras = {}
        for k, v in kwargs.items():
            if k in block_ok:
                passed[k] = v
            else:
                extras[k] = v
        ebk = dict(passed.pop('extra_block_kwargs', None) or {})
        layer_ebk = {k: v for k, v in ebk.items() if k in layer_ok}
        extras.update({k: v for k, v in ebk.items() if k not in layer_ok})
        if layer_ebk:
            passed['extra_block_kwargs'] = layer_ebk
        if mhc is not None:
            passed['mhc_multistream'] = mhc
        self._bridge_extras = extras or None
        return passed

    def _forward_tracked(self, input_ids, position_ids, hidden_states, attention_mask, mhc_multistream=None, **kwargs):
        """Manual per-depth loop: releases' block forward accepts no per-depth hooks."""
        caps = get_mtp_capabilities()
        extra = dict(kwargs.pop('extra_block_kwargs', None) or {})
        kwargs = {**kwargs, **extra}

        offset = get_mtp_layer_offset(self.config, self.vp_stage)
        assert offset == 0, (f'MTP does not support a pipeline offset (got {offset}): process_mtp_loss chunks the '
                             'output by 1 + mtp_num_layers, which only matches when no offset chunks are prepended.')
        hs_list = list(torch.chunk(hidden_states, 1 + offset, dim=0))
        hs = hs_list[offset]

        if mhc_multistream is not None:
            mhc_chunks = list(torch.chunk(mhc_multistream, 1 + offset, dim=0))
            depth_in = mhc_chunks[offset]
        else:
            depth_in = hs

        # COMPAT(dev): mtp_detach_heads is dev-only.
        if caps.has_detach_heads and getattr(self.config, 'mtp_detach_heads', False):
            depth_in = depth_in.detach()

        n = self.config.mtp_num_layers or 0
        repeated = getattr(self.config, 'mtp_use_repeated_layer', False)

        # Keys the upstream layer forward declares go through verbatim;
        # everything else is a bridge extra for the inner TransformerLayer.
        allowed = set(caps.layer_forward_params)
        base_kw = {k: v for k, v in kwargs.items() if k in allowed}
        self._bridge_extras = {k: v for k, v in kwargs.items() if k not in allowed} or None

        for it in range(n):
            idx = 0 if repeated else it
            layer = self.layers[idx]

            lk = dict(base_kw)
            lk['_bridge_depth'] = it + 1

            res = layer(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=depth_in,
                attention_mask=attention_mask,
                **lk)

            values = tuple(res)
            if len(values) == 5:
                hs_out, input_ids, position_ids, pmask, mtp_input_mask = values
                base_kw['padding_mask'] = pmask
                if 'mtp_input_mask' not in allowed:
                    raise RuntimeError('Upstream MTP layer returns mtp_input_mask but does not accept it')
                base_kw['mtp_input_mask'] = mtp_input_mask
            elif len(values) == 4:
                hs_out, input_ids, position_ids, pmask = values
                base_kw['padding_mask'] = pmask
            elif len(values) == 3:
                hs_out, input_ids, position_ids = values
            else:
                raise ValueError(f'Unexpected MTP layer result length: {len(values)}')

            if mhc_multistream is not None:
                mhc_chunks.append(hs_out)
                hs_list.append(layer._postprocess(hs_out))
                depth_in = hs_out
            else:
                hs_list.append(hs_out)
                depth_in = hs_out

        return torch.cat(hs_list, dim=0)


def prepare_mtp_loss_mask(labels: Optional[torch.Tensor], loss_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Merge ``labels == -100`` into *loss_mask* so upstream
    ``process_mtp_loss`` sees a clean mask."""
    if labels is None:
        return loss_mask
    valid = labels != -100
    if loss_mask is None:
        return valid
    return loss_mask & valid


def process_mtp_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    output_layer: Callable,
    output_weight: Optional[torch.Tensor],
    runtime_gather_output: Optional[bool],
    is_training: bool,
    compute_language_model_loss: Callable,
    config,
    cp_group=None,
    packed_seq_params=None,
    **extra,
) -> torch.Tensor:
    """Call upstream ``process_mtp_loss`` with the kwargs the installed version declares.

    This wrapper cannot be dropped: 0.18 has no ``tp_group``/``input_ids`` parameters, while
    0.19+ and dev need them, and ``gpt_model`` passes them unconditionally so that one call site
    serves every supported version. *loss_mask* must already exclude ``-100`` tokens;
    *output_layer* follows the upstream convention ``(hs, weight=..., runtime_gather_output=...)
    -> (logits, bias)``.  ``scale_logits_fn`` is never forwarded so the MTP loss stays in sync
    with the main loss (which does not apply MuP scaling).
    """
    accepted_kwargs = set(_PROCESS_MTP_LOSS_PARAMS)
    return _upstream_process_mtp_loss(
        hidden_states=hidden_states,
        labels=labels,
        loss_mask=loss_mask,
        output_layer=output_layer,
        output_weight=output_weight,
        runtime_gather_output=runtime_gather_output,
        is_training=is_training,
        compute_language_model_loss=compute_language_model_loss,
        config=config,
        cp_group=cp_group,
        packed_seq_params=packed_seq_params,
        **{
            k: v
            for k, v in extra.items() if k in accepted_kwargs
        })


def adapt_mtp_block_spec(config, mtp_block_spec):
    """Swap the upstream layer class for :class:`MultiTokenPredictionLayer`.

    A spec that already carries a model-specific layer subclass is left untouched:
    those layers own semantics the bridge adapter must not replace.
    """
    if mtp_block_spec is None:
        return None
    for ls in mtp_block_spec.layer_specs:
        module = getattr(ls, 'module', None)
        if module is None or module is McoreMultiTokenPredictionLayer:
            ls.module = MultiTokenPredictionLayer
    return mtp_block_spec


def get_mtp_block_cls(config):
    """Block class for MTP construction (used by ModelLoader patch ctx)."""
    return MultiTokenPredictionBlock
