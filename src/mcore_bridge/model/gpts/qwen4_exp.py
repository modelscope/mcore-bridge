# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import math
import megatron.core
import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import contextmanager
from copy import deepcopy
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm, TERowParallelLinear
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec, get_gpt_mtp_block_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.gated_delta_net import GatedDeltaNetSubmodules
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.tensor_parallel.mappings import (gather_from_tensor_model_parallel_region,
                                                    scatter_to_sequence_parallel_region)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.utils import make_viewless_tensor
from torch import nn
from transformers.utils import is_torch_npu_available
from typing import List, Optional

from mcore_bridge.utils import get_env_args, get_local_layer_specs, get_logger
from mcore_bridge.utils.megatron_utils import reconstruct_tensor_cp

from ..modules import (QSA_SPARSE_KERNEL_ENV, GatedDeltaNet, MultiTokenPredictionLayer, QSAIndexer,
                       QSASparseCoreAttention, Qwen4ExpTextGatedResidual, Qwen4ExpTextPLELayer, TransformerBlock,
                       TransformerLayer, qsa_sparse_supported, use_qsa_sparse_kernel)
from ..modules.ple import Qwen4ExpTextNGramEmbedding
from ..register import ModelLoader
from .qwen3_next import Qwen3NextBridge, Qwen3NextRMSNorm, Qwen3NextSelfAttention

logger = get_logger()

_HC_WEIGHT_KEYS = (
    'hc_norm.weight',
    'input_mix_weight_down.weight',
    'input_mix_weight_up.weight',
    'block_inject_weight.weight',
)


class Qwen4ExpGDN(GatedDeltaNet):
    # upstream uses config.activation_func as the act_fn for both the gated output
    # norm and the conv1d; but the conv1d path asserts act_fn in ['silu', 'swish'],
    # so setting it to sigmoid would be rejected. Override only the output gate here.
    def _apply_gated_norm(self, x, gate):
        x_dtype = x.dtype
        x = x.reshape(-1, x.shape[-1])
        y = self.out_norm(x)
        gate = gate.reshape(-1, gate.shape[-1])
        output_gate_type = self.config.output_gate_type
        gate_act = torch.sigmoid if output_gate_type == 'sigmoid' else F.silu
        y = y * gate_act(gate.float())
        return y.to(x_dtype)


class Qwen4ExpLayer(TransformerLayer):
    # refer: transformers Qwen4ExpTextDecoderLayer
    def __init__(self, config, submodules, layer_number: int = 1, **kwargs):
        super().__init__(config, submodules, layer_number, **kwargs)
        self.ple = None
        if self.layer_number in config.ple_layer_ids:
            self.ple = Qwen4ExpTextPLELayer(
                config, config.ple_layer_ids.index(self.layer_number), pg_collection=self.pg_collection)
        is_linear_attention = self._resolve_is_linear_attention(config)
        if not is_linear_attention and config.indexer_n_heads is not None:
            self.self_attention.indexer = QSAIndexer(config, tp_group=self.tp_group)
            if qsa_sparse_supported(config.kv_channels):
                attn = self.self_attention
                attn.core_attention = QSASparseCoreAttention(
                    attn.core_attention, config, softmax_scale=config.softmax_scale)
        self.attn_hyper_connection = Qwen4ExpTextGatedResidual(config)
        self.mlp_hyper_connection = Qwen4ExpTextGatedResidual(config)

    # override in MTP layer
    def _resolve_is_linear_attention(self, config):
        return config.linear_attention_freq[self.layer_number - 1]

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        attention_mask = kwargs.get('attention_mask')
        packed_seq_params: PackedSeqParams = kwargs.get('packed_seq_params')
        attn_kwargs = dict(
            attention_mask=attention_mask,
            inference_context=kwargs.get('inference_context'),
            rotary_pos_emb=kwargs.get('rotary_pos_emb'),
            rotary_pos_cos=kwargs.get('rotary_pos_cos'),
            rotary_pos_sin=kwargs.get('rotary_pos_sin'),
            attention_bias=kwargs.get('attention_bias'),
            packed_seq_params=packed_seq_params,
            sequence_len_offset=kwargs.get('sequence_len_offset'),
        )
        if self.ple is not None:
            input_ids = kwargs.get('input_ids')
            assert input_ids is not None, 'PLE layers require input_ids in extra_block_kwargs'
            hidden_states = hidden_states + self.ple(hidden_states, input_ids, packed_seq_params)

        # attention sub-block (mirrors transformers Qwen4ExpTextDecoderLayer.forward)
        hidden_states, hyper_input, injection_weights = self.attn_hyper_connection(hidden_states)
        qsa_selection, sparse = self._qsa_select(hidden_states, attn_kwargs, kwargs.get('position_ids'))
        if qsa_selection is not None:
            # sparse: int64 indices consumed by QSASparseCoreAttention; mask
            # fallback: bool mask consumed by TE under attn_mask_type=arbitrary.
            attn_kwargs = dict(attn_kwargs, attention_mask=qsa_selection)
        # `arbitrary` mask type is only needed for the bool-mask (TE) path; the
        # sparse kernel reads the indices and ignores attn_mask_type.
        with self._patch_apply_rotary_pos_emb(), self._qsa_arbitrary_mask(qsa_selection is not None and not sparse):
            hidden_states, _ = self.self_attention(hidden_states=hidden_states, **attn_kwargs)
        injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        hidden_states = hyper_input + injection.flatten(-2)

        # mlp sub-block
        hidden_states, hyper_input, injection_weights = self.mlp_hyper_connection(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        hidden_states = hyper_input + injection.flatten(-2)
        return hidden_states, None

    @contextmanager
    def _qsa_arbitrary_mask(self, enabled: bool):
        """Temporarily switch the attention to `arbitrary` mask type.

        `Attention.forward` takes no `attn_mask_type` argument -- it reads
        `self.attn_mask_type` (and core_attention's) -- so a custom mask is
        silently ignored unless the type is flipped. Restored in `finally` so a
        raising forward cannot leave the layer stuck in the slower unfused mode.
        """
        if not enabled:
            yield
            return
        attn = self.self_attention
        targets = [attn]
        core = getattr(attn, 'core_attention', None)
        if core is not None:
            targets.append(core)
        saved = [(t, t.attn_mask_type) for t in targets if hasattr(t, 'attn_mask_type')]
        for t, _ in saved:
            t.attn_mask_type = AttnMaskType.arbitrary
        try:
            yield
        finally:
            for t, old in saved:
                t.attn_mask_type = old

    def _qsa_select(self, hidden_states, attn_kwargs, position_ids=None):
        """Choose the QSA selection representation for this forward.

        Returns ``(selection, is_sparse)``. ``is_sparse`` means ``selection`` is the
        int64 index tensor consumed by ``QSASparseCoreAttention`` (sbhd and thd,
        with or without SP/CP); otherwise it is the bool TE mask from the legacy
        path, or ``None`` for full attention. CP needs the all_gather comm type
        (the selection has to see every key before attention runs; ring/p2p
        cannot provide that), mirroring mcore DSA's restriction.
        """
        indexer = getattr(self.self_attention, 'indexer', None)
        sparse_ok = isinstance(getattr(self.self_attention, 'core_attention', None), QSASparseCoreAttention)
        if indexer is None:
            return None, False
        packed_seq_params: PackedSeqParams = attn_kwargs.get('packed_seq_params')
        is_thd = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        cp_size = self.config.context_parallel_size
        needs_kernel = is_thd or cp_size > 1

        # sbhd with CP==1
        if not needs_kernel:
            return self._qsa_select_mask(hidden_states, attn_kwargs), False

        # From here the mask path is not an option, so by default a failure raises instead
        # of silently degrading to dense attention (which would diverge from the sparse
        # rollout without telling anyone). An explicit opt-out via the env var is the one
        # sanctioned fallback: full attention, warned about once.
        if not sparse_ok:
            if not use_qsa_sparse_kernel():
                logger.warning_once(f'QSA sparse kernel is disabled via {QSA_SPARSE_KERNEL_ENV}=0; '
                                    f'falling back to full attention ({"packing/thd" if is_thd else f"CP={cp_size}"}).')
                return None, False
            raise RuntimeError(f'QSA needs the sparse kernel here ({"packing/thd" if is_thd else f"CP={cp_size}"}), '
                               'but QSASparseCoreAttention was not installed -- triton is missing or '
                               f'kv_channels={getattr(self.config, "kv_channels", None)} is not a power of two. '
                               'Use --padding_free false with context_parallel_size 1 to take the bool-mask path, '
                               f'or set {QSA_SPARSE_KERNEL_ENV}=0 to fall back to full attention.')
        if cp_size > 1 and getattr(self.config, 'cp_comm_type', None) != 'all_gather':
            raise RuntimeError(f"QSA sparse selection with context_parallel_size={cp_size} requires "
                               f"cp_comm_type='all_gather' (got {getattr(self.config, 'cp_comm_type', None)!r}): the "
                               'selection has to see every key before attention runs, which ring/p2p cannot provide.')
        rotary_pos_emb = attn_kwargs.get('rotary_pos_emb')
        if rotary_pos_emb is None:
            raise RuntimeError('QSA sparse selection needs rotary_pos_emb (blocks rotate at their first '
                               'token position) but it was not passed to the layer.')
        if is_thd:
            indices = self._qsa_select_indices_thd(hidden_states, rotary_pos_emb, packed_seq_params, position_ids)
        else:
            indices = self._qsa_select_indices_sbhd(hidden_states, rotary_pos_emb)
        return indices, True

    def _qsa_select_indices_sbhd(self, hidden_states, rotary_pos_emb):
        if self.config.sequence_parallel and self.config.tensor_model_parallel_size > 1:
            hidden_states = gather_from_sequence_parallel_region(hidden_states, tensor_parallel_output_grad=False)
        if self.config.context_parallel_size > 1:
            hidden_states = reconstruct_tensor_cp(hidden_states, None, dim=0)
            rotary_pos_emb = reconstruct_tensor_cp(rotary_pos_emb, None, dim=0)
        return self.self_attention.indexer.selection_as_token_indices(hidden_states, rotary_pos_emb)

    def _qsa_select_indices_thd(self, hidden_states, rotary_pos_emb, packed_seq_params, position_ids=None):
        if self.config.sequence_parallel and self.config.tensor_model_parallel_size > 1:
            hidden_states = gather_from_sequence_parallel_region(hidden_states, tensor_parallel_output_grad=False)
        psp_for_cp = None
        if self.config.context_parallel_size > 1:
            # TE's packed CP partition (thd_get_partitioned_indices) requires
            # int32 cu; the training pipeline produces int32, but normalize callers
            # that hand us int64.
            if packed_seq_params.cu_seqlens_q is not None \
                    and packed_seq_params.cu_seqlens_q.dtype != torch.int32:
                psp_for_cp = copy.copy(packed_seq_params)
                psp_for_cp.cu_seqlens_q = packed_seq_params.cu_seqlens_q.to(torch.int32)
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    psp_for_cp.cu_seqlens_q_padded = packed_seq_params.cu_seqlens_q_padded.to(torch.int32)
            else:
                psp_for_cp = packed_seq_params
            local_len = hidden_states.shape[0]
            hidden_states = reconstruct_tensor_cp(hidden_states, psp_for_cp, dim=0)
        # Per-token rotary angles. Without rope fusion gpt_model already indexes
        # the freq table by position_ids, so what arrives is per-token (zigzag-
        # sharded under CP -- undo it like hidden). With fusion the raw table
        # arrives and must be indexed by the (CP-reconstructed) per-doc ids.
        freqs = rotary_pos_emb
        if self.config.context_parallel_size > 1:
            fused_table = (
                self.config.position_embedding_type != 'mrope'
                and (self.config.apply_rope_fusion or freqs.shape[0] != local_len))
            if fused_table:
                if position_ids is None:
                    raise RuntimeError('QSA thd selection under CP needs position_ids to index the fused rotary '
                                       'table (apply_rope_fusion=true hands over the raw table, not per-token '
                                       'freqs). Pass position_ids, or set --apply_rope_fusion false.')
                pos = reconstruct_tensor_cp(position_ids, psp_for_cp, dim=1)
                freqs = freqs[pos.reshape(-1)]
            else:
                freqs = reconstruct_tensor_cp(freqs, psp_for_cp, dim=0)
        else:
            fused_table = freqs.shape[0] != hidden_states.shape[0]
            if fused_table:
                raise RuntimeError(f'QSA thd selection got a fused rotary table ({freqs.shape[0]} rows for '
                                   f'{hidden_states.shape[0]} tokens): apply_rope_fusion=true hands over the raw '
                                   'table rather than per-token freqs. Set --apply_rope_fusion false.')
        # the CP reconstruct (like TE's thd kernels) works in the padded pack
        # space, so align against the padded cu when present
        cu = packed_seq_params.cu_seqlens_q_padded
        if cu is None:
            cu = packed_seq_params.cu_seqlens_q
        if cu is None:
            raise RuntimeError('QSA thd selection needs packed_seq_params.cu_seqlens_q to find document '
                               'boundaries, but it is missing.')
        cu = Qwen4ExpTextPLELayer._normalize_cu_seqlens(cu, hidden_states.shape[0])
        hidden_tok = hidden_states.reshape(hidden_states.shape[0], -1)
        return self.self_attention.indexer.select_token_indices_thd(
            hidden_tok, freqs, cu, force_materialize=self.config.context_parallel_size > 1)

    def _qsa_select_mask(self, hidden_states, attn_kwargs):
        # Bool-mask QSA on TE's `arbitrary` mask. Only reached for sbhd with CP==1 --
        # _qsa_selection() routes thd and CP>1 to the kernel, because TE rejects an
        # arbitrary mask under thd and this path never gathers keys across CP ranks.
        # Returning None means full attention, which here only happens when the
        # sequence is short enough that selection is a no-op anyway (selection_as_mask
        # short-circuits at max_blocks <= block_topk).
        indexer = self.self_attention.indexer
        if indexer is None:
            return None
        rotary_pos_emb = attn_kwargs.get('rotary_pos_emb')
        if rotary_pos_emb is None:
            raise RuntimeError('QSA bool-mask selection needs rotary_pos_emb (blocks rotate at their first '
                               'token position) but it was not passed to the layer.')
        if self.config.sequence_parallel and self.config.tensor_model_parallel_size > 1:
            hidden_states = gather_from_sequence_parallel_region(hidden_states, tensor_parallel_output_grad=False)
        return indexer.selection_as_mask(hidden_states, rotary_pos_emb)


class Qwen4ExpTransformerBlock(TransformerBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.config
        if config.hc_count is None:
            raise ValueError('Qwen4Exp requires config.hc_count (checkpoint has hc_count=4).')
        if config.hc_count > 1 and self.has_final_layernorm_in_this_stage():
            # Final contraction (use_combine=False matches the checkpoint:
            # hyper_connection_mixer has no block_inject_weight).
            self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(config, use_combine=False)


class Qwen4ExpMTPInnerLayer(Qwen4ExpLayer):

    def _resolve_is_linear_attention(self, config):
        return False


class Qwen4ExpMTPStreamNorm(nn.Module):
    """Zero-centered RMSNorm over the full multi-stream (``hc_count * hidden_size``).

    Qwen3.8-Flash-Next's MTP ``pre_fc_norm_hidden`` normalizes the concatenated ``hc_count`` streams
    jointly (a single GemmaRMSNorm over n*H with a per-element affine), unlike Megatron's mHC MTP which
    normalizes each H-sized stream independently. The MTP spec builds norms with ``hidden_size=H``, so
    scale by ``hc_count`` here.
    """

    def __init__(self, config, hidden_size, eps):
        super().__init__()
        self.dim = config.hc_count * hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(self.dim, dtype=config.params_dtype))
        self.weight.sequence_parallel = config.sequence_parallel

    def forward(self, x):
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return ((1.0 + self.weight.float()) * x).to(input_dtype)


class Qwen4ExpMultiTokenPredictionLayer(MultiTokenPredictionLayer):
    """Qwen3.8-Flash-Next MTP head: the ``residual_linear_shared`` fusion over the gated-HC backbone.

    The backbone runs Qwen4Exp's own gated hyper-connections (``hc_count`` streams) but NOT Megatron's
    ``enable_hyper_connections`` mHC. The MTP head still needs Megatron's mHC *projection* form
    (separate ``e_proj``/``h_proj`` rather than a fused ``eh_proj``), because the checkpoint stores
    ``fc_embedding``/``fc_hidden`` (H->H each) and adds the embedding residual to every stream. So the
    subtree is built with hyper-connections enabled via a config copy -- which selects e_proj/h_proj --
    then Megatron's mHC-only contraction params (``hc_head_*``) and ``final_layernorm`` are dropped in
    favour of Qwen4Exp's own ``hyper_connection_mixer``, matching the checkpoint and the vLLM draft
    model. ``_concat_embeddings`` normalizes the multi-stream jointly (``Qwen4ExpMTPStreamNorm``) and
    ``_postprocess`` contracts with the mixer, so the layer returns the multi-stream and the MTP block
    applies ``_postprocess`` for the loss head.
    """

    def __init__(self, config, submodules, *args, **kwargs):
        mtp_config = copy.copy(config)
        mtp_config.enable_hyper_connections = True
        super().__init__(mtp_config, submodules, *args, **kwargs)
        for name in ('hc_head_fn', 'hc_head_base', 'hc_head_scale', 'final_layernorm'):
            if hasattr(self, name):
                delattr(self, name)
        self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(config, use_combine=False)

    def _concat_embeddings(self, hidden_states, decoder_input):
        # hidden_states: pre-mixer multi-stream [s, b, n*H]; decoder_input: rolled-token embedding [s, b, H].
        n = self.config.hc_count
        h = self.config.hidden_size
        decoder_input = self.enorm(decoder_input)
        decoder_input = make_viewless_tensor(inp=decoder_input, requires_grad=True, keep_graph=True)
        # Qwen4Exp normalizes the full multi-stream jointly (pre_fc_norm_hidden over n*H), not per-stream.
        hs = self.hnorm(hidden_states)
        hs = make_viewless_tensor(inp=hs, requires_grad=True, keep_graph=True).unflatten(-1, (n, h))
        e_out, _ = self.e_proj(decoder_input)  # fc_embedding -> [s, b, H/tp]
        h_out, _ = self.h_proj(hs)  # fc_hidden -> [s, b, n, H/tp]
        out = e_out.unsqueeze(2) + h_out  # add the embedding residual to every stream
        out = gather_from_tensor_model_parallel_region(out, group=self.tp_group)
        # Read the shape AFTER the gather: under sequence parallel the column-parallel projections
        # all-gather the sequence dim, so a pre-projection `s` would be stale.
        s, b, n_out, h_dim = out.shape
        out = out.reshape(s, b, n_out * h_dim)
        if self.sequence_parallel:
            out = scatter_to_sequence_parallel_region(out, group=self.tp_group)
        return out

    def _postprocess(self, hidden_states):
        # Contract the multi-stream [s, b, n*H] to [s, b, H] with Qwen4Exp's gated mixer (no final norm).
        return self.hyper_connection_mixer(hidden_states)

    def _get_embeddings(self,
                        input_ids,
                        position_ids,
                        embedding,
                        hidden_states,
                        packed_seq_params=None,
                        decoder_input=None):
        input_ids, position_ids, decoder_input, hidden_states = super()._get_embeddings(
            input_ids, position_ids, embedding, hidden_states, packed_seq_params, decoder_input)
        # Stash the rolled ids so _proj_and_transformer_layer can forward them to the inner
        # Qwen4ExpMTPInnerLayer (its QSA indexer needs position_ids under packing/CP; PLE is absent).
        self._mtp_input_ids = input_ids
        self._mtp_position_ids = position_ids
        return input_ids, position_ids, decoder_input, hidden_states

    def _proj_and_transformer_layer(self, *args, **kwargs):
        kwargs.setdefault('input_ids', getattr(self, '_mtp_input_ids', None))
        kwargs.setdefault('position_ids', getattr(self, '_mtp_position_ids', None))
        return super()._proj_and_transformer_layer(*args, **kwargs)


class Qwen4ExpBridge(Qwen3NextBridge):
    hf_mixer_prefix = 'model.'

    def _get_hf_experts_attr(self, is_mtp: bool = False):
        # The checkpoint stores experts as packed per-layer tensors
        # (`mlp.experts.gate_up_proj` / `mlp.experts.down_proj`).
        return True, True

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        mg_attn = None if mg_layer is None else mg_layer.self_attention
        is_linear_attention = self.config.linear_attention_freq[layer_idx]
        if is_linear_attention:
            # GDN weights; this model has no input_layernorm for GDN layers.
            hf_state_dict.update(
                self._set_linear_attn_state(mg_attn, hf_state_dict, 'linear_attn.', layer_idx, to_mcore))
        else:
            # Dense QSA-equivalent attention (qkv + output gate + q/k norms).
            hf_state_dict.update(self._set_attn_state(mg_attn, hf_state_dict, 'self_attn.', layer_idx, to_mcore))
            has_indexer = mg_attn is not None and getattr(mg_attn, 'indexer', None) is not None
            has_indexer = self._reduce_tensor_pp_group(has_indexer, to_mcore)
            if has_indexer:
                indexer = None if mg_attn is None else mg_attn.indexer
                for mg_key, hf_key in [('index_qk_proj.weight', 'self_attn.indexer.index_qk_proj.weight'),
                                       ('q_layernorm.weight', 'self_attn.indexer.q_layernorm.weight'),
                                       ('k_layernorm.weight', 'self_attn.indexer.k_layernorm.weight')]:
                    self._set_state_dict(indexer, mg_key, hf_state_dict, hf_key, to_mcore)
        return hf_state_dict

    def _set_layer_mlp(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool, is_mtp: bool = False):
        mg_mlp = None if mg_layer is None else mg_layer.mlp
        is_moe = mg_mlp is not None and hasattr(mg_mlp, 'experts')
        if not to_mcore:
            is_moe = torch.tensor([is_moe], dtype=torch.bool, device='cuda')
            if self.pp_size > 1:
                dist.all_reduce(is_moe, group=self.pp_group)
        if is_moe:
            hf_state_dict.update(
                self._set_moe_state(
                    mg_mlp, hf_state_dict, f'{self.hf_mlp_prefix}.', layer_idx, to_mcore, is_mtp=is_mtp))
        else:
            hf_state_dict.update(
                self._set_mlp_state(mg_mlp, hf_state_dict, f'{self.hf_mlp_prefix}.', layer_idx, to_mcore))
        # No post_attention_layernorm in this model (HC norms replace it).
        return hf_state_dict

    def _set_layer_hc(self, mg_layer, hf_state_dict, to_mcore: bool):
        for key in ['attn_hyper_connection', 'mlp_hyper_connection']:
            hyper_connection = None if mg_layer is None else getattr(mg_layer, key)
            for weight_key in _HC_WEIGHT_KEYS:
                self._set_state_dict(hyper_connection, weight_key, hf_state_dict, f'{key}.{weight_key}', to_mcore)

    # --- PLE -----------------------------------------------------------------
    _PLE_NGRAM_BUFFERS = ('layer_multipliers', 'ngram_heads_offsets', 'ngram_heads_vocab_sizes')

    def _get_tp_split_dim(self, mg_key):
        # PLE weights are replicated across TP; in particular `conv1d.weight`
        # must not use the dim-0 split that applies to the GDN conv1d.
        if getattr(self, '_converting_ple', False):
            return None
        return super()._get_tp_split_dim(mg_key)

    def _get_pp_src_rank(self, has_module: bool) -> int:
        """Global rank of the PP stage holding the module (all-reduce MAX)."""
        holder = torch.tensor([dist.get_rank() if has_module else -1], dtype=torch.long, device='cuda')
        if self.pp_size > 1:
            dist.all_reduce(holder, op=dist.ReduceOp.MAX, group=self.pp_group)
        return int(holder.item())

    def _broadcast_pp_weight(self, tensor, pp_src_rank: int):
        """Cross-pp transfer of one exported weight through the tp-aligned pp
        group. `tensor` is non-None only on the exporting rank (tp rank 0 of the
        stage owning the PLE layer, i.e. the src of its pp group); the other pp
        members receive it. The payload is streamed through
        `_chunked_broadcast_pp` so no full-size GPU buffer is materialized, and
        it always rides as raw bytes (flattened uint8): any current or future
        low-width dtype (fp8 e4m3/e5m2, int8, packed fp4, ...) is transported
        without NCCL dtype concerns; the meta carries the original shape/dtype
        for the receiver to view back. Groups whose src has nothing to transfer
        (tp != 0 coords) exchange only the empty meta.
        """
        meta = [None if tensor is None else [list(tensor.shape), str(tensor.dtype).replace('torch.', '')]]
        dist.broadcast_object_list(meta, src=pp_src_rank, group=self.pp_group)
        if meta[0] is None:
            return None
        shape, dtype_name = meta[0]
        dtype = getattr(torch, dtype_name)
        if tensor is not None:
            payload = tensor.contiguous().flatten().view(torch.uint8)
            self._chunked_broadcast_pp(payload, list(payload.shape), payload.dtype, pp_src_rank, self.pp_group)
            return tensor
        byte_count = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        out = self._chunked_broadcast_pp(None, [byte_count], torch.uint8, pp_src_rank, self.pp_group)
        return out.view(dtype).view(shape)

    def _iter_ple_table_export(self, ple, pp_src_rank, layer_prefix: str):
        """Lazily export the PLE table shards one by one. On the exporting rank
        (tp rank 0 of the owning stage) each shard is assembled by the chunked
        all_reduce inside ``iter_export_table_to_hf`` and immediately streamed
        to the other pp members; on the other stages it is received as it
        arrives. Nothing accumulates: only the consumer (the safetensors
        writer) drives the pace, so no rank ever holds more than a single shard
        of the 100GB-scale table in host memory. The per-shard collectives keep
        every rank of the tp-aligned pp groups in lockstep, exactly like the
        synchronous version this generator replaces.
        """
        parts = self.config.split_ngram_parts
        scale_key = Qwen4ExpTextNGramEmbedding._NGRAM_SCALE_KEY
        shard_prefix = 'ple.ple_embedding.ngram_embedding'
        # On tp != 0 ranks of the owning stage the table iterator executes the
        # same all_reduces but yields nothing (shards exist only on tp rank 0).
        table_iter = ple.ple_embedding.iter_export_table_to_hf() if ple is not None else iter(())
        for i in range(parts):
            shard = next(table_iter, (None, None))[1] if ple is not None else None
            if self.pp_size > 1:
                shard = self._broadcast_pp_weight(shard, pp_src_rank)
            if shard is not None:
                yield f'{layer_prefix}{shard_prefix}.shard_{i}.weight', shard
        # The scale is a tiny scalar; a pickled broadcast is fine.
        scale = None
        if ple is not None:
            for k, v in table_iter:
                if k == scale_key:
                    scale = v
        if self.pp_size > 1:
            obj = [scale]
            dist.broadcast_object_list(obj, src=pp_src_rank, group=self.pp_group)
            scale = obj[0]
        if scale is not None:
            yield f'{layer_prefix}{scale_key}', scale

    def _set_layer_ple(self, mg_layer, hf_state_dict, to_mcore: bool, layer_prefix: str = ''):
        ple = None if mg_layer is None else getattr(mg_layer, 'ple', None)
        if to_mcore:
            # Only the stage owning the PLE layer reaches this path, so it
            # must not run pp collectives (other pp ranks never enter here).
            if ple is None:
                return
            pp_src_rank = None
        else:
            # to_hf: every pp rank calls this for every layer, so the pp
            # collectives below stay in sync across stages.
            pp_src_rank = self._get_pp_src_rank(ple is not None)
            has_ple = self._reduce_tensor_pp_group(ple is not None, to_mcore)
            if not has_ple:
                return
        # `ple` is only non-None on the pp stage owning the PLE layer, so the offload
        # flag has to be reduced across pp before it can gate the loop below -- that
        # loop runs pp collectives (broadcast_object_list) and export_table_to_hf runs
        # tp ones, and stages disagreeing on whether to enter would deadlock.
        ple_offloaded = self._reduce_tensor_pp_group(ple is not None and ple.ple_embedding.cpu_offload, to_mcore)
        if to_mcore:
            # A PEFT/adapter checkpoint carries no PLE n-gram buffers -- those come from the base
            # checkpoint that the adapter is applied on top of -- so a peft-format load must skip
            # them instead of KeyError-ing on `ple.ple_embedding.layer_multipliers`.
            skip_ngram_state = self._peft_format
        else:
            skip_ngram_state = not self._is_saving and (self._peft_format or ple_offloaded)
        for buf in () if skip_ngram_state else self._PLE_NGRAM_BUFFERS:
            if to_mcore:
                buffer = getattr(ple.ple_embedding, buf)
                buffer.copy_(hf_state_dict[f'ple.ple_embedding.{buf}'].load().to(buffer.device))
            else:
                tensor = getattr(ple.ple_embedding, buf).data.clone() if ple is not None else None
                if self.pp_size > 1:
                    obj = [tensor]
                    dist.broadcast_object_list(obj, src=pp_src_rank, group=self.pp_group)
                    tensor = obj[0]
                # Written directly into the state dict (bypasses _get_weight,
                # which normally applies _target_device).
                if tensor is not None and self._target_device is not None:
                    tensor = tensor.to(self._target_device)
                hf_state_dict[f'ple.ple_embedding.{buf}'] = tensor
        if not skip_ngram_state and to_mcore:
            # The table's only ingestion path: fill from the HF checkpoint shards.
            ple.ple_embedding.fill_table_from_hf(hf_state_dict)
        if not to_mcore and not skip_ngram_state:
            # Stream the table shards one at a time instead of accumulating the
            # 100GB-scale table in host memory: the generator below is only
            # consumed at _convert's yield point (the safetensors writer drives
            # the pace), and each shard is broadcast to the other pp members as
            # it is produced, then released once written.
            self._pending_export_iter = self._iter_ple_table_export(ple, pp_src_rank, layer_prefix)
        else:
            self._pending_export_iter = None
        self._converting_ple = True
        try:
            for mg_key, hf_key in [('key_proj.weight', 'ple.key_proj.weight'),
                                   ('value_proj.weight', 'ple.value_proj.weight'),
                                   ('norm_key.weight', 'ple.norm_key.weight'),
                                   ('norm_query.weight', 'ple.norm_query.weight'),
                                   ('norm_conv.weight', 'ple.norm_conv.weight'),
                                   ('conv1d.weight', 'ple.conv1d.weight')]:
                self._set_state_dict(ple, mg_key, hf_state_dict, hf_key, to_mcore)
        finally:
            self._converting_ple = False

    def _set_layer_state(self, mg_layer, hf_state_dict, hf_prefix: str, layer_idx: int, to_mcore: bool):
        hf_prefix = f'{hf_prefix}{layer_idx}.'
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        hf_state_dict.update(self._set_layer_attn(mg_layer, hf_state_dict, layer_idx, to_mcore))
        hf_state_dict.update(self._set_layer_mlp(mg_layer, hf_state_dict, layer_idx, to_mcore))
        self._set_layer_hc(mg_layer, hf_state_dict, to_mcore)
        if (layer_idx + 1) in (self.config.ple_layer_ids or []):
            self._set_layer_ple(mg_layer, hf_state_dict, to_mcore, layer_prefix=hf_prefix)
        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _set_final_layernorm(self, lm_model, hf_state_dict, to_mcore):
        # This architecture has no final layernorm: the HC norms and the
        # hyper_connection_mixer contraction replace it, and the checkpoint
        # carries no `norm` weight.
        pass

    def _convert_post_process(self, mg_model, hf_state_dict, hf_prefix: str, to_mcore):
        res = super()._convert_post_process(mg_model, hf_state_dict, hf_prefix, to_mcore)
        lm_model = mg_model.language_model if self.is_multimodal else mg_model
        hc_count = self.config.hc_count
        if hc_count > 1:
            # The mixer only exists on the stage holding the final layernorm.
            mixer_keys = ['hc_norm.weight', 'input_mix_weight_down.weight', 'input_mix_weight_up.weight']
            # super() returns {} in to_mcore mode: read from the incoming full
            # state dict; in to_hf mode write into the dict super() returns.
            mixer_sd = hf_state_dict if to_mcore else res
            for key in mixer_keys:
                self._set_state_dict(lm_model, f'decoder.hyper_connection_mixer.{key}', mixer_sd,
                                     f'{self.hf_mixer_prefix}hyper_connection_mixer.{key}', to_mcore)
        return res

    def _convert_mtp_extra(self, mtp_layer, hf_state_dict, to_mcore, origin_hf_state_dict):
        # Qwen3.8-Flash-Next's MTP head lives at the `mtp.` level (not under `mtp.layers.i`):
        # pre_fc_norm_embedding/pre_fc_norm_hidden -> enorm/hnorm, fc_embedding/fc_hidden -> e_proj/h_proj
        # (the residual_linear_shared fusion), plus its own hyper_connection_mixer for the contraction.
        # There is no fused eh_proj and no final norm (the mixer is the contraction).
        sd = self._remove_prefix(origin_hf_state_dict, 'mtp.')
        for mg_key, key in [('enorm.weight', 'pre_fc_norm_embedding.weight'),
                            ('hnorm.weight', 'pre_fc_norm_hidden.weight'), ('e_proj.weight', 'fc_embedding.weight'),
                            ('h_proj.weight', 'fc_hidden.weight')]:
            self._set_state_dict(mtp_layer, mg_key, sd, key, to_mcore)
        self._fp8_skip_modules.update({'mtp.fc_embedding', 'mtp.fc_hidden'})
        mixer = None if mtp_layer is None else getattr(mtp_layer, 'hyper_connection_mixer', None)
        for key in ('hc_norm.weight', 'input_mix_weight_down.weight', 'input_mix_weight_up.weight'):
            self._set_state_dict(mixer, key, sd, f'hyper_connection_mixer.{key}', to_mcore)
        if not to_mcore:
            origin_hf_state_dict.update(self._add_prefix(sd, 'mtp.'))

    def _convert_mtp_layer(self, lm_model, hf_state_dict, hf_prefix: str, layer_idx: int, to_mcore: bool):
        mtp_layer = lm_model.mtp.layers[layer_idx] if hasattr(lm_model, 'mtp') else None
        hf_prefix = f'{hf_prefix}{layer_idx}.'  # 'mtp.layers.0.'
        if to_mcore:
            origin_hf_state_dict = hf_state_dict
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
            if len(hf_state_dict) == 0:
                logger.info(f'MTP layer {layer_idx} safetensors weights not found, '
                            'this part will be randomly initialized.')
                for param in mtp_layer.parameters():
                    if param.ndim == 2:
                        mtp_layer.config.init_method(param.data)
                return {}
        else:
            origin_hf_state_dict = {}
            hf_state_dict = {}
        self._convert_mtp_extra(mtp_layer, hf_state_dict, to_mcore, origin_hf_state_dict)
        # Inner block: a full-attention + MoE Qwen4ExpLayer with its own gated hyper-connections.
        # layer_idx=-1 routes _set_layer_attn through linear_attention_freq[-1] (the backbone's last
        # layer, full_attention), matching the MTP head, which is always full-attention.
        inner = None if mtp_layer is None else mtp_layer.transformer_layer
        hf_state_dict.update(self._set_layer_attn(inner, hf_state_dict, -1, to_mcore))
        hf_state_dict.update(self._set_layer_mlp(inner, hf_state_dict, -1, to_mcore, is_mtp=True))
        self._set_layer_hc(inner, hf_state_dict, to_mcore)
        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
            hf_state_dict.update(origin_hf_state_dict)
        return hf_state_dict


class Qwen4ExpLoader(ModelLoader):
    transformer_block = Qwen4ExpTransformerBlock

    def _get_moe_layer_pattern(self) -> List[bool]:
        config = self.config
        freq = config.moe_layer_freq
        if isinstance(freq, list):
            return [bool(x) for x in freq]
        # int N: one MoE every N layers (mcore convention: i % N == N - 1).
        return [i % freq == freq - 1 for i in range(config.num_layers)]

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        config = self.config
        config.hetereogenous_dist_checkpoint = True
        if config.context_parallel_size > 1 and getattr(config, 'cp_comm_type', None) in (None, 'p2p'):
            logger.warning_once(
                "Qwen4-Exp QSA under context parallelism requires cp_comm_type='all_gather'; "
                f"got {getattr(config, 'cp_comm_type', None)!r} (mcore's default), promoting to 'all_gather'.")
            config.cp_comm_type = 'all_gather'
        moe_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            use_kitchen=config.use_kitchen,
        )
        if config.num_moe_experts is not None:
            dense_spec = get_gpt_layer_with_transformer_engine_spec(
                num_experts=None,
                moe_grouped_gemm=config.moe_grouped_gemm,
                qk_layernorm=config.qk_layernorm,
                multi_latent_attention=config.multi_latent_attention,
                use_kitchen=config.use_kitchen,
            )
        else:
            dense_spec = moe_spec
        gdn_spec = ModuleSpec(
            module=Qwen4ExpGDN,
            submodules=GatedDeltaNetSubmodules(
                in_proj=TEColumnParallelLinear,
                out_norm=TENorm,
                out_proj=TERowParallelLinear,
            ),
        )
        moe_pattern = self._get_moe_layer_pattern()
        layer_specs = []
        for layer_idx, is_linear_attention in enumerate(config.linear_attention_freq):
            layer_spec = deepcopy(moe_spec if moe_pattern[layer_idx] else dense_spec)
            if is_linear_attention:
                layer_spec.submodules.self_attention = deepcopy(gdn_spec)
            else:
                layer_spec.submodules.self_attention.submodules.linear_qkv = TEColumnParallelLinear
                layer_spec.submodules.self_attention.module = Qwen3NextSelfAttention
                if hasattr(layer_spec.submodules.self_attention.submodules, 'q_layernorm'):
                    layer_spec.submodules.self_attention.submodules.q_layernorm = Qwen3NextRMSNorm
                if hasattr(layer_spec.submodules.self_attention.submodules, 'k_layernorm'):
                    layer_spec.submodules.self_attention.submodules.k_layernorm = Qwen3NextRMSNorm
            # This model has no per-layer layernorms (HC norms replace them).
            layer_spec.submodules.input_layernorm = IdentityOp
            if hasattr(layer_spec.submodules, 'pre_mlp_layernorm'):
                layer_spec.submodules.pre_mlp_layernorm = IdentityOp
            layer_specs.append(layer_spec)

        local_layer_specs = get_local_layer_specs(config, layer_specs, vp_stage=vp_stage)
        # No final layernorm in this model; keep the slot so the HC mixer
        # stage logic (has_final_layernorm_in_this_stage) still triggers.
        block_spec = TransformerBlockSubmodules(layer_specs=local_layer_specs, layer_norm=IdentityOp)
        return block_spec

    def _set_transformer_layer(self, transformer_layer_spec):
        for layer_spec in transformer_layer_spec.layer_specs:
            layer_spec.module = Qwen4ExpLayer

    def get_mtp_block_spec(self, transformer_layer_spec, vp_stage: Optional[int] = None):
        mtp_block_spec = get_gpt_mtp_block_spec(
            self.config, transformer_layer_spec, use_transformer_engine=True, vp_stage=vp_stage)
        if mtp_block_spec is not None:
            for layer_spec in mtp_block_spec.layer_specs:
                sub = layer_spec.submodules
                # The residual_linear_shared head needs Megatron's mHC *projection* form (separate
                # e_proj/h_proj slots). megatron-core <= 0.18 only has the fused eh_proj slot, and
                # assigning e_proj/h_proj there would silently no-op and surface later as an
                # AttributeError on self.e_proj -- so reject early and name the fix.
                if not (hasattr(sub, 'e_proj') and hasattr(sub, 'h_proj')):
                    raise NotImplementedError(
                        'Qwen3.8-Flash-Next MTP requires a Megatron whose MultiTokenPredictionLayerSubmodules '
                        'exposes e_proj/h_proj (megatron-core >= 0.19 / dev); got '
                        f'{megatron.core.__version__}.')
                layer_spec.module = Qwen4ExpMultiTokenPredictionLayer
                # residual_linear_shared head: separate e_proj/h_proj (fc_embedding/fc_hidden), a joint
                # multi-stream hnorm (pre_fc_norm_hidden over n*H), and no fused eh_proj. layer_norm
                # (final_layernorm) is built then dropped by the layer -- the mixer is the contraction.
                sub.enorm = TENorm
                sub.hnorm = Qwen4ExpMTPStreamNorm
                sub.eh_proj = None
                sub.e_proj = TEColumnParallelLinear
                sub.h_proj = TEColumnParallelLinear
                sub.layer_norm = TENorm
                # The MTP inner block is always full-attention (config.mtp.layer_types), independent of
                # the backbone layer numbering that Qwen4ExpLayer would otherwise read.
                sub.mtp_model_layer.module = Qwen4ExpMTPInnerLayer
        return mtp_block_spec

    def build_model(
        self,
        pre_process=True,
        post_process=True,
        vp_stage: Optional[int] = None,
    ):
        model = super().build_model(pre_process, post_process, vp_stage)
        lm_model = model.language_model if hasattr(model, 'language_model') else model
        # The GDN out_norm uses ones-style weights, unlike the zero-centered
        # HC norms, so opt it out of layernorm_zero_centered_gamma.
        for layer in lm_model.decoder.layers:
            if hasattr(layer.self_attention, 'out_norm'):
                out_norm = layer.self_attention.out_norm
                out_norm.zero_centered_gamma = False
                if not is_torch_npu_available():
                    assert hasattr(out_norm, 'zero_centered_gamma')
                if hasattr(out_norm, 'config'):
                    out_norm.config = copy.copy(out_norm.config)
                    out_norm.config.layernorm_zero_centered_gamma = False
        return model
