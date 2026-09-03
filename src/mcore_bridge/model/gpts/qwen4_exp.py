# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import contextmanager
from copy import deepcopy
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm, TERowParallelLinear
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.ssm.gated_delta_net import GatedDeltaNetSubmodules
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.packed_seq_params import PackedSeqParams

from transformers.utils import is_torch_npu_available
from typing import List, Optional

from mcore_bridge.utils import get_env_args, get_local_layer_specs, get_logger
from mcore_bridge.utils.megatron_utils import reconstruct_tensor_cp

from ..modules import (GatedDeltaNet, QSAIndexer, QSASparseCoreAttention, Qwen4ExpTextGatedResidual,
                       Qwen4ExpTextPLELayer, TransformerBlock, TransformerLayer, qsa_sparse_supported)
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
        is_linear_attention = config.linear_attention_freq[self.layer_number - 1]
        if not is_linear_attention and config.indexer_n_heads is not None:
            self.self_attention.indexer = QSAIndexer(config)
            if qsa_sparse_supported(config.kv_channels or 0):
                attn = self.self_attention
                attn.core_attention = QSASparseCoreAttention(
                    attn.core_attention, config, softmax_scale=getattr(config, 'softmax_scale', None))
        self.attn_hyper_connection = Qwen4ExpTextGatedResidual(config)
        self.mlp_hyper_connection = Qwen4ExpTextGatedResidual(config)

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
        path, or ``None`` for full attention. CP needs the allgather comm type
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

        # From here the mask path is not an option, so every failure raises instead of
        # silently degrading to dense attention (which would diverge from the sparse
        # rollout without telling anyone).
        if not sparse_ok:
            raise RuntimeError(
                f'QSA needs the sparse kernel here ({"packing/thd" if is_thd else f"CP={cp_size}"}), '
                'but QSASparseCoreAttention was not installed -- triton is missing or '
                f'kv_channels={getattr(self.config, "kv_channels", None)} is not a power of two. '
                'Use --padding_free false with context_parallel_size 1 to take the bool-mask path.')
        if cp_size > 1 and getattr(self.config, 'cp_comm_type', None) != 'allgather':
            raise RuntimeError(
                f"QSA sparse selection with context_parallel_size={cp_size} requires "
                f"cp_comm_type='allgather' (got {getattr(self.config, 'cp_comm_type', None)!r}): the "
                'selection has to see every key before attention runs, which ring/p2p cannot provide.')
        rotary_pos_emb = attn_kwargs.get('rotary_pos_emb')
        if rotary_pos_emb is None:
            raise RuntimeError(
                'QSA sparse selection needs rotary_pos_emb (blocks rotate at their first '
                'token position) but it was not passed to the layer.')
        if is_thd:
            indices = self._qsa_select_indices_thd(
                hidden_states, rotary_pos_emb, packed_seq_params, position_ids)
        else:
            indices = self._qsa_select_indices_sbhd(hidden_states, rotary_pos_emb)
        return indices, True

    def _qsa_select_indices_sbhd(self, hidden_states, rotary_pos_emb):
        if self.config.sequence_parallel and self.config.tensor_model_parallel_size > 1:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False)
        if self.config.context_parallel_size > 1:
            hidden_states = reconstruct_tensor_cp(hidden_states, None, dim=0)
            rotary_pos_emb = reconstruct_tensor_cp(rotary_pos_emb, None, dim=0)
        return self.self_attention.indexer.selection_as_token_indices(hidden_states, rotary_pos_emb)

    def _qsa_select_indices_thd(self, hidden_states, rotary_pos_emb, packed_seq_params,
                                position_ids=None):
        if self.config.sequence_parallel and self.config.tensor_model_parallel_size > 1:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False)
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
            hidden_states = reconstruct_tensor_cp(hidden_states, psp_for_cp, dim=0)
        # Per-token rotary angles. Without rope fusion gpt_model already indexes
        # the freq table by position_ids, so what arrives is per-token (zigzag-
        # sharded under CP -- undo it like hidden). With fusion the raw table
        # arrives and must be indexed by the (CP-reconstructed) per-doc ids.
        freqs = rotary_pos_emb
        fused_table = freqs.shape[0] != hidden_states.shape[0]
        if self.config.context_parallel_size > 1:
            if fused_table:
                if position_ids is None:
                    raise RuntimeError(
                        'QSA thd selection under CP needs position_ids to index the fused rotary '
                        'table (apply_rope_fusion=true hands over the raw table, not per-token '
                        'freqs). Pass position_ids, or set --apply_rope_fusion false.')
                pos = reconstruct_tensor_cp(position_ids, psp_for_cp, dim=1)
                freqs = freqs[pos.reshape(-1)]
            else:
                freqs = reconstruct_tensor_cp(freqs, psp_for_cp, dim=0)
        elif fused_table:
            # Same problem without CP, and here there is no reconstruct step to hide
            # behind: the indexer would slice the raw table's first T rows, treating
            # row i as token i's angle. In a packed batch token i sits at in-document
            # position i - cu[doc], so those angles belong to the wrong positions --
            # silently degrading the selection instead of failing.
            raise RuntimeError(
                f'QSA thd selection got a fused rotary table ({freqs.shape[0]} rows for '
                f'{hidden_states.shape[0]} tokens): apply_rope_fusion=true hands over the raw '
                'table rather than per-token freqs. Set --apply_rope_fusion false.')
        # the CP reconstruct (like TE's thd kernels) works in the padded pack
        # space, so align against the padded cu when present
        cu = packed_seq_params.cu_seqlens_q_padded
        if cu is None:
            cu = packed_seq_params.cu_seqlens_q
        if cu is None:
            raise RuntimeError(
                'QSA thd selection needs packed_seq_params.cu_seqlens_q to find document '
                'boundaries, but it is missing.')
        cu = Qwen4ExpTextPLELayer._normalize_cu_seqlens(cu, hidden_states.shape[0])
        hidden_tok = hidden_states.reshape(hidden_states.shape[0], -1)
        return self.self_attention.indexer.select_token_indices_thd(hidden_tok, freqs, cu)

    def _qsa_select_mask(self, hidden_states, attn_kwargs):
        # Bool-mask QSA on TE's `arbitrary` mask. Only reached for sbhd with CP==1 --
        # _qsa_selection() routes thd and CP>1 to the kernel, because TE rejects an
        # arbitrary mask under thd and this path never gathers keys across CP ranks.
        # Returning None means full attention, which here only happens when the
        # sequence is short enough that selection is a no-op anyway (selection_as_mask
        # short-circuits at max_blocks <= block_topk).
        indexer = getattr(self.self_attention, 'indexer', None)
        if indexer is None:
            return None
        rotary_pos_emb = attn_kwargs.get('rotary_pos_emb')
        if rotary_pos_emb is None:
            raise RuntimeError(
                'QSA bool-mask selection needs rotary_pos_emb (blocks rotate at their first '
                'token position) but it was not passed to the layer.')
        if self.config.sequence_parallel and self.config.tensor_model_parallel_size > 1:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False)
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

    def _set_layer_ple(self, mg_layer, hf_state_dict, to_mcore: bool):
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
        ple_offloaded = self._reduce_tensor_pp_group(
            ple is not None and ple.ple_embedding.cpu_offload, to_mcore)
        skip_ngram_state = not to_mcore and not self._is_saving and (
            self._peft_format or ple_offloaded)
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
        elif not skip_ngram_state and ple is not None:
            ple.ple_embedding.export_table_to_hf(hf_state_dict)
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
            self._set_layer_ple(mg_layer, hf_state_dict, to_mcore)
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
        # Context parallelism: PLE gathers the full sequence internally (undoing the
        # CP zigzag) and GDN carries its own CP handling (a2a CP<->HP plus CP-aware
        # cu_seqlens), so CP is no longer blanket-rejected here. Left unasserted so
        # it can be exercised; QSA layers run dense attention, which mcore's
        # attention already supports under CP.
        if getattr(config, 'mtp_num_layers', None):
            raise NotImplementedError('Qwen4-Exp MTP is not supported yet')
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
