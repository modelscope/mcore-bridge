# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import contextmanager
from copy import deepcopy
from megatron.core import mpu
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm, TERowParallelLinear
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.ssm.gated_delta_net import GatedDeltaNetSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from transformers.utils import is_torch_npu_available
from typing import List, Optional

from mcore_bridge.utils import get_local_layer_specs, get_logger

from ..modules import (GatedDeltaNet, QSAIndexer, Qwen4ExpTextGatedResidual, Qwen4ExpTextPLELayer, TransformerBlock,
                       TransformerLayer)
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
        output_gate_type = getattr(self.config, 'output_gate_type', None)
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
        if not is_linear_attention and getattr(config, 'indexer_n_heads', None) is not None:
            self.self_attention.indexer = QSAIndexer(config)
        self.attn_hyper_connection = Qwen4ExpTextGatedResidual(config)
        self.mlp_hyper_connection = Qwen4ExpTextGatedResidual(config)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        attention_mask = kwargs.get('attention_mask')
        packed_seq_params = kwargs.get('packed_seq_params')
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
        qsa_mask = self._qsa_select_mask(hidden_states, attn_kwargs)
        if qsa_mask is not None:
            # failed to full atention
            attn_kwargs = dict(attn_kwargs, attention_mask=qsa_mask)
        with self._patch_apply_rotary_pos_emb(), self._qsa_arbitrary_mask(qsa_mask is not None):
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

    def _warn_qsa_fallback_once(self, reason: str) -> None:
        # warning_once dedupes on the message, so every QSA layer can call this and
        # the user still sees it exactly once per distinct reason.
        get_logger().warning_once(f'QSA sparse selection disabled: {reason}')

    def _qsa_select_mask(self, hidden_states, attn_kwargs):
        # return None means full attention
        # TODO: support padding_free & cp
        indexer = getattr(self.self_attention, 'indexer', None)
        if indexer is None:
            return None
        if attn_kwargs.get('packed_seq_params') is not None:
            self._warn_qsa_fallback_once(
                'packing/padding_free is enabled (qkv_format=thd), which TE cannot combine with a '
                'custom attention mask. QSA layers fall back to full attention -- training will '
                'differ from sparse inference beyond the indexer budget. Pass `--padding_free false` '
                'to enable QSA sparse selection.')
            return None
        if self.config.context_parallel_size > 1:
            self._warn_qsa_fallback_once(
                f'context_parallel_size={self.config.context_parallel_size} > 1 is not supported by '
                'the QSA indexer yet (block pooling needs keys from other CP ranks). QSA layers fall '
                'back to full attention -- training will differ from sparse inference beyond the '
                'indexer budget.')
            return None
        rotary_pos_emb = attn_kwargs.get('rotary_pos_emb')
        if rotary_pos_emb is None:
            return None
        return indexer.select_mask(hidden_states, rotary_pos_emb)


class Qwen4ExpTransformerBlock(TransformerBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.config
        hc_count = getattr(config, 'hc_count', 0) or 0
        if hc_count > 1 and self.has_final_layernorm_in_this_stage():
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
    # (mcore attribute name, hf checkpoint suffix). Both sides now use the
    # transformers buffer names; the table is kept so these buffers stay on
    # the dedicated PLE conversion path (pp broadcast, no TP split).
    _PLE_NGRAM_BUFFERS = (
        ('layer_multipliers', 'layer_multipliers'),
        ('ngram_heads_offsets', 'ngram_heads_offsets'),
        ('ngram_heads_vocab_sizes', 'ngram_heads_vocab_sizes'),
    )

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

    def _set_ple_ngram_embedding(self, ple, hf_state_dict, to_mcore: bool, pp_src_rank: int):
        # The checkpoint shards the (padded) table into `parts` uniform row
        # blocks, so shard boundaries must be derived from the padded size.
        total = parts = dim = None
        if ple is not None:
            total = ple.ple_embedding.ngram_embedding.num_embeddings
            parts = ple.ple_embedding.split_ngram_parts
            dim = ple.ple_embedding.head_dim
        if not to_mcore and self.pp_size > 1:
            obj = [(total, parts, dim)]
            dist.broadcast_object_list(obj, src=pp_src_rank, group=self.pp_group)
            total, parts, dim = obj[0]
        if to_mcore and ple is None:
            return
        shard_size = (total + parts - 1) // parts
        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_ranks = dist.get_process_group_ranks(self.tp_group)
        emb = ple.ple_embedding.ngram_embedding if ple is not None else None
        dtype = emb.weight.dtype if emb is not None else self.config.params_dtype
        device = emb.weight.device if emb is not None else torch.cuda.current_device()
        per_partition = (emb.num_embeddings_per_partition if emb is not None else (total + tp_size - 1) // tp_size)
        tp_start = tp_rank * per_partition if emb is not None else 0
        tp_end = min((tp_rank + 1) * per_partition, total) if emb is not None else 0
        if to_mcore:
            for i in range(parts):
                key = f'ple.ple_embedding.ngram_embedding.shard_{i}.weight'
                if key not in hf_state_dict:
                    continue
                cs, ce = i * shard_size, min((i + 1) * shard_size, total)
                s, e = max(cs, tp_start), min(ce, tp_end)
                if s < e:
                    weight = hf_state_dict[key].load()
                    emb.weight.data[s - tp_start:e - tp_start] = weight[s - cs:e - cs].to(emb.weight.dtype)
        else:
            for i in range(parts):
                cs, ce = i * shard_size, min((i + 1) * shard_size, total)
                pieces = []
                for r in range(tp_size):
                    r_start = r * per_partition
                    r_end = min((r + 1) * per_partition, total)
                    s, e = max(cs, r_start), min(ce, r_end)
                    if s >= e:
                        continue
                    if emb is not None and r == tp_rank:
                        piece = emb.weight.data[s - tp_start:e - tp_start].clone()
                    else:
                        piece = torch.empty(e - s, dim, dtype=dtype, device=device)
                    dist.broadcast(piece, src=tp_ranks[r], group=self.tp_group)
                    pieces.append(piece)
                shard = torch.cat(pieces, dim=0)
                if self.pp_size > 1:
                    dist.broadcast(shard, src=pp_src_rank, group=self.pp_group)
                # Written directly into the state dict (bypasses _get_weight,
                # which normally applies _target_device).
                if self._target_device is not None:
                    shard = shard.to(self._target_device)
                hf_state_dict[f'ple.ple_embedding.ngram_embedding.shard_{i}.weight'] = shard

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
        for mg_buf, hf_buf in self._PLE_NGRAM_BUFFERS:
            if to_mcore:
                buffer = getattr(ple.ple_embedding, mg_buf)
                buffer.copy_(hf_state_dict[f'ple.ple_embedding.{hf_buf}'].load().to(buffer.device))
            else:
                tensor = getattr(ple.ple_embedding, mg_buf).data.clone() if ple is not None else None
                if self.pp_size > 1:
                    obj = [tensor]
                    dist.broadcast_object_list(obj, src=pp_src_rank, group=self.pp_group)
                    tensor = obj[0]
                # Written directly into the state dict (bypasses _get_weight,
                # which normally applies _target_device).
                if tensor is not None and self._target_device is not None:
                    tensor = tensor.to(self._target_device)
                hf_state_dict[f'ple.ple_embedding.{hf_buf}'] = tensor
        self._set_ple_ngram_embedding(ple, hf_state_dict, to_mcore, pp_src_rank)
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
        lm_model = getattr(mg_model, 'language_model') if self.is_multimodal else mg_model
        hc_count = getattr(self.config, 'hc_count', 0) or 0
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
