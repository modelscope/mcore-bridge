# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import torch
from contextlib import contextmanager
from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from typing import Optional

from mcore_bridge.bridge import GPTBridge
from mcore_bridge.utils import Fp8Dequantizer, fp4_to_fp8

from ..constant import ModelType
from ..gpt_model import GPTModel
from ..register import ModelLoader, ModelMeta, register_model
from ..rope import get_rope_inv_freq

try:
    from megatron.core.pipeline_parallel.fine_grained_activation_offload import \
        FineGrainedActivationOffloadingInterface as off_interface
    from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import \
        DSv4HybridSelfAttention as McoreDSv4HybridSelfAttention
    from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import _q_rms_norm
    from megatron.core.typed_torch import apply_module
except ImportError:
    McoreDSv4HybridSelfAttention = object
    _q_rms_norm = None
    apply_module = None
    off_interface = None


@contextmanager
def _patch_YarnRotaryEmbedding(config):
    """Temporarily patch missing rope scaling attrs on config for YarnRotaryEmbedding init.

    YarnRotaryEmbedding requires beta_fast/beta_slow/mscale/mscale_all_dim on config,
    but DeepSeek-V4 HF config may not include them. This context manager sets defaults
    on entry and removes them on exit, keeping the config clean (the resulting
    YarnRotaryEmbedding module will be deleted later anyway).
    """
    defaults = {
        'original_max_position_embeddings': 4096,
        'beta_fast': 32.0,
        'beta_slow': 1.0,
        'mscale': 1.0,
        'mscale_all_dim': 0.0,
    }
    added = []
    for attr, value in defaults.items():
        if getattr(config, attr, None) is None:
            setattr(config, attr, value)
            added.append(attr)
    try:
        yield config
    finally:
        # Restore: remove attrs that were temporarily added
        for attr in added:
            delattr(config, attr)


class DSv4HybridSelfAttention(McoreDSv4HybridSelfAttention):

    def __init__(self, config, *args, **kwargs):
        assert McoreDSv4HybridSelfAttention is not object, (
            'Please install the Megatron-Core dev branch: '
            '`pip install git+https://github.com/NVIDIA/Megatron-LM@dev`')
        with _patch_YarnRotaryEmbedding(config):
            super().__init__(config, *args, **kwargs)
        self.layer_type = self.config.hf_config.layer_types[self.layer_number - 1]
        self.rope_layer_type = 'main' if self.layer_type == 'sliding_attention' else 'compress'

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        rotary_pos_emb=None,
        *,
        inference_params=None,
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # s = sequence length, b = batch size, h = hidden size, n = num attention heads
        # Attention heads [s, b, n*h]
        assert (hidden_states.ndim == 3), f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"
        if packed_seq_params is not None:
            assert (packed_seq_params.local_cp_size
                    is None), 'dynamic_context_parallel is not supported with MLA yet and is planned for future. \
            Please disable dynamic_context_parallel.'

        assert (inference_context is None
                and inference_params is None), 'Inference is not supported for DSv4HybridSelfAttention.'

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            cu_seqlens_q = packed_seq_params.cu_seqlens_q
            cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        # q_compressed: [s, b, q_lora_rank]
        q_compressed, _ = self.linear_q_down_proj(hidden_states)

        kv_compressed = hidden_states

        if packed_seq_params is not None:
            # If sequence packing, TE expect [t, h, d] shaped qkv input.
            # In Megatron-Core, the qkv shape is [t, 1, h, d].
            # So we need to reshape qkv from [t, 1, h, d] to [t, h, d].
            q_compressed = q_compressed.squeeze(1)

        # =========================================
        # Apply norm
        # =========================================

        if self.config.q_lora_rank is not None:
            # q_compressed: [num_tokens, q_lora_rank]
            q_compressed = apply_module(self.q_layernorm)(q_compressed)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================

        def qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, rotary_pos_emb):
            """
            Apply the up projection and RoPE to the query and key.
            When sequence packing enabled, the input tensors adopt a packed shape of [t, ...];
            otherwise, they maintain the unpacked shape [s, b, ...]. In subsequent code comments,
            we uniformly use [num_tokens, ...] to denote [s, b, ...] or [t, ...] for two cases.
            """
            # q_compressed: [num_tokens, q_lora_rank]
            # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
            q, _ = self.linear_q_up_proj(q_compressed)

            # q: [num_tokens, n, q_head_dim]
            q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)
            q = _q_rms_norm(q, self.config.layernorm_epsilon)

            kv, _ = self.linear_kv_proj(kv_compressed)
            kv = self.kv_layernorm(kv)

            # [num_tokens, qk_pos_emb_head_dim] -> [num_tokens, 1, qk_pos_emb_head_dim]
            q_len = q.size()[0]
            if packed_seq_params is None or self.config.context_parallel_size == 1:
                # Shorten rotary_pos_emb to the sequence length when inference_params
                # is not provided. This makes sure we can run forward directly with
                # any sequence length. During training, the sequence length is always
                # the full rotary_pos_emb length, except for sequence packing + CP.
                # When sequence packing and context parallel are both enabled, the
                # position embedding will not split rotary_pos_emb, so it may exceed
                # the sequence length on this CP rank, but we need the full rotary_pos_emb
                # to cover the full sequence, so we do not shorten it here.
                rotary_pos_emb = rotary_pos_emb[0:q_len]

            # q_no_pe: [num_tokens, n, qk_head_dim]
            # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
            pos_dim = self.config.qk_pos_emb_head_dim
            q_no_pe, q_pos_emb = torch.split(q, [q.shape[-1] - pos_dim, pos_dim], dim=-1)

            # RoPE and query (shared for wkv and latent)
            # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
            q_pos_emb = apply_rotary_pos_emb(
                q_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_q,
                cp_group=self.pg_collection.cp,
                mla_rotary_interleaved=True,
                mla_output_remove_interleaving=True,
            )
            # query: [num_tokens, n, (qk_head_dim + v_head_dim)]
            query = torch.cat([q_no_pe, q_pos_emb], dim=-1)

            kv_no_pe, k_pos_emb = torch.split(kv, [kv.size(-1) - pos_dim, pos_dim], dim=-1)

            # k_pos_emb:[num_tokens, 1, qk_pos_emb_head_dim]
            k_pos_emb = apply_rotary_pos_emb(
                k_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
                cp_group=self.pg_collection.cp,
                mla_rotary_interleaved=True,
                mla_output_remove_interleaving=True,
            )

            # Single head: key = value = [num_tokens, 1, v_head_dim]
            kv = torch.cat([kv_no_pe, k_pos_emb], dim=-1).unsqueeze(-2)
            key = kv
            value = kv

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            return query, key, value

        if self.recompute_up_proj:
            quantization = self.config.fp8 or self.config.fp4
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput(fp8=quantization)
            query, key, value = self.qkv_up_checkpoint.checkpoint(qkv_up_proj_and_rope_apply, q_compressed,
                                                                  kv_compressed, rotary_pos_emb)
        else:
            query, key, value = qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, rotary_pos_emb)

        return query, key, value, q_compressed, kv_compressed

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
    ):
        """Forward pass for DeepSeek-v4 Hybrid Attention"""
        rotary_pos_emb = rotary_pos_emb[self.rope_layer_type]
        assert (attention_bias is None), 'Attention bias should not be passed into DSv4HybridAttention.'
        assert (rotary_pos_cos is None
                and rotary_pos_sin is None), 'DSv4HybridAttention does not support Flash Decoding'
        assert (not rotary_pos_cos_sin), 'Flash-infer rope has not been tested with DSv4HybridAttention.'
        assert (inference_context is None
                and inference_params is None), 'Inference is not supported for DSv4HybridAttention.'

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        query, key, value, q_compressed, kv_compressed = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            rotary_pos_emb=rotary_pos_emb,
            inference_context=inference_context,
        )

        # TODO: Currently, TE can only accept contiguous tensors for MLA
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        # ==================================
        # core attention computation
        # ==================================
        # Need corresponding TE change
        core_attn_manager = off_interface(self.offload_core_attention and self.training, query, 'core_attn')
        with core_attn_manager as query:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                packed_seq_params=packed_seq_params,
                x=hidden_states,
                qr=q_compressed,
            )
        core_attn_out = core_attn_manager.group_offload(core_attn_out, forced_released_tensors=[query, key, value])

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        # inverse RoPE on last qk_pos_emb_head_dim of each head
        seq_len = core_attn_out.size(0)
        n_heads = self.num_attention_heads_per_partition
        pos_dim = self.config.qk_pos_emb_head_dim
        core_attn_out = core_attn_out.view(seq_len, core_attn_out.size(1), n_heads, -1)
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if packed_seq:
            cu_seqlens_kv = (
                packed_seq_params.cu_seqlens_kv_padded
                if packed_seq_params.cu_seqlens_kv_padded is not None else packed_seq_params.cu_seqlens_kv)
        else:
            cu_seqlens_kv = None

        content_part, rot_part = torch.split(core_attn_out, [core_attn_out.size(-1) - pos_dim, pos_dim], dim=-1)
        rot_part = apply_rotary_pos_emb(
            rot_part,
            rotary_pos_emb,
            self.config,
            cu_seqlens=cu_seqlens_kv,
            cp_group=self.pg_collection.cp,
            mla_rotary_interleaved=True,
            inverse=True,
            mla_output_remove_interleaving=True,
        )
        core_attn_out = torch.cat([content_part, rot_part], dim=-1)
        core_attn_out = core_attn_out.view(seq_len, core_attn_out.size(1), -1)

        # Grouped output
        core_attn_out = core_attn_out.view(core_attn_out.size(0), core_attn_out.size(1), self.o_local_groups, -1)
        wo_a_weight = self.linear_o_group_proj.view(self.o_local_groups, self.config.o_lora_rank, -1)
        core_attn_out = torch.einsum('...gd,grd->...gr', core_attn_out, wo_a_weight)
        core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)

        # =================
        # Output. [sq, b, h]
        # =================
        attn_proj_manager = off_interface(self.offload_attn_proj, core_attn_out, 'attn_proj')
        with attn_proj_manager as core_attn_out:
            output, bias = self.linear_proj(core_attn_out)
        output = attn_proj_manager.group_offload(output, forced_released_tensors=[core_attn_out])

        return output, bias


class DeepseekV4GPTModel(GPTModel):

    def _init_mla_softmax_scale(self, config):
        pass

    def _get_rotary_pos_emb(self, decoder_input, position_ids, packed_seq_params, inference_context=None):
        rotary_seq_len = RotaryEmbedding.get_rotary_seq_len(self, inference_context, self.decoder, decoder_input,
                                                            self.config, packed_seq_params)
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        compress_rotary_pos_emb = self.compress_rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        rotary_pos_emb = {'main': rotary_pos_emb, 'compress': compress_rotary_pos_emb}
        return rotary_pos_emb, None, None

    def _set_inv_freq(self):
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


class DeepseekV4Loader(ModelLoader):
    model_cls = DeepseekV4GPTModel

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import \
            get_transformer_block_with_experimental_attention_variant_spec
        transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(self.config, vp_stage)
        for layer_spec in transformer_layer_spec.layer_specs:
            layer_spec.submodules.self_attention.module = DSv4HybridSelfAttention
        return transformer_layer_spec


class DeepseekV4Bridge(GPTBridge):
    hf_mtp_prefix = 'model.mtp'
    hf_embed_key = 'model.embed.weight'
    hf_attn_prefix = 'attn'
    hf_mlp_prefix = 'ffn'
    hf_lm_head_key = 'model.head.weight'
    hf_score_key = 'model.score.weight'
    hf_input_layernorm_key = 'attn_norm.weight'
    hf_post_attention_layernorm_key = 'ffn_norm.weight'
    hf_expert_bias_key = 'gate.bias'

    def _convert_hf_state_dict(self, hf_state_dict, to_mcore):
        res = super()._convert_hf_state_dict(hf_state_dict, to_mcore)
        if to_mcore:
            res = self._add_prefix(res, 'model.')
            new_res = {}
            for k, v in res.items():
                if k.endswith('.scale'):
                    k = k[:-len('.scale')] + '.weight_scale_inv'
                new_res[k] = v
            res = new_res
        else:
            res = self._remove_prefix(res, 'model.')
            new_res = {}
            for k, v in res.items():
                if k.endswith('.weight_scale_inv'):
                    k = k[:-len('.weight_scale_inv')] + '.scale'
                new_res[k] = v
            res = new_res
        return res

    def _set_moe_state(
        self,
        mg_mlp,
        hf_state_dict,
        hf_prefix: str,
        layer_idx: int,
        to_mcore: bool,
        is_mtp: bool = False,
    ):
        if to_mcore:
            hf_state_dict = {
                k.replace('.w1.', '.gate_proj.').replace('.w3.', '.up_proj.').replace('.w2.', '.down_proj.'): v
                for k, v in hf_state_dict.items()
            }
        hf_state_dict = super()._set_moe_state(mg_mlp, hf_state_dict, hf_prefix, layer_idx, to_mcore, is_mtp)
        if not to_mcore:
            hf_state_dict = {
                k.replace('.gate_proj.', '.w1.').replace('.up_proj.', '.w3.').replace('.down_proj.', '.w2.'): v
                for k, v in hf_state_dict.items()
            }
        return hf_state_dict

    def _set_mla_attn_state(
        self,
        mg_attn,
        hf_state_dict,
        hf_prefix: str,
        layer_idx: int,
        to_mcore: bool,
    ):
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        self._set_state_dict(mg_attn, 'linear_proj.weight', hf_state_dict, 'wo_b.weight', to_mcore)
        self._set_state_dict(mg_attn, 'linear_o_group_proj', hf_state_dict, 'wo_a.weight', to_mcore)
        self._set_state_dict(mg_attn, 'linear_q_down_proj.weight', hf_state_dict, 'wq_a.weight', to_mcore)
        self._set_state_dict(mg_attn, 'linear_q_up_proj.weight', hf_state_dict, 'wq_b.weight', to_mcore)
        self._set_state_dict(mg_attn, 'linear_kv_proj.weight', hf_state_dict, 'wkv.weight', to_mcore)
        self._set_state_dict(mg_attn, 'core_attention.attn_sink', hf_state_dict, 'attn_sink', to_mcore)
        if self.config.qk_layernorm:
            self._set_state_dict(mg_attn, 'q_layernorm.weight', hf_state_dict, 'q_norm.weight', to_mcore)
            self._set_state_dict(mg_attn, 'kv_layernorm.weight', hf_state_dict, 'kv_norm.weight', to_mcore)
        has_compressor = False if mg_attn is None else mg_attn.core_attention.compressor is not None
        has_indexer = False if mg_attn is None else mg_attn.core_attention.indexer is not None
        has_compressor = self._reduce_tensor_pp_group(has_compressor, to_mcore)
        has_indexer = self._reduce_tensor_pp_group(has_indexer, to_mcore)
        if has_compressor:
            for mg_key, hf_key in zip(['ape', 'linear_wkv.weight', 'linear_wgate.weight', 'norm.weight'],
                                      ['ape', 'wkv.weight', 'wgate.weight', 'norm.weight']):
                self._set_state_dict(mg_attn, f'core_attention.compressor.{mg_key}', hf_state_dict,
                                     f'compressor.{hf_key}', to_mcore)
        if has_indexer:
            for mg_key, hf_key in zip(['linear_wq_b.weight', 'linear_weights_proj.weight'],
                                      ['wq_b.weight', 'weights_proj.weight']):
                self._set_state_dict(mg_attn, f'core_attention.indexer.{mg_key}', hf_state_dict, f'indexer.{hf_key}',
                                     to_mcore)
            for mg_key, hf_key in zip(['ape', 'linear_wkv.weight', 'linear_wgate.weight', 'norm.weight'],
                                      ['ape', 'wkv.weight', 'wgate.weight', 'norm.weight']):
                self._set_state_dict(mg_attn, f'core_attention.indexer.compressor.{mg_key}', hf_state_dict,
                                     f'indexer.compressor.{hf_key}', to_mcore)

        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _set_final_layernorm(self, lm_model, hf_state_dict, to_mcore):
        super()._set_final_layernorm(lm_model, hf_state_dict, to_mcore)
        for key in ['hc_head_base', 'hc_head_fn', 'hc_head_scale']:
            self._set_state_dict(lm_model, f'decoder.{key}', hf_state_dict, f'model.{key}', to_mcore)

    def _set_router(self, mg_mlp, hf_state_dict, to_mcore, **kwargs):
        is_hash_layer = False if mg_mlp is None else mg_mlp.router.is_hash_layer
        is_hash_layer = self._reduce_tensor_pp_group(is_hash_layer, to_mcore)
        if is_hash_layer:
            self._set_state_dict(mg_mlp, 'router.tid2eid', hf_state_dict, 'gate.tid2eid', to_mcore)
            kwargs['moe_router_enable_expert_bias'] = False
        super()._set_router(mg_mlp, hf_state_dict, to_mcore, **kwargs)

    def _convert_mtp_extra(self, mtp_layer, hf_state_dict, to_mcore, origin_hf_state_dict):
        for key in ['enorm.weight', 'hnorm.weight', 'e_proj.weight', 'h_proj.weight']:
            self._set_state_dict(mtp_layer, key, hf_state_dict, key, to_mcore)
        self._set_state_dict(mtp_layer, 'final_layernorm.weight', hf_state_dict, 'norm.weight', to_mcore)
        for key in ['hc_head_base', 'hc_head_fn', 'hc_head_scale']:
            self._set_state_dict(mtp_layer, key, hf_state_dict, key, to_mcore)

    def _convert_mtp_embeds(self, lm_model, hf_state_dict, to_mcore):
        if not to_mcore:
            self._set_state_dict(lm_model, 'embedding.word_embeddings.weight', hf_state_dict, 'emb.tok_emb.weight',
                                 to_mcore)
            if self.config.untie_embeddings_and_output_weights:
                self._set_state_dict(lm_model, 'output_layer.weight', hf_state_dict, 'head.weight', to_mcore)

    def _set_param(self, param, tensor, scale_inv):
        is_fp4 = tensor.dtype == torch.int8 and tensor.shape[-1] * 2 == param.shape[-1]
        if not is_fp4:
            return super()._set_param(param, tensor, scale_inv)
        tensor = fp4_to_fp8(tensor)
        tensor = tensor.reshape(*param.shape)
        scale_inv = scale_inv.reshape(-1, scale_inv.shape[-1])
        tensor = Fp8Dequantizer(block_size='auto').convert(tensor, scale_inv)
        if self._is_fp8_param(param):
            param._high_precision_init_val.copy_(tensor)
        param.data.copy_(tensor)


register_model(
    ModelMeta(
        ModelType.deepseek_v4,
        ['deepseek_v4'],
        bridge_cls=DeepseekV4Bridge,
        loader=DeepseekV4Loader,
    ))
