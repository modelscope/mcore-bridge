try:
    from megatron.core.transformer.experimental_attention_variant.absorbed_mla import \
        AbsorbedMLASelfAttention as McoreAbsorbedMLASelfAttention
except ImportError:
    McoreAbsorbedMLASelfAttention = object

import torch
from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import (gather_from_sequence_parallel_region,
                                                    gather_from_tensor_model_parallel_region,
                                                    scatter_to_sequence_parallel_region)
from megatron.core.utils import deprecate_inference_params


class AbsorbedMLASelfAttention(McoreAbsorbedMLASelfAttention):

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        packed_seq_params=None,
        inference_context=None,
        rotary_pos_emb=None,
        *,
        inference_params=None,
    ):
        """
        Derives absorbed q, compressed q, and compressed kv tensors from `hidden_states`.
        """
        # s = sequence length, b = batch size, h = hidden size
        from megatron.core.utils import get_pg_size
        assert (hidden_states.ndim == 3), f"hidden_states should be 3D, [s, b, h], got {hidden_states.ndim}D"
        if packed_seq_params is not None:
            assert (packed_seq_params.local_cp_size
                    is None), 'dynamic context parallel is not supported with MLA yet and is planned for future. \
            Please disable dynamic context parallel.'

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
            if packed_seq_params.cu_seqlens_kv_padded is not None:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            else:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # =========================================
        # Q down projection
        # =========================================
        if self.config.q_lora_rank is not None:
            # if linear_q_down_proj is ColumnParallelLinear:
            #     q_compressed: [s, b, q_lora_rank / TP]
            # elif linear_q_down_proj is Linear:
            #     q_compressed: [s / TP, b, q_lora_rank]
            q_compressed, _ = self.linear_q_down_proj(hidden_states)

            # When output is sharded (ColumnParallelLinear), two things are needed to be
            # identical to a normal Linear.
            #   1. Manually gather output to restore output dim q_lora_rank;
            #   2. Scatter sequence back to s / TP if sequence-parallel since it was
            #      gathered by ColumnParallelLinear.
            if q_compressed.size(-1) != self.config.q_lora_rank:
                q_compressed = gather_from_tensor_model_parallel_region(q_compressed)
                if self.config.sequence_parallel:
                    q_compressed = scatter_to_sequence_parallel_region(q_compressed)
        else:
            q_compressed = hidden_states

        # =========================================
        # KV down projection
        # =========================================
        # if linear_kv_down_proj is ColumnParallelLinear:
        #     kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim) / TP]
        # elif linear_kv_down_proj is Linear:
        #     kv_combined: [s / TP, b, (kv_lora_rank + qk_pos_emb_head_dim)]
        kv_combined, _ = self.linear_kv_down_proj(hidden_states)
        if kv_combined.size(-1) != self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim:
            # kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim)]
            kv_combined = gather_from_tensor_model_parallel_region(kv_combined)
            # kv_compressed:[s, b, kv_lora_rank], k_pos_emb: [s, b, qk_pos_emb_head_dim]
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1)
            if self.config.sequence_parallel:
                # kv_compressed:[s / TP, b, kv_lora_rank]
                kv_compressed = scatter_to_sequence_parallel_region(kv_compressed)
        else:
            # kv_compressed:[s / TP, b, kv_lora_rank], k_pos_emb: [s / TP, b, qk_pos_emb_head_dim]
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1)
            if get_pg_size(self.tp_group) > 1 and self.config.sequence_parallel:
                # k_pos_emb: [s, b, qk_pos_emb_head_dim]
                k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb, group=self.tp_group)

        if packed_seq_params is not None:
            assert q_compressed.ndim == 3 and q_compressed.size(1) == 1
            assert kv_compressed.ndim == 3 and kv_compressed.size(1) == 1
            assert k_pos_emb.ndim == 3 and k_pos_emb.size(1) == 1
            # If sequence packing, TE expect [t, h, d] shaped qkv input.
            # In Megatron-Core, the qkv shape is [t, 1, h, d].
            # So we need to reshape qkv from [t, 1, h, d] to [t, h, d].
            q_compressed = q_compressed.squeeze(1)
            kv_compressed = kv_compressed.squeeze(1)
            k_pos_emb = k_pos_emb.squeeze(1)

        # =========================================
        # Apply norm
        # =========================================
        if self.config.q_lora_rank is not None:
            # q_compressed: [num_tokens, q_lora_rank]
            q_compressed = self.q_layernorm(q_compressed)

        kv_compressed = self.kv_layernorm(kv_compressed)
        # Because we won't apply V up projection to the compressed KV, so we need to gather it
        # manually.
        if get_pg_size(self.tp_group) > 1 and self.config.sequence_parallel:
            kv_compressed = gather_from_sequence_parallel_region(kv_compressed, group=self.tp_group)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================

        def qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb):
            """
            Apply the up projection and RoPE to the query and key.
            When sequence packing enabled, the input tensors adopt a packed shape of [t, ...];
            otherwise, they maintain the unpacked shape [s, b, ...]. In subsequent code comments,
            we uniformly use [num_tokens, ...] to denote [s, b, ...] or [t, ...] for two cases.
            """
            if self.config.q_lora_rank is not None:
                # q_compressed: [num_tokens, q_lora_rank]
                # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
                q, _ = self.linear_q_up_proj(q_compressed)
            else:
                # q_compressed: [num_tokens, hidden_size]
                # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
                q, _ = self.linear_q_proj(q_compressed)

            # q: [num_tokens, n, q_head_dim]
            q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)

            # [num_tokens, kv_lora_rank] -> [num_tokens, 1, kv_lora_rank]
            kv_compressed = torch.unsqueeze(kv_compressed, -2)
            # [num_tokens, qk_pos_emb_head_dim] -> [num_tokens, 1, qk_pos_emb_head_dim]
            k_pos_emb = torch.unsqueeze(k_pos_emb, -2)

            k_up_weight, _ = self._get_kv_up_weights()

            q_len = q.size()[0]
            if inference_context is not None:
                # add offset to the sequence start for inference
                sequence_start = inference_context.sequence_len_offset
                sequence_end = sequence_start + q_len
                rotary_pos_emb = rotary_pos_emb[sequence_start:sequence_end]
            elif packed_seq_params is None or self.config.context_parallel_size == 1:
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
            q_no_pe, q_pos_emb = torch.split(q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1)

            # Absorb k_up_weight into q_no_pe
            # q_absorbed: [num_tokens, n, kv_lora_rank]
            q_absorbed = torch.einsum('...nd,ndk->...nk', q_no_pe, k_up_weight)
            q_absorbed = q_absorbed.contiguous()
            assert q_absorbed.ndim == q.ndim
            assert q_absorbed.shape[:-1] == q.shape[:-1]
            assert q_absorbed.size(-1) == self.config.kv_lora_rank

            # Apply RoPE to q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
            q_pos_emb = apply_rotary_pos_emb(
                q_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_q,
                cp_group=self.pg_collection.cp,
                mla_rotary_interleaved=True,
            )
            # k_pos_emb:[num_tokens, 1, qk_pos_emb_head_dim]
            k_pos_emb = apply_rotary_pos_emb(
                k_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
                cp_group=self.pg_collection.cp,
                mla_rotary_interleaved=True,
            )

            # query: [num_tokens, n, (kv_lora_rank + qk_pos_emb_head_dim)]
            q_absorbed = torch.cat([q_absorbed, q_pos_emb], dim=-1)
            # key: [num_tokens, 1, (kv_lora_rank + qk_pos_emb_head_dim)]
            kv_compressed = torch.cat([kv_compressed, k_pos_emb], dim=-1)

            assert q_absorbed.is_contiguous()
            assert kv_compressed.is_contiguous()

            return q_absorbed, kv_compressed

        if self.recompute_up_proj:
            quantization = self.config.fp8 or self.config.fp4
            assert not quantization, 'FP8/FP4 is not supported for AbsorbedMLA'
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput(fp8=quantization)
            q_absorbed, kv_compressed = self.qkv_up_checkpoint.checkpoint(qkv_up_proj_and_rope_apply, q_compressed,
                                                                          kv_compressed, k_pos_emb, rotary_pos_emb)
        else:
            assert not self.cache_mla_latents, 'cache_mla_latents is not supported for AbsorbedMLA'
            q_absorbed, kv_compressed = qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb,
                                                                   rotary_pos_emb)

        return q_absorbed, kv_compressed, q_compressed

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
        from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
            _apply_absorbed_v_up_projection, _restore_packed_thd_batch_dim)
        """Forward pass for multi-latent attention with matrix absorption"""
        assert attention_bias is None, 'Attention bias should not be passed into MLA.'
        assert (rotary_pos_cos is None and rotary_pos_sin is None), 'MLA does not support Flash Decoding'
        assert not rotary_pos_cos_sin, 'Flash-infer rope has not been tested with MLA.'
        assert not (self.training and self.cache_mla_latents), 'cache_mla_latents conflicts with training.'
        assert (inference_context is None and inference_params is None), 'Inference is not supported for AbsorbedMLA'

        # =====================
        # Query, Key, and Value
        # =====================
        q_absorbed, kv_compressed, q_compressed = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            packed_seq_params,
            rotary_pos_emb=rotary_pos_emb,
            inference_context=inference_context)

        assert q_absorbed.is_contiguous()
        assert q_compressed.is_contiguous()
        assert kv_compressed.is_contiguous()
        v_up_weight = self._get_v_up_weight()

        # ==================================
        # Core attention computation
        # ==================================
        if self.config.experimental_attention_variant == 'dsa':
            if packed_seq_params is None:
                packed_seq_params = PackedSeqParams()
            # for easy injection of rotary_pos_emb (patch)
            packed_seq_params.rotary_pos_emb = rotary_pos_emb
            if self.config.llm_model_type == 'glm_moe_dsa':
                topk_holder = (
                    self.core_attention._get_index_share_topk_holder(packed_seq_params, attention_mask)
                    if self.core_attention.index_share else None)
                if self.core_attention.skip_topk and self.core_attention.source_layer not in topk_holder:
                    raise ValueError(f'DSA: Layer {self.layer_number} is a "shared" indexer layer but no '
                                     f'"full" layer precedes it in this PP stage. Please adjust '
                                     f'`--pipeline_model_parallel_layout` to ensure each PP stage starts with '
                                     f'a "full" indexer layer. indexer_types: {self.config.hf_config.indexer_types}.')
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                q_absorbed,
                kv_compressed,
                hidden_states,
                q_compressed,
                attention_mask,
                v_up_weight,
                position_ids=position_ids,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                q_absorbed,
                kv_compressed,
                value=None,
                attention_mask=attention_mask,
                x=hidden_states,
                qr=q_compressed,
                up_v_weight=v_up_weight,
                position_ids=position_ids,
                packed_seq_params=packed_seq_params,
                attn_mask_type=self.attn_mask_type,
            )

        # ==================================
        # Apply V up projection
        # ==================================
        core_consumed_v_up_projection = getattr(self.core_attention, 'consumes_absorbed_v_up_projection', False)
        core_attn_out = _apply_absorbed_v_up_projection(
            core_attn_out,
            v_up_weight,
            self.num_attention_heads_per_partition,
            self.config.kv_lora_rank,
            self.config.v_head_dim,
            core_consumed_v_up_projection,
        )

        core_attn_out = _restore_packed_thd_batch_dim(core_attn_out, hidden_states, packed_seq_params)

        assert core_attn_out.ndim == hidden_states.ndim
        assert core_attn_out.shape[0] == (hidden_states.shape[0] * self.config.tensor_model_parallel_size), (
            f"{core_attn_out.shape[0]} != "
            f"{hidden_states.shape[0]} * "
            f"{self.config.tensor_model_parallel_size}")
        assert core_attn_out.shape[1:-1] == hidden_states.shape[1:-1]
        assert core_attn_out.size(-1) == (self.config.v_head_dim * self.num_attention_heads_per_partition)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        # =================
        # Output. [sq, b, h]
        # =================
        output, bias = self.linear_proj(core_attn_out)

        return output, bias
