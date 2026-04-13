import megatron.core
import peft
import sys
import torch
import torch.nn.functional as F
from functools import partial
from megatron.core import InferenceParams, mpu, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.extensions.transformer_engine import TEGroupedLinear, TELinear
from megatron.core.models.common.embeddings import rope_utils
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.rotary_pos_embedding import MultimodalRotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import (gather_from_sequence_parallel_region,
                                                    gather_from_tensor_model_parallel_region,
                                                    scatter_to_sequence_parallel_region)
from megatron.core.transformer import TransformerLayer
from megatron.core.transformer.multi_latent_attention import MLASelfAttention, MultiLatentAttention
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
from megatron.core.utils import deprecate_inference_params
from packaging import version
from peft.tuners.tuners_utils import BaseTuner
from torch import nn
from transformers.utils import is_torch_npu_available
from typing import Callable, List, Optional, Tuple

from mcore_bridge.utils import get_logger, is_flash_attn_3_available

mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')
logger = get_logger()


def _patch_flash_attn():
    # flash_attention_3
    if is_flash_attn_3_available():
        import flash_attn_interface
        sys.modules['flash_attn_3.flash_attn_interface'] = flash_attn_interface


def _patch_transformer_engine():
    import transformer_engine
    try:
        from transformer_engine.pytorch.attention import apply_rotary_pos_emb
    except ImportError:
        try:
            transformer_engine.pytorch.attention.apply_rotary_pos_emb = (
                transformer_engine.pytorch.attention.rope.apply_rotary_pos_emb)
        except (ImportError, AttributeError):
            logger.warning('Failed to patch apply_rotary_pos_emb.')
    try:
        from transformer_engine.pytorch.attention import _SplitAlongDim
    except ImportError:
        try:
            transformer_engine.pytorch.attention._SplitAlongDim = (transformer_engine.pytorch.utils.SplitAlongDim)
        except (ImportError, AttributeError):
            logger.warning('Failed to patch _SplitAlongDim.')


def _patch_mla_attention():
    # support thd

    # Code borrowed from NVIDIA/Megatron-LM
    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        **kwargs,
    ):
        """Forward pass for multi-latent attention"""
        assert attention_bias is None, 'Attention bias should not be passed into MLA.'
        assert (rotary_pos_cos is None and rotary_pos_sin is None), 'MLA does not support Flash Decoding'

        # hidden_states: [sq, b, h]

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        # query: [96, 1, 16, 128], key:[96, 1, 16, 128], value:[96, 1, 16, 128]
        query, key, value, q_compressed, kv_compressed = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            rotary_pos_emb=rotary_pos_emb,
            inference_context=inference_context,
        )

        # ===================================================
        # Adjust key, value for inference
        # ===================================================
        # rotary_pos_emb = None
        if mcore_013:
            query, key, value, _, attn_mask_type, _ = self._adjust_key_value_for_inference(
                inference_context, query, key, value, rotary_pos_emb=None)
        else:
            query, key, value, _, attn_mask_type = self._adjust_key_value_for_inference(
                inference_context, query, key, value, rotary_pos_emb=None)

        # TODO: Currently, TE can only accept contiguous tensors for MLA
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        # ==================================
        # core attention computation
        # ==================================
        # Need corresponding TE change
        thd_qkv_format = packed_seq_params and packed_seq_params.qkv_format == 'thd'
        v_dim = value.shape[-1]
        if thd_qkv_format and query.shape[-1] != v_dim:
            value = F.pad(value, [0, query.shape[-1] - v_dim])
            self.core_attention.hidden_size_per_attention_head_v = value.shape[-1]
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, packed_seq_params=packed_seq_params)
        else:
            extra_kwargs = {}
            if self.config.experimental_attention_variant == 'dsa':
                # For dsa we need to pass in the original hidden states and the compressed
                # query representation.
                extra_kwargs['x'] = hidden_states
                extra_kwargs['qr'] = q_compressed
                # for easy injection of rotary_pos_emb (patch)
                packed_seq_params = (packed_seq_params, rotary_pos_emb)
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                packed_seq_params=packed_seq_params,
                attn_mask_type=attn_mask_type,
                **extra_kwargs,
            )
        if thd_qkv_format:
            if core_attn_out.ndim == 2:
                core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-1], -1, value.shape[-1])
            if query.shape[-1] != v_dim:
                core_attn_out = core_attn_out[..., :v_dim]
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        # =================
        # Output. [sq, b, h]
        # =================
        output, bias = self.linear_proj(core_attn_out)

        return output, bias

    MultiLatentAttention.forward = forward

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
        assert (hidden_states.ndim == 3), f'hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D'

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if packed_seq_params is not None:
            cu_seqlens_q = packed_seq_params.cu_seqlens_q
            cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # =========================================
        # QKV down projection and layernorm
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

            q_compressed = self.q_layernorm(q_compressed)
        else:
            q_compressed = hidden_states

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
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                # k_pos_emb: [s, b, qk_pos_emb_head_dim]
                k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb)

        kv_compressed = self.kv_layernorm(kv_compressed)

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

            # kv: [num_tokens, n * (qk_head_dim + v_head_dim)]
            kv, _ = self.linear_kv_up_proj(kv_compressed)

            # kv: [num_tokens, n, (qk_head_dim + v_head_dim)]
            kv = kv.view(
                *kv.size()[:-1],
                self.num_attention_heads_per_partition,
                self.config.qk_head_dim + self.config.v_head_dim,
            )

            q_len = q.size()[0]
            if inference_context is not None:
                # add offset to the sequence start for inference
                sequence_start = inference_context.sequence_len_offset
                sequence_end = sequence_start + q_len
                rotary_pos_emb = rotary_pos_emb[sequence_start:sequence_end]
            # Remove the else branch to fix cp.

            # [num_tokens, qk_pos_emb_head_dim] -> [num_tokens, 1, qk_pos_emb_head_dim]
            k_pos_emb = torch.unsqueeze(k_pos_emb, -2)

            # q_no_pe: [num_tokens, n, qk_head_dim]
            # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
            q_no_pe, q_pos_emb = torch.split(q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1)

            # k_no_pe: [num_tokens, n, qk_head_dim]
            # value: [num_tokens, n, v_head_dim]
            k_no_pe, value = torch.split(kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1)
            # This function will be patched and supports mscale.
            # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
            q_pos_emb = apply_rotary_pos_emb(
                q_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_q,
            )
            # k_pos_emb:[num_tokens, 1, qk_pos_emb_head_dim]
            k_pos_emb = apply_rotary_pos_emb(
                k_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
            )

            # query: [num_tokens, n, (qk_head_dim + v_head_dim)]
            query = torch.cat([q_no_pe, q_pos_emb], dim=-1)

            # key: [num_tokens, n, (qk_head_dim + v_head_dim)]Add commentMore actions
            if k_pos_emb.ndim == 4:
                k_pos_emb = k_pos_emb.expand(-1, -1, self.num_attention_heads_per_partition, -1)
            else:
                assert k_pos_emb.ndim == 3
                k_pos_emb = k_pos_emb.expand(-1, self.num_attention_heads_per_partition, -1)
            key = torch.cat([k_no_pe, k_pos_emb], dim=-1)

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            return query, key, value

        if packed_seq_params is not None:
            # If sequence packing, TE expect [t, h, d] shaped qkv input.
            # In Megatron-Core, the qkv shape is [t, 1, h, d].
            # So we need to reshape qkv from [t, 1, h, d] to [t, h, d].
            q_compressed = q_compressed.squeeze(1)
            kv_compressed = kv_compressed.squeeze(1)
            k_pos_emb = k_pos_emb.squeeze(1)

        if self.recompute_up_proj:
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            query, key, value = self.qkv_up_checkpoint.checkpoint(qkv_up_proj_and_rope_apply, q_compressed,
                                                                  kv_compressed, k_pos_emb, rotary_pos_emb)
        else:
            query, key, value = qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb)

        return query, key, value, q_compressed, kv_compressed

    MLASelfAttention.get_query_key_value_tensors = get_query_key_value_tensors


def _patch_peft_BaseTuner():
    _origin_get_tied_target_modules = BaseTuner._get_tied_target_modules

    def _get_tied_target_modules(self, model: nn.Module) -> List[str]:
        try:
            return _origin_get_tied_target_modules(self, model)
        except AttributeError:
            tied_target_modules = []
            if model.share_embeddings_and_output_weights:
                for target_module in self.targeted_module_names:
                    if target_module.split('.')[-1] in ['output_layer', 'embedding']:
                        tied_target_modules.append(target_module)
            return tied_target_modules

    BaseTuner._get_tied_target_modules = _get_tied_target_modules


def _patch_TEGroupedLinear():

    def sharded_state_dict(
            self,
            prefix: str = '',
            sharded_offsets: Tuple[Tuple[int, int, int]] = (),
            metadata: Optional[dict] = None,
    ):
        return self._sharded_state_dict_grouped(None, prefix, sharded_offsets, metadata)

    TEGroupedLinear.sharded_state_dict = sharded_state_dict


def _patch_mtp():

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        context: torch.Tensor = None,
        context_mask: torch.Tensor = None,
        rotary_pos_emb: torch.Tensor = None,
        rotary_pos_cos: torch.Tensor = None,
        rotary_pos_sin: torch.Tensor = None,
        attention_bias: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        sequence_len_offset: torch.Tensor = None,
        embedding=None,
    ):
        """
        Execute the forward pass through the Multi-Token Prediction (MTP) layer.

        Args:
            input_ids (Tensor): Input token IDs .
            position_ids (Tensor): Positional IDs of the input tokens.
            hidden_states (Tensor): Hidden states tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention, if applicable.
            context_mask (Tensor, optional): Mask for cross-attention context, if applicable.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Tensor, optional): Cosine component of rotary positional embeddings.
            rotary_pos_sin (Tensor, optional): Sine component of rotary positional embeddings.
            sequence_len_offset (Tensor, optional): Offset for sequence length, if applicable.
            embedding (Callable): The embedding module from gpt model to compute the decoder input.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """
        # TODO: Multimodal compatible
        assert context is None, 'multi token prediction + cross attention is not yet supported.'
        input_ids, position_ids, decoder_input, hidden_states = self._get_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            embedding=embedding,
            packed_seq_params=packed_seq_params,
            hidden_states=hidden_states,
        )
        assert not self.transformer_layer.self_attention.config.apply_rope_fusion
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if self.config.position_embedding_type == 'rope' and packed_seq:
            assert position_ids.shape[0] == 1, f'position_ids.shape: {position_ids.shape}'
            rotary_pos_emb = rotary_pos_emb[position_ids[0]]
        else:
            # mrope or not packed_seq
            rotary_pos_emb = torch.roll(rotary_pos_emb, shifts=-self.layer_number, dims=0)
        if self.config.recompute_granularity == 'full' and self.training:
            hidden_states = self._checkpointed_forward(
                partial(
                    self._proj_and_transformer_layer,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                ),
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
            )
        else:
            hidden_states = self._proj_and_transformer_layer(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
        return hidden_states, input_ids, position_ids

    MultiTokenPredictionLayer.forward = forward

    def _get_embeddings(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        embedding: Callable,
        hidden_states: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        from megatron.core.transformer.multi_token_prediction import roll_tensor
        from megatron.core.utils import make_viewless_tensor

        # Calc logits for the current Multi-Token Prediction (MTP) layers.
        input_ids, _ = roll_tensor(
            input_ids,
            shifts=-1,
            dims=-1,
            cp_group=self.cp_group,
            packed_seq_params=packed_seq_params,
        )
        position_ids, _ = roll_tensor(
            position_ids,
            shifts=-1,
            dims=-1,
            cp_group=self.cp_group,
            packed_seq_params=packed_seq_params,
        )
        # embedding
        if isinstance(embedding, tuple):
            embedding, decoder_input = embedding
        else:
            decoder_input = None
        if decoder_input is None:
            decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            enable_sp = self.config.sequence_parallel and self.config.tensor_model_parallel_size > 1
            if enable_sp:
                decoder_input = gather_from_sequence_parallel_region(decoder_input)
            decoder_input, _ = roll_tensor(
                decoder_input.transpose(0, 2),
                shifts=-1,
                dims=-1,
                cp_group=self.cp_group,
                packed_seq_params=packed_seq_params,
            )
            decoder_input = decoder_input.transpose(0, 2).contiguous()
            if enable_sp:
                decoder_input = scatter_to_sequence_parallel_region(decoder_input)
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        return input_ids, position_ids, decoder_input, hidden_states

    MultiTokenPredictionLayer._get_embeddings = _get_embeddings


def _patch_peft_ModulesToSaveWrapper():
    if version.parse(peft.__version__) >= version.parse('0.16'):
        from peft.utils import other as peft_module
    else:
        from peft.tuners import tuners_utils as peft_module

    from .tuners.utils import tuners_sharded_state_dict

    OriginModulesToSaveWrapper = peft_module.ModulesToSaveWrapper

    class ModulesToSaveWrapper(OriginModulesToSaveWrapper):

        def sharded_state_dict(
                self,
                prefix: str = '',
                sharded_offsets: Tuple[Tuple[int, int, int]] = (),
                metadata: Optional[dict] = None,
        ) -> ShardedStateDict:
            sharded_state_dict = tuners_sharded_state_dict(self, prefix, sharded_offsets, metadata)
            if prefix in {'output_layer.', 'language_model.output_layer.'}:
                for k in list(sharded_state_dict.keys()):
                    if '_extra_state' in k:
                        # Old GPT checkpoints only stored the output layer weight key. So we remove the
                        # _extra_state key but check that it doesn't contain any data anyway
                        output_extra_state = sharded_state_dict.pop(k, None)
                        assert not (output_extra_state and output_extra_state.data
                                    ), f'Expected output layer extra state to be empty, got: {output_extra_state}'
                # fix error
                if f'{prefix}modules_to_save.default.weight' in sharded_state_dict:
                    sharded_state_dict[f'{prefix}weight'] = sharded_state_dict[
                        f'{prefix}modules_to_save.default.weight']
            return sharded_state_dict

    peft_module.ModulesToSaveWrapper = ModulesToSaveWrapper
    peft_module.OriginModulesToSaveWrapper = OriginModulesToSaveWrapper


def _patch_TransformerLayer():
    _origin_forward = TransformerLayer.forward

    def forward(self, *_args, **kwargs):
        """
        Perform a forward pass through the transformer layer.

        This method calls the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.
        """
        if not mcore_013:
            return _origin_forward(self, *_args, **kwargs)
        hidden_states, context = self._forward_attention(*_args, **kwargs)
        mlp_padding_free = self.config.mlp_padding_free and 'attention_mask' in kwargs
        mask = None
        if mlp_padding_free and hidden_states.shape[1] > 1:
            mask = ((~kwargs['attention_mask']).sum(dim=(1, 2)) > 0).t()
            hidden_states = hidden_states[mask][:, None]
        output = self._forward_mlp(hidden_states, kwargs.get('inference_context', None))
        if mask is not None:
            new_output = hidden_states.new_zeros((*mask.shape, output.shape[-1]))
            new_output[mask] = output.squeeze(1)
            output = new_output
        return output, context

    TransformerLayer.forward = forward


def _patch_TELinear():

    def __repr__(self):
        if is_torch_npu_available():
            # MindSpeed 0.15.x changes some TE debug fields to
            # input_size/output_size. Keep this compatibility on the NPU path
            # only so GPU and older versions retain their original field
            # semantics.
            in_features = getattr(self, 'in_features', getattr(self, 'input_size', None))
            out_features = getattr(self, 'out_features', getattr(self, 'output_size', None))
            use_bias = getattr(self, 'use_bias', getattr(self, 'bias', None) is not None)
            tp_size = getattr(self, 'tp_size', None)
            if tp_size is None:
                parallel_mode = getattr(self, 'parallel_mode', None)
                tp_size = 1 if parallel_mode == 'duplicated' else 'unknown'
        else:
            in_features = self.in_features
            out_features = self.out_features
            use_bias = self.use_bias
            tp_size = self.tp_size
        return (f'{type(self).__name__}(in_features={in_features}, '
                f'out_features={out_features}, bias={use_bias}, TP={tp_size})')

    TELinear.__repr__ = __repr__


def _patch_mrope():
    # Code borrowed from huggingface/transformers
    def apply_interleaved_mrope(freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    # Code borrowed from NVIDIA/Megatron-LM
    def forward(self, position_ids, mrope_section: List[int], mrope_interleaved: bool = False) -> torch.Tensor:
        seq = position_ids.to(device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        # shape (3, bs, dim, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].expand(3, seq.shape[1], -1, 1)
        # shape (3, bs, 1, seq_length)
        seq_expanded = seq[:, :, None, :].float()
        # shape (3, bs, seq_length, dim)
        freqs = (inv_freq_expanded @ seq_expanded).transpose(2, 3)
        if mrope_interleaved:
            freqs = apply_interleaved_mrope(freqs, mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            # first part even vector components, second part odd vector components,
            #  2 * dim in dimension size
            if self.rotary_interleaved:
                emb = torch.cat([m[i % 3] for i, m in enumerate(freqs.split(mrope_section, dim=-1))], dim=-1)
                emb = emb.repeat_interleave(2, dim=-1)
            else:
                emb = torch.cat((freqs, freqs), dim=-1)  # shape (3, bs, seq_length, 2 * dim)
                # generate freqs with mrope_section
                # shape (bs, seq_length, 2 * dim)
                mrope_section = mrope_section * 2
                emb = torch.cat([m[i % 3] for i, m in enumerate(emb.split(mrope_section, dim=-1))], dim=-1)

        # shape (seq_length, bs, 1, 2 * dim)
        emb = emb[..., None, :].transpose(0, 1).contiguous()
        return emb

    MultimodalRotaryEmbedding.forward = forward
    _origin_apply_rotary_pos_emb_thd = rope_utils._apply_rotary_pos_emb_thd

    def _apply_rotary_pos_emb_thd(
        t: torch.Tensor,
        cu_seqlens: torch.Tensor,
        freqs: torch.Tensor,
        rotary_interleaved: bool = False,
        multi_latent_attention: bool = False,
        mscale: float = 1.0,
        cp_group: torch.distributed.ProcessGroup = None,
        **kwargs,
    ) -> torch.Tensor:
        """A baseline implementation of applying RoPE for `thd` format.

        Args:
            t (Tensor): Input tensor T is of shape [t, h, d]
            cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
            with shape [b + 1] and dtype torch.int32.
            freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]
            cp_group (torch.distributed.ProcessGroup): The context parallel group

        Returns:
            Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
        """
        if cp_group is not None:
            cp_size = cp_group.size()
        else:
            cp_size = mpu.get_context_parallel_world_size()
        cu_seqlens_for_batched = cu_seqlens // cp_size
        use_batched_rope = (freqs.dim() >= 1 and freqs.shape[0] == cu_seqlens_for_batched[-1]).item()
        if not use_batched_rope:
            logger.warning_once('Using non-batched RoPE, which may affect performance.')
            if mcore_013:
                kwargs['cp_group'] = cp_group
            return _origin_apply_rotary_pos_emb_thd(
                t,
                cu_seqlens,
                freqs,
                rotary_interleaved=rotary_interleaved,
                multi_latent_attention=multi_latent_attention,
                mscale=mscale,
                **kwargs,
            )

        return rope_utils._apply_rotary_pos_emb_bshd(
            t.unsqueeze(1),
            freqs,
            rotary_interleaved=rotary_interleaved,
            multi_latent_attention=multi_latent_attention,
            mscale=mscale,
            **kwargs,
        ).squeeze(1)

    rope_utils._apply_rotary_pos_emb_thd = _apply_rotary_pos_emb_thd


def _patch_dsa():

    from megatron.core.models.gpt import experimental_attention_variant_module_specs
    from megatron.core.transformer.experimental_attention_variant.dsa import rotate_activation
    DSAIndexer = experimental_attention_variant_module_specs.DSAIndexer

    class NewDSAIndexer(DSAIndexer):

        def forward_before_topk(
            self,
            x: torch.Tensor,
            qr: torch.Tensor,
            packed_seq_params: Optional[PackedSeqParams] = None,
        ):
            """All computations before topk."""
            # =========================================
            # Gather inputs if sp is enabled
            # =========================================
            packed_seq_params, rotary_pos_emb = packed_seq_params  # patch
            assert packed_seq_params is None, 'Packed sequence is not supported for DSAttention'

            if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
                x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
                qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)

            # =========================================
            # Get sequence length and batch size
            # =========================================
            seqlen, bsz, _ = x.size()

            # =========================================
            # q linear and apply rope to q
            # =========================================
            # [seqlen, batch, q_lora_rank] -> [seqlen, batch, index_n_heads * index_head_dim]
            q, _ = self.linear_wq_b(qr)
            # [seqlen, batch, index_n_heads * index_head_dim]
            #   -> [seqlen, batch, index_n_heads, index_head_dim]
            q = q.reshape(seqlen, bsz, self.index_n_heads, self.index_head_dim)
            q = self._apply_rope(q, rotary_pos_emb)  # mscale will be passed in by patch

            # =========================================
            # k linear and apply rope to k
            # =========================================
            # [seqlen, batch, hidden_size] -> [seqlen, batch, index_head_dim]
            k, _ = self.linear_wk(x)
            k = self.k_norm(k)
            # [seqlen, batch, index_head_dim] -> [seqlen, batch, 1, index_head_dim]
            k = k.reshape(seqlen, bsz, 1, self.index_head_dim)
            k = self._apply_rope(k, rotary_pos_emb)
            # [seqlen, batch, 1, index_head_dim] -> [seqlen, batch, index_head_dim]
            k = k.reshape(seqlen, bsz, self.index_head_dim)

            # =========================================
            # Rotate activation
            # =========================================
            q = rotate_activation(q)
            k = rotate_activation(k)

            # =========================================
            # Prepare weights for index scores
            # =========================================
            # [seqlen, batch, hidden_size] -> [seqlen, batch, index_n_heads]
            weights, _ = self.linear_weights_proj(x)
            weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale

            return q, k, weights

        def _apply_rope(self, x: torch.Tensor, rotary_pos_emb: torch.Tensor):
            """Apply RoPE to the input tensor."""
            # x_nope [seqlen, batch, *, index_head_dim - qk_pos_emb_head_dim]
            # x_pe   [seqlen, batch, *, qk_pos_emb_head_dim]
            x_pe, x_nope = torch.split(
                x, [self.index_head_dim - self.qk_pos_emb_head_dim, self.qk_pos_emb_head_dim], dim=-1)
            origin_multi_latent_attention = self.config.multi_latent_attention
            try:
                self.config.multi_latent_attention = self.config.dsa_indexer_rotary_interleaved
                x_pe = apply_rotary_pos_emb(
                    x_pe,
                    rotary_pos_emb,
                    config=self.config,
                    cu_seqlens=None,
                    cp_group=self.pg_collection.cp,
                )
            finally:
                self.config.multi_latent_attention = origin_multi_latent_attention
            # [seqlen, batch, *, index_head_dim]
            x = torch.cat([x_pe, x_nope], dim=-1)
            return x

        def forward_with_scores(
            self,
            x: torch.Tensor,
            qr: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            packed_seq_params: Optional[PackedSeqParams] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass for DSA Indexer that returns both index scores and top-k indices.

            This is used when KL loss is enabled to compare indexer scores with true attention scores.

            Args:
                x: hidden states [seqlen, batch, hidden_size].
                qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
                mask: Attention mask [batch, seqlen, seqlen].
                packed_seq_params: Packed sequence parameters for variable length sequences.

            Returns:
                index_scores: Index scores [batch, seqlen, seqlen].
                topk_indices: Top-k indices [batch, seqlen, index_topk].
            """
            try:
                from megatron.core.transformer.experimental_attention_variant.dsa import fused_qk_topk_naive
            except ImportError:
                raise ImportError('fused_qk_topk_naive is not available. Please install megatron-core from source. '
                                  '`pip install git+https://github.com/NVIDIA/Megatron-LM.git`')
            # [seqlen, batch, index_n_heads * index_head_dim]
            # [seqlen, batch, index_head_dim]
            # [seqlen, batch, index_n_heads]
            q, k, weights = self.forward_before_topk(x, qr, packed_seq_params)

            # [batch, seqlen, seqlen], [batch, seqlen, index_topk]
            index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, self.index_topk, mask)

            return index_scores, topk_indices

        def forward(self,
                    x: torch.Tensor,
                    qr: torch.Tensor,
                    mask: Optional[torch.Tensor] = None,
                    packed_seq_params: Optional[PackedSeqParams] = None):
            """
            Forward pass for DSA Indexer.

            Args:
                x: hidden states [seqlen, batch, hidden_size].
                qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
                mask: Attention mask [batch, seqlen, seqlen].
                packed_seq_params: Packed sequence parameters for variable length sequences.

            Returns:
                topk_indices: Top-k indices for sparse attention [batch, seqlen, index_topk].
            """
            _, topk_indices = self.forward_with_scores(x, qr, mask, packed_seq_params)
            return topk_indices

    experimental_attention_variant_module_specs.DSAIndexer = NewDSAIndexer


def apply_patch():
    _patch_flash_attn()
    _patch_transformer_engine()
    # patch peft
    try:
        _patch_peft_BaseTuner()
        _patch_peft_ModulesToSaveWrapper()
    except Exception:
        logger.warning('Failed to patch peft.')
    # patch module
    _patch_mla_attention()
    _patch_TEGroupedLinear()
    _patch_mtp()
    _patch_TransformerLayer()
    _patch_TELinear()
    _patch_mrope()
    from mcore_bridge import tuners  # apply patch
    try:
        _patch_dsa()
    except ImportError:
        pass
