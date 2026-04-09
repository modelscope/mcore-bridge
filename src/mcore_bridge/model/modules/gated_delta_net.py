# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.nn.functional as F
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from typing import List, Optional

try:
    from fla.modules.convolution import causal_conv1d
    from fla.modules.l2norm import l2norm
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError:
    causal_conv1d = None
    l2norm = None
    chunk_gated_delta_rule = None

try:
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet as _GatedDeltaNet
    from megatron.core.ssm.gated_delta_net import torch_chunk_gated_delta_rule
except ImportError:
    _GatedDeltaNet = object


# Code borrowed from NVIDIA/Megatron-LM
def _unpack_sequence(x, cu_seqlens, dim=1):
    unpacked_x = []
    num_seqs = cu_seqlens.shape[0] - 1
    for i in range(num_seqs):
        idx_start = cu_seqlens[i].item()
        idx_end = cu_seqlens[i + 1].item()
        chunked_index = [slice(None)] * dim + [slice(idx_start, idx_end)]
        unpacked_x.append(x[tuple(chunked_index)])
    return unpacked_x


# Code borrowed from NVIDIA/Megatron-LM
# Avoid the warning caused by `param[slices]`
def get_parameter_local_cp(
    param: torch.Tensor,
    dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
) -> torch.Tensor:
    """Get the local parameter for the current context parallel rank.

    Args:
        param (torch.Tensor): The entire parameter to get the local parameter for.
        dim (int): The dimension to split the parameter along. Usually the dimension of head.
        cp_group (torch.distributed.ProcessGroup): The context parallel group.
        split_sections (Optional[List[int]]): If not None,
            first split the parameter along the dimension dim into sections,
            then get the local hidden parallel weights separately,
            finally concatenate the local hidden parallel weights along the dimension dim.

    Returns:
        torch.Tensor: The local parameter for the current context parallel rank.
    """

    cp_size = cp_group.size()
    cp_rank = cp_group.rank()

    # No need to split if CP size is 1.
    if cp_size == 1:
        return param

    # Split first if needed.
    if split_sections is not None:
        inputs = torch.split(param, split_sections, dim=dim)
        outputs = []
        for p in inputs:
            p = get_parameter_local_cp(p, dim, cp_group)
            outputs.append(p)
        return torch.cat(outputs, dim=dim)

    # Slice the parameter.
    slices = [slice(None)] * param.dim()
    dim_size = param.size(dim=dim)
    slices[dim] = slice(cp_rank * dim_size // cp_size, (cp_rank + 1) * dim_size // cp_size)
    param = param[tuple(slices)]
    return param


class GatedDeltaNet(_GatedDeltaNet):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ):
        """
        Perform a forward pass through the GDN module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) GDN output and bias.

        """
        # TODO: Deal with attention_mask
        from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push

        inference_context = deprecate_inference_params(inference_context, inference_params)

        seq_len, batch, _ = hidden_states.shape
        cp_size = self.config.context_parallel_size
        seq_len = seq_len * self.sp_size * cp_size

        if inference_context is not None:
            assert (
                inference_context.is_static_batching()), 'GDN does not currently support dynamic inference batching.'
            assert not self.config.sequence_parallel
            # TODO: support inference
            raise NotImplementedError('GDN does not support inference for now.')

        cu_seqlens = None if packed_seq_params is None else packed_seq_params.cu_seqlens_q
        # Input projection
        nvtx_range_push(suffix='in_proj')
        qkvzba, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix='in_proj')

        if cp_size > 1:
            from megatron.core.ssm.gated_delta_net import tensor_a2a_cp2hp, tensor_a2a_hp2cp
            if cu_seqlens is not None:
                unpacked_qkvzba = _unpack_sequence(qkvzba, cu_seqlens // self.cp_size, dim=0)
                outputs = []
                for qkvzba_i in unpacked_qkvzba:
                    qkvzba_i = tensor_a2a_cp2hp(
                        qkvzba_i,
                        seq_dim=0,
                        head_dim=-1,
                        cp_group=self.pg_collection.cp,
                    )
                    outputs.append(qkvzba_i)
                qkvzba = torch.cat(outputs, dim=0)
            else:
                # CP All to All: CP to HP
                qkvzba = tensor_a2a_cp2hp(
                    qkvzba,
                    seq_dim=0,
                    head_dim=-1,
                    cp_group=self.pg_collection.cp,
                )

        # Transpose: s b x --> b s x
        # From sbhd to bshd format
        qkvzba = qkvzba.transpose(0, 1)

        # Split, reorder, and reshape the tensor into q, k, v, gate, beta, alpha
        num_key_heads_per_device = self.num_key_heads // self.tp_size // cp_size
        qkvzba = qkvzba.view(qkvzba.shape[:-1]
                             + (num_key_heads_per_device, qkvzba.shape[-1] // num_key_heads_per_device))
        qkv, gate, beta, alpha = torch.split(
            qkvzba,
            [
                (self.qk_dim * 2 + self.v_dim) // self.num_key_heads,
                self.v_dim // self.num_key_heads,
                self.num_value_heads // self.num_key_heads,
                self.num_value_heads // self.num_key_heads,
            ],
            dim=-1,
        )
        gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)
        beta = beta.reshape(batch, seq_len, -1)
        alpha = alpha.reshape(batch, seq_len, -1)
        qkv = qkv.reshape(batch, seq_len, -1)

        # Convolution on qkv
        nvtx_range_push(suffix='conv1d')
        if cp_size > 1:
            conv1d_weight = get_parameter_local_cp(
                self.conv1d.weight,
                dim=0,
                cp_group=self.pg_collection.cp,
            )
            conv1d_bias = (
                get_parameter_local_cp(
                    self.conv1d.bias,
                    dim=0,
                    cp_group=self.pg_collection.cp,
                ) if self.conv_bias else None)
        else:
            conv1d_weight = self.conv1d.weight
            conv1d_bias = self.conv1d.bias

        if (causal_conv1d is None) or self.config.deterministic_mode:
            assert cu_seqlens is None, 'Packed sequences are not supported when fla is not available.'
            qkv = qkv.transpose(1, 2).contiguous()  # b, s, d -> b, d, s
            conv_out = F.conv1d(
                input=qkv,
                weight=conv1d_weight,
                bias=conv1d_bias,
                stride=self.conv1d.stride,
                padding=self.conv1d.padding,
                dilation=self.conv1d.dilation,
            )
            qkv = self.act_fn(conv_out[..., :seq_len])
            qkv = qkv.transpose(1, 2)  # b, d, s -> b, s, d
        else:
            assert self.activation in ['silu', 'swish']
            qkv = causal_conv1d(
                x=qkv,
                weight=conv1d_weight.squeeze(1),  # d, 1, w -> d, w
                bias=conv1d_bias,
                activation=self.activation,
                cu_seqlens=cu_seqlens,
            )[0]
        nvtx_range_pop(suffix='conv1d')
        # Split qkv into query, key, and value
        qkv = qkv.view(qkv.shape[:-1] + (num_key_heads_per_device, qkv.shape[-1] // num_key_heads_per_device))
        query, key, value = torch.split(
            qkv,
            [self.qk_dim // self.num_key_heads, self.qk_dim // self.num_key_heads, self.v_dim // self.num_key_heads],
            dim=-1,
        )
        query = query.reshape(batch, seq_len, -1, self.key_head_dim)
        key = key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)
        # Apply L2 norm to query and key
        if self.use_qk_l2norm:
            query = l2norm(query.contiguous())
            key = l2norm(key.contiguous())
        if self.num_value_heads // self.num_key_heads > 1:
            query = query.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)
            key = key.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)

        # Make contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        gate = gate.contiguous()
        beta = beta.contiguous()
        alpha = alpha.contiguous()

        # Calculate g and beta
        nvtx_range_push(suffix='g_and_beta')
        if cp_size > 1:
            A_log_local_cp = get_parameter_local_cp(self.A_log, dim=0, cp_group=self.pg_collection.cp)
            dt_bias_local_cp = get_parameter_local_cp(self.dt_bias, dim=0, cp_group=self.pg_collection.cp)
        else:
            A_log_local_cp, dt_bias_local_cp = self.A_log, self.dt_bias
        g = -A_log_local_cp.exp() * F.softplus(alpha.float() + dt_bias_local_cp)  # In fp32
        beta = beta.sigmoid()
        nvtx_range_pop(suffix='g_and_beta')

        nvtx_range_push(suffix='gated_delta_rule')
        if self.config.deterministic_mode:
            assert cu_seqlens is None, ('cu_seqlens is not supported for torch_chunk_gated_delta_rule for now.')
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
            )
        else:
            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                cu_seqlens=cu_seqlens,
            )
        nvtx_range_pop(suffix='gated_delta_rule')

        # RMSNorm
        nvtx_range_push(suffix='gated_norm')
        norm_out = self._apply_gated_norm(core_attn_out, gate)
        nvtx_range_pop(suffix='gated_norm')

        # Transpose: b s x --> s b x
        # From bshd back to sbhd format
        norm_out = norm_out.reshape(batch, seq_len, -1)
        norm_out = norm_out.transpose(0, 1).contiguous()
        if cp_size > 1:
            if cu_seqlens is not None:
                unpacked_norm_out = _unpack_sequence(norm_out, cu_seqlens, dim=0)
                outputs = []
                for norm_out_i in unpacked_norm_out:
                    norm_out_i = tensor_a2a_hp2cp(norm_out_i, seq_dim=0, head_dim=-1, cp_group=self.pg_collection.cp)
                    outputs.append(norm_out_i)
                norm_out = torch.cat(outputs, dim=0)
            else:
                norm_out = tensor_a2a_hp2cp(norm_out, seq_dim=0, head_dim=-1, cp_group=self.pg_collection.cp)

        # Output projection
        nvtx_range_push(suffix='out_proj')
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix='out_proj')

        return out, out_bias
