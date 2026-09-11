# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEGroupedLinear
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import make_sharded_tensor_for_checkpoint
from transformers.utils import is_torch_npu_available
from typing import Optional, Tuple

from mcore_bridge.utils import get_current_device

_GMM_SPLIT_ITEM_FORWARD_OR_DINPUT = 2
_GMM_SPLIT_ITEM_DWEIGHT = 3
_GMM_GROUP_BY_M_AXIS = 0
_GMM_GROUP_BY_K_AXIS = 2
_GMM_GROUP_LIST_IS_EXPERT_SIZES = 1


def _has_moe_local_expert_grouping(base_layer) -> bool:
    config = getattr(base_layer, 'config', None)
    num_moe_experts = getattr(config, 'num_moe_experts', None)
    if num_moe_experts is None:
        return False
    ep_size = getattr(config, 'expert_model_parallel_size', 1)
    if not isinstance(ep_size, int) or ep_size <= 0:
        return False
    if num_moe_experts % ep_size != 0:
        return False
    return getattr(base_layer, 'num_gemms', None) == num_moe_experts // ep_size


def is_expert_layer(base_layer) -> bool:
    is_expert = getattr(base_layer, 'is_expert', None)
    if is_expert is not None:
        return bool(is_expert)
    if is_torch_npu_available() and isinstance(base_layer, TEGroupedLinear):
        if getattr(base_layer, 'explicit_expert_comm', False):
            return True
        has_moe_local_expert_grouping = _has_moe_local_expert_grouping(base_layer)
        if getattr(base_layer, 'expert_parallel', False):
            return has_moe_local_expert_grouping
        # MCore's TEGroupedLinear requires is_expert=True at construction but
        # does not retain that argument. With EP=ETP=1 there is no explicit
        # expert communication, so use the TEGroupedMLP invariant: one grouped
        # GEMM slot per local expert.
        return has_moe_local_expert_grouping
    return False


class NpuGroupedLoraLinear(nn.Module):
    """Generic grouped linear for low-rank NPU LoRA adapters."""

    def __init__(self,
                 num_gemms: int,
                 input_size: int,
                 output_size: int,
                 *,
                 config,
                 bias: bool,
                 is_expert: bool = False):
        super().__init__()
        self.num_gemms = num_gemms
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = bias
        self.is_expert = is_expert
        self.parallel_mode = None
        device = torch.device('cpu') if config.use_cpu_initialization else get_current_device()
        dtype = config.params_dtype
        for i in range(num_gemms):
            self.register_parameter(
                f'weight{i}',
                nn.Parameter(torch.empty(output_size, input_size, device=device, dtype=dtype)),
            )
            if bias:
                self.register_parameter(
                    f'bias{i}',
                    nn.Parameter(torch.empty(output_size, device=device, dtype=dtype)),
                )

    @property
    def weight(self):
        return self.weight0

    def _set_expert_replica_id(self, sharded_tensor):
        replica_id = sharded_tensor.replica_id
        assert len(replica_id) == 3, f'Expected replica_id to be in (PP, TP, DP) format, got: {replica_id}'
        if getattr(sharded_tensor, 'is_data_parallel_fully_shard', False):
            edp_replica_id = 0
        else:
            edp_replica_id = parallel_state.get_expert_data_parallel_rank()
        sharded_tensor.replica_id = (*replica_id[:2], edp_replica_id)
        return sharded_tensor

    def sharded_state_dict(
            self,
            prefix: str = '',
            sharded_offsets: Tuple[Tuple[int, int, int]] = (),
            metadata: Optional[dict] = None,
    ):
        if not self.is_expert:
            return make_sharded_tensors_for_checkpoint(
                self.state_dict(prefix='', keep_vars=True),
                prefix,
                sharded_offsets=sharded_offsets,
            )

        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        num_global_experts = parallel_state.get_expert_model_parallel_world_size() * self.num_gemms
        local_expert_indices_offset = parallel_state.get_expert_model_parallel_rank() * self.num_gemms
        ep_axis = len(sharded_offsets)
        sharded_state_dict = {}

        for i in range(self.num_gemms):
            global_expert_idx = local_expert_indices_offset + i
            if singleton_local_shards:
                key_prefix = f'{global_expert_idx}.{prefix}'
                new_sharded_offsets = sharded_offsets
            else:
                key_prefix = prefix
                new_sharded_offsets = (*sharded_offsets, (ep_axis, global_expert_idx, num_global_experts))

            for param_name in ('weight', 'bias'):
                local_name = f'{param_name}{i}'
                param = getattr(self, local_name, None)
                if param is None:
                    continue
                sharded_tensor = make_sharded_tensor_for_checkpoint(
                    param,
                    f'{key_prefix}{param_name}',
                    prepend_offsets=new_sharded_offsets,
                    tp_group=parallel_state.get_expert_tensor_parallel_group(),
                    dp_cp_group=parallel_state.get_expert_data_parallel_group(),
                )
                sharded_state_dict[f'{prefix}{local_name}'] = self._set_expert_replica_id(sharded_tensor)

        return sharded_state_dict

    def _fallback_forward(self, x, m_splits):
        if isinstance(m_splits, torch.Tensor):
            m_splits = m_splits.tolist()
        outputs = []
        offset = 0
        for i, split_size in enumerate(m_splits):
            split_size = int(split_size)
            x_i = x[offset:offset + split_size]
            offset += split_size
            weight = getattr(self, f'weight{i}')
            bias = getattr(self, f'bias{i}', None)
            outputs.append(F.linear(x_i, weight, bias))
        if offset != x.shape[0]:
            raise RuntimeError(f'Grouped LoRA token split mismatch: got {offset}, expected {x.shape[0]}')
        return torch.cat(outputs, dim=0) if outputs else x.new_empty((*x.shape[:-1], self.output_size)), None

    def _can_use_grouped_matmul(self, x):
        # PEFT's lora_bias=True adds a bias to LoRA-B and is not the mainstream
        # Swift path. Keep that uncommon adapter shape on the generic PyTorch
        # path until it is explicitly needed and verified.
        return is_torch_npu_available() and x.device.type == 'npu' and x.dim() == 2 and not self.use_bias

    def forward(self, x, m_splits):
        if not self._can_use_grouped_matmul(x):
            return self._fallback_forward(x, m_splits)
        weights = [getattr(self, f'weight{i}') for i in range(self.num_gemms)]
        return _NpuGroupedLoraLinearGMM.apply(x, m_splits, weights, *[weight.T for weight in weights]), None


class _NpuGroupedLoraLinearGMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_tensor, m_splits, weights, *weight_input_T):
        import torch_npu

        if isinstance(m_splits, torch.Tensor):
            group_list = m_splits
            if group_list.device.type == 'cpu':
                group_list = group_list.to(device=input_tensor.device, dtype=torch.int64)
            else:
                group_list = group_list.to(dtype=torch.int64)
        else:
            group_list = torch.tensor(m_splits, device=input_tensor.device, dtype=torch.int64)
        output = torch_npu.npu_grouped_matmul(
            [input_tensor],
            weight_input_T,
            bias=None,
            group_list=group_list,
            split_item=_GMM_SPLIT_ITEM_FORWARD_OR_DINPUT,
            group_type=_GMM_GROUP_BY_M_AXIS,
            group_list_type=_GMM_GROUP_LIST_IS_EXPERT_SIZES,
        )[0]
        ctx.group_list = group_list
        ctx.save_for_backward(input_tensor, *weights)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        import torch_npu

        input_tensor = ctx.saved_tensors[0]
        weights = ctx.saved_tensors[1:]
        group_list = ctx.group_list
        grad_input = torch_npu.npu_grouped_matmul(
            [grad_output],
            weights,
            bias=None,
            group_list=group_list,
            split_item=_GMM_SPLIT_ITEM_FORWARD_OR_DINPUT,
            group_type=_GMM_GROUP_BY_M_AXIS,
            group_list_type=_GMM_GROUP_LIST_IS_EXPERT_SIZES,
        )[0]
        grad_weight_T = torch_npu.npu_grouped_matmul(
            [input_tensor.T],
            [grad_output],
            bias=None,
            group_list=group_list,
            split_item=_GMM_SPLIT_ITEM_DWEIGHT,
            group_type=_GMM_GROUP_BY_K_AXIS,
            group_list_type=_GMM_GROUP_LIST_IS_EXPERT_SIZES,
        )[0]
        return grad_input, None, None, *grad_weight_T
