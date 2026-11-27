# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.nn.functional as F
from copy import deepcopy
from functools import partial
from megatron.core import mpu
from megatron.core.extensions.transformer_engine import (TEColumnParallelGroupedLinear, TELinear,
                                                         TERowParallelGroupedLinear)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.transformer.utils import (ensure_metadata_has_dp_cp_group, make_sharded_tensors_for_checkpoint,
                                             sharded_state_dict_default)
from torch import nn
from torch.utils.checkpoint import checkpoint

from mcore_bridge.tuners.lora import LoraParallelLinear

ATTENTION_PROJECTIONS = (
    'q_proj.linear',
    'k_proj.linear',
    'v_proj.linear',
    'relative_position_bias.linear',
    'o_proj.gate_proj',
    'o_proj.linear',
)


def require_inkling2_loop():
    try:
        from megatron.core.transformer.module import mark_keep_in_fp32
        from transformers.models.inkling2_loop import modeling_inkling2_loop
    except ImportError as error:
        raise ImportError(
            'Inkling-2-Mini requires native Transformers support and Megatron mark_keep_in_fp32.') from error
    return modeling_inkling2_loop, mark_keep_in_fp32


def _patch_bounded_fused_adam():
    from transformer_engine.pytorch.optimizers import fused_adam

    original = fused_adam.multi_tensor_applier
    if getattr(original, '_bounded_inkling_adam', False):
        return

    def bounded_apply(operation, noop_flag, tensor_lists, *args):
        count = len(tensor_lists[0])
        chunk_size = 1024
        if count <= chunk_size:
            return original(operation, noop_flag, tensor_lists, *args)
        if any(len(tensors) != count for tensors in tensor_lists):
            raise ValueError('FusedAdam tensor lists must be aligned before chunking.')
        # Keep the Adam group step unchanged while chunking lists to respect TE's handle limit.
        for start in range(0, count, chunk_size):
            original(operation, noop_flag, [tensors[start:start + chunk_size] for tensors in tensor_lists], *args)

    bounded_apply._bounded_inkling_adam = True
    fused_adam.multi_tensor_applier = bounded_apply


def _apply_with_fp32_buffers(module, apply, fn, recurse):
    buffers = [(child, name, buffer) for child in module.modules()
               for name, buffer in child.named_buffers(recurse=False) if getattr(buffer, 'keep_in_fp32', False)]
    result = apply(fn, recurse=recurse)
    _, mark_keep_in_fp32 = require_inkling2_loop()
    for child, name, buffer in buffers:
        device = child._buffers[name].device
        child._buffers[name] = mark_keep_in_fp32(buffer.to(device=device, dtype=torch.float32))
    return result


def _sharded_children(module, prefix, sharded_offsets, metadata):
    metadata = ensure_metadata_has_dp_cp_group(metadata)
    own_state = {name: parameter for name, parameter in module.named_parameters(recurse=False)}
    own_state.update({
        name: buffer
        for name, buffer in module.named_buffers(recurse=False) if name not in module._non_persistent_buffers_set
    })
    result = make_sharded_tensors_for_checkpoint(
        own_state,
        prefix, {},
        sharded_offsets,
        tp_group=mpu.get_tensor_model_parallel_group(),
        dp_cp_group=metadata['dp_cp_group'])
    for name, child in module.named_children():
        child_prefix = f'{prefix}{name}.'
        if hasattr(child, 'sharded_state_dict'):
            result.update(sharded_state_dict_default(child, child_prefix, sharded_offsets, metadata))
        else:
            result.update(_sharded_children(child, child_prefix, sharded_offsets, metadata))
    return result


class Inkling2LoopLoraLinear(LoraParallelLinear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _, mark_keep_in_fp32 = require_inkling2_loop()
        for factor in (self.lora_A, self.lora_B):
            for parameter in factor.parameters():
                mark_keep_in_fp32(parameter)
        self._unmerged_weight = None

    def get_delta_weights(self, adapter):
        device_type = self.get_base_layer().weight.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            delta = self.lora_B[adapter].weight.float() @ self.lora_A[adapter].weight.float()
            if self.fan_in_fan_out:
                delta = delta.T
            return [delta * self.scaling[adapter]]

    def merge(self, safe_merge=False, adapter_names=None):
        weight = self.get_base_layer().weight
        original_storage = weight.data
        if not self.merged:
            self._unmerged_weight = weight.detach().clone()
        try:
            super().merge(safe_merge=safe_merge, adapter_names=adapter_names)
        except Exception:
            self.unmerge()
            self._unmerged_weight = None
            raise
        finally:
            if weight.data_ptr() != original_storage.data_ptr():
                with torch.no_grad():
                    original_storage.copy_(weight.data)
                weight.data = original_storage
        if not self.merged:
            self._unmerged_weight = None

    def unmerge(self):
        if not self.merged:
            return
        # Restore the pre-merge weight exactly; BF16 addition and subtraction are not reversible.
        with torch.no_grad():
            weight = self.get_base_layer().weight
            weight.copy_(self._unmerged_weight.to(weight.device))
        self.merged_adapters.clear()
        self._unmerged_weight = None


class Inkling2LoopLinear(TELinear):
    lora_cls = Inkling2LoopLoraLinear


class Inkling2TensorLinear(MegatronModule):

    def __init__(self, input_size, output_size, config):
        super().__init__(config)
        self.linear = Inkling2LoopLinear(
            input_size,
            output_size,
            config=config,
            parallel_mode='duplicated',
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False)

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, hidden_states):
        output, bias = self.linear(hidden_states)
        if bias is not None:
            output = output + bias
        return output


class Inkling2SoftcappedSwiGLU(nn.Module):

    def __init__(self, config):
        super().__init__()
        text_config = config.hf_config.text_config
        self.gate_softcap = text_config.mlp_gate_softcap
        self.up_softcap = text_config.mlp_up_softcap

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        gate, up = hidden_states.float().chunk(2, dim=-1)
        gate_factor = torch.tanh(gate / self.gate_softcap) * self.gate_softcap
        up_factor = torch.tanh(up / self.up_softcap) * self.up_softcap
        return (gate_factor * torch.sigmoid(gate) * up_factor).to(dtype)


class Inkling2Experts(TEGroupedMLP):

    def forward(self, hidden_states, tokens_per_expert, permuted_probs, output_buffer=None, grad_input_buffer=None):
        if output_buffer is not None or grad_input_buffer is not None:
            raise ValueError('Inkling-2-Mini does not support fused expert output buffers.')
        counts = tokens_per_expert.tolist() if isinstance(tokens_per_expert, torch.Tensor) else tokens_per_expert
        hidden_states, bias = self.linear_fc1(hidden_states, counts)
        if bias is not None:
            raise ValueError('Inkling-2-Mini experts must be bias-free.')
        if self.training and self.activation_recompute and torch.is_grad_enabled():
            hidden_states = checkpoint(self.activation_func, hidden_states, use_reentrant=False)
        else:
            hidden_states = self.activation_func(hidden_states)
        output, bias = self.linear_fc2(hidden_states, counts)
        if bias is not None:
            raise ValueError('Inkling-2-Mini experts must be bias-free.')
        # Apply routing weights after the down projection and accumulate outputs in FP32.
        return output.float() * permuted_probs.reshape(-1, 1).float(), None


class Inkling2Router(TopKRouter):

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        _, mark_keep_in_fp32 = require_inkling2_loop()
        self.weight.data = self.weight.data.float()
        mark_keep_in_fp32(self.weight)
        self.register_buffer('e_score_correction_bias', torch.zeros(config.num_moe_experts, dtype=torch.float32))
        mark_keep_in_fp32(self.e_score_correction_bias)

    def _apply(self, fn, recurse=True):
        return _apply_with_fp32_buffers(self, super()._apply, fn, recurse)

    def routing(self, logits, padding_mask=None):
        scores = torch.sigmoid(logits.float().reshape(-1, self.config.num_moe_experts))
        indices = torch.topk(
            scores + self.e_score_correction_bias.to(scores.device), self.config.moe_router_topk, dim=-1,
            sorted=False).indices
        selected_scores = scores.gather(1, indices)
        selected_scores = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)
        selected_scores = selected_scores * self.config.moe_router_topk_scaling_factor
        probabilities = torch.zeros_like(scores).scatter(1, indices, selected_scores)
        routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter(1, indices, True)
        if padding_mask is not None:
            valid = ~padding_mask.reshape(-1, 1)
            probabilities = probabilities * valid
            routing_map = routing_map & valid
        return probabilities, routing_map

    def gating(self, input):
        return F.linear(input.float(), self.weight.float())

    def forward(self, input, padding_mask=None, **kwargs):
        return self.routing(self.gating(input), padding_mask)


class Inkling2TokenDispatcher(MoEAlltoAllTokenDispatcher):

    def combine_postprocess(self, tokens):
        routing_map = self.routing_map
        token_ids = self.reversed_local_input_permutation_mapping
        slots = (routing_map.cumsum(dim=-1).T.contiguous()[routing_map.T] - 1).long()
        num_tokens, hidden = self.hidden_shape_before_permute
        topk = self.config.moe_router_topk
        flat_ids = token_ids * topk + slots
        ordered = tokens.new_zeros((num_tokens * topk, hidden)).index_copy(0, flat_ids, tokens)
        ordered = ordered.view(num_tokens, topk, hidden)
        output = tokens.new_zeros((num_tokens, hidden))
        # Match the official eager expert accumulation order in FP32.
        for slot in range(topk):
            output = output + ordered[:, slot]
        return output.view(self.hidden_shape)


class Inkling2TensorMoE(MegatronModule):

    def __init__(self, config, layer_number, pg_collection):
        super().__init__(config)
        if pg_collection is None:
            from megatron.core.transformer.moe.moe_layer import get_default_pg_collection

            pg_collection = get_default_pg_collection()
        expert_submodules = GroupedMLPSubmodules(
            linear_fc1=TEColumnParallelGroupedLinear,
            linear_fc2=TERowParallelGroupedLinear,
            activation_func=Inkling2SoftcappedSwiGLU)
        experts = partial(Inkling2Experts, submodules=expert_submodules)
        self.native = MoELayer(
            config=config,
            submodules=MoESubmodules(experts=experts, router=Inkling2Router),
            layer_number=layer_number,
            pg_collection=pg_collection)
        self.native.token_dispatcher = Inkling2TokenDispatcher(
            self.native.num_local_experts, self.native.local_expert_indices, config=config, pg_collection=pg_collection)

    def forward(self, hidden_states):
        output, bias = self.native(hidden_states.transpose(0, 1).contiguous())
        if bias is not None:
            output = output + bias
        return output.to(hidden_states.dtype).transpose(0, 1).contiguous()


def _materialize_hf_parameters(module, config):
    _, mark_keep_in_fp32 = require_inkling2_loop()
    device = 'cpu' if config.use_cpu_initialization else torch.cuda.current_device()
    fp32_names = ('post_attention_convolution', 'post_mlp_convolution', 'sconv.conv1d', 'message_gate_proj',
                  'forget_gate_proj')
    for path, child in module.named_modules():
        for name, parameter in list(child.named_parameters(recurse=False)):
            if not parameter.is_meta:
                continue
            full_name = f'{path}.{name}'
            keep_fp32 = name == 'query' or any(part in full_name for part in fp32_names)
            dtype = torch.float32 if keep_fp32 else config.params_dtype
            value = torch.empty(parameter.shape, dtype=dtype, device=device)
            new_parameter = nn.Parameter(value, requires_grad=parameter.requires_grad)
            if keep_fp32:
                mark_keep_in_fp32(new_parameter)
            child.register_parameter(name, new_parameter)
            with torch.no_grad():
                if name == 'query' or name == 'static_weight':
                    new_parameter.zero_()
                elif name == 'scale' or isinstance(child, nn.LayerNorm) or child.__class__.__name__.endswith('RMSNorm'):
                    new_parameter.fill_(1)
                else:
                    config.init_method(new_parameter)


def _packed_document_layout(packed_seq_params, batch, length, device):
    valid = torch.ones((batch, length), dtype=torch.bool, device=device)
    if packed_seq_params is None:
        return torch.zeros((batch, length), dtype=torch.int32, device=device), valid
    cu = packed_seq_params.cu_seqlens_q
    physical = packed_seq_params.cu_seqlens_q_padded
    physical = cu if physical is None else physical
    if batch != 1 or packed_seq_params.qkv_format != 'thd':
        raise ValueError('Inkling packed inputs require THD layout with batch=1.')
    for offsets in (cu, physical):
        if (offsets.ndim != 1 or offsets.numel() < 2 or offsets.dtype not in (torch.int32, torch.int64)
                or offsets[0] != 0 or (offsets[1:] < offsets[:-1]).any()):
            raise ValueError('Packed sequence offsets must be nondecreasing integers starting at zero.')
    cu, physical = cu.to(device), physical.to(device)
    lengths = cu.diff()
    if physical.shape != cu.shape or physical[-1] > length or (lengths > physical.diff()).any():
        raise ValueError('Packed sequence lengths exceed their physical token layout.')
    tokens = torch.arange(length, device=device)
    documents = torch.bucketize(tokens, physical[1:], right=True)
    safe_documents = documents.clamp_max(lengths.numel() - 1)
    valid = (documents < lengths.numel()) & (tokens - physical[safe_documents] < lengths[safe_documents])
    return documents.to(torch.int32).unsqueeze(0), valid.unsqueeze(0)


class Inkling2LoopBlock(MegatronModule):

    def __init__(self,
                 config,
                 spec=None,
                 pre_process=True,
                 post_process=True,
                 pg_collection=None,
                 vp_stage=None,
                 **kwargs):
        super().__init__(config)
        hf, _ = require_inkling2_loop()
        _patch_bounded_fused_adam()
        self.hf_config = deepcopy(config.hf_config.text_config)
        self.hf_config._attn_implementation = 'eager'
        self.input_tensor = None
        self.pg_collection = pg_collection
        self.layers = nn.ModuleList()
        classes = {
            (True, False): hf.Inkling2LoopSlidingDecoderLayer,
            (False, False): hf.Inkling2LoopFullDecoderLayer,
            (True, True): hf.Inkling2LoopSlidingLoopDecoderLayer,
            (False, True): hf.Inkling2LoopFullLoopDecoderLayer,
        }
        start, end = config.inkling_loop_start, config.inkling_loop_end
        for index in range(config.num_layers):
            is_sliding = self.hf_config.layer_types[index] == 'hybrid_sliding'
            with torch.device('meta'):
                layer = classes[(is_sliding, start <= index < end)](self.hf_config, index)
            layer.layer_number = index + 1
            for path in ATTENTION_PROJECTIONS:
                parent_path, field = path.rsplit('.', 1)
                parent = layer.self_attn.get_submodule(parent_path)
                original = getattr(parent, field)
                setattr(parent, field, Inkling2TensorLinear(original.in_features, original.out_features, config))
            layer.mlp = Inkling2TensorMoE(config, index + 1, pg_collection)
            _materialize_hf_parameters(layer, config)
            self.layers.append(layer)
        with torch.device('meta'):
            self.embedding_norm = hf.Inkling2LoopRMSNorm(config.hidden_size, self.hf_config.rms_norm_eps)
            self.final_layernorm = hf.Inkling2LoopRMSNorm(config.hidden_size, self.hf_config.rms_norm_eps)
            self.final_input = hf.Inkling2LoopBlockAttentionResidual(self.hf_config)
            self.message_input = hf.Inkling2LoopMessageAttentionResidual(self.hf_config)
        for module in (self.embedding_norm, self.final_layernorm, self.final_input, self.message_input):
            _materialize_hf_parameters(module, config)

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor

    def _run_layer(self, layer, hidden_states, block_states, messages, common_kwargs):

        def run(hidden, blocks, cross_pass):
            return layer(hidden, blocks, **({} if cross_pass is None else {'messages': cross_pass}), **common_kwargs)

        if self.training and self.config.recompute_granularity == 'full' and torch.is_grad_enabled():
            return checkpoint(run, hidden_states, block_states, messages, use_reentrant=False)
        return run(hidden_states, block_states, messages)

    def forward(self,
                hidden_states,
                attention_mask=None,
                position_ids=None,
                packed_seq_params=None,
                inference_context=None,
                padding_mask=None,
                image_embeds=None,
                image_mask=None,
                **kwargs):
        if inference_context is not None:
            raise NotImplementedError(
                'Export Inkling-2-Mini to HF for inference; Megatron KV-cache inference is unsupported.')
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        input_dtype = hidden_states.dtype
        hidden_states = self.embedding_norm(hidden_states)
        if image_embeds is not None:
            mask = image_mask.to(device=hidden_states.device).unsqueeze(-1).expand_as(hidden_states)
            hidden_states = hidden_states.masked_scatter(mask, image_embeds.to(hidden_states.device,
                                                                               hidden_states.dtype))
        batch, length = hidden_states.shape[:2]
        if length > self.config.max_position_embeddings:
            raise ValueError('The sequence exceeds the Inkling-2-Mini checkpoint context length.')
        if position_ids is None:
            position_ids = torch.arange(length, device=hidden_states.device).expand(batch, -1)
        if position_ids.shape != (batch, length):
            raise ValueError('Inkling-2-Mini position_ids must match [batch, sequence].')
        seq_idx, valid = _packed_document_layout(packed_seq_params, batch, length, hidden_states.device)
        distance = torch.arange(length, device=hidden_states.device)
        distance = distance[:, None] - distance[None, :]
        masked = (distance < 0).expand(batch, -1, -1) | (seq_idx[:, :, None] != seq_idx[:, None, :])
        if attention_mask is not None:
            if isinstance(attention_mask, dict):
                attention_mask = attention_mask['full_attention']
            if attention_mask.ndim == 4:
                masked = masked | attention_mask[:, 0].bool()
            elif attention_mask.ndim == 2:
                padding_mask = ~attention_mask.bool()
            else:
                raise ValueError('Inkling-2-Mini expects a 2D padding mask or a 4D Megatron attention mask.')
        if padding_mask is not None:
            valid = valid & ~padding_mask
        masked = masked | ~valid[:, None, :]
        negative = torch.finfo(torch.float32).min
        full_mask = torch.zeros((batch, 1, length, length), dtype=torch.float32, device=hidden_states.device)
        full_mask = full_mask.masked_fill(masked[:, None], negative)
        local_mask = full_mask.masked_fill((distance >= self.hf_config.sliding_window_size)[None, None], negative)
        masks = {'full_attention': full_mask, 'sliding_attention': local_mask}
        conv_seq_idx = seq_idx if (seq_idx[:, 1:] != seq_idx[:, :-1]).any() else None
        common = {
            'relative_distance': distance,
            'position_ids': position_ids,
            'seq_idx': conv_seq_idx,
            'conv_mask': valid
        }
        block_states = hidden_states.new_zeros(batch, length, 0, self.config.hidden_size)

        def visit(index, hidden, blocks, messages=None):
            layer = self.layers[index]
            call_kwargs = {**common, 'attention_mask': masks[layer.attention_type]}
            output = self._run_layer(layer, hidden, blocks, messages, call_kwargs)
            if layer.is_block_boundary:
                blocks = torch.cat((blocks, hidden.unsqueeze(-2)), dim=-2)
            return output, blocks

        start, end = self.config.inkling_loop_start, self.config.inkling_loop_end
        for index in range(start):
            hidden_states, block_states = visit(index, hidden_states, block_states)
        initial_hidden, initial_blocks = hidden_states, block_states
        initial_block_count = block_states.shape[-2]
        for index in range(start, end):
            hidden_states, block_states = visit(index, hidden_states, block_states)
        for _ in range(1, self.config.inkling_loop_num_passes):
            previous_blocks = block_states[..., initial_block_count + 1:, :]
            previous_hidden = hidden_states
            hidden_states, block_states = initial_hidden, initial_blocks
            for index in range(start, end):
                messages = self.message_input(previous_blocks, previous_hidden, index - start)
                hidden_states, block_states = visit(index, hidden_states, block_states, messages)
        for index in range(end, self.config.num_layers):
            hidden_states, block_states = visit(index, hidden_states, block_states)
        hidden_states = self.final_input(block_states, hidden_states)[0]
        return self.final_layernorm(hidden_states).to(input_dtype).transpose(0, 1).contiguous()

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        return _sharded_children(self, prefix, sharded_offsets, metadata)
