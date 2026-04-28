# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
from contextlib import nullcontext

import torch
from torch import nn
from transformers import PretrainedConfig

from megatron.core import tensor_parallel
from megatron.core.enums import Fp8Recipe
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt import gpt_model
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import VocabParallelEmbedding
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.transformer_layer import TransformerLayer, get_transformer_layer_offset
from megatron.core.utils import WrappedTensor, deprecate_inference_params, get_pg_rank, make_viewless_tensor

from mcore_bridge.bridge import MultimodalGPTBridge
from mcore_bridge.model.gpt_model import GPTModel
from mcore_bridge.model.mm_gpt_model import MultimodalGPTModel
from mcore_bridge.utils import get_logger

from ..constant import ModelType
from ..register import ModelLoader, ModelMeta, register_model
from .utils import HuggingFaceVit

logger = get_logger()


class Gemma4RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (hidden_states * self.weight.float()).to(input_dtype)


class Gemma4SelfAttention(SelfAttention):

    def __init__(self, config, submodules, *args, **kwargs):
        layer_number = kwargs.get('layer_number', 1)
        layer_idx = layer_number - 1
        layer_types = getattr(config, 'layer_types', None) or []
        layer_type = layer_types[layer_idx] if layer_idx < len(layer_types) else 'sliding_attention'
        is_sliding = layer_type == 'sliding_attention'

        local_config = copy.copy(config)
        local_config.kv_channels = config.kv_channels if is_sliding else (config.global_kv_channels or config.kv_channels)
        local_config.num_query_groups = (config.num_query_groups if is_sliding or config.num_global_query_groups is None
                                         else config.num_global_query_groups)
        super().__init__(local_config, submodules, *args, **kwargs)
        self.layer_type = layer_type
        self.post_self_attn_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.post_self_attn_layernorm.to(device=next(self.linear_proj.parameters()).device, dtype=config.params_dtype)

    def forward(self, hidden_states, *args, **kwargs):
        output, bias = super().forward(hidden_states, *args, **kwargs)
        if bias is not None:
            output = output + bias
            bias = None
        output = self.post_self_attn_layernorm(output)
        return output, bias


class Gemma4MLP(MLP):

    def __init__(self, config, submodules, *, layer_number: int, tp_group=None):
        local_config = copy.copy(config)
        first_kv_shared_layer_idx = config.num_layers - getattr(config, 'num_kv_shared_layers', 0)
        is_kv_shared_layer = layer_number - 1 >= first_kv_shared_layer_idx > 0
        if getattr(config, 'use_double_wide_mlp', False) and is_kv_shared_layer:
            local_config.ffn_hidden_size = config.ffn_hidden_size * 2
        super().__init__(local_config, submodules, tp_group=tp_group)
        self.post_mlp_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.post_mlp_layernorm.to(device=next(self.linear_fc2.parameters()).device, dtype=config.params_dtype)

    def forward(self, hidden_states, *args, **kwargs):
        output, bias = super().forward(hidden_states, *args, **kwargs)
        if bias is not None:
            output = output + bias
            bias = None
        output = self.post_mlp_layernorm(output)
        return output, bias


class Gemma4TransformerLayer(TransformerLayer):

    def __init__(self, config, submodules, *args, **kwargs):
        super().__init__(config, submodules, *args, **kwargs)
        self.mlp = Gemma4MLP(
            config,
            submodules.mlp.submodules,
            layer_number=self.layer_number,
            tp_group=self.pg_collection.tp,
        )
        self.hidden_size_per_layer_input = getattr(config, 'hidden_size_per_layer_input', 0) or 0
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = nn.Linear(config.hidden_size, self.hidden_size_per_layer_input, bias=False)
            self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, config.hidden_size, bias=False)
            self.post_per_layer_input_norm = Gemma4RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
            device = next(self.self_attention.parameters()).device
            self.per_layer_input_gate.to(device=device, dtype=config.params_dtype)
            self.per_layer_projection.to(device=device, dtype=config.params_dtype)
            self.post_per_layer_input_norm.to(device=device, dtype=config.params_dtype)

    def forward(self, *args, per_layer_input=None, **kwargs):
        hidden_states, context = super().forward(*args, **kwargs)
        if per_layer_input is not None and self.hidden_size_per_layer_input:
            if per_layer_input.dim() == hidden_states.dim() + 1:
                per_layer_input = per_layer_input[..., self.layer_number - 1, :]
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = torch.nn.functional.gelu(hidden_states, approximate='tanh')
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states
        return hidden_states, context


class Gemma4TransformerBlock(gpt_model.TransformerBlock):

    def _checkpointed_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        attention_bias: torch.Tensor,
        packed_seq_params: PackedSeqParams,
        use_inner_quantization_context: bool,
        padding_mask: torch.Tensor | None = None,
        extract_layer_indices=None,
        layer_offset: int = 0,
        per_layer_input: torch.Tensor | None = None,
    ):
        if per_layer_input is None:
            return super()._checkpointed_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                use_inner_quantization_context=use_inner_quantization_context,
                padding_mask=padding_mask,
                extract_layer_indices=extract_layer_indices,
                layer_offset=layer_offset,
            )

        if extract_layer_indices is None:
            extract_layer_indices = set()
        intermediate_hidden_states = []

        def custom(start: int, end: int):

            def custom_forward(
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                padding_mask=None,
                per_layer_input=None,
            ):
                for index in range(start, end):
                    layer = self._get_layer(index)
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(self.config, layer.layer_number - 1)
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(self.config, layer.layer_number - 1)
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()

                    with inner_quantization_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            attention_bias=attention_bias,
                            inference_context=None,
                            packed_seq_params=packed_seq_params,
                            padding_mask=padding_mask,
                            per_layer_input=per_layer_input,
                        )
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            if self.config.fp8 or self.config.fp4:
                return gpt_model.te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    padding_mask,
                    per_layer_input,
                )
            return tensor_parallel.checkpoint(
                forward_func,
                self.config.distribute_saved_activations,
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                padding_mask,
                per_layer_input,
            )

        if self.config.recompute_method == 'uniform':
            layer_idx = 0
            while layer_idx < self.num_layers_per_pipeline_rank:
                chunk_end = min(layer_idx + self.config.recompute_num_layers, self.num_layers_per_pipeline_rank)
                hidden_states, context = checkpoint_handler(custom(layer_idx, chunk_end))
                for idx in range(layer_idx, chunk_end):
                    if (idx + layer_offset) in extract_layer_indices and idx == chunk_end - 1:
                        intermediate_hidden_states.append(hidden_states)
                layer_idx += self.config.recompute_num_layers
        elif self.config.recompute_method == 'block':
            recompute_skip_num_layers = 0
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    layer_idx >= recompute_skip_num_layers
                    and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
                else:
                    hidden_states, context = custom(layer_idx, layer_idx + 1)(
                        hidden_states,
                        attention_mask,
                        context,
                        context_mask,
                        rotary_pos_emb,
                        padding_mask,
                        per_layer_input,
                    )
                if (layer_idx + layer_offset) in extract_layer_indices:
                    intermediate_hidden_states.append(hidden_states)
        else:
            raise ValueError('Invalid activation recompute method.')

        if len(extract_layer_indices) > 0:
            return hidden_states, intermediate_hidden_states
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor | WrappedTensor,
        attention_mask: torch.Tensor | None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        extract_layer_indices=None,
        *,
        inference_params: BaseInferenceContext | None = None,
        dynamic_inference_decode_only: bool | None = None,
        per_layer_input: torch.Tensor | None = None,
    ):
        if per_layer_input is None:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                padding_mask=padding_mask,
                extract_layer_indices=extract_layer_indices,
                inference_params=inference_params,
                dynamic_inference_decode_only=dynamic_inference_decode_only,
            )

        inference_context = deprecate_inference_params(inference_context, inference_params)
        if extract_layer_indices is None:
            extract_layer_indices = set()
        intermediate_hidden_states = []

        pp_group = self.pg_collection.pp if hasattr(self.pg_collection, 'pp') else None
        layer_offset = get_transformer_layer_offset(self.config, self.vp_stage, get_pg_rank(pp_group))

        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()
        if not self.pre_process:
            hidden_states = self.input_tensor

        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        rng_context = tensor_parallel.get_cuda_rng_tracker().fork() if self.config.sequence_parallel else nullcontext()

        if self.config.fp8:
            use_outer_quantization_context = self.config.fp8_recipe == Fp8Recipe.delayed
            use_inner_quantization_context = self.config.fp8_recipe != Fp8Recipe.delayed
            outer_quantization_context = get_fp8_context(self.config) if use_outer_quantization_context else nullcontext()
        elif self.config.fp4:
            use_inner_quantization_context = True
            outer_quantization_context = nullcontext()
        else:
            use_inner_quantization_context = False
            outer_quantization_context = nullcontext()

        with rng_context, outer_quantization_context:
            if self.config.recompute_granularity == 'full' and self.training:
                checkpointed_result = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    use_inner_quantization_context=use_inner_quantization_context,
                    padding_mask=padding_mask,
                    extract_layer_indices=extract_layer_indices,
                    layer_offset=layer_offset,
                    per_layer_input=per_layer_input,
                )
                if len(extract_layer_indices) > 0:
                    hidden_states, intermediate_hidden_states = checkpointed_result
                else:
                    hidden_states = checkpointed_result
            else:
                for l_no, layer in enumerate(self.layers):
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(self.config, layer.layer_number - 1)
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(self.config, layer.layer_number - 1)
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()

                    with self.offload_context, inner_quantization_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            rotary_pos_cos_sin=rotary_pos_cos_sin,
                            attention_bias=attention_bias,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                            padding_mask=padding_mask,
                            per_layer_input=per_layer_input,
                        )
                    if (
                        torch.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)
                    if (l_no + layer_offset) in extract_layer_indices:
                        intermediate_hidden_states.append(hidden_states)

        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        if not self.pre_process and len(self.layers) == 0 and not self.final_layernorm:
            hidden_states = hidden_states.clone()
        if len(extract_layer_indices) > 0:
            return hidden_states, intermediate_hidden_states
        return hidden_states


class Gemma4GPTModel(GPTModel):

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.hidden_size_per_layer_input = getattr(config, 'hidden_size_per_layer_input', 0) or 0
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = VocabParallelEmbedding(
                config.vocab_size_per_layer_input,
                config.num_layers * self.hidden_size_per_layer_input,
                init_method=config.init_method,
                reduce_scatter_embeddings=False,
                config=config,
                tp_group=self.pg_collection.tp,
            )
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size,
                config.num_layers * self.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_projection_norm = Gemma4RMSNorm(self.hidden_size_per_layer_input, eps=config.layernorm_epsilon)
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection_scale = config.hidden_size**-0.5
            device = self.embedding.word_embeddings.weight.device
            self.embed_tokens_per_layer.to(device=device, dtype=config.params_dtype)
            self.per_layer_model_projection.to(device=device, dtype=config.params_dtype)
            self.per_layer_projection_norm.to(device=device, dtype=config.params_dtype)

    def get_per_layer_inputs(self, input_ids: torch.Tensor):
        per_layer_inputs = self.embed_tokens_per_layer(input_ids)
        per_layer_inputs = per_layer_inputs.reshape(*input_ids.shape, self.config.num_layers, self.hidden_size_per_layer_input)
        if per_layer_inputs.dim() == 4:
            per_layer_inputs = per_layer_inputs.transpose(0, 1).contiguous()
        return per_layer_inputs

    def project_per_layer_inputs(self, inputs_embeds: torch.Tensor, per_layer_inputs: torch.Tensor | None = None):
        per_layer_projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        if per_layer_inputs is None:
            return per_layer_projection
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        decoder_input: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_context=None,
        packed_seq_params=None,
        extra_block_kwargs: dict = None,
        runtime_gather_output=None,
        *,
        inference_params=None,
        loss_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        extra_block_kwargs = dict(extra_block_kwargs or {})
        if self.hidden_size_per_layer_input and decoder_input is not None:
            per_layer_inputs = self.get_per_layer_inputs(input_ids)
            extra_block_kwargs['per_layer_input'] = self.project_per_layer_inputs(decoder_input, per_layer_inputs)
        return super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            runtime_gather_output=runtime_gather_output,
            inference_params=inference_params,
            loss_mask=loss_mask,
            **kwargs,
        )


class Gemma4MultimodalGPTModel(MultimodalGPTModel):

    def __init__(self, config, transformer_layer_spec, pre_process=True, post_process=True, *_args, **kwargs):
        nn.Module.__init__(self)
        self.config = config
        self.pre_process = pre_process
        self.post_process = post_process
        self.language_model = Gemma4GPTModel(config, transformer_layer_spec, pre_process, post_process, *_args, **kwargs)
        self.vp_stage = self.language_model.vp_stage
        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights
        self.model_meta = config.model_meta
        self.visual = None
        if pre_process and self.model_meta.visual_cls is not None:
            self.visual = self.model_meta.visual_cls(config)


class Gemma4Bridge(MultimodalGPTBridge):

    @staticmethod
    def _is_sliding_layer(config, layer_idx: int) -> bool:
        layer_types = getattr(config, 'layer_types', None) or []
        return layer_idx >= len(layer_types) or layer_types[layer_idx] == 'sliding_attention'

    def _patch_layer_attention_config(self, layer_idx: int):
        class _Ctx:

            def __init__(self, bridge):
                self.bridge = bridge
                self.orig_kv_channels = bridge.config.kv_channels
                self.orig_num_query_groups = bridge.config.num_query_groups

            def __enter__(self):
                if not Gemma4Bridge._is_sliding_layer(self.bridge.config, layer_idx):
                    self.bridge.config.kv_channels = self.bridge.config.global_kv_channels or self.bridge.config.kv_channels
                    if self.bridge.config.num_global_query_groups is not None:
                        self.bridge.config.num_query_groups = self.bridge.config.num_global_query_groups

            def __exit__(self, exc_type, exc, tb):
                self.bridge.config.kv_channels = self.orig_kv_channels
                self.bridge.config.num_query_groups = self.orig_num_query_groups

        return _Ctx(self)

    def _get_tp_split_dim(self, mg_key):
        if mg_key == 'embed_tokens_per_layer.weight':
            return 0
        return super()._get_tp_split_dim(mg_key)

    def _set_attn_state(self, mg_attn, hf_state_dict, hf_prefix: str, layer_idx: int, to_mcore: bool):
        with self._patch_layer_attention_config(layer_idx):
            return super()._set_attn_state(mg_attn, hf_state_dict, hf_prefix, layer_idx, to_mcore)

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        hf_state_dict = super()._set_layer_attn(mg_layer, hf_state_dict, layer_idx, to_mcore)
        self._set_state_dict(mg_layer, 'self_attention.post_self_attn_layernorm.weight', hf_state_dict,
                             'post_attention_layernorm.weight', to_mcore)
        return hf_state_dict

    def _set_layer_mlp(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        hf_state_dict = super()._set_layer_mlp(mg_layer, hf_state_dict, layer_idx, to_mcore)
        self._set_state_dict(mg_layer, 'mlp.post_mlp_layernorm.weight', hf_state_dict,
                             'post_feedforward_layernorm.weight', to_mcore)
        if getattr(self.config, 'hidden_size_per_layer_input', 0):
            self._set_state_dict(mg_layer, 'per_layer_input_gate.weight', hf_state_dict, 'per_layer_input_gate.weight',
                                 to_mcore)
            self._set_state_dict(mg_layer, 'per_layer_projection.weight', hf_state_dict, 'per_layer_projection.weight',
                                 to_mcore)
            self._set_state_dict(mg_layer, 'post_per_layer_input_norm.weight', hf_state_dict,
                                 'post_per_layer_input_norm.weight', to_mcore)
        return hf_state_dict

    def _convert_pre_process(self, mg_model, hf_state_dict, hf_prefix: str, to_mcore):
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        lm_model = getattr(mg_model, 'language_model') if self.is_multimodal else mg_model
        self._set_state_dict(lm_model, 'embedding.word_embeddings.weight', hf_state_dict, self.hf_embed_key, to_mcore)
        if getattr(self.config, 'hidden_size_per_layer_input', 0):
            self._set_state_dict(lm_model, 'embed_tokens_per_layer.weight', hf_state_dict,
                                 'model.language_model.embed_tokens_per_layer.weight', to_mcore)
            self._set_state_dict(lm_model, 'per_layer_model_projection.weight', hf_state_dict,
                                 'model.language_model.per_layer_model_projection.weight', to_mcore)
            self._set_state_dict(lm_model, 'per_layer_projection_norm.weight', hf_state_dict,
                                 'model.language_model.per_layer_projection_norm.weight', to_mcore)
        if self.is_multimodal:
            for prefix, mg_prefix in self.module_mapping.items():
                mg_module = getattr(mg_model.visual, mg_prefix)
                hf_state_dict.update(self._set_module(mg_module, hf_state_dict, f'{hf_prefix}{prefix}.', to_mcore))
        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict


class Gemma4Loader(ModelLoader):
    model_cls = Gemma4MultimodalGPTModel

    def _patch_transformer_block(self):
        if hasattr(gpt_model, 'OriginTransformerBlock'):
            return
        gpt_model.OriginTransformerBlock = gpt_model.TransformerBlock
        gpt_model.TransformerBlock = Gemma4TransformerBlock

    def __init__(self, config):
        super().__init__(config)
        self._patch_transformer_block()

    def get_transformer_layer_spec(self, vp_stage=None):
        self.config.qk_layernorm = True
        layer_spec = self._get_transformer_layer_spec()
        layer_spec.module = Gemma4TransformerLayer
        layer_spec.submodules.self_attention.module = Gemma4SelfAttention
        return layer_spec


class Gemma4Vit(HuggingFaceVit):
    module_mapping = {
        'model.vision_tower': 'vision_tower',
        'model.embed_vision': 'embed_vision',
        'model.audio_tower': 'audio_tower',
        'model.embed_audio': 'embed_audio',
    }
    _vision_tower = ['vision_tower', 'audio_tower']
    _aligner = ['embed_vision', 'embed_audio']

    @staticmethod
    def _expand_modal_mask(mask: torch.Tensor, inputs_embeds: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 2 and inputs_embeds.dim() == 3:
            if mask.shape[:2] == inputs_embeds.shape[:2]:
                pass
            elif mask.shape[0] == inputs_embeds.shape[1] and mask.shape[1] == inputs_embeds.shape[0]:
                mask = mask.transpose(0, 1)
        return mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

    def prepare_model(self, hf_config: PretrainedConfig):
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4AudioModel,
            Gemma4Model,
            Gemma4MultimodalEmbedder,
            Gemma4VisionModel,
        )

        self.vision_tower = Gemma4VisionModel._from_config(hf_config.vision_config)
        self.embed_vision = Gemma4MultimodalEmbedder(hf_config.vision_config, hf_config.text_config).to(
            self.vision_tower.dtype)
        self.audio_tower = None
        self.embed_audio = None
        if hf_config.audio_config is not None:
            self.audio_tower = Gemma4AudioModel._from_config(hf_config.audio_config)
            self.embed_audio = Gemma4MultimodalEmbedder(hf_config.audio_config, hf_config.text_config).to(
                self.audio_tower.dtype)
        self.model_cls = Gemma4Model

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        input_ids = kwargs['input_ids']
        pixel_values = kwargs.get('pixel_values')
        pixel_values_videos = kwargs.get('pixel_values_videos')
        input_features = kwargs.get('input_features')
        input_features_mask = kwargs.get('input_features_mask')

        image_mask, video_mask, audio_mask = self.get_placeholder_mask(input_ids=input_ids, inputs_embeds=inputs_embeds)

        if pixel_values is None and pixel_values_videos is None and input_features is None:
            dummy = self._get_dummy_dependency(inputs_embeds)
            return inputs_embeds + dummy * 0.

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                image_position_ids=kwargs.get('image_position_ids'),
                return_dict=True,
            ).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = self._expand_modal_mask(image_mask, inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

        if pixel_values_videos is not None:
            video_features = self.get_video_features(
                pixel_values_videos=pixel_values_videos,
                video_position_ids=kwargs.get('video_position_ids'),
                return_dict=True,
            ).pooler_output
            video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
            video_mask = self._expand_modal_mask(video_mask, inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_features)

        if input_features is not None and input_features_mask is not None:
            audio_output = self.get_audio_features(input_features, input_features_mask, return_dict=True)
            audio_features = audio_output.pooler_output
            audio_mask_from_encoder = audio_output.attention_mask
            audio_features = audio_features[audio_mask_from_encoder]
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_mask = self._expand_modal_mask(audio_mask, inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        return inputs_embeds

    def _get_dummy_dependency(self, inputs_embeds):
        deps = []
        for module_name in ('vision_tower', 'embed_vision', 'audio_tower', 'embed_audio'):
            module = getattr(self, module_name, None)
            if module is None:
                continue
            try:
                deps.append(next(module.parameters()).mean())
            except StopIteration:
                continue
        if not deps:
            return inputs_embeds.new_zeros(())
        return sum(dep.to(inputs_embeds.device, inputs_embeds.dtype) for dep in deps)

    def get_placeholder_mask(self, *args, **kwargs):
        with self.patch_hf_config():
            return self.model_cls.get_placeholder_mask(self, *args, **kwargs)

    def get_image_features(self, *args, **kwargs):
        with self.patch_hf_config():
            return self.model_cls.get_image_features(self, *args, **kwargs)

    def get_video_features(self, *args, **kwargs):
        with self.patch_hf_config():
            return self.model_cls.get_video_features(self, *args, **kwargs)

    def get_audio_features(self, *args, **kwargs):
        with self.patch_hf_config():
            return self.model_cls.get_audio_features(self, *args, **kwargs)


register_model(
    ModelMeta(
        ModelType.gemma4,
        ['gemma4'],
        bridge_cls=Gemma4Bridge,
        visual_cls=Gemma4Vit,
        loader=Gemma4Loader,
    ))
