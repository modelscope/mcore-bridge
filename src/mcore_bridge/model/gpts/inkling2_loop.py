# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import re
import torch
from collections import defaultdict
from dataclasses import dataclass
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import ModulesToSaveWrapper
from safetensors import safe_open
from typing import Optional

from mcore_bridge.bridge import MultimodalGPTBridge
from mcore_bridge.config import ModelConfig
from mcore_bridge.utils import is_master
from mcore_bridge.utils.safetensors import LazyTensor

from ..constant import ModelType
from ..gpt_model import GPTModel
from ..mm_gpt_model import MultimodalGPTModel
from ..mm_gpts.utils import HuggingFaceVit
from ..modules.inkling2_loop import (ATTENTION_PROJECTIONS, Inkling2LoopBlock, _apply_with_fp32_buffers,
                                     _sharded_children, require_inkling2_loop)
from ..register import ModelLoader, ModelMeta, register_model


@dataclass
class Inkling2LoopConfig(ModelConfig):
    inkling_loop_start: Optional[int] = None
    inkling_loop_end: Optional[int] = None
    inkling_loop_num_passes: Optional[int] = None
    use_te_activation_func: bool = True
    bias_activation_fusion: bool = False

    def __post_init__(self):
        require_inkling2_loop()
        text = self.hf_config.text_config
        loop_fields = {
            'inkling_loop_start': text.loop_layer_index_start,
            'inkling_loop_end': text.loop_layer_index_end,
            'inkling_loop_num_passes': text.loop_num_passes,
        }
        for name, value in loop_fields.items():
            if getattr(self, name) is None:
                setattr(self, name, value)
            elif getattr(self, name) != value:
                raise ValueError(f'{name} must match the checkpoint; changing the loop changes the model.')
        loop_config = getattr(text, 'loop_config', {})
        if isinstance(loop_config, dict) and loop_config.get('is_varloop', False):
            raise ValueError('Variable-pass Inkling research models are not supported.')
        for name in ('tensor_model_parallel_size', 'pipeline_model_parallel_size', 'context_parallel_size',
                     'expert_tensor_parallel_size'):
            if getattr(self, name) != 1:
                raise ValueError(f'Inkling-2-Mini requires {name}=1; only DP and EP are supported.')
        if self.sequence_parallel or self.virtual_pipeline_model_parallel_size is not None:
            raise ValueError('Inkling-2-Mini does not support SP or virtual pipeline parallelism.')
        if self.mtp_num_layers:
            raise ValueError('Inkling-2-Mini carries an independent DSpark draft; set mtp_num_layers=None.')
        unsupported_precision = {
            name: getattr(self, name)
            for name in ('fp8', 'fp4', 'fp8_param', 'fp4_param', 'fp16') if getattr(self, name)
        }
        if unsupported_precision:
            raise ValueError('Inkling-2-Mini training supports BF16/FP32 only, without FP8/FP4 parameter gathering: '
                             f'{unsupported_precision}. Disable these options.')
        if self.task_type != 'causal_lm' or self.position_embedding_type != 'none' or self.rope_scaling is not None:
            raise ValueError('Inkling-2-Mini supports causal LM with relative attention, not RoPE or scoring heads.')
        if self.moe_router_load_balancing_type not in ('none', ['none']) or self.moe_router_enable_expert_bias:
            raise ValueError(
                'Inkling-2-Mini uses a fixed correction bias; disable online expert bias and auxiliary balancing.')
        if self.moe_router_score_function != 'sigmoid' or self.moe_router_pre_softmax:
            raise ValueError('Inkling-2-Mini requires sigmoid routing with normalization after top-k selection.')
        if any(
                getattr(self, name) is not None
                for name in ('moe_router_num_groups', 'moe_router_group_topk', 'moe_router_topk_limited_devices')):
            raise ValueError('Inkling-2-Mini does not support group-limited expert routing.')
        if self.moe_enable_routing_replay:
            raise ValueError('Inkling-2-Mini does not support routing replay; use router_replay_mode="disabled".')
        if self.bias_activation_fusion:
            raise ValueError('Inkling-2-Mini uses a custom soft-capped expert activation; '
                             'set bias_activation_fusion=False.')
        if self.moe_router_dtype != 'fp32' or not self.use_te_activation_func or not self.moe_grouped_gemm:
            raise ValueError('Inkling-2-Mini requires FP32 routing and its custom grouped-expert activation.')
        if (self.moe_token_dispatcher_type != 'alltoall' or self.moe_enable_deepep
                or self.moe_shared_expert_intermediate_size):
            raise ValueError('Inkling-2-Mini supports the native alltoall dispatcher without shared experts.')
        if self.moe_use_grouped_tensor:
            raise ValueError('Inkling-2-Mini checkpoint conversion requires per-expert grouped-linear weights; '
                             'set moe_use_grouped_tensor=False.')
        if self.moe_permute_fusion:
            raise ValueError('Inkling-2-Mini ordered FP32 expert accumulation requires moe_permute_fusion=False.')
        if self.moe_expert_capacity_factor is not None or self.moe_apply_probs_on_input:
            raise ValueError(
                'Inkling-2-Mini does not support dropping tokens or applying router probabilities to inputs.')
        if self.use_transformer_engine_op_fuser or self.cuda_graph_impl != 'none' or self.delay_wgrad_compute:
            raise ValueError('Inkling-2-Mini does not support TE op fusion, CUDA graphs or delayed weight gradients.')
        if self.recompute_granularity == 'selective' and set(self.recompute_modules) - {'moe', 'moe_act'}:
            raise ValueError(
                'Use full recomputation for Inkling-2-Mini attention; selective supports moe/moe_act only.')
        if self.recompute_granularity == 'full' and (self.recompute_method not in (None, 'uniform')
                                                     or self.recompute_num_layers not in (None, 1)):
            raise ValueError('Inkling-2-Mini full recompute uses uniform one-layer groups; set recompute_num_layers=1.')
        super().__post_init__()


class Inkling2LoopVit(HuggingFaceVit):
    module_mapping = {'model.vision_tower': 'vision_tower'}
    _vision_tower = ['vision_tower']
    _aligner = ['vision_tower.merger']

    def prepare_attn_impl(self):
        self.hf_config.vision_config._attn_implementation = self.config.vit_attn_impl or 'eager'

    def prepare_model(self, hf_config):
        from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel

        self.vision_tower = Inkling2VisionModel._from_config(hf_config.vision_config)
        _, mark_keep_in_fp32 = require_inkling2_loop()
        # Native vision RoPE tables are FP32 even when the trainable weights are BF16.
        for buffer in self.vision_tower.rotary_emb.buffers():
            mark_keep_in_fp32(buffer)

    def _apply(self, fn, recurse=True):
        return _apply_with_fp32_buffers(self, super()._apply, fn, recurse)

    def prepare_language_model(self, hf_config):
        self.vision_tower = None

    def _image_mask(self, inputs):
        input_ids = inputs['input_ids']
        token_types = inputs.get('mm_token_type_ids')
        if token_types is None:
            return input_ids == self.hf_config.image_token_id
        if token_types.shape != input_ids.shape or ((token_types != 0) & (token_types != 1)).any():
            raise ValueError('Inkling mm_token_type_ids must match input_ids and contain only text/image types.')
        image_mask = token_types == 1
        if (image_mask & (input_ids != self.hf_config.image_token_id)).any():
            raise ValueError('Inkling image types must identify image placeholder tokens.')
        return image_mask

    def get_inputs_embeds_language_model(self, inputs_embeds, **kwargs):
        if (kwargs.get('pixel_values') is not None or kwargs.get('image_grid_thw') is not None
                or self._image_mask(kwargs).any()):
            raise ValueError('Inkling language_model_only=True rejects images; set language_model_only=False.')
        return inputs_embeds

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        pixel_values = kwargs.get('pixel_values')
        grid = kwargs.get('image_grid_thw')
        image_mask = self._image_mask(kwargs)
        if (pixel_values is None) != (grid is None):
            raise ValueError('Inkling image inputs require both pixel_values and image_grid_thw.')
        if pixel_values is None:
            if image_mask.any():
                raise ValueError('Inkling image placeholders require pixel_values and image_grid_thw.')
            if any(parameter.requires_grad for parameter in self.vision_tower.parameters()):
                size = self.hf_config.vision_config.patch_size * self.hf_config.vision_config.spatial_merge_size
                dummy = inputs_embeds.new_zeros(1, size, size, 3)
                features = self.vision_tower(
                    pixel_values=dummy, image_grid_thw=torch.ones(1, 3, dtype=torch.long)).pooler_output
                # Text-only ranks must participate in replicated vision gradient reduction.
                inputs_embeds = inputs_embeds + features.sum().to(inputs_embeds.dtype) * 0.
            return inputs_embeds
        count = grid.prod(-1).sum().item()
        if image_mask.sum().item() != count:
            raise ValueError('Inkling image placeholder and grid token counts differ.')
        features = self.vision_tower(pixel_values=pixel_values, image_grid_thw=grid.cpu()).pooler_output
        if features.shape != (count, inputs_embeds.shape[-1]):
            raise ValueError('Inkling image feature and placeholder shapes differ.')
        # Vision features bypass the text embedding normalization inside the decoder.
        return {'inputs_embeds': inputs_embeds, 'image_mask': image_mask, 'image_embeds': features}

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        return _sharded_children(self, prefix, sharded_offsets, metadata)


class Inkling2LoopGPTModel(GPTModel):

    def forward(self, input_ids, position_ids, attention_mask=None, **kwargs):
        extra = dict(kwargs.pop('extra_block_kwargs', None) or {})
        extra['position_ids'] = position_ids
        return super().forward(input_ids, position_ids, attention_mask, extra_block_kwargs=extra, **kwargs)

    def _forward_output_layer(self, hidden_states, *args, **kwargs):
        logits = super()._forward_output_layer(hidden_states, *args, **kwargs).float()
        text = self.config.hf_config.text_config
        logits = torch.tanh(logits / text.final_logit_softcapping) * text.final_logit_softcapping
        indices = torch.arange(logits.shape[-1], device=logits.device)
        return logits.masked_fill(indices >= text.unpadded_vocab_size, float('-inf'))


class Inkling2LoopModel(MultimodalGPTModel):
    language_model_cls = Inkling2LoopGPTModel


class Inkling2LoopLoader(ModelLoader):
    model_cls = Inkling2LoopModel
    transformer_block = Inkling2LoopBlock

    def get_transformer_layer_spec(self, vp_stage=None):
        return TransformerBlockSubmodules(layer_specs=[])

    def get_mtp_block_spec(self, transformer_layer_spec, vp_stage=None):
        return None


class Inkling2LoopBridge(MultimodalGPTBridge):
    hf_layers_prefix = 'model.llm.layers'
    hf_embed_key = 'model.llm.embed.weight'
    hf_final_layernorm_key = 'model.llm.norm.weight'
    hf_lm_head_key = 'model.llm.unembed.weight'

    def __init__(self, config):
        super().__init__(config)
        self._source_native_format = False
        self._source_dtypes = {}
        self._source_model_dir = config.hf_config.name_or_path or None

    @staticmethod
    def _check_adapter_modules(models, peft_format=False):
        projections = '|'.join(re.escape(path) for path in ATTENTION_PROJECTIONS)
        pattern = rf'(?:^|\.)language_model\.decoder\.layers\.(\d+)\.self_attn\.({projections})\.linear$'
        unsupported, mapped = [], set()
        for model in models:
            for name, module in model.named_modules():
                if isinstance(module, ModulesToSaveWrapper):
                    unsupported.append(name)
                elif isinstance(module, BaseTunerLayer) and (peft_format or not module.merged):
                    match = re.search(pattern, name)
                    if match is None:
                        unsupported.append(name)
                    else:
                        mapped.add((match.group(1), match.group(2)))
        if unsupported:
            raise ValueError(f'Inkling HF export cannot preserve these unmerged adapters: {sorted(set(unsupported))}. '
                             'Use the six attention projection targets or merge and unload before full export.')
        return mapped

    @staticmethod
    def _check_adapter_keys(state, targets):
        projections = '|'.join(re.escape(path) for path in ATTENTION_PROJECTIONS)
        pattern = (rf'(?:^|\.)model\.(?:language_model|llm)\.layers\.(\d+)\.self_attn\.'
                   rf'({projections})\.lora_[AB]\.weight$')
        unsupported = [
            key for key in state
            if (match := re.search(pattern, key)) is None or (match.group(1), match.group(2)) not in targets
        ]
        if unsupported:
            raise ValueError(f'Inkling bridge cannot load these unmapped adapter tensors: {unsupported}.')

    def load_weights(self, mg_models, hf_model_dir, peft_format=False, adapter_name='default', converter=None):
        if peft_format:
            from mcore_bridge.utils import SafetensorLazyLoader
            targets = self._check_adapter_modules(mg_models, peft_format=True)
            with SafetensorLazyLoader(hf_model_dir) as source:
                self._check_adapter_keys(source.get_state_dict(), targets)
        if not peft_format:
            from mcore_bridge.utils import SafetensorLazyLoader
            with SafetensorLazyLoader(hf_model_dir) as source:
                for key, filename in source._weight_map.items():
                    self._source_dtypes[key] = source._open_file(filename).get_slice(key).get_dtype()
        return super().load_weights(
            mg_models, hf_model_dir, peft_format=peft_format, adapter_name=adapter_name, converter=converter)

    def _convert_hf_state_dict(self, state, to_mcore):
        if to_mcore:
            self._source_native_format = not self._peft_format and any(
                key.startswith('model.language_model.') for key in state)
        result = {}
        for key, value in state.items():
            if key.startswith(('model.visual.', 'model.vision_tower.')):
                key = self._convert_vision_key(key, to_mcore)
            if to_mcore and key.startswith('model.language_model.layers.'):
                key = key.replace('model.language_model.layers.', 'model.llm.layers.', 1)
            elif not to_mcore and self._peft_format and key.startswith('model.llm.layers.'):
                key = key.replace('model.llm.layers.', 'model.language_model.layers.', 1)
            if not to_mcore and not self._peft_format and value is not None:
                if self._source_dtypes.get(key) == 'BF16' and value.dtype == torch.float32:
                    rounded = value.bfloat16()
                    if torch.equal(value, rounded.float()):
                        value = rounded
            result[key] = value
        return result

    @staticmethod
    def _convert_vision_key(key, to_mcore):
        source_prefix, target_prefix = 'model.visual.', 'model.vision_tower.'
        if not to_mcore:
            source_prefix, target_prefix = target_prefix, source_prefix
        if not key.startswith(source_prefix):
            return key
        name = key[len(source_prefix):]
        position = ('backbone.embeddings.position_embedding', 'embeddings.position_embedding.weight')
        source, target = position if to_mcore else position[::-1]
        if name == source:
            return target_prefix + target
        prefixes = (
            ('backbone.embeddings.', 'embeddings.'),
            ('backbone.ln_pre.', 'pre_layernorm.'),
            ('backbone.layers.', 'layers.'),
            ('proj_in.', 'merger.linear_1.'),
            ('proj_out.', 'merger.linear_2.'),
        )
        for source, target in prefixes:
            source, target = (source, target) if to_mcore else (target, source)
            if name.startswith(source):
                name = target + name[len(source):]
                break
        for source, target in (
            ('.ln_1.', '.layernorm_before.'),
            ('.ln_2.', '.layernorm_after.'),
            ('.mlp.c_fc.', '.mlp.fc1.'),
            ('.mlp.c_proj.', '.mlp.fc2.'),
            ('.ls_1.gamma', '.lambda_1'),
            ('.ls_2.gamma', '.lambda_2'),
        ):
            source, target = (source, target) if to_mcore else (target, source)
            name = name.replace(source, target, 1)
        return target_prefix + name

    def _set_layer_state(self, layer, state, hf_prefix, index, to_mcore):
        prefix = f'{hf_prefix}{index}.'
        local = self._remove_prefix(state, prefix) if to_mcore else {}
        native_names = self._peft_format or to_mcore and self._source_native_format
        projection_sources = ('attn.wq_du', 'attn.wk_dv', 'attn.wv_dv', 'attn.wr_du', 'attn.gate_attn', 'attn.wo_ud')
        for path, source in zip(ATTENTION_PROJECTIONS, projection_sources):
            hf_key = f'self_attn.{path}.weight' if native_names else f'{source}.weight'
            self._set_state_dict(layer, f'self_attn.{path}.linear.weight', local, hf_key, to_mcore)
        if not self._peft_format:
            mapping = {
                'self_attn.sink_proj.weight': 'attn.attn_sink_proj.weight',
                'self_attn.q_proj.norm.scale': 'attn.q_l2_clip.scale',
                'self_attn.relative_position_bias.norm.scale': 'attn.rel_l2_clip.scale',
                'self_attn.relative_position_bias.proj': 'attn.rel_logits_proj.proj',
                'post_attention_layernorm.weight': 'mlp_norm.weight',
                'post_attention_convolution.conv1d.weight': 'attn_sconv.weight',
                'post_mlp_convolution.conv1d.weight': 'mlp_sconv.weight',
            }
            if layer.attention_type == 'sliding_attention':
                mapping['input_layernorm.weight'] = 'attn_norm.weight'
                for kind in ('k', 'v'):
                    mapping[f'self_attn.{kind}_proj.sconv.conv1d.weight'] = f'attn.{kind}_sconv.weight'
            else:
                for kind in ('q', 'k', 'v'):
                    mapping[f'{kind}_input_layernorm.weight'] = f'{kind}_input_norm.weight'
                for kind in ('k', 'v'):
                    mapping[f'self_attn.{kind}_proj.sconv.static_weight'] = f'attn.{kind}_sconv.static_w'
                    mapping[f'self_attn.{kind}_proj.sconv.weight_proj.weight'] = f'attn.{kind}_sconv.weight_proj'
            if self.config.inkling_loop_start <= index < self.config.inkling_loop_end:
                for kind, source in (('q', 'attn'), ('k', 'k'), ('v', 'v'), ('mlp', 'mlp')):
                    for target, weight in (('down_proj', 'down_weight'), ('forget_gate_proj', 'forget_weight'),
                                           ('message_gate_proj', 'gain_weight')):
                        mapping[f'{kind}_message_gate.{target}.weight'] = f'controllers.{source}.{weight}'
            for target, source in mapping.items():
                self._set_state_dict(layer, target, local, target if native_names else source, to_mcore)
            moe = layer.mlp.native
            self._set_state_dict(moe, 'router.weight', local, 'mlp.gate.weight', to_mcore)
            self._set_state_dict(moe, 'router.e_score_correction_bias', local,
                                 'mlp.gate.e_score_correction_bias' if native_names else 'mlp.gate.bias', to_mcore)
            local.update(self._set_experts(moe.experts, local, to_mcore))
        result = self._add_prefix(local, prefix) if not to_mcore else {}
        if not self._peft_format:
            queries = [('mlp_input.query', f'model.llm.attn_res.mlp_queries.{index}')]
            if index > 0:
                queries.append(('attention_input.query', f'model.llm.attn_res.attn_queries.{index - 1}'))
            for target, source in queries:
                if to_mcore:
                    query_state, key = (local, target) if native_names else (state, source)
                    self._set_state_dict(layer, target, query_state, key, True)
                else:
                    self._set_state_dict(layer, target, result, source, False)
        return result

    def _set_experts(self, experts, state, to_mcore):
        local_count = self.config.num_moe_experts // self.ep_size
        names = (('linear_fc1', 'mlp.experts.w13_weight', 'mlp.experts.gate_up_proj'),
                 ('linear_fc2', 'mlp.experts.w2_weight', 'mlp.experts.down_proj'))
        result = {}
        from transformers.core_model_loading import Inkling2LoopDequantizeExperts

        for target, source, native_source in names:
            linear = getattr(experts, target)
            if to_mcore:
                source_key = native_source if self._source_native_format else source
                first_expert = self.ep_rank * local_count
                op = Inkling2LoopDequantizeExperts()
                for offset in range(0, local_count, op.EXPERTS_PER_CHUNK):
                    count = min(op.EXPERTS_PER_CHUNK, local_count - offset)
                    indices = slice(first_expert + offset, first_expert + offset + count)
                    value = state[source_key].load_slice((indices, slice(None), slice(None)))
                    if value.dtype not in (torch.bfloat16, torch.float32):
                        patterns = [
                            source_key + suffix for suffix in ('', '_scale', '_amax', '_outer_scale_denominator')
                        ]
                        tensors = {patterns[0]: value, patterns[1]: state[patterns[1]].load_slice(indices)}
                        if value.dtype == torch.uint8:
                            tensors[patterns[2]] = state[patterns[2]].load_slice(indices)
                            tensors[patterns[3]] = state[patterns[3]].load()
                        value = op.convert(
                            tensors, patterns, [source_key], model=linear,
                            full_layer_name=f'weight{offset}')[source_key]
                    if target == 'linear_fc1' and not self._source_native_format:
                        value = value.reshape(count, -1, 2, value.shape[-1]).transpose(1, 2).reshape(value.shape)
                    weights = [getattr(linear, f'weight{i}') for i in range(offset, offset + count)]
                    self._set_weight(weights, value.reshape(-1, value.shape[-1]), f'{target}.weight', is_expert=True)
            else:
                complete = None
                for ep_rank in range(self.ep_size):
                    weights = [getattr(linear, f'weight{i}').data
                               for i in range(local_count)] if ep_rank == self.ep_rank else None
                    value, _ = self._get_weight(weights, f'{target}.weight', is_expert=True)
                    if value is None:
                        continue
                    if target == 'linear_fc1':
                        value = value.reshape(local_count, 2, -1, value.shape[-1]).transpose(1, 2).reshape(value.shape)
                    if complete is None:
                        complete = value.new_empty(self.config.num_moe_experts, *value.shape[1:])
                    complete[ep_rank * local_count:(ep_rank + 1) * local_count] = value
                if complete is not None:
                    result[source] = complete
        return result

    def _convert_pre_process(self, model, state, hf_prefix, to_mcore):
        if self._source_native_format and to_mcore:
            state = dict(state)
            state[self.hf_embed_key] = state['model.language_model.embed_tokens.weight']
        result = super()._convert_pre_process(model, state, hf_prefix, to_mcore)
        if not self._peft_format:
            key = ('model.language_model.embed_tokens.embed_norm.weight'
                   if self._source_native_format and to_mcore else 'model.llm.embed_norm.weight')
            self._set_state_dict(model.language_model.decoder, 'embedding_norm.weight', state if to_mcore else result,
                                 key, to_mcore)
        return result

    def _convert_post_process(self, model, state, hf_prefix, to_mcore):
        if self._source_native_format and to_mcore:
            state = dict(state)
            state[self.hf_final_layernorm_key] = state['model.language_model.norm.weight']
            state[self.hf_lm_head_key] = state['lm_head.weight']
        if to_mcore and not self._peft_format and self.hf_lm_head_key not in state:
            raise ValueError('Inkling-2-Mini full checkpoints must contain the unembedding weight.')
        result = super()._convert_post_process(model, state, hf_prefix, to_mcore)
        if not self._peft_format:
            decoder = model.language_model.decoder
            key = ('model.language_model.final_input.query'
                   if self._source_native_format and to_mcore else 'model.llm.attn_res.output_query')
            self._set_state_dict(decoder, 'final_input.query', state if to_mcore else result, key, to_mcore)
            message_key = 'model.language_model.message_input.query'
            q_key, kv_key = 'model.llm.middle_cycle_cross_pass_queries', 'model.llm.middle_cycle_cross_pass_kv_queries'
            if to_mcore:
                if self._source_native_format:
                    value = state[message_key].load()
                else:
                    queries = state[q_key].load().reshape(-1, 2, self.config.hidden_size)
                    kv_queries = state[kv_key].load().reshape_as(queries)
                    value = torch.cat((queries[:, :1], kv_queries, queries[:, 1:]), dim=1)
                self._set_state_dict(decoder, 'message_input.query', {message_key: LazyTensor(value)}, message_key,
                                     True)
            else:
                value, _ = self._get_weight(decoder.message_input.query.data, 'message_input.query')
                if value is not None:
                    result[q_key] = value[:, [0, 3]].flatten(0, 1).contiguous()
                    result[kv_key] = value[:, 1:3].flatten(0, 1).contiguous()
        return result

    def patch_export_hf_config(self, hf_config):
        hf_config.dtype = self.config.params_dtype
        hf_config.text_config.dtype = self.config.params_dtype
        for config in (hf_config, hf_config.text_config, hf_config.vision_config):
            config.__dict__.pop('quantization_config', None)

    def _save_missing_weights(self, saver, saved_keys, source_model_dir=None):
        if not is_master() or not self.config.language_model_only:
            return
        from mcore_bridge.utils import SafetensorLazyLoader
        source_dir = source_model_dir or self._source_model_dir
        with SafetensorLazyLoader(source_dir) as source:
            for key, tensor in source.get_state_dict().items():
                # Copy only frozen vision weights; stale language-model aliases could overwrite trained weights.
                if key.startswith(('model.visual.', 'model.vision_tower.')) and key not in saved_keys:
                    saver.add_tensor(key, tensor.load())

    def save_weights(self, models, output_dir, peft_format=False, **kwargs):
        self._check_adapter_modules(models, peft_format)
        kwargs.setdefault('save_missing_weights', not peft_format)
        if self.config.language_model_only and not peft_format and not kwargs['save_missing_weights']:
            raise ValueError('Complete Inkling HF export requires vision weights; set save_missing_weights=True '
                             'or pass the source HF directory when language_model_only=True.')
        super().save_weights(models, output_dir, peft_format=peft_format, **kwargs)
        if peft_format or not is_master():
            return
        source_precision = os.path.join(self._source_model_dir, 'hf_quant_config.json')
        if not os.path.isfile(source_precision):
            return
        with open(source_precision, encoding='utf-8') as f:
            precision = json.load(f)
        index_path = os.path.join(output_dir, 'model.safetensors.index.json')
        if os.path.isfile(index_path):
            with open(index_path, encoding='utf-8') as f:
                filenames = set(json.load(f)['weight_map'].values())
        else:
            filenames = {'model.safetensors'}
        dtypes = {}
        dtype_names = {'BF16': 'bfloat16', 'F32': 'float32'}
        for filename in filenames:
            with safe_open(os.path.join(output_dir, filename), framework='pt') as handle:
                for key in handle.keys():
                    dtypes[key] = dtype_names[handle.get_slice(key).get_dtype()]
        moved = defaultdict(list)
        for group in precision['model']['parameter_groups']:
            kept = []
            for key in group['tensors']:
                if any(
                        key.endswith(f'.experts.{weight}{suffix}') for weight in ('w13_weight', 'w2_weight')
                        for suffix in ('_scale', '_amax', '_outer_scale_denominator')):
                    continue
                if key not in dtypes:
                    raise ValueError(f'Inkling HF export lost a tensor referenced by precision metadata: {key}')
                if group['output_dtype'] == dtypes[key]:
                    kept.append(key)
                else:
                    moved[dtypes[key]].append(key)
            group['tensors'] = kept
        for dtype, keys in moved.items():
            precision['model']['parameter_groups'].append({
                'compute_dtype': dtype,
                'output_dtype': dtype,
                'required_activation_quantization': None,
                'weight_block_shape': [],
                'weight_transform': 'cast',
                'tensors': sorted(keys),
            })
        precision['model']['parameter_groups'] = [
            group for group in precision['model']['parameter_groups'] if group['tensors']
        ]
        with open(os.path.join(output_dir, 'hf_quant_config.json'), 'w', encoding='utf-8') as f:
            json.dump(precision, f, indent=2)


register_model(
    ModelMeta(
        ModelType.inkling2_loop,
        ['inkling2_loop'],
        bridge_cls=Inkling2LoopBridge,
        visual_cls=Inkling2LoopVit,
        loader=Inkling2LoopLoader,
        config_cls=Inkling2LoopConfig,
        mtp_depth_field=None,
    ))
