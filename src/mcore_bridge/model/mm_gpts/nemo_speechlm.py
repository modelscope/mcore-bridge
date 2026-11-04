# Copyright (c) ModelScope Contributors. All rights reserved.
"""Nemotron Transcribe with a replicated audio tower and a Nemotron-H backbone."""
import os
import re
import torch
from dataclasses import dataclass
from functools import lru_cache, partial
from transformers import PretrainedConfig
from typing import Optional

from mcore_bridge.config import ModelConfig
from mcore_bridge.utils.safetensors import LazyTensor

from ..constant import ModelType
from ..gpts.nemotron_h import NemotronHBridge, NemotronHLoader
from ..hybrid_model import HybridModel
from ..mm_gpt_model import MultimodalGPTModel
from ..register import ModelMeta, register_model
from .utils import HuggingFaceVit


def _load_nemo_expert(weight, expert_idx):
    return weight.load_slice((expert_idx, )).transpose(0, 1)


@dataclass
class NemoSpeechLMConfig(ModelConfig):
    make_vocab_size_divisible_by: int = 128
    actual_vocab_size: Optional[int] = None

    def __post_init__(self):
        runtime_vocab_size = self.padded_vocab_size
        output_vocab_size = self.actual_vocab_size
        divisor = self.make_vocab_size_divisible_by * self.tensor_model_parallel_size
        self.padded_vocab_size = ((runtime_vocab_size + divisor - 1) // divisor) * divisor
        super().__post_init__()
        self.actual_vocab_size = runtime_vocab_size if output_vocab_size is None else output_vocab_size


class NemoSpeechLMBridge(NemotronHBridge):
    hf_embed_key = 'llm.model.embed_tokens.weight'
    hf_layers_prefix = 'llm.model.layers'
    hf_final_layernorm_key = 'llm.model.norm.weight'
    hf_lm_head_key = 'llm.lm_head.weight'
    hf_mtp_prefix = 'llm.mtp.layers'
    hf_state_dict_mapping = {
        'mixer._fp32_params.A_log': 'mixer.A_log',
        'mixer._fp32_params.D': 'mixer.D',
        'mixer._fp32_params.dt_bias': 'mixer.dt_bias',
    }

    def _convert_hf_state_dict(self, hf_state_dict, to_mcore):
        result = {}
        for key, value in hf_state_dict.items():
            for old_key, new_key in self.hf_state_dict_mapping.items():
                if not to_mcore:
                    old_key, new_key = new_key, old_key
                if old_key in key:
                    key = key.replace(old_key, new_key)
                    break
            result[key] = value
        return result

    def _set_mlp_state(self,
                       mg_mlp,
                       hf_state_dict,
                       hf_prefix: str,
                       layer_idx: int,
                       to_mcore: bool,
                       ep_rank=None,
                       is_mtp: bool = False):
        if not to_mcore or ep_rank is None or self._peft_format:
            return super()._set_mlp_state(mg_mlp, hf_state_dict, hf_prefix, layer_idx, to_mcore, ep_rank, is_mtp)
        hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        num_local_experts = self.config.num_moe_experts // self.ep_size
        start_idx = ep_rank * num_local_experts
        for source, target in [('gate_and_up_projs', 'up_proj'), ('down_projs', 'down_proj')]:
            weight = hf_state_dict.pop(f'{source}.weight', None)
            if weight is None:
                weight = hf_state_dict.pop(source)
            for local_idx in range(num_local_experts):
                expert_idx = start_idx + local_idx
                hf_state_dict[f'{expert_idx}.{target}.weight'] = LazyTensor(
                    loader=partial(_load_nemo_expert, weight, expert_idx))
        return super()._set_mlp_state(mg_mlp, hf_state_dict, '', layer_idx, True, ep_rank, is_mtp)

    def _pack_nemo_experts(self, hf_state_dict):
        for hf_name, nemo_name in [('up_proj', 'gate_and_up_projs'), ('down_proj', 'down_projs')]:
            pattern = re.compile(rf'^(.*)\.(\d+)\.{hf_name}\.weight$')
            weights = {}
            prefixes = set()
            for key in list(hf_state_dict):
                match = pattern.match(key)
                if match:
                    weights[int(match.group(2))] = hf_state_dict.pop(key)
                    prefixes.add(match.group(1))
            if weights:
                if len(prefixes) != 1:
                    raise ValueError(f'MoE {hf_name} spans multiple experts prefixes: {sorted(prefixes)}')
                if len(weights) != self.config.num_moe_experts:
                    raise ValueError(f'MoE {hf_name} has {len(weights)} experts, expected '
                                     f'{self.config.num_moe_experts}')
                prefix = prefixes.pop()
                if prefix.endswith('.experts'):
                    prefix = prefix[:-len('.experts')]
                hf_state_dict[f'{prefix}.experts.{nemo_name}'] = torch.stack(
                    [weights[i].transpose(0, 1) for i in range(self.config.num_moe_experts)])
        return hf_state_dict

    def _set_moe_state(self, mg_mlp, hf_state_dict, hf_prefix, layer_idx, to_mcore, is_mtp=False):
        result = super()._set_moe_state(mg_mlp, hf_state_dict, hf_prefix, layer_idx, to_mcore, is_mtp)
        if not to_mcore and not self._peft_format:
            result = self._pack_nemo_experts(result)
        return result

    def _set_module(self, mg_module, hf_state_dict, hf_prefix, to_mcore):
        if self._peft_format or not to_mcore or hf_prefix != 'perception.':
            return super()._set_module(mg_module, hf_state_dict, hf_prefix, to_mcore)
        if mg_module is None:
            return {}
        state = {
            key: value.load()
            for key, value in self._remove_prefix(hf_state_dict, hf_prefix).items() if not key.endswith('._extra_state')
        }
        incompatible = mg_module.load_state_dict(state, strict=False)
        missing = [key for key in incompatible.missing_keys if not key.endswith('._extra_state')]
        unexpected = [key for key in incompatible.unexpected_keys if not key.endswith('._extra_state')]
        if missing or unexpected:
            raise ValueError(f'Perception checkpoint mismatch: missing={missing}, unexpected={unexpected}')
        return {}

    def get_hf_meta_model(self):
        hf_config = self.config.hf_config
        if not hasattr(hf_config, 'audio_config'):
            raise ValueError('Nemotron Transcribe requires a newer Transformers version (>5.21) and '
                             'checkpoint metadata with audio_config; legacy NeMo metadata is not supported.')
        from transformers import AutoModelForMultimodalLM
        with torch.device('meta'):
            return AutoModelForMultimodalLM.from_config(hf_config)

    def save_weights(self, mg_models, output_dir, peft_format=False, save_missing_weights=False, **kwargs):
        if not peft_format:
            source_model_dir = (
                save_missing_weights if isinstance(save_missing_weights, str) else self._source_model_dir)
            if source_model_dir is None or not os.path.isdir(source_model_dir):
                raise ValueError(
                    'Exporting a complete Nemotron Transcribe checkpoint requires the source checkpoint to '
                    'preserve MTP, CTC and _extra_state weights. Pass the original --model directory.')
            save_missing_weights = True
        return super().save_weights(
            mg_models, output_dir, peft_format=peft_format, save_missing_weights=save_missing_weights, **kwargs)


@lru_cache(maxsize=1)
def _perception_transforms():
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    from transformers.core_model_loading import WeightConverter, WeightRenaming

    transforms = get_checkpoint_conversion_mapping('nemo_speechlm')
    if not transforms:
        raise ImportError('Nemotron Transcribe requires a newer Transformers version (>5.21) with the '
                          'nemo_speechlm checkpoint conversion mapping.')
    renamings = [transform for transform in transforms if isinstance(transform, WeightRenaming)]
    converters = [transform for transform in transforms if isinstance(transform, WeightConverter)]
    if len(renamings) + len(converters) != len(transforms):
        raise ValueError('Unsupported Nemotron Transcribe checkpoint transforms.')
    return (renamings, converters, [transform.reverse_transform() for transform in renamings],
            [transform.reverse_transform() for transform in converters])


def _strip_perception_target(key):
    if key.startswith('model.audio_tower.'):
        return 'encoder.' + key[len('model.audio_tower.'):]
    if key.startswith('model.multi_modal_projector.'):
        return 'proj.' + key[len('model.multi_modal_projector.'):]
    return None


def _convert_perception_state(state):
    from transformers.core_model_loading import rename_source_key

    renamings, converters, _, _ = _perception_transforms()
    out = {}
    for key, tensor in state.items():
        source = key if key.startswith('perception.') else f'perception.{key}'
        target, pattern = rename_source_key(source, renamings, converters)
        if pattern is not None and 'w_qkv' in pattern:
            base = target.rsplit('self_attn.q_proj.weight', 1)[0]
            for part, value in zip(('q', 'k', 'v'), tensor.chunk(3, dim=0), strict=True):
                mapped = _strip_perception_target(f'{base}self_attn.{part}_proj.weight')
                if mapped is not None:
                    out[mapped] = value
            continue
        mapped = _strip_perception_target(target)
        if mapped is not None:
            out[mapped] = tensor
    return out


def _convert_perception_state_inverse(state):
    from transformers.core_model_loading import rename_source_key

    _, _, renamings, converters = _perception_transforms()
    out, qkv, seen = {}, {}, set()
    for key, tensor in state.items():
        if key.startswith('proj.'):
            source = f'model.multi_modal_projector.{key[len("proj."):]}'
        elif key.startswith('encoder.'):
            source = f'model.audio_tower.{key[len("encoder."):]}'
        else:
            continue
        source, pattern = rename_source_key(source, renamings, converters, reverse=True)
        if not source.startswith('perception.'):
            continue
        source = source[len('perception.'):]
        seen.add(key)
        if pattern is not None:
            match = re.search(r'([qkv])_proj', pattern)
            if match is not None:
                if not source.endswith('attn.w_qkv.weight'):
                    raise ValueError(f'Unexpected qkv mapping target: {source}')
                base = source.rsplit('attn.w_qkv.weight', 1)[0]
                qkv.setdefault(base, {})[match.group(1)] = tensor
                continue
        out[source] = tensor
    for base, values in qkv.items():
        out[f'{base}attn.w_qkv.weight'] = torch.cat([values['q'], values['k'], values['v']], dim=0)
    unmapped = [key for key in state if not key.endswith('_extra_state') and key not in seen]
    if unmapped:
        raise ValueError(f'Perception keys not mapped back to checkpoint names: {unmapped[:8]}')
    return out


class _TransformersPerception(torch.nn.Module):

    def __init__(self, hf_config):
        super().__init__()
        from transformers import Nemotron3_5TranscribeAudioModel
        self.encoder = Nemotron3_5TranscribeAudioModel._from_config(hf_config.audio_config)
        self.proj = torch.nn.Linear(hf_config.audio_config.hidden_size, hf_config.text_config.hidden_size)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(_convert_perception_state(state_dict), strict=strict, assign=assign)

    def state_dict(self, *args, **kwargs):
        return _convert_perception_state_inverse(super().state_dict(*args, **kwargs))

    def get_audio_features(self, input_features, input_features_mask):
        output = self.encoder(input_features, input_features_mask=input_features_mask, return_dict=True)
        audio_embeds = self.proj(output.last_hidden_state)
        lengths = input_features_mask.sum(-1).to(torch.long)
        # Same factor as processor.subsampling_factor / the template's token estimate.
        subsampling_factor = self.encoder.config.subsampling_factor
        post_lengths = (lengths + subsampling_factor - 1) // subsampling_factor
        valid_mask = torch.arange(audio_embeds.shape[1], device=post_lengths.device)[None, :] < post_lengths[:, None]
        return audio_embeds[valid_mask]


class NemoSpeechLMVit(HuggingFaceVit):
    module_mapping = {'perception': 'perception'}
    _vision_tower = ['perception.encoder']
    _aligner = ['perception.proj']
    test_mm_type = 'audio'

    def prepare_model(self, hf_config: PretrainedConfig):
        self.perception = _TransformersPerception(hf_config)
        self.audio_token_id = hf_config.audio_token_id

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        input_ids = kwargs['input_ids']
        input_features = kwargs.get('input_features')
        input_features_mask = kwargs.get('input_features_mask')
        is_dummy = input_features is None
        if is_dummy:
            # 104 frames = 13 subsampled frames; must be a multiple of subsampling_factor.
            num_mel_bins = int(self.config.hf_config.audio_config.asr_config.num_mel_bins)
            input_features = inputs_embeds.new_zeros((1, 104, num_mel_bins), dtype=torch.float32)
            input_features_mask = input_ids.new_ones((1, 104), dtype=torch.bool)
        audio_embeds = self.perception.get_audio_features(input_features, input_features_mask)
        if is_dummy:
            return inputs_embeds + audio_embeds.mean().to(inputs_embeds.dtype) * 0.
        audio_mask = input_ids == self.audio_token_id
        expected = (int(audio_mask.sum()), inputs_embeds.shape[-1])
        if audio_embeds.shape != expected:
            raise ValueError(f'Audio embeddings {tuple(audio_embeds.shape)} do not match placeholders {expected}.')
        audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        return inputs_embeds.masked_scatter(audio_mask.unsqueeze(-1).expand_as(inputs_embeds), audio_embeds)


class NemoSpeechLMGPTModel(MultimodalGPTModel):
    language_model_cls = HybridModel

    def __init__(self, config, *args, **kwargs):
        bridge = config.bridge
        config.bridge = None
        try:
            super().__init__(config, *args, **kwargs)
        finally:
            config.bridge = bridge


class NemoSpeechLMLoader(NemotronHLoader):
    model_cls = NemoSpeechLMGPTModel

    def __init__(self, config):
        if config.context_parallel_size > 1:
            raise NotImplementedError('Nemotron Transcribe requires --context_parallel_size 1.')
        if config.mtp_num_layers and not getattr(config, 'mtp_use_repeated_layer', False):
            raise NotImplementedError('Nemotron Transcribe MTP uses one repeated physical pattern; '
                                      'use --mtp_shared_weights true.')
        if config.fp8 or getattr(config, 'fp4', None):
            raise NotImplementedError('Nemotron Transcribe FP8/FP4 training is not supported.')
        if getattr(config, 'padding_free', False) or getattr(config, 'packing', False):
            raise NotImplementedError('Nemotron Transcribe requires --padding_free false --packing false.')
        super().__init__(config)


register_model(
    ModelMeta(
        ModelType.nemo_speechlm, ['nemo_speechlm'],
        bridge_cls=NemoSpeechLMBridge,
        visual_cls=NemoSpeechLMVit,
        loader=NemoSpeechLMLoader,
        config_cls=NemoSpeechLMConfig,
        mtp_prefix='llm.mtp.layers',
        mtp_depth_field='mtp.num_nextn_predict_layers',
        mtp_repeated_field='mtp.use_repeated_layer',
        mtp_layers_per_depth=2))
