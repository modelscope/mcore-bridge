# Copyright (c) ModelScope Contributors. All rights reserved.
"""Nemotron-H-Omni: replicated RADIO/projector with a Nemotron-H language backbone."""
import os
import re
import torch
from peft.utils import ModulesToSaveWrapper
from transformers import PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.utils.versions import require_version
from types import SimpleNamespace

from mcore_bridge.utils import get_logger

from ..constant import ModelType
from ..gpts.nemotron_h import NemotronHBridge, NemotronHLoader
from ..hybrid_model import HybridModel
from ..mm_gpt_model import MultimodalGPTModel
from ..register import ModelMeta, register_model
from .utils import HuggingFaceVit

_MODELING_MODULE = 'modeling_nemotron_h_omni'
_PACKED_EVS_ERROR = ('EVS requires --padding_free false --packing false, or disabling it by unsetting '
                     'USE_NEMOTRON_EVS; packed sample boundaries are not rebuilt.')

logger = get_logger()


def _evs_env_enabled() -> bool:
    """Return whether EVS video pruning is enabled for training (default: disabled)."""
    return os.environ.get('USE_NEMOTRON_EVS', '').lower() in ('1', 'true', 'on')


def _radio_checkpoint_to_module(radio_model, checkpoint_sd):
    """Apply the official RADIO rename/fused-QKV conversion (inverse of HF export)."""
    from copy import deepcopy
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import WeightConverter, WeightRenaming, dot_natural_key, rename_source_key
    conversions = get_model_conversion_mapping(radio_model, add_legacy=True)
    renamings = [c for c in conversions if isinstance(c, WeightRenaming)]
    converters = [c for c in conversions if isinstance(c, WeightConverter)]
    pattern_to_converter = {k: c for c in converters for k in c.source_patterns}
    mapping = {}
    for key, tensor in sorted(checkpoint_sd.items(), key=lambda kv: dot_natural_key(kv[0])):
        renamed, source_pattern = rename_source_key(key, renamings, converters, reverse=False)
        if source_pattern is None:
            mapping.setdefault(renamed, WeightRenaming(key, renamed))
            source_pattern = key
        else:
            mapping.setdefault(renamed, deepcopy(pattern_to_converter[source_pattern]))
        mapping[renamed].add_tensor(renamed, key, source_pattern, tensor)
    module_sd = {}
    for name, converter in mapping.items():
        realized = converter.convert(name, model=radio_model, config=radio_model.config)
        for target, value in realized.items():
            module_sd[target] = value[0] if isinstance(value, list) else value
    return module_sd


def _radio_module_to_checkpoint(radio_model, module_sd):
    from transformers.core_model_loading import revert_weight_conversion
    return revert_weight_conversion(radio_model, module_sd)


_PROJECTOR_TO_MODULE = {'0': 'norm', '1': 'linear1', '3': 'linear2'}
_PROJECTOR_TO_CHECKPOINT = {v: k for k, v in _PROJECTOR_TO_MODULE.items()}


def _rename_first_segment(state_dict, table):
    renamed = {}
    for key, value in state_dict.items():
        head, sep, rest = key.partition('.')
        renamed[f'{table[head]}{sep}{rest}' if head in table else key] = value
    return renamed


class NemotronH_OmniBridge(NemotronHBridge):
    hf_embed_key = 'language_model.backbone.embeddings.weight'
    hf_layers_prefix = 'language_model.backbone.layers'
    hf_final_layernorm_key = 'language_model.backbone.norm_f.weight'
    hf_lm_head_key = 'language_model.lm_head.weight'
    hf_mtp_prefix = 'language_model.mtp.layers'

    _RADIO_PREFIX = 'vision_model.'
    _PROJECTOR_PREFIX = 'mlp1.'
    _VISION_PREFIXES = (_RADIO_PREFIX, _PROJECTOR_PREFIX, 'vision_projector.vision_final_layernorm.')

    def _set_module(self, mg_module, hf_state_dict, hf_prefix, to_mcore):
        if hf_prefix not in self._VISION_PREFIXES:
            return super()._set_module(mg_module, hf_state_dict, hf_prefix, to_mcore)
        if self._peft_format:
            # Language-only adapters must still skip the frozen vision subtree. A visual HF
            # adapter needs its own fused-QKV adapter mapping; full/merged export is supported.
            if to_mcore:
                has_adapter = bool(self._remove_prefix(hf_state_dict, hf_prefix))
            else:
                has_adapter = mg_module is not None and any('lora_' in k or 'modules_to_save.' in k
                                                            for k in mg_module.state_dict())
            if self._reduce_tensor_pp_group(has_adapter, to_mcore):
                raise NotImplementedError('HF visual LoRA/modules_to_save adapters are not supported for Omni; '
                                          'use a full or merged checkpoint, or language-only LoRA targets.')
            return {}
        if hf_prefix not in (self._RADIO_PREFIX, self._PROJECTOR_PREFIX):
            return super()._set_module(mg_module, hf_state_dict, hf_prefix, to_mcore)
        is_radio = hf_prefix == self._RADIO_PREFIX
        if to_mcore:
            if mg_module is None:
                return {}
            raw = {k: v.load() for k, v in self._remove_prefix(hf_state_dict, hf_prefix).items()}
            state = (
                _radio_checkpoint_to_module(mg_module, raw) if is_radio else _rename_first_segment(
                    raw, _PROJECTOR_TO_MODULE))
            incompatible = mg_module.load_state_dict(state, strict=False)
            # RADIO's identity layer scales are legitimately absent from released checkpoints.
            patterns = getattr(type(mg_module), '_keys_to_ignore_on_load_missing', None) or []
            missing = [k for k in incompatible.missing_keys if not any(re.search(p, k) for p in patterns)]
            assert not missing, f'{hf_prefix} missing_keys: {missing}'
            return {}
        export_module = None
        if mg_module is not None:
            # Match GPTBridge's full export wrapper handling BEFORE RADIO's fused conversion.
            state = {}
            for key, value in mg_module.state_dict().items():
                if 'lora_A.' in key or 'lora_B.' in key or 'original_module.' in key:
                    continue
                if 'modules_to_save.' in key and f'modules_to_save.{self._adapter_name}.' not in key:
                    continue
                key = key.replace('base_layer.', '').replace(f'modules_to_save.{self._adapter_name}.', '')
                state[key] = value
            # Conversion discovery must see the canonical RADIO root, not the PEFT copy's prefix.
            conversion_module = (
                mg_module.modules_to_save[self._adapter_name]
                if isinstance(mg_module, ModulesToSaveWrapper) else mg_module)
            state = (
                _radio_module_to_checkpoint(conversion_module, state) if is_radio else _rename_first_segment(
                    state, _PROJECTOR_TO_CHECKPOINT))
            export_module = SimpleNamespace(state_dict=lambda: state)
        # Delegate device placement AND PP broadcasts, including ranks without a visual module.
        return super()._set_module(export_module, hf_state_dict, hf_prefix, to_mcore)


def _media_masks(input_ids, image_token_mask, video_token_mask, token_id):
    masks = []
    for name, mask in [('image_token_mask', image_token_mask), ('video_token_mask', video_token_mask)]:
        if mask is None:
            mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if mask.dtype != torch.bool or mask.shape != input_ids.shape:
            raise ValueError(f'{name} must be bool and have input_ids.shape')
        masks.append(mask.to(input_ids.device))
    image_mask, video_mask = masks
    if (image_mask & video_mask).any() or not torch.equal(image_mask | video_mask, input_ids == token_id):
        raise ValueError('Independent image/video masks must partition all visual placeholder tokens exactly.')
    return image_mask, video_mask


def _compress_evs_inputs(input_ids, position_ids, attention_mask, labels, keep, image_mask, video_mask, multiple):
    """Compress samples, then map already-shifted labels to the next retained token."""
    if attention_mask is None:
        valid = torch.ones_like(keep)
    elif attention_mask.dim() == 2:
        valid = attention_mask.bool()
    elif attention_mask.dim() == 4 and attention_mask.dtype == torch.bool:
        valid = ~attention_mask.all(dim=(1, 2))
        valid = valid.expand_as(keep)
    else:
        raise ValueError('EVS requires a 2D valid-token mask or 4D boolean Megatron attention mask.')
    if ((image_mask | video_mask) & ~valid).any():
        raise ValueError('Visual token masks must not include padding positions.')
    indices = [(row & row_valid).nonzero(as_tuple=True)[0] for row, row_valid in zip(keep, valid)]
    max_length = max(1, max(index.numel() for index in indices))
    length = (max_length + multiple - 1) // multiple * multiple
    shape = (input_ids.shape[0], length)
    new_ids = input_ids.new_zeros(shape)
    new_pos = position_ids.new_zeros(shape)
    new_image = image_mask.new_zeros(shape)
    new_video = video_mask.new_zeros(shape)
    new_labels = None if labels is None else labels.new_full(shape, -100)
    loss_indices = None if labels is None else input_ids.new_full(shape, -1)
    heads = attention_mask.shape[1] if attention_mask is not None and attention_mask.dim() == 4 else 1
    new_attention = torch.ones((shape[0], heads, length, length), dtype=torch.bool, device=input_ids.device)
    for batch, index in enumerate(indices):
        n = index.numel()
        new_ids[batch, :n] = input_ids[batch, index]
        new_pos[batch, :n] = position_ids[batch, index]
        new_image[batch, :n] = image_mask[batch, index]
        new_video[batch, :n] = video_mask[batch, index]
        if labels is not None and n > 1:
            # Labels are already shifted; each position targets the next retained token.
            target_indices = index[1:] - 1
            new_labels[batch, :n - 1] = labels[batch, target_indices]
            loss_indices[batch, :n - 1] = target_indices
        if attention_mask is not None and attention_mask.dim() == 4:
            source = attention_mask[0 if attention_mask.shape[0] == 1 else batch]
            new_attention[batch, :, :n, :n] = source[:, index][:, :, index]
        else:
            new_attention[batch, :, :n, :n] = torch.ones((n, n), dtype=torch.bool, device=input_ids.device).triu(1)
    return new_ids, new_pos, new_attention, new_labels, new_image, new_video, loss_indices


class NemotronH_OmniVit(HuggingFaceVit):
    module_mapping = {
        'vision_model': 'vision_model',
        'mlp1': 'vision_projector.mlp1',
        'vision_projector.vision_final_layernorm': 'vision_projector.vision_final_layernorm',
    }
    _vision_tower = ['vision_model']
    _aligner = ['vision_projector']

    def prepare_model(self, hf_config: PretrainedConfig):
        RadioModel = get_class_from_dynamic_module('modeling_radio.RadioModel', hf_config.name_or_path)
        VisionProjector = get_class_from_dynamic_module(
            f'{_MODELING_MODULE}.NemotronH_Omni_Reasoning_V3VisionProjector', hf_config.name_or_path)
        self.compute_retention_mask = get_class_from_dynamic_module(f'{_MODELING_MODULE}.compute_retention_mask',
                                                                    hf_config.name_or_path)
        self.vision_model = RadioModel._from_config(hf_config.vision_config)
        self.vision_model.make_preprocessor_external()
        self.vision_projector = VisionProjector(hf_config).to(self.vision_model.dtype)
        self.img_context_token_id = hf_config.img_context_token_id
        self.video_pruning_rate = hf_config.video_pruning_rate
        self.patch_size = hf_config.vision_config.patch_size

    def encode_media(self, pixel_values, pixel_values_videos, prune=False):
        image_embeds, video_embeds, retentions = [], [], []
        for image in pixel_values or []:
            embeds = self.vision_projector(image, self.vision_model)
            image_embeds.append(embeds.flatten(0, 1))
        for clip in pixel_values_videos or []:
            embeds = self.vision_projector.forward_video(clip, self.vision_model)
            flat = embeds.flatten(0, 1)
            if prune:
                h, w = clip.shape[-2:]
                ratio = self.vision_projector.downsample_ratio
                h, w = int((h // self.patch_size) * ratio), int((w // self.patch_size) * ratio)
                if self.vision_projector.ps_version == 'v1':
                    h, w = w, h
                if h * w != embeds.shape[1]:
                    raise ValueError('Projected video token count disagrees with its pixel H/W grid.')
                retention = self.compute_retention_mask(
                    video_embeds=embeds, thw=(embeds.shape[0], h, w), spatial_merge_size=1,
                    q=self.video_pruning_rate).reshape(-1).to(
                        device=flat.device, dtype=torch.bool)
                retentions.append(retention)
                flat = flat[retention]
            video_embeds.append(flat)
        return (torch.cat(image_embeds) if image_embeds else None, torch.cat(video_embeds) if video_embeds else None,
                torch.cat(retentions) if retentions else None)

    @staticmethod
    def _scatter(inputs_embeds, mask, embeds, name):
        count = int(mask.sum())
        if embeds is None:
            if count:
                raise ValueError(f'{name} mask has {count} tokens but no features.')
            return inputs_embeds
        if embeds.ndim != 2 or embeds.shape != (count, inputs_embeds.shape[-1]):
            raise ValueError(f'{name} features {tuple(embeds.shape)} do not match {count} placeholder tokens.')
        return inputs_embeds.masked_scatter(
            mask.to(inputs_embeds.device).unsqueeze(-1), embeds.to(inputs_embeds.device, inputs_embeds.dtype))

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        image_mask, video_mask = _media_masks(kwargs['input_ids'], kwargs.get('image_token_mask'),
                                              kwargs.get('video_token_mask'), self.img_context_token_id)
        precomputed = kwargs.get('omni_embeddings')
        if precomputed is None:
            image_embeds, video_embeds, _ = self.encode_media(
                kwargs.get('pixel_values'), kwargs.get('pixel_values_videos'))
        else:
            image_embeds, video_embeds = precomputed
        inputs_embeds = self._scatter(inputs_embeds, image_mask, image_embeds, 'image')
        inputs_embeds = self._scatter(inputs_embeds, video_mask, video_embeds, 'video')
        if image_embeds is None and video_embeds is None:
            # HuggingFaceVit contract: text-only batches keep the vision graph with zero gradient.
            side = self.patch_size * int(1 / self.vision_projector.downsample_ratio)
            dummy = torch.zeros((1, 3, side, side), dtype=self.vision_model.dtype, device=inputs_embeds.device)
            embeds = self.vision_projector(dummy, self.vision_model)
            inputs_embeds = inputs_embeds + embeds.mean().to(inputs_embeds.device) * 0.
        return inputs_embeds


class NemotronH_OmniGPTModel(MultimodalGPTModel):
    language_model_cls = HybridModel

    def forward(self,
                input_ids,
                position_ids,
                attention_mask=None,
                decoder_input=None,
                labels=None,
                inference_params=None,
                packed_seq_params=None,
                **kwargs):
        if any(kwargs.get(key) is not None for key in ('sound_clips', 'input_features', 'audio_values')):
            raise NotImplementedError('Omni audio inputs are not supported; use image/video/text inputs only.')
        kwargs.pop('loss_scale', None)
        compressed = False
        original_labels = labels
        loss_indices = None
        rate = self.config.hf_config.video_pruning_rate
        evs = rate > 0 and _evs_env_enabled()
        if rate > 0 and not evs and kwargs.get('pixel_values_videos'):
            logger.info_once(f'Videos are trained without EVS (video_pruning_rate={rate}); '
                             'set USE_NEMOTRON_EVS=1 to enable video token pruning.')
        if evs and kwargs.get('pixel_values_videos') and (inference_params is not None
                                                          or kwargs.get('inference_context') is not None):
            # Compression/right padding does not update cache offsets or last-token selection.
            raise NotImplementedError('Omni EVS cached inference is not supported; unset USE_NEMOTRON_EVS '
                                      '(or set video_pruning_rate=0), or use a full forward without '
                                      'inference_params/inference_context.')
        if evs and packed_seq_params is not None:
            raise NotImplementedError(_PACKED_EVS_ERROR)
        if evs and kwargs.get('pixel_values_videos') and not self.config.language_model_only:
            if self.config.pipeline_model_parallel_size > 1:
                raise NotImplementedError('EVS requires --pipeline_model_parallel_size 1.')
            if decoder_input is not None:
                raise ValueError('EVS requires input_ids, not precomputed decoder_input.')
            image_mask, video_mask = _media_masks(input_ids, kwargs.get('image_token_mask'),
                                                  kwargs.get('video_token_mask'), self.visual.img_context_token_id)
            image_embeds, video_embeds, retention = self.visual.encode_media(
                kwargs.get('pixel_values'), kwargs['pixel_values_videos'], prune=True)
            if retention.numel() != int(video_mask.sum()):
                raise ValueError('Video features and video_token_mask have different token counts.')
            kwargs['omni_embeddings'] = (image_embeds, video_embeds)
            if not retention.all():
                keep = torch.ones_like(input_ids, dtype=torch.bool)
                keep[video_mask] = retention.to(input_ids.device)
                (input_ids, position_ids, attention_mask, labels, image_mask, video_mask,
                 loss_indices) = _compress_evs_inputs(input_ids, position_ids, attention_mask, labels, keep, image_mask,
                                                      video_mask, self.config.tensor_model_parallel_size)
                compressed = True
                kwargs['image_token_mask'], kwargs['video_token_mask'] = image_mask, video_mask
                kwargs.pop('loss_mask', None)
                # Any caller-provided padding mask describes the old sequence.
                kwargs.pop('padding_mask', None)
        output = super().forward(input_ids, position_ids, attention_mask, decoder_input, labels, inference_params,
                                 packed_seq_params, **kwargs)
        if original_labels is not None and compressed:
            valid = loss_indices >= 0
            restored = output.new_zeros(original_labels.shape)
            return restored.scatter_add(1, loss_indices.clamp_min(0), output * valid)
        return output


class NemotronH_OmniLoader(NemotronHLoader):
    model_cls = NemotronH_OmniGPTModel

    def __init__(self, config):
        require_version('transformers>=5.16.1',
                        'Nemotron-H-Omni RADIO weight conversion requires transformers>=5.16.1.')
        if config.context_parallel_size > 1:
            raise NotImplementedError('Omni requires --context_parallel_size 1.')
        if config.mtp_num_layers:
            raise NotImplementedError('Omni MTP is not supported; set --mtp_num_layers 0 (do not alter HF llm_config).')
        if config.fp8:
            raise NotImplementedError('Omni FP8 is not supported; disable --fp8.')
        if config.hf_config.sound_config is not None:
            raise NotImplementedError(
                'Omni audio models (sound_config) are not supported; use a vision/text checkpoint.')
        rate = config.hf_config.video_pruning_rate
        if not 0 <= rate <= 1:
            raise ValueError('video_pruning_rate must be in [0, 1].')
        if rate > 0:
            if _evs_env_enabled():
                # Loader runs on every PP stage, before constructing any tower/decoder.
                if config.pipeline_model_parallel_size > 1:
                    raise NotImplementedError('EVS requires --pipeline_model_parallel_size 1, or disabling it '
                                              'by unsetting USE_NEMOTRON_EVS.')
                if getattr(config, 'padding_free', False) or getattr(config, 'packing', False):
                    raise NotImplementedError(_PACKED_EVS_ERROR)
            else:
                logger.info_once(f'Nemotron EVS is off by default (video_pruning_rate={rate}); training runs '
                                 'without video token pruning. Set USE_NEMOTRON_EVS=1 to enable it.')
        super().__init__(config)


register_model(
    ModelMeta(
        ModelType.nemotron_h_omni, ['nemotron_h_omni'],
        bridge_cls=NemotronH_OmniBridge,
        visual_cls=NemotronH_OmniVit,
        loader=NemotronH_OmniLoader))
