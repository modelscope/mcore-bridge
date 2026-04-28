# Copyright (c) ModelScope Contributors. All rights reserved.
from transformers import PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from mcore_bridge.bridge import GPTBridge

from ..constant import ModelType
from ..register import ModelMeta, register_model
from .utils import HuggingFaceVit


class LLavaOneVision1_5Vit(HuggingFaceVit):
    module_mapping = {'visual': 'visual'}
    _vision_tower = ['visual']
    _aligner = ['visual.merger']

    def prepare_model(self, hf_config: PretrainedConfig):
        VisualModel = get_class_from_dynamic_module(
            'modeling_llavaonevision1_5.RiceTransformerPretrainedModel',
            hf_config.name_or_path)
        self.visual = VisualModel._from_config(hf_config.vision_config)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        return self._hf_get_inputs_embeds(inputs_embeds, kwargs, self.visual, self.hf_config)


register_model(
    ModelMeta(
        ModelType.llava_onevision1_5,
        ['llava_onevision1_5'],
        bridge_cls=GPTBridge,
        visual_cls=LLavaOneVision1_5Vit,
    ))
