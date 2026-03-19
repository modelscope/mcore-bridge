# Copyright (c) ModelScope Contributors. All rights reserved.
from transformers import PreTrainedConfig

from mcore_bridge.bridge import MultimodalGPTBridge

from ..constant import ModelType
from ..gpts.glm4 import Glm4Bridge, Glm4Loader
from ..register import ModelMeta, register_model
from .utils import HuggingFaceVit


class Glm4vVit(HuggingFaceVit):
    module_mapping = {'model.visual': 'visual'}
    _vision_tower = ['visual']
    _aligner = ['visual.merger']

    def prepare_model(self, hf_config: PreTrainedConfig):
        pass

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        return self._hf_get_inputs_embeds(inputs_embeds, kwargs, self.visual, self.processor, self.hf_config)


register_model(ModelMeta(
    ModelType.glm4v_moe,
    ['glm4v_moe'],
    bridge_cls=MultimodalGPTBridge,
    visual_cls=Glm4vVit,
))


class Glm4vBridge(Glm4Bridge, MultimodalGPTBridge):
    pass


register_model(ModelMeta(
    ModelType.glm4v,
    ['glm4v'],
    bridge_cls=Glm4vBridge,
    visual_cls=Glm4vVit,
    loader=Glm4Loader,
))
