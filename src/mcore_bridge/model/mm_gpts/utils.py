# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from abc import ABC, abstractmethod
from contextlib import contextmanager
from megatron.core.models.huggingface import HuggingFaceModule as _HuggingFaceModule
from transformers import PreTrainedModel
from transformers.utils import ContextManagers

from mcore_bridge.config import ModelConfig
from mcore_bridge.utils import deep_getattr


@contextmanager
def patch_hf_initialize_weight():

    _origin_initialize_weight = PreTrainedModel._initialize_weights

    def _initialize_weight(self, *args, **kwargs):
        return

    PreTrainedModel._initialize_weights = _initialize_weight
    try:
        yield
    finally:
        PreTrainedModel._initialize_weights = _origin_initialize_weight


@contextmanager
def patch_get_dynamic_module():
    origin_get_cached_module_file = dynamic_module_utils.get_cached_module_file

    def new_get_cached_module_file(pretrained_model_name_or_path, *args, **kwargs):
        with safe_ddp_context(hash_id=str(pretrained_model_name_or_path)):
            return origin_get_cached_module_file(pretrained_model_name_or_path, *args, **kwargs)

    dynamic_module_utils.get_cached_module_file = new_get_cached_module_file
    try:
        yield
    finally:
        dynamic_module_utils.get_cached_module_file = origin_get_cached_module_file


@contextmanager
def patch_device_map_meta(model_cls):
    __origin_init__ = model_cls.__init__

    def __init__(self, *args, **kwargs):
        with torch.device('meta'):
            __origin_init__(self, *args, **kwargs)

    model_cls.__init__ = __init__

    try:
        yield
    finally:
        model_cls.__init__ = __origin_init__


class HuggingFaceVit(_HuggingFaceModule, ABC):
    module_mapping = {}  # hf -> mcore

    def __init__(self, config: ModelConfig, ignore_init_model_cls=None):
        super().__init__(config)
        attn_impl = config.vit_attn_impl or 'flash_attention_2'
        config.hf_config.torch_dtype = config.params_dtype
        if config.attention_backend.name == 'flash':
            config.hf_config._attn_implementation = attn_impl
        from transformers.models.qwen3_5 import Qwen3_5VisionModel
        hf_config = config.hf_config
        self.visual = Qwen3_5VisionModel._from_config(hf_config.vision_config)
        self.to('cuda')

    def prepare_model(self, hf_model):
        pass

    # @abstractmethod
    # def get_inputs_embeds(self, inputs_embeds, **kwargs):
    #     pass
