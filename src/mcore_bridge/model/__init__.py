# Copyright (c) ModelScope Contributors. All rights reserved.
from . import gpts, mm_gpts
from .constant import ModelType
from .gpt_model import GPTModel
from .mm_gpt_model import MultimodalGPTModel
from .mtp_detect import resolve_mtp_config, resolve_mtp_num_layers
from .register import MODEL_MAPPING, get_mcore_model, get_mcore_model_type, get_model_meta
