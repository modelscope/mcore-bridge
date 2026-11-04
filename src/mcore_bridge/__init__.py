# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
from typing import TYPE_CHECKING

from .patcher import apply_patch
from .utils import _LazyModule

apply_patch()

if TYPE_CHECKING:
    from .bridge import GPTBridge
    from .config import MLAModelConfig, ModelConfig, hf_to_mcore_config
    from .model import get_mcore_model, resolve_mtp_config, resolve_mtp_num_layers
    from .tuners import LoraParallelLinear
    from .utils import get_logger, set_random_seed, split_cp_inputs
    from .version import __release_datetime__, __version__
else:
    _import_structure = {
        'bridge': ['GPTBridge'],
        'config': ['MLAModelConfig', 'ModelConfig', 'hf_to_mcore_config'],
        'model': ['get_mcore_model', 'resolve_mtp_config', 'resolve_mtp_num_layers'],
        'tuners': ['LoraParallelLinear'],
        'utils': ['get_logger', 'set_random_seed', 'split_cp_inputs'],
        'version': ['__release_datetime__', '__version__'],
    }

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
