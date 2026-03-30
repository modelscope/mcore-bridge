# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
from typing import TYPE_CHECKING

from .patcher import apply_patch
from .utils import _LazyModule

apply_patch()

if TYPE_CHECKING:
    from .bridge import GPTBridge
    from .config import ModelConfig, hf_to_mcore_config
    from .model import get_mcore_model
    from .tuners import LoraParallelLinear
    from .utils import get_logger, set_random_seed
    from .version import __release_datetime__, __version__
else:
    _import_structure = {
        'bridge': ['GPTBridge'],
        'config': ['ModelConfig', 'hf_to_mcore_config'],
        'model': ['get_mcore_model'],
        'tuners': ['LoraParallelLinear'],
        'utils': ['get_logger', 'set_random_seed'],
        'version': ['__release_datetime__', '__version__'],
    }

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
