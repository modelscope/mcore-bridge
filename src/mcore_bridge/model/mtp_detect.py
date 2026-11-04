# Copyright (c) ModelScope Contributors. All rights reserved.
import glob
import json
import os
import re
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from mcore_bridge.utils import get_logger

if TYPE_CHECKING:
    from .register import ModelMeta

logger = get_logger()


def _cfg_get(config, key: str):
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def _cfg_get_path(config, path: str):
    for key in path.split('.'):
        config = _cfg_get(config, key)
        if config is None:
            return None
    return config


def _get_declared_depth(hf_config, field: Optional[str]) -> Optional[int]:
    if not field:
        return None
    for config in (hf_config, _cfg_get(hf_config, 'text_config')):
        value = _cfg_get_path(config, field)
        if value is None:
            continue
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _get_declared_bool(hf_config, field: Optional[str]) -> bool:
    return bool(_cfg_get_path(hf_config, field)) if field else False


def _read_weight_keys(model_dir: Optional[str]) -> Optional[List[str]]:
    if not model_dir or not os.path.isdir(model_dir):
        return None
    for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        index_path = os.path.join(model_dir, name)
        if os.path.isfile(index_path):
            with open(index_path, encoding='utf-8') as f:
                weight_map = json.load(f).get('weight_map', {})
            return list(weight_map)
    safetensor_files = sorted(glob.glob(os.path.join(model_dir, '*.safetensors')))
    if safetensor_files:
        from safetensors import safe_open
        keys = set()
        for path in safetensor_files:
            with safe_open(path, framework='pt') as f:
                keys.update(f.keys())
        return sorted(keys)
    bin_files = sorted(glob.glob(os.path.join(model_dir, '*.bin')))
    if bin_files:
        import torch
        keys = set()
        for path in bin_files:
            state_dict = torch.load(path, map_location='meta', mmap=True, weights_only=True)
            keys.update(state_dict.keys())
        return sorted(keys)
    return None


def _count_standard_sets(keys: List[str], num_layers: int) -> int:
    # Text-only checkpoints use `model.layers.*`, multimodal ones use
    # `model.language_model.layers.*`; both describe the same decoder stack.
    pattern = re.compile(r'^(?:model\.)?(?:language_model\.)?layers\.(\d+)\.')
    indices = set()
    for key in keys:
        match = pattern.match(key)
        if match and int(match.group(1)) >= num_layers:
            indices.add(int(match.group(1)))
    return len(indices)


def _count_prefixed_sets(keys: List[str], prefix: str) -> int:
    prefix = prefix.strip('.')
    pattern = re.compile(rf'^{re.escape(prefix)}\.(?:layers\.)?(\d+)\.')
    indices = {int(match.group(1)) for key in keys if (match := pattern.match(key))}
    if indices:
        return len(indices)

    # Legacy multimodal checkpoints may wrap an otherwise bare prefix.
    name = prefix.split('.')[-1]
    wrapper = r'(?:model\.)?(?:language_model\.)?'
    pattern = re.compile(rf'^{wrapper}{re.escape(name)}\.(?:layers\.)?(\d+)\.')
    indices = {int(match.group(1)) for key in keys if (match := pattern.match(key))}
    if indices:
        return len(indices)
    top_level = re.compile(rf'^(?:{re.escape(prefix)}\.|{wrapper}{re.escape(name)}\.)')
    return 1 if any(top_level.match(key) for key in keys) else 0


def _count_mtp_sets(keys: List[str], prefix: str, hf_config) -> int:
    if prefix in ('model.layers', 'layers'):
        text_config = _cfg_get(hf_config, 'text_config')
        num_layers = _cfg_get(text_config, 'num_hidden_layers') or _cfg_get(hf_config, 'num_hidden_layers')
        if num_layers is None:
            return 0
        return _count_standard_sets(keys, int(num_layers))
    return _count_prefixed_sets(keys, prefix)


def resolve_mtp_num_layers(model_meta: 'ModelMeta',
                           hf_config,
                           model_dir: Optional[str],
                           *,
                           warn: bool = True) -> Optional[int]:
    """Resolve the MTP depth stored in a checkpoint.

    The config's declared depth is only a hint: some checkpoints declare MTP but omit
    the weights (for example MiniCPM-V), so the weight keys decide whether MTP is
    actually available.
    """
    prefix = model_meta.mtp_prefix
    if not prefix:
        return None
    declared = _get_declared_depth(hf_config, model_meta.mtp_depth_field)
    keys = _read_weight_keys(model_dir)
    if keys is None:
        if warn:
            logger.warning(f'Could not inspect MTP weights in {model_dir!r}; MTP is disabled. '
                           'Pass `--mtp_num_layers` explicitly if you want to train it.')
        return None
    detected_modules = _count_mtp_sets(keys, prefix, hf_config)
    if detected_modules == 0:
        if declared and warn:
            logger.warning(f'The model config declares {declared} MTP layer(s), but no MTP weights were found '
                           f'under {prefix!r}; MTP is disabled for this checkpoint.')
        return None

    layers_per_depth = int(model_meta.mtp_layers_per_depth or 1)
    if detected_modules % layers_per_depth:
        raise ValueError(f'MTP weights under {prefix!r} contain {detected_modules} module(s), which is not divisible '
                         f'by mtp_layers_per_depth={layers_per_depth}.')
    physical_depths = detected_modules // layers_per_depth
    repeated = _get_declared_bool(hf_config, model_meta.mtp_repeated_field)
    if repeated:
        if physical_depths != 1:
            raise ValueError(f'{model_meta.mtp_repeated_field} is true, but the checkpoint stores '
                             f'{physical_depths} physical MTP depth(s) under {prefix!r}.')
        detected = declared or 1
        if declared and warn:
            logger.info(f'MTP uses one repeated physical pattern {prefix!r} for {declared} logical depth(s).')
        return detected

    detected = physical_depths
    if declared and declared != detected and warn:
        logger.warning(f'The model config declares {declared} MTP layer(s), but the checkpoint stores '
                       f'{detected} physical set(s) under {prefix!r}; using {detected}.')
    return detected


def resolve_mtp_config(model_meta: 'ModelMeta', hf_config, model_dir: Optional[str],
                       requested_num_layers: Optional[Union[int, str]],
                       requested_shared_weights: Optional[bool]) -> Tuple[Optional[int], bool]:
    """Resolve MTP depth, automatically reusing weights only from a single physical set."""
    physical = resolve_mtp_num_layers(model_meta, hf_config, model_dir, warn=requested_num_layers == 'auto')
    if requested_num_layers == 'auto':
        num_layers = physical
    elif requested_num_layers is None:
        num_layers = None
    else:
        num_layers = int(requested_num_layers)
    if requested_shared_weights is None:
        repeated = _get_declared_bool(hf_config, model_meta.mtp_repeated_field)
        shared_weights = bool(repeated or physical == 1 and num_layers and num_layers > 1)
    else:
        shared_weights = requested_shared_weights
    if requested_num_layers == 'auto' or requested_shared_weights is None:
        logger.info(f'MTP configuration: mtp_num_layers={num_layers}, mtp_shared_weights={shared_weights}')
    return num_layers, shared_weights
