import copy
import json
import os
from contextlib import contextmanager
from torch import nn
from transformers.utils import strtobool
from typing import Callable, Dict, Optional, TypeVar, Union

from .logger import get_logger

logger = get_logger()


# code borrowed from modelscope/ms-swift
def json_parse_to_dict(value: Union[str, Dict, None], strict: bool = True) -> Union[str, Dict]:
    """Convert a JSON string or JSON file into a dict"""
    # If the value could potentially be a string, it is generally advisable to set strict to False.
    if value is None:
        value = {}
    elif isinstance(value, str):
        if os.path.exists(value):  # local path
            with open(value, 'r', encoding='utf-8') as f:
                value = json.load(f)
        else:  # json str
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                if strict:
                    logger.error(f"Unable to parse json string: '{value}'.")
                    raise
    return value


_T = TypeVar('_T')


# code borrowed from modelscope/ms-swift
def get_env_args(args_name: str, type_func: Callable[[str], _T], default_value: Optional[_T]) -> Optional[_T]:
    args_name_upper = args_name.upper()
    value = os.getenv(args_name_upper)
    if value is None:
        value = default_value
        log_info = (f'Setting {args_name}: {default_value}. '
                    f'You can adjust this hyperparameter through the environment variable: `{args_name_upper}`.')
    else:
        if type_func is bool:
            value = strtobool(value)
        value = type_func(value)
        log_info = f'Using environment variable `{args_name_upper}`, Setting {args_name}: {value}.'
    logger.info_once(log_info)
    return value


def deep_getattr(obj, attr: str, default=None):
    attrs = attr.split('.')
    for a in attrs:
        if obj is None:
            break
        if isinstance(obj, dict):
            obj = obj.get(a, default)
        else:
            obj = getattr(obj, a, default)
    return obj


@contextmanager
def patch_deepcopy():
    _origin_deepcopy = copy.deepcopy
    copy_keys = ('tp_group', '_tp_group', 'config')

    def new_deepcopy(x, *args, **kwargs):
        if not isinstance(x, nn.Module):
            return _origin_deepcopy(x, *args, **kwargs)

        saved_values = {}
        for key in copy_keys:
            val = getattr(x, key, None)
            if val is not None:
                saved_values[key] = val
                setattr(x, key, None)

        try:
            res = _origin_deepcopy(x, *args, **kwargs)
        finally:
            for key, value in saved_values.items():
                setattr(x, key, value)

        for key, value in saved_values.items():
            setattr(res, key, value)
        return res

    copy.deepcopy = new_deepcopy
    try:
        yield
    finally:
        copy.deepcopy = _origin_deepcopy
