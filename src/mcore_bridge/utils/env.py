# Copyright (c) ModelScope Contributors. All rights reserved.
# code borrowed from modelscope/ms-swift
import os
from typing import Tuple

from transformers.utils import strtobool

from .logger import get_logger

logger = get_logger()


def use_hf_hub():
    return strtobool(os.environ.get('USE_HF', '0'))


def get_dist_setting() -> Tuple[int, int, int, int]:
    """return rank, local_rank, world_size, local_world_size"""
    rank = int(os.getenv('RANK', -1))
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE') or os.getenv('_PATCH_WORLD_SIZE') or 1)
    # compat deepspeed launch
    local_world_size = int(os.getenv('LOCAL_WORLD_SIZE', None) or os.getenv('LOCAL_SIZE', 1))
    return rank, local_rank, world_size, local_world_size


def get_node_setting():
    node_rank = int(os.getenv('NODE_RANK', 0))
    nnodes = int(os.getenv('NNODES', 1))
    return node_rank, nnodes


def is_local_master():
    local_rank = get_dist_setting()[1]
    return local_rank in {-1, 0}


def is_master():
    rank = get_dist_setting()[0]
    return rank in {-1, 0}


def is_last_rank():
    rank, _, world_size, _ = get_dist_setting()
    return rank in {-1, world_size - 1}


def is_dist():
    """Determine if the training is distributed"""
    rank, local_rank, _, _ = get_dist_setting()
    return rank >= 0 and local_rank >= 0
