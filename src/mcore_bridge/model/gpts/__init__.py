# Copyright (c) ModelScope Contributors. All rights reserved.
from importlib.util import find_spec

from . import (bailing_hybrid, bailing_moe, deepseek_v4, glm4, glm_moe_dsa, hunyuan, llm, minimax_m2, olmoe, qwen3_emb,
               qwen3_next)

if find_spec('megatron.core.models.hybrid') is not None:
    from . import nemotron_h
