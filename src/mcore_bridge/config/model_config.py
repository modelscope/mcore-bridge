# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import re
import torch.nn.functional as F
from dataclasses import dataclass
from megatron.core import mpu
from megatron.core.transformer import TransformerConfig
from transformers import PretrainedConfig
from transformers.utils import is_torch_npu_available
from transformers.utils.versions import require_version
from typing import List, Literal, Optional, Union

from mcore_bridge.utils import get_logger, json_parse_to_dict

logger = get_logger()


# code borrowed from NVIDIA/Megatron-LM
def _eval_pattern(pattern):
    """ Validate and evaluate a string containing a Python list expression """
    assert isinstance(pattern, str)

    # validate input, only allow comma, digits, [, ], (, ), +, and *
    if bool(re.compile(r'[^,\d\[\]\(\)\+\*]').search(pattern)):
        raise ValueError(f'Invalid pattern: {pattern}')

    return eval(pattern)


# code borrowed from NVIDIA/Megatron-LM
def no_rope_freq_type(x):
    """ Controls which layers to skip performing Rotary Position Embedding.
    - An integer N: Represents a 1:N ratio, meaning RoPE is skipped every N-1 layers.
    - A string "N": Same as above, but provided as a string
    - A string containing a Python list expression that defines a custom pattern, e.g.:
      "([0]*3+[1]*1)*3" evaluates to [0,0,0,1,0,0,0,1,0,0,0,1]
      where 1 indicates rope is skipped on the layer.
      This allows defining arbitrary patterns of rope skipping.
      The pattern length must match the total number of transformer layers.
      Examples:
          "([1]+[0]*23)": Only first layer has rope skipped for a 24-layer network.
          "([0]*3+[1]*1)*2": Every 4 layers the rope is skipped on the last layer. Repeat twice.
    """
    if x is None or isinstance(x, int):
        return x
    assert isinstance(x, str)
    if '[' in x:
        # it's a custom pattern
        return _eval_pattern(x)
    else:
        # it's a single int but in str
        return int(x)


def linear_attn_freq_type(x):
    """Frequency between LA (linear attention) layers and SDPA (scaled dot-product attention) layers.

    Accepts either:
    - An integer N: Represents a (N-1):N ratio, meaning (N-1) LA layers for every 1 SDPA layer
    - A string "N": Same as above, but provided as a string
    - A string containing a Python list expression that defines a custom pattern, e.g.:
      "([1]*3+[0]*1)*3" evaluates to [1,1,1,0,1,1,1,0,1,1,1,0]
      where 1 indicates an LA layer and 0 indicates a SDPA layer.
      This allows defining arbitrary patterns of LA and SDPA layers.
      The pattern length must match the total number of transformer layers.
      Examples:
          "([0]+[1]*23)": 1 SDPA layer followed by 23 LA layers
          "([1]*3+[0]*2)*2": Three LA layers followed by two SDPA layers, repeated twice.
    """
    return no_rope_freq_type(x)


# code borrowed from NVIDIA/Megatron-LM
def moe_freq_type(x):
    """Frequency between MoE layers and Dense layers.

    Accepts either:
    - An integer N: Represents a 1:N ratio, meaning one expert layer for every N-1 dense layers
    - A string "N": Same as above, but provided as a string
    - A string containing a Python list expression that defines a custom pattern, e.g.:
      "([1]*3+[0]*1)*3" evaluates to [1,1,1,0,1,1,1,0,1,1,1,0]
      where 1 indicates an expert layer and 0 indicates a dense layer.
      This allows defining arbitrary patterns of expert and dense layers.
      The pattern length must match the total number of transformer layers.
      Examples:
          "([0]+[1]*23)": 1 dense layer followed by 23 experts layers
          "([1]*3+[0]*2)*2": Three expert layers followed by two dense layers, repeated twice.
    """
    if isinstance(x, int):
        return x
    assert isinstance(x, str)
    if '[' in x:
        # it's a custom pattern
        return _eval_pattern(x)
    else:
        # it's a single int but in str
        return int(x)


# code borrowed from NVIDIA/Megatron-LM
def tuple_type(x):
    """
    Convert a string to a tuple of integers.
    Examples:
        "1,2,3" -> (1, 2, 3)
        "(1,2,3)" -> (1, 2, 3)
    """
    if x is None or isinstance(x, tuple):
        return x
    assert isinstance(x, str)
    return tuple(int(i) for i in x.strip('()').split(','))


@dataclass
class ModelConfig(TransformerConfig):
    mcore_model_type: Optional[str] = None  # Inferred from hf_model_type by default
    hf_model_type: Optional[str] = None
    llm_model_type: Optional[str] = None
    padded_vocab_size: Optional[int] = None
    rope_scaling: Optional[Union[dict, str]] = None

    # model
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    ffn_hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_query_groups: Optional[int] = None
    softmax_type: Literal['vanilla', 'off-by-one', 'learnable'] = 'vanilla'
    window_size: Optional[str] = None
    window_attn_skip_freq: Optional[str] = None
    max_position_embeddings: Optional[int] = None

    position_embedding_type: Literal['learned_absolute', 'rope', 'mrope', 'none'] = 'rope'
    rotary_base: int = 10000
    rotary_percent: float = 1.
    rotary_interleaved: bool = False
    original_max_position_embeddings: Optional[int] = None
    partial_rotary_factor: Optional[float] = None
    mrope_section: Optional[List[int]] = None
    # qwen3_vl, qwen3_omni
    mrope_interleaved: bool = False

    normalization: Literal['LayerNorm', 'RMSNorm'] = 'RMSNorm'
    layernorm_epsilon: float = 1e-5
    swiglu: bool = True
    quick_geglu: bool = False
    activation_func_clamp_value: Optional[float] = None
    glu_linear_offset: float = 0.
    untie_embeddings_and_output_weights: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = True
    attention_dropout: float = 0.
    hidden_dropout: float = 0.
    kv_channels: Optional[int] = None
    qk_layernorm: bool = False
    qk_l2_norm: bool = False
    no_rope_freq: Optional[int] = None
    moe_apply_probs_on_input: Optional[bool] = None

    # moe
    num_moe_experts: Optional[int] = None
    moe_layer_freq: str = '1'
    moe_ffn_hidden_size: Optional[int] = None
    moe_shared_expert_intermediate_size: Optional[int] = None

    moe_router_topk: int = 2
    moe_router_num_groups: Optional[int] = None
    moe_router_group_topk: Optional[int] = None
    moe_router_pre_softmax: bool = False
    moe_router_score_function: Literal['sigmoid', 'softmax'] = 'softmax'
    moe_router_bias_update_rate: float = 1e-3
    moe_router_enable_expert_bias: bool = False
    moe_router_topk_scaling_factor: Optional[float] = None
    moe_router_load_balancing_type: Literal['aux_loss', 'seq_aux_loss', 'global_aux_loss', 'sinkhorn',
                                            'none'] = 'aux_loss'
    moe_shared_expert_gate: bool = False

    # mla
    multi_latent_attention: bool = False
    q_lora_rank: Optional[int] = None
    kv_lora_rank: int = 32
    qk_head_dim: int = 128
    qk_pos_emb_head_dim: int = 64
    v_head_dim: int = 128

    # qwen3_next/qwen3_5
    linear_attention_freq: Optional[str] = None
    linear_num_key_heads: Optional[int] = None
    linear_num_value_heads: Optional[int] = None
    linear_key_head_dim: Optional[int] = None
    linear_value_head_dim: Optional[int] = None
    linear_conv_kernel_dim: Optional[int] = None
    layernorm_zero_centered_gamma: bool = False
    attention_output_gate: bool = False

    # dsa
    experimental_attention_variant: Optional[Literal['gated_delta_net', 'dsa']] = None
    dsa_indexer_n_heads: Optional[int] = None
    dsa_indexer_head_dim: Optional[int] = None
    dsa_indexer_topk: Optional[int] = None
    dsa_indexer_loss_coeff: Optional[float] = None
    dsa_indexer_use_sparse_loss: bool = False
    dsa_indexer_rotary_interleaved: bool = False

    # visual
    hf_config: Optional[PretrainedConfig] = None
    vit_gradient_checkpointing: Optional[bool] = None
    vit_attn_impl: Optional[str] = None  # e.g. 'flash_attention_2'
    vit_gradient_checkpointing_kwargs: Optional[Union[dict, str]] = None

    # Override
    perform_initialization: bool = False
    apply_query_key_layer_scaling: Optional[bool] = None
    moe_router_dtype: Literal['none', 'fp32', 'fp64'] = 'fp32'
    moe_token_dispatcher_type: Literal['allgather', 'alltoall', 'flex'] = 'alltoall'
    moe_grouped_gemm: bool = True
    variable_seq_lengths: bool = True

    overlap_p2p_comm: bool = True
    persist_layer_norm: bool = True
    deallocate_pipeline_outputs: bool = True
    cp_comm_type: str = 'p2p'

    # other
    task_type: Literal['causal_lm', 'seq_cls', 'embedding', 'generative_reranker'] = 'causal_lm'
    num_labels: Optional[int] = None
    mlp_padding_free: bool = False

    _mindspeed_defaults_cache = None

    def _augment_mindspeed_defaults(self):
        if not is_torch_npu_available():
            return

        if ModelConfig._mindspeed_defaults_cache is None:
            defaults = {}
            try:
                import mindspeed.features_manager as mfm
                import sys
                from argparse import ArgumentParser
                from mindspeed.arguments import process_args

                original_features = list(mfm.FEATURES_LIST)
                full_features = mfm.create_features_list()
                mfm.FEATURES_LIST.clear()
                mfm.FEATURES_LIST.extend(full_features)
                try:
                    parser = ArgumentParser()
                    process_args(parser)
                    # Parse args from sys.argv
                    args, _ = parser.parse_known_args([])
                    defaults = vars(args)
                finally:
                    mfm.FEATURES_LIST.clear()
                    mfm.FEATURES_LIST.extend(original_features)
            except Exception as e:
                logger.warning(f'Failed to get MindSpeed defaults, which may cause issues on NPU: {e}')
                defaults = {}
            ModelConfig._mindspeed_defaults_cache = defaults

        for name, value in ModelConfig._mindspeed_defaults_cache.items():
            if not hasattr(self, name):
                setattr(self, name, value)
            elif hasattr(self, name) and getattr(self, name) is None and value is not None:
                setattr(self, name, value)

    def __post_init__(self):
        from mcore_bridge.model import get_mcore_model_type, get_model_meta
        self._augment_mindspeed_defaults()
        self._format_config()
        if self.experimental_attention_variant is not None:
            require_version('megatron-core>=0.16.0.dev',
                            'experimental attention variant requires megatron-core>=0.16.0')
        if isinstance(self.moe_router_dtype, str) and self.moe_router_dtype.lower() == 'none':
            self.moe_router_dtype = None
        if self.moe_shared_expert_intermediate_size == 0:
            self.moe_shared_expert_intermediate_size = None
        if self.num_moe_experts is not None:
            if self.moe_ffn_hidden_size is None:
                self.moe_ffn_hidden_size = self.ffn_hidden_size
        if self.rope_scaling is not None:
            self.rope_scaling = json_parse_to_dict(self.rope_scaling)
            if 'type' in self.rope_scaling and 'rope_type' not in self.rope_scaling:
                self.rope_scaling['rope_type'] = self.rope_scaling['type']

        if self.add_bias_linear:
            self.add_qkv_bias = True
        if self.swiglu:
            self.activation_func = F.silu
            self.gated_linear_unit = True
        if self.quick_geglu:
            # megatron-core>=0.14.0
            try:
                from megatron.core.fusions.fused_bias_geglu import quick_gelu
            except ImportError:
                from megatron.core.activations import quick_gelu
            assert not self.swiglu
            self.gated_linear_unit = True
            self.activation_func = quick_gelu
        if self.pipeline_dtype is None:
            self.pipeline_dtype = self.params_dtype
        if self.apply_query_key_layer_scaling is None:
            self.apply_query_key_layer_scaling = self.fp16
        if self.apply_query_key_layer_scaling:
            os.environ['NVTE_APPLY_QK_LAYER_SCALING'] = '1'
        # patch rotary_interleaved
        super().__post_init__()

        self._check_npu()
        if self.mcore_model_type is None:
            self.mcore_model_type = get_mcore_model_type(self.hf_model_type)
        self.model_meta = get_model_meta(self.mcore_model_type)
        self.is_multimodal = self.model_meta.visual_cls is not None
        if self.is_multimodal:
            if self.hf_config is None:
                raise ValueError('Multimodal model must specify hf_config.')
        self.is_moe_model = self.num_moe_experts is not None
        self.bridge = self.model_meta.bridge_cls(self)

    def _format_config(self):
        if self.window_size is not None:
            self.window_size = tuple_type(self.window_size)
        if self.window_attn_skip_freq is not None:
            self.window_attn_skip_freq = moe_freq_type(self.window_attn_skip_freq)
        if self.no_rope_freq is not None:
            self.no_rope_freq = no_rope_freq_type(self.no_rope_freq)
        if self.moe_layer_freq is not None:
            self.moe_layer_freq = moe_freq_type(self.moe_layer_freq)
        if self.linear_attention_freq is not None:
            self.linear_attention_freq = linear_attn_freq_type(self.linear_attention_freq)
            if isinstance(self.linear_attention_freq, int):
                self.linear_attention_freq = [
                    0 if ((i + 1) % self.linear_attention_freq == 0) else 1 for i in range(self.num_layers)
                ]

    def _check_npu(self):
        MAX_NPU_EXPERTS_PER_EP = 128
        num_experts = self.num_moe_experts
        expert_model_parallel_size = mpu.get_expert_model_parallel_world_size()
        if is_torch_npu_available() and num_experts and num_experts > MAX_NPU_EXPERTS_PER_EP:
            required_ep = (num_experts + MAX_NPU_EXPERTS_PER_EP - 1) // MAX_NPU_EXPERTS_PER_EP
            if expert_model_parallel_size < required_ep:
                logger.warning(f'{">" * 20} WARNING {"<" * 20}\n'
                               f'MindSpeed on NPU supports up to {MAX_NPU_EXPERTS_PER_EP} experts per EP group. '
                               f'num_experts={num_experts}, '
                               f'expert_model_parallel_size={expert_model_parallel_size}. '
                               f'Please set expert_model_parallel_size (EP) to {required_ep} '
                               f'(num_experts / {MAX_NPU_EXPERTS_PER_EP}) or higher.')
