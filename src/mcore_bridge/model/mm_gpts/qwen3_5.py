# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.nn.functional as F
from fla.modules.conv.causal_conv1d import causal_conv1d as _fla_causal_conv1d
from megatron.core.extensions.transformer_engine import _get_extra_te_kwargs
from megatron.core.models.huggingface import HuggingFaceModule as _HuggingFaceModule
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region, scatter_to_sequence_parallel_region
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig

from mcore_bridge.utils import get_env_args

from ..constant import ModelType
from ..gpts.qwen3_next import Qwen3NextBridge, Qwen3NextLoader, resolve_gdn_attention_mask
from ..register import ModelMeta, register_model
from .utils import HuggingFaceVit

try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet as _Qwen3_5MoeGatedDeltaNet
except ImportError:
    _Qwen3_5MoeGatedDeltaNet = object


class Qwen3_5MoeGatedDeltaNet(_HuggingFaceModule, _Qwen3_5MoeGatedDeltaNet):

    def __init__(self, config: TransformerConfig, submodules: SelfAttentionSubmodules, layer_number: int, **kwargs):
        assert config.context_parallel_size == 1, 'Qwen3_5 currently does not support context parallel.'
        assert _Qwen3_5MoeGatedDeltaNet is not object, 'please update the `transformers` version.'
        if getattr(config, 'layer_types', None) is None:
            freq = config.linear_attention_freq
            if isinstance(freq, int):
                freq = [0 if (i + 1) % freq == 0 else 1 for i in range(config.num_layers)]
            config.layer_types = ['linear_attention' if f else 'full_attention' for f in freq]
        _Qwen3_5MoeGatedDeltaNet.__init__(self, config, layer_number - 1)
        self.config = config
        extra_kwargs = _get_extra_te_kwargs(config)
        self.to(dtype=extra_kwargs['params_dtype'], device=extra_kwargs['device'])
        self.causal_conv1d_fn = self._fla_conv_fn
        self._cur_cu_seqlens = None

    def _fla_conv_fn(self, x, weight, bias, activation, seq_idx=None):
        x_btd = x.transpose(1, 2)
        if isinstance(activation, str):
            act = activation
        elif activation is F.silu or activation == 'silu':
            act = 'silu'
        else:
            act = None
        out, _ = _fla_causal_conv1d(x_btd, weight=weight, bias=bias, activation=act, cu_seqlens=self._cur_cu_seqlens)
        return out.transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        config = self.config
        if config.sequence_parallel and config.tensor_model_parallel_size > 1:
            hidden_states = gather_from_sequence_parallel_region(hidden_states, tensor_parallel_output_grad=False)
        packed_seq_params = kwargs.get('packed_seq_params')
        thd_format = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if thd_format:
            cu_seq_lens_q = packed_seq_params.cu_seqlens_q
            self._cur_cu_seqlens = cu_seq_lens_q
            attention_mask = None
        else:
            cu_seq_lens_q = None
            self._cur_cu_seqlens = None
            attention_mask = resolve_gdn_attention_mask(kwargs)
        hidden_states = hidden_states.transpose(0, 1)
        with get_cuda_rng_tracker().fork('data-parallel-rng'):
            res = super().forward(
                hidden_states=hidden_states, attention_mask=attention_mask, cu_seq_lens_q=cu_seq_lens_q)
        res = res.transpose(0, 1).contiguous()
        if config.sequence_parallel and config.tensor_model_parallel_size > 1:
            res = scatter_to_sequence_parallel_region(res)
        return res, None


class Qwen3_5Vit(HuggingFaceVit):
    module_mapping = {'model.visual': 'visual'}
    _vision_tower = ['visual']
    _aligner = ['visual.merger']

    def prepare_model(self, hf_config):
        from transformers.models.qwen3_5 import Qwen3_5VisionModel
        self.visual = Qwen3_5VisionModel._from_config(hf_config.vision_config)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        return self._hf_get_inputs_embeds(inputs_embeds, kwargs, self.visual, self.hf_config)


class Qwen3_5Bridge(Qwen3NextBridge):
    hf_layers_prefix = 'model.language_model.layers'
    hf_embed_key = 'model.language_model.embed_tokens.weight'
    hf_final_layernorm_key = 'model.language_model.norm.weight'


class Qwen3_5Loader(Qwen3NextLoader):
    gated_delta_net = Qwen3_5MoeGatedDeltaNet


use_mcore_gdn = get_env_args('USE_MCORE_GDN', bool, True)

if not use_mcore_gdn:
    register_model(
        ModelMeta(
            ModelType.qwen3_5,
            ['qwen3_5', 'qwen3_5_moe'],
            bridge_cls=Qwen3_5Bridge,
            visual_cls=Qwen3_5Vit,
            loader=Qwen3_5Loader,
        ))
