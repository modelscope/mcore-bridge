# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from megatron.core.extensions.transformer_engine import _get_extra_te_kwargs
from megatron.core.models.huggingface import HuggingFaceModule as _HuggingFaceModule
from megatron.core.tensor_parallel import (gather_from_sequence_parallel_region,
                                           reduce_scatter_to_sequence_parallel_region)
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
        _Qwen3_5MoeGatedDeltaNet.__init__(self, config, layer_number)
        self.config = config
        extra_kwargs = _get_extra_te_kwargs(config)
        self.to(dtype=extra_kwargs['params_dtype'], device=extra_kwargs['device'])

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        config = self.config
        if config.sequence_parallel and config.tensor_model_parallel_size > 1:
            hidden_states = gather_from_sequence_parallel_region(hidden_states)
        seq_len = hidden_states.shape[0]
        packed_seq_params = kwargs.get('packed_seq_params')
        thd_format = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        # Note: for packed inputs, we do not perform padding_free unpadding.
        # Doing so would allow different sequences to see each other; for efficiency we keep this implementation.
        if thd_format:
            new_hidden_states = hidden_states.new_zeros(
                (packed_seq_params.num_samples, packed_seq_params.max_seqlen_q.item(), hidden_states.shape[-1]))
            attention_mask = hidden_states.new_zeros(
                (packed_seq_params.num_samples, packed_seq_params.max_seqlen_q.item()), dtype=torch.bool)
            cu_seqlens_q = packed_seq_params.cu_seqlens_q
            for i in range(packed_seq_params.num_samples):
                start, end = cu_seqlens_q[i], cu_seqlens_q[i + 1]
                attention_mask[i, :end - start] = True
                new_hidden_states[i, :end - start] = hidden_states[start:end, 0]
            hidden_states = new_hidden_states
        else:
            hidden_states = hidden_states.transpose(0, 1)
            attention_mask = resolve_gdn_attention_mask(kwargs)
        res = super().forward(hidden_states=hidden_states, attention_mask=attention_mask)
        if thd_format:
            res = res[attention_mask][:, None]
            res = torch.concat([res, res.new_zeros(seq_len - res.shape[0], 1, res.shape[2])])
        else:
            res = res.transpose(0, 1).contiguous()
        if config.sequence_parallel and config.tensor_model_parallel_size > 1:
            # Quick fix for dropout issue, awaiting ms-swift 4.0 refactor
            res = reduce_scatter_to_sequence_parallel_region(res) / config.tensor_model_parallel_size
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
