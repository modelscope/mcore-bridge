# Copyright (c) ModelScope Contributors. All rights reserved.
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, apply_swiglu_sharded_factory
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.utils import sharded_state_dict_default
from typing import Optional

from mcore_bridge.config import ModelConfig

from ..constant import ModelType
from ..register import ModelLoader, ModelMeta, register_model
from .olmoe import OLMoEBridge, OLMoESelfAttention


class Olmo2SelfAttention(OLMoESelfAttention):
    """OLMo-2/3 attention.

    Inherits OLMoE-style full-channel q/k RMSNorm, and additionally applies
    a post-attention RMSNorm on the o_proj output (before the residual add),
    matching the HF post-norm architecture (no input layernorm in HF).
    """

    def __init__(self, config: ModelConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.post_self_attn_layernorm = build_module(
            TENorm,
            hidden_size=self.config.hidden_size,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def forward(self, hidden_states, *args, **kwargs):
        output, bias = super().forward(hidden_states, *args, **kwargs)
        assert bias is None, 'OLMo-2/3 self attention does not support bias.'
        output = self.post_self_attn_layernorm(output)
        return output, bias


class Olmo2MLP(MLP):
    """OLMo-2/3 MLP: applies a post-MLP RMSNorm before the residual add."""

    def __init__(self, config: ModelConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.post_mlp_layernorm = build_module(
            TENorm,
            hidden_size=self.config.hidden_size,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def forward(self, hidden_states, *args, **kwargs):
        output, bias = super().forward(hidden_states, *args, **kwargs)
        assert bias is None, 'OLMo-2/3 MLP does not support bias.'
        output = self.post_mlp_layernorm(output)
        return output, bias

    def sharded_state_dict(self,
                           prefix: str = '',
                           sharded_offsets: tuple = (),
                           metadata: Optional[dict] = None) -> ShardedStateDict:
        sharded_state_dict = {}
        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        for name, module in self._modules.items():
            sub_sd = sharded_state_dict_default(module, f'{prefix}{name}.', sharded_offsets, metadata)
            if self.config.gated_linear_unit and name == 'linear_fc1':
                for k, v in sub_sd.items():
                    if k in (f'{prefix}{name}.weight', f'{prefix}{name}.bias'):
                        sub_sd[k] = apply_swiglu_sharded_factory(v, sharded_offsets, singleton_local_shards)
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict


class Olmo2Bridge(OLMoEBridge):
    """OLMo-2/3 bridge.

    OLMo-2/3 is a post-norm only architecture: there is no `input_layernorm`
    nor `pre_feedforward_layernorm` on the HF side. Each layer instead has:
      * `post_attention_layernorm.weight`  -- after self-attn, before residual
      * `post_feedforward_layernorm.weight` -- after MLP, before residual
    Together with OLMoE-style full-channel q/k_norm.
    """

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        mg_attn = None if mg_layer is None else mg_layer.self_attention
        # q/k/v/o + full-channel q_norm/k_norm via the inherited OLMoE path.
        hf_state_dict.update(
            self._set_attn_state(mg_attn, hf_state_dict, f'{self.hf_attn_prefix}.', layer_idx, to_mcore))
        # No HF `input_layernorm.weight` exists; map the HF post-attn norm
        # to the post_self_attn_layernorm we attach in Olmo2SelfAttention.
        self._set_state_dict(mg_layer, 'self_attention.post_self_attn_layernorm.weight', hf_state_dict,
                             'post_attention_layernorm.weight', to_mcore)
        return hf_state_dict

    def _set_layer_mlp(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool, is_mtp: bool = False):
        mg_mlp = None if mg_layer is None else mg_layer.mlp
        hf_state_dict.update(
            self._set_mlp_state(mg_mlp, hf_state_dict, f'{self.hf_mlp_prefix}.', layer_idx, to_mcore))
        # No HF `pre_feedforward_layernorm.weight` exists; map the HF
        # post-MLP norm to the post_mlp_layernorm we attach in Olmo2MLP.
        self._set_state_dict(mg_layer, 'mlp.post_mlp_layernorm.weight', hf_state_dict,
                             'post_feedforward_layernorm.weight', to_mcore)
        return hf_state_dict


class Olmo2Loader(ModelLoader):

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        transformer_layer_spec = super().get_transformer_layer_spec(vp_stage)
        for layer_spec in transformer_layer_spec.layer_specs:
            # OLMo-2/3 has no pre-norm: drop the layernorm fused into linear_qkv/linear_fc1
            # and explicitly mark input_layernorm / pre_mlp_layernorm as identity ops.
            layer_spec.submodules.input_layernorm = IdentityOp
            layer_spec.submodules.pre_mlp_layernorm = IdentityOp
            layer_spec.submodules.self_attention.submodules.linear_qkv = TEColumnParallelLinear
            layer_spec.submodules.mlp.submodules.linear_fc1 = TEColumnParallelLinear
            # Attach post-norms via custom SelfAttention / MLP modules.
            layer_spec.submodules.self_attention.module = Olmo2SelfAttention
            self._set_mlp_spec(layer_spec.submodules, Olmo2MLP)
        return transformer_layer_spec


register_model(ModelMeta(
    ModelType.olmo2,
    ['olmo2'],
    bridge_cls=Olmo2Bridge,
    loader=Olmo2Loader,
))

register_model(ModelMeta(
    ModelType.olmo3,
    ['olmo3'],
    bridge_cls=Olmo2Bridge,
    loader=Olmo2Loader,
))
