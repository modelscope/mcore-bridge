# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from typing import Optional

from mcore_bridge.bridge import GPTBridge

from ..constant import ModelType
from ..register import ModelLoader, ModelMeta, register_model
from .bailing_moe import BailingMoeBridge


class NemotronHBridge(BailingMoeBridge):
    """Bridge for Nemotron-3.5 Hybrid (Mamba2 + Attention + MoE) model.

    Handles weight conversion between HuggingFace and Megatron-Core formats
    for three layer types determined by hybrid_override_pattern:
      M = Mamba2 SSM layer
      E = MoE expert layer (128 routed + 1 shared)
      * = Attention layer (GQA)
    """
    hf_embed_key = 'backbone.embeddings.weight'
    hf_layers_prefix = 'backbone.layers'
    hf_final_layernorm_key = 'backbone.norm_f.weight'
    hf_lm_head_key = 'lm_head.weight'
    hf_attn_prefix = 'mixer'
    hf_mlp_prefix = 'mixer'
    hf_input_layernorm_key = 'norm.weight'
    hf_o_proj_key = 'o_proj'
    hf_q_norm_key = 'q_norm.weight'
    hf_k_norm_key = 'k_norm.weight'
    hf_gate_key = 'gate.weight'
    hf_expert_bias_key = 'gate.e_score_correction_bias'
    hf_shared_expert_key = 'shared_experts'

    # Nemotron attention uses separate q/k/v projections; restore the base
    # implementation (BailingMoeBridge overrides it with fused query_key_value).
    _set_qkv = GPTBridge._set_qkv

    def _get_layer_type(self, layer_idx):
        """Parse hybrid_layer_pattern to get layer type.

        `layer_idx == -1` is used by the MTP path; MTP layers have their own pattern
        (`mtp_hybrid_layer_pattern`) and must not index the backbone pattern.
        """
        pattern = self.config.hybrid_layer_pattern
        assert 0 <= layer_idx < len(pattern), (
            f'layer_idx {layer_idx} out of range for hybrid_layer_pattern of length {len(pattern)}. '
            'MTP layers must be dispatched via mtp_hybrid_layer_pattern, not the backbone pattern.')
        return {'M': 'mamba', 'E': 'moe', '*': 'attention', '-': 'mlp'}[pattern[layer_idx]]

    def _get_tp_split_dim(self, mg_key: Optional[str]) -> Optional[int]:
        # `D` and `conv1d_{weight,bias}` are flat nn.Parameters on MambaMixer (no dot in the
        # relative key for `D`), so the base class keyword lookup cannot classify them.
        if mg_key in {'D', 'conv1d_weight', 'conv1d_bias'}:
            return 0
        if mg_key == 'mixer.norm.weight':
            # Inner gated RMSNorm of the Mamba mixer is sharded over d_inner.
            return 0
        if mg_key is not None and mg_key.split('.', 1)[0] in {'linear_fc1', 'linear_fc1_up'}:
            # relu^2 is non-gated, so linear_fc1 is a plain [ffn, hidden] column-parallel
            # weight. The base class returns 1 because it assumes the gated [2, X, Y] layout.
            return 0
        return super()._get_tp_split_dim(mg_key)

    def _get_hf_experts_attr(self, is_mtp: bool = False):
        # Not hf_grouped, not gate_up merged format.
        return False, False

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        """Dispatch attention/mamba weight conversion based on layer type."""
        layer_type = self._get_layer_type(layer_idx)
        if layer_type == 'attention':
            mg_attn = None if mg_layer is None else mg_layer.self_attention
            hf_state_dict.update(
                self._set_attn_state(mg_attn, hf_state_dict, f'{self.hf_attn_prefix}.', layer_idx, to_mcore))
            # Pre-norm is fused into linear_qkv (TELayerNormColumnParallelLinear).
            self._set_state_dict(mg_layer, 'self_attention.linear_qkv.layer_norm_weight', hf_state_dict,
                                 self.hf_input_layernorm_key, to_mcore)
        elif layer_type == 'mamba':
            hf_state_dict.update(self._set_mamba_state(mg_layer, hf_state_dict, layer_idx, to_mcore))
            # MambaLayer keeps a standalone pre-norm (`norm`, not `input_layernorm`).
            self._set_state_dict(mg_layer, 'norm.weight', hf_state_dict, self.hf_input_layernorm_key, to_mcore)
        # 'moe' layers have no attention/mixer part here; their norm is handled in _set_layer_mlp
        return hf_state_dict

    def _set_mamba_state(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        """Convert Mamba2 SSM weights under the `mixer.` prefix.

        MambaMixer keeps `conv1d_weight` / `conv1d_bias` as flat nn.Parameters (not a
        `conv1d` submodule), and `in_proj` is a single packed [z, x, B, C, dt] projection.
        The HF checkpoint uses exactly the same packed layout and shapes, so every tensor
        maps 1:1 with no reordering.
        """
        hf_prefix = f'{self.hf_attn_prefix}.'
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        mg_mixer = None if mg_layer is None else mg_layer.mixer
        self._set_state_dict(mg_mixer, 'in_proj.weight', hf_state_dict, 'in_proj.weight', to_mcore)
        self._set_state_dict(mg_mixer, 'conv1d_weight', hf_state_dict, 'conv1d.weight', to_mcore)
        self._set_state_dict(mg_mixer, 'conv1d_bias', hf_state_dict, 'conv1d.bias', to_mcore)
        self._set_state_dict(mg_mixer, 'A_log', hf_state_dict, 'A_log', to_mcore)
        self._set_state_dict(mg_mixer, 'D', hf_state_dict, 'D', to_mcore)
        self._set_state_dict(mg_mixer, 'dt_bias', hf_state_dict, 'dt_bias', to_mcore)
        self._set_state_dict(mg_mixer, 'out_proj.weight', hf_state_dict, 'out_proj.weight', to_mcore)
        # Inner gated RMSNorm, present when the mixer uses rmsnorm.
        has_inner_norm = False if mg_mixer is None else getattr(mg_mixer, 'norm', None) is not None
        has_inner_norm = self._reduce_tensor_pp_group(has_inner_norm, to_mcore)
        if has_inner_norm:
            self._set_state_dict(mg_layer, 'mixer.norm.weight', hf_state_dict, 'norm.weight', to_mcore)
        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _set_layer_mlp(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool, is_mtp: bool = False):
        """Dispatch MoE weight conversion for E-type layers."""
        layer_type = self._get_layer_type(layer_idx)
        if layer_type == 'moe':
            mg_mlp = None if mg_layer is None else mg_layer.mlp
            hf_state_dict.update(
                self._set_moe_state(
                    mg_mlp, hf_state_dict, f'{self.hf_mlp_prefix}.', layer_idx, to_mcore, is_mtp=is_mtp))
            self._set_state_dict(mg_layer, 'pre_mlp_layernorm.weight', hf_state_dict, self.hf_input_layernorm_key,
                                 to_mcore)
        # mamba / attention layers have no MLP sub-module
        return hf_state_dict

    def _set_mlp_state(
        self,
        mg_mlp,
        hf_state_dict,
        hf_prefix: str,
        layer_idx: int,
        to_mcore: bool,
        ep_rank: Optional[int] = None,
        is_mtp: bool = False,
    ):
        """Map linear_fc1 <-> up_proj 1:1.

        Nemotron uses relu^2, which is non-gated: there is no gate_proj, so linear_fc1 is a
        plain [ffn, hidden] tensor rather than the merged [gate_proj; up_proj] layout the
        base class assumes. That single assumption is why the base implementation cannot be
        reused here; everything else still goes through `_set_state_dict`.
        """
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        is_expert = ep_rank is not None
        if not self._peft_format:
            if is_expert:
                num_local_experts = self.config.num_moe_experts // self.ep_size
                start_idx = ep_rank * num_local_experts
                for mg_name, hf_name in [('linear_fc1', 'up_proj'), ('linear_fc2', 'down_proj')]:
                    mg_linear = None if mg_mlp is None else getattr(mg_mlp, mg_name)
                    # `linear_fc1_up` aliases linear_fc1 to bypass the base gated-fc1
                    # reshape; TP dim is registered for both names in _get_tp_split_dim.
                    tp_key = 'linear_fc1_up.weight' if mg_name == 'linear_fc1' else 'linear_fc2.weight'
                    if to_mcore:
                        weight = torch.concat(
                            [
                                hf_state_dict[f'{start_idx + i}.{hf_name}.weight'].load()
                                for i in range(num_local_experts)
                            ],
                            dim=0)
                        self._set_weight([getattr(mg_linear, f'weight{i}') for i in range(num_local_experts)],
                                         weight,
                                         tp_key,
                                         is_expert=True)
                    else:
                        mg_weight = None if mg_linear is None else [
                            getattr(mg_linear, f'weight{i}').data for i in range(num_local_experts)
                        ]
                        # `_get_weight` reshapes to [num_local_experts, ffn, hidden].
                        weight, _ = self._get_weight(mg_weight, tp_key, is_expert=True)
                        if weight is not None:
                            for i in range(num_local_experts):
                                hf_state_dict[f'{start_idx + i}.{hf_name}.weight'] = weight[i].clone()
                        del weight
            else:
                # dense MLP / shared expert. `linear_fc1_up` is an alias of `linear_fc1` that
                # avoids the base `_get_weight` gated-fc1 reshape (which forces a [2, X, Y]
                # view); the real module path is still linear_fc1.
                if to_mcore:
                    self._set_weight(mg_mlp.linear_fc1.weight, hf_state_dict['up_proj.weight'].load(),
                                     'linear_fc1_up.weight')
                else:
                    fc1 = None if mg_mlp is None else mg_mlp.linear_fc1.weight.data
                    weight, _ = self._get_weight(fc1, 'linear_fc1_up.weight')
                    if weight is not None:
                        hf_state_dict['up_proj.weight'] = weight.clone()
                    del weight
                self._set_state_dict(mg_mlp, 'linear_fc2.weight', hf_state_dict, 'down_proj.weight', to_mcore)
        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict


def _build_mamba_layer_cls():
    """MambaLayer subclass that matches TransformerBlock's calling convention.

    Three mismatches have to be absorbed so a Mamba layer can live inside a plain
    GPTModel/TransformerBlock instead of mcore's dedicated MambaStack:

    * `TransformerBlock.build_layer` always passes `vp_stage=`, while
      `MambaLayer.__init__` only accepts `pp_layer_offset`.
    * `TransformerLayer` adds the pipeline offset to `layer_number` internally, but
      `MambaLayer` stores it verbatim and expects the caller to supply
      `pp_layer_offset`. Without this, PP>1 ranks report local layer numbers (e.g.
      [1, 4] instead of [3, 4]) and `GPTBridge._convert` indexes the wrong layer.
    * `TransformerBlock.forward` calls layers with the full `TransformerLayer.forward`
      keyword set (`context`, `attention_bias`, `rotary_pos_cos`, ...) and unpacks a
      `(hidden_states, context)` pair; `MambaLayer.forward` accepts only a small subset
      and returns just `hidden_states`.
    """
    from megatron.core.ssm.mamba_layer import MambaLayer
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    class _MambaLayerCompat(MambaLayer):

        def __init__(self, config, submodules, layer_number: int = 1, *args, vp_stage=None, **kwargs):
            offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
            super().__init__(
                config,
                submodules,
                layer_number=layer_number + offset,
                *args,
                pp_layer_offset=offset,
                **kwargs)
            self.vp_stage = vp_stage

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            context=None,
            context_mask=None,
            rotary_pos_emb=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            rotary_pos_cos_sin=None,
            attention_bias=None,
            inference_context=None,
            packed_seq_params=None,
            sequence_len_offset=None,
            padding_mask=None,
            *,
            inference_params=None,
        ):
            # Mamba has no cross-attention and no positional encoding; the extra
            # TransformerLayer kwargs are inapplicable and intentionally dropped.
            hidden_states = super().forward(
                hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
            )
            # TransformerBlock unpacks `(hidden_states, context)`.
            return hidden_states, context

    return _MambaLayerCompat


class NemotronHLoader(ModelLoader):
    """Loader for Nemotron-3.5 that builds dynamic layer specs.

    Uses hybrid_layer_pattern to assign different layer specs:
      M -> MambaLayer (SSM)
      E -> MoE layer (routed + shared experts)
      * -> Standard attention (GQA)
      - -> Dense MLP

    MTP is not supported yet: the HF checkpoint stores MTP layers under a separate
    `mtp.layers.*` tree with its own `mtp_hybrid_override_pattern`, whose two-level
    index flattening is not implemented here.
    """

    _mamba_layer_cls = None

    def __init__(self, config):
        super().__init__(config)
        if config.mtp_num_layers:
            raise NotImplementedError(
                'nemotron_h MTP conversion is not implemented. The HF checkpoint keeps MTP under '
                '`mtp.layers.*` with its own mtp_hybrid_override_pattern, which requires dedicated '
                'index-flattening mappings. Set mtp_num_layers=None to convert the backbone only.')

    def get_transformer_layer_spec(self, vp_stage=None):
        """Build per-layer specs based on hybrid_layer_pattern.

        Each layer holds exactly ONE mixer, so the unused half of the standard
        (attention + MLP) layer must be stripped, otherwise every 'E' layer would build an
        unused attention block and every '*' layer an unused MLP -- ~600M phantom params
        for this checkpoint, randomly initialized and picked up by the optimizer.

        `moe_layer_freq` (derived from the same pattern in parser.py) already decides which
        layers get a MoE vs dense MLP; here we drop whichever submodule the layer type does
        not use, and swap 'M' layers for MambaLayer entirely.
        """
        from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
        pattern = self.config.hybrid_layer_pattern
        transformer_layer_spec = super().get_transformer_layer_spec(vp_stage=vp_stage)
        # `super()` returns only this PP/VPP stage's layers, so local index 0 is not
        # necessarily global layer 0. The pattern is indexed globally.
        offset = get_transformer_layer_offset(self.config, vp_stage=vp_stage)
        for i, layer_spec in enumerate(transformer_layer_spec.layer_specs):
            ch = pattern[offset + i]
            if ch == 'M':
                # A fresh spec per layer: layer_specs entries must not alias each other,
                # matching the base class `_deepcopy_layer_spec` contract.
                transformer_layer_spec.layer_specs[i] = self._get_mamba_layer_spec()
                continue
            submodules = layer_spec.submodules
            if ch == '*':
                # Attention-only layer: no FFN. Its pre-norm is fused into linear_qkv.
                # `mlp_bda` must go too: it unpacks its input as (output, bias), which an
                # IdentityOp mlp does not produce.
                submodules.mlp = IdentityOp
                submodules.pre_mlp_layernorm = IdentityOp
                submodules.mlp_bda = IdentityFuncOp
            else:
                # 'E'/'-': FFN-only layer. Its pre-norm is fused into the MLP's fc1
                # (or pre_mlp_layernorm for MoE), so drop attention and its norm.
                submodules.self_attention = IdentityOp
                submodules.input_layernorm = IdentityOp
                submodules.self_attn_bda = IdentityFuncOp
        return transformer_layer_spec

    def _get_mamba_layer_spec(self):
        """Build a MambaLayer spec for Mamba2 SSM layers.

        The Bridge expects a standalone pre-norm (`norm`) and a mixer with
        plain `in_proj`/`out_proj`, so TENorm + TEColumnParallelLinear are used
        instead of the fused TELayerNormColumnParallelLinear variant.
        """
        try:
            from megatron.core.extensions.transformer_engine import (TEColumnParallelLinear, TENorm,
                                                                     TERowParallelLinear)
            from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
            from megatron.core.ssm.mamba_layer import MambaLayerSubmodules
            from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
            from megatron.core.transformer.spec_utils import ModuleSpec
        except ImportError as e:
            raise ImportError('NemotronHLoader requires a megatron-core version with Mamba2 SSM support '
                              '(megatron.core.ssm.mamba_layer / mamba_mixer).') from e
        if self._mamba_layer_cls is None:
            self._mamba_layer_cls = _build_mamba_layer_cls()
        return ModuleSpec(
            module=self._mamba_layer_cls,
            submodules=MambaLayerSubmodules(
                norm=TENorm,
                mixer=ModuleSpec(
                    module=MambaMixer,
                    submodules=MambaMixerSubmodules(
                        in_proj=TEColumnParallelLinear,
                        out_proj=TERowParallelLinear,
                    ),
                ),
                mamba_bda=get_bias_dropout_add,
            ),
        )


register_model(
    ModelMeta(
        ModelType.nemotron_h,
        ['nemotron_h'],
        bridge_cls=NemotronHBridge,
        loader=NemotronHLoader,
    ))
