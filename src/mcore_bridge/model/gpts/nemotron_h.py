# Copyright (c) ModelScope Contributors. All rights reserved.
"""Nemotron-3.5 (hybrid Mamba2 + Attention + MoE) on megatron-core's HybridModel.

Upstream deprecated `GPTModel` in favour of `HybridModel` (Megatron-LM #5911). On
`HybridModel` one pattern symbol *is* one layer, so the `IdentityOp` stripping and the
`MambaLayer` compat shim that a `GPTModel` build would need are both unnecessary here,
and MTP can span several heterogeneous inner layers.
"""
import torch
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm, TERowParallelLinear
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from typing import Optional

from mcore_bridge.bridge import GPTBridge
from mcore_bridge.tuners import LoraParallelLinear
from mcore_bridge.utils import get_logger

from ..constant import ModelType
from ..hybrid_model import HybridModel
from ..register import ModelLoader, ModelMeta, register_model

logger = get_logger()


class NemotronHBridge(GPTBridge):
    """HuggingFace <-> Megatron-Core weight conversion for Nemotron-3.5.

    Layer families come from `hybrid_layer_pattern`, one symbol per layer:
      M = Mamba2 SSM,  E = MoE (routed + shared),  * = attention (GQA),  - = dense MLP

    All three families sit under the same HF `mixer.` prefix, so dispatch is driven by
    the pattern rather than by the key name.
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
    hf_mtp_prefix = 'mtp.layers'
    hf_mtp_final_layernorm_key = 'final_layernorm.weight'

    _LAYER_TYPES = {'M': 'mamba', 'E': 'moe', '*': 'attention', '-': 'mlp'}

    def _get_layer_type(self, layer_idx: int):
        """Resolve a layer's family from the pattern.

        A negative index means "MTP inner layer i" (see `_convert_mtp_layer`), which is
        described by `mtp_hybrid_override_pattern` rather than the backbone pattern.
        """
        if layer_idx < 0:
            pattern = self.config.mtp_hybrid_override_pattern
            idx = -layer_idx - 1
        else:
            pattern = self.config.hybrid_layer_pattern
            idx = layer_idx
        assert 0 <= idx < len(pattern), f'layer index {idx} out of range for pattern {pattern!r}'
        return self._LAYER_TYPES[pattern[idx]]

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
        # Experts are stored one module per expert, with separate up/down projections.
        return False, False

    def _set_final_layernorm(self, lm_model, hf_state_dict, to_mcore):
        # `HybridStack` names the trailing norm `final_norm` (`TransformerBlock` uses
        # `final_layernorm`).
        self._set_state_dict(lm_model, 'decoder.final_norm.weight', hf_state_dict, self.hf_final_layernorm_key,
                             to_mcore)

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        """Convert the sequence-mixing half of a layer: attention or Mamba."""
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
        # 'moe'/'-' layers have no sequence mixer; their norm is handled in _set_layer_mlp.
        return hf_state_dict

    def _set_mamba_state(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        """Convert Mamba2 SSM weights under the `mixer.` prefix.

        MambaMixer keeps `conv1d_weight` / `conv1d_bias` as flat nn.Parameters (not a
        `conv1d` submodule), and `in_proj` is a single packed [z, x, B, C, dt] projection.
        The HF checkpoint uses the same packed layout, so at TP=1 every tensor maps 1:1.

        Under TP the packed tensors need per-block slicing rather than one contiguous cut:
        upstream sizes each block by its own local width (`d_inner_local_tp`,
        `ngroups_local_tp * d_state`, `nheads_local_tp`), so a rank owns a slice of *every*
        block. A naive split of the concatenation would land inside one block and silently
        hand a rank the wrong projections -- see `_split_packed_dim0`.
        """
        hf_prefix = f'{self.hf_attn_prefix}.'
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        mg_mixer = None if mg_layer is None else mg_layer.mixer
        self._set_mamba_in_proj(mg_mixer, hf_state_dict, to_mcore)
        self._set_mamba_conv1d(mg_mixer, hf_state_dict, to_mcore)
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

    def _mamba_block_sizes(self):
        """Global dim-0 sizes of the [z, x, B, C, dt] blocks packed into `in_proj`.

        Derived from `self.config` rather than the mixer instance, so it stays valid on PP
        ranks that do not hold this Mamba layer (`mg_mixer is None`).
        """
        d_inner = self.config.mamba_num_heads * self.config.mamba_head_dim
        bc = self.config.mamba_num_groups * self.config.mamba_state_dim
        return [d_inner, d_inner, bc, bc, self.config.mamba_num_heads]

    def _split_packed_dim0(self, tensor, block_sizes):
        """Take this TP rank's slice out of each packed block along dim 0."""
        out, offset = [], 0
        for size in block_sizes:
            local = size // self.tp_size
            start = offset + self.tp_rank * local
            out.append(tensor[start:start + local])
            offset += size
        return torch.cat(out, dim=0)

    def _merge_packed_dim0(self, gathered, block_sizes):
        """Inverse of `_split_packed_dim0`.

        `_all_gather_tp` concatenates the per-rank shards along dim 0, so `gathered` reads
        [rank0 blocks..., rank1 blocks..., ...]. Regroup it back into whole global blocks.
        """
        local_total = sum(size // self.tp_size for size in block_sizes)
        shards = [gathered[i * local_total:(i + 1) * local_total] for i in range(self.tp_size)]
        blocks, offset = [], 0
        for size in block_sizes:
            local = size // self.tp_size
            blocks.append(torch.cat([s[offset:offset + local] for s in shards], dim=0))
            offset += local
        return torch.cat(blocks, dim=0)

    def _set_mamba_packed(self, mg_param, hf_state_dict, hf_key, block_sizes, to_mcore: bool):
        """Load/export a packed Mamba tensor whose dim-0 blocks are each TP-sharded.

        `mg_param` is None on a PP rank that does not own this layer. Both collectives still
        have to run on every rank: the TP all-gather tolerates None, and the PP broadcast is
        what actually hands the merged tensor to the non-owning ranks -- returning early
        instead would drop this layer from the export entirely.
        """
        if to_mcore:
            if mg_param is None:
                return
            weight = hf_state_dict[hf_key].load()
            # `_set_weight` would split by `_get_tp_split_dim`, i.e. one contiguous cut, so
            # slice per block here and hand it the already-local shard (tp_dim None).
            self._set_weight(mg_param, self._split_packed_dim0(weight, block_sizes), None)
        else:
            gathered = self._all_gather_tp(None if mg_param is None else mg_param.data, 0, False)
            merged = None if gathered is None else self._merge_packed_dim0(gathered, block_sizes)
            # Non-owning PP ranks receive the merged tensor here; owning ranks send it.
            merged = self._broadcast_ep_pp(merged, False)
            # `_all_gather_tp` leaves the result on cuda; the generic export path applies
            # `_target_device` when it writes into hf_state_dict, so do the same here.
            if self._target_device is not None:
                merged = merged.to(self._target_device)
            hf_state_dict[hf_key] = merged

    def _set_mamba_in_proj(self, mg_mixer, hf_state_dict, to_mcore: bool):
        """`in_proj` packs [z, x, B, C, dt]; each block is TP-sharded on its own."""
        if self.tp_size == 1:
            self._set_state_dict(mg_mixer, 'in_proj.weight', hf_state_dict, 'in_proj.weight', to_mcore)
            return
        mg_param = None if mg_mixer is None else mg_mixer.in_proj.weight
        self._set_mamba_packed(mg_param, hf_state_dict, 'in_proj.weight', self._mamba_block_sizes(), to_mcore)

    def _set_mamba_conv1d(self, mg_mixer, hf_state_dict, to_mcore: bool):
        """conv1d mirrors `in_proj` minus the dt block: [x, B, C] along dim 0."""
        if self.tp_size == 1:
            self._set_state_dict(mg_mixer, 'conv1d_weight', hf_state_dict, 'conv1d.weight', to_mcore)
            self._set_state_dict(mg_mixer, 'conv1d_bias', hf_state_dict, 'conv1d.bias', to_mcore)
            return
        d_inner = self.config.mamba_num_heads * self.config.mamba_head_dim
        bc = self.config.mamba_num_groups * self.config.mamba_state_dim
        blocks = [d_inner, bc, bc]
        for mg_name, hf_name in (('conv1d_weight', 'conv1d.weight'), ('conv1d_bias', 'conv1d.bias')):
            mg_param = None if mg_mixer is None else getattr(mg_mixer, mg_name)
            self._set_mamba_packed(mg_param, hf_state_dict, hf_name, blocks, to_mcore)

    def _set_layer_mlp(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool, is_mtp: bool = False):
        """Convert the channel-mixing half of a layer: MoE for 'E'."""
        if self._get_layer_type(layer_idx) == 'moe':
            mg_mlp = None if mg_layer is None else mg_layer.mlp
            hf_state_dict.update(
                self._set_moe_state(
                    mg_mlp, hf_state_dict, f'{self.hf_mlp_prefix}.', layer_idx, to_mcore, is_mtp=is_mtp))
            self._set_state_dict(mg_layer, 'pre_mlp_layernorm.weight', hf_state_dict, self.hf_input_layernorm_key,
                                 to_mcore)
        # mamba / attention layers have no MLP sub-module.
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
        reused here; everything else still goes through `_set_state_dict` / `_set_weight`.
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
                    # Under LoRA the expert linear is wrapped, and the per-expert `weight{i}`
                    # live on the wrapped module. This branch exports merged base weights
                    # (`_peft_format` is False), so unwrap before indexing.
                    if isinstance(mg_linear, LoraParallelLinear):
                        mg_linear = mg_linear.base_layer
                    # `linear_fc1_up` aliases linear_fc1 to bypass the base gated-fc1
                    # reshape; TP dim is registered for both names in _get_tp_split_dim.
                    tp_key = 'linear_fc1_up.weight' if mg_name == 'linear_fc1' else 'linear_fc2.weight'
                    if to_mcore:
                        weight = torch.concat([
                            hf_state_dict[f'{start_idx + i}.{hf_name}.weight'].load() for i in range(num_local_experts)
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
                # dense MLP / shared expert, same non-gated fc1 handling as above.
                fc1_module = None if mg_mlp is None else mg_mlp.linear_fc1
                if isinstance(fc1_module, LoraParallelLinear):
                    fc1_module = fc1_module.base_layer
                if to_mcore:
                    self._set_weight(fc1_module.weight, hf_state_dict['up_proj.weight'].load(), 'linear_fc1_up.weight')
                else:
                    fc1 = None if fc1_module is None else fc1_module.weight.data
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

    def _convert_mtp_layer(self, lm_model, hf_state_dict, hf_prefix: str, layer_idx: int, to_mcore: bool):
        """Map one MTP depth, whose inner layers span several HF indices.

        With `mtp_hybrid_override_pattern='*E'` a depth holds two inner layers, and HF stores
        them as two `mtp.layers.{0,1}` entries: index 0 carries `enorm`/`hnorm`/`eh_proj`
        plus the attention mixer, index 1 the MoE mixer plus `final_layernorm`. mcore keeps
        both under `mtp.layers[depth].mtp_model_layer.layers[i]`, so the base class
        assumption of a single `mtp_layer.transformer_layer` does not hold.
        """
        pattern = self.config.mtp_hybrid_override_pattern
        mtp_layer = lm_model.mtp.layers[layer_idx] if hasattr(lm_model, 'mtp') else None
        n_inner = len(pattern)
        exported = {}
        for inner_idx in range(n_inner):
            inner_prefix = f'{hf_prefix}{layer_idx * n_inner + inner_idx}.'
            if to_mcore:
                inner_sd = self._remove_prefix(hf_state_dict, inner_prefix)
                if not inner_sd:
                    logger.info(f'MTP inner layer {inner_prefix} safetensors weights not found, '
                                'this part will be randomly initialized.')
                    continue
            else:
                inner_sd = {}
            inner_layer = None if mtp_layer is None else mtp_layer.mtp_model_layer.layers[inner_idx]
            # enorm/hnorm/eh_proj live on the MTP layer itself and only exist on inner 0.
            if inner_idx == 0:
                for key in ['enorm.weight', 'hnorm.weight', 'eh_proj.weight']:
                    self._set_state_dict(mtp_layer, key, inner_sd, key, to_mcore)
                self._fp8_skip_modules.update({'eh_proj'})
            if inner_idx == n_inner - 1:
                self._set_state_dict(mtp_layer, 'final_layernorm.weight', inner_sd, self.hf_mtp_final_layernorm_key,
                                     to_mcore)
            # Negative index selects `mtp_hybrid_override_pattern` in `_get_layer_type`.
            mtp_layer_idx = -(inner_idx + 1)
            inner_sd.update(self._set_layer_attn(inner_layer, inner_sd, mtp_layer_idx, to_mcore))
            inner_sd.update(self._set_layer_mlp(inner_layer, inner_sd, mtp_layer_idx, to_mcore, is_mtp=True))
            if not to_mcore:
                exported.update(self._add_prefix(inner_sd, inner_prefix))
        return {} if to_mcore else exported


class NemotronHLoader(ModelLoader):
    model_cls = HybridModel

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        """Return a `HybridStack` spec with a standalone pre-norm on Mamba layers.

        Upstream's default mamba spec fuses the pre-norm into `in_proj`
        (`TELayerNormColumnParallelLinear`), which renames the weight to
        `mixer.in_proj.layer_norm_weight`. This checkpoint stores it as a separate
        `norm.weight`, so `TENorm` keeps the Bridge mapping one-to-one.
        """
        submodules = HybridStackSubmodules(
            **{
                field: getattr(hybrid_stack_spec.submodules, field)
                for field in hybrid_stack_spec.submodules.__dataclass_fields__
            })
        # Separate norm from in_proj: the fused TELayerNormColumnParallelLinear would rename the
        # weight to `mixer.in_proj.layer_norm_weight`, while this checkpoint stores a standalone
        # `norm.weight`. Keeping them separate makes the Bridge mapping one-to-one.
        submodules.mamba_layer = ModuleSpec(
            module=MambaLayer,
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
        return ModuleSpec(module=HybridStack, submodules=submodules)

    def build_model(self, pre_process=True, post_process=True, vp_stage: Optional[int] = None):
        """Build via `HybridModel`, skipping the base class's layer_specs post-processing.

        `ModelLoader.build_model` rewrites `spec.layer_specs` (MLA / router / TransformerLayer
        substitution); a `HybridStack` spec exposes per-layer-family submodules instead, and
        this model needs none of those substitutions.
        """
        model = self.model_cls(
            config=self.config,
            transformer_layer_spec=self.get_transformer_layer_spec(vp_stage=vp_stage),
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )
        self._set_linear_is_expert(model)
        return model


register_model(ModelMeta(
    ModelType.nemotron_h,
    ['nemotron_h'],
    bridge_cls=NemotronHBridge,
    loader=NemotronHLoader,
))
