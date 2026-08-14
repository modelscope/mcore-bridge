# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import torch
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import build_module
from torch import Tensor
from transformers import AutoModel
from typing import Optional, Tuple

from mcore_bridge.bridge import MultimodalGPTBridge

from ..constant import ModelType
from ..gpt_model import GPTModel
from ..mm_gpt_model import MultimodalGPTModel
from ..modules import TransformerLayer
from ..register import ModelLoader, ModelMeta, register_model
from .utils import HuggingFaceVit


class MuseGlimmerRMSNormNoScale(torch.nn.Module):
    """RMSNorm without a learnable scale, mirroring HF `MuseGlimmerRMSNorm(with_scale=False)`."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)).to(orig_dtype)


class MuseGlimmerVit(HuggingFaceVit):
    # The aligner is two-stage: `vision_adapter` (fc1/fc2) reduces the pooled patch features and
    # `vision_projection` maps them onto the text hidden size.
    module_mapping = {
        'model.vision_tower': 'vision_tower',
        'model.vision_adapter': 'vision_adapter',
        'model.vision_projection': 'vision_projection',
    }
    _vision_tower = ['vision_tower']
    _aligner = ['vision_adapter', 'vision_projection']

    def prepare_model(self, hf_config):
        from transformers.models.muse_glimmer import MuseGlimmerVisionModel
        from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerVisionAdapter
        self.vision_tower = MuseGlimmerVisionModel._from_config(hf_config.vision_config)
        self.vision_adapter = MuseGlimmerVisionAdapter(hf_config)
        self.vision_projection = torch.nn.Linear(
            hf_config.projector_hidden_size, hf_config.text_config.hidden_size, bias=False)
        # Weightless RMSNorm applied to the projected perception embeddings before they are
        # scattered into `inputs_embeds`.
        self.perception_emb_norm = MuseGlimmerRMSNormNoScale(
            hf_config.text_config.hidden_size, eps=hf_config.text_config.rms_norm_eps)
        # HF wraps the token embedding in `MuseGlimmerTextNormedEmbedding`, which applies a weightless
        # RMSNorm on top of the lookup. Megatron's VocabParallelEmbedding has no such hook, so it is
        # applied here -- before the vision scatter, since vision embeddings carry
        # `perception_emb_norm` instead and must not be normalized twice.
        self.embed_norm = MuseGlimmerRMSNormNoScale(
            hf_config.text_config.hidden_size, eps=hf_config.text_config.rms_norm_eps)

    @staticmethod
    def _cast(x, module):
        # tower / adapter / projection are three separate top-level modules here, so a frozen vision
        # tower or a fp32 precision check can leave them in different dtypes; re-cast per boundary.
        return x.to(next(module.parameters()).dtype)

    def _encode_vision(self, pixel_values, grid_thw):
        # Mirrors HF `MuseGlimmerModel.get_image_features`; videos go through the very same path.
        hidden_states = self.vision_tower(
            self._cast(pixel_values, self.vision_tower), grid_thw=grid_thw).last_hidden_state
        hidden_states = self.vision_adapter(self._cast(hidden_states, self.vision_adapter))
        hidden_states = self.vision_projection(self._cast(hidden_states, self.vision_projection))
        return self.perception_emb_norm(hidden_states)

    def get_inputs_embeds_language_model(self, inputs_embeds, **kwargs):
        return self.embed_norm(inputs_embeds)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        inputs_embeds = self.embed_norm(inputs_embeds)
        input_ids = kwargs['input_ids']
        hf_config = self.hf_config
        pixel_values = kwargs.get('pixel_values')
        pixel_values_videos = kwargs.get('pixel_values_videos')
        image_grid_thw = kwargs.get('image_grid_thw')
        video_grid_thw = kwargs.get('video_grid_thw')
        vision_config = HuggingFaceVit._get_vision_config(hf_config)
        if pixel_values is None and pixel_values_videos is None:  # plain-text
            # Keep the vision tower in the autograd graph so that DDP/FSDP see gradients for every
            # rank even when a micro-batch happens to carry no media.
            # Mirrors `MuseGlimmerVisionPatchEmbedder`: patch_temporal * 3 channels * patch_size**2.
            hidden_size = vision_config.patch_temporal * 3 * vision_config.patch_size**2
            dummy = torch.zeros(16 * 16, hidden_size, device=input_ids.device)
            embeds = self._encode_vision(dummy, input_ids.new_tensor([[1, 16, 16]]))
            inputs_embeds = inputs_embeds + embeds.mean().to(device=inputs_embeds.device) * 0.
            return {'inputs_embeds': inputs_embeds}

        for values, grid_thw, token_id in [(pixel_values, image_grid_thw, hf_config.image_token_id),
                                           (pixel_values_videos, video_grid_thw, hf_config.video_token_id)]:
            if values is None:
                continue
            embeds = self._encode_vision(values, grid_thw)
            embeds = embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            mask = (input_ids == token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(mask, embeds)
        return {'inputs_embeds': inputs_embeds}


class MuseGlimmerSelfAttention(SelfAttention):
    """SelfAttention with weightless qk_norm, a constant query scale and a sigmoid output gate."""

    def __init__(self, config, submodules, *args, **kwargs):
        text_config = config.hf_config.text_config
        submodules = self._strip_qk_layernorm(submodules)
        super().__init__(config, submodules, *args, **kwargs)
        # HF normalizes q/k with a scale-free RMSNorm, so there is no weight to bridge; the parent
        # applies these two modules itself. The query is additionally multiplied by a constant while
        # the key is left untouched.
        self.q_layernorm = MuseGlimmerRMSNormNoScale(self.hidden_size_per_attention_head, eps=text_config.rms_norm_eps)
        self.k_layernorm = MuseGlimmerRMSNormNoScale(self.hidden_size_per_attention_head, eps=text_config.rms_norm_eps)
        self.qk_scale_factor = text_config.qk_scale_factor
        # `gate_proj` produces one scalar per (head, channel) of the local attention output, so it is
        # sharded exactly like `linear_qkv`'s query part -- column parallel with no gather.
        self.gate_proj = build_module(
            TEColumnParallelLinear,
            config.hidden_size,
            text_config.num_attention_heads * self.hidden_size_per_attention_head,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='gate_proj',
            tp_group=self.pg_collection.tp,
        )
        # HF gates the attention output with the *pre-attention* activation right before the output
        # projection. Stashing the gate and applying it through a pre-hook on `linear_proj` keeps the
        # `(output, bias)` contract of `SelfAttention.forward` intact for the inference paths.
        self._attn_gate = None
        self.linear_proj.register_forward_pre_hook(self._apply_attn_gate)

    def _apply_attn_gate(self, module, args):
        gate = self._attn_gate
        if gate is None:
            return None
        core_attn_out = args[0]
        gate = gate.to(core_attn_out.dtype).reshape(core_attn_out.shape)
        return (core_attn_out * gate, ) + tuple(args[1:])

    @staticmethod
    def _strip_qk_layernorm(submodules):
        # The spec may carry TENorm for q/k; replace it so the parent does not allocate weights that
        # this model does not have.
        submodules.q_layernorm = IdentityOp
        submodules.k_layernorm = IdentityOp
        return submodules

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, *args, **kwargs):
        # The parent already applies `q_layernorm` / `k_layernorm` (our scale-free RMSNorm) on the
        # per-head view, so only the constant query scale is left to add. Re-normalizing here would
        # be a no-op at best and would cancel the scale at worst, since RMSNorm is scale-invariant.
        query, key, value = super().get_query_key_value_tensors(hidden_states, key_value_states, *args, **kwargs)
        return query * self.qk_scale_factor, key, value

    def _input_layernorm(self, hidden_states: Tensor) -> Tensor:
        # `TransformerLayer.input_layernorm` is an IdentityOp: the real norm is fused into
        # `linear_qkv` (TELayerNormColumnParallelLinear), so `forward` receives the *un-normalized*
        # hidden states. HF feeds `gate_proj` the normalized ones, hence the recomputation here.
        module = self.linear_qkv
        # With LoRA the fused linear is wrapped, and the norm weight stays on the base layer.
        while not hasattr(module, 'layer_norm_weight'):
            module = module.base_layer
        weight = module.layer_norm_weight
        x = hidden_states.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.config.layernorm_epsilon)
        if self.config.layernorm_zero_centered_gamma:
            x = x * (1.0 + weight.float())
        else:
            x = x * weight.float()
        return x.to(hidden_states.dtype)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        gate, _ = self.gate_proj(self._input_layernorm(hidden_states))
        self._attn_gate = torch.sigmoid(gate.to(torch.float32))
        try:
            return super().forward(hidden_states, attention_mask, **kwargs)
        finally:
            self._attn_gate = None


class MuseGlimmerTransformerLayer(TransformerLayer):

    def __init__(self, config, submodules, *args, **kwargs):
        super().__init__(config, submodules, *args, **kwargs)
        text_config = config.hf_config.text_config
        hidden_size = config.hidden_size
        # Sandwich norm: HF uses `rms_norm_eps` for the two pre-norms (handled by the parent through
        # `linear_qkv.layer_norm_weight` / `linear_fc1.layer_norm_weight`) but a much smaller
        # `post_norm_eps` for the two post-norms.
        post_eps = text_config.post_norm_eps
        self.post_attention_layernorm = build_module(TENorm, hidden_size=hidden_size, config=config, eps=post_eps)
        self.post_feedforward_layernorm = build_module(TENorm, hidden_size=hidden_size, config=config, eps=post_eps)
        # `layer_rope_theta[i] == 0` marks a NoPE layer: HF passes `position_embeddings=None` there.
        self.use_rope = bool(text_config.layer_rope_theta[self.layer_number - 1])
        # The post-norms must run on the sublayer output *before* the residual add. Injecting them as
        # module hooks keeps the parent's `_forward_attention` / `_forward_mlp` (and their offload,
        # recompute and fused-TP paths) untouched instead of reproducing that body here.
        self.self_attention.register_forward_hook(self._norm_attn_output)
        self.mlp.register_forward_hook(self._norm_mlp_output)
        if not self.use_rope:
            self.self_attention.register_forward_pre_hook(self._drop_rope, with_kwargs=True)

    def _norm_attn_output(self, module, args, output):
        out, bias = output
        assert bias is None, 'post_attention_layernorm would be applied before the bias add'
        return self.post_attention_layernorm(out), bias

    def _norm_mlp_output(self, module, args, output):
        out, bias = output
        assert bias is None, 'post_feedforward_layernorm would be applied before the bias add'
        return self.post_feedforward_layernorm(out), bias

    @staticmethod
    def _drop_rope(module, args, kwargs):
        for key in ['rotary_pos_emb', 'rotary_pos_cos', 'rotary_pos_sin', 'rotary_pos_cos_sin']:
            if key in kwargs:
                kwargs[key] = None
        return args, kwargs


class MuseGlimmerTextGPTModel(GPTModel):

    def _forward_output_layer(self, hidden_states, *args, **kwargs):
        # Hooking the output layer instead of `forward` is what keeps the training path correct: with
        # labels, `GPTModel.forward` returns the loss (also a Tensor), which must not be softcapped.
        logits = super()._forward_output_layer(hidden_states, *args, **kwargs)
        text_config = self.config.hf_config.text_config
        softcap = text_config.final_logit_softcapping
        if softcap is None:
            return logits
        # HF scales the logits then squashes them through tanh: `T * tanh(logits * mult / T)`.
        logits = logits * text_config.output_multiplier
        logits = logits / softcap
        logits = torch.tanh(logits)
        return logits * softcap


class MuseGlimmerGPTModel(MultimodalGPTModel):
    language_model_cls = MuseGlimmerTextGPTModel


class MuseGlimmerBridge(MultimodalGPTBridge):
    # `self_attention.gate_proj` is a ColumnParallelLinear, so its rows follow the query heads under
    # TP. mcore's MLP uses `linear_fc1`, so this name cannot collide with the HF `mlp.gate_proj`.
    additional_dim0_keys = {'gate_proj'}

    def _set_qk_layernorm(self, mg_attn, hf_state_dict, to_mcore, **kwargs):
        # q/k norms are scale-free in this architecture, so there is nothing to bridge.
        return hf_state_dict

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        hf_state_dict = super()._set_layer_attn(mg_layer, hf_state_dict, layer_idx, to_mcore)
        # The module has to be `self_attention` rather than the layer: `_get_tp_split_dim` resolves LoRA
        # keys from the *front* (`key.lora_B.default.weight`), so any extra prefix would hide `gate_proj`
        # from `additional_dim0_keys` and the adapter would be saved as an un-gathered TP shard.
        mg_attn = None if mg_layer is None else mg_layer.self_attention
        self._set_state_dict(mg_attn, 'gate_proj.weight', hf_state_dict, f'{self.hf_attn_prefix}.gate_proj.weight',
                             to_mcore)
        return hf_state_dict

    def _set_layer_mlp(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool, is_mtp: bool = False):
        mg_mlp = None if mg_layer is None else mg_layer.mlp
        hf_state_dict.update(self._set_mlp_state(mg_mlp, hf_state_dict, f'{self.hf_mlp_prefix}.', layer_idx, to_mcore))
        self._set_state_dict(mg_layer, 'mlp.linear_fc1.layer_norm_weight', hf_state_dict,
                             'pre_feedforward_layernorm.weight', to_mcore)
        return hf_state_dict

    def _set_layer_state(self, mg_layer, hf_state_dict, hf_prefix: str, layer_idx: int, to_mcore: bool):
        hf_prefix = f'{hf_prefix}{layer_idx}.'
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        hf_state_dict.update(self._set_layer_attn(mg_layer, hf_state_dict, layer_idx, to_mcore))
        hf_state_dict.update(self._set_layer_mlp(mg_layer, hf_state_dict, layer_idx, to_mcore))
        for key in ['post_attention_layernorm', 'post_feedforward_layernorm']:
            self._set_state_dict(mg_layer, f'{key}.weight', hf_state_dict, f'{key}.weight', to_mcore)
        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict


class MuseGlimmerLoader(ModelLoader):
    model_cls = MuseGlimmerGPTModel

    def build_model(self, pre_process=True, post_process=True, vp_stage: Optional[int] = None):
        model = super().build_model(pre_process, post_process, vp_stage)
        # `layernorm_zero_centered_gamma` is set globally for the four centered per-layer norms, but
        # HF's final `norm` is a plain RMSNorm (`x * w`), so it has to opt out individually.
        lm_model = model.language_model if hasattr(model, 'language_model') else model
        final_layernorm = getattr(lm_model.decoder, 'final_layernorm', None)
        if final_layernorm is not None:
            final_layernorm.zero_centered_gamma = False
            if hasattr(final_layernorm, 'config'):
                final_layernorm.config = copy.copy(final_layernorm.config)
                final_layernorm.config.layernorm_zero_centered_gamma = False
        return model

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        layer_specs = get_gpt_decoder_block_spec(
            self.config, use_transformer_engine=True, normalization=self.config.normalization, vp_stage=vp_stage)
        for layer_spec in layer_specs.layer_specs:
            layer_spec.submodules.self_attention.module = MuseGlimmerSelfAttention
        return layer_specs

    def _set_transformer_layer(self, transformer_layer_spec):
        for layer_spec in transformer_layer_spec.layer_specs:
            layer_spec.module = MuseGlimmerTransformerLayer


register_model(
    ModelMeta(
        ModelType.muse_glimmer,
        ['muse_glimmer'],
        bridge_cls=MuseGlimmerBridge,
        visual_cls=MuseGlimmerVit,
        loader=MuseGlimmerLoader,
    ))
