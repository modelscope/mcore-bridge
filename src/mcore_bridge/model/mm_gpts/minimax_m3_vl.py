# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.distributed as dist
from transformers import PretrainedConfig

from mcore_bridge.bridge import GPTBridge

from ..constant import ModelType
from ..register import ModelMeta, register_model
from .utils import HuggingFaceVit


class MinimaxM3Vit(HuggingFaceVit):
    module_mapping = {'model.vision_tower': 'vision_tower', 'model.multi_modal_projector': 'multi_modal_projector'}
    _vision_tower = ['vision_tower']
    _aligner = ['multi_modal_projector']

    def prepare_model(self, hf_config: PretrainedConfig):
        from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (MiniMaxM3VLMultiModalProjector,
                                                                              MiniMaxM3VLVisionModel)
        self.vision_tower = MiniMaxM3VLVisionModel(hf_config.vision_config).to(hf_config.dtype)
        self.multi_modal_projector = MiniMaxM3VLMultiModalProjector(hf_config).to(hf_config.dtype)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        return self._hf_get_inputs_embeds(inputs_embeds, kwargs, self.visual, self.hf_config)


class MinimaxM3Bridge(GPTBridge):
    hf_layers_prefix = 'language_model.model.layers'
    hf_mtp_prefix = 'language_model.model.layers'
    hf_embed_key = 'language_model.model.embed_tokens.weight'
    hf_final_layernorm_key = 'language_model.model.norm.weight'
    hf_lm_head_key = 'language_model.lm_head.weight'
    hf_expert_bias_key = 'e_score_correction_bias'

    def _set_moe_state(
        self,
        mg_mlp,
        hf_state_dict,
        hf_prefix: str,
        layer_idx: int,
        to_mcore: bool,
        is_mtp: bool = False,
    ):
        if to_mcore:
            hf_state_dict = {
                k.replace('.w1.', '.gate_proj.').replace('.w3.', '.up_proj.').replace('.w2.', '.down_proj.'): v
                for k, v in hf_state_dict.items()
            }
        hf_state_dict = super()._set_moe_state(mg_mlp, hf_state_dict, hf_prefix, layer_idx, to_mcore, is_mtp)
        if not to_mcore:
            hf_state_dict = {
                k.replace('.gate_proj.', '.w1.').replace('.up_proj.', '.w3.').replace('.down_proj.', '.w2.'): v
                for k, v in hf_state_dict.items()
            }
        return hf_state_dict


register_model(
    ModelMeta(
        ModelType.minimax_m3_vl,
        ['minimax_m3_vl'],
        bridge_cls=MinimaxM3Bridge,
        visual_cls=MinimaxM3Vit,
    ))
