# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from transformers import AutoModel, PretrainedConfig
from typing import Optional

from mcore_bridge.bridge import MultimodalGPTBridge
from mcore_bridge.config import ModelConfig

from ..constant import ModelType
from ..gpt_model import GPTModel
from ..mm_gpt_model import MultimodalGPTModel
from ..register import ModelLoader, ModelMeta, register_model
from ..rope import get_rope_inv_freq
from .utils import HuggingFaceVit


class Gemma4Vit(HuggingFaceVit):
    module_mapping = {
        'model.vision_tower': 'vision_tower',
        'model.embed_vision': 'embed_vision',
        'model.audio_tower': 'audio_tower',
        'model.embed_audio': 'embed_audio',
    }
    _vision_tower = ['vision_tower', 'audio_tower']
    _aligner = ['embed_vision', 'embed_audio']
    support_multimodal = False

    def prepare_model(self, hf_config: PretrainedConfig):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder
        self.vision_tower = AutoModel.from_config(hf_config.vision_config)
        self.audio_tower = AutoModel.from_config(hf_config.audio_config) if hf_config.audio_config is not None else None
        self.embed_vision = Gemma4MultimodalEmbedder(hf_config.vision_config, hf_config.text_config)
        self.embed_audio = (
            Gemma4MultimodalEmbedder(hf_config.audio_config, hf_config.text_config)
            if hf_config.audio_config is not None else None)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        return inputs_embeds


class Gemma4SelfAttention(SelfAttention):

    def __init__(
        self,
        config: ModelConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        *args,
        **kwargs,
    ):
        text_config = config.hf_config.text_config
        super().__init__(config, submodules, layer_number, *args, **kwargs)


class Gemma4Bridge(MultimodalGPTBridge):
    pass


class Gemma4TextGPTModel(GPTModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print()

    def _set_inv_freq(self):
        rope_scaling = self.config.rope_scaling
        self.config.rope_scaling = rope_scaling['sliding_attention']
        new_inv_freq, attention_scaling = get_rope_inv_freq(self.config)
        assert attention_scaling == 1, 'not support'
        self.rotary_pos_emb.inv_freq = new_inv_freq.to(self.rotary_pos_emb.inv_freq.device)
        # full
        self.full_rotary_pos_emb = copy.copy(self.rotary_pos_emb)
        self.config.rope_scaling = rope_scaling['full_attention']
        kwargs = {}
        if self.config.rope_scaling['rope_type'] == 'proportional':
            kwargs['head_dim_key'] = 'global_head_dim'
        new_inv_freq, attention_scaling = get_rope_inv_freq(self.config, **kwargs)
        assert attention_scaling == 1, 'not support'
        self.full_rotary_pos_emb.inv_freq = new_inv_freq
        self.attention_scaling = attention_scaling

        self.config.rope_scaling = rope_scaling


class Gemma4GPTModel(MultimodalGPTModel):
    language_model_cls = Gemma4TextGPTModel


class Gemma4Loader(ModelLoader):
    model_cls = Gemma4GPTModel

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        layer_specs = get_gpt_decoder_block_spec(
            self.config, use_transformer_engine=True, normalization=self.config.normalization, vp_stage=vp_stage)
        for layer_spec in layer_specs.layer_specs:
            layer_spec.submodules.self_attention.module = Gemma4SelfAttention
        return layer_specs


register_model(
    ModelMeta(
        ModelType.gemma4,
        ['gemma4'],
        bridge_cls=Gemma4Bridge,
        visual_cls=Gemma4Vit,
        loader=Gemma4Loader,
    ))
