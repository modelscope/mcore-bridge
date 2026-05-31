# Copyright (c) ModelScope Contributors. All rights reserved.
import megatron.core
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.extensions.transformer_engine import TEGroupedLinear, TELayerNormColumnParallelLinear, TELinear
from megatron.core.models.gpt import gpt_model
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec, get_gpt_mtp_block_spec
from megatron.core.transformer.moe.router import TopKRouter as McoreTopKRouter
from megatron.core.transformer.multi_latent_attention import MLASelfAttention as McoreMLASelfAttention
from megatron.core.transformer.transformer_layer import TransformerLayer as McoreTransformerLayer
from packaging import version
from torch import nn
from typing import TYPE_CHECKING, List, Optional, Type, Union

from mcore_bridge.bridge import GPTBridge
from mcore_bridge.config import ModelConfig
from mcore_bridge.utils import get_logger

from .modules import MLASelfAttention, MultiTokenPredictionLayer, TopKRouter, TransformerBlock, TransformerLayer

if TYPE_CHECKING:
    from .gpt_model import GPTModel
    from .mm_gpt_model import MultimodalGPTModel

MODEL_MAPPING = {}
logger = get_logger()


@dataclass
class ModelMeta:
    model_type: str
    model_types: List[str]

    bridge_cls: Type[GPTBridge] = GPTBridge
    visual_cls: Optional[Type[nn.Module]] = None
    is_multimodal: bool = False
    loader: Optional[Type['ModelLoader']] = None

    def __post_init__(self):
        if self.visual_cls is not None:
            self.is_multimodal = True
        if self.loader is None:
            self.loader = ModelLoader


def register_model(model_meta: ModelMeta, *, exist_ok: bool = False):
    model_type = model_meta.model_type
    if not exist_ok and model_type in MODEL_MAPPING:
        raise ValueError(f'The `{model_type}` has already been registered in the MODEL_MAPPING.')
    MODEL_MAPPING[model_type] = model_meta


model_type_mapping = None


def get_mcore_model_type(hf_model_type: str) -> Optional[str]:
    global model_type_mapping
    if model_type_mapping is None:
        model_type_mapping = {}
        for k, model_meta in MODEL_MAPPING.items():
            for _model_type in model_meta.model_types:
                model_type_mapping[_model_type] = k
    return model_type_mapping.get(hf_model_type)


def get_model_meta(mcore_model_type: str) -> ModelMeta:
    return MODEL_MAPPING.get(mcore_model_type)


class ModelLoader:
    model_cls = None
    transformer_block = TransformerBlock

    def __init__(self, config: ModelConfig):
        from mcore_bridge.model import GPTModel, MultimodalGPTModel
        self.config = config
        if self.model_cls is None:
            self.model_cls = MultimodalGPTModel if config.is_multimodal else GPTModel

    def _set_mlp_spec(self, layer_submodules, mlp_module, mlp_key='mlp'):
        mlp_spec = getattr(layer_submodules, mlp_key)
        if isinstance(mlp_spec, partial):
            mlp_spec = partial(
                mlp_module.as_mlp_submodule if hasattr(mlp_module, 'as_mlp_submodule') else mlp_module, *mlp_spec.args,
                **mlp_spec.keywords)
            setattr(layer_submodules, mlp_key, mlp_spec)
        else:
            mlp_spec.module = mlp_module

    def _replace_spec_dsa(self, layer_spec):
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            _get_backend_spec_provider, get_dsa_module_spec_for_backend)
        backend = _get_backend_spec_provider(config=self.config)
        dsa_spec = get_dsa_module_spec_for_backend(self.config, backend)
        if self.config.qk_layernorm:
            linear_q_up_proj = backend.column_parallel_linear()
            # fix megatron-core
            dsa_spec.submodules.q_layernorm = backend.layer_norm(for_qk=True)
            dsa_spec.submodules.kv_layernorm = backend.layer_norm(for_qk=True)
            dsa_spec.submodules.linear_q_up_proj = linear_q_up_proj
            dsa_spec.submodules.linear_kv_up_proj = linear_q_up_proj
        layer_spec.submodules.self_attention = dsa_spec

    @contextmanager
    def _patch_experimental_attention_variant(self):
        experimental_attention_variant = self.config.experimental_attention_variant
        self.config.experimental_attention_variant = None
        try:
            yield
        finally:
            self.config.experimental_attention_variant = experimental_attention_variant

    def _deepcopy_layer_spec(self, transformer_layer_spec):
        for i, layer_spec in enumerate(transformer_layer_spec.layer_specs):
            transformer_layer_spec.layer_specs[i] = deepcopy(layer_spec)

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        with self._patch_experimental_attention_variant():
            transformer_layer_spec = get_gpt_decoder_block_spec(
                self.config,
                use_transformer_engine=True,
                normalization=self.config.normalization,
                qk_l2_norm=self.config.qk_l2_norm,
                vp_stage=vp_stage)
            self._deepcopy_layer_spec(transformer_layer_spec)
        if self.config.experimental_attention_variant == 'dsa':
            for layer_spec in transformer_layer_spec.layer_specs:
                self._replace_spec_dsa(layer_spec)
        return transformer_layer_spec

    def get_mtp_block_spec(self, transformer_layer_spec, vp_stage: Optional[int] = None):
        mtp_block_spec = get_gpt_mtp_block_spec(
            self.config, transformer_layer_spec, use_transformer_engine=True, vp_stage=vp_stage)
        if mtp_block_spec is not None:
            for layer_spec in mtp_block_spec.layer_specs:
                layer_spec.module = MultiTokenPredictionLayer
        return mtp_block_spec

    def _set_shared_expert_gate(self, transformer_layer_spec):
        mcore_016 = version.parse(megatron.core.__version__) >= version.parse('0.16.0rc0')
        if (not mcore_016 and self.config.moe_shared_expert_gate and self.config.num_moe_experts
                and self.config.moe_shared_expert_intermediate_size):
            for layer_spec in transformer_layer_spec.layer_specs:
                if hasattr(layer_spec.submodules.mlp.submodules, 'shared_experts'):
                    layer_spec.submodules.mlp.submodules.shared_experts.params = {'gate': True}

    def _set_transformer_layer(self, transformer_layer_spec):
        if self.config.enable_hyper_connections:
            return
        for layer_spec in transformer_layer_spec.layer_specs:
            if layer_spec.module is McoreTransformerLayer:
                layer_spec.module = TransformerLayer

    def _replace_mla_attention(self, transformer_layer_spec):
        for layer_spec in transformer_layer_spec.layer_specs:
            self_attention = layer_spec.submodules.self_attention
            if self_attention.module is McoreMLASelfAttention:
                self_attention.module = MLASelfAttention

    def _replace_router(self, transformer_layer_spec, mlp_key='mlp'):
        for layer_spec in transformer_layer_spec.layer_specs:
            mlp_spec = getattr(layer_spec.submodules, mlp_key, None)
            if mlp_spec is not None:
                if isinstance(mlp_spec, partial):
                    mlp_submodules = mlp_spec.keywords.get('submodules')
                else:
                    mlp_submodules = getattr(mlp_spec, 'submodules', None)
                if getattr(mlp_submodules, 'router', None) is McoreTopKRouter:
                    mlp_submodules.router = TopKRouter

    def build_model(
        self,
        pre_process=True,
        post_process=True,
        vp_stage: Optional[int] = None,
    ) -> Union['GPTModel', 'MultimodalGPTModel']:
        transformer_layer_spec = self.get_transformer_layer_spec(vp_stage=vp_stage)
        self._set_shared_expert_gate(transformer_layer_spec)
        self._set_transformer_layer(transformer_layer_spec)
        self._replace_mla_attention(transformer_layer_spec)
        self._replace_router(transformer_layer_spec)
        mtp_block_spec = None
        if self.config.mtp_num_layers is not None:
            mtp_block_spec = self.get_mtp_block_spec(transformer_layer_spec, vp_stage=vp_stage)
        with self._patch_transformer_block():
            model = self.model_cls(
                config=self.config,
                transformer_layer_spec=transformer_layer_spec,
                pre_process=pre_process,
                post_process=post_process,
                mtp_block_spec=mtp_block_spec,
                vp_stage=vp_stage,
            )
        self._set_linear_is_expert(model)
        return model

    @contextmanager
    def _patch_transformer_block(self):
        TransformerBlock = gpt_model.TransformerBlock
        gpt_model.TransformerBlock = self.transformer_block
        try:
            yield
        finally:
            gpt_model.TransformerBlock = TransformerBlock

    def _set_linear_is_expert(self, model):
        for n, module in model.named_modules():
            if '.local_experts.' in n and isinstance(module, (TELinear, TELayerNormColumnParallelLinear)) or isinstance(
                    module, TEGroupedLinear):
                module.is_expert = True


def get_mcore_model(config: ModelConfig) -> List[nn.Module]:
    loader = config.model_meta.loader(config)
    model_type = ModelType.encoder_or_decoder
    if mpu.get_pipeline_model_parallel_world_size() > 1 and config.virtual_pipeline_model_parallel_size is not None:
        models = []
        for i in range(config.virtual_pipeline_model_parallel_size):
            pre_process = mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=i)
            post_process = mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=i)
            model = loader.build_model(pre_process, post_process, vp_stage=i)
            models.append(model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        model = loader.build_model(pre_process=pre_process, post_process=post_process)
        models = [model]
    for model in models:
        model.model_type = model_type
        model.prepare_inputs_for_generation = None
    return models
