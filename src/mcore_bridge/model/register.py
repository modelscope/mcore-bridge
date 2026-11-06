# Copyright (c) ModelScope Contributors. All rights reserved.
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.extensions.transformer_engine import (TEGroupedLinear, TELayerNormColumnParallelLinear, TELinear,
                                                         TENorm)
from megatron.core.models.gpt import gpt_model
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_layer_specs, get_gpt_mtp_block_spec
from megatron.core.transformer.moe.router import TopKRouter as McoreTopKRouter
from megatron.core.transformer.multi_latent_attention import MLASelfAttention as McoreMLASelfAttention
from megatron.core.transformer.multi_token_prediction import get_mtp_num_layers_to_build
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import TransformerLayer as McoreTransformerLayer
from torch import nn
from typing import TYPE_CHECKING, List, Optional, Type, Union

from mcore_bridge.bridge import GPTBridge
from mcore_bridge.config import ModelConfig
from mcore_bridge.utils import get_local_layer_specs, get_logger

from .modules import (AbsorbedMLASelfAttention, DSAIndexer, MLASelfAttention, TopKRouter, TransformerBlock,
                      TransformerLayer)
from .mtp import adapt_mtp_block_spec, get_mtp_block_cls

try:
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import \
        get_transformer_layer_with_experimental_attention_variant_spec as _build_experimental_layer_specs
except ImportError:  # megatron-core 0.18: layer-level entry point absent
    _build_experimental_layer_specs = None

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
    config_cls: Type[ModelConfig] = ModelConfig
    mtp_prefix: Optional[str] = None
    # Most HF checkpoints declare MTP depth here; Qwen models override it.
    # Dotted paths are accepted for composite configs.
    mtp_depth_field: Optional[str] = 'num_nextn_predict_layers'
    mtp_repeated_field: Optional[str] = None
    # Number of physical modules in one MTP prediction step (for hybrid MTP).
    mtp_layers_per_depth: int = 1

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
    final_layer_norm_cls = TENorm

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

    def decoder_layer_specs(self):
        """All decoder layer specs, model-patched, independent of pp/vpp slicing.

        Final entry point. Subclasses extend `_build_decoder_layer_specs` (construction) and,
        rarely, `_patch_layer_specs` (class replacement) so the model-level patch below can
        never be skipped.
        """
        return self._patch_layer_specs(self._build_decoder_layer_specs())

    def _build_decoder_layer_specs(self):
        """Construct every decoder layer; subclasses replace this when upstream cannot."""
        variant = self.config.experimental_attention_variant
        if variant is None:
            layer_specs = self._plain_decoder_layer_specs()
        elif _build_experimental_layer_specs is None:
            raise NotImplementedError(f'{type(self).__name__} with experimental_attention_variant={variant!r} requires '
                                      'megatron-core >= 0.19 (the layer-level experimental entry point).')
        else:
            layer_specs = _build_experimental_layer_specs(self.config)
        layer_specs = [deepcopy(layer_spec) for layer_spec in layer_specs]
        if variant == 'dsa':
            self._replace_dsa_indexer(layer_specs)
        return layer_specs

    def _plain_decoder_layer_specs(self):
        return get_gpt_decoder_layer_specs(
            self.config,
            use_transformer_engine=True,
            normalization=self.config.normalization,
            qk_l2_norm=self.config.qk_l2_norm)

    def _replace_dsa_indexer(self, layer_specs):
        # Upstream's indexer lacks bridge's FP8 init / SP gather / rope mscale.
        for layer_spec in layer_specs:
            core_attention = layer_spec.submodules.self_attention.submodules.core_attention
            core_attention.submodules.indexer.module = DSAIndexer

    def _patch_layer_specs(self, layer_specs):
        # The three helpers below take a block spec, so wrap the list once.
        block = TransformerBlockSubmodules(layer_specs=layer_specs)
        self._set_transformer_layer(block)
        self._replace_mla_attention(block)
        self._replace_router(block)
        return layer_specs

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        """The local block of one pipeline stage (historical name: returns a block spec)."""
        return TransformerBlockSubmodules(
            layer_specs=get_local_layer_specs(self.config, self.decoder_layer_specs(), vp_stage=vp_stage),
            layer_norm=self.final_layer_norm_cls)

    def get_mtp_block_spec(self, transformer_layer_spec, vp_stage: Optional[int] = None):
        # Upstream returns None for stages that hold no MTP layer. Check that first so
        # embedding-only / loss-only stages never build a full spec (or hit a version error).
        if get_mtp_num_layers_to_build(self.config, vp_stage=vp_stage) == 0:
            return None
        if not transformer_layer_spec.layer_specs:
            cls = type(self)
            if (cls._build_decoder_layer_specs is ModelLoader._build_decoder_layer_specs
                    and cls.get_transformer_layer_spec is not ModelLoader.get_transformer_layer_spec):
                raise NotImplementedError(
                    f'{cls.__name__} only overrides get_transformer_layer_spec; its extra spec '
                    'changes never reach decoder_layer_specs(), so an MTP-only stage would build a '
                    'template that differs from the normal layout. Override `_build_decoder_layer_specs` '
                    '(and call super()) instead.')
            transformer_layer_spec = TransformerBlockSubmodules(
                layer_specs=[self.decoder_layer_specs()[-1]], layer_norm=transformer_layer_spec.layer_norm)
        mtp_block_spec = get_gpt_mtp_block_spec(
            self.config, transformer_layer_spec, use_transformer_engine=True, vp_stage=vp_stage)
        return self._patch_mtp_block_spec(adapt_mtp_block_spec(self.config, mtp_block_spec))

    def _patch_mtp_block_spec(self, mtp_block_spec):
        """Model-level MTP tweaks; subclasses replace the module/norm classes here."""
        return mtp_block_spec

    def _set_transformer_layer(self, transformer_layer_spec):
        # HC models pick their own layer class; only the vanilla one is swapped.
        if getattr(self.config, 'enable_hyper_connections', False):  # field exists from 0.20
            return
        for layer_spec in transformer_layer_spec.layer_specs:
            if layer_spec.module is McoreTransformerLayer:
                layer_spec.module = TransformerLayer

    def _replace_mla_attention(self, transformer_layer_spec):
        for layer_spec in transformer_layer_spec.layer_specs:
            # Hybrid models (e.g. nemotron_h) may have layers with no attention submodule
            # at all (MambaLayer) or with it replaced by IdentityOp (FFN-only layers).
            self_attention = getattr(layer_spec.submodules, 'self_attention', None)
            if not hasattr(self_attention, 'module'):
                continue
            if self_attention.module is McoreMLASelfAttention:
                self_attention.module = MLASelfAttention
            elif getattr(self_attention.module, '__name__', None) == 'AbsorbedMLASelfAttention':
                self_attention.module = AbsorbedMLASelfAttention

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
        mtp_block_spec = None
        if self.config.mtp_num_layers is not None:
            mtp_block_spec = self.get_mtp_block_spec(transformer_layer_spec, vp_stage=vp_stage)
        with self._patch_transformer_block(), self._patch_mtp_block():
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

    # GPTModel picks its block class at construction time; swap it while the model is built.
    @contextmanager
    def _patch_transformer_block(self):
        TransformerBlock = gpt_model.TransformerBlock
        gpt_model.TransformerBlock = self.transformer_block
        try:
            yield
        finally:
            gpt_model.TransformerBlock = TransformerBlock

    @contextmanager
    def _patch_mtp_block(self):
        """Temporarily replace the MTP block class used by upstream GPTModel
        so that :class:`MultiTokenPredictionBlock` is instantiated instead."""
        if self.config.mtp_num_layers is None:
            yield
            return
        original = getattr(gpt_model, 'MultiTokenPredictionBlock', None)
        if original is None:
            yield
            return
        gpt_model.MultiTokenPredictionBlock = get_mtp_block_cls(self.config)
        try:
            yield
        finally:
            gpt_model.MultiTokenPredictionBlock = original

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
