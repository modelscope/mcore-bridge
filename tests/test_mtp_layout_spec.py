# Copyright (c) ModelScope Contributors. All rights reserved.
"""Regression tests for MTP layer-spec resolution on stages without decoder layers.

An MTP-only pipeline stage used to hand an empty local block to upstream's MTP spec
builder, which indexes ``layer_specs[-1]`` and raises ``IndexError`` (ms-swift#9823).
The MTP template must come from the full, patched, un-sliced layer list instead.
"""
import importlib
import inspect
import pytest
import torch
from copy import deepcopy
from megatron.core import parallel_state
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from unittest import mock

import mcore_bridge  # noqa: F401  (applies runtime patches at import)
from mcore_bridge.model.modules import TransformerLayer as BridgeTransformerLayer
from mcore_bridge.model.register import ModelLoader, _build_experimental_layer_specs


def _config(layout='Ettt|mL', num_layers=3, pp=2):
    return TransformerConfig(
        pipeline_dtype=torch.float32,
        num_layers=num_layers,
        hidden_size=128,
        num_attention_heads=4,
        num_query_groups=4,
        ffn_hidden_size=256,
        normalization='RMSNorm',
        add_bias_linear=False,
        gated_linear_unit=True,
        pipeline_model_parallel_size=pp,
        pipeline_model_parallel_layout=layout,
        mtp_num_layers=1,
    )


def _loader(loader_cls, config):
    # Bypass __init__: these tests only exercise spec resolution, not model building.
    loader = loader_cls.__new__(loader_cls)
    loader.config = config
    return loader


def _rank1_last_stage():
    return mock.patch.multiple(
        parallel_state,
        get_pipeline_model_parallel_rank=mock.Mock(return_value=1),
        get_virtual_pipeline_model_parallel_world_size=mock.Mock(return_value=None),
    )


def test_decoder_layer_specs_has_no_stage_argument():
    """A stage argument would turn the full layer list back into a local slice."""
    assert 'vp_stage' not in inspect.signature(ModelLoader.decoder_layer_specs).parameters


def test_decoder_layer_specs_returns_every_layer_regardless_of_stage():
    loader = _loader(ModelLoader, _config())
    assert len(loader.decoder_layer_specs()) == 3


def test_experimental_variant_uses_upstream_layer_entry():
    """Variant models go through upstream's layer-level experimental entry (>= 0.19)."""
    if _build_experimental_layer_specs is None:
        pytest.skip('layer-level experimental entry requires megatron-core >= 0.19')
    config = _config()
    config.experimental_attention_variant = 'dsa'
    config.multi_latent_attention = True
    loader = _loader(ModelLoader, config)
    assert len(loader.decoder_layer_specs()) == 3


def test_dsa_attention_spec_is_not_dropped():
    """A plain-MLA fallback keeps the class name but loses the indexer, so check both."""
    if _build_experimental_layer_specs is None:
        pytest.skip('layer-level experimental entry requires megatron-core >= 0.19')
    config = _config()
    config.experimental_attention_variant = 'dsa'
    config.multi_latent_attention = True
    config.qk_layernorm = True
    loader = _loader(ModelLoader, config)
    self_attention = loader.decoder_layer_specs()[0].submodules.self_attention
    core = self_attention.submodules.core_attention
    assert self_attention.module.__module__.startswith('mcore_bridge')
    assert core.module.__name__ == 'DSAttention'
    assert hasattr(core.submodules, 'indexer')


_MIGRATED_LOADERS = [
    ('mcore_bridge.model.gpts.glm4', 'Glm4Loader'),
    ('mcore_bridge.model.gpts.olmoe', 'OLMoELoader'),
    ('mcore_bridge.model.gpts.minimax_m2', 'MinimaxM2Loader'),
    ('mcore_bridge.model.mm_gpts.llama4', 'Llama4Loader'),
    ('mcore_bridge.model.gpts.glm_moe_dsa', 'GlmMoeDsaLoader'),
    ('mcore_bridge.model.gpts.deepseek_v4', 'DeepseekV4Loader'),
]

# Only GDN still needs the 0.18 block-level fallback; DeepSeek-V4 now requires 0.20 outright.
_MIGRATED_WITH_018_FALLBACK = [
    ('mcore_bridge.model.gpts.qwen3_next_gdn', 'Qwen3NextLoader'),
]


@pytest.mark.parametrize('module_name,class_name', _MIGRATED_LOADERS)
def test_migrated_loader_extends_the_full_list_only(module_name, class_name):
    """A migrated loader must not keep a second slicing path."""
    loader_cls = getattr(importlib.import_module(module_name), class_name)
    assert loader_cls._build_decoder_layer_specs is not ModelLoader._build_decoder_layer_specs
    assert loader_cls.get_transformer_layer_spec is ModelLoader.get_transformer_layer_spec


@pytest.mark.parametrize('module_name,class_name', _MIGRATED_WITH_018_FALLBACK)
def test_loader_with_018_fallback_overrides_both(module_name, class_name):
    loader_cls = getattr(importlib.import_module(module_name), class_name)
    assert loader_cls._build_decoder_layer_specs is not ModelLoader._build_decoder_layer_specs
    assert loader_cls.get_transformer_layer_spec is not ModelLoader.get_transformer_layer_spec


def test_glm_moe_dsa_swaps_shared_indexer_attention():
    """dsa_indexer_topk_freq > 1 must install the bridge's shared-indexer attention subclass."""
    from mcore_bridge.model.gpts.glm_moe_dsa import DSAttention, GlmMoeDsaLoader

    if getattr(DSAttention, '_HOLDER_ATTR', None) is None:
        pytest.skip('shared indexer requires megatron-core >= 0.19')
    config = _config()
    config.experimental_attention_variant = 'dsa'
    config.multi_latent_attention = True
    config.qk_layernorm = True
    config.dsa_indexer_topk_freq = 2
    loader = _loader(GlmMoeDsaLoader, config)
    core = loader.decoder_layer_specs()[0].submodules.self_attention.submodules.core_attention
    assert core.module is DSAttention


def test_non_mtp_empty_stage_returns_none_without_building_a_template():
    """Embedding-only stages must not build a full spec (or raise) just to find no template."""

    class HalfLoader(ModelLoader):

        def get_transformer_layer_spec(self, vp_stage=None):
            return TransformerBlockSubmodules(layer_specs=[])

    loader = _loader(HalfLoader, _config('E|t|t|mL', num_layers=2))
    with mock.patch.multiple(
            parallel_state,
            get_pipeline_model_parallel_rank=mock.Mock(return_value=0),
            get_virtual_pipeline_model_parallel_world_size=mock.Mock(return_value=2),
    ):
        assert loader.get_mtp_block_spec(TransformerBlockSubmodules(layer_specs=[]), vp_stage=0) is None


def test_mtp_only_stage_raises_only_when_the_stage_needs_mtp():
    """Simulated 0.18: the version error appears on MTP stages only, never on empty ones."""

    class NoLayerLevelLoader(ModelLoader):

        def _build_decoder_layer_specs(self):
            raise NotImplementedError('simulated 0.18: no layer-level entry')

        def get_transformer_layer_spec(self, vp_stage=None):
            return TransformerBlockSubmodules(layer_specs=[])

    loader = _loader(NoLayerLevelLoader, _config())
    with _rank1_last_stage():
        with pytest.raises(NotImplementedError, match='layer-level'):
            loader.get_mtp_block_spec(TransformerBlockSubmodules(layer_specs=[]), vp_stage=0)


def test_template_uses_the_last_layer_not_the_first():
    """Heterogeneous fixture: only an implementation taking layer 0 would pass the marker."""

    class MarkerAttention:
        """Never built in the passing path; it only makes layer 0 distinguishable."""

    class HeteroLoader(ModelLoader):

        def _build_decoder_layer_specs(self):
            layer_specs = [deepcopy(spec) for spec in self._plain_decoder_layer_specs()]
            layer_specs[0].submodules.self_attention.module = MarkerAttention
            return layer_specs

    loader = _loader(HeteroLoader, _config())
    with _rank1_last_stage():
        local = loader.get_transformer_layer_spec(vp_stage=0)
        mtp_block_spec = loader.get_mtp_block_spec(local, vp_stage=0)
    inner = mtp_block_spec.layer_specs[0].submodules.mtp_model_layer
    assert inner.submodules.self_attention.module is not MarkerAttention


def test_real_gdn_loader_mtp_only_uses_full_last_layer():
    """Real Qwen3.5 loader + MTP-only layout: the template must be the full list's last layer."""
    from mcore_bridge.model.gpts.qwen3_next import Qwen3NextLoader, Qwen3NextSelfAttention

    config = _config('Ett|t|t|mL', num_layers=4)
    config.hf_model_type = 'qwen3_next'
    config.linear_attention_freq = [1, 0, 1, 0]  # last layer is full attention
    config.hetereogenous_dist_checkpoint = False
    config.layernorm_epsilon = 1e-6
    config.rms_norm_eps = 1e-6
    config.hidden_act = 'silu'
    config.dtype = torch.float32
    config.params_dtype = torch.float32

    loader = _loader(Qwen3NextLoader, config)
    with mock.patch.multiple(
            parallel_state,
            get_pipeline_model_parallel_rank=mock.Mock(return_value=1),
            get_virtual_pipeline_model_parallel_world_size=mock.Mock(return_value=2),
    ):
        local = loader.get_transformer_layer_spec(vp_stage=1)
        assert local.layer_specs == []
        mtp_block_spec = loader.get_mtp_block_spec(local, vp_stage=1)
        full_last = loader.decoder_layer_specs()[-1]
    inner = mtp_block_spec.layer_specs[0].submodules.mtp_model_layer
    assert inner.submodules.self_attention.module is Qwen3NextSelfAttention
    assert inner.submodules.self_attention.module is full_last.submodules.self_attention.module


def test_mtp_only_stage_resolves_template_from_full_layer_list():
    """`Ettt|mL`: rank1 owns only MTP+loss; the local block is empty but MTP still builds."""
    loader = _loader(ModelLoader, _config())
    with _rank1_last_stage():
        local = loader.get_transformer_layer_spec(vp_stage=0)
        assert local.layer_specs == []
        mtp_block_spec = loader.get_mtp_block_spec(local, vp_stage=0)
    assert mtp_block_spec is not None
    assert len(mtp_block_spec.layer_specs) == 1


def test_mtp_template_carries_the_model_patch():
    """The template must be the patched layer, not upstream's vanilla TransformerLayer."""
    loader = _loader(ModelLoader, _config())
    with _rank1_last_stage():
        local = loader.get_transformer_layer_spec(vp_stage=0)
        mtp_block_spec = loader.get_mtp_block_spec(local, vp_stage=0)
    inner = mtp_block_spec.layer_specs[0].submodules.mtp_model_layer
    assert inner.module is BridgeTransformerLayer


def test_template_is_the_last_layer_of_the_full_list():
    """The local block never participates: the template mirrors decoder_layer_specs()[-1]."""
    loader = _loader(ModelLoader, _config())
    with _rank1_last_stage():
        local = loader.get_transformer_layer_spec(vp_stage=0)
        assert local.layer_specs == []
        mtp_block_spec = loader.get_mtp_block_spec(local, vp_stage=0)
    inner = mtp_block_spec.layer_specs[0].submodules.mtp_model_layer
    full_last = loader.decoder_layer_specs()[-1]
    assert inner.module is full_last.module
    assert inner.submodules.self_attention.module is full_last.submodules.self_attention.module


def test_half_overridden_loader_fails_loud():
    """A loader that only overrides the sliced entry point must not get a silent template."""

    class HalfLoader(ModelLoader):

        def get_transformer_layer_spec(self, vp_stage=None):
            return TransformerBlockSubmodules(layer_specs=[])

    loader = _loader(HalfLoader, _config())
    with _rank1_last_stage():
        with pytest.raises(NotImplementedError, match='_build_decoder_layer_specs'):
            loader.get_mtp_block_spec(TransformerBlockSubmodules(layer_specs=[]), vp_stage=0)
