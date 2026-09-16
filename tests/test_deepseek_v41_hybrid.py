# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for the DeepSeek-V4.1 GPT-space -> HybridStack config derivation (B1a).

Pure logic, no GPU / distributed init required.
"""
from mcore_bridge.model.gpts.deepseek_v41_hybrid import HybridLayerConfig, derive_hybrid_layer_config


def test_tiny_all_moe_zero_ratio():
    # tiny checkpoint: 4 layers, sliding-window only (ratio 0), every layer MoE.
    r = derive_hybrid_layer_config(4, [0, 0, 0, 0], [1, 1, 1, 1])
    assert isinstance(r, HybridLayerConfig)
    assert r.hybrid_layer_pattern == 'DEDEDEDE'
    assert r.num_layers == 8
    assert r.csa_compress_ratios == [0] * 8
    assert r.csa2_kv_source_layers == []
    assert r.csa2_index_source_layers == []
    assert r.csa2_candidate_source_layer is None


def test_matches_csa2_pipeline_reference_layout():
    # The upstream test_csa2_pipeline._config bakes the *hybrid* ratios directly as
    # [0,0,2,0,2,0,1,0,1,0,1,0] (even slots = real per-attention ratio). Deriving from the
    # GPT-space 6-layer config must reproduce that exact 2x layout, and re-map the source
    # layers from GPT index j to hybrid attention index 2*j.
    r = derive_hybrid_layer_config(
        6,
        [0, 2, 2, 1, 1, 1],
        [1] * 6,
        csa2_kv_source_layers=[1, 3],
        csa2_index_source_layers=[2],
        csa2_candidate_source_layer=1,
    )
    assert r.csa_compress_ratios == [0, 0, 2, 0, 2, 0, 1, 0, 1, 0, 1, 0]
    assert r.hybrid_layer_pattern == 'DEDEDEDEDEDE'
    assert r.csa2_kv_source_layers == [2, 6]
    assert r.csa2_index_source_layers == [4]
    assert r.csa2_candidate_source_layer == 2


def test_dense_prefix_from_first_k_dense_replace():
    # first_k_dense_replace=1 -> moe_layer_freq [0,1,1,1]; the first MLP slot is dense '-'.
    r = derive_hybrid_layer_config(4, [0, 0, 0, 0], [0, 1, 1, 1])
    assert r.hybrid_layer_pattern == 'D-DEDEDE'


def test_int_moe_layer_freq_convention():
    # int N -> layer i is MoE iff i % N == 0 (megatron-core moe_logging convention).
    r = derive_hybrid_layer_config(4, [0] * 4, 2)
    assert r.hybrid_layer_pattern == 'DED-DED-'


def test_no_experts_all_dense():
    r = derive_hybrid_layer_config(3, [0, 0, 0], None)
    assert r.hybrid_layer_pattern == 'D-D-D-'
    assert r.num_layers == 6


def test_ratio_length_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        derive_hybrid_layer_config(4, [0, 0, 0], [1, 1, 1, 1])
    with pytest.raises(ValueError):
        derive_hybrid_layer_config(4, [0, 0, 0, 0], [1, 1, 1])


def test_loader_build_hybrid_config_does_not_mutate_original():
    # The loader derives its own config copy so the golden GPT path stays intact.
    from types import SimpleNamespace

    from mcore_bridge.model.gpts.deepseek_v41_hybrid import DeepseekV41HybridLoader

    loader = object.__new__(DeepseekV41HybridLoader)
    loader.config = SimpleNamespace(
        num_layers=4,
        csa_compress_ratios=[0, 0, 0, 0],
        moe_layer_freq=[1, 1, 1, 1],
        csa2_kv_source_layers=[],
        csa2_index_source_layers=[],
        csa2_candidate_source_layer=None,
        mtp_num_layers=1,
        is_hybrid_model=False,
        hybrid_layer_pattern=None,
    )
    original = loader.config
    cfg = loader._build_hybrid_config()

    assert cfg is not original
    assert cfg.num_layers == 8
    assert cfg.hybrid_layer_pattern == 'DEDEDEDE'
    assert cfg.csa_compress_ratios == [0] * 8
    assert cfg.moe_layer_freq == [0, 1, 0, 1, 0, 1, 0, 1]
    assert cfg.is_hybrid_model is True
    # MTP disabled for the B1 backbone-only path.
    assert cfg.mtp_num_layers is None
    # golden GPT config left untouched
    assert original.num_layers == 4
    assert original.mtp_num_layers == 1
    assert original.is_hybrid_model is False


def _make_bridge(engram_layer_ids, enable_hyper_connections):
    from types import SimpleNamespace

    from mcore_bridge.model.gpts.deepseek_v41_hybrid import DeepseekV41HybridBridge

    bridge = object.__new__(DeepseekV41HybridBridge)
    bridge.config = SimpleNamespace(
        engram_layer_ids=engram_layer_ids, enable_hyper_connections=enable_hyper_connections)
    return bridge


def test_hybrid_layer_state_fans_out_attn_and_mlp():
    # A single HF layer i must fan out to hybrid layer 2*i (attention half) and 2*i+1 (MLP half),
    # each unwrapping ``inner_layer`` and routing the wrapper's single hyper-connection to the
    # matching hc channel. Engram fires only on the attention half of an engram layer.
    from types import SimpleNamespace

    bridge = _make_bridge(engram_layer_ids=[1], enable_hyper_connections=True)
    calls = []
    bridge._set_layer_attn = lambda inner, local, hf_idx, to_mcore: calls.append(('attn', inner, hf_idx)) or {}
    bridge._set_layer_mlp = lambda inner, local, hf_idx, to_mcore: calls.append(('mlp', inner, hf_idx)) or {}
    bridge._set_one_hyper_connection = lambda hc, local, hf_key, to_mcore: calls.append(('hc', hf_key, hc))
    bridge._set_layer_engram = lambda mg_layer, local, to_mcore: calls.append(('engram', mg_layer))

    wrappers = {
        idx: SimpleNamespace(inner_layer=f'inner{idx}', hyper_connection=f'hc{idx}')
        for idx in range(4)
    }
    for idx in range(4):
        res = bridge._set_hybrid_layer_state(wrappers[idx], {}, 'model.layers.', idx, to_mcore=False)
        assert isinstance(res, dict)

    assert ('attn', 'inner0', 0) in calls
    assert ('mlp', 'inner1', 0) in calls
    assert ('attn', 'inner2', 1) in calls
    assert ('mlp', 'inner3', 1) in calls
    # each wrapper's single hyper-connection maps to attn (even) / ffn (odd)
    assert ('hc', 'attn', 'hc0') in calls
    assert ('hc', 'ffn', 'hc1') in calls
    assert ('hc', 'attn', 'hc2') in calls
    assert ('hc', 'ffn', 'hc3') in calls
    # engram only on the attention half of HF layer 1
    engram_calls = [c for c in calls if c[0] == 'engram']
    assert len(engram_calls) == 1
    assert engram_calls[0][1] is wrappers[2]


def test_hybrid_layer_state_without_wrapper_uses_layer_directly():
    # enable_hyper_connections=False: no wrapper, so the payload is the layer itself and no
    # hyper-connection channel is written.
    bridge = _make_bridge(engram_layer_ids=[], enable_hyper_connections=False)
    seen = {}
    bridge._set_layer_attn = lambda inner, local, hf_idx, to_mcore: seen.update(attn_inner=inner) or {}
    bridge._set_one_hyper_connection = lambda *a, **k: seen.update(hc=True)

    plain_layer = object()  # no ``inner_layer`` attribute
    bridge._set_hybrid_layer_state(plain_layer, {}, 'model.layers.', 0, to_mcore=False)
    assert seen['attn_inner'] is plain_layer
    assert 'hc' not in seen


def test_hybrid_layer_state_prefixes_by_hf_index_on_export():
    # Both halves of HF layer i write under ``model.layers.{i}.`` so the exported checkpoint
    # keeps one layer per index.
    bridge = _make_bridge(engram_layer_ids=[], enable_hyper_connections=False)
    bridge._set_layer_mlp = lambda inner, local, hf_idx, to_mcore: {'ffn.w1.weight': 1}

    res = bridge._set_hybrid_layer_state(object(), {}, 'model.layers.', 5, to_mcore=False)
    assert res == {'model.layers.2.ffn.w1.weight': 1}


def test_engram_placement_and_hf_layer_id_round_trip():
    # HF layer e -> attention hybrid layer_number 2*e + 1; the bridge maps it back to e so the
    # engram_num_embeddings validation (keyed by 0-based HF ids) still resolves.
    from types import SimpleNamespace

    from mcore_bridge.model.gpts.deepseek_v41_hybrid import DeepseekV41HybridBridge, DeepseekV41HybridLoader

    loader = object.__new__(DeepseekV41HybridLoader)
    assert loader._engram_placement_layer_ids([1, 3]) == (3, 7)
    assert loader._engram_placement_layer_ids([0]) == (1,)

    bridge = object.__new__(DeepseekV41HybridBridge)
    for hf_id in (0, 1, 3, 10):
        layer_number = 2 * hf_id + 1
        assert bridge._engram_hf_layer_id(SimpleNamespace(layer_number=layer_number)) == hf_id


import pytest  # noqa: E402

from mcore_bridge.model.gpts.deepseek_v41_hybrid import (  # noqa: E402
    DeepseekV41HyperConnectionHybridLayer, HyperConnectionHybridLayer)

requires_hybrid = pytest.mark.skipif(
    DeepseekV41HyperConnectionHybridLayer is None, reason='megatron hybrid stack not importable')


@requires_hybrid
def test_hc_wrapper_declines_fast_path_only_for_engram_layers():
    # The V4.1 wrapper subclass returns None (forcing the full-forward `_call_inner_layer`
    # path, which applies Engram) iff the inner layer carries an Engram module; otherwise it
    # must delegate unchanged to the base fast path.
    from types import SimpleNamespace
    from unittest.mock import patch

    engram_layer = object.__new__(DeepseekV41HyperConnectionHybridLayer)
    engram_layer.inner_layer = SimpleNamespace(engram=object())
    assert engram_layer._call_inner_transformer_layer_without_local_bda('h', 'mask') is None

    plain_layer = object.__new__(DeepseekV41HyperConnectionHybridLayer)
    plain_layer.inner_layer = SimpleNamespace(engram=None)
    sentinel = object()
    with patch.object(
            HyperConnectionHybridLayer,
            '_call_inner_transformer_layer_without_local_bda',
            return_value=sentinel) as base_call:
        assert plain_layer._call_inner_transformer_layer_without_local_bda('h', 'mask') is sentinel
        base_call.assert_called_once()


@requires_hybrid
def test_rewrap_swaps_class_only_on_engram_wrappers():
    # Post-build retrofit: only wrappers whose inner layer built an Engram module get the
    # subclass; every other wrapper keeps the base class (and its fast path).
    from types import SimpleNamespace

    from mcore_bridge.model.gpts.deepseek_v41_hybrid import DeepseekV41HybridLoader

    engram_wrapper = object.__new__(HyperConnectionHybridLayer)
    engram_wrapper.inner_layer = SimpleNamespace(engram=object())
    plain_wrapper = object.__new__(HyperConnectionHybridLayer)
    plain_wrapper.inner_layer = SimpleNamespace(engram=None)
    model = SimpleNamespace(decoder=SimpleNamespace(layers=[engram_wrapper, plain_wrapper]))

    loader = object.__new__(DeepseekV41HybridLoader)
    loader.config = SimpleNamespace(enable_hyper_connections=True)
    loader._rewrap_engram_hyper_connection_layers(model)

    assert type(engram_wrapper) is DeepseekV41HyperConnectionHybridLayer
    assert type(plain_wrapper) is HyperConnectionHybridLayer


@requires_hybrid
def test_rewrap_noop_without_hyper_connections():
    # No wrapping happens at all when hyper-connections are off, so nothing to retrofit.
    from types import SimpleNamespace

    from mcore_bridge.model.gpts.deepseek_v41_hybrid import DeepseekV41HybridLoader

    engram_wrapper = object.__new__(HyperConnectionHybridLayer)
    engram_wrapper.inner_layer = SimpleNamespace(engram=object())
    model = SimpleNamespace(decoder=SimpleNamespace(layers=[engram_wrapper]))

    loader = object.__new__(DeepseekV41HybridLoader)
    loader.config = SimpleNamespace(enable_hyper_connections=False)
    loader._rewrap_engram_hyper_connection_layers(model)

    assert type(engram_wrapper) is HyperConnectionHybridLayer

