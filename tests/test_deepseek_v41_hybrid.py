# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for the DeepSeek-V4.1 GPT-space -> HybridStack config derivation (B1a).

Pure logic, no GPU / distributed init required.
"""
import pytest  # noqa: E402

from mcore_bridge.model.gpts.deepseek_v41 import DeepseekV41HyperConnectionHybridLayer  # noqa: E402
from mcore_bridge.model.gpts.deepseek_v41 import (HybridLayerConfig, HyperConnectionHybridLayer,
                                                  derive_hybrid_layer_config)


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
    # The loader derives its own config copy so the caller's config is never mutated.
    from types import SimpleNamespace

    from mcore_bridge.model.gpts.deepseek_v41 import DeepseekV41Loader

    loader = object.__new__(DeepseekV41Loader)
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
    # MTP stays disabled (V4.1 ``mtp.*`` keys are DSpark).
    assert cfg.mtp_num_layers is None
    # caller config left untouched
    assert original.num_layers == 4
    assert original.mtp_num_layers == 1
    assert original.is_hybrid_model is False


def _make_bridge(engram_layer_ids, enable_hyper_connections):
    from types import SimpleNamespace

    from mcore_bridge.model.gpts.deepseek_v41 import DeepseekV41Bridge

    bridge = object.__new__(DeepseekV41Bridge)
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

    wrappers = {idx: SimpleNamespace(inner_layer=f'inner{idx}', hyper_connection=f'hc{idx}') for idx in range(4)}
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

    from mcore_bridge.model.gpts.deepseek_v41 import DeepseekV41Bridge, DeepseekV41Loader

    loader = object.__new__(DeepseekV41Loader)
    assert loader._engram_placement_layer_ids([1, 3]) == (3, 7)
    assert loader._engram_placement_layer_ids([0]) == (1, )

    bridge = object.__new__(DeepseekV41Bridge)
    for hf_id in (0, 1, 3, 10):
        layer_number = 2 * hf_id + 1
        assert bridge._engram_hf_layer_id(SimpleNamespace(layer_number=layer_number)) == hf_id


def test_num_hybrid_layers_normalizes_both_layer_spaces():
    # ``_convert`` sees ``self.config`` in two layer spaces. On load it is the GPT-space config
    # (num_layers == N, no pattern) whose built decoder holds 2*N layers; on export it is the
    # doubled hybrid config (num_layers == 2*N, pattern populated). Both must yield 2*N so the
    # loop matches the decoder's real layer count -- a regression guard for the export bug where
    # ``2 * num_layers`` on the doubled config over-counted and dereferenced None layers.
    from types import SimpleNamespace

    from mcore_bridge.model.gpts.deepseek_v41 import DeepseekV41Bridge

    load_cfg = SimpleNamespace(num_layers=4, hybrid_layer_pattern=None)
    export_cfg = SimpleNamespace(num_layers=8, hybrid_layer_pattern='DEDEDEDE')
    assert DeepseekV41Bridge._num_hybrid_layers(load_cfg) == 8
    assert DeepseekV41Bridge._num_hybrid_layers(export_cfg) == 8
    # A config missing the attribute entirely is treated as GPT-space (load).
    assert DeepseekV41Bridge._num_hybrid_layers(SimpleNamespace(num_layers=3)) == 6


requires_hybrid = pytest.mark.skipif(
    DeepseekV41HyperConnectionHybridLayer is None, reason='megatron hybrid stack not importable')


@requires_hybrid
def test_hc_wrapper_applies_engram_on_nstream_before_delegating():
    # The V4.1 wrapper subclass overrides ``forward``: for an Engram-carrying inner layer it adds
    # the n-stream Engram delta to ``hidden_states`` BEFORE delegating to the base wrapper forward
    # (aggregation + fast-path attention), so the delta lands on the pre-aggregation streams.
    # A plain inner layer delegates unchanged with no Engram add.
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch

    # Engram-carrying layer: a constant unit delta is added, so the tensor handed to the base
    # forward is the input plus that delta.
    def engram(hidden_states, input_ids, inference_context):
        return torch.ones_like(hidden_states)

    engram_layer = object.__new__(DeepseekV41HyperConnectionHybridLayer)
    engram_layer.inner_layer = SimpleNamespace(engram=engram)
    h = torch.zeros(2, 1, 4)
    ids = torch.zeros(2, 1, dtype=torch.long)
    with patch.object(HyperConnectionHybridLayer, 'forward', return_value='OUT') as base_fwd:
        assert engram_layer.forward(h, input_ids=ids) == 'OUT'
    passed = base_fwd.call_args.args[0]
    assert torch.equal(passed, torch.ones_like(h))  # delta added before delegating

    # Plain layer (no Engram): delegate unchanged, forwarding the original tensor untouched.
    plain_layer = object.__new__(DeepseekV41HyperConnectionHybridLayer)
    plain_layer.inner_layer = SimpleNamespace(engram=None)
    with patch.object(HyperConnectionHybridLayer, 'forward', return_value='OUT') as base_fwd:
        assert plain_layer.forward(h, input_ids=ids) == 'OUT'
    assert base_fwd.call_args.args[0] is h


@requires_hybrid
def test_hc_wrapper_requires_input_ids_for_engram_layer():
    # An Engram layer cannot run without token IDs (needed for the n-gram hash), so forward
    # raises rather than silently dropping the Engram contribution.
    import pytest
    import torch
    from types import SimpleNamespace

    engram_layer = object.__new__(DeepseekV41HyperConnectionHybridLayer)
    engram_layer.inner_layer = SimpleNamespace(engram=lambda *a, **k: 0)
    with pytest.raises(ValueError, match='input token IDs'):
        engram_layer.forward(torch.zeros(2, 1, 4), input_ids=None)


@requires_hybrid
def test_rewrap_swaps_class_only_on_engram_wrappers():
    # Post-build retrofit: only wrappers whose inner layer built an Engram module get the
    # subclass; every other wrapper keeps the base class (and its fast path).
    from types import SimpleNamespace

    from mcore_bridge.model.gpts.deepseek_v41 import DeepseekV41Loader

    engram_wrapper = object.__new__(HyperConnectionHybridLayer)
    engram_wrapper.inner_layer = SimpleNamespace(engram=object())
    plain_wrapper = object.__new__(HyperConnectionHybridLayer)
    plain_wrapper.inner_layer = SimpleNamespace(engram=None)
    model = SimpleNamespace(decoder=SimpleNamespace(layers=[engram_wrapper, plain_wrapper]))

    loader = object.__new__(DeepseekV41Loader)
    loader.config = SimpleNamespace(enable_hyper_connections=True)
    loader._rewrap_engram_hyper_connection_layers(model)

    assert type(engram_wrapper) is DeepseekV41HyperConnectionHybridLayer
    assert type(plain_wrapper) is HyperConnectionHybridLayer


@requires_hybrid
def test_rewrap_noop_without_hyper_connections():
    # No wrapping happens at all when hyper-connections are off, so nothing to retrofit.
    from types import SimpleNamespace

    from mcore_bridge.model.gpts.deepseek_v41 import DeepseekV41Loader

    engram_wrapper = object.__new__(HyperConnectionHybridLayer)
    engram_wrapper.inner_layer = SimpleNamespace(engram=object())
    model = SimpleNamespace(decoder=SimpleNamespace(layers=[engram_wrapper]))

    loader = object.__new__(DeepseekV41Loader)
    loader.config = SimpleNamespace(enable_hyper_connections=False)
    loader._rewrap_engram_hyper_connection_layers(model)

    assert type(engram_wrapper) is HyperConnectionHybridLayer


from mcore_bridge.model.gpts.deepseek_v41 import DeepseekV41HybridStackModel  # noqa: E402


def _seg_config(pattern, **overrides):
    from types import SimpleNamespace
    cfg = dict(
        hybrid_layer_pattern=pattern,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        num_layers_in_first_pipeline_stage=None,
        num_layers_in_last_pipeline_stage=None,
        pipeline_model_parallel_layout=None,
        mtp_num_layers=None,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


@requires_hybrid
def test_segment_main_pattern_block_aligned_even_split():
    # 4 blocks over pp=2 -> two whole blocks per stage.
    cfg = _seg_config('DEDEDEDE', pipeline_model_parallel_size=2)
    assert DeepseekV41HybridStackModel._segment_main_pattern(cfg) == 'DEDE|DEDE'


@requires_hybrid
def test_segment_main_pattern_uneven_split_front_loads_extra_blocks():
    # 4 blocks over pp=3 -> divmod(4,3)=(1,1): first stage gets 2 blocks, the rest 1 each.
    cfg = _seg_config('DEDEDEDE', pipeline_model_parallel_size=3)
    assert DeepseekV41HybridStackModel._segment_main_pattern(cfg) == 'DEDE|DE|DE'


@requires_hybrid
def test_segment_main_pattern_vpp_multiplies_stages():
    # pp=2 * vp=2 = 4 stages, consecutive segments ordered (vp0,pp0),(vp0,pp1),(vp1,pp0),(vp1,pp1)
    # to match upstream segment_index = vp_stage * pp_size + pp_rank.
    cfg = _seg_config('DEDEDEDE', pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2)
    assert DeepseekV41HybridStackModel._segment_main_pattern(cfg) == 'DE|DE|DE|DE'


@requires_hybrid
def test_segment_main_pattern_pp1_is_noop():
    cfg = _seg_config('DEDEDEDE', pipeline_model_parallel_size=1)
    assert DeepseekV41HybridStackModel._segment_main_pattern(cfg) == 'DEDEDEDE'


@requires_hybrid
def test_segment_main_pattern_respects_explicit_pipes_and_uneven_layout():
    # An explicit '|' layout or num_layers_in_first/last_pipeline_stage is passed through
    # untouched; upstream + the post-build even-boundary guard validate it.
    assert DeepseekV41HybridStackModel._segment_main_pattern(_seg_config('DEDE|DEDE',
                                                                         pipeline_model_parallel_size=2)) == 'DEDE|DEDE'
    assert DeepseekV41HybridStackModel._segment_main_pattern(
        _seg_config('DEDEDEDE', pipeline_model_parallel_size=2, num_layers_in_first_pipeline_stage=2)) == 'DEDEDEDE'


@requires_hybrid
def test_segment_main_pattern_raises_when_stage_gets_no_block():
    import pytest

    # 2 blocks cannot cover 4 stages.
    with pytest.raises(ValueError, match='at least one attention'):
        DeepseekV41HybridStackModel._segment_main_pattern(_seg_config('DEDE', pipeline_model_parallel_size=4))


@requires_hybrid
def test_segment_main_pattern_rejects_pipeline_layout():
    import pytest
    with pytest.raises(ValueError, match='pipeline_model_parallel_layout'):
        DeepseekV41HybridStackModel._segment_main_pattern(
            _seg_config('DEDEDEDE', pipeline_model_parallel_size=2, pipeline_model_parallel_layout=[[0], [1]]))


@requires_hybrid
def test_resolve_hybrid_layer_pattern_segments_then_defers_to_base():
    # With MTP off the resolver just returns the block-segmented main pattern; pp=1 returns the
    # bare pattern (base resolver, no MTP suffix appended).
    assert DeepseekV41HybridStackModel._resolve_hybrid_layer_pattern(
        _seg_config('DEDEDEDE', pipeline_model_parallel_size=2)) == 'DEDE|DEDE'
    assert DeepseekV41HybridStackModel._resolve_hybrid_layer_pattern(
        _seg_config('DEDEDEDE', pipeline_model_parallel_size=1)) == 'DEDEDEDE'


from mcore_bridge.model.gpts.deepseek_v41 import DeepseekV41Bridge, DeepseekV41Loader  # noqa: E402

# --- DSpark (``mtp.*``) draft stack -------------------------------------------------------------


def _dspark_bridge(dspark_num_layers=1):
    # object.__new__ so no distributed init; only the DSpark dispatch fields are needed. Record
    # calls into the shared ``_convert_dspark_stack`` so we assert dispatch + stage guards without
    # building a real stack (that is the GPU acceptance step).
    from types import SimpleNamespace

    bridge = object.__new__(DeepseekV41Bridge)
    bridge.config = SimpleNamespace(dspark_num_layers=dspark_num_layers)
    calls = []
    bridge._convert_dspark_stack = (lambda language_model, dspark, hf_state_dict, hf_prefix, to_mcore: (calls.append(
        (language_model, dspark)) or iter(['SENTINEL'])))
    return bridge, calls


def test_hybrid_convert_additional_layers_maps_dspark_via_lm():
    # The hybrid model is text-only (no ``language_model`` wrapper), so ``_lm`` resolves the model
    # itself; the DSpark stack attached there is mapped through the shared base helper.
    from types import SimpleNamespace

    bridge, calls = _dspark_bridge()
    dspark = object()
    mg_model = SimpleNamespace(dspark=dspark)  # no ``language_model`` -> _lm returns mg_model
    out = list(bridge._convert_additional_layers(mg_model, {}, 'prefix.', to_mcore=True, is_pp_last_stage=True))
    assert out == ['SENTINEL']
    assert calls == [(mg_model, dspark)]


def test_hybrid_convert_additional_layers_resolves_language_model_wrapper():
    # When a multimodal wrapper is present, ``_lm`` unwraps it and the
    # DSpark stack is looked up on the nested language model.
    from types import SimpleNamespace

    bridge, calls = _dspark_bridge()
    dspark = object()
    language_model = SimpleNamespace(dspark=dspark)
    mg_model = SimpleNamespace(language_model=language_model)
    out = list(bridge._convert_additional_layers(mg_model, {}, 'prefix.', to_mcore=True, is_pp_last_stage=True))
    assert out == ['SENTINEL']
    assert calls == [(language_model, dspark)]


def test_hybrid_convert_additional_layers_skips_without_dspark():
    # No draft stack configured -> nothing to convert, base helper untouched.
    from types import SimpleNamespace

    bridge, calls = _dspark_bridge(dspark_num_layers=0)
    mg_model = SimpleNamespace(dspark=object())
    out = list(bridge._convert_additional_layers(mg_model, {}, 'prefix.', to_mcore=True, is_pp_last_stage=True))
    assert out == []
    assert calls == []


def test_hybrid_convert_additional_layers_load_skips_non_last_stage():
    # On load only the final stage owns the stack; earlier stages are guarded out.
    from types import SimpleNamespace

    bridge, calls = _dspark_bridge()
    mg_model = SimpleNamespace()  # no ``dspark`` on this stage
    out = list(bridge._convert_additional_layers(mg_model, {}, 'prefix.', to_mcore=True, is_pp_last_stage=False))
    assert out == []
    assert calls == []


def test_hybrid_convert_additional_layers_export_uses_proxy_on_non_last_stage():
    # On export a non-last PP stage has no draft stack, but it must still issue the same collective
    # sequence as the final stage, so an empty structural proxy with the configured layer count
    # stands in (skipping quietly would desynchronize the pp group).
    from types import SimpleNamespace

    bridge, calls = _dspark_bridge()
    mg_model = SimpleNamespace()  # no ``dspark``
    out = list(bridge._convert_additional_layers(mg_model, {}, 'prefix.', to_mcore=False, is_pp_last_stage=False))
    assert out == ['SENTINEL']
    (language_model, dspark), = calls
    assert language_model is mg_model
    assert dspark.layers == [None] * bridge.config.dspark_num_layers


def test_hybrid_convert_additional_layers_raises_on_last_stage_without_stack():
    import pytest
    from types import SimpleNamespace

    bridge, _ = _dspark_bridge()
    mg_model = SimpleNamespace()  # last stage but stack missing -> misbuilt model
    with pytest.raises(RuntimeError, match='DSpark weights require the draft stack'):
        list(bridge._convert_additional_layers(mg_model, {}, 'prefix.', to_mcore=False, is_pp_last_stage=True))


# --- multimodal wrapper hosting the hybrid backbone ---------------------------------------------


def test_multimodal_wrapper_hosts_hybrid_backbone():
    # The wrapper is the stock multimodal model with the language-model class swapped for the
    # PP-capable hybrid backbone; everything else (vision tower, image-embed injection) is inherited.
    from mcore_bridge.model.gpts.deepseek_v41 import DeepseekV41HybridStackModel, DeepseekV41MultimodalModel
    from mcore_bridge.model.mm_gpt_model import MultimodalGPTModel
    assert issubclass(DeepseekV41MultimodalModel, MultimodalGPTModel)
    assert DeepseekV41MultimodalModel.language_model_cls is DeepseekV41HybridStackModel


def test_hybrid_stack_exposes_extra_forward_keys():
    # ``MultimodalGPTModel.forward`` reads ``language_model.extra_forward_keys``; the hybrid backbone
    # must expose the same empty default the GPT ``GPTModel`` carries.
    assert DeepseekV41HybridStackModel.extra_forward_keys == []


def test_hybrid_loader_model_cls_is_multimodal_wrapper():
    from mcore_bridge.model.gpts.deepseek_v41 import DeepseekV41MultimodalModel
    assert DeepseekV41Loader.model_cls is DeepseekV41MultimodalModel


def _pre_process_bridge(monkeypatch):
    # Stub the base implementation (word embeddings + config-guarded vision/aligner block) and the
    # marker writes so we observe delegation + ordering without a real model.
    from mcore_bridge.bridge.gpt_bridge import GPTBridge

    called, markers = [], []
    monkeypatch.setattr(GPTBridge, '_convert_pre_process', lambda self, mg, sd, pfx, tm: called.append(
        (mg, pfx, tm)) or {'SUPER': True})
    bridge = object.__new__(DeepseekV41Bridge)
    bridge._set_state_dict = (lambda mod, mkey, sd, hkey, tm: markers.append((mod, mkey, sd, hkey, tm)))
    return bridge, called, markers


def test_pre_process_delegates_to_base_and_maps_image_markers(monkeypatch):
    # First PP stage of a multimodal model: delegate to the base bridge, then map the three
    # ``image_*`` markers, which live on the wrapper (not the backbone).
    from types import SimpleNamespace

    bridge, called, markers = _pre_process_bridge(monkeypatch)
    mg_model = SimpleNamespace(visual=object())
    hf_state_dict = {'x': 1}
    out = bridge._convert_pre_process(mg_model, hf_state_dict, '', to_mcore=True)
    assert out == {'SUPER': True}
    assert called == [(mg_model, '', True)]
    # On load the markers are read out of the incoming HF dict.
    assert markers == [(mg_model, f'visual.{name}', hf_state_dict, f'model.{name}', True)
                       for name in ('image_start', 'image_end', 'image_newline')]


def test_pre_process_runs_vision_block_even_without_visual(monkeypatch):
    # A non-first PP stage builds ``visual=None``, but the vision block must still run: every rank
    # has to issue the same pp-group collectives, otherwise the last stage's per-layer ``has_model``
    # all-reduce reads a stale value and ``_convert`` raises ``StopIteration``.
    from types import SimpleNamespace

    bridge, called, markers = _pre_process_bridge(monkeypatch)
    mg_model = SimpleNamespace(visual=None)
    out = bridge._convert_pre_process(mg_model, {'x': 1}, '', to_mcore=False)
    assert out == {'SUPER': True}
    assert called == [(mg_model, '', False)]
    # On export the markers are written into the dict the base call returned.
    assert [(item[1], item[2] is out, item[4]) for item in markers
            ] == [(f'visual.{name}', True, False) for name in ('image_start', 'image_end', 'image_newline')]


def test_hybrid_set_word_embeddings_resolves_via_lm():
    # ``_set_word_embeddings`` must resolve the LM through ``_lm`` so both the wrapper and a bare
    # backbone map ``embedding.word_embeddings.weight`` onto the right module.
    from types import SimpleNamespace

    bridge = object.__new__(DeepseekV41Bridge)
    bridge.hf_embed_key = 'model.embed_tokens.weight'
    recorded = []
    bridge._set_state_dict = lambda mod, mkey, sd, hkey, tm: recorded.append((mod, mkey, hkey, tm))

    language_model = SimpleNamespace(tag='lm')
    bridge._set_word_embeddings(SimpleNamespace(language_model=language_model), {}, to_mcore=True)
    bare = SimpleNamespace()  # no wrapper -> _lm returns the model itself
    bridge._set_word_embeddings(bare, {}, to_mcore=False)
    assert recorded == [
        (language_model, 'embedding.word_embeddings.weight', 'model.embed_tokens.weight', True),
        (bare, 'embedding.word_embeddings.weight', 'model.embed_tokens.weight', False),
    ]


@requires_hybrid
def test_hybrid_forward_unpacks_extra_block_kwargs(monkeypatch):
    # ``MultimodalGPTModel.forward`` funnels the decoder's extra kwargs through ``extra_block_kwargs``
    # (the GPTModel calling convention), but upstream ``HybridModel.forward`` has no such parameter --
    # it threads ``input_ids`` itself. The hybrid stack must unpack that container before delegating,
    # strip visual keys, and forward anything else, otherwise a text-only wrapper run raises
    # ``HybridModel.forward() got an unexpected keyword argument 'extra_block_kwargs'``.
    from mcore_bridge.model.gpts import deepseek_v41 as hyb

    received = {}
    monkeypatch.setattr(hyb.HybridModel, 'forward', lambda self, *a, **k: received.update(args=a, kwargs=k) or 'OUT')
    stack = object.__new__(DeepseekV41HybridStackModel)  # no distributed init; forward is self-contained
    out = DeepseekV41HybridStackModel.forward(
        stack, input_ids=1, extra_block_kwargs={
            'image_grid_thw': 7,
            'foo': 'bar'
        })
    assert out == 'OUT'
    kwargs = received['kwargs']
    assert 'extra_block_kwargs' not in kwargs  # container unpacked, not forwarded verbatim
    assert 'image_grid_thw' not in kwargs  # visual key stripped
    assert kwargs['foo'] == 'bar'  # unknown extra kwarg still threaded through
    assert kwargs['input_ids'] == 1


@requires_hybrid
def test_hybrid_forward_rejects_pixel_values_from_extra_block_kwargs(monkeypatch):
    # Defense in depth: a multimodal batch that smuggles ``pixel_values`` via ``extra_block_kwargs``
    # must still hit the text-only guard (the wrapper injects image embeds and clears them, so the
    # backbone never legitimately sees pixels).
    from mcore_bridge.model.gpts import deepseek_v41 as hyb

    monkeypatch.setattr(hyb.HybridModel, 'forward', lambda self, *a, **k: 'OUT')
    stack = object.__new__(DeepseekV41HybridStackModel)
    with pytest.raises(NotImplementedError, match='text-only'):
        DeepseekV41HybridStackModel.forward(stack, input_ids=1, extra_block_kwargs={'pixel_values': 1})
