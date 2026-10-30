# Copyright (c) ModelScope Contributors. All rights reserved.
"""Contract tests for the upstream-MTP adaptation layer.

Only the properties that would regress silently are locked here: the upstream
block must stay unpatched, model-specific layer specs must survive adaptation,
the construction-time class patch must be restored, and the loss/extras hooks
must keep their documented behaviour.
"""
import torch
from types import SimpleNamespace

import mcore_bridge  # noqa: F401  (ensures apply_patch ran at import)
from mcore_bridge.model import mtp as mtp_mod
from mcore_bridge.model.mtp import (MultiTokenPredictionBlock, MultiTokenPredictionLayer, adapt_mtp_block_spec,
                                    get_mtp_capabilities, prepare_mtp_loss_mask)


def test_import_does_not_globally_patch_upstream_block():
    """The refactor must NOT monkey-patch the upstream block forward."""
    from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionBlock as McoreBlock
    assert McoreBlock.forward.__module__ == 'megatron.core.transformer.multi_token_prediction', McoreBlock.forward


def test_adapt_mtp_block_spec_preserves_model_specific_layer():
    """A layer already specialized by a model must not be replaced by the adapter."""

    class CustomLayer:
        pass

    layer_spec = SimpleNamespace(module=CustomLayer)
    adapt_mtp_block_spec(SimpleNamespace(), SimpleNamespace(layer_specs=[layer_spec]))
    assert layer_spec.module is CustomLayer


def test_prepare_mtp_loss_mask_merges_ignore_index():
    labels = torch.tensor([[5, 6, -100, 7], [8, -100, 9, 10]])
    # no incoming mask -> only -100 filtered
    mask = prepare_mtp_loss_mask(labels, None)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [[True, True, False, True], [True, False, True, True]]
    # incoming mask is AND-ed with the -100 filter
    incoming = torch.tensor([[True, True, True, False], [True, True, True, True]])
    merged = prepare_mtp_loss_mask(labels, incoming)
    assert merged.tolist() == [[True, True, False, False], [True, False, True, True]]
    # labels=None -> mask returned untouched
    assert prepare_mtp_loss_mask(None, incoming) is incoming


def test_mtp_block_patch_restores_after_build():
    """A leaked class patch would make later model builds depend on test order."""
    from megatron.core.models.gpt import gpt_model

    from mcore_bridge.model.register import ModelLoader

    before = gpt_model.MultiTokenPredictionBlock
    loader = SimpleNamespace(config=SimpleNamespace(mtp_num_layers=1))
    with ModelLoader._patch_mtp_block(loader):
        assert gpt_model.MultiTokenPredictionBlock is MultiTokenPredictionBlock
    assert gpt_model.MultiTokenPredictionBlock is before


def test_mtp_block_patch_noop_without_mtp():
    from megatron.core.models.gpt import gpt_model

    from mcore_bridge.model.register import ModelLoader

    before = gpt_model.MultiTokenPredictionBlock
    loader = SimpleNamespace(config=SimpleNamespace(mtp_num_layers=None))
    with ModelLoader._patch_mtp_block(loader):
        assert gpt_model.MultiTokenPredictionBlock is before
    assert gpt_model.MultiTokenPredictionBlock is before


def test_get_embeddings_preserves_upstream_tuple_shape(monkeypatch):
    """The override must mirror 5/6-value upstream returns, including mtp_input_mask."""

    class _EmbeddingLayer(MultiTokenPredictionLayer):

        def __init__(self):
            torch.nn.Module.__init__(self)
            self.config = SimpleNamespace(
                sequence_parallel=False,
                tensor_model_parallel_size=1,
                mtp_decoder_input_detach=False,
                mtp_detach_heads=False,
            )
            self._pending_decoder_input = None
            self._pending_depth = 0

    layer = _EmbeddingLayer()
    ids = torch.zeros(1, 2, dtype=torch.long)
    dec = torch.zeros(1, 2, 4)
    hs = torch.zeros(1, 2, 4)
    pmask = torch.zeros(1, 2, dtype=torch.bool)
    mtp_mask = torch.ones(1, 2, dtype=torch.bool)

    for result in ((ids, ids, pmask, dec, hs), (ids, ids, pmask, mtp_mask, dec, hs)):
        monkeypatch.setattr(
            mtp_mod.McoreMultiTokenPredictionLayer, '_get_embeddings', lambda *args, _result=result, **kwargs: _result)
        out = layer._get_embeddings(input_ids=ids, position_ids=ids)
        assert len(out) == len(result)
        assert torch.equal(out[-2], dec)
        if len(result) == 6:
            assert out[3] is mtp_mask


def test_get_inner_layer_kwargs_routes_into_extras(monkeypatch):
    """Model hooks feed the inner transformer layer through the staged extras channel."""
    captured = {}

    def fake_upstream_forward(self, *args, **kwargs):
        captured['extras'] = dict(self._pending_extras or {})
        return 'ok'

    class _HookLayer(MultiTokenPredictionLayer):

        def __init__(self):
            torch.nn.Module.__init__(self)

        def _get_inner_layer_kwargs(self, input_ids, position_ids):
            return {'per_depth_ids': input_ids}

    # Warm the capability cache BEFORE replacing the upstream forward: the probe
    # reads its signature, and caching a fake `(*args, **kwargs)` signature would
    # poison every later model build in the same process.
    get_mtp_capabilities()
    monkeypatch.setattr(mtp_mod.McoreMultiTokenPredictionLayer, 'forward', fake_upstream_forward)

    layer = _HookLayer()
    layer.__dict__.update(_bridge_depth=0, _pending_decoder_input=None, _pending_extras=None, _pending_depth=0)
    ids = torch.zeros(1, 2, dtype=torch.long)
    out = layer(input_ids=ids, position_ids=ids, hidden_states=torch.zeros(1, 1, 4))
    assert out == 'ok'
    assert torch.equal(captured['extras']['per_depth_ids'], ids)
