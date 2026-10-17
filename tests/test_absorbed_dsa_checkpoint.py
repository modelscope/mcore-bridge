# Copyright (c) ModelScope Contributors. All rights reserved.
"""Selective recompute of absorbed DSA must forward x and qr as keywords."""
import importlib.util
import sys
import torch
import types
from pathlib import Path


def _load_absorbed():
    root = Path(__file__).resolve().parents[1] / 'src' / 'mcore_bridge' / 'model' / 'modules' / 'absorbed_mla.py'

    def module(name):
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    megatron = module('megatron')
    core = module('megatron.core')
    megatron.core = core

    def checkpoint(function, distribute_saved_activations, *args):
        return function(*args)

    core.tensor_parallel = types.SimpleNamespace(checkpoint=checkpoint)
    rope = module('megatron.core.models.common.embeddings.rope_utils')
    rope.apply_rotary_pos_emb = lambda *args, **kwargs: None
    for name in (
            'megatron.core.models',
            'megatron.core.models.common',
            'megatron.core.models.common.embeddings',
            'megatron.core.tensor_parallel',
            'megatron.core.transformer',
            'megatron.core.transformer.experimental_attention_variant',
    ):
        module(name)
    mappings = module('megatron.core.tensor_parallel.mappings')
    mappings.gather_from_sequence_parallel_region = lambda *args, **kwargs: None
    mappings.gather_from_tensor_model_parallel_region = lambda *args, **kwargs: None
    mappings.scatter_to_sequence_parallel_region = lambda *args, **kwargs: None
    packed = module('megatron.core.packed_seq_params')
    packed.PackedSeqParams = type('PackedSeqParams', (), {})
    utils = module('megatron.core.utils')
    utils.deprecate_inference_params = lambda *args, **kwargs: None

    spec = importlib.util.spec_from_file_location('absorbed_mla_under_test', root)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_checkpoint_passes_hidden_and_compressed_query_as_keywords():
    saved = sys.modules.copy()
    try:
        absorbed = _load_absorbed()
        seen = {}

        class Core:

            def __call__(self, query, key, **kwargs):
                seen['query'] = query
                seen['key'] = key
                seen['kwargs'] = kwargs
                return query

        layer = absorbed.AbsorbedMLASelfAttention()
        layer.core_attention = Core()
        layer.attn_mask_type = 'causal'
        q = torch.tensor([1.0])
        kv = torch.tensor([2.0])
        hidden = torch.tensor([3.0])
        qr = torch.tensor([4.0])
        up = torch.tensor([5.0])
        mask = torch.tensor([6.0])
        pos = torch.tensor([7])
        packed = object()
        absorbed.tensor_parallel.checkpoint = lambda function, distribute, *args: function(*args)
        result = layer._checkpoint_absorbed_core_attention(
            q, kv, {
                'x': hidden,
                'qr': qr,
                'attention_mask': mask,
                'up_v_weight': up,
                'position_ids': pos,
                'packed_seq_params': packed,
                'attn_mask_type': 'causal',
            })
    finally:
        sys.modules.clear()
        sys.modules.update(saved)
    assert torch.equal(result, q)
    assert torch.equal(seen['query'], q)
    assert torch.equal(seen['key'], kv)
    assert seen['kwargs']['value'] is None
    assert torch.equal(seen['kwargs']['x'], hidden)
    assert torch.equal(seen['kwargs']['qr'], qr)
    assert torch.equal(seen['kwargs']['up_v_weight'], up)
    assert torch.equal(seen['kwargs']['attention_mask'], mask)
    assert seen['kwargs']['packed_seq_params'] is packed
