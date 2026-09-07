# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import torch
from pathlib import Path


def _load_materialize_rope():
    module_path = Path(__file__).parents[1] / 'src/mcore_bridge/model/modules/qsa_indexer.py'
    spec = importlib.util.spec_from_file_location('qsa_indexer', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._materialize_rope


def test_materialize_rope_preserves_mrope_batch_dimension():
    materialize_rope = _load_materialize_rope()
    seq_len, batch_size, rope_dim = 16, 2, 64
    freqs = torch.randn(seq_len, batch_size, 1, rope_dim)

    cos, sin = materialize_rope(freqs, seq_len, torch.float32, 1.0)
    expected = freqs.squeeze(2).permute(1, 0, 2)

    assert cos.shape == (batch_size, seq_len, rope_dim)
    torch.testing.assert_close(cos, expected.cos())
    torch.testing.assert_close(sin, expected.sin())

    changed_freqs = freqs.clone()
    changed_freqs[:, 1].add_(0.5)
    changed_cos, changed_sin = materialize_rope(changed_freqs, seq_len, torch.float32, 1.0)

    torch.testing.assert_close(changed_cos[0], cos[0])
    torch.testing.assert_close(changed_sin[0], sin[0])
    assert not torch.equal(changed_cos[1], cos[1])
    assert not torch.equal(changed_sin[1], sin[1])
