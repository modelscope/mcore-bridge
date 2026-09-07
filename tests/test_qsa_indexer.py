# Copyright (c) ModelScope Contributors. All rights reserved.
import torch

from mcore_bridge.model.modules.qsa_indexer import _materialize_rope, _rotate_half


def test_materialize_rope_preserves_mrope_batch_dimension():
    """MRoPE positions differ per sample, so ``freq_b`` must stay its own dim.

    The bug this pins: flattening ``[s, b, 1, rot]`` to ``[s, b * rot]`` makes
    ``rot`` come out as ``b * rot``, and every downstream ``[..., :rot]`` slice
    then reads the wrong half -- silently, and only when ``b > 1``.
    """
    seq_len, batch_size, rope_dim = 16, 2, 64
    freqs = torch.randn(seq_len, batch_size, 1, rope_dim)

    cos, sin = _materialize_rope(freqs, seq_len, torch.float32, 1.0)
    expected = freqs.squeeze(2).permute(1, 0, 2)

    # rot must stay rope_dim, not batch_size * rope_dim
    assert cos.shape == (batch_size, seq_len, rope_dim), \
        f'expected [b, s, rot] = {(batch_size, seq_len, rope_dim)}, got {tuple(cos.shape)}'
    torch.testing.assert_close(cos, expected.cos())
    torch.testing.assert_close(sin, expected.sin())

    # Perturbing sample 1 must leave sample 0 untouched: no cross-sample bleed.
    changed_freqs = freqs.clone()
    changed_freqs[:, 1].add_(0.5)
    changed_cos, changed_sin = _materialize_rope(changed_freqs, seq_len, torch.float32, 1.0)

    torch.testing.assert_close(changed_cos[0], cos[0])
    torch.testing.assert_close(changed_sin[0], sin[0])
    assert not torch.equal(changed_cos[1], cos[1]), 'sample 1 should have changed'
    assert not torch.equal(changed_sin[1], sin[1]), 'sample 1 should have changed'


def test_materialize_rope_applies_mscale_and_dtype():
    """``mscale`` mirrors the attention path's attention_scaling; dtype is honoured."""
    freqs = torch.randn(8, 1, 1, 32)
    mscale = 1.7

    cos, sin = _materialize_rope(freqs, 8, torch.bfloat16, mscale)
    ref = freqs.squeeze(2).permute(1, 0, 2)

    assert cos.dtype is torch.bfloat16 and sin.dtype is torch.bfloat16
    torch.testing.assert_close(cos.float(), (ref.cos() * mscale).bfloat16().float())
    torch.testing.assert_close(sin.float(), (ref.sin() * mscale).bfloat16().float())


def test_materialize_rope_truncates_to_seq_len():
    """A longer freq table is sliced to ``seq_len`` (CP hands over full-length tables)."""
    freqs = torch.randn(64, 3, 1, 16)

    cos, _ = _materialize_rope(freqs, 20, torch.float32, 1.0)

    assert cos.shape == (3, 20, 16)
    torch.testing.assert_close(cos, freqs[:20].squeeze(2).permute(1, 0, 2).cos())


def test_rotate_half_matches_reference():
    """``_rotate_half`` is the standard (-x2, x1) split the attention path uses."""
    x = torch.randn(2, 5, 8)

    got = _rotate_half(x)

    x1, x2 = x[..., :4], x[..., 4:]
    torch.testing.assert_close(got, torch.cat((-x2, x1), dim=-1))
