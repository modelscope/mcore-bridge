import pytest
import torch

from mcore_bridge.model.modules.engram import to_sequence_major


@pytest.mark.parametrize('batch,seq,dim', [(1, 918, 6144), (1, 7, 4), (3, 50, 128), (2, 1, 16)])
def test_to_sequence_major_matches_transpose(batch, seq, dim):
    embedded = torch.randn(batch, seq, dim)
    out = to_sequence_major(embedded)
    assert out.shape == (seq, batch, dim)
    torch.testing.assert_close(out, embedded.transpose(0, 1))


@pytest.mark.parametrize('batch,seq,dim', [(1, 918, 6144), (1, 7, 4), (3, 50, 128)])
def test_to_sequence_major_has_canonical_strides(batch, seq, dim):
    # `transpose(0, 1).contiguous()` leaves stride (dim, seq * dim, 1) behind for batch == 1 because
    # the size-1 dim is ignored by the contiguity check; the helper must normalize it to (dim, dim, 1).
    out = to_sequence_major(torch.randn(batch, seq, dim))
    assert out.is_contiguous()
    assert out.stride() == (batch * dim, dim, 1)
    if batch == 1:
        legacy = torch.randn(1, seq, dim).transpose(0, 1).contiguous()
        assert legacy.stride() == (dim, seq * dim, 1)
        assert out.stride() == (dim, dim, 1)
