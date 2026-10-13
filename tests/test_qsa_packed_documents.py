# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import pytest
import torch

from mcore_bridge.model.modules import qsa_indexer as qi


def _global_reference(q, keys, boundaries, ratio, topk):
    lengths = boundaries[1:] - boundaries[:-1]
    blocks = lengths // ratio
    token_doc = torch.repeat_interleave(torch.arange(len(lengths), device=q.device), lengths)
    block_doc = torch.repeat_interleave(torch.arange(len(lengths), device=q.device), blocks)
    block_offsets = torch.cumsum(blocks, 0) - blocks
    block_pos = torch.arange(len(keys), device=q.device) - block_offsets[block_doc]
    visible = (torch.arange(len(q), device=q.device) - boundaries[token_doc] + 1) // ratio
    scores = torch.einsum('thd,kd->thk', q.float(), keys.float()).relu().sum(1) / math.sqrt(q.shape[-1])
    valid = (token_doc[:, None] == block_doc[None, :]) & (block_pos[None, :] < visible[:, None])
    selected = scores.masked_fill(~valid, float('-inf')).sort(dim=-1, descending=True, stable=True).indices
    selected = selected[:, :min(topk, len(keys))]
    return selected.masked_fill(~valid.gather(1, selected), -1)


@pytest.mark.parametrize('lengths', [[33, 18, 49], [0, 3, 0, 13, 2], [3, 1], [65]])
@pytest.mark.parametrize('ties', [False, True])
@pytest.mark.parametrize('chunk_bytes', [256, 1 << 30])
def test_document_scores_match_global_masked_selection(monkeypatch, lengths, ties, chunk_bytes):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(17)
    ratio, topk, heads, dim = 4, 5, 3, 16
    boundaries = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=device)
    q = torch.randn(sum(lengths), heads, dim, device=device)
    keys = torch.randn(sum(length // ratio for length in lengths), dim, device=device)
    if ties:
        q.zero_()
    expected = _global_reference(q, keys, boundaries, ratio, topk)
    monkeypatch.setattr(qi, '_QSA_INDEX_SCORE_CHUNK_BYTES', chunk_bytes)
    selected, keep = qi._score_packed_blocks(q, keys, boundaries, ratio, topk)
    torch.testing.assert_close(selected.masked_fill(~keep, -1), expected, rtol=0, atol=0)


def test_packed_scoring_skips_cross_document_pairs(monkeypatch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lengths = [32, 32, 32, 32]
    boundaries = torch.tensor([0, 32, 64, 96, 128], device=device)
    q = torch.randn(sum(lengths), 3, 16, device=device)
    keys = torch.randn(32, 16, device=device)
    original = torch.einsum
    pairs = []

    def record(equation, queries, blocks):
        pairs.append(queries.shape[0] * blocks.shape[0])
        return original(equation, queries, blocks)

    monkeypatch.setattr(torch, 'einsum', record)
    qi._score_packed_blocks(q, keys, boundaries, 4, 5)
    assert sum(pairs) == 4 * 32 * 8
    assert sum(pairs) < q.shape[0] * keys.shape[0]
