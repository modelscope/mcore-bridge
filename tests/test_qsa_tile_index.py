"""CPU-only: a packed selection must attend exactly its keys, and the tile map must
list every attended key's tile."""
import pytest
import random

from mcore_bridge.model.modules.kernels import qsa_kernels as qk
from mcore_bridge.model.modules.kernels.qsa_block_sparse_attn import build_tile_index, build_tile_index_pair

torch = pytest.importorskip('torch')
pytest.importorskip('triton')

BQ = 64
_rng = random.Random(0)
PACKS = [
    ([61, 128], 4),
    ([61, 61, 61, 200], 4),
    ([64, 128], 4),
    ([1, 1, 1, 1, 1, 200], 4),
    ([3, 20], 4),
    ([1, 3, 20, 37, 9], 4),
    *[([_rng.randint(1, 300) for _ in range(_rng.randint(2, 6))], _rng.choice([3, 4, 5, 8])) for _ in range(6)],
]


def _selection_rows(lens, ratio, seed=0):
    """Indexer-shaped rows: whole document-relative blocks (with holes) + causal tail."""
    cu = torch.tensor([0, *lens]).cumsum(0)
    total = int(cu[-1])
    starts = cu[:-1]
    seg = torch.zeros(total, dtype=torch.long)
    seg[starts[1:]] = 1
    seg = seg.cumsum(0)
    positions = torch.arange(total) - starts[seg]
    layout = qk._PackedBlockLayout(cu, ratio, 'cpu')
    g = torch.Generator().manual_seed(seed)

    rows = []
    for t in range(total):
        seq = int(seg[t])
        p = int(positions[t])
        visible = (p + 1) // ratio
        blocks = torch.randperm(visible, generator=g)[:min(2, visible)].sort().values if visible else \
            torch.zeros(0, dtype=torch.long)
        full = (blocks[:, None] * ratio + torch.arange(ratio)[None, :]).reshape(-1) + int(starts[seq])
        qn = (p + 1) // ratio
        tail = torch.arange(int(starts[seq]) + qn * ratio, min(int(starts[seq]) + qn * ratio + ratio, t + 1))
        row = torch.cat([full, tail]).unique()
        rows.append(row)
    idx = torch.full((total, max(r.numel() for r in rows)), -1, dtype=torch.int64)
    for t, row in enumerate(rows):
        idx[t, :row.numel()] = row
    return layout, idx


def _admitted(layout, idx, block_size, sel, lo, hi):
    """Kernel membership rule: selected block column and inside [lo, hi]."""
    total = idx.shape[0]
    keys = torch.arange(total)
    tok_base = layout.token_start
    blk_base = layout.token_block_start
    blk = blk_base.unsqueeze(1) + (keys[None, :] - tok_base.unsqueeze(1)) // block_size
    inside = (blk >= 0) & (blk < sel.shape[1])
    gathered = torch.gather(sel.long(), 1, blk.clamp(0, sel.shape[1] - 1))
    return (gathered != 0) & inside & (keys[None, :] >= lo[:, None]) & (keys[None, :] <= hi[:, None])


def _tile_sets(lst, cnt):
    return [set(lst[i, :int(cnt[i])].tolist()) for i in range(cnt.numel())]


@pytest.mark.parametrize('lens,ratio', PACKS, ids=lambda x: str(x))
def test_packed_off_grid_attends_exactly_the_selection(lens, ratio):
    layout, idx = _selection_rows(lens, ratio)
    sel, lo, hi, blk_base, tok_base, block_first, block_last = qk._block_form(idx, layout, ratio)

    exact = torch.zeros(idx.shape[0], idx.shape[0], dtype=torch.bool)
    rows = torch.arange(idx.shape[0]).unsqueeze(1).expand_as(idx)
    valid = idx >= 0
    exact[rows[valid], idx[valid]] = True
    # rows may list the current block whole; the kernel clips at [lo, hi]
    keys = torch.arange(idx.shape[0])
    exact &= (keys[None, :] >= lo[:, None]) & (keys[None, :] <= hi[:, None])

    admitted = _admitted(layout, idx, ratio, sel, lo, hi)
    extra = int((admitted & ~exact).sum())
    missed = int((exact & ~admitted).sum())
    assert extra == 0, f'{lens} R={ratio}: {extra} keys attended outside the selection'
    assert missed == 0, f'{lens} R={ratio}: {missed} selected keys dropped'


@pytest.mark.parametrize('lens,ratio', PACKS, ids=lambda x: str(x))
def test_packed_off_grid_tile_lists_cover_every_attended_key(lens, ratio):
    layout, idx = _selection_rows(lens, ratio)
    sel, lo, hi, blk_base, tok_base, block_first, block_last = qk._block_form(idx, layout, ratio)
    admitted = _admitted(layout, idx, ratio, sel, lo, hi)
    total = idx.shape[0]

    for bk, pair in ((64, False), (32, True)):
        if pair:
            klist, kcnt, qlist, qcnt = build_tile_index_pair(sel, block_first, block_last, BQ, bk)
        else:
            klist, kcnt = build_tile_index(sel, block_first, block_last, BQ, bk)
            qlist, qcnt = build_tile_index_pair(sel, block_first, block_last, BQ, bk)[2:]
        k_tiles = _tile_sets(klist, kcnt)
        q_tiles = _tile_sets(qlist, qcnt)
        missing = 0
        for t in range(total):
            for key in admitted[t].nonzero().flatten().tolist():
                if key // bk not in k_tiles[t // BQ]:
                    missing += 1
                if t // BQ not in q_tiles[key // bk]:
                    missing += 1
        assert missing == 0, f'{lens} R={ratio} BK={bk}: tile lists miss {missing} (query, key) pairs'


class _PackedSeqParams:

    def __init__(self, cu, padded=None):
        self.cu_seqlens_q = cu
        self.cu_seqlens_q_padded = padded


@pytest.mark.parametrize(
    'cu,padded,total,expected',
    [
        ([0, 3, 23], None, 23, [0, 3, 23]),  # plain padding_free packing
        ([3, 23], None, 23, [0, 3, 23]),  # missing leading 0
        ([5, 8, 28], None, 23, [0, 3, 23]),  # per-batch offset
        ([0, 3, 23], [0, 4, 28], 28, [0, 4, 28]),  # CP: padded space wins
        ([0, 3, 23], None, 28, None),  # nothing describes 28 tokens -> loud
    ])
def test_packed_cu_seqlens_alignment(cu, padded, total, expected):
    psp = _PackedSeqParams(None if cu is None else torch.tensor(cu, dtype=torch.int32),
                           None if padded is None else torch.tensor(padded, dtype=torch.int32))
    if expected is None:
        with pytest.raises(ValueError):
            qk._packed_cu_seqlens(psp, total)
        return
    aligned = qk._packed_cu_seqlens(psp, total)
    assert aligned.tolist() == expected
    assert aligned.dtype == torch.long
