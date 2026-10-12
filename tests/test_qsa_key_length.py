# Copyright (c) ModelScope Contributors. All rights reserved.
"""The QSA block bitmap is sized from the key length, not the query count."""
import ast
import torch
from pathlib import Path


def _bitmap():
    path = Path(__file__).resolve().parents[1] / 'src/mcore_bridge/model/modules/kernels/qsa_block_sparse_attn.py'
    tree = ast.parse(path.read_text())
    fn = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'selection_to_block_bitmap')
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {'Tensor': torch.Tensor, 'torch': torch}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['selection_to_block_bitmap']


def test_short_query_can_flag_a_later_key_block():
    bitmap = _bitmap()
    # 2 queries, keys of length 8, block 4. Index 6 is in block 1, past the query count.
    indices = torch.tensor([[6, -1], [-1, -1]])
    flags = bitmap(indices, 8, 4)
    assert flags.shape == (2, 2)
    assert int(flags[0, 1]) == 1
    assert int(flags[0, 0]) == 0


def _index_path(captured):
    path = Path(__file__).resolve().parents[1] / 'src/mcore_bridge/model/modules/kernels/qsa_block_sparse_attn.py'
    tree = ast.parse(path.read_text())
    names = {'selection_to_block_bitmap', 'qsa_sparse_attention_from_indices'}
    fns = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]

    def fake_triton(q, k, v, sel, lo, hi, blk_base, tok_base, scale, block_size):
        captured['blk_base'] = blk_base
        captured['tok_base'] = tok_base
        return q

    module = ast.Module(body=fns, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        'Tensor': torch.Tensor,
        'torch': torch,
        'qsa_block_sparse_attention_triton': fake_triton,
    }
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['qsa_sparse_attention_from_indices']


def test_index_path_bases_follow_the_query_count_when_keys_are_longer():
    captured = {}
    fn = _index_path(captured)
    q = torch.zeros(2, 1, 4)
    k = torch.zeros(8, 1, 4)
    indices = torch.tensor([[6, -1], [-1, -1]])
    out = fn(q, k, k, indices, 1.0, 4)
    assert out.shape == q.shape
    assert captured['blk_base'].shape == (2,)
    assert captured['tok_base'].shape == (2,)
    assert int(captured['blk_base'].sum()) == 0
    assert int(captured['tok_base'].sum()) == 0
