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
