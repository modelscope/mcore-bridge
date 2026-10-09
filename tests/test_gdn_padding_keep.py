# Copyright (c) ModelScope Contributors. All rights reserved.
"""Left-padding positions are zeroed before the GDN recurrence."""
import ast
import torch
from pathlib import Path


def _keep():
    path = Path(__file__).resolve().parents[1] / 'src/mcore_bridge/model/modules/gated_delta_net.py'
    tree = ast.parse(path.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'gdn_padding_keep')
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {'torch': torch}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['gdn_padding_keep']


def test_left_pad_column_is_dropped():
    keep = _keep()
    # [batch, 1, seq, seq], True = masked. Batch 0 has the first key padded.
    mask = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    mask[0, :, :, 0] = True
    out = keep(mask, seq_len=3, batch=1)
    assert out.shape == (3, 1, 1)
    assert torch.equal(out.squeeze(), torch.tensor([0, 1, 1]))


def test_unrelated_shape_is_ignored():
    keep = _keep()
    mask = torch.ones(2, 3, dtype=torch.bool)
    assert keep(mask, seq_len=4, batch=2) is None
