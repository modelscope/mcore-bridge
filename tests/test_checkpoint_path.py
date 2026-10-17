# Copyright (c) ModelScope Contributors. All rights reserved.
"""Weight-map filenames cannot leave the checkpoint directory."""
import ast
import os
from pathlib import Path


def _join():
    path = Path(__file__).resolve().parents[1] / 'src/mcore_bridge/utils/safetensors.py'
    tree = ast.parse(path.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'checkpoint_path')
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {'os': os}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['checkpoint_path']


def test_relative_shard_stays_inside(tmp_path):
    join = _join()
    shard = tmp_path / 'model-00001.safetensors'
    shard.write_bytes(b'x')
    assert join(str(tmp_path), 'model-00001.safetensors') == os.path.realpath(shard)


def test_parent_and_absolute_names_are_rejected(tmp_path):
    join = _join()
    outside = tmp_path.parent / 'outside.safetensors'
    outside.write_bytes(b'x')
    for name in ('../outside.safetensors', str(outside)):
        try:
            join(str(tmp_path), name)
        except ValueError:
            continue
        raise AssertionError(name)
