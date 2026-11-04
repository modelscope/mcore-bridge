# Copyright (c) ModelScope Contributors. All rights reserved.
"""DSv4 RoPE must not hand packed cu_seqlens to Megatron's zigzag helper."""
import ast
import torch
from pathlib import Path


def _load_apply():
    path = Path(__file__).resolve().parents[1] / 'src/mcore_bridge/model/gpts/deepseek_v4.py'
    tree = ast.parse(path.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_apply_mla_rope')
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    captured = {}

    def apply_rotary_pos_emb(t, freqs, **kwargs):
        captured['kwargs'] = kwargs
        captured['t'] = t
        captured['freqs'] = freqs
        return t

    namespace = {'apply_rotary_pos_emb': apply_rotary_pos_emb}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['_apply_mla_rope'], captured


def test_packed_cu_seqlens_are_not_forwarded():
    apply, captured = _load_apply()
    tokens = torch.zeros(4, 2, 8)
    freqs = torch.zeros(4, 1, 1, 4)
    cu = torch.tensor([0, 4])
    apply(tokens, freqs, config=object(), cu_seqlens=cu, cp_group=object())
    assert captured['kwargs']['cu_seqlens'] is None
    assert captured['kwargs']['mla_rotary_interleaved'] is True
    assert captured['t'] is tokens
    assert captured['freqs'] is freqs


def test_misaligned_frequencies_still_fail():
    apply, _ = _load_apply()
    try:
        apply(torch.zeros(4, 1, 4), torch.zeros(3, 1, 1, 2), config=object(), cu_seqlens=None, cp_group=None)
    except AssertionError as error:
        assert 'row-aligned' in str(error)
    else:
        raise AssertionError('expected a row-alignment failure')
