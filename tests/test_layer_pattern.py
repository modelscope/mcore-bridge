# Copyright (c) ModelScope Contributors. All rights reserved.
"""Layer patterns are parsed, not eval'd, and huge repetitions are refused."""
import ast
from pathlib import Path


def _parser():
    path = Path(__file__).resolve().parents[1] / 'src/mcore_bridge/config/model_config.py'
    tree = ast.parse(path.read_text())
    body = [
        node for node in tree.body if (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == '_MAX_PATTERN_ITEMS' for target in node.targets)) or (
                isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {'_eval_pattern', '_PatternParser'})
    ]
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {'re': __import__('re')}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['_eval_pattern']


def test_documented_pattern_matches_the_expanded_list():
    parse = _parser()
    assert parse('([0]*3+[1]*1)*3') == [0, 0, 0, 1] * 3
    assert parse('([1]+[0]*2)') == [1, 0, 0]


def test_exponent_and_huge_repeat_are_rejected():
    parse = _parser()
    for pattern in ('[0]*10**9', '[0]*(10**9)', '[0]*1000000'):
        try:
            parse(pattern)
        except ValueError:
            continue
        raise AssertionError(pattern)
