# Copyright (c) ModelScope Contributors. All rights reserved.
"""Gemma 4 attention dims must be read the same way before and after transformers 5.15 (per-layer configs).
The parser functions are loaded from source so most cases run on CPU without Transformer Engine."""
import ast
import pytest
from pathlib import Path
from typing import Any, Dict, Tuple

gemma4_config = pytest.importorskip('transformers.models.gemma4.configuration_gemma4')

_NAMES = {'config_mapping', 'has_per_layer_config', 'get_gemma4_attention_dims', 'get_gemma4_rope_head_dim_key'}


def _load_parser():
    path = Path(__file__).resolve().parents[1] / 'src/mcore_bridge/config/parser.py'
    body = [
        node for node in ast.parse(path.read_text()).body
        if (isinstance(node, ast.FunctionDef) and node.name in _NAMES | {'_convert_config'}) or (
            isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) in _NAMES)
    ]
    namespace = {'Any': Any, 'Dict': Dict, 'Tuple': Tuple}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), 'exec'), namespace)
    return namespace


# Attention fields of the public checkpoints (fewer layers); expected (head_dim, kv_heads) for sliding / full.
_CASES = {
    'gemma-4-31B': (dict(
        hidden_size=5376,
        num_attention_heads=32,
        num_key_value_heads=16,
        num_global_key_value_heads=4,
        attention_k_eq_v=True), (256, 16), (512, 4)),
    'gemma-4-E2B': (dict(
        hidden_size=1536,
        num_attention_heads=8,
        num_key_value_heads=1,
        num_global_key_value_heads=None,
        attention_k_eq_v=False), (256, 1), (512, 1)),
}


def _make_config(case):
    rope = {
        'full_attention': {
            'partial_rotary_factor': 0.25,
            'rope_theta': 1000000.0,
            'rope_type': 'proportional'
        },
        'sliding_attention': {
            'rope_theta': 10000.0,
            'rope_type': 'default'
        },
    }
    text_config = dict(
        num_hidden_layers=6,
        layer_types=['sliding_attention'] * 5 + ['full_attention'],
        head_dim=256,
        global_head_dim=512,
        hidden_size_per_layer_input=0,
        rope_parameters=rope,
        **_CASES[case][0])
    return gemma4_config.Gemma4Config(text_config=text_config)


@pytest.mark.parametrize('case', list(_CASES))
def test_attention_dims(case):
    parser = _load_parser()
    config = _make_config(case)
    _, sliding, full = _CASES[case]
    assert parser['get_gemma4_attention_dims'](config.text_config, 'sliding_attention') == sliding
    assert parser['get_gemma4_attention_dims'](config.text_config, 'full_attention') == full
    # The generic mapping must not read per-layer attributes (it raised on transformers>=5.15).
    res = parser['_convert_config'](config)
    assert res.get('kv_channels') in (None, sliding[0]) and res.get('num_query_groups') in (None, sliding[1])


@pytest.mark.parametrize('case', list(_CASES))
def test_full_attention_rope_matches_hf(case):
    torch = pytest.importorskip('torch')
    modeling = pytest.importorskip('transformers.models.gemma4.modeling_gemma4')
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    text_config = _make_config(case).text_config
    head_dim_key = _load_parser()['get_gemma4_rope_head_dim_key'](text_config)
    inv_freq, _ = ROPE_INIT_FUNCTIONS['proportional'](
        text_config, 'cpu', layer_type='full_attention', head_dim_key=head_dim_key)
    assert inv_freq.numel() == 512 // 2
    torch.testing.assert_close(inv_freq, modeling.Gemma4TextRotaryEmbedding(text_config).full_attention_inv_freq)


@pytest.mark.parametrize('case', list(_CASES))
def test_hf_to_mcore_config(case):
    pytest.importorskip('transformer_engine')
    from mcore_bridge.config import hf_to_mcore_config
    res = hf_to_mcore_config(_make_config(case))
    assert (res['kv_channels'], res['num_query_groups']) == _CASES[case][1]
