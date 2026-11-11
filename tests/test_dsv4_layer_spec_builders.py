# Copyright (c) ModelScope Contributors. All rights reserved.
"""Exercise the V4 loader's builder adapter without importing GPU dependencies."""
import ast
import copy
import pytest
from functools import partial
from pathlib import Path
from types import SimpleNamespace


class _Module:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _Compressor(_Module):
    pass


class _Indexer(_Module):
    pass


def _load_loader(specs):
    # Match the CPU-only AST harness used by test_dsv4_mla_rope.py.
    path = Path(__file__).resolve().parents[1] / 'src/mcore_bridge/model/gpts/deepseek_v4.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DeepseekV4Loader')
    namespace = dict(
        copy=copy,
        partial=partial,
        ModelLoader=object,
        DeepseekV4GPTModel=object,
        McoreDSv4HybridSelfAttention=_Module,
        DSv4HybridSelfAttention=_Module,
        Compressor=_Compressor,
        CSAIndexer=_Indexer,
        _build_all_layer_specs=lambda config: specs)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
    loader = namespace['DeepseekV4Loader']()
    loader.config = object()
    return loader


def _builder(as_partial, submodules):
    if as_partial:
        return partial(_Module, 'bound-argument', submodules=submodules, option=7)
    return SimpleNamespace(module=_Module, submodules=submodules, params={'option': 7})


def _submodules(builder):
    return builder.keywords['submodules'] if isinstance(builder, partial) else builder.submodules


def _assert_module(builder, expected):
    if isinstance(builder, partial):
        built = builder(runtime_option=11)
        assert type(built) is expected
        assert built.args == ('bound-argument', )
        assert built.kwargs == dict(submodules=_submodules(builder), option=7, runtime_option=11)
    else:
        assert builder.module is expected
        assert builder.params == {'option': 7}


@pytest.mark.parametrize('core_partial', [False, True])
@pytest.mark.parametrize('indexer_partial', [False, True])
@pytest.mark.parametrize('compressor_partial', [False, True])
def test_patch_builders_preserves_constructor_arguments(core_partial, indexer_partial, compressor_partial):
    compressor = _builder(compressor_partial, SimpleNamespace(linear_wkv='wkv', norm='norm'))
    indexer = _builder(indexer_partial, SimpleNamespace(compressor=compressor, linear_wq_b='wq'))
    core = _builder(core_partial, SimpleNamespace(compressor=compressor, indexer=indexer))
    attention = SimpleNamespace(module=object, submodules=SimpleNamespace(core_attention=core))
    original = SimpleNamespace(submodules=SimpleNamespace(self_attention=attention))
    patched = _load_loader([original])._build_decoder_layer_specs()[0]

    assert patched.submodules.self_attention.module is _Module
    patched_core = patched.submodules.self_attention.submodules.core_attention
    _assert_module(patched_core, _Module)
    core_submodules = _submodules(patched_core)
    _assert_module(core_submodules.compressor, _Compressor)
    _assert_module(core_submodules.indexer, _Indexer)
    indexer_submodules = _submodules(core_submodules.indexer)
    _assert_module(indexer_submodules.compressor, _Compressor)
    assert vars(_submodules(core_submodules.compressor)) == {'linear_wkv': 'wkv', 'norm': 'norm'}
    assert indexer_submodules.linear_wq_b == 'wq'
    # The loader must patch its deep copy, not upstream's shared specs.
    assert attention.module is object
    _assert_module(compressor, _Module)
    _assert_module(indexer, _Module)


@pytest.mark.parametrize('as_partial', [False, True])
@pytest.mark.parametrize('fields', [{}, {'compressor': None, 'indexer': None}])
def test_optional_builders_remain_absent(as_partial, fields):
    core = _builder(as_partial, SimpleNamespace(**fields))
    attention = SimpleNamespace(module=object, submodules=SimpleNamespace(core_attention=core))
    spec = SimpleNamespace(submodules=SimpleNamespace(self_attention=attention))
    patched = _load_loader([spec])._build_decoder_layer_specs()[0]
    assert vars(_submodules(patched.submodules.self_attention.submodules.core_attention)) == fields
