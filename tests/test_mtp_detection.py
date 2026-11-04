import json
from types import SimpleNamespace

from mcore_bridge.model.mtp_detect import resolve_mtp_config, resolve_mtp_num_layers
from mcore_bridge.model.register import ModelMeta, get_model_meta


def _write_index(model_dir, keys):
    weight_map = {key: 'model-00001-of-00001.safetensors' for key in keys}
    with open(model_dir / 'model.safetensors.index.json', 'w') as f:
        json.dump({'weight_map': weight_map}, f)


def _standard_meta():
    return ModelMeta('test', ['test'], mtp_prefix='model.layers', mtp_depth_field='num_nextn_predict_layers')


def test_standard_layout_reads_config_and_keys(tmp_path):
    _write_index(tmp_path, ['model.layers.0.q_proj.weight', 'model.layers.2.eh_proj.weight'])
    config = SimpleNamespace(num_hidden_layers=2, num_nextn_predict_layers=1)
    assert resolve_mtp_num_layers(_standard_meta(), config, str(tmp_path)) == 1


def test_standard_layout_requires_mtp_keys(tmp_path):
    _write_index(tmp_path, ['model.layers.0.q_proj.weight'])
    config = SimpleNamespace(num_hidden_layers=2, num_nextn_predict_layers=1)
    assert resolve_mtp_num_layers(_standard_meta(), config, str(tmp_path)) is None


def test_mtp_prefix_without_config_field(tmp_path):
    _write_index(tmp_path, ['mtp.layers.0.input_layernorm.weight'])
    config = SimpleNamespace(num_hidden_layers=2)
    meta = ModelMeta('test', ['test'], mtp_prefix='mtp')
    assert resolve_mtp_num_layers(meta, config, str(tmp_path)) == 1


def test_mtp_prefix_top_level_only(tmp_path):
    _write_index(tmp_path, ['mtp.fc_embedding.weight'])
    config = SimpleNamespace(num_hidden_layers=2)
    meta = ModelMeta('test', ['test'], mtp_prefix='mtp')
    assert resolve_mtp_num_layers(meta, config, str(tmp_path)) == 1


def test_no_mtp_layout_ignores_mtp_keys(tmp_path):
    _write_index(tmp_path, ['mtp.layers.0.input_layernorm.weight'])
    config = SimpleNamespace(num_hidden_layers=2, num_nextn_predict_layers=1)
    assert resolve_mtp_num_layers(ModelMeta('test', ['test']), config, str(tmp_path)) is None


def test_standard_layout_multimodal_prefix(tmp_path):
    _write_index(tmp_path, [
        'model.language_model.layers.0.q_proj.weight',
        'model.language_model.layers.2.eh_proj.weight',
    ])
    config = SimpleNamespace(num_hidden_layers=2, num_nextn_predict_layers=1)
    assert resolve_mtp_num_layers(_standard_meta(), config, str(tmp_path)) == 1


def test_mtp_prefix_multimodal_wrapper(tmp_path):
    _write_index(tmp_path, ['model.language_model.mtp.layers.0.input_layernorm.weight'])
    config = SimpleNamespace(num_hidden_layers=2)
    meta = ModelMeta('test', ['test'], mtp_prefix='mtp')
    assert resolve_mtp_num_layers(meta, config, str(tmp_path)) == 1


def test_model_mtp_prefix(tmp_path):
    _write_index(tmp_path, ['model.mtp.layers.0.eh_proj.weight'])
    config = SimpleNamespace(num_hidden_layers=2)
    meta = ModelMeta('test', ['test'], mtp_prefix='model.mtp')
    assert resolve_mtp_num_layers(meta, config, str(tmp_path)) == 1


def test_model_mtp_direct_numbering(tmp_path):
    _write_index(tmp_path, ['model.mtp.0.e_proj.weight', 'model.mtp.1.e_proj.weight'])
    config = SimpleNamespace(num_hidden_layers=2, num_nextn_predict_layers=2)
    meta = get_model_meta('deepseek_v4')
    resolution = resolve_mtp_config(meta, config, str(tmp_path), 'auto', None)
    assert resolution == (2, False)


def test_default_depth_field(tmp_path):
    _write_index(tmp_path, ['model.layers.2.eh_proj.weight'])
    config = SimpleNamespace(num_hidden_layers=2, num_nextn_predict_layers=1)
    meta = ModelMeta('test', ['test'], mtp_prefix='model.layers')
    assert resolve_mtp_num_layers(meta, config, str(tmp_path)) == 1


def test_registered_mtp_layouts():
    expected = {
        'gpt': ('model.layers', 'num_nextn_predict_layers'),
        'glm_moe_dsa': ('model.layers', 'num_nextn_predict_layers'),
        'glm5_next': ('model.layers', 'num_nextn_predict_layers'),
        'deepseek_v4': ('model.mtp', 'num_nextn_predict_layers'),
        'qwen3_next': ('mtp', 'mtp_num_hidden_layers'),
        'qwen3_5': ('mtp', 'mtp_num_hidden_layers'),
        'qwen4_exp': ('mtp', 'mtp_num_hidden_layers'),
        'nemotron_h': ('mtp', 'mtp_num_hidden_layers'),
    }
    for model_type, layout in expected.items():
        meta = get_model_meta(model_type)
        if meta is not None:
            assert (meta.mtp_prefix, meta.mtp_depth_field) == layout
    for model_type in ('deepseek_v41', 'minicpmv4_6', 'nemotron_h_omni'):
        meta = get_model_meta(model_type)
        if meta is not None:
            assert meta.mtp_prefix is None


def test_resolve_mtp_config_auto_and_shared(tmp_path):
    _write_index(tmp_path, ['model.layers.2.eh_proj.weight'])
    config = SimpleNamespace(num_hidden_layers=2, num_nextn_predict_layers=1)
    meta = _standard_meta()
    cases = (
        ('auto', None, 1, False),
        (3, None, 3, True),
        (3, False, 3, False),
        (1, None, 1, False),
        (3, True, 3, True),
        (None, None, None, False),
        (0, None, 0, False),
    )
    for requested_depth, requested_shared, depth, shared in cases:
        resolution = resolve_mtp_config(meta, config, str(tmp_path), requested_depth, requested_shared)
        assert resolution == (depth, shared)


def test_registered_models_auto_resolution(tmp_path):
    cases = {
        'gpt': ({
            'num_hidden_layers': 2,
            'num_nextn_predict_layers': 1
        }, ['model.layers.2.eh_proj.weight']),
        'glm_moe_dsa': ({
            'num_hidden_layers': 2,
            'num_nextn_predict_layers': 1
        }, ['model.layers.2.eh_proj.weight']),
        'glm5_next': ({
            'num_hidden_layers': 2,
            'num_nextn_predict_layers': 1
        }, ['model.language_model.layers.2.eh_proj.weight']),
        'deepseek_v4': ({
            'num_hidden_layers': 2,
            'num_nextn_predict_layers': 0
        }, ['model.mtp.layers.0.eh_proj.weight']),
        'qwen3_next': ({
            'num_hidden_layers': 2
        }, ['mtp.layers.0.input_layernorm.weight']),
        'qwen3_5': ({
            'num_hidden_layers': 2,
            'mtp_num_hidden_layers': 1
        }, ['mtp.layers.0.input_layernorm.weight']),
        'qwen4_exp': ({
            'num_hidden_layers': 2,
            'mtp_num_hidden_layers': 1
        }, ['mtp.fc_embedding.weight', 'mtp.layers.0.input_layernorm.weight']),
        'nemotron_h': ({
            'num_hidden_layers': 2
        }, ['mtp.layers.0.input_layernorm.weight']),
    }
    for model_type, (config_kwargs, keys) in cases.items():
        meta = get_model_meta(model_type)
        if meta is None:
            continue
        model_dir = tmp_path / model_type
        model_dir.mkdir()
        _write_index(model_dir, keys)
        assert resolve_mtp_num_layers(meta, SimpleNamespace(**config_kwargs), str(model_dir)) == 1


def test_multiple_physical_sets(tmp_path):
    _write_index(tmp_path, ['model.layers.2.eh_proj.weight', 'model.layers.3.eh_proj.weight'])
    config = SimpleNamespace(num_hidden_layers=2, num_nextn_predict_layers=2)
    assert resolve_mtp_num_layers(_standard_meta(), config, str(tmp_path)) == 2
    cases = (('auto', None, 2, False), (3, None, 3, False), (3, True, 3, True))
    for requested_depth, requested_shared, depth, shared in cases:
        resolution = resolve_mtp_config(_standard_meta(), config, str(tmp_path), requested_depth, requested_shared)
        assert resolution == (depth, shared)


def test_three_physical_sets_keep_independent_weights(tmp_path):
    _write_index(tmp_path, [f'model.layers.{i}.eh_proj.weight' for i in range(2, 5)])
    config = SimpleNamespace(num_hidden_layers=2, num_nextn_predict_layers=3)
    for requested_depth in ('auto', 3):
        resolution = resolve_mtp_config(_standard_meta(), config, str(tmp_path), requested_depth, None)
        assert resolution == (3, False)


def test_unreadable_weights_disables_mtp(tmp_path):
    config = SimpleNamespace(num_hidden_layers=2, num_nextn_predict_layers=1)
    assert resolve_mtp_num_layers(_standard_meta(), config, str(tmp_path)) is None
