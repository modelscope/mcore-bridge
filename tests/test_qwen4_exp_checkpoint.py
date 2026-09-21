# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
import pytest
import torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import PretrainedConfig
from types import SimpleNamespace

from mcore_bridge.utils import qwen4_exp_checkpoint as mcore_bridge_checkpoint
from mcore_bridge.utils.qwen4_exp_checkpoint import (qwen4_exp_export_config, restore_qwen4_exp_fixed_assets,
                                                     validate_ple_checkpoint)

LAYERS_PREFIX = 'model.language_model.layers'
PLE_PREFIX = f"{LAYERS_PREFIX}.1.ple.ple_embedding."


def _write_weights(directory: Path, tensors: dict, *, indexed: bool = False) -> None:
    directory.mkdir(exist_ok=True)
    save_file(tensors, directory / 'model.safetensors', metadata={'format': 'pt'})
    if indexed:
        (directory / 'model.safetensors.index.json').write_text(
            json.dumps({
                'metadata': {
                    'total_size': -1,
                    'test_metadata': 'preserved'
                },
                'weight_map': dict.fromkeys(tensors, 'model.safetensors'),
            }))


@pytest.fixture
def checkpoints(tmp_path):
    source = tmp_path / 'source'
    output = tmp_path / 'output'
    source_tensors = {
        'model.language_model.layers.0.mlp.experts.gate_up_proj': torch.ones(2, 2, dtype=torch.bfloat16),
        'lm_head.weight': torch.full((2, 2), 2.0, dtype=torch.bfloat16),
        'model.visual.blocks.0.attn.qkv.weight': torch.arange(4, dtype=torch.bfloat16).reshape(2, 2),
        'model.visual.patch_embed.proj.weight': torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        'model.visual.pos_embed.weight': torch.arange(6, dtype=torch.float32),
        'model.visual.merger.bias': torch.tensor(2.0),
        'mtp.fc_hidden.weight': torch.ones(2, 2),
    }
    exported_tensors = {
        'model.language_model.layers.0.mlp.experts.gate_up_proj': torch.full((2, 2), 9.0, dtype=torch.bfloat16),
        'lm_head.weight': torch.full((2, 2), 7.0, dtype=torch.bfloat16),
    }
    _write_weights(source, source_tensors)
    (source / 'config.json').write_text(json.dumps({'model_type': 'qwen4_exp'}))
    _write_weights(output, exported_tensors)
    return source, output, source_tensors, exported_tensors


def _read_indexed_weights(directory: Path) -> tuple[dict, dict]:
    index = json.loads((directory / 'model.safetensors.index.json').read_text())
    tensors = {}
    for filename in set(index['weight_map'].values()):
        with safe_open(directory / filename, framework='pt', device='cpu') as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key)
    return index, tensors


@pytest.mark.parametrize('indexed', [False, True])
def test_restore_only_visual_assets_preserves_trained_text_and_index(checkpoints, indexed):
    source, output, source_tensors, exported_tensors = checkpoints
    _write_weights(source, source_tensors, indexed=indexed)
    _write_weights(output, exported_tensors, indexed=indexed)
    text_shard_hash = hashlib.sha256((output / 'model.safetensors').read_bytes()).hexdigest()

    report = restore_qwen4_exp_fixed_assets(str(source), str(output), language_model_only=True, max_shard_size_bytes=32)

    index, tensors = _read_indexed_weights(output)
    assert set(tensors) == set(source_tensors) - {'mtp.fc_hidden.weight'}
    assert report['omitted_mtp_keys'] == ['mtp.fc_hidden.weight']
    assert report['restored_keys'] == sorted(key for key in source_tensors if key.startswith('model.visual.'))
    assert len(report['new_shards']) >= 2
    for filename in report['new_shards']:
        shard_bytes = sum(tensors[key].numel() * tensors[key].element_size()
                          for key, shard_name in index['weight_map'].items() if shard_name == filename)
        assert shard_bytes <= 32
    for key, value in tensors.items():
        expected = (exported_tensors[key] if key in exported_tensors else source_tensors[key])
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert report['total_size'] == sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    assert index['metadata']['total_size'] == report['total_size']
    if indexed:
        assert index['metadata']['test_metadata'] == 'preserved'
    # Transformers prioritizes the monolithic filename over an index.
    assert not (output / 'model.safetensors').exists()
    exported_filename = index['weight_map']['lm_head.weight']
    assert (hashlib.sha256((output / exported_filename).read_bytes()).hexdigest() == text_shard_hash)
    assert not list(output.glob('*.pending'))


def test_restore_repeated_call_is_idempotent(checkpoints):
    source, output, _, _ = checkpoints
    restore_qwen4_exp_fixed_assets(str(source), str(output), language_model_only=True)
    first_files = {path.name: path.read_bytes() for path in output.iterdir()}

    report = restore_qwen4_exp_fixed_assets(str(source), str(output), language_model_only=True)

    assert report['restored_keys'] == []
    assert report['new_shards'] == []
    assert {path.name: path.read_bytes() for path in output.iterdir()} == first_files


def test_restore_hf_sharded_loader_reads_text_and_visual_assets(checkpoints):
    from transformers.trainer_utils import load_sharded_checkpoint

    source, output, source_tensors, exported_tensors = checkpoints
    restore_qwen4_exp_fixed_assets(str(source), str(output), language_model_only=True)
    model = torch.nn.Module()
    for key, tensor in source_tensors.items():
        if key.startswith('mtp.'):
            continue
        module = model
        components = key.split('.')
        for component in components[:-1]:
            if not hasattr(module, component):
                module.add_module(component, torch.nn.Module())
            module = getattr(module, component)
        module.register_parameter(components[-1], torch.nn.Parameter(torch.zeros_like(tensor)))

    result = load_sharded_checkpoint(model, str(output), strict=True, prefer_safe=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    for key, tensor in model.state_dict().items():
        expected = (exported_tensors[key] if key in exported_tensors else source_tensors[key])
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    'missing_key',
    ['lm_head.weight', 'model.language_model.layers.0.mlp.experts.gate_up_proj'],
)
def test_restore_rejects_missing_text_before_writing(checkpoints, missing_key):
    source, output, _, exported_tensors = checkpoints
    exported_tensors.pop(missing_key)
    _write_weights(output, exported_tensors)
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    with pytest.raises(ValueError, match='missing required non-restorable tensors'):
        restore_qwen4_exp_fixed_assets(str(source), str(output), language_model_only=True)

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_restore_does_not_copy_missing_ple_or_nested_mtp(checkpoints):
    source, output, source_tensors, _ = checkpoints
    source_tensors['model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight'] = torch.ones(2, 2)
    source_tensors['model.language_model.mtp.fc_hidden.weight'] = torch.ones(2, 2)
    _write_weights(source, source_tensors)

    with pytest.raises(ValueError, match='missing required non-restorable tensors'):
        restore_qwen4_exp_fixed_assets(str(source), str(output), language_model_only=True)

    assert not list(output.glob('model-fixed-visual-*.safetensors'))


def test_restore_rejects_unknown_output_keys(checkpoints):
    source, output, _, exported_tensors = checkpoints
    exported_tensors['unexpected.weight'] = torch.ones(2)
    _write_weights(output, exported_tensors)

    with pytest.raises(ValueError, match='unknown checkpoint tensors'):
        restore_qwen4_exp_fixed_assets(str(source), str(output), language_model_only=True)


def test_restore_rejects_text_shape_change(checkpoints):
    source, output, _, exported_tensors = checkpoints
    exported_tensors['lm_head.weight'] = torch.ones(1, 2)
    _write_weights(output, exported_tensors)

    with pytest.raises(ValueError, match='shape differs from source'):
        restore_qwen4_exp_fixed_assets(str(source), str(output), language_model_only=True)


@pytest.mark.parametrize('dtype', [torch.int64, torch.bool, torch.float16])
def test_restore_rejects_changed_tensor_dtype(checkpoints, dtype):
    source, output, _, exported_tensors = checkpoints
    exported_tensors['lm_head.weight'] = exported_tensors['lm_head.weight'].to(dtype)
    _write_weights(output, exported_tensors)

    with pytest.raises(ValueError, match='dtype differs from source'):
        restore_qwen4_exp_fixed_assets(str(source), str(output), language_model_only=True)
    assert not list(output.glob('model-fixed-visual-*.safetensors'))


def test_restore_preserves_source_fp32_buffer_dtype(checkpoints):
    source, output, source_tensors, exported_tensors = checkpoints
    key = 'model.language_model.layers.0.linear_attn.A_log'
    source_tensors[key] = torch.ones(2, dtype=torch.float32)
    exported_tensors[key] = torch.full((2, ), 2.0, dtype=torch.float32)
    _write_weights(source, source_tensors)
    _write_weights(output, exported_tensors)

    restore_qwen4_exp_fixed_assets(str(source), str(output), language_model_only=True)

    _, tensors = _read_indexed_weights(output)
    assert tensors[key].dtype == torch.float32
    torch.testing.assert_close(tensors[key], exported_tensors[key], rtol=0, atol=0)


@pytest.mark.parametrize('bad_path', ['../source/model.safetensors', '/outside/model.safetensors'])
@pytest.mark.parametrize('side', ['source', 'output'])
def test_restore_rejects_nonlocal_index_shard_paths(checkpoints, bad_path, side):
    source, output, source_tensors, exported_tensors = checkpoints
    directory = source if side == 'source' else output
    tensors = source_tensors if side == 'source' else exported_tensors
    _write_weights(directory, tensors, indexed=True)
    index_path = directory / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    index['weight_map']['lm_head.weight'] = bad_path
    index_path.write_text(json.dumps(index))

    with pytest.raises(ValueError, match='Invalid checkpoint shard path'):
        restore_qwen4_exp_fixed_assets(str(source), str(output), language_model_only=True)


@pytest.mark.parametrize('side', ['source', 'output'])
def test_restore_rejects_external_symlink_shard(checkpoints, side):
    source, output, _, _ = checkpoints
    directory, external = (source, output) if side == 'source' else (output, source)
    (directory / 'external.safetensors').symlink_to(external / 'model.safetensors')

    with pytest.raises(ValueError, match='escapes its directory'):
        restore_qwen4_exp_fixed_assets(str(source), str(output), language_model_only=True)


def test_restore_rejects_external_index_symlink(checkpoints):
    source, output, source_tensors, _ = checkpoints
    _write_weights(source, source_tensors, indexed=True)
    (output / 'model.safetensors.index.json').symlink_to(source / 'model.safetensors.index.json')

    with pytest.raises(ValueError, match='Checkpoint index escapes'):
        restore_qwen4_exp_fixed_assets(str(source), str(output), language_model_only=True)


def test_restore_visual_mode_requires_exporter_to_include_vision(checkpoints):
    source, output, _, _ = checkpoints

    with pytest.raises(ValueError, match='missing required non-restorable tensors'):
        restore_qwen4_exp_fixed_assets(str(source), str(output), language_model_only=False)


@pytest.mark.parametrize('mtp_enabled', [False, True])
def test_restore_mtp_contract_is_explicit(checkpoints, mtp_enabled):
    source, output, source_tensors, exported_tensors = checkpoints
    if not mtp_enabled:
        exported_tensors['mtp.fc_hidden.weight'] = source_tensors['mtp.fc_hidden.weight']
        _write_weights(output, exported_tensors)
    message = ('declared omitted' if not mtp_enabled else 'missing required non-restorable')

    with pytest.raises(ValueError, match=message):
        restore_qwen4_exp_fixed_assets(str(source), str(output), language_model_only=True, mtp_enabled=mtp_enabled)


def test_restore_mtp_whitelist_cannot_hide_missing_text(checkpoints):
    source, output, _, _ = checkpoints

    with pytest.raises(ValueError, match='exact MTP tensor keys only'):
        restore_qwen4_exp_fixed_assets(
            str(source),
            str(output),
            language_model_only=True,
            omitted_mtp_keys=['lm_head.weight'],
        )


def test_restore_rejects_index_that_disagrees_with_tensor_files(checkpoints):
    source, output, _, exported_tensors = checkpoints
    _write_weights(output, exported_tensors, indexed=True)
    index_path = output / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    index['weight_map']['ghost.weight'] = 'model.safetensors'
    index_path.write_text(json.dumps(index))

    with pytest.raises(ValueError, match='index disagrees'):
        restore_qwen4_exp_fixed_assets(str(source), str(output), language_model_only=True)


def test_restore_oversized_tensor_gets_its_own_shard(checkpoints):
    source, output, source_tensors, _ = checkpoints
    source_tensors['model.visual.patch_embed.proj.weight'] = torch.ones(20)
    _write_weights(source, source_tensors)

    report = restore_qwen4_exp_fixed_assets(str(source), str(output), language_model_only=True, max_shard_size_bytes=32)

    for filename in report['new_shards']:
        with safe_open(output / filename, framework='pt', device='cpu') as handle:
            sizes = [handle.get_tensor(key).numel() * handle.get_tensor(key).element_size() for key in handle.keys()]
            assert sum(sizes) <= 32 or len(sizes) == 1


def test_restore_staging_failure_preserves_existing_export(checkpoints, monkeypatch):
    source, output, _, _ = checkpoints
    original_save = mcore_bridge_checkpoint.save_file
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    calls = 0

    def fail_second_shard(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('disk full')
        return original_save(*args, **kwargs)

    monkeypatch.setattr(mcore_bridge_checkpoint, 'save_file', fail_second_shard)

    with pytest.raises(OSError, match='disk full'):
        restore_qwen4_exp_fixed_assets(str(source), str(output), language_model_only=True, max_shard_size_bytes=32)

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_restore_index_publish_failure_rolls_back_new_shards(checkpoints, monkeypatch):
    source, output, _, _ = checkpoints
    original_replace = mcore_bridge_checkpoint.os.replace
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    def fail_index_publish(src, dst):
        if Path(dst).name == 'model.safetensors.index.json':
            raise OSError('cannot publish index')
        return original_replace(src, dst)

    monkeypatch.setattr(mcore_bridge_checkpoint.os, 'replace', fail_index_publish)

    with pytest.raises(OSError, match='cannot publish index'):
        restore_qwen4_exp_fixed_assets(str(source), str(output), language_model_only=True)

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize('invalid', [None, 'missing', 'mixed', 'mixed_float', 'shape', 'stale_scale', 'non_ple'])
def test_restore_fp8_ple_conversion_requires_complete_float_export(checkpoints, invalid, dtype):
    source, output, source_tensors, exported_tensors = checkpoints
    prefix = 'model.language_model.layers.0.ple.ple_embedding.ngram_embedding'
    shards = [f"{prefix}.shard_{index}.weight" for index in range(2)]
    scale = f"{prefix}.weight_scale"
    for key in shards:
        source_tensors[key] = torch.ones(2, 2).to(torch.float8_e4m3fn)
        exported_tensors[key] = torch.full((2, 2), 7.0, dtype=dtype)
    source_tensors[scale] = torch.tensor(0.5)
    if invalid == 'missing':
        del exported_tensors[shards[1]]
    elif invalid == 'mixed':
        exported_tensors[shards[1]] = source_tensors[shards[1]]
    elif invalid == 'mixed_float':
        other_dtype = torch.float32 if dtype != torch.float32 else torch.float16
        exported_tensors[shards[1]] = exported_tensors[shards[1]].to(other_dtype)
    elif invalid == 'shape':
        exported_tensors[shards[1]] = torch.ones(1, 2, dtype=dtype)
    elif invalid == 'stale_scale':
        exported_tensors[scale] = source_tensors[scale]
    elif invalid == 'non_ple':
        source_tensors['lm_head.weight'] = source_tensors['lm_head.weight'].to(torch.float8_e4m3fn)
    _write_weights(source, source_tensors)
    _write_weights(output, exported_tensors)

    if invalid is not None:
        with pytest.raises(ValueError, match='PLE conversion|dtype differs'):
            restore_qwen4_exp_fixed_assets(str(source), str(output), language_model_only=True)
        assert not list(output.glob('model-fixed-visual-*.safetensors'))
        return

    restore_qwen4_exp_fixed_assets(str(source), str(output), language_model_only=True)

    index, tensors = _read_indexed_weights(output)
    assert scale not in index['weight_map']
    assert set(tensors) == set(source_tensors) - {scale, 'mtp.fc_hidden.weight'}
    for key in shards:
        assert tensors[key].dtype == dtype
        torch.testing.assert_close(tensors[key], exported_tensors[key], rtol=0, atol=0)


@pytest.fixture
def ple_checkpoint():
    config = SimpleNamespace(
        hf_model_type='qwen4_exp',
        ngram_size=3,
        heads_per_ngram=1,
        ple_embed_dim=4,
        split_ngram_parts=3,
        make_ngram_vocab_size_divisible_by=4,
        ple_layer_ids=[2],
    )
    tensors = {
        PLE_PREFIX + 'layer_multipliers': torch.tensor([11, 13, 17]),
        PLE_PREFIX + 'ngram_heads_offsets': torch.tensor([0, 5]),
        PLE_PREFIX + 'ngram_heads_vocab_sizes': torch.tensor([5, 7]),
        PLE_PREFIX + 'ngram_embedding.weight_scale': torch.tensor(0.25),
    }
    for part in range(config.split_ngram_parts):
        tensors[f"{PLE_PREFIX}ngram_embedding.shard_{part}.weight"] = (
            torch.arange(8, dtype=torch.float32).reshape(4, 2).to(torch.float8_e4m3fn))
    return config, tensors


@pytest.mark.parametrize('indexed', [False, True])
def test_ple_checkpoint_complete_assets_validate_without_loading_tables(tmp_path, ple_checkpoint, indexed):
    config, tensors = ple_checkpoint
    save_file(tensors, tmp_path / 'model.safetensors')
    if indexed:
        (tmp_path / 'model.safetensors.index.json').write_text(
            json.dumps({'weight_map': dict.fromkeys(tensors, 'model.safetensors')}))

    manifest = validate_ple_checkpoint(str(tmp_path), config, LAYERS_PREFIX)

    assert set(manifest) == set(tensors)
    buffer_key = PLE_PREFIX + 'layer_multipliers'
    assert (manifest[buffer_key]['sha256'] == hashlib.sha256(tensors[buffer_key].numpy().tobytes()).hexdigest())
    shard_key = PLE_PREFIX + 'ngram_embedding.shard_0.weight'
    assert manifest[shard_key] == {'shape': [4, 2], 'dtype': 'F8_E4M3'}


@pytest.mark.parametrize(
    'missing_key',
    [
        'ngram_embedding.shard_1.weight',
        'ngram_embedding.weight_scale',
        'layer_multipliers',
    ],
)
def test_ple_checkpoint_missing_required_asset_raises(tmp_path, ple_checkpoint, missing_key):
    config, tensors = ple_checkpoint
    del tensors[PLE_PREFIX + missing_key]
    save_file(tensors, tmp_path / 'model.safetensors')

    with pytest.raises(ValueError, match='Missing required PLE checkpoint tensor'):
        validate_ple_checkpoint(str(tmp_path), config, LAYERS_PREFIX)


def test_ple_checkpoint_index_ghost_tensor_raises(tmp_path, ple_checkpoint):
    config, tensors = ple_checkpoint
    weight_map = dict.fromkeys(tensors, 'model.safetensors')
    del tensors[PLE_PREFIX + 'ngram_embedding.shard_1.weight']
    save_file(tensors, tmp_path / 'model.safetensors')
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weight_map}))

    with pytest.raises(ValueError, match='index points to an absent tensor'):
        validate_ple_checkpoint(str(tmp_path), config, LAYERS_PREFIX)


@pytest.mark.parametrize(
    ('key', 'replacement', 'message'),
    [
        ('ngram_embedding.shard_1.weight', torch.ones(3, 2), 'Invalid PLE shard shape'),
        ('ngram_heads_offsets', torch.tensor([0, 4]), 'Invalid PLE hash-table'),
        (
            'layer_multipliers',
            torch.tensor([11, 13, 17], dtype=torch.int32),
            'hash-buffer',
        ),
        (
            'ngram_embedding.weight_scale',
            torch.tensor(float('nan')),
            'finite and positive',
        ),
        ('ngram_embedding.weight_scale', torch.tensor(0.0), 'finite and positive'),
        ('ngram_embedding.weight_scale', torch.ones(2), 'must be scalar'),
    ],
)
def test_ple_checkpoint_corrupt_metadata_raises(tmp_path, ple_checkpoint, key, replacement, message):
    config, tensors = ple_checkpoint
    tensors[PLE_PREFIX + key] = replacement
    save_file(tensors, tmp_path / 'model.safetensors')

    with pytest.raises(ValueError, match=message):
        validate_ple_checkpoint(str(tmp_path), config, LAYERS_PREFIX)


def test_ple_checkpoint_dequantized_table_without_scale_is_valid(tmp_path, ple_checkpoint):
    config, tensors = ple_checkpoint
    del tensors[PLE_PREFIX + 'ngram_embedding.weight_scale']
    for key in tensors:
        if '.shard_' in key:
            tensors[key] = tensors[key].to(torch.bfloat16)
    save_file(tensors, tmp_path / 'model.safetensors')

    manifest = validate_ple_checkpoint(str(tmp_path), config, LAYERS_PREFIX)

    assert manifest[PLE_PREFIX + 'ngram_embedding.shard_0.weight']['dtype'] == 'BF16'


def test_ple_checkpoint_hash_must_match_model_configuration(tmp_path, ple_checkpoint):
    config, tensors = ple_checkpoint
    save_file(tensors, tmp_path / 'model.safetensors')
    layer = torch.nn.Module()
    layer.layer_number = 2
    layer.ple = torch.nn.Module()
    layer.ple.ple_embedding = torch.nn.Module()
    for key, value in tensors.items():
        name = key.removeprefix(PLE_PREFIX)
        if '.' not in name:
            layer.ple.ple_embedding.register_buffer(name, value.clone())
    validate_ple_checkpoint(str(tmp_path), config, LAYERS_PREFIX, [layer])
    layer.ple.ple_embedding.layer_multipliers[0] += 1

    with pytest.raises(ValueError, match='disagrees with the model configuration'):
        validate_ple_checkpoint(str(tmp_path), config, LAYERS_PREFIX, [layer])


class _SourceConfig(PretrainedConfig):
    model_type = 'qwen4_exp'


def _source_config():
    return _SourceConfig(
        text_config=PretrainedConfig(
            mtp={
                'hybrid': True,
                'num_hidden_layers': 1,
                'layer_types': ['full_attention'],
            },
            mtp_num_hidden_layers=1,
            mtp_use_dedicated_embeddings=False,
        ))


def test_export_config_removes_mtp_on_a_copy_only(tmp_path):
    config = _source_config()
    before = config.to_dict()

    exported = qwen4_exp_export_config(config, mtp_enabled=False)
    exported.save_pretrained(tmp_path)

    saved = json.loads((tmp_path / 'config.json').read_text())
    assert saved['text_config']['mtp'] is None
    assert saved['text_config']['mtp_num_hidden_layers'] == 0
    assert config.to_dict() == before
    assert exported.text_config is not config.text_config


def test_enabled_mtp_export_preserves_configuration_without_aliasing():
    config = _source_config()

    exported = qwen4_exp_export_config(config, mtp_enabled=True)

    assert exported.to_dict() == config.to_dict()
    exported.text_config.mtp['num_hidden_layers'] = 9
    assert config.text_config.mtp['num_hidden_layers'] == 1


@pytest.mark.parametrize('mtp_enabled', [False, True])
def test_export_qsa_uses_portable_layer_names_without_mutating_runtime(tmp_path, mtp_enabled):
    config = _source_config()
    config.text_config.layer_types = ['linear_attention', 'qwen_sparse_attention']
    before = config.to_dict()

    exported = qwen4_exp_export_config(config, mtp_enabled=mtp_enabled)
    exported.save_pretrained(tmp_path)

    saved = json.loads((tmp_path / 'config.json').read_text())
    assert saved['text_config']['layer_types'] == [
        'linear_attention',
        'full_attention',
    ]
    assert config.to_dict() == before


def test_exact_qwen4_exp_hf_tiny_export_config_has_no_mtp(tmp_path):
    if importlib.util.find_spec('transformers.models.qwen4_exp') is None:
        pytest.skip('Qwen4-Exp requires the pinned Transformers runtime')
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration

    config = Qwen4ExpConfig(
        text_config={
            'vocab_size': 32,
            'hidden_size': 16,
            'num_hidden_layers': 4,
            'num_attention_heads': 2,
            'num_key_value_heads': 1,
            'head_dim': 8,
            'linear_key_head_dim': 8,
            'linear_value_head_dim': 8,
            'linear_num_key_heads': 1,
            'linear_num_value_heads': 1,
            'num_experts': 2,
            'num_experts_per_tok': 1,
            'moe_intermediate_size': 8,
            'shared_expert_intermediate_size': 8,
            'hc_count': 2,
            'hc_lowrank': 4,
            'ple_layer_ids': [2],
            'ple_embed_dim': 4,
            'heads_per_ngram': 1,
            'ngram_vocab_size_base': 11,
            'make_ngram_vocab_size_divisible_by': 4,
            'split_ngram_parts': 2,
            'eos_token_id': 2,
            'indexer_n_heads': 1,
            'indexer_kv_heads': 1,
            'indexer_head_dim': 8,
            'indexer_budget': 4,
            'indexer_compress_ratio': 2,
            'mtp': {
                'num_hidden_layers': 1
            },
            'mtp_num_hidden_layers': 1,
            'rope_parameters': {
                'rope_type': 'default',
                'rope_theta': 10000.0
            },
        },
        vision_config={
            'depth': 1,
            'hidden_size': 16,
            'intermediate_size': 16,
            'num_heads': 4,
            'patch_size': 2,
            'temporal_patch_size': 1,
            'out_hidden_size': 16,
            'num_position_embeddings': 16,
        },
    )
    before = config.to_dict()
    exported = qwen4_exp_export_config(config, mtp_enabled=False)
    exported.save_pretrained(tmp_path)
    reloaded = Qwen4ExpConfig.from_pretrained(tmp_path)

    model = Qwen4ExpForConditionalGeneration._from_config(reloaded, attn_implementation='eager')

    assert not any('mtp' in name.split('.') for name, _ in model.named_modules())
    assert reloaded.text_config.mtp is None
    assert reloaded.text_config.mtp_num_hidden_layers == 0
    assert config.to_dict() == before
