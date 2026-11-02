# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU DCP roundtrips and malformed/incomplete-export regressions."""

import json
import math
import pickle
import pytest
import torch
import torch.distributed.checkpoint as dcp
import weakref
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint.metadata import (ChunkStorageMetadata, Metadata, MetadataIndex, TensorProperties,
                                                   TensorStorageMetadata)
from types import SimpleNamespace

from mcore_bridge.checkpoint import export as export_module
from mcore_bridge.checkpoint import export_checkpoint
from mcore_bridge.checkpoint.dcp_stream import DcpReader, headers, verify_digest, write_shard
from mcore_bridge.checkpoint.deepseek_v41_mapping import Piece, Target, check_partition, flat_chunk


@pytest.fixture
def fixture(tmp_path):
    checkpoint, base = tmp_path / 'checkpoint', tmp_path / 'base'
    checkpoint.mkdir()
    base.mkdir()
    text = {
        'num_hidden_layers': 2,
        'n_routed_experts': 2,
        'hidden_size': 2,
        'hc_mult': 2,
        'vocab_size': 4,
        'kv_source_layer_ids': [0],
        'moe_intermediate_size': 3,
        'engram_layer_ids': [0],
        'engram_max_ngram_size': 2,
        'engram_n_heads': 2,
        'engram_head_dim': 2,
        'engram_num_embeddings': [8]
    }
    tensors, expected = {}, {}

    def add(source, target, shape, dtype=torch.bfloat16):
        value = (torch.arange(math.prod(shape)).reshape(shape) + len(tensors) * 3).to(dtype)
        tensors['language_model.' + source] = value
        if target:
            expected[target] = value
        return value

    add('embedding.word_embeddings.weight', 'embed.weight', (4, 2))
    add('output_layer.weight', 'head.weight', (4, 2))
    add('decoder.final_norm.weight', 'norm.weight', (2, ))
    for layer in range(2):
        hf = f'layers.{layer}.'
        for parity, channel in ((0, 'attn'), (1, 'ffn')):
            prefix = f'decoder.layers.{2 * layer + parity}.'
            add(prefix + 'hyper_connection.mapping_proj.weight', hf + f'hc_{channel}_fn', (8, 4), torch.float32)
            add(prefix + 'hyper_connection.bias', hf + f'hc_{channel}_base', (8, ), torch.float32)
            scales = [
                add(prefix + 'hyper_connection.alpha_' + suffix, None, (1, ), torch.float32)
                for suffix in ('pre', 'post', 'res')
            ]
            expected[hf + f'hc_{channel}_scale'] = torch.cat(scales)
            inner = prefix + 'inner_layer.'
            if parity == 0:
                for source, target, shape in (
                    ('input_layernorm.weight', 'attn_norm.weight', (2, )),
                    ('self_attention.linear_q_down_proj.weight', 'attn.wq_a.weight', (2, 2)),
                    ('self_attention.linear_o_group_proj', 'attn.wo_a.weight', (2, 2)),
                    ('self_attention.core_attention.indexer.linear_wk.weight', 'attn.indexer.wk.weight', (2, 2)),
                ):
                    add(inner + source, hf + target, shape)
            else:
                add(inner + 'pre_mlp_layernorm.weight', hf + 'ffn_norm.weight', (2, ))
                add(inner + 'mlp.router.weight', hf + 'ffn.gate.weight', (2, 2))
                shared = add(inner + 'mlp.shared_experts.linear_fc1.weight', None, (6, 2))
                expected[hf + 'ffn.shared_experts.w1.weight'] = shared[:3]
                expected[hf + 'ffn.shared_experts.w3.weight'] = shared[3:]
                add(inner + 'mlp.shared_experts.linear_fc2.weight', hf + 'ffn.shared_experts.w2.weight', (2, 3))
                fc1 = add(inner + 'mlp.experts.experts.linear_fc1.weight', None, (2, 6, 2))
                fc2 = add(inner + 'mlp.experts.experts.linear_fc2.weight', None, (2, 2, 3))
                for expert in range(2):
                    expected[hf + f'ffn.experts.{expert}.w1.weight'] = fc1[expert, :3]
                    expected[hf + f'ffn.experts.{expert}.w3.weight'] = fc1[expert, 3:]
                    expected[hf + f'ffn.experts.{expert}.w2.weight'] = fc2[expert]
    prefix = 'decoder.layers.0.inner_layer.engram.'
    tables = [add(prefix + f'embedding.tables.{i}.weight', None, (rows, 2)) for i, rows in enumerate((3, 5))]
    expected['layers.0.engram.embed.weight'] = torch.cat(tables)
    key = add(prefix + 'key_projection.weight', None, (4, 4))
    value = add(prefix + 'value_projection.weight', None, (2, 4))
    expected['layers.0.engram.wkv.weight'] = torch.cat([key, value])
    for source, target in (('query_norm', 'q_weight'), ('key_norm', 'k_weight')):
        expected['layers.0.engram.' + target] = add(prefix + source + '.weight', None, (4, )).reshape(2, 2)
    # Model, optimizer and extra-state records coexist; only model tensors are read.
    tensors['optimizer.state.large'] = torch.full((16, ), float('nan'))
    tensors['language_model.decoder.layers.0._extra_state'] = {'ignored': True}
    base_tensors = {key: torch.zeros_like(value) for key, value in expected.items()}
    packed = 'layers.0.ffn.experts.0.w1.weight'
    base_tensors[packed] = torch.zeros((3, 1), dtype=torch.int8)
    base_tensors[packed.removesuffix('.weight') + '.scale'] = torch.ones((1, 1))
    base_tensors['mtp.weight'] = torch.zeros(2)
    base_tensors['vision.weight'] = torch.ones(2, dtype=torch.bfloat16)
    save_file(base_tensors, str(base / 'base.safetensors'))
    (base / 'model.safetensors.index.json').write_text(
        json.dumps({'weight_map': {
            key: 'base.safetensors'
            for key in base_tensors
        }}))
    (base / 'config.json').write_text(
        json.dumps({
            'model_type': 'deepseek_v41',
            'text_config': text,
            'quantization_config': {
                'quant_method': 'fp8'
            },
            'eos_token_id': 1
        }))
    for asset in ('tokenizer.json', 'tokenizer_config.json'):
        (base / asset).write_text('{}')
    (base / 'generation_config.json').write_text('{"eos_token_id": 1, "temperature": 0.75}')
    (base / 'encoding').mkdir()
    (base / 'encoding' / 'reference.py').write_text('# fixture\n')
    (checkpoint / 'args.json').write_text(
        json.dumps({
            'model_type': 'deepseek_v41',
            'model': str(base),
            'tuner_type': 'full',
            'bf16': True,
            'language_model_only': True,
            'tensor_model_parallel_size': 1,
            'save_safetensors': False
        }))
    (checkpoint / 'latest_checkpointed_iteration.txt').write_text('1')
    return SimpleNamespace(
        checkpoint=checkpoint, base=base, tensors=tensors, expected=expected, output=tmp_path / 'hf', text=text)


def save_chunks(fixture, prefix='language_model.'):
    """Write actual tensor serialization with deliberately reversed EP/PP row metadata."""
    directory = fixture.checkpoint / 'iter_0000001'
    directory.mkdir()
    specs, storage = {}, {}
    for number, (key, value) in enumerate(fixture.tensors.items()):
        if not isinstance(value, torch.Tensor):
            continue
        key = prefix + key.removeprefix('language_model.') if key.startswith('language_model.') else key
        shape = tuple(value.shape)
        split = max(1, shape[0] // 2)
        cuts = (0, split, shape[0]) if split < shape[0] else (0, shape[0])
        chunks = []
        for part, (lo, hi) in enumerate(zip(cuts, cuts[1:])):
            filename = f'__{number}_{part}.distcp'
            offsets, sizes = (lo, ) + (0, ) * (value.ndim - 1), (hi - lo, ) + shape[1:]
            torch.save(value[lo:hi].clone(), directory / filename)
            storage[MetadataIndex(key, torch.Size(offsets))] = SimpleNamespace(
                relative_path=filename, offset=0, length=(directory / filename).stat().st_size)
            chunks.append(ChunkStorageMetadata(torch.Size(offsets), torch.Size(sizes)))
        specs[key] = TensorStorageMetadata(TensorProperties(dtype=value.dtype), value.size(), chunks[::-1])
    (directory / '.metadata').write_bytes(pickle.dumps(Metadata(state_dict_metadata=specs, storage_data=storage)))
    return directory


@pytest.mark.parametrize('prefix', ['language_model.', 'model.language_model.', 'model0.language_model.', ''])
@pytest.mark.parametrize('writers', [1, 2])
def test_real_dcp_save_full_roundtrip(fixture, prefix, writers):
    state = {
        prefix + key.removeprefix('language_model.') if key.startswith('language_model.') else key: value
        for key, value in fixture.tensors.items()
    }
    dcp.save(state, checkpoint_id=fixture.checkpoint / 'iter_0000001')
    plan = export_checkpoint(fixture.checkpoint, fixture.base, fixture.output, preflight=True)
    assert not fixture.output.exists()
    assert plan['source_prefix'] == prefix
    report = export_checkpoint(fixture.checkpoint, fixture.base, fixture.output, workers=writers, max_shard_size=64)
    assert report['passed'] and report['independent_dcp_samples']['passed']
    assert (fixture.output / '_SUCCESS').exists()
    actual, weight_map = headers(fixture.output)
    assert set(actual) == set(fixture.expected) | {'vision.weight'}
    for key, expected in fixture.expected.items():
        with safe_open(str(fixture.output / weight_map[key]), framework='pt') as handle:
            value = handle.get_tensor(key)
        assert value.dtype == expected.dtype
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    config = json.loads((fixture.output / 'config.json').read_text())
    assert 'quantization_config' not in config
    assert config['text_config']['num_nextn_predict_layers'] == 0
    assert json.loads((fixture.output / 'generation_config.json').read_text())['temperature'] == 0.75
    assert json.loads((fixture.base / 'config.json').read_text())['quantization_config']
    for record in report['shards']:
        verify_digest(fixture.output / record['file'], record['sha256'])
    with pytest.raises(FileExistsError):
        export_checkpoint(fixture.checkpoint, fixture.base, fixture.output)


def test_shuffled_ep_chunks_and_oversized_engram(fixture):
    save_chunks(fixture)
    report = export_checkpoint(fixture.checkpoint, fixture.base, fixture.output, max_shard_size=16)
    _, weight_map = headers(fixture.output)
    key = 'layers.0.engram.embed.weight'
    with safe_open(str(fixture.output / weight_map[key]), framework='pt') as handle:
        assert list(handle.keys()) == [key]
        torch.testing.assert_close(handle.get_tensor(key), fixture.expected[key], rtol=0, atol=0)
    assert report['independent_dcp_samples']['passed']


def test_releases_record_before_loading_next(fixture, monkeypatch):
    directory = save_chunks(fixture)
    reader = DcpReader(directory)
    key = 'language_model.embedding.word_embeddings.weight'
    previous = []
    load = torch.load

    def tracked_load(*args, **kwargs):
        assert not previous or previous[-1]() is None
        value = load(*args, **kwargs)
        previous.append(weakref.ref(value))
        return value

    monkeypatch.setattr(torch, 'load', tracked_load)
    target = Target('embed.weight', (4, 2), 'torch.bfloat16', (Piece(key, 0, 8), ))
    write_shard(fixture.output.with_suffix('.safetensors'), [target], reader)
    assert len(previous) == 2 and all(value() is None for value in previous)


@pytest.mark.parametrize('field,value', [('tensor_model_parallel_size', 2), ('bf16', False), ('fp8', 'hybrid'),
                                         ('mtp_num_layers', 1), ('dspark_num_layers', 1), ('tuner_type', 'lora'),
                                         ('language_model_only', False), ('use_megatron_fsdp', True)])
def test_unsupported_training_args_before_output(fixture, field, value):
    args_path = fixture.checkpoint / 'args.json'
    args = json.loads(args_path.read_text())
    args[field] = value
    args_path.write_text(json.dumps(args))
    with pytest.raises(ValueError):
        export_checkpoint(fixture.checkpoint, fixture.base, fixture.output)
    assert not fixture.output.exists()


@pytest.mark.parametrize('problem', ['hole', 'overlap', 'truncated', 'path', 'dtype', 'unknown'])
def test_bad_metadata_before_output(fixture, problem):
    directory = save_chunks(fixture)
    path = directory / '.metadata'
    metadata = pickle.loads(path.read_bytes())
    key = 'language_model.embedding.word_embeddings.weight'
    spec = metadata.state_dict_metadata[key]
    if problem == 'hole':
        spec.chunks = spec.chunks[:1]
    elif problem == 'overlap':
        spec.chunks.append(spec.chunks[0])
    elif problem == 'dtype':
        spec.properties.dtype = torch.float16
    elif problem == 'unknown':
        metadata.state_dict_metadata['language_model.unknown'] = spec
        for index, location in list(metadata.storage_data.items()):
            if index.fqn == key:
                metadata.storage_data[MetadataIndex('language_model.unknown', index.offset)] = location
    else:
        location = next(iter(metadata.storage_data.values()))
        if problem == 'path':
            location.relative_path = '../escape.distcp'
        else:
            (directory / location.relative_path).write_bytes(b'')
    path.write_bytes(pickle.dumps(metadata))
    with pytest.raises(ValueError):
        export_checkpoint(fixture.checkpoint, fixture.base, fixture.output)
    assert not fixture.output.exists()


@pytest.mark.parametrize('problem', ['nan', 'write', 'readback', 'sample'])
def test_failure_never_marks_success(fixture, monkeypatch, problem):
    if problem == 'nan':
        fixture.tensors['language_model.embedding.word_embeddings.weight'][0, 0] = float('nan')
    save_chunks(fixture)

    def fail(*args, **kwargs):
        raise OSError('injected interrupted export')

    if problem == 'write':
        monkeypatch.setattr(export_module, 'write_shard', fail)
    elif problem == 'readback':
        monkeypatch.setattr(export_module, 'verify_digest', fail)
    elif problem == 'sample':
        build, check, _ = export_module._EXPORTERS['deepseek_v41']
        monkeypatch.setitem(export_module._EXPORTERS, 'deepseek_v41', (build, check, fail))
    with pytest.raises((ValueError, OSError)):
        export_checkpoint(fixture.checkpoint, fixture.base, fixture.output)
    assert fixture.output.exists()
    assert not (fixture.output / '_SUCCESS').exists()
    with pytest.raises(FileExistsError):
        export_checkpoint(fixture.checkpoint, fixture.base, fixture.output)


def test_interval_and_unsupported_tp_layout():
    for intervals in ([(0, 2), (3, 4)], [(0, 3), (2, 4)]):
        with pytest.raises(ValueError):
            check_partition(intervals, 4, 'test')
    assert flat_chunk((17, 4), (9, 0), (8, 4)) == (36, 32)
    with pytest.raises(ValueError):
        flat_chunk((17, 4), (0, 1), (17, 2))
