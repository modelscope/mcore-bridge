# Copyright (c) ModelScope Contributors. All rights reserved.
"""Re-exporting into the same directory must not leave an old shard set behind.

`safetensors` loaders read `model.safetensors.index.json` in preference to
`model.safetensors`, so shards from a larger earlier export kept being loaded
after a re-export that fits in fewer files: the reload returned the old tensors
(same keys, old values) instead of the ones just written.
"""
import json
import pytest
import torch

from mcore_bridge.utils.safetensors import SafetensorLazyLoader, StreamingSafetensorSaver


def _files(save_dir):
    return sorted(path.name for path in save_dir.iterdir())


def _previous_export(save_dir, shard_count, keys_per_shard):
    """Write an index plus `shard_count` shards, as an earlier run would."""
    from safetensors.torch import save_file

    weight_map = {}
    index = 0
    for shard in range(1, shard_count + 1):
        name = f'model-{shard:05d}-of-{shard_count:05d}.safetensors'
        shard_tensors = {}
        for _ in range(keys_per_shard):
            shard_tensors[f'w{index}'] = torch.zeros(3)
            weight_map[f'w{index}'] = name
            index += 1
        save_file(shard_tensors, str(save_dir / name))
    (save_dir / 'model.safetensors.index.json').write_text(
        json.dumps({
            'metadata': {
                'total_size': index * 12
            },
            'weight_map': weight_map,
        }))
    return weight_map


def _saver(save_dir, max_shard_size):
    saver = StreamingSafetensorSaver(save_dir=str(save_dir), max_shard_size='5GB')
    # The constructor's string form only accepts whole GB; set bytes directly so the
    # tests can force shard boundaries with tiny tensors.
    saver.max_shard_size = max_shard_size
    return saver


def test_single_file_reexport_drops_stale_index_and_shards(tmp_path):
    stale = _previous_export(tmp_path, shard_count=2, keys_per_shard=2)

    saver = _saver(tmp_path, max_shard_size=1000)
    saver.add_tensor('w0', torch.arange(4, dtype=torch.float32))
    saver.finalize()

    assert _files(tmp_path) == ['model.safetensors']
    assert set(SafetensorLazyLoader(str(tmp_path))._weight_map) == {'w0'}
    loaded = SafetensorLazyLoader(str(tmp_path)).get_state_dict()['w0'].load()
    assert loaded.tolist() == [0.0, 1.0, 2.0, 3.0]
    # The stale export must not stay reachable through the old index.
    assert not (tmp_path / 'model.safetensors.index.json').exists()
    assert stale  # the earlier export really did write shards


def test_fewer_shards_reexport_drops_extra_shards(tmp_path):
    _previous_export(tmp_path, shard_count=4, keys_per_shard=1)
    before = set(_files(tmp_path))

    saver = _saver(tmp_path, max_shard_size=30)  # two 12-byte tensors per shard
    for index in range(4):
        saver.add_tensor(f'w{index}', torch.zeros(3, dtype=torch.float32))
    saver.finalize()

    after = set(_files(tmp_path))
    weight_map = json.loads((tmp_path / 'model.safetensors.index.json').read_text())['weight_map']
    assert set(weight_map.values()) == {'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'}
    assert after == set(weight_map.values()) | {'model.safetensors.index.json'}
    assert before - after, 'the stale shards of the 4-shard export should be gone'


def test_reexport_keeps_the_files_it_just_wrote(tmp_path):
    _previous_export(tmp_path, shard_count=2, keys_per_shard=2)

    saver = _saver(tmp_path, max_shard_size=1000)
    saver.add_tensor('w0', torch.zeros(3))
    saver.finalize()
    saver = _saver(tmp_path, max_shard_size=1000)
    saver.add_tensor('w0', torch.ones(3))
    saver.finalize()

    assert _files(tmp_path) == ['model.safetensors']
    assert SafetensorLazyLoader(str(tmp_path)).get_state_dict()['w0'].load().tolist() == [1.0, 1.0, 1.0]


def test_adapter_export_leaves_model_files_alone(tmp_path):
    """PEFT exports go to adapter_model.safetensors and must not delete model weights."""
    from safetensors.torch import save_file

    save_file({'w0': torch.zeros(3)}, str(tmp_path / 'model.safetensors'))
    saver = _saver(tmp_path, max_shard_size=1000)
    saver.peft_format = True
    saver.add_tensor('lora_a', torch.zeros(2))
    saver.finalize()

    assert {'model.safetensors', 'adapter_model.safetensors'} <= set(_files(tmp_path))
