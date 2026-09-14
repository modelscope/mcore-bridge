"""Verify that `save_weights(save_missing_weights=True)` restores tensors that
Megatron never materializes.

The scenario mirrors DeepSeek-V4-Flash-0731, whose `mtp.*` (DSpark) weights are
not supported by Megatron: without the restore step they silently disappear from
the exported checkpoint.
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# The megatron entrypoint initializes torch.distributed via env:// rendezvous.
os.environ.setdefault('RANK', '0')
os.environ.setdefault('LOCAL_RANK', '0')
os.environ.setdefault('WORLD_SIZE', '1')
os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '29513')

import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

MODEL_ID = 'Qwen/Qwen2-0.5B-Instruct'
MODEL_TYPE = 'qwen2'  # a copied dir loses the model id, so type/template cannot be inferred
TEMPLATE = 'qwen'
# Stand-ins for weights of a submodule Megatron does not know about.
EXTRA_WEIGHTS = {
    'mtp.0.main_proj.weight': torch.randn(8, 24, dtype=torch.bfloat16),
    'mtp.0.main_norm.weight': torch.randn(8, dtype=torch.bfloat16),
    'mtp.2.markov_head.markov_w1.weight': torch.randn(16, 4, dtype=torch.bfloat16),
}


def _build_checkpoint_with_extra_weights(dst_dir: str) -> str:
    """Copy the source model and inject weights that the bridge cannot consume."""
    from swift import safe_snapshot_download
    src_dir = safe_snapshot_download(MODEL_ID)
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    index_path = os.path.join(dst_dir, 'model.safetensors.index.json')
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_name = sorted(set(index['weight_map'].values()))[0]
    else:
        index = None
        shard_name = 'model.safetensors'

    shard_path = os.path.join(dst_dir, shard_name)
    state_dict = load_file(shard_path)
    state_dict.update(EXTRA_WEIGHTS)
    save_file(state_dict, shard_path, metadata={'format': 'pt'})

    if index is not None:
        for key in EXTRA_WEIGHTS:
            index['weight_map'][key] = shard_name
        with open(index_path, 'w') as f:
            json.dump(index, f)
    return dst_dir


def _export(model_dir: str, output_dir: str, save_missing_weights: bool):
    """Round-trip HF -> mcore -> HF through the real bridge."""
    from swift.megatron import MegatronExportArguments, megatron_export_main
    mcore_dir = f'{output_dir}-mcore'
    megatron_export_main(
        MegatronExportArguments(
            model=model_dir,
            model_type=MODEL_TYPE,
            template=TEMPLATE,
            to_mcore=True,
            output_dir=mcore_dir,
            exist_ok=True,
            torch_dtype='bfloat16',
        ))
    megatron_export_main(
        MegatronExportArguments(
            mcore_model=mcore_dir,
            # `model` is only used as the source of the weights Megatron cannot export.
            model=model_dir,
            model_type=MODEL_TYPE,
            template=TEMPLATE,
            to_hf=True,
            output_dir=output_dir,
            exist_ok=True,
            torch_dtype='bfloat16',
            save_missing_weights=save_missing_weights,
        ))
    return output_dir


def _load_exported(output_dir: str):
    index_path = os.path.join(output_dir, 'model.safetensors.index.json')
    state_dict = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            shards = sorted(set(json.load(f)['weight_map'].values()))
    else:
        shards = ['model.safetensors']
    for shard in shards:
        state_dict.update(load_file(os.path.join(output_dir, shard)))
    return state_dict


def test_save_missing_weights():
    """The injected weights must reappear byte-for-byte when the flag is on."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = _build_checkpoint_with_extra_weights(os.path.join(tmp_dir, 'src'))
        output_dir = _export(model_dir, os.path.join(tmp_dir, 'restored'), save_missing_weights=True)
        state_dict = _load_exported(output_dir)

        for key, expected in EXTRA_WEIGHTS.items():
            assert key in state_dict, f'{key} was not restored'
            assert torch.equal(state_dict[key], expected), f'{key} was altered during restore'
        # The regular weights must still be exported by Megatron, not copied blindly.
        assert 'model.layers.0.self_attn.q_proj.weight' in state_dict


def test_save_missing_weights_disabled():
    """With the flag off the behaviour is unchanged: the extra weights are dropped."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = _build_checkpoint_with_extra_weights(os.path.join(tmp_dir, 'src'))
        output_dir = _export(model_dir, os.path.join(tmp_dir, 'dropped'), save_missing_weights=False)
        state_dict = _load_exported(output_dir)

        for key in EXTRA_WEIGHTS:
            assert key not in state_dict, f'{key} leaked into the export'
        assert 'model.layers.0.self_attn.q_proj.weight' in state_dict


if __name__ == '__main__':
    test_save_missing_weights()
    test_save_missing_weights_disabled()
