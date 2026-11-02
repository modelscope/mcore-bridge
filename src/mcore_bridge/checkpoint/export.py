# Copyright (c) ModelScope Contributors. All rights reserved.
# Adapted from https://github.com/DongHande/EchoCraft, Apache-2.0.
# Revision 59720b8449b8debef77133dd4496e9a9381a7f67, megatron_to_hf/export_deepseek_v41.py.
"""Export a supported Megatron DCP checkpoint without constructing a model."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import copy
import hashlib
import json
import math
import os
import shutil
import time
import torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

from .dcp_stream import DTYPES, DcpReader, headers, verify_digest, write_shard
from .deepseek_v41_mapping import Target, build_targets, check_base_layout
from .deepseek_v41_samples import verify_samples

_EXPORTERS = {'deepseek_v41': (build_targets, check_base_layout, verify_samples)}


def _dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def _plan_shards(targets: list[Target], limit: int) -> list[list[Target]]:
    shards, pending, size = [], [], 0
    for target in targets:
        if pending and size + target.size_bytes > limit:
            shards.append(pending)
            pending, size = [], 0
        pending.append(target)
        size += target.size_bytes
    if pending:
        shards.append(pending)
    return shards


def _copy_assets(base: Path, output: Path, config: dict) -> dict[str, str]:
    hashes = {}
    for name in ('tokenizer.json', 'tokenizer_config.json', 'LICENSE', 'encoding', 'inference',
                 'preprocessor_config.json', 'processor_config.json', 'generation_config.json'):
        source = base / name
        if not source.exists():
            if name.startswith('tokenizer') or name == 'encoding':
                raise ValueError(f'Required base asset missing: {source}')
            continue
        if source.is_dir():
            shutil.copytree(source, output / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        else:
            shutil.copyfile(source, output / name)
        for path in ([output / name] if source.is_file() else sorted((output / name).rglob('*'))):
            if path.is_file():
                relative = str(path.relative_to(output))
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != hashlib.sha256((base / relative).read_bytes()).hexdigest():
                    raise ValueError(f'Asset copy differs: {relative}')
                hashes[relative] = digest
    config = copy.deepcopy(config)
    config.pop('quantization_config', None)
    text = config['text_config']
    text.pop('quantization_config', None)
    config['dtype'] = text['dtype'] = 'bfloat16'
    text['num_nextn_predict_layers'] = 0
    text['dspark_block_size'] = 0
    text['dspark_target_layer_ids'] = []
    _dump(output / 'config.json', config)
    if not (output / 'generation_config.json').exists():
        _dump(output / 'generation_config.json',
              {key: config[key]
               for key in ('bos_token_id', 'eos_token_id', 'pad_token_id') if key in config})
    return hashes


def export_checkpoint(
    checkpoint: str | Path,
    base_model: str | Path,
    output: str | Path,
    *,
    workers: int = 1,
    max_shard_size: int = 5_000_000_000,
    preflight: bool = False,
) -> dict:
    """Export a BF16 full, TP1, language-only DeepSeek V4.1 DCP checkpoint on CPU.

    Args:
        checkpoint: Checkpoint root with Swift args.json and an iteration tracker.
        base_model: Original HF directory supplying layout, unchanged vision and assets.
        output: New output directory. An existing directory is never overwritten.
        workers: Concurrent shard writers. Memory scales with the largest DCP record
            times this value, rather than with full-model or Engram table size.
        max_shard_size: Target shard size in bytes. A larger single tensor stays intact.
        preflight: Validate metadata/layout and return the plan without creating output.

    Returns:
        The export plan for preflight, otherwise the completed validation report.
    """
    if not 1 <= workers <= 8 or max_shard_size < 1:
        raise ValueError('workers must be between 1 and 8 and max_shard_size must be positive')
    checkpoint, base, output = Path(checkpoint).resolve(), Path(base_model).resolve(), Path(output).absolute()
    if output.exists():
        raise FileExistsError(f'Preserving existing output; choose a new directory: {output}')
    training = json.loads((checkpoint / 'args.json').read_text())
    model_type = training.get('model_type')
    if model_type not in _EXPORTERS:
        raise ValueError(f'Unsupported checkpoint model_type: {model_type}')
    if not training.get('language_model_only'):
        raise ValueError('Only language_model_only checkpoints are supported')
    if any(
            training.get(key) for key in ('mtp_num_layers', 'dspark_num_layers', 'fp8', 'fp8_param_gather',
                                          'use_megatron_fsdp', 'save_safetensors')):
        raise ValueError('MTP/DSpark/FP8/FSDP/safetensors checkpoints are not supported')
    if not training.get('bf16') or training.get('tensor_model_parallel_size') != 1:
        raise ValueError('This exporter requires a BF16 TP1 checkpoint')
    if any(training.get(key) not in (None, 'full') for key in ('tuner_type', 'sft_type', 'train_type')):
        raise ValueError('Adapter checkpoints are not supported')
    if training.get('model') and Path(training['model']).resolve() != base:
        raise ValueError('Base model differs from checkpoint args.json')
    iteration = int((checkpoint / 'latest_checkpointed_iteration.txt').read_text().strip())
    if iteration < 1:
        raise ValueError('Checkpoint iteration must be positive')
    reader = DcpReader(checkpoint / f'iter_{iteration:07d}')
    config = json.loads((base / 'config.json').read_text())
    if config.get('model_type') != model_type:
        raise ValueError('Base model_type differs from checkpoint')
    base_headers, base_map = headers(base)
    build, check, verify = _EXPORTERS[model_type]
    targets = build(reader.model, config['text_config'])
    visual = check(targets, base_headers)
    for name in ('tokenizer.json', 'tokenizer_config.json', 'encoding'):
        if not (base / name).exists():
            raise ValueError(f'Required base asset missing: {name}')
    total_size = sum(t.size_bytes for t in targets) + sum(
        math.prod(base_headers[k]['shape']) * {
            'BF16': 2,
            'F32': 4
        }[base_headers[k]['dtype']] for k in visual)
    shards = _plan_shards(targets, max_shard_size)
    plan = {
        'model_type': model_type,
        'checkpoint': str(checkpoint),
        'iteration': iteration,
        'output': str(output),
        'source_prefix': reader.source_prefix,
        'model_tensors': len(reader.model),
        'language_tensors': len(targets),
        'visual_tensors': len(visual),
        'tensor_bytes': total_size,
        'language_shards': len(shards),
        'workers': workers,
        'max_shard_size': max_shard_size,
        'metadata_sha256': reader.metadata_sha256,
        'torch_version': torch.__version__
    }
    if preflight:
        return plan
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < total_size * 1.05:
        raise ValueError('Insufficient output space')
    start = time.monotonic()
    output.mkdir()
    _dump(output / 'export-plan.json', plan)
    records, weight_map = [], {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for index, shard in enumerate(shards, 1):
            name = f'model-{index:05d}-of-{len(shards):05d}.safetensors'
            weight_map.update({t.name: name for t in shard})
            futures.append(pool.submit(write_shard, output / name, shard, reader))
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
            _dump(output / 'export-progress.json', {'completed_shards': len(records), 'total_shards': len(shards)})
    if visual:
        visual_tensors = {}
        for key in visual:
            with safe_open(str(base / base_map[key]), framework='pt', device='cpu') as handle:
                visual_tensors[key] = handle.get_tensor(key)
        name = 'model-visual-base.safetensors'
        save_file(visual_tensors, str(output / name), metadata={'format': 'pt'})
        with safe_open(str(output / name), framework='pt', device='cpu') as handle:
            for key, expected in visual_tensors.items():
                if not torch.isfinite(expected).all().item() or not torch.equal(handle.get_tensor(key), expected):
                    raise ValueError(f'Visual copy differs: {key}')
        del visual_tensors
        weight_map.update({key: name for key in visual})
    _dump(output / 'model.safetensors.index.json', {'metadata': {'total_size': total_size}, 'weight_map': weight_map})
    asset_hashes = _copy_assets(base, output, config)
    actual, actual_map = headers(output)
    if set(actual) != set(weight_map) or actual_map != weight_map:
        raise ValueError('Output tensor index differs')
    for target in targets:
        item = actual[target.name]
        if tuple(item['shape']) != target.shape or item['dtype'] != DTYPES[target.dtype]:
            raise ValueError(f'Output shape/dtype differs: {target.name}')
        lo, hi = item['data_offsets']
        if hi - lo != target.size_bytes:
            raise ValueError(f'Output tensor size differs: {target.name}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(verify_digest, output / r['file'], r['sha256']) for r in records]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    if reader.metadata_sha256 != hashlib.sha256((reader.directory / '.metadata').read_bytes()).hexdigest():
        raise ValueError('Checkpoint metadata changed during export')
    samples = verify(reader.directory, output, config['text_config'], reader.source_prefix)
    report = {
        **plan, 'passed': True,
        'elapsed_seconds': time.monotonic() - start,
        'tensor_count': len(actual),
        'dtype_counts': dict(collections.Counter(v['dtype'] for v in actual.values())),
        'language_finite_elements': sum(r['finite_elements'] for r in records),
        'language_full_readback_sha256': True,
        'visual_exact_comparisons': len(visual),
        'independent_dcp_samples': samples,
        'shards': sorted(records, key=lambda r: r['file']),
        'asset_sha256': asset_hashes,
        'limits': 'Conversion and byte-readback validation; inference and model quality are not tested.'
    }
    _dump(output / 'export-validation.json', report)
    with (output / '_SUCCESS').open('x') as stream:
        stream.write('Export and validation complete\n')
        stream.flush()
        os.fsync(stream.fileno())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-shard-size', type=int, default=5_000_000_000, help='Shard size in bytes')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(**vars(args)), indent=2))
