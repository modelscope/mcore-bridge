# SPDX-License-Identifier: Apache-2.0
"""Validate Qwen4-Exp PLE assets and complete HF text-training checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import tempfile
import torch
from collections.abc import Collection
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Any

_VISION_PREFIX = 'model.visual.'
_MTP_PREFIXES = ('mtp.', )


def validate_ple_checkpoint(
    path: str,
    config: Any,
    layers_prefix: str,
    models: list[torch.nn.Module] | None = None,
) -> dict[str, dict[str, Any]]:
    """Check PLE assets without materializing the large embedding tables.

    Hashes cover the small lookup buffers, not the table contents. Full checkpoint
    file hashes belong to the experiment's source manifest.
    """
    model_dir = Path(path)
    index_path = model_dir / 'model.safetensors.index.json'
    if index_path.is_file():
        with index_path.open() as stream:
            weight_map = json.load(stream)['weight_map']
    else:
        with safe_open(model_dir / 'model.safetensors', framework='pt', device='cpu') as handle:
            weight_map = dict.fromkeys(handle.keys(), 'model.safetensors')

    heads = (config.ngram_size - 1) * config.heads_per_ngram
    parts = config.split_ngram_parts
    divisor = config.make_ngram_vocab_size_divisible_by
    if heads <= 0 or parts <= 0 or divisor <= 0 or config.ple_embed_dim % heads:
        raise ValueError('Invalid Qwen4-Exp PLE head, shard, or embedding dimensions.')
    head_dim = config.ple_embed_dim // heads
    manifest = {}
    with ExitStack() as stack:
        files = {}

        def tensor_handle(key: str):
            if key not in weight_map:
                raise ValueError(f"Missing required PLE checkpoint tensor: {key}")
            filename = weight_map[key]
            if filename not in files:
                files[filename] = stack.enter_context(safe_open(model_dir / filename, framework='pt', device='cpu'))
            handle = files[filename]
            if key not in handle.keys():
                raise ValueError(f"PLE checkpoint index points to an absent tensor: {key}")
            return handle

        for layer_id in config.ple_layer_ids:
            prefix = f"{layers_prefix}.{layer_id - 1}.ple.ple_embedding."
            buffers = {}
            for name, length in (
                ('layer_multipliers', config.ngram_size),
                ('ngram_heads_offsets', heads),
                ('ngram_heads_vocab_sizes', heads),
            ):
                key = prefix + name
                handle = tensor_handle(key)
                metadata = handle.get_slice(key)
                if metadata.get_shape() != [length] or metadata.get_dtype() != 'I64':
                    raise ValueError(f"Invalid PLE hash-buffer shape or dtype: {key}")
                buffer = handle.get_tensor(key)
                buffers[name] = buffer
                manifest[key] = {
                    'shape': [length],
                    'dtype': 'I64',
                    'sha256': hashlib.sha256(buffer.numpy().tobytes()).hexdigest(),
                }
            sizes = buffers['ngram_heads_vocab_sizes']
            offsets = buffers['ngram_heads_offsets']
            expected_offsets = torch.cat((sizes.new_zeros(1), sizes.cumsum(0)[:-1]))
            if not bool(torch.all(sizes > 0)) or not torch.equal(offsets, expected_offsets):
                raise ValueError(f"Invalid PLE hash-table sizes or offsets: {prefix}")
            total = ((int(sizes.sum()) + divisor - 1) // divisor) * divisor
            shard_size = (total + parts - 1) // parts
            shard_dtypes = set()
            for part in range(parts):
                key = f"{prefix}ngram_embedding.shard_{part}.weight"
                metadata = tensor_handle(key).get_slice(key)
                expected_shape = [
                    max(0, min(shard_size, total - part * shard_size)),
                    head_dim,
                ]
                shape = metadata.get_shape()
                dtype = metadata.get_dtype()
                if shape != expected_shape:
                    raise ValueError(f"Invalid PLE shard shape for {key}: expected {expected_shape}, got {shape}")
                if dtype not in ('BF16', 'F16', 'F32', 'F8_E4M3'):
                    raise ValueError(f"Unsupported PLE shard dtype for {key}: {dtype}")
                shard_dtypes.add(dtype)
                manifest[key] = {'shape': shape, 'dtype': dtype}
            if len(shard_dtypes) != 1:
                raise ValueError(f"Mixed PLE shard dtypes: {prefix}")
            scale_key = f"{prefix}ngram_embedding.weight_scale"
            if 'F8_E4M3' in shard_dtypes or scale_key in weight_map:
                handle = tensor_handle(scale_key)
                metadata = handle.get_slice(scale_key)
                if metadata.get_shape() not in ([], [1]):
                    raise ValueError(f"PLE weight scale must be scalar: {scale_key}")
                scale = handle.get_tensor(scale_key).float()
                if not bool(torch.all(torch.isfinite(scale) & (scale > 0))):
                    raise ValueError(f"PLE weight scale must be finite and positive: {scale_key}")
                manifest[scale_key] = {
                    'shape': metadata.get_shape(),
                    'dtype': metadata.get_dtype(),
                }
    # These buffers are deterministic functions of config/seed. Check the local
    # model before loading can overwrite a mismatched hash function from disk.
    for model in models or []:
        for layer in model.modules():
            ple = getattr(layer, 'ple', None)
            if ple is None or not hasattr(layer, 'layer_number'):
                continue
            prefix = f"{layers_prefix}.{layer.layer_number - 1}.ple.ple_embedding."
            for name in (
                    'layer_multipliers',
                    'ngram_heads_offsets',
                    'ngram_heads_vocab_sizes',
            ):
                buffer = getattr(ple.ple_embedding, name).detach().cpu().contiguous()
                fingerprint = hashlib.sha256(buffer.numpy().tobytes()).hexdigest()
                if fingerprint != manifest[prefix + name]['sha256']:
                    raise ValueError(f"PLE hash buffer disagrees with the model configuration: {prefix}{name}")
    return manifest


def qwen4_exp_export_config(hf_config: Any, *, mtp_enabled: bool) -> Any:
    """Return an export-only config; keep the running model's config untouched."""
    exported = deepcopy(hf_config)
    text_config = getattr(exported, 'text_config', exported)
    layer_types = getattr(text_config, 'layer_types', None)
    if layer_types is not None:
        # Newer Transformers normalizes this checkpoint spelling to QSA at
        # load time. Preserve the original portable spelling when exporting:
        # the pinned SGLang Qwen4-Exp loader selects its QSA implementation
        # through "full_attention", not "qwen_sparse_attention".
        text_config.layer_types = [
            'full_attention' if kind == 'qwen_sparse_attention' else kind for kind in layer_types
        ]
    if not mtp_enabled:
        text_config.mtp = None
        text_config.mtp_num_hidden_layers = 0
        for config in (exported, text_config):
            if hasattr(config, 'mtp'):
                config.mtp = None
            for field in (
                    'mtp_num_hidden_layers',
                    'mtp_num_layers',
                    'num_nextn_predict_layers',
            ):
                if hasattr(config, field):
                    setattr(config, field, 0)
    return exported


def _checkpoint_inventory(directory: Path, ) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, Any]]:
    """Read tensor headers and verify the index against files without loading weights."""
    index_path = directory / 'model.safetensors.index.json'
    index = None
    if index_path.is_file():
        if not index_path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Checkpoint index escapes its directory: {index_path}")
        with index_path.open() as stream:
            index = json.load(stream)
        if not isinstance(index.get('weight_map'), dict):
            raise ValueError(f"Invalid safetensors weight_map: {index_path}")
        for filename in index['weight_map'].values():
            if (not isinstance(filename, str) or Path(filename).is_absolute() or '..' in Path(filename).parts
                    or Path(filename).suffix != '.safetensors'):
                raise ValueError(f"Invalid checkpoint shard path: {filename!r}")
    filenames = {path.name for path in directory.glob('*.safetensors')}
    if index is not None:
        filenames.update(index['weight_map'].values())
    if not filenames:
        raise ValueError(f"No safetensors weight files in {directory}")
    weight_map = {}
    tensors = {}
    for filename in sorted(filenames):
        path = directory / filename
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Checkpoint shard escapes its directory: {path}")
        # safe_open validates the file format before offsets are used for the
        # same byte-accounting convention as MegatronEngine's index rebuild.
        with safe_open(path, framework='pt', device='cpu') as handle:
            with path.open('rb') as stream:
                header_size = struct.unpack('<Q', stream.read(8))[0]
                header = json.loads(stream.read(header_size))
            for key in handle.keys():
                if key in weight_map:
                    raise ValueError(f"Duplicate checkpoint tensor {key} in {directory}")
                begin, end = header[key]['data_offsets']
                weight_map[key] = filename
                tensors[key] = {
                    'shape': header[key]['shape'],
                    'dtype': header[key]['dtype'],
                    'nbytes': end - begin,
                }
    if index is not None and index['weight_map'] != weight_map:
        raise ValueError(f"Safetensors index disagrees with actual tensor files in {directory}")
    return weight_map, tensors, {} if index is None else index.get('metadata', {})


def restore_qwen4_exp_fixed_assets(
    source_path: str,
    output_path: str,
    *,
    language_model_only: bool,
    mtp_enabled: bool = False,
    omitted_mtp_keys: Collection[str] | None = None,
    max_shard_size_bytes: int = 256 * 1024 * 1024,
) -> dict[str, Any]:
    """Validate an HF export and restore only missing ``model.visual.*`` tensors.

    Call on the saving rank after the bridge finished writing all tensor shards.
    Callers own distributed error propagation and config/tokenizer preservation.
    Each bucket is bounded by ``max_shard_size_bytes``; an indivisible larger
    tensor gets its own shard. Existing tensors are never copied over. With MTP
    disabled, only top-level ``mtp.*`` keys are allowed to be omitted; callers
    may narrow that whitelist by providing exact ``omitted_mtp_keys``.
    """
    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    if source == output:
        raise ValueError('Source checkpoint and output checkpoint must differ.')
    if max_shard_size_bytes <= 0:
        raise ValueError('max_shard_size_bytes must be positive.')
    with (source / 'config.json').open() as stream:
        config = json.load(stream)
    if config.get('model_type') != 'qwen4_exp':
        raise ValueError('Fixed-asset restoration requires a qwen4_exp source checkpoint.')
    source_map, source_tensors, _ = _checkpoint_inventory(source)
    output_map, output_tensors, output_metadata = _checkpoint_inventory(output)
    source_keys = set(source_map)
    output_keys = set(output_map)
    if omitted_mtp_keys is None:
        omitted_mtp_keys = (set() if mtp_enabled else {key for key in source_keys if key.startswith(_MTP_PREFIXES)})
    else:
        omitted_mtp_keys = set(omitted_mtp_keys)
    if mtp_enabled and omitted_mtp_keys:
        raise ValueError('Enabled MTP cannot have omitted checkpoint keys.')
    if any(not key.startswith(_MTP_PREFIXES) for key in omitted_mtp_keys):
        raise ValueError('The omission whitelist accepts exact MTP tensor keys only.')
    if omitted_mtp_keys - source_keys:
        raise ValueError(f"Omitted MTP keys are absent from the source: {sorted(omitted_mtp_keys - source_keys)}")
    if omitted_mtp_keys & output_keys:
        raise ValueError('The output contains MTP tensors explicitly declared omitted.')
    unknown = output_keys - source_keys
    if unknown:
        raise ValueError(f"Export contains unknown checkpoint tensors: {sorted(unknown)}")
    # The bridge exports dequantized PLE tables in the parameter dtype.
    # Their original FP8 scale must disappear, but only after every source
    # shard has been exported with the same shape in the new representation.
    ple_groups: dict[str, dict[int, str]] = {}
    for key in source_keys:
        match = re.fullmatch(
            r'(model\.language_model\.layers\.\d+\.ple\.ple_embedding'
            r'\.ngram_embedding)\.shard_(\d+)\.weight',
            key,
        )
        if match:
            ple_groups.setdefault(match[1], {})[int(match[2])] = key
    converted_ple_keys: set[str] = set()
    omitted_ple_scales: set[str] = set()
    dequantized_dtypes = {'BF16', 'F16', 'F32'}
    for prefix, shards in ple_groups.items():
        if not any(source_tensors[key]['dtype'] == 'F8_E4M3'
                   and output_tensors.get(key, {}).get('dtype') in dequantized_dtypes for key in shards.values()):
            continue
        output_dtypes = {output_tensors.get(key, {}).get('dtype') for key in shards.values()}
        if (len(output_dtypes) != 1 or not output_dtypes <= dequantized_dtypes
                or set(shards) != set(range(len(shards)))
                or any(source_tensors[key]['dtype'] != 'F8_E4M3'
                       or output_tensors[key]['shape'] != source_tensors[key]['shape'] for key in shards.values())):
            raise ValueError(f"Incomplete or invalid floating-point PLE conversion: {prefix}")
        scale_key = f"{prefix}.weight_scale"
        if scale_key not in source_keys or scale_key in output_keys:
            raise ValueError(f"Floating-point PLE conversion requires removing the source scale: {scale_key}")
        converted_ple_keys.update(shards.values())
        omitted_ple_scales.add(scale_key)
    vision_keys = {key for key in source_keys if key.startswith(_VISION_PREFIX)}
    missing = source_keys - output_keys - omitted_mtp_keys - omitted_ple_scales
    copy_keys = missing & vision_keys if language_model_only else set()
    missing_required = missing - copy_keys
    if missing_required:
        raise ValueError(f"Export is missing required non-restorable tensors: {sorted(missing_required)}")
    for key in output_keys:
        if output_tensors[key]['shape'] != source_tensors[key]['shape']:
            raise ValueError(f"Export tensor shape differs from source: {key}")
        if (output_tensors[key]['dtype'] != source_tensors[key]['dtype'] and key not in converted_ple_keys):
            raise ValueError(f"Export tensor dtype differs from source: {key}: "
                             f"{source_tensors[key]['dtype']} -> {output_tensors[key]['dtype']}")
    if language_model_only and not vision_keys:
        raise ValueError('The source contains no recognized model.visual.* fixed assets.')

    bucket: dict[str, torch.Tensor] = {}
    bucket_size = 0
    next_shard = 1
    new_shards = []
    pending_shards = []

    def flush_bucket(cleanup: ExitStack) -> None:
        nonlocal bucket_size, next_shard
        if not bucket:
            return
        filename = f"model-fixed-visual-{next_shard:05d}.safetensors"
        while (output / filename).exists():
            next_shard += 1
            filename = f"model-fixed-visual-{next_shard:05d}.safetensors"
        with tempfile.NamedTemporaryFile(
                dir=output, prefix='.fixed-visual-', suffix='.pending', delete=False) as stream:
            temporary_path = Path(stream.name)
        cleanup.callback(temporary_path.unlink, missing_ok=True)
        save_file(bucket, temporary_path, metadata={'format': 'pt'})
        pending_shards.append((temporary_path, output / filename))
        for key in bucket:
            output_map[key] = filename
            output_tensors[key] = source_tensors[key]
        new_shards.append(filename)
        bucket.clear()
        bucket_size = 0
        next_shard += 1

    with ExitStack() as cleanup:
        # Group reads by source shard. get_tensor touches only selected vision
        # tensors, even when that file also stores trainable text weights.
        for filename in sorted({source_map[key] for key in copy_keys}):
            with safe_open(source / filename, framework='pt', device='cpu') as handle:
                for key in sorted(key for key in copy_keys if source_map[key] == filename):
                    nbytes = source_tensors[key]['nbytes']
                    if bucket and bucket_size + nbytes > max_shard_size_bytes:
                        flush_bucket(cleanup)
                    bucket[key] = handle.get_tensor(key)
                    bucket_size += nbytes
                    if bucket_size >= max_shard_size_bytes:
                        flush_bucket(cleanup)
        flush_bucket(cleanup)
        for temporary_path, shard_path in pending_shards:
            os.replace(temporary_path, shard_path)
            cleanup.callback(shard_path.unlink, missing_ok=True)

        # HF prefers model.safetensors over an index, so it must become a shard
        # when extra files are added. The trained tensor bytes remain unchanged.
        if ('model.safetensors' in output_map.values() and len(set(output_map.values())) > 1):
            shard_number = 1
            filename = f"model-exported-{shard_number:05d}.safetensors"
            while (output / filename).exists():
                shard_number += 1
                filename = f"model-exported-{shard_number:05d}.safetensors"
            os.replace(output / 'model.safetensors', output / filename)
            cleanup.callback(os.replace, output / filename, output / 'model.safetensors')
            output_map = {key: filename if value == 'model.safetensors' else value for key, value in output_map.items()}

        total_size = sum(tensor['nbytes'] for tensor in output_tensors.values())
        output_metadata = dict(output_metadata, total_size=total_size)
        index = {
            'metadata': output_metadata,
            'weight_map': dict(sorted(output_map.items())),
        }
        with tempfile.NamedTemporaryFile(
                mode='w',
                dir=output,
                prefix='.model-index-',
                suffix='.pending',
                delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            cleanup.callback(temporary_path.unlink, missing_ok=True)
            json.dump(index, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output / 'model.safetensors.index.json')
        cleanup.pop_all()
    return {
        'restored_keys': sorted(copy_keys),
        'omitted_mtp_keys': sorted(omitted_mtp_keys),
        'new_shards': new_shards,
        'total_size': total_size,
    }
