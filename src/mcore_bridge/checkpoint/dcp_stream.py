# Copyright (c) ModelScope Contributors. All rights reserved.
# Original implementation: https://github.com/DongHande/EchoCraft (Apache-2.0).
# Adapted from DongHande/EchoCraft, Apache-2.0, revision 59720b8449b8debef77133dd4496e9a9381a7f67.
# Source: llm_infra/training/swift_distillation/megatron_to_hf/ (DeepSeek V4.1 exporter).
"""Bounded-memory DCP reads and safetensors writes, without model initialization."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pickle
import re
import struct
import torch
from collections.abc import Iterator
from pathlib import Path
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata

from .deepseek_v41_mapping import Target, check_partition, flat_chunk

DTYPES = {'torch.bfloat16': 'BF16', 'torch.float32': 'F32'}
BLOCK_BYTES = 64 * 1024 * 1024


def headers(directory: Path) -> tuple[dict, dict[str, str]]:
    index = json.loads((directory / 'model.safetensors.index.json').read_text())
    weight_map = index['weight_map']
    result = {}
    for filename in sorted(set(weight_map.values())):
        if Path(filename).name != filename:
            raise ValueError('Unsafe shard filename')
        path = directory / filename
        with path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            if length > 100_000_000:
                raise ValueError(f"Header too large: {path}")
            header = json.loads(stream.read(length))
        cursor = 0
        for key, value in sorted(((k, v) for k, v in header.items() if k != '__metadata__'),
                                 key=lambda kv: kv[1]['data_offsets'][0]):
            start, stop = value['data_offsets']
            if key in result or weight_map.get(key) != filename or start != cursor or stop <= start:
                raise ValueError(f"Invalid header/index: {key}")
            cursor = stop
            result[key] = value
        if path.stat().st_size != 8 + length + cursor:
            raise ValueError(f"Truncated shard: {path}")
    if result.keys() != weight_map.keys():
        raise ValueError('Incomplete safetensors index')
    return result, weight_map


class DcpReader:
    """Read trusted torch DCP records individually, rejecting gaps and overlaps."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        with (directory / '.metadata').open('rb') as stream:
            metadata: Metadata = pickle.load(stream)
        self.metadata_sha256 = hashlib.sha256((directory / '.metadata').read_bytes()).hexdigest()
        if not isinstance(metadata, Metadata) or not isinstance(metadata.storage_data, dict):
            raise ValueError('Unsupported DCP metadata: expected PyTorch FileSystemWriter metadata')
        locations = {(i.fqn, tuple(i.offset or ())): v for i, v in metadata.storage_data.items()}
        keys = [key for key, value in metadata.state_dict_metadata.items() if isinstance(value, TensorStorageMetadata)]
        prefixes = ('language_model.', 'model.language_model.', 'model0.language_model.', '')
        candidates = [prefix for prefix in prefixes if prefix + 'embedding.word_embeddings.weight' in keys]
        if len(candidates) != 1:
            raise ValueError('Expected exactly one supported language-model prefix in DCP metadata')
        self.source_prefix = candidates[0]
        self.model = {}
        self.chunks = {}
        file_sizes: dict[str, int] = {}
        for key, spec in metadata.state_dict_metadata.items():
            if not isinstance(spec, TensorStorageMetadata):
                continue
            if self.source_prefix:
                if not key.startswith(self.source_prefix):
                    if re.match(r'^(?:model\d*\.)?(?:language_model\.|embedding\.|decoder\.|output_layer\.)', key):
                        raise ValueError(f'Multiple model wrappers are unsupported: {key}')
                    continue
            elif not key.startswith(('embedding.', 'decoder.', 'output_layer.')):
                continue
            canonical = 'language_model.' + key.removeprefix(self.source_prefix)
            shape = tuple(spec.size)
            dtype = str(spec.properties.dtype)
            if dtype not in DTYPES:
                raise ValueError(f"Unsupported training dtype: {key}, {dtype}")
            chunks = []
            for chunk in spec.chunks:
                start, length = flat_chunk(shape, tuple(chunk.offsets), tuple(chunk.sizes))
                location = locations[(key, tuple(chunk.offsets))]
                if Path(location.relative_path).name != location.relative_path:
                    raise ValueError('Unsafe DCP record path')
                if location.offset < 0 or location.length <= 0:
                    raise ValueError('Invalid DCP record offset')
                if location.relative_path not in file_sizes:
                    file_sizes[location.relative_path] = (directory / location.relative_path).stat().st_size
                if file_sizes[location.relative_path] < location.offset + location.length:
                    raise ValueError(f"Truncated DCP record: {key}")
                chunks.append((start, length, location))
            chunks.sort(key=lambda c: c[0])
            check_partition([(s, s + n) for s, n, _ in chunks], math.prod(shape), key)
            self.model[canonical] = {'shape': shape, 'dtype': dtype}
            self.chunks[canonical] = chunks

    def parts(self, key: str, start: int, length: int) -> Iterator[tuple[int, torch.Tensor]]:
        if start < 0 or length <= 0 or start + length > math.prod(self.model[key]['shape']):
            raise ValueError(f'DCP selection outside tensor: {key}')
        for chunk_start, chunk_length, location in self.chunks[key]:
            lo, hi = max(start, chunk_start), min(start + length, chunk_start + chunk_length)
            if lo >= hi:
                continue
            with (self.directory / location.relative_path).open('rb') as stream:
                stream.seek(location.offset)
                payload = stream.read(location.length)
            if len(payload) != location.length:
                raise ValueError(f"Short DCP read: {key}")
            value = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
            del payload
            if not isinstance(value, torch.Tensor) or value.numel() != chunk_length:
                raise ValueError(f"Invalid DCP tensor: {key}")
            if str(value.dtype) != self.model[key]['dtype'] or not value.is_contiguous():
                raise ValueError(f"Invalid DCP dtype/layout: {key}")
            yield lo - start, value.reshape(-1)[lo - chunk_start:hi - chunk_start]
            del value


def make_header(targets: list[Target]) -> tuple[bytes, dict[str, int]]:
    header: dict[str, object] = {'__metadata__': {'format': 'pt'}}
    offsets = {}
    cursor = 0
    for target in targets:
        offsets[target.name] = cursor
        header[target.name] = {
            'dtype': DTYPES[target.dtype],
            'shape': list(target.shape),
            'data_offsets': [cursor, cursor + target.size_bytes]
        }
        cursor += target.size_bytes
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * ((-len(encoded)) % 8)
    return struct.pack('<Q', len(encoded)) + encoded, offsets


def write_shard(path: Path, targets: list[Target], reader: DcpReader) -> dict[str, object]:
    """Write contiguous pieces and hash all bytes; the oversized Engram stays one tensor."""
    header, offsets = make_header(targets)
    digest = hashlib.sha256(header)
    elements = 0
    with path.open('xb', buffering=0) as stream:
        stream.write(header)
        for target in targets:
            covered = 0
            for piece in target.pieces:
                for offset, value in reader.parts(piece.source, piece.start, piece.length):
                    byte_offset = (piece.destination + offset) * value.element_size()
                    expected = len(header) + offsets[target.name] + byte_offset
                    if stream.tell() != expected:
                        raise ValueError(f"Nonsequential write: {target.name}")
                    step = BLOCK_BYTES // value.element_size()
                    for begin in range(0, value.numel(), step):
                        block = value[begin:begin + step]
                        if not torch.isfinite(block).all().item():
                            raise ValueError(f"NaN/Inf in checkpoint: {piece.source}")
                        raw = memoryview(block.view(torch.uint8).numpy())
                        digest.update(raw)
                        while raw:
                            written = stream.write(raw)
                            if not written:
                                raise OSError('Short safetensors write')
                            raw = raw[written:]
                    covered += value.numel()
                    elements += value.numel()
                    del raw, block, value
            if covered != math.prod(target.shape):
                raise ValueError(f"Incomplete write: {target.name}")
        os.fsync(stream.fileno())
    return {
        'file': path.name,
        'sha256': digest.hexdigest(),
        'bytes': path.stat().st_size,
        'tensors': len(targets),
        'finite_elements': elements
    }


def verify_digest(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(BLOCK_BYTES):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Readback SHA256 mismatch: {path}")
