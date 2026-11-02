# Copyright (c) ModelScope Contributors. All rights reserved.
# Original implementation: https://github.com/DongHande/EchoCraft (Apache-2.0).
# Adapted from DongHande/EchoCraft, Apache-2.0, revision 59720b8449b8debef77133dd4496e9a9381a7f67.
# Source: llm_infra/training/swift_distillation/megatron_to_hf/ (DeepSeek V4.1 exporter).
"""DeepSeek V4.1 hybrid DCP tensors to the official unquantized HF layout."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Piece:
    source: str
    start: int
    length: int
    destination: int = 0


@dataclass(frozen=True)
class Target:
    name: str
    shape: tuple[int, ...]
    dtype: str
    pieces: tuple[Piece, ...]

    @property
    def size_bytes(self) -> int:
        return math.prod(self.shape) * {'torch.bfloat16': 2, 'torch.float32': 4}[self.dtype]


ATTENTION = {
    'input_layernorm.weight': 'attn_norm.weight',
    'self_attention.linear_q_down_proj.weight': 'attn.wq_a.weight',
    'self_attention.linear_q_up_proj.weight': 'attn.wq_b.weight',
    'self_attention.linear_kv_proj.weight': 'attn.wkv.weight',
    'self_attention.linear_o_group_proj': 'attn.wo_a.weight',
    'self_attention.linear_proj.weight': 'attn.wo_b.weight',
    'self_attention.q_layernorm.weight': 'attn.q_norm.weight',
    'self_attention.kv_layernorm.weight': 'attn.kv_norm.weight',
    'self_attention.core_attention.attn_sink': 'attn.attn_sink',
    'self_attention.core_attention.compressor.linear_wkv.weight': 'attn.compressor.wkv.weight',
    'self_attention.core_attention.compressor.linear_wgate.weight': 'attn.compressor.wgate.weight',
    'self_attention.core_attention.compressor.norm.weight': 'attn.compressor.norm.weight',
    'self_attention.core_attention.indexer.linear_wq_b.weight': 'attn.indexer.wq_b.weight',
    'self_attention.core_attention.indexer.linear_weights_proj.weight': 'attn.indexer.weights_proj.weight',
    'self_attention.core_attention.indexer.linear_wk.weight': 'attn.indexer.wk.weight',
    'self_attention.core_attention.indexer.k_norm.weight': 'attn.indexer.k_norm.weight',
}
MLP = {
    'pre_mlp_layernorm.weight': 'ffn_norm.weight',
    'mlp.router.weight': 'ffn.gate.weight',
    'mlp.router.expert_bias': 'ffn.gate.bias',
    'mlp.router.expert_bias_vl': 'ffn.gate.bias_vl',
    'mlp.shared_experts.linear_fc2.weight': 'ffn.shared_experts.w2.weight',
}
ENDPOINTS = {
    'language_model.embedding.word_embeddings.weight': 'embed.weight',
    'language_model.output_layer.weight': 'head.weight',
    'language_model.decoder.final_norm.weight': 'norm.weight',
}


def check_partition(intervals: list[tuple[int, int]], length: int, name: str) -> None:
    cursor = 0
    for start, stop in sorted(intervals):
        if start != cursor or stop <= start:
            raise ValueError(f"Overlapping or missing tensor region: {name}, {cursor}, {(start, stop)}")
        cursor = stop
    if cursor != length:
        raise ValueError(f"Incomplete tensor: {name}, {cursor} != {length}")


def flat_chunk(shape: tuple[int, ...], offsets: tuple[int, ...], sizes: tuple[int, ...]) -> tuple[int, int]:
    """Accept only contiguous rectangular chunks; reject unsupported TP layouts."""
    if len(shape) != len(offsets) or len(shape) != len(sizes):
        raise ValueError('Chunk dimensionality mismatch')
    if any(o < 0 or n <= 0 or o + n > d for d, o, n in zip(shape, offsets, sizes)):
        raise ValueError('Chunk outside tensor bounds')
    strides = [math.prod(shape[i + 1:]) for i in range(len(shape))]
    start = sum(o * s for o, s in zip(offsets, strides))
    last = sum((o + n - 1) * s for o, n, s in zip(offsets, sizes, strides))
    length = math.prod(sizes)
    if last - start + 1 != length:
        raise ValueError('Noncontiguous DCP chunk; this export supports TP1 checkpoints')
    return start, length


def build_targets(model: dict, text_config: dict) -> list[Target]:
    """Map every source element exactly once, including all EP Engram table rows."""
    targets: list[Target] = []
    consumed: set[str] = set()

    def add(name: str, sources: list[str], shape: tuple[int, ...] | None = None) -> None:
        dtype = model[sources[0]]['dtype']
        if dtype not in ('torch.bfloat16', 'torch.float32'):
            raise ValueError(f'Unsupported target dtype: {name}, {dtype}')
        pieces = []
        destination = 0
        for source in sources:
            if model[source]['dtype'] != dtype:
                raise ValueError(f"Mixed source dtypes: {name}")
            length = math.prod(model[source]['shape'])
            pieces.append(Piece(source, 0, length, destination))
            destination += length
        actual_shape = tuple(model[sources[0]]['shape']) if shape is None else shape
        targets.append(Target(name, actual_shape, dtype, tuple(pieces)))
        consumed.update(sources)

    for source, name in ENDPOINTS.items():
        add(name, [source])
    layers = text_config['num_hidden_layers']
    experts = text_config['n_routed_experts']
    for layer in range(layers):
        hf = f"layers.{layer}."
        for parity, channel in ((0, 'attn'), (1, 'ffn')):
            prefix = f"language_model.decoder.layers.{2 * layer + parity}."
            for suffix, name in (('mapping_proj.weight', 'fn'), ('bias', 'base')):
                add(hf + f"hc_{channel}_{name}", [prefix + 'hyper_connection.' + suffix])
            add(hf + f"hc_{channel}_scale",
                [prefix + 'hyper_connection.alpha_' + part for part in ('pre', 'post', 'res')], (3, ))
            inner = prefix + 'inner_layer.'
            for source_suffix, name in (ATTENTION if parity == 0 else MLP).items():
                source = inner + source_suffix
                if source in model:
                    add(hf + name, [source])
            if parity == 0 and layer in text_config['engram_layer_ids']:
                ep = inner + 'engram.'
                table_keys = sorted(
                    (k for k in model if re.fullmatch(re.escape(ep) + r'embedding.tables.\d+.weight', k)),
                    key=lambda k: int(k.split('.')[-2]),
                )
                count = (text_config['engram_max_ngram_size'] - 1) * text_config['engram_n_heads']
                if [int(k.split('.')[-2]) for k in table_keys] != list(range(count)):
                    raise ValueError(f"Engram subtable IDs missing: {layer}")
                rows = sum(model[k]['shape'][0] for k in table_keys)
                expected_rows = text_config['engram_num_embeddings'][text_config['engram_layer_ids'].index(layer)]
                if rows != expected_rows:
                    raise ValueError(f"Engram row count mismatch: {layer}")
                add(hf + 'engram.embed.weight', table_keys, (rows, text_config['engram_head_dim']))
                hidden, mult = text_config['hidden_size'], text_config['hc_mult']
                width = count * text_config['engram_head_dim']
                add(hf + 'engram.wkv.weight', [ep + 'key_projection.weight', ep + 'value_projection.weight'],
                    ((mult + 1) * hidden, width))
                for source, target in (('query_norm', 'q_weight'), ('key_norm', 'k_weight')):
                    add(hf + 'engram.' + target, [ep + source + '.weight'], (mult, hidden))
            if parity == 1:
                for family in ('shared_experts', 'experts.experts'):
                    for linear in ('linear_fc1', 'linear_fc2'):
                        source = inner + f"mlp.{family}.{linear}.weight"
                        if family == 'shared_experts' and linear == 'linear_fc2':
                            continue
                        dims = tuple(model[source]['shape'])
                        is_expert = family == 'experts.experts'
                        if is_expert and dims[0] != experts:
                            raise ValueError(f"Expert count mismatch: {source}")
                        shape = dims[1:] if is_expert else dims
                        if len(shape) != 2 or (linear == 'linear_fc1' and shape[0] % 2):
                            raise ValueError(f'Unsupported expert tensor shape: {source}, {dims}')
                        parts = ('w1', 'w3') if linear == 'linear_fc1' else ('w2', )
                        out_shape = (shape[0] // len(parts), shape[1])
                        length = math.prod(out_shape)
                        for expert in range(experts if is_expert else 1):
                            name = hf + (f"ffn.experts.{expert}." if is_expert else 'ffn.shared_experts.')
                            for part, target in enumerate(parts):
                                start = (expert * len(parts) + part) * length
                                targets.append(
                                    Target(name + target + '.weight', out_shape, model[source]['dtype'],
                                           (Piece(source, start, length), )))
                        consumed.add(source)
    if consumed != model.keys():
        raise ValueError(f"Unmapped DCP tensors: {sorted(model.keys() - consumed)[:10]}")
    validate_mapping(model, targets)
    return targets


def validate_mapping(model: dict, targets: list[Target]) -> None:
    if len({t.name for t in targets}) != len(targets):
        raise ValueError('Duplicate HF target')
    source_intervals: dict[str, list[tuple[int, int]]] = {key: [] for key in model}
    for target in targets:
        intervals = []
        for piece in target.pieces:
            source_intervals[piece.source].append((piece.start, piece.start + piece.length))
            intervals.append((piece.destination, piece.destination + piece.length))
        check_partition(intervals, math.prod(target.shape), target.name)
    for source, intervals in source_intervals.items():
        check_partition(intervals, math.prod(model[source]['shape']), source)


def check_base_layout(targets: list[Target], base_headers: dict) -> list[str]:
    expected = {k for k in base_headers if not k.startswith('mtp.') and not k.endswith('.scale')}
    visual = {
        k
        for k in expected if k.startswith(('vision.', 'aligner.')) or k in ('image_start', 'image_end', 'image_newline')
    }
    actual = {t.name for t in targets}
    if actual != expected - visual:
        raise ValueError(f"HF keys differ: missing={sorted(expected - visual - actual)[:10]}, "
                         f"extra={sorted(actual - expected)[:10]}")
    for target in targets:
        base = base_headers[target.name]
        shape = list(base['shape'])
        if base['dtype'] == 'I8' and target.name.removesuffix('.weight') + '.scale' in base_headers:
            shape[-1] *= 2  # The official FP4 expert stores two values per int8 byte.
        if tuple(shape) != target.shape:
            raise ValueError(f"HF shape differs: {target.name}, {target.shape} != {shape}")
    if any(base_headers[k]['dtype'] not in ('BF16', 'F32') for k in visual):
        raise ValueError('Quantized visual weights require a separate conversion')
    return sorted(visual)
