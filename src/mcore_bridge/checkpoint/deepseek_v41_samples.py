# Copyright (c) ModelScope Contributors. All rights reserved.
# Original implementation: https://github.com/DongHande/EchoCraft (Apache-2.0).
# Adapted from DongHande/EchoCraft, Apache-2.0, revision 59720b8449b8debef77133dd4496e9a9381a7f67.
# Source: llm_infra/training/swift_distillation/megatron_to_hf/ (DeepSeek V4.1 exporter).
"""Independent DCP/HF samples for endpoints, CSA2, mHC, MoE and Engram boundaries."""

from __future__ import annotations

import io
import json
import math
import pickle
import torch
from pathlib import Path
from safetensors import safe_open


def verify_samples(directory: Path, output: Path, text: dict, source_prefix: str = 'language_model.') -> dict:
    """Use rectangular DCP selections independently of the exporter's flat mapping."""
    with (directory / '.metadata').open('rb') as stream:
        metadata = pickle.load(stream)
    metadata.state_dict_metadata = {
        'language_model.' + key.removeprefix(source_prefix): value
        for key, value in metadata.state_dict_metadata.items() if key.startswith(source_prefix)
    }
    locations = {(i.fqn, tuple(i.offset or ())): v for i, v in metadata.storage_data.items()}
    locations = {
        ('language_model.' + key.removeprefix(source_prefix), offset): value
        for (key, offset), value in locations.items() if key.startswith(source_prefix)
    }
    weight_map = json.loads((output / 'model.safetensors.index.json').read_text())['weight_map']
    results = []

    def source_slice(key: str, selection: tuple[tuple[int, int], ...]) -> torch.Tensor:
        spec = metadata.state_dict_metadata[key]
        result = torch.empty(tuple(hi - lo for lo, hi in selection), dtype=spec.properties.dtype)
        covered = 0
        for chunk in spec.chunks:
            bounds = [(max(lo, offset), min(hi, offset + size))
                      for (lo, hi), offset, size in zip(selection, chunk.offsets, chunk.sizes)]
            if any(hi <= lo for lo, hi in bounds):
                continue
            location = locations[(key, tuple(chunk.offsets))]
            with (directory / location.relative_path).open('rb') as stream:
                stream.seek(location.offset)
                payload = stream.read(location.length)
            value = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True).reshape(tuple(chunk.sizes))
            del payload
            src = tuple(slice(lo - offset, hi - offset) for (lo, hi), offset in zip(bounds, chunk.offsets))
            dst = tuple(slice(lo - begin, hi - begin) for (lo, hi), (begin, _) in zip(bounds, selection))
            result[dst] = value[src]
            covered += math.prod(hi - lo for lo, hi in bounds)
            del value
        if covered != result.numel():
            raise ValueError(f"DCP sample coverage differs: {key}")
        return result

    def compare(key: str,
                source: str,
                selection: tuple[tuple[int, int], ...],
                hf_selection: tuple[tuple[int, int], ...] | None = None) -> None:
        hf_selection = selection if hf_selection is None else hf_selection
        expected = source_slice(source, selection)
        with safe_open(str(output / weight_map[key]), framework='pt', device='cpu') as handle:
            actual = handle.get_slice(key)[tuple(slice(lo, hi) for lo, hi in hf_selection)]
        expected = expected.reshape(actual.shape)
        if actual.dtype != expected.dtype or not torch.equal(actual, expected):
            raise ValueError(f"Independent DCP/HF sample differs: {key}, {selection}")
        results.append({
            'key': key,
            'source': source,
            'source_slice': selection,
            'hf_slice': hf_selection,
            'elements': actual.numel(),
            'exact_equal': True
        })

    hidden, vocab = text['hidden_size'], text['vocab_size']
    for source, target in (('embedding.word_embeddings.weight', 'embed.weight'), ('output_layer.weight',
                                                                                  'head.weight')):
        for row in (0, vocab // 2, vocab - 1):
            compare(target, 'language_model.' + source, ((row, row + 1), (0, hidden)))
    compare('norm.weight', 'language_model.decoder.final_norm.weight', ((0, hidden), ))
    for layer in sorted({0, text['num_hidden_layers'] - 1, *text.get('kv_source_layer_ids', [])}):
        source = f"language_model.decoder.layers.{2 * layer}."
        hf = f"layers.{layer}."
        for src, dst in (('linear_q_down_proj.weight', 'wq_a.weight'), ('linear_o_group_proj', 'wo_a.weight'),
                         ('core_attention.indexer.linear_wk.weight', 'indexer.wk.weight')):
            key = source + 'inner_layer.self_attention.' + src
            if key in metadata.state_dict_metadata:
                dims = metadata.state_dict_metadata[key].size
                compare(hf + 'attn.' + dst, key, ((0, min(16, dims[0])), (0, dims[1])))
        for parity, channel in ((0, 'attn'), (1, 'ffn')):
            key = f"language_model.decoder.layers.{2 * layer + parity}.hyper_connection.mapping_proj.weight"
            dims = metadata.state_dict_metadata[key].size
            compare(hf + f"hc_{channel}_fn", key, ((0, dims[0]), (0, dims[1])))
            scale = torch.cat([
                source_slice(f"language_model.decoder.layers.{2 * layer + parity}."
                             f"hyper_connection.alpha_{suffix}", ((0, 1), )) for suffix in ('pre', 'post', 'res')
            ])
            with safe_open(str(output / weight_map[hf + f"hc_{channel}_scale"]), framework='pt') as handle:
                if not torch.equal(handle.get_tensor(hf + f"hc_{channel}_scale"), scale):
                    raise ValueError('mHC alpha ordering differs')
            results.append({'key': hf + f"hc_{channel}_scale", 'elements': 3, 'exact_equal': True})
    intermediate, experts = text['moe_intermediate_size'], text['n_routed_experts']
    for layer in (0, text['num_hidden_layers'] - 1):
        mg = f"language_model.decoder.layers.{2 * layer + 1}.inner_layer.mlp."
        hf = f"layers.{layer}.ffn."
        for expert in (0, experts // 2, experts - 1):
            for offset, name in ((0, 'w1'), (intermediate, 'w3')):
                compare(hf + f"experts.{expert}.{name}.weight", mg + 'experts.experts.linear_fc1.weight',
                        ((expert, expert + 1), (offset, offset + intermediate), (0, hidden)),
                        ((0, intermediate), (0, hidden)))
            compare(hf + f"experts.{expert}.w2.weight", mg + 'experts.experts.linear_fc2.weight',
                    ((expert, expert + 1), (0, hidden), (0, intermediate)), ((0, hidden), (0, intermediate)))
        for offset, name in ((0, 'w1'), (intermediate, 'w3')):
            compare(hf + f"shared_experts.{name}.weight", mg + 'shared_experts.linear_fc1.weight',
                    ((offset, offset + intermediate), (0, hidden)), ((0, intermediate), (0, hidden)))
    for layer in text['engram_layer_ids']:
        prefix = f"language_model.decoder.layers.{2 * layer}.inner_layer.engram."
        tables = sorted((k for k in metadata.state_dict_metadata if k.startswith(prefix + 'embedding.tables.')),
                        key=lambda k: int(k.split('.')[-2]))
        rows_before = 0
        for table, key in enumerate(tables):
            spec = metadata.state_dict_metadata[key]
            if table in (0, len(tables) - 1):
                chunks = sorted(spec.chunks, key=lambda c: c.offsets[0])
                rows = sorted(r for r in {0, spec.size[0] - 1, chunks[-1].offsets[0] - 1, chunks[-1].offsets[0]}
                              if 0 <= r < spec.size[0])
                for row in rows:
                    compare(f"layers.{layer}.engram.embed.weight", key, ((row, row + 1), (0, spec.size[1])),
                            ((rows_before + row, rows_before + row + 1), (0, spec.size[1])))
            rows_before += spec.size[0]
        width = metadata.state_dict_metadata[prefix + 'key_projection.weight'].size[1]
        for name, offset, rows in (('key_projection', 0, hidden * text['hc_mult']), ('value_projection',
                                                                                     hidden * text['hc_mult'], hidden)):
            compare(f"layers.{layer}.engram.wkv.weight", prefix + name + '.weight', ((0, rows), (0, width)),
                    ((offset, offset + rows), (0, width)))
    print(json.dumps({'independent_samples': len(results), 'exact_equal': True}), flush=True)
    return {
        'passed': True,
        'comparisons': len(results),
        'elements_compared': sum(r['elements'] for r in results),
        'results': results
    }
