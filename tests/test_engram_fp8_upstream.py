# Copyright (c) ModelScope Contributors. All rights reserved.
"""Fp8EngramEmbedding against the real Megatron Engram embedding; skipped without the upstream package
(stub-based coverage lives in ``test_engram_fp8_tables.py``).
"""
import pytest
import torch
from types import SimpleNamespace

upstream_embedding = pytest.importorskip('megatron.core.models.engram.distributed_embedding')

from mcore_bridge.model.mm_gpts.deepseek_v41 import DeepseekV41Bridge  # noqa: E402
from mcore_bridge.model.modules.engram import DeepseekV41Engram  # noqa: E402
from mcore_bridge.model.modules.engram_fp8 import Fp8EngramEmbedding, Fp8EngramTable, is_fp8_table  # noqa: E402

BLOCK = 32
WIDTH = 256
SIZES = (96, 128)


def _config(*, fp8_declared=False):
    return SimpleNamespace(
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16,
        sequence_parallel=False,
        deterministic_mode=False,
        perform_initialization=False,
        hf_config=SimpleNamespace(quantization_config={'scale_fmt': 'ue8m0'} if fp8_declared else None))


def _bits(total_rows, seed=0):
    generator = torch.Generator().manual_seed(seed)
    payload = (torch.randn(total_rows, WIDTH, generator=generator) * 4).to(torch.float8_e4m3fn)
    scales = torch.randint(120, 130, (total_rows, WIDTH // BLOCK), generator=generator, dtype=torch.uint8)
    return payload, scales


def _build_pair():
    upstream = upstream_embedding.EPShardedMultiTableEmbedding(
        config=_config(),
        table_sizes=SIZES,
        embedding_dim=WIDTH,
        init_method=None,
        ep_group=None,
        tp_group=None,
        expt_dp_group=None)
    ours = Fp8EngramEmbedding(
        config=_config(),
        table_sizes=SIZES,
        embedding_dim=WIDTH,
        ep_group=None,
        tp_group=None,
        expt_dp_group=None,
        block_size=BLOCK)
    payload, scales = _bits(sum(SIZES))
    offset = 0
    for table, fp8_table in zip(upstream.tables, ours.tables):
        rows = table.local_num_embeddings
        assert fp8_table.local_num_embeddings == rows
        fp8_table.fill_rows(0, rows, payload[offset:offset + rows], scales[offset:offset + rows])
        table.weight.data.copy_(fp8_table(torch.arange(rows)).to(table.weight.dtype))
        offset += rows
    return upstream, ours


def test_forward_matches_upstream_bf16_bit_for_bit():
    upstream, ours = _build_pair()
    torch.manual_seed(0)
    ids = torch.randint(0, min(SIZES), (3, 4, len(SIZES)))
    reference = upstream(ids)
    got = ours(ids)
    assert reference.shape == got.shape == (3, 4, len(SIZES), WIDTH)
    assert torch.equal(reference, got)


def test_fp8_store_holds_no_bf16_shard_and_halves_storage():
    upstream, ours = _build_pair()
    bf16_bytes = sum(t.weight.numel() * t.weight.element_size() for t in upstream.tables)
    fp8_bytes = sum(t.local_storage_bytes for t in ours.tables)
    assert sum(t.weight.numel() for t in ours.tables) == 0
    assert fp8_bytes < bf16_bytes
    assert fp8_bytes == sum(t.fp8_rows.numel() + t.fp8_scales.numel() for t in ours.tables)


def test_row_ranges_follow_the_upstream_partition():
    upstream, ours = _build_pair()
    assert [(t.row_start, t.row_end) for t in ours.tables] == [(t.row_start, t.row_end) for t in upstream.tables]


def test_fp8_tables_subclass_the_real_upstream_table():
    """Guards against a stub module leaking into this file: the base must be Megatron's own class."""
    _, ours = _build_pair()
    for table in ours.tables:
        assert isinstance(table, Fp8EngramTable)
        assert type(table).__mro__[1] is upstream_embedding.EPShardedEmbeddingTable


def test_swap_replace_the_bf16_embedding_with_untrainable_tables():
    upstream = upstream_embedding.EPShardedMultiTableEmbedding(
        config=_config(fp8_declared=True),
        table_sizes=SIZES,
        embedding_dim=WIDTH,
        init_method=None,
        ep_group=None,
        tp_group=None,
        expt_dp_group=None)
    holder = SimpleNamespace(config=_config(fp8_declared=True), embedding=upstream)

    DeepseekV41Engram._maybe_use_fp8_tables(holder)
    assert isinstance(holder.embedding, Fp8EngramEmbedding)
    assert all(is_fp8_table(table) for table in holder.embedding.tables)
    assert [tuple(t.fp8_rows.shape) for t in holder.embedding.tables] == [(SIZES[0], WIDTH), (SIZES[1], WIDTH)]
    # No bf16 shard is left behind and the placeholder parameter carries no gradient state.
    assert all(t.weight.numel() == 0 and not t.weight.requires_grad for t in holder.embedding.tables)


def _export_bridge():
    bridge = object.__new__(DeepseekV41Bridge)
    bridge.config = SimpleNamespace(engram_num_embeddings=[sum(SIZES)], engram_layer_ids=[1])
    bridge._peft_format = False
    bridge._skip_unsupported_export = False
    bridge._ENGRAM_LOAD_CHUNK_ROWS = 8
    return bridge


def _fake_layer(embedding):
    return SimpleNamespace(
        engram=SimpleNamespace(
            embedding=embedding,
            layer_number=3,
            num_streams=2,
            hidden_size=4,
            engram_config=SimpleNamespace(total_memory_dim=WIDTH),
            key_projection=SimpleNamespace(weight=torch.nn.Parameter(torch.zeros(8, WIDTH, dtype=torch.bfloat16))),
            value_projection=SimpleNamespace(weight=torch.nn.Parameter(torch.zeros(4, WIDTH, dtype=torch.bfloat16))),
            query_norm=SimpleNamespace(weight=torch.nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))),
            key_norm=SimpleNamespace(weight=torch.nn.Parameter(torch.zeros(8, dtype=torch.bfloat16)))))


def test_export_writes_fp8_planes_and_round_trips_into_both_stores():
    _, fp8_embedding = _build_pair()
    bridge = _export_bridge()
    hf = {}
    bridge._set_layer_engram(_fake_layer(fp8_embedding), hf, to_mcore=False)
    assert hf['engram.embed.weight'].dtype == torch.float8_e4m3fn
    assert hf['engram.embed.weight_scale_inv'].dtype == torch.float8_e8m0fnu
    assert tuple(hf['engram.embed.weight'].shape) == (sum(SIZES), WIDTH)
    assert torch.equal(hf['engram.embed.weight'].view(torch.uint8),
                       torch.cat([t.fp8_rows for t in fp8_embedding.tables], dim=0))

    state = {
        'engram.embed.weight': _Tensor(hf['engram.embed.weight']),
        'engram.embed.weight_scale_inv': _Tensor(hf['engram.embed.weight_scale_inv'])
    }
    ids = torch.stack([torch.randint(0, SIZES[0], (8, )), torch.randint(0, SIZES[1], (8, ))], dim=-1)

    fp8_reload = Fp8EngramEmbedding(
        config=_config(),
        table_sizes=SIZES,
        embedding_dim=WIDTH,
        ep_group=None,
        tp_group=None,
        expt_dp_group=None,
        block_size=BLOCK)
    bridge._load_engram_embedding(_fake_layer(fp8_reload).engram, state)
    assert torch.equal(fp8_reload(ids), fp8_embedding(ids))

    bf16_reload = upstream_embedding.EPShardedMultiTableEmbedding(
        config=_config(),
        table_sizes=SIZES,
        embedding_dim=WIDTH,
        init_method=None,
        ep_group=None,
        tp_group=None,
        expt_dp_group=None)
    bridge._load_engram_embedding(_fake_layer(bf16_reload).engram, state)
    reference = torch.cat([fp8_embedding.tables[0].fp8_rows, fp8_embedding.tables[1].fp8_rows])
    scales = torch.cat([fp8_embedding.tables[0].fp8_scales, fp8_embedding.tables[1].fp8_scales])
    from mcore_bridge.model.modules.engram_fp8 import decode_fp8_rows
    expected = decode_fp8_rows(reference, scales, BLOCK, torch.bfloat16)
    assert torch.equal(bf16_reload.tables[0].weight, expected[:SIZES[0]])
    assert torch.equal(bf16_reload.tables[1].weight, expected[SIZES[0]:])


def test_export_keeps_bf16_tables_as_one_flat_tensor():
    upstream, _ = _build_pair()
    bridge = _export_bridge()
    hf = {}
    bridge._set_layer_engram(_fake_layer(upstream), hf, to_mcore=False)
    assert hf['engram.embed.weight'].dtype == torch.bfloat16
    assert 'engram.embed.weight_scale_inv' not in hf
    assert torch.equal(hf['engram.embed.weight'], torch.cat([t.weight.data for t in upstream.tables], dim=0))


class _Tensor:

    def __init__(self, tensor):
        self.tensor = tensor

    def load(self):
        return self.tensor

    def load_slice(self, slices):
        return self.tensor[slices]


def test_swap_keeps_upstream_trainability_on_the_bf16_path():
    upstream = upstream_embedding.EPShardedMultiTableEmbedding(
        config=_config(),
        table_sizes=SIZES,
        embedding_dim=WIDTH,
        init_method=None,
        ep_group=None,
        tp_group=None,
        expt_dp_group=None)
    holder = SimpleNamespace(config=_config(), embedding=upstream)
    DeepseekV41Engram._maybe_use_fp8_tables(holder)
    assert holder.embedding is upstream
    # A bf16 checkpoint keeps upstream's semantics: the run's own policy decides whether to freeze.
    assert all(table.weight.requires_grad for table in holder.embedding.tables)


def test_cpu_offload_survives_complete_engram_initialization(monkeypatch):
    from mcore_bridge.model.modules.engram_cpu import CPUShardedEmbedding

    monkeypatch.setenv('NGRAM_EMBEDDING_CPU_OFFLOAD', '1')
    config = _config(fp8_declared=True)
    config.enable_hyper_connections = False
    config.hidden_size = 4
    config.layernorm_epsilon = 1e-6
    config.init_method = lambda value: value
    engram_config = SimpleNamespace(
        tokenizer_remap=None,
        multipliers=lambda layer: (1, 3),
        table_sizes=lambda layer: SIZES,
        head_dim=32,
        total_memory_dim=64,
        kernel_size=1,
        max_ngram_order=2,
        variant_spec=SimpleNamespace(projection_bias=False, zero_centered_gamma=False))
    module = DeepseekV41Engram(
        config,
        engram_config=engram_config,
        layer_number=1,
        pg_collection=SimpleNamespace(tp=None, ep=None, expt_dp=None))
    assert isinstance(module.embedding, CPUShardedEmbedding)
    module.to(dtype=torch.float32)
    assert all(t.weight.device.type == 'cpu' and not t.weight.requires_grad for t in module.embedding.tables)
    assert not list(module.embedding.parameters())
