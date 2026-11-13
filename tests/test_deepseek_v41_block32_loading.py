# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

from mcore_bridge.model.gpts.deepseek_v4 import DeepseekV4Bridge
from mcore_bridge.model.mm_gpts.deepseek_v41 import DeepseekV41Bridge


@pytest.fixture
def bridge():
    result = DeepseekV41Bridge.__new__(DeepseekV41Bridge)
    result.config = SimpleNamespace(
        hf_config=SimpleNamespace(quantization_config={'weight_block_size': [32, 32]}),
        task_type='causal_lm',
        padded_vocab_size=1,
    )
    result.tp_size = result.etp_size = 1
    result.tp_rank = result.etp_rank = 0
    result._peft_format = False
    result._fp8_quantizer = None
    return result


def _source_and_expected(scale_dtype):
    source = ((torch.arange(4096).reshape(64, 64) % 8).float() - 4).to(torch.float8_e4m3fn)
    scales = torch.tensor([[0.5, 1.], [2., 4.]]).to(scale_dtype)
    expected = source.float() * scales.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
    return source, scales, expected


@pytest.mark.parametrize('scale_dtype', [torch.float32, torch.float8_e8m0fnu])
@pytest.mark.parametrize('mg_key, tp_dim', [('linear_q_up_proj.weight', 0), ('linear_proj.weight', 1)])
def test_projection_loading_uses_block32_scales(bridge, scale_dtype, mg_key, tp_dim):
    source, scales, expected = _source_and_expected(scale_dtype)
    bridge.tp_size = 2
    for rank in range(2):
        bridge.tp_rank = rank
        local_expected = expected.chunk(2, dim=tp_dim)[rank]
        projection = torch.nn.Linear(local_expected.shape[1], local_expected.shape[0], bias=False, dtype=torch.bfloat16)
        module = SimpleNamespace(**{mg_key.split('.')[0]: projection})
        state = {
            'projection.weight': SimpleNamespace(load=lambda: source),
            'projection.weight_scale_inv': SimpleNamespace(load=lambda: scales),
        }
        with patch.object(
                DeepseekV4Bridge,
                'fp8_quantizer',
                new_callable=PropertyMock,
                side_effect=AssertionError('Block32 source must bypass the TE block128 quantizer')):
            bridge._set_state_dict(module, mg_key, state, 'projection.weight', to_mcore=True)
        torch.testing.assert_close(projection.weight.detach(), local_expected.to(torch.bfloat16), rtol=0, atol=0)


def test_batched_source_scales_are_reshaped_with_parameters(bridge):
    source, scales, expected = _source_and_expected(torch.float32)
    parameter = torch.nn.Parameter(torch.empty(2, 64, 64, dtype=torch.bfloat16))
    bridge._set_param(parameter, source.unsqueeze(0).repeat(2, 1, 1), scales.unsqueeze(0).repeat(2, 1, 1))
    torch.testing.assert_close(
        parameter.detach(), expected.unsqueeze(0).repeat(2, 1, 1).to(torch.bfloat16), rtol=0, atol=0)


def test_block32_source_rejects_fp8_training_parameters(bridge):
    source, scales, _ = _source_and_expected(torch.float32)
    parameter = torch.nn.Parameter(torch.zeros(64, 64, dtype=torch.bfloat16))
    with patch.object(bridge, '_is_fp8_param', return_value=True):
        with pytest.raises(ValueError, match='BF16 training parameters'):
            bridge._set_param(parameter, source, scales)
    assert torch.count_nonzero(parameter) == 0


@pytest.mark.parametrize('quantization', [None, {}, {'weight_block_size': [128, 128]}])
def test_other_quantization_formats_delegate_to_v4(bridge, quantization, monkeypatch):
    monkeypatch.setattr('mcore_bridge.model.mm_gpts.deepseek_v41.is_torch_npu_available', lambda: False)
    bridge.config.hf_config.quantization_config = quantization
    source, scales, _ = _source_and_expected(torch.float32)
    parameter = torch.nn.Parameter(torch.empty(64, 64, dtype=torch.bfloat16))
    with patch.object(DeepseekV4Bridge, '_set_param') as parent:
        bridge._set_param(parameter, source, scales)
    parent.assert_called_once_with(parameter, source, scales)


def test_unquantized_source_loading_stays_unchanged(bridge):
    source = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    parameter = torch.nn.Parameter(torch.zeros_like(source))
    bridge._set_param(parameter, source, None)
    torch.testing.assert_close(parameter.detach(), source, rtol=0, atol=0)


@pytest.mark.parametrize('scale_dtype', [torch.float32, torch.float8_e8m0fnu])
def test_engram_wkv_block32_loading_preserves_row_scaled_tables(bridge, scale_dtype):
    source, scales, expected = _source_and_expected(scale_dtype)
    table = SimpleNamespace(
        global_num_embeddings=4,
        local_num_embeddings=4,
        row_start=0,
        weight=torch.nn.Parameter(torch.empty(4, 64, dtype=torch.bfloat16)),
    )
    engram = SimpleNamespace(
        layer_number=1,
        num_streams=1,
        hidden_size=32,
        engram_config=SimpleNamespace(total_memory_dim=64),
        embedding=SimpleNamespace(tables=[table]),
        key_projection=SimpleNamespace(weight=torch.nn.Parameter(torch.empty(32, 64, dtype=torch.bfloat16))),
        value_projection=SimpleNamespace(weight=torch.nn.Parameter(torch.empty(32, 64, dtype=torch.bfloat16))),
        query_norm=SimpleNamespace(weight=torch.nn.Parameter(torch.empty(32))),
        key_norm=SimpleNamespace(weight=torch.nn.Parameter(torch.empty(32))),
    )
    bridge.config.engram_num_embeddings = [4]
    bridge.config.engram_layer_ids = [0]
    raw_table = source[:4]
    row_scales = torch.tensor([[2., 4.]]).expand(4, 2).to(scale_dtype)
    table_slices = []

    def load_rows(rows):
        table_slices.append((rows.start, rows.stop))
        return raw_table[rows]

    state = {
        'engram.embed.weight': SimpleNamespace(load_slice=load_rows),
        'engram.embed.weight_scale_inv': SimpleNamespace(load_slice=lambda rows: row_scales[rows]),
        'engram.wkv.weight': SimpleNamespace(load=lambda: source),
        'engram.wkv.weight_scale_inv': SimpleNamespace(load=lambda: scales),
        'engram.q_weight': SimpleNamespace(load=lambda: torch.arange(32).float()),
        'engram.k_weight': SimpleNamespace(load=lambda: torch.arange(32).float() + 1),
    }
    bridge._set_layer_engram(SimpleNamespace(engram=engram), state, to_mcore=True)
    torch.testing.assert_close(engram.key_projection.weight.detach(), expected[:32].to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(
        engram.value_projection.weight.detach(), expected[32:].to(torch.bfloat16), rtol=0, atol=0)
    table_expected = raw_table.float() * row_scales.float().repeat_interleave(32, 1)
    torch.testing.assert_close(table.weight.detach(), table_expected.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(engram.query_norm.weight.detach(), torch.arange(32).float(), rtol=0, atol=0)
    torch.testing.assert_close(engram.key_norm.weight.detach(), torch.arange(32).float() + 1, rtol=0, atol=0)
    assert table_slices == [(0, 4)]


def test_engram_unquantized_and_invalid_scale_shapes():
    source, _, _ = _source_and_expected(torch.float32)
    assert DeepseekV41Bridge._dequantize_engram_rows(source, None) is source
    for scales in (torch.ones(3, 3), torch.ones(64, 3), torch.ones(64)):
        with pytest.raises(ValueError):
            DeepseekV41Bridge._dequantize_engram_rows(source, scales)
