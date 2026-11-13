# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch

from mcore_bridge.model.mm_gpts import deepseek_v41 as module


@pytest.mark.parametrize('shape', [(7, 12), (7, 3, 12), (7, 1, 3, 12)])
@pytest.mark.parametrize('inverse', [False, True])
def test_mla_rope_token_aligned_forward_backward(shape, inverse):
    from types import SimpleNamespace

    from mcore_bridge.model.gpts.deepseek_v4 import _apply_mla_rope

    torch.manual_seed(42)
    source = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    reference = source.detach().clone().requires_grad_(True)
    # Include reset and non-contiguous positions as in packing and CP.
    positions = torch.tensor([0, 1, 0, 1, 2, 9, 10], dtype=torch.float64)
    angles = positions[:, None] * torch.tensor([1., .1, .01, .001], dtype=torch.float64)
    freqs = torch.cat((angles, angles), dim=-1)[:, None, None, :]
    actual = _apply_mla_rope(
        source,
        freqs,
        config=SimpleNamespace(apply_rope_fusion=False, rotary_interleaved=False, mrope_section=None),
        cu_seqlens=torch.tensor([0, 2, 5, 7], dtype=torch.int32),
        cp_group=SimpleNamespace(size=lambda: 1),
        inverse=inverse)
    angles = angles.reshape(7, *([1] * (len(shape) - 2)), 4)
    cosine, sine = angles.cos(), angles.sin() * (-1 if inverse else 1)
    even, odd = reference[..., :8:2], reference[..., 1:8:2]
    rotated = torch.stack((even * cosine - odd * sine, odd * cosine + even * sine), dim=-1).flatten(-2)
    expected = torch.cat((rotated, reference[..., 8:]), dim=-1)
    assert actual.shape == source.shape
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[..., 8:], source[..., 8:], rtol=0, atol=0)
    gradient = torch.randn_like(expected)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(source.grad, reference.grad)


def test_npu_fp4_load_matches_independent_values(monkeypatch):
    monkeypatch.setattr(module, 'is_torch_npu_available', lambda: True)
    bridge = object.__new__(module.DeepseekV41Bridge)
    parameter = torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)
    packed = torch.tensor([[0x10, 0x32], [0x54, 0x76], [0x98, 0xba], [0xdc, 0xfe]], dtype=torch.uint8).view(torch.int8)
    scale = torch.tensor([[2.], [4.], [2.], [4.]])
    bridge._set_param(parameter, packed, scale)
    expected = torch.tensor([[0., 1., 2., 3.], [8., 12., 16., 24.], [0., -1., -2., -3.], [-8., -12., -16., -24.]])
    torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


def test_non_npu_parameter_loading_preserves_parent_path(monkeypatch):
    monkeypatch.setattr(module, 'is_torch_npu_available', lambda: False)
    calls = []
    monkeypatch.setattr(module.DeepseekV4Bridge, '_set_param', lambda self, *args: calls.append(args))
    bridge = object.__new__(module.DeepseekV41Bridge)
    parameter, source, scale = torch.zeros(2, 2), torch.ones(2, 2), None
    bridge._set_param(parameter, source, scale)
    assert calls == [(parameter, source, scale)]


@pytest.mark.parametrize('rows', [64, 100])
def test_engram_block_scales_match_expanded_reference(rows):
    weight = (torch.arange(rows * 64).reshape(rows, 64) % 7 - 3).to(torch.float8_e4m3fn)
    scale = torch.arange(1, ((rows + 31) // 32) * 2 + 1).reshape(-1, 2).float()
    expected = weight.float() * scale.repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)[:rows]
    actual = module.DeepseekV41Bridge._dequantize_engram_rows(weight, scale)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('npu', [False, True])
def test_fp8_parameter_tail_blocks(monkeypatch, npu):
    from types import SimpleNamespace

    monkeypatch.setattr(module, 'is_torch_npu_available', lambda: npu)
    bridge = object.__new__(module.DeepseekV41Bridge)
    bridge.config = SimpleNamespace(hf_config=SimpleNamespace(quantization_config={'weight_block_size': [32, 32]}))
    weight = torch.ones(100, 64).to(torch.float8_e4m3fn)
    scales = torch.arange(1, 9).reshape(4, 2).float()
    target = torch.nn.Parameter(torch.zeros(100, 64), requires_grad=False)
    bridge._set_param(target, weight, scales)
    expected = scales.repeat_interleave(32, 0).repeat_interleave(32, 1)[:100]
    torch.testing.assert_close(target, expected, rtol=0, atol=0)


def test_engram_row_scales_preserve_existing_format():
    weight = (torch.arange(32).reshape(4, 8) % 7 - 3).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5, 2.], [4., 8.], [2., 0.5], [8., 4.]])
    expected = weight.float() * scale.repeat_interleave(4, dim=1)
    actual = module.DeepseekV41Bridge._dequantize_engram_rows(weight, scale)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_pipeline_rotary_uses_hidden_payload_sequence():
    from megatron.core.pipeline_parallel.pipeline_payload import PipelinePayload
    from types import SimpleNamespace

    class Payload(PipelinePayload):
        tensors = (torch.zeros(3), torch.zeros(7, 1, 8))

        @property
        def tensor_specs(self):
            return (SimpleNamespace(name='side_input'), SimpleNamespace(name='hidden_states'))

    class Rotary:

        def get_rotary_seq_len(self, *args):
            raise AssertionError('Native tensor-only length lookup cannot consume a typed payload')

        def __call__(self, length, **kwargs):
            return torch.arange(length)

    for cp, sp, tp, expected in [(1, False, 4, 7), (2, False, 4, 14), (2, True, 3, 42)]:
        model = SimpleNamespace(
            decoder=SimpleNamespace(input_tensor=Payload()),
            config=SimpleNamespace(context_parallel_size=cp, sequence_parallel=sp, tensor_model_parallel_size=tp),
            rotary_pos_emb=Rotary(),
            compress_rotary_pos_emb=Rotary())
        result = module.DeepseekV41HybridStackModel._dsv4_rotary_pos_emb(model, None, None)
        for frequencies in result.values():
            torch.testing.assert_close(frequencies, torch.arange(expected))
