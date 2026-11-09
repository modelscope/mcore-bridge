# Copyright (c) ModelScope Contributors. All rights reserved.
"""Optional ffpa-attn path for Gemma 4 full-attention layers (GEMMA4_FFPA_ATTN=1). The CUDA tests use the
31B full-attention shapes under TP4 and need ffpa-attn on a Hopper (SM90) GPU."""
import ast
import pytest
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from types import ModuleType, SimpleNamespace

_NAMES = {'get_ffpa_softmax_scale', 'ffpa_varlen_attention'}
NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 8, 1, 512


def _load():
    path = Path(__file__).resolve().parents[1] / 'src/mcore_bridge/model/mm_gpts/gemma4.py'
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in _NAMES]
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {'torch': torch, 'PackedSeqParams': object}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace


def test_softmax_scale_follows_model_config():
    get_scale = _load()['get_ffpa_softmax_scale']
    # Gemma 4 sets `softmax_scale = 1.0` (q/k norms); the kernel default 1/sqrt(512) is wrong for it.
    assert get_scale(SimpleNamespace(softmax_scale=1.0), HEAD_DIM) == 1.0
    assert get_scale(SimpleNamespace(softmax_scale=None), HEAD_DIM) == HEAD_DIM**-0.5


def _gemma4_softmax_scale():
    """Scale from the converted Gemma 4 config, as the attention layer reads it."""
    pytest.importorskip('transformer_engine')
    gemma4_config = pytest.importorskip('transformers.models.gemma4.configuration_gemma4')
    from mcore_bridge.config import hf_to_mcore_config
    text_config = dict(
        num_hidden_layers=6,
        layer_types=['sliding_attention'] * 5 + ['full_attention'],
        hidden_size=5376,
        num_attention_heads=32,
        head_dim=256,
        global_head_dim=512,
        num_key_value_heads=16,
        num_global_key_value_heads=4,
        attention_k_eq_v=True,
        hidden_size_per_layer_input=0)
    config = SimpleNamespace(**hf_to_mcore_config(gemma4_config.Gemma4Config(text_config=text_config)))
    return _load()['get_ffpa_softmax_scale'](config, HEAD_DIM)


def _cuda_ffpa():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip('needs a Hopper (SM90) GPU')
    pytest.importorskip('ffpa_attn.cute')
    return _load()['ffpa_varlen_attention']


def _packed(lens):
    cu_seqlens = torch.tensor([0] + lens, device='cuda').cumsum(0).to(torch.int32)
    return SimpleNamespace(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=max(lens),
        max_seqlen_kv=max(lens))


def _inputs(total, requires_grad=True):
    # Gemma 4 applies RMSNorm to q and k, so their entries have unit scale.
    q, k, v = (torch.randn(total, h, HEAD_DIM, device='cuda') for h in (NUM_HEADS, NUM_KV_HEADS, NUM_KV_HEADS))
    q, k = F.rms_norm(q, (HEAD_DIM, )), F.rms_norm(k, (HEAD_DIM, ))
    return [x.to(torch.bfloat16).requires_grad_(requires_grad) for x in (q, k, v)]


def _reference(q, k, v, cu_seqlens, scale):
    outs = []
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        qq = q[start:end].float().transpose(0, 1)
        kk, vv = (x[start:end].float().transpose(0, 1).repeat_interleave(NUM_HEADS // NUM_KV_HEADS, 0) for x in (k, v))
        outs.append(F.scaled_dot_product_attention(qq, kk, vv, is_causal=True, scale=scale).transpose(0, 1))
    return torch.cat(outs)


def _rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize('lens', [[37, 1000, 3, 512, 2620], [8192], [1500, 1700, 1800, 1600, 1592]])
def test_ffpa_matches_reference(lens):
    attention = _cuda_ffpa()
    scale = _gemma4_softmax_scale()
    assert scale == 1.0
    torch.manual_seed(0)
    packed = _packed(lens)
    q, k, v = _inputs(sum(lens))
    out = attention(q, k, v, packed, scale)
    grad = torch.randn_like(out)
    out.backward(grad)

    q_ref, k_ref, v_ref = (x.detach().float().requires_grad_() for x in (q, k, v))
    ref = _reference(q_ref, k_ref, v_ref, packed.cu_seqlens_q, scale)
    ref.backward(grad.float())
    assert _rel_err(out, ref) < 1e-2
    for x, x_ref in ((q, q_ref), (k, k_ref), (v, v_ref)):
        assert _rel_err(x.grad, x_ref.grad) < 2e-2
    # The default 1/sqrt(head_dim) scale would give a clearly different result.
    assert _rel_err(attention(q, k, v, packed, HEAD_DIM**-0.5), ref) > 0.1


def test_ffpa_documents_are_isolated():
    attention = _cuda_ffpa()
    torch.manual_seed(0)
    lens = [37, 1000, 3, 512, 2620]
    packed = _packed(lens)
    q, k, v = _inputs(sum(lens), requires_grad=False)
    out = attention(q, k, v, packed, 1.0)
    first_end = lens[0]
    k2, v2 = k.clone(), v.clone()
    k2[:first_end] += 3.0
    v2[:first_end] -= 2.0
    out2 = attention(q, k2, v2, packed, 1.0)
    assert torch.equal(out[first_end:], out2[first_end:])


def test_padded_packed_sequences_are_rejected():
    attention = _cuda_ffpa()
    packed = _packed([16, 16])
    packed.cu_seqlens_q_padded = torch.tensor([0, 32, 64], device='cuda', dtype=torch.int32)
    q, k, v = _inputs(32, requires_grad=False)
    with pytest.raises(NotImplementedError, match='padded packed sequences'):
        attention(q, k, v, packed, 1.0)


# The tests below call the real `Gemma4SelfAttention` methods on a stand-in `self`; building the full module would
# need a Megatron process group setup and a Gemma 4 HF config.


def _attention_cls():
    pytest.importorskip('megatron.core')
    from mcore_bridge.model.mm_gpts.gemma4 import Gemma4SelfAttention
    return Gemma4SelfAttention


def _layer(is_sliding=False, head_dim=HEAD_DIM, cp_size=1, attention_dropout=0.0):
    return SimpleNamespace(
        is_sliding=is_sliding,
        head_dim=head_dim,
        pg_collection=SimpleNamespace(cp=SimpleNamespace(size=lambda: cp_size)),
        config=SimpleNamespace(attention_dropout=attention_dropout, softmax_scale=1.0))


@pytest.fixture
def ffpa_env(monkeypatch):
    """GEMMA4_FFPA_ATTN=1 on a Hopper device with an importable CuTeDSL backend (both faked)."""
    monkeypatch.setenv('GEMMA4_FFPA_ATTN', '1')
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda *args: (9, 0))
    cute = ModuleType('ffpa_attn.cute')
    cute._ffpa_attn_varlen_cute = object()
    monkeypatch.setitem(sys.modules, 'ffpa_attn', ModuleType('ffpa_attn'))
    monkeypatch.setitem(sys.modules, 'ffpa_attn.cute', cute)
    return monkeypatch


@pytest.mark.parametrize('kwargs, enabled', [(dict(), True), (dict(is_sliding=True), False),
                                             (dict(head_dim=256), False)])
def test_switch_covers_head_dim_512_full_attention_layers_only(ffpa_env, kwargs, enabled):
    layer = _layer(**kwargs)
    _attention_cls()._init_ffpa_attn(layer)
    assert layer.use_ffpa_attn is enabled
    if enabled:
        assert layer.ffpa_softmax_scale == 1.0


def test_switch_with_installed_package(monkeypatch):
    _cuda_ffpa()
    monkeypatch.setenv('GEMMA4_FFPA_ATTN', '1')
    layer = _layer()
    _attention_cls()._init_ffpa_attn(layer)
    assert layer.use_ffpa_attn is True


def test_switch_is_off_by_default(ffpa_env):
    ffpa_env.delenv('GEMMA4_FFPA_ATTN')
    layer = _layer()
    _attention_cls()._init_ffpa_attn(layer)
    assert layer.use_ffpa_attn is False


def test_missing_cute_backend_raises_import_error(ffpa_env):
    ffpa_env.setitem(sys.modules, 'ffpa_attn.cute', None)  # what Python does for a module that failed to import
    with pytest.raises(ImportError, match='CuTeDSL backend'):
        _attention_cls()._init_ffpa_attn(_layer())


@pytest.mark.parametrize('capability', [(8, 0), (10, 0)])
def test_non_hopper_gpu_is_rejected(ffpa_env, capability):
    ffpa_env.setattr(torch.cuda, 'get_device_capability', lambda *args: capability)
    with pytest.raises(NotImplementedError, match='Hopper'):
        _attention_cls()._init_ffpa_attn(_layer())


@pytest.mark.parametrize('kwargs, match', [(dict(cp_size=2), 'context parallelism'),
                                           (dict(attention_dropout=0.1), 'attention dropout')])
def test_unsupported_configs_are_rejected(ffpa_env, kwargs, match):
    with pytest.raises(NotImplementedError, match=match):
        _attention_cls()._init_ffpa_attn(_layer(**kwargs))


def test_unsupported_masks_are_rejected():
    cls = _attention_cls()
    from megatron.core.transformer.enums import AttnMaskType
    layer = SimpleNamespace(attn_mask_type=AttnMaskType.arbitrary)
    with pytest.raises(NotImplementedError, match='causal masks only'):
        cls._forward_ffpa_attention(layer, None, None, None, None, None)
    layer.attn_mask_type = AttnMaskType.causal
    with pytest.raises(NotImplementedError, match='attention_bias'):
        cls._forward_ffpa_attention(layer, None, None, None, torch.zeros(1), None)


@pytest.mark.parametrize('use_ffpa_attn, qkv_format, expected', [(True, 'thd', 'ffpa'), (True, 'sbhd', 'te'),
                                                                 (True, None, 'te'), (False, 'thd', 'te')])
def test_core_attention_dispatch(use_ffpa_attn, qkv_format, expected):
    cls = _attention_cls()
    calls = []
    out = torch.zeros(4, 2, 3)
    layer = SimpleNamespace(
        use_ffpa_attn=use_ffpa_attn,
        attn_mask_type=None,
        checkpoint_core_attention=False,
        training=True,
        _forward_ffpa_attention=lambda *args: calls.append('ffpa') or out,
        core_attention=lambda *args, **kwargs: calls.append('te') or out)
    packed = None if qkv_format is None else SimpleNamespace(qkv_format=qkv_format)
    cls._forward_core_attention(layer, None, None, None, None, packed_seq_params=packed)
    assert calls == [expected]


def test_recompute_matches_plain_forward():
    _cuda_ffpa()
    cls = _attention_cls()
    from megatron.core.transformer.enums import AttnMaskType
    torch.manual_seed(0)
    lens = [37, 1000, 3, 512, 2620]
    packed = _packed(lens)
    q, k, v = _inputs(sum(lens))
    grad = torch.randn(sum(lens), NUM_HEADS, HEAD_DIM, device='cuda', dtype=torch.bfloat16)
    results = []
    for checkpoint in (False, True):
        layer = SimpleNamespace(
            attn_mask_type=AttnMaskType.causal,
            checkpoint_core_attention=checkpoint,
            training=True,
            ffpa_softmax_scale=1.0)
        qq, kk, vv = (x.detach().clone().requires_grad_() for x in (q, k, v))
        out = cls._forward_ffpa_attention(layer, qq, kk, vv, None, packed)
        out.backward(grad)
        results.append((out.detach(), qq.grad, kk.grad, vv.grad))
    for plain, recomputed in zip(*results):
        assert torch.equal(plain, recomputed)
