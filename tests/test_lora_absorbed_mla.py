# Copyright (c) ModelScope Contributors. All rights reserved.
"""Adapter-aware weight for MLA matrix absorption.

Absorbed MLA reads ``linear_kv_up_proj.weight`` instead of calling the module, so a LoRA wrapper
left both the frozen base and no autograd edge: correct on step 0, inert afterwards. Each test
below guards one failure: (0) missing import wiring; (1) no grads; (2) wrong or partial effective
weight; (3) step-0 drift; (4) double counting or a 1-ulp mismatch with ``merge()``; (5) ``.weight``
semantics; (6) full-parameter path broken by the fallback; (7) combined-only layout; (8) silent
base fallback for a new wrapper; (9) non-``AttributeError`` on grouped; (10) mutating the layer;
(11) double wrapping; (12) dtype or delta loss in BF16; (13) TP2 geometry; (14) disable-after-merge
diverging from ``forward``; (15) a training-time ``lora_dropout`` being silently ignored; (16) a
lora_bias term silently dropped; (17) package import breaking on a Core revision without the getter;
(18) the use-time guard missing a split ``linear_k_up_proj`` / ``linear_v_up_proj`` adapter.
"""
import os
import pytest
import subprocess
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')

HEADS, QK_DIM, V_DIM, KV_LORA, RANK, ALPHA = 2, 2, 2, 3, 2, 2
TP = 2

requires_two_gpus = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='requires two CUDA devices')
requires_nccl = pytest.mark.skipif(not dist.is_nccl_available(), reason='requires NCCL')


@contextmanager
def _parallel_context(tp=1):
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group('nccl', init_method=f'file:///tmp/lora-absorbed-{uuid.uuid4().hex}', rank=0,
                                world_size=1)
    already_initialized = parallel_state.model_parallel_is_initialized()
    if not already_initialized:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp)
        model_parallel_cuda_manual_seed(123)
    try:
        yield
    finally:
        if not already_initialized:
            parallel_state.destroy_model_parallel()


@pytest.fixture(scope='module', autouse=True)
def _mpu():
    # Production imports the bridge before building a model; this also installs the patch asserted below.
    import mcore_bridge  # noqa: F401
    with _parallel_context():
        yield


def _config(tp=1, dtype=torch.float32):
    from megatron.core.transformer import TransformerConfig
    return TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=HEADS * tp,
        tensor_model_parallel_size=tp,
        params_dtype=dtype,
    )


def _wrap(dtype=torch.float32, tp=1, seed=0, out_features=None, lora_dropout=0.0, lora_bias=False):
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear

    from mcore_bridge.tuners.lora import LoraParallelLinear
    torch.manual_seed(seed)
    config = _config(tp, dtype)
    base = TEColumnParallelLinear(
        input_size=KV_LORA,
        output_size=out_features or HEADS * (QK_DIM + V_DIM),
        config=config,
        init_method=config.init_method,
        gather_output=False,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
    ).to(dtype)
    base.weight.requires_grad_(False)
    lora = LoraParallelLinear(
        base,
        'default',
        r=RANK,
        lora_alpha=ALPHA,
        lora_dropout=lora_dropout,
        init_lora_weights=True,
        lora_bias=lora_bias)
    return base, lora


def _effective(lora, heads=None, head_dim=QK_DIM + V_DIM):
    """Independent FP32 oracle: base + (B @ A) * scaling, viewed per head."""
    heads = HEADS if heads is None else heads
    weight_a, weight_b = lora.lora_A['default'].weight, lora.lora_B['default'].weight
    merged = lora.get_base_layer().weight.float() + (weight_b.float() @ weight_a.float()) * lora.scaling['default']
    return merged.view(heads, head_dim, KV_LORA)


def _holder(layer, heads=None):
    return SimpleNamespace(
        _uses_combined_kv_up_projection=True,
        num_attention_heads_per_partition=HEADS if heads is None else heads,
        config=SimpleNamespace(kv_lora_rank=KV_LORA, qk_head_dim=QK_DIM, v_head_dim=V_DIM),
        linear_kv_up_proj=layer,
    )


def _native():
    return pytest.importorskip(
        'megatron.core.transformer.experimental_attention_variant.absorbed_mla').AbsorbedMLASelfAttention


def _loss(k, v):
    q = torch.randn(5, 1, k.shape[0], k.shape[1], device=k.device, dtype=k.dtype, requires_grad=True)
    latent = torch.randn(5, 1, k.shape[0], KV_LORA, device=k.device, dtype=k.dtype, requires_grad=True)
    return (torch.einsum('sbhd,hdk->sbhk', q, k).square().mean()
            + torch.einsum('sbhk,hdk->sbhd', latent, v).square().mean())


def test_import_wires_the_absorbed_getter():
    assert getattr(_native(), '_mcore_lora_patched', False), (
        'importing mcore_bridge must route the absorbed getter through the adapter accessor')


def test_k_absorption_trains_the_adapter():
    base, lora = _wrap()
    getter = _native()._get_kv_up_weights
    weight_a, weight_b = lora.lora_A['default'].weight, lora.lora_B['default'].weight
    frozen = base.weight.detach().clone()
    optimizer = torch.optim.SGD([weight_a, weight_b], lr=0.5)

    optimizer.zero_grad(set_to_none=True)
    k, v = getter(_holder(lora))
    _loss(k, v).backward()
    assert weight_b.grad is not None and weight_b.grad.count_nonzero() > 0, 'B must receive the K/V signal'
    assert weight_a.grad is not None, 'A must be in the graph even when its grad is zero at B=0'
    assert k.requires_grad and v.requires_grad
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    k, v = getter(_holder(lora))
    _loss(k, v).backward()
    assert weight_a.grad is not None and weight_a.grad.count_nonzero() > 0, 'A must update once B is nonzero'
    optimizer.step()

    assert weight_b.count_nonzero() > 0, 'the adapter must actually move'
    assert weight_a.count_nonzero() > 0
    assert torch.equal(frozen, base.weight.detach()), 'the frozen base must never change'


def test_absorbed_weight_matches_core_layout():
    """Oracle = Core's own getter on a plain projection holding the effective weight.

    This pins the copied view/slice to upstream's layout; a third handwritten view could not detect
    an upstream layout change that keeps the element count.
    """
    _, lora = _wrap()
    with torch.no_grad():
        lora.lora_B['default'].weight.normal_()
    effective = _effective(lora).reshape(HEADS * (QK_DIM + V_DIM), KV_LORA)
    expected = _origin_getter()(_holder(_plain_projection(effective)))
    k, v = _native()._get_kv_up_weights(_holder(lora))
    torch.testing.assert_close(k, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(v, expected[1], rtol=0, atol=0)
    # The V getter takes the same path; a fix that only touches K would leave this on the base weight.
    holder = _holder(lora)
    holder._get_kv_up_weights = lambda: _native()._get_kv_up_weights(holder)
    torch.testing.assert_close(_native()._get_v_up_weight(holder), expected[1], rtol=0, atol=0)


def test_zero_initialized_adapter_reproduces_unadapted_output():
    base, lora = _wrap()
    k, v = _native()._get_kv_up_weights(_holder(lora))
    upstream_k, upstream_v = _origin_getter()(_holder(lora))
    torch.testing.assert_close(k, upstream_k, rtol=0, atol=0)
    torch.testing.assert_close(v, upstream_v, rtol=0, atol=0)
    base_view = base.weight.detach().view(HEADS, QK_DIM + V_DIM, KV_LORA)
    assert torch.equal(k, base_view[:, :QK_DIM])
    assert torch.equal(v, base_view[:, QK_DIM:])


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16], ids=['fp32', 'bf16'])
def test_disabled_and_merged_adapters_fall_back_to_base(dtype):
    base, lora = _wrap(dtype=dtype)
    for name in ('lora_A', 'lora_B'):
        getattr(lora, name)['default'].to(torch.float32)
    with torch.no_grad():
        lora.lora_B['default'].weight.normal_(std=0.5)
    delta = (lora.lora_B['default'].weight.float()
             @ lora.lora_A['default'].weight.float()) * lora.scaling['default']
    # What `merge()` writes into the base: one FP32 add, rounded once to the base dtype.
    merged_reference = (base.weight.float() + delta).to(dtype)
    base_view = base.weight.detach().view(HEADS, QK_DIM + V_DIM, KV_LORA)

    lora.enable_adapters(False)
    torch.testing.assert_close(_native()._get_kv_up_weights(_holder(lora))[0], base_view[:, :QK_DIM], rtol=0, atol=0)
    lora.enable_adapters(True)

    # Bit-equality with `merge()`, or training-time and exported weights drift by one ulp.
    torch.testing.assert_close(lora.get_weight_for_absorption(), merged_reference, rtol=0, atol=0)

    lora.merge(safe_merge=True)
    assert lora.merged, 'merge() must mark the adapter as merged so the delta is not added twice'
    k, _ = _native()._get_kv_up_weights(_holder(lora))
    torch.testing.assert_close(k, merged_reference.view(HEADS, QK_DIM + V_DIM, KV_LORA)[:, :QK_DIM], rtol=0, atol=0)


def test_weight_property_still_returns_the_frozen_base():
    base, lora = _wrap()
    assert lora.weight is base.weight
    assert not hasattr(lora.get_weight_for_absorption(), 'base_layer')


def test_plain_projection_without_adapter_is_unchanged():
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
    config = _config()
    plain = TEColumnParallelLinear(
        input_size=KV_LORA,
        output_size=HEADS * (QK_DIM + V_DIM),
        config=config,
        init_method=config.init_method,
        gather_output=False,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
    )
    assert not hasattr(plain, 'get_weight_for_absorption')
    k, v = _native()._get_kv_up_weights(_holder(plain))
    base_view = plain.weight.detach().view(HEADS, QK_DIM + V_DIM, KV_LORA)
    torch.testing.assert_close(k, base_view[:, :QK_DIM], rtol=0, atol=0)
    torch.testing.assert_close(v, base_view[:, QK_DIM:], rtol=0, atol=0)


def _origin_getter():
    """The upstream ``_get_kv_up_weights`` body, unwrapped from the bridge patch."""
    getter = _native()._get_kv_up_weights
    for cell in getattr(getter, '__closure__', None) or ():
        value = cell.cell_contents
        if callable(value) and getattr(value, '__name__', '') == '_get_kv_up_weights':
            return value
    return getter


def _plain_projection(weight):
    """Plain TE linear holding `weight`, so Core's own getter can serve as the layout oracle."""
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
    config = _config()
    plain = TEColumnParallelLinear(
        input_size=KV_LORA,
        output_size=weight.shape[0],
        config=config,
        init_method=config.init_method,
        gather_output=False,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
    )
    with torch.no_grad():
        plain.weight.copy_(weight.to(plain.weight.dtype))
    return plain


def test_split_layout_projections_are_both_routed():
    # 0.18/0.20 only implement the combined layout; the split branches are dev-era.
    if 'linear_v_up_proj' not in _origin_getter().__code__.co_names:
        pytest.skip('this Megatron revision only implements the combined kv-up layout')
    _, k_layer = _wrap(seed=0, out_features=HEADS * QK_DIM)
    _, v_layer = _wrap(seed=1, out_features=HEADS * V_DIM)
    with torch.no_grad():
        k_layer.lora_B['default'].weight.normal_()
        v_layer.lora_B['default'].weight.normal_()
    holder = SimpleNamespace(
        _uses_combined_kv_up_projection=False,
        num_attention_heads_per_partition=HEADS,
        config=SimpleNamespace(qk_head_dim=QK_DIM, v_head_dim=V_DIM, kv_lora_rank=KV_LORA),
        linear_k_up_proj=k_layer,
        linear_v_up_proj=v_layer,
    )
    expected = _origin_getter()(
        SimpleNamespace(
            _uses_combined_kv_up_projection=False,
            num_attention_heads_per_partition=HEADS,
            config=SimpleNamespace(qk_head_dim=QK_DIM, v_head_dim=V_DIM, kv_lora_rank=KV_LORA),
            linear_k_up_proj=_plain_projection(_effective(k_layer, head_dim=QK_DIM).reshape(-1, KV_LORA)),
            linear_v_up_proj=_plain_projection(_effective(v_layer, head_dim=V_DIM).reshape(-1, KV_LORA)),
        ))
    k, v = _native()._get_kv_up_weights(holder)
    torch.testing.assert_close(k, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(v, expected[1], rtol=0, atol=0)


def test_adapter_without_accessor_fails_loudly(monkeypatch):
    from mcore_bridge.tuners.lora import LoraParallelLinear
    _, lora = _wrap()
    monkeypatch.delattr(LoraParallelLinear, 'get_weight_for_absorption')
    with pytest.raises(TypeError, match='get_weight_for_absorption'):
        _native()._get_kv_up_weights(_holder(lora))


def test_grouped_projection_raises_attribute_error():
    _, lora = _wrap()
    lora.is_grouped = True
    with pytest.raises(AttributeError):
        lora.get_weight_for_absorption()


def test_grouped_projection_fails_at_the_read_site():
    # A grouped base has no single weight: the read site must fail loudly, not use `.weight`.
    _, lora = _wrap()
    lora.is_grouped = True
    with pytest.raises(AttributeError):
        _native()._get_kv_up_weights(_holder(lora))


def test_disable_after_merge_matches_forward():
    base, lora = _wrap()
    unadapted = base.weight.detach().clone()
    with torch.no_grad():
        lora.lora_B['default'].weight.normal_(std=0.5)
    lora.merge(safe_merge=True)
    assert not torch.equal(base.weight.detach(), unadapted), 'merge() must have written the delta'
    lora.enable_adapters(False)

    # `forward()` unmerges before using the base; the accessor must not report the merged weight.
    # The merge/unmerge round-trip is not bitwise in FP32, hence the loose relative tolerance.
    tolerance = dict(rtol=1e-4, atol=1e-7)
    torch.testing.assert_close(lora.get_weight_for_absorption(), unadapted, **tolerance)
    assert not lora.merged, 'the merged delta must be removed, as `forward()` does'
    torch.testing.assert_close(_native()._get_kv_up_weights(_holder(lora))[0],
                               unadapted.view(HEADS, QK_DIM + V_DIM, KV_LORA)[:, :QK_DIM], **tolerance)


def test_training_dropout_is_rejected():
    base, lora = _wrap(lora_dropout=0.1)
    with torch.no_grad():
        lora.lora_B['default'].weight.normal_(std=0.5)
    lora.train()
    # Ignoring it would silently change the computation the config asked for.
    with pytest.raises(NotImplementedError, match='lora_dropout'):
        lora.get_weight_for_absorption()
    with pytest.raises(NotImplementedError, match='lora_dropout'):
        _native()._get_kv_up_weights(_holder(lora))

    lora.eval()  # dropout is the identity in eval, so the static weight is exact
    reference = base.weight.float() + (lora.lora_B['default'].weight.float()
                                       @ lora.lora_A['default'].weight.float()) * lora.scaling['default']
    torch.testing.assert_close(lora.get_weight_for_absorption(), reference.to(base.weight.dtype), rtol=0, atol=0)


@pytest.mark.parametrize('field', ['linear_kv_up_proj', 'linear_k_up_proj', 'linear_v_up_proj'])
def test_missing_hook_guard_rejects_adapter_wrapped_projection(field):
    """Old revisions split the projection, so the guard must cover all three attributes."""
    from mcore_bridge.patcher import _guard_absorbed_mla_without_hook

    class _Layer:
        def forward(self):
            return 'ok'

    _guard_absorbed_mla_without_hook(_Layer)
    _, lora = _wrap()
    wrapped = _Layer()
    setattr(wrapped, field, lora)
    with pytest.raises(NotImplementedError, match=field):
        wrapped.forward()

    plain = _Layer()
    setattr(plain, field, _wrap()[0])
    assert plain.forward() == 'ok'


def test_lora_bias_is_rejected():
    _, lora = _wrap(lora_bias=True)
    with pytest.raises(NotImplementedError, match='lora_bias'):
        lora.get_weight_for_absorption()
    with pytest.raises(NotImplementedError, match='lora_bias'):
        _native()._get_kv_up_weights(_holder(lora))


def test_import_survives_a_revision_without_the_getter():
    script = ('import megatron.core.transformer.experimental_attention_variant.absorbed_mla as m\n'
              'delattr(m.AbsorbedMLASelfAttention, "_get_kv_up_weights")\n'
              'import mcore_bridge\n'
              'print("imported", getattr(m.AbsorbedMLASelfAttention, "_mcore_lora_guard", False))\n')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'imported True' in result.stdout, 'the package must import and install the use-time guard'


def test_getter_does_not_mutate_the_layer():
    class _Attention(torch.nn.Module):
        """Real layer shape: the projection is a registered submodule."""

        def __init__(self, layer):
            super().__init__()
            self.linear_kv_up_proj = layer
            self._uses_combined_kv_up_projection = True
            self.num_attention_heads_per_partition = HEADS
            self.config = SimpleNamespace(kv_lora_rank=KV_LORA, qk_head_dim=QK_DIM, v_head_dim=V_DIM)

    _, lora = _wrap()
    attention = _Attention(lora)
    registered = dict(attention._modules)
    _native()._get_kv_up_weights(attention)
    assert attention.linear_kv_up_proj is lora
    assert attention._modules == registered
    assert 'linear_kv_up_proj' not in attention.__dict__


def test_patch_is_idempotent():
    from mcore_bridge.patcher import _patch_absorbed_mla_lora
    native = _native()
    was_wired = getattr(native, '_mcore_lora_patched', False)
    original = native._get_kv_up_weights
    try:
        _patch_absorbed_mla_lora()
        wired = native._get_kv_up_weights
        _patch_absorbed_mla_lora()
        assert native._get_kv_up_weights is wired, 'a repeated patch call must not wrap again'
    finally:
        if not was_wired:
            # Do not let this test mask a missing apply_patch() wiring for the tests that follow.
            native._get_kv_up_weights = original
            if hasattr(native, '_mcore_lora_patched'):
                delattr(native, '_mcore_lora_patched')


def test_bf16_base_with_fp32_adapter_keeps_the_delta_visible():
    base, lora = _wrap(dtype=torch.bfloat16)
    for name in ('lora_A', 'lora_B'):
        getattr(lora, name)['default'].to(torch.float32)
    with torch.no_grad():
        lora.lora_B['default'].weight.normal_(std=0.05)  # ~1e-2, above the BF16 ulp of these weights
    merged = lora.get_weight_for_absorption()
    assert merged.dtype is torch.bfloat16
    reference = base.weight.float() + (lora.lora_B['default'].weight.float()
                                       @ lora.lora_A['default'].weight.float()) * lora.scaling['default']
    delta = reference - base.weight.float()
    assert delta.abs().max() > 0
    error = ((merged.float() - base.weight.float()) - delta).abs().max() / delta.abs().max()
    assert error < 2**-7, f'the delta must survive its own rounding, got {error}'

    k, v = _native()._get_kv_up_weights(_holder(lora))  # bf16 activation dtype, same graph
    assert k.dtype is torch.bfloat16 and k.requires_grad
    _loss(k, v).backward()
    assert lora.lora_B['default'].weight.grad is not None


def _tp2_worker(rank, init_file):
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=f'file://{init_file}', rank=rank, world_size=TP)
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    try:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=TP)
        model_parallel_cuda_manual_seed(123)
        base, lora = _wrap(tp=TP, out_features=HEADS * TP * (QK_DIM + V_DIM))
        local_heads = HEADS  # the projection carries a realistic global geometry
        getter = _native()._get_kv_up_weights
        with torch.no_grad():
            lora.lora_B['default'].weight.normal_()
        k, v = getter(_holder(lora, heads=local_heads))
        reference = _effective(lora, heads=local_heads)  # base and B are the local shard, A is replicated
        torch.testing.assert_close(k, reference[:, :QK_DIM], rtol=0, atol=0)
        torch.testing.assert_close(v, reference[:, QK_DIM:], rtol=0, atol=0)
        assert tuple(lora.get_weight_for_absorption().shape) == tuple(base.weight.shape)

        optimizer = torch.optim.SGD([lora.lora_B['default'].weight], lr=0.1)
        optimizer.zero_grad(set_to_none=True)
        k, v = getter(_holder(lora, heads=local_heads))
        _loss(k, v).backward()
        grad = lora.lora_B['default'].weight.grad
        assert grad is not None and grad.count_nonzero() > 0, f'rank {rank}: B must get a shard-local gradient'
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@requires_two_gpus
@requires_nccl
def test_tp2_column_parallel_absorption_matches_local_shard(tmp_path):
    mp.spawn(_tp2_worker, args=(str(tmp_path / 'lora-absorbed-tp2'), ), nprocs=TP, join=True)
