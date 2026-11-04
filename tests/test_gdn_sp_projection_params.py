# Copyright (c) ModelScope Contributors. All rights reserved.
"""SP+TP: the GDN sequence-parallel toggle has to reach the projection parameters.

``GatedDeltaNet.forward`` gathers the sequence before the projections and calls
``_force_sequence_parallel_off`` so the projections run without sequence
parallelism. Their gradient is then complete on every tensor parallel rank, so
Megatron's finalizer (``finalize_model_grads``) must not sum it over the TP group.
The finalizer decides from ``param.sequence_parallel``, and Transformer Engine
marks the parameters it wants reduced with that same flag (for example
``layer_norm_weight`` of a ``LayerNormLinear``). With the flag left on, every rank
holds the complete gradient and still has it summed, which multiplies it by the
tensor parallel size -- the reported #241 case was a 4x gamma gradient at TP4.

The test drives the real ``_force_sequence_parallel_off`` on a Transformer Engine
shaped stand-in and reads the reduction back off ``main_grad`` after Megatron's own
``_allreduce_non_tensor_model_parallel_grads``. A negative control restores the
flag and must reproduce the amplification, so this fails if the parameter half of
the toggle goes missing.

Runs on ``gloo`` (CPU), so it needs no GPU.
"""
import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import types

TP = 2
PROJECTIONS = ('in_proj', 'in_proj_qkvz', 'in_proj_ba', 'out_proj')


class _FakeLayerNormLinear(nn.Module):
    """Stand-in for a Transformer Engine ``LayerNormLinear``.

    Transformer Engine flags the module and the parameters it wants reduced over
    the tensor parallel group with ``sequence_parallel``; ``layer_norm_weight`` is
    the one the reported case loses in the finalizer. ``weight`` carries no flag,
    so it is the control for "parameters without the flag are left alone".
    """

    def __init__(self, mark_parameters: bool = True):
        super().__init__()
        self.sequence_parallel = True
        self.layer_norm_weight = nn.Parameter(torch.ones(4))
        self.weight = nn.Parameter(torch.ones(2, 4))
        if mark_parameters:
            self.layer_norm_weight.sequence_parallel = True


def _module():
    """Return ``gated_delta_net`` or skip when Megatron/Transformer Engine is absent."""
    pytest.importorskip('megatron.core')
    try:
        from mcore_bridge.model.modules import gated_delta_net
    except (ImportError, AttributeError) as exc:
        # mcore_bridge applies its Megatron/TE patches at import time, so an
        # environment without a full Transformer Engine cannot load the module.
        pytest.skip(f'mcore_bridge is not importable here: {exc!r}')
    return gated_delta_net


def _root(**modules):
    """Only the projection attributes the toggle walks are needed."""
    attributes = dict.fromkeys(PROJECTIONS)
    attributes.update(modules)
    return types.SimpleNamespace(**attributes)


class _FinalizeChunk(nn.Module):
    """Chunk with the DDP surface ``_allreduce_non_tensor_model_parallel_grads`` touches."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.ddp_config = types.SimpleNamespace(use_megatron_fsdp=False)

    def finish_grad_sync(self, force_all_reduce=False):
        pass


def test_toggle_marks_every_projection_and_their_parameters():
    gated_delta_net = _module()
    layers = {name: _FakeLayerNormLinear() for name in PROJECTIONS}
    root = _root(**layers)

    gated_delta_net._force_sequence_parallel_off(root)

    for name, layer in layers.items():
        assert layer.sequence_parallel is False, f'{name} module flag'
        assert layer.layer_norm_weight.sequence_parallel is False, f'{name} parameter flag'
        # The unmarked weight must be left alone: adding the attribute would make
        # the finalizer start reducing a parameter that never needed it.
        assert not hasattr(layer.weight, 'sequence_parallel'), f'{name} untouched parameter'


def test_toggle_ignores_missing_projections():
    """A ``linear_decoupled_in_proj`` build replaces ``in_proj`` with ``IdentityOp``."""
    gated_delta_net = _module()
    layer = _FakeLayerNormLinear()
    root = _root(in_proj_qkvz=layer, in_proj_ba=layer)

    gated_delta_net._force_sequence_parallel_off(root)

    assert layer.sequence_parallel is False
    assert layer.layer_norm_weight.sequence_parallel is False


def test_out_norm_keeps_its_flag():
    """``out_norm`` is deliberately excluded: its weight is shared by all heads.

    The head axis is TP-split, so that gradient really is partial and still needs
    the sum. Forcing the flag off on it would drop a reduction that is required.
    """
    gated_delta_net = _module()
    out_norm = _FakeLayerNormLinear()
    root = _root(in_proj=_FakeLayerNormLinear(), out_norm=out_norm)

    gated_delta_net._force_sequence_parallel_off(root)

    assert out_norm.sequence_parallel is True
    assert out_norm.layer_norm_weight.sequence_parallel is True


def _worker(rank, world_size, init_file, flagged, queue):
    """Run Megatron's own reduction on one gloo rank (CPU)."""
    dist.init_process_group('gloo', init_method='file://' + init_file, rank=rank, world_size=world_size)
    from megatron.core import parallel_state
    from megatron.core.distributed.finalize_model_grads import _allreduce_non_tensor_model_parallel_grads
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)
    try:
        config = types.SimpleNamespace(sequence_parallel=True, qk_layernorm=False)
        layer = _FakeLayerNormLinear()
        # What the finalizer sees after the backward: the projection ran without
        # sequence parallelism, so each rank already holds the complete gradient.
        # ``flagged`` is the state the toggle leaves on the parameter -- False with
        # the fix, True before it (the negative control).
        layer.layer_norm_weight.sequence_parallel = flagged
        layer.layer_norm_weight.main_grad = torch.ones(4)
        layer.weight.main_grad = torch.ones(2, 4)
        _allreduce_non_tensor_model_parallel_grads([_FinalizeChunk(layer)], config)
        queue.put((rank, layer.layer_norm_weight.main_grad.tolist(), layer.weight.main_grad.tolist()))
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize(
    'flagged,expected',
    [(False, 1.0), (True, float(TP))],
    ids=['flag-off-what-the-toggle-leaves', 'flag-on-negative-control'],
)
def test_finalizer_sums_the_gamma_only_when_the_parameter_is_flagged(flagged, expected, tmp_path):
    """Megatron's own reduction, on the two parameter states the toggle can leave behind.

    Megatron's rule is not re-implemented here: the Transformer Engine shaped
    stand-in goes through the real ``_allreduce_non_tensor_model_parallel_grads`` on
    two gloo ranks and the reduction is read off ``main_grad``. With the flag off
    (what ``_force_sequence_parallel_off`` leaves), the complete gradient stays at
    1x, which is what #241 asks for. The control case is the pre-fix state and has
    to come out at ``tp_size`` times, so this fails if the parameter half of the
    toggle goes missing.
    """
    _module()  # skip cleanly when megatron.core is unavailable
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    mp.spawn(_worker, args=(TP, os.path.join(str(tmp_path), 'init'), flagged, queue), nprocs=TP, join=True)
    results = [queue.get(timeout=30) for _ in range(TP)]
    for rank, gamma, weight in results:
        assert gamma == [expected] * 4, f'rank {rank} (flagged={flagged}) gamma {gamma}'
        # ``weight`` carries no flag, so the finalizer never reduces it.
        assert weight == [[1.0] * 4] * 2, f'rank {rank} (flagged={flagged}) weight {weight}'
