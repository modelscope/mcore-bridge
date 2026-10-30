# Copyright (c) ModelScope Contributors. All rights reserved.
"""The GDN sequence-parallel toggle has to reach the projection parameters.

``GatedDeltaNet`` gathers the sequence before the projections and turns sequence parallelism off on them for
that call, because Transformer Engine then all-reduces the projection gradient itself. Megatron's gradient
finalizer decides from ``param.sequence_parallel`` (see ``finalize_model_grads``) whether a gradient still
has to be summed over the tensor parallel group, so the flag has to be off on the parameters as well: with
only the module attribute off, the finalizer sums the already complete gamma gradient once more and
multiplies it by the tensor parallel size. Parameters that do not carry the flag are left alone.
"""
import os
import pytest
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import types

TP = 2


class _FakeLayerNormLinear(nn.Module):
    """Stand-in for a Transformer Engine ``LayerNormLinear``.

    Transformer Engine flags the module and the parameters it wants reduced over the tensor parallel group
    with ``sequence_parallel``; ``layer_norm_weight`` is the one the reported case loses in the finalizer.
    """

    def __init__(self, sequence_parallel: bool = True, mark_parameters: bool = True):
        super().__init__()
        self.sequence_parallel = sequence_parallel
        self.layer_norm_weight = nn.Parameter(torch.ones(4))
        self.weight = nn.Parameter(torch.ones(2, 4))
        if mark_parameters:
            self.layer_norm_weight.sequence_parallel = sequence_parallel


def _gdn():
    """Return ``GatedDeltaNet``, skipping when Megatron (or the Transformer Engine it imports) is absent."""
    pytest.importorskip('megatron.core')
    try:
        from mcore_bridge.model.modules.gated_delta_net import GatedDeltaNet
    except (ImportError, AttributeError) as exc:
        # mcore_bridge applies its Megatron/Transformer Engine patches at import time, so an environment
        # without a full Transformer Engine cannot load the module; skip rather than fail.
        pytest.skip(f'mcore_bridge is not importable here: {exc!r}')
    return GatedDeltaNet


def _finalize_helper():
    """Import Megatron's own ``_allreduce_non_tensor_model_parallel_grads`` (the finalizer's TP reduction)."""
    try:
        from megatron.core import parallel_state
        from megatron.core.distributed.finalize_model_grads import _allreduce_non_tensor_model_parallel_grads
    except (ImportError, AttributeError) as exc:
        pytest.skip(f'megatron.core is not importable here: {exc!r}')
    return _allreduce_non_tensor_model_parallel_grads, parallel_state


def _projections(**modules):
    """Only the four projection attributes the toggle walks are needed."""
    attributes = {'in_proj': None, 'in_proj_qkvz': None, 'in_proj_ba': None, 'out_proj': None}
    attributes.update(modules)
    return types.SimpleNamespace(**attributes)


def test_toggle_marks_the_parameters_and_the_restore_keeps_them_off():
    gated_delta_net = _gdn()
    layer = _FakeLayerNormLinear()
    model = _projections(in_proj=layer)
    saved = gated_delta_net._set_linear_sequence_parallel(model, False)

    assert layer.sequence_parallel is False
    assert layer.layer_norm_weight.sequence_parallel is False
    assert len(saved) == 2
    assert saved[0][0] is layer
    assert saved[1][0] is layer.layer_norm_weight

    gated_delta_net._restore_linear_sequence_parallel(model, saved)
    # The module goes back to its configured value, the parameter stays off: the finalizer reads it after
    # the backward, long after the module attribute was restored. This is the pair of attributes the
    # finalizer test below feeds to Megatron's own reduction.
    assert layer.sequence_parallel is True
    assert layer.layer_norm_weight.sequence_parallel is False


def test_parameters_without_the_flag_are_left_alone():
    gated_delta_net = _gdn()
    layer = _FakeLayerNormLinear(mark_parameters=False)
    model = _projections(in_proj=layer)
    saved = gated_delta_net._set_linear_sequence_parallel(model, False)

    assert not hasattr(layer.weight, 'sequence_parallel')
    assert len(saved) == 1
    assert saved[0][0] is layer

    gated_delta_net._restore_linear_sequence_parallel(model, saved)
    assert layer.sequence_parallel is True


class _FinalizeChunk(nn.Module):
    """Standalone chunk with the DDP surface ``_allreduce_non_tensor_model_parallel_grads`` touches."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.ddp_config = types.SimpleNamespace(use_megatron_fsdp=False)

    def finish_grad_sync(self, force_all_reduce=False):
        pass


def _finalize_worker(rank, world_size, init_file, flagged, queue):
    """Run Megatron's own ``_allreduce_non_tensor_model_parallel_grads`` on one gloo rank (CPU)."""
    dist.init_process_group('gloo', init_method='file://' + init_file, rank=rank, world_size=world_size)
    from megatron.core import parallel_state
    from megatron.core.distributed.finalize_model_grads import _allreduce_non_tensor_model_parallel_grads
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)
    try:
        config = types.SimpleNamespace(sequence_parallel=True, qk_layernorm=False)
        layer = _FakeLayerNormLinear()
        # What the layer looks like to the finalizer after ``forward``: the module attribute is back on and
        # the parameter carries whatever the toggle left on it -- False with this fix, True before it.
        layer.sequence_parallel = True
        layer.layer_norm_weight.sequence_parallel = flagged
        layer.layer_norm_weight.main_grad = torch.ones(4)
        layer.weight.main_grad = torch.ones(2, 4)
        _allreduce_non_tensor_model_parallel_grads([_FinalizeChunk(layer)], config)
        queue.put((rank, layer.layer_norm_weight.main_grad.tolist(), layer.weight.main_grad.tolist()))
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize(
    'flagged,expected_gamma',
    [(False, 1.0), (True, float(TP))],
    ids=['param-flag-off', 'param-flag-on-control'],
)
def test_megatrons_finalizer_sums_the_gamma_only_when_the_parameter_is_flagged(flagged, expected_gamma):
    """Megatron's own reduction, on the two parameter states the toggle can leave behind.

    Megatron's rule is not re-implemented here: the Transformer Engine shaped stand-in goes through the real
    ``_allreduce_non_tensor_model_parallel_grads`` on two gloo ranks and the reduction it performs is read off
    ``main_grad``. With the flag off -- what the fixed ``_set_linear_sequence_parallel(..., False)`` leaves on
    the parameter, see the pin above -- the gamma gradient stays at 1x, which is what the issue asks for. The
    control case is the pre-fix state and has to come out at ``tp_size`` times, so this pin fails if the
    parameter half of the toggle goes missing. The unmarked ``weight`` is never reduced either way.
    """
    _finalize_helper()  # skip cleanly when megatron.core is unavailable
    with tempfile.TemporaryDirectory() as tmp:
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()
        mp.spawn(
            _finalize_worker,
            args=(TP, os.path.join(tmp, 'init'), flagged, queue),
            nprocs=TP,
            join=True,
        )
        results = [queue.get(timeout=30) for _ in range(TP)]
    for rank, gamma, weight in results:
        assert gamma == [expected_gamma] * 4, f'rank {rank} (flagged={flagged}) gamma {gamma}'
        assert weight == [[1.0] * 4] * 2, f'rank {rank} (flagged={flagged}) weight {weight} should be untouched'
