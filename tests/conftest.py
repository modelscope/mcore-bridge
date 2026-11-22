# Copyright (c) ModelScope Contributors. All rights reserved.
"""Test tiering, mirroring ms-swift and twinkle: unmarked runs on CPU, ``accel(n)`` needs n
accelerators (skip when fewer), ``slow`` is daily-only (``pytest -m "not slow"`` on a PR).

The GPU contract of each file is declared here once instead of per case: the numerical-contract
suites spawn their own ranks inside one visible device set, so ``accel(1)`` means "needs a GPU",
and the two-device CP suite asks for two. Files carrying their own ``skipif`` guards keep them;
the markers only add the tier and the capability gate.
"""
import ctypes
import importlib.util
import os

import pytest


def _preload_pip_cudnn():
    """Pin cuDNN symbol resolution to the pip wheel before Transformer Engine loads.

    On boxes carrying BOTH a system cuDNN (/usr/lib, e.g. 9.14) and torch's pip nvidia-cudnn
    (e.g. 9.19), TE's DT_NEEDED libcudnn.so.9 maps the system main library alongside the pip
    one (same SONAME, two objects). cuDNN then lazily dlopens its sublibraries across the two
    installations and a 9.14 main + 9.19 sublibrary mix dies with
    CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED inside fused attention -- but only once some earlier
    test has imported the cudnn frontend, so it surfaces as order-dependent suite pollution.
    Loading the pip main + every sublibrary RTLD_GLOBAL first makes all later resolutions reuse
    one consistent family. No-op where the pip wheel is absent.
    """
    spec = importlib.util.find_spec('nvidia.cudnn')
    if spec is None or not spec.submodule_search_locations:
        return
    libdir = os.path.join(spec.submodule_search_locations[0], 'lib')
    main = os.path.join(libdir, 'libcudnn.so.9')
    if not os.path.exists(main):
        return
    ctypes.CDLL(main, mode=ctypes.RTLD_GLOBAL)
    for name in sorted(os.listdir(libdir)):
        if name.startswith('libcudnn_') and name.endswith('.so.9'):
            ctypes.CDLL(os.path.join(libdir, name), mode=ctypes.RTLD_GLOBAL)


_preload_pip_cudnn()

#: file stem -> accelerators needed; every other test file stays CPU-tier.
ACCEL_FILES = {
    'test_deepseek_v41_engram': 1,
    'test_deepseek_v41_packing_cp': 2,
    'test_gdn_sp_projection_params': 1,
    'test_glm5_hybrid': 1,
    'test_glm5_next': 1,
    'test_hybrid_compat': 1,
    'test_lora_replicated_base': 1,
    'test_llm': 1,
    'test_mllm': 1,
    'test_mtp_checkpoint': 1,
    'test_mtp_runtime': 1,
    'test_nemo_speechlm': 1,
    'test_nemotron_omni': 1,
    'test_ple_cp_grad': 1,
    'test_ple_fp8_scale': 1,
    'test_ple_kernels': 1,
    'test_ple_sp_tp_grad': 1,
    'test_qsa_indexer': 1,
    'test_qsa_packed_documents': 1,
    'test_qsa_sparse_kernels': 1,
    'test_qwen4_exp_mtp': 1,
    'test_qwen4_exp_units': 1,
    # legacy swift export path: MegatronExportArguments initializes an nccl process group.
    'test_save_missing_weights': 1,
}

#: Legacy swift export conversion-precision e2e: real model weights, minutes per model, warm-cache
#: only (each _test_model skips a cold miss). Daily tier, never the PR gate.
SLOW_FILES = {'test_llm', 'test_mllm'}


def _accelerator_count() -> int:
    try:
        import torch
    except ImportError:
        return 0
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    npu = getattr(torch, 'npu', None)
    return npu.device_count() if npu is not None and npu.is_available() else 0


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: daily tier, may run long')
    config.addinivalue_line('markers', 'accel(N): needs at least N accelerators (skip when fewer)')


def pytest_collection_modifyitems(config, items):
    has_megatron = importlib.util.find_spec('megatron.core') is not None
    available = _accelerator_count()
    for item in items:
        if not has_megatron:
            item.add_marker(pytest.mark.skip(reason='megatron-core not installed'))
            continue
        needed = ACCEL_FILES.get(item.path.stem)
        marker = item.get_closest_marker('accel')
        if marker is not None:
            needed = max(needed or 0, marker.args[0] if marker.args else 1)
        if needed is not None:
            item.add_marker(pytest.mark.accel(needed))
            if available < needed:
                item.add_marker(pytest.mark.skip(reason=f'needs {needed} accelerator(s), found {available}'))
        if item.path.stem in SLOW_FILES:
            item.add_marker(pytest.mark.slow)
