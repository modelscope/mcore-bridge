# Copyright (c) ModelScope Contributors. All rights reserved.
"""The QSA sparse kernel stays off unless CUDA can compile it."""
import ast
from pathlib import Path


def _load(have_triton, cuda):
    path = Path(__file__).resolve().parents[1] / 'src/mcore_bridge/model/modules/kernels/qsa_kernels.py'
    tree = ast.parse(path.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'qsa_sparse_supported')
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    warnings = []

    class _Cuda:

        @staticmethod
        def is_available():
            return cuda

    class _Logger:

        @staticmethod
        def warning_once(message):
            warnings.append(message)

    namespace = {
        'HAVE_TRITON': have_triton,
        'QSA_SPARSE_KERNEL_ENV': 'QSA_SPARSE_KERNEL',
        'logger': _Logger(),
        'torch': type('Torch', (), {'cuda': _Cuda})(),
        'use_qsa_sparse_kernel': lambda: True,
    }
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['qsa_sparse_supported'], warnings


def test_non_cuda_does_not_enable_the_kernel():
    supported, warnings = _load(have_triton=True, cuda=False)
    assert supported(64) is False
    assert warnings and 'CUDA-only' in warnings[0]


def test_cuda_power_of_two_head_stays_enabled():
    supported, warnings = _load(have_triton=True, cuda=True)
    assert supported(128) is True
    assert warnings == []


def test_non_power_of_two_head_stays_disabled():
    supported, _ = _load(have_triton=True, cuda=True)
    assert supported(96) is False
