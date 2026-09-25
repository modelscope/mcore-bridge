# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for the memory-bounded linear cross-entropy implementation."""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F


MODULE_PATH = Path(__file__).parents[1] / 'src/mcore_bridge/model/chunked_linear_ce.py'


def _load_module():
    spec = importlib.util.spec_from_file_location('chunked_linear_ce_under_test', MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load {MODULE_PATH}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_chunked(function, hidden, weight, labels, chunk_size, tail_pad_multiple, grad_scale):
    hidden = hidden.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    loss = function.apply(
        hidden,
        weight,
        labels,
        None,
        0,
        chunk_size,
        False,
        tail_pad_multiple,
    )
    (loss * grad_scale).sum().backward()
    return loss.detach(), hidden.grad.detach(), weight.grad.detach()


def _run_native(hidden, weight, labels, grad_scale):
    hidden = hidden.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    seq_len, batch_size, hidden_size = hidden.shape
    logits = hidden.reshape(seq_len * batch_size, hidden_size) @ weight.t()
    targets = labels.transpose(0, 1).contiguous().reshape(-1)
    loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=-100)
    loss = loss.reshape(seq_len, batch_size).transpose(0, 1).contiguous()
    (loss * grad_scale).sum().backward()
    return loss.detach(), hidden.grad.detach(), weight.grad.detach()


class ChunkedLinearCrossEntropyTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_config_and_environment_precedence(self):
        config = SimpleNamespace(chunked_linear_ce_chunk_size=2048)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.module.parse_chunked_linear_ce_chunk_size(config), 2048)

        with patch.dict(os.environ, {'LINEAR_CE_CHUNK_SIZE': '2k'}, clear=True):
            self.assertEqual(self.module.parse_chunked_linear_ce_chunk_size(config), 2048)

        with patch.dict(
                os.environ,
                {'CHUNKED_LINEAR_CE_CHUNK_SIZE': '4k', 'LINEAR_CE_CHUNK_SIZE': '2k'},
                clear=True):
            self.assertEqual(self.module.parse_chunked_linear_ce_chunk_size(config), 4096)

        with patch.dict(os.environ, {'CHUNKED_LINEAR_CE_CHUNK_SIZE': '0'}, clear=True):
            self.assertEqual(self.module.parse_chunked_linear_ce_chunk_size(config), 0)

        with patch.dict(os.environ, {'CHUNKED_LINEAR_CE_CHUNK_SIZE': '-1'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'must be >= 0'):
                self.module.parse_chunked_linear_ce_chunk_size(config)

    def test_forward_backward_matches_native_ce(self):
        torch.manual_seed(20260831)
        seq_len, batch_size, hidden_size, vocab_size = 11, 2, 7, 19
        hidden = torch.randn(seq_len, batch_size, hidden_size, dtype=torch.float32) * 0.1
        weight = torch.randn(vocab_size, hidden_size, dtype=torch.float32) * 0.1
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
        labels[0, :3] = -100
        labels[1, 5:8] = -100
        grad_scale = torch.randn(batch_size, seq_len, dtype=torch.float32)

        native = _run_native(hidden, weight, labels, grad_scale)
        chunked = _run_chunked(
            self.module.ChunkedLinearCrossEntropyLoss,
            hidden,
            weight,
            labels,
            chunk_size=5,
            tail_pad_multiple=4,
            grad_scale=grad_scale,
        )

        for native_value, chunked_value in zip(native, chunked):
            self.assertTrue(torch.allclose(native_value, chunked_value, atol=2e-5, rtol=2e-5))

        ignored = labels == -100
        self.assertEqual(float(chunked[0][ignored].abs().sum()), 0.0)
        self.assertEqual(float(chunked[1][ignored.transpose(0, 1)].abs().sum()), 0.0)

    def test_legacy_class_alias_is_preserved(self):
        self.assertIs(
            self.module.ChunkedLinearCrossEntropy,
            self.module.ChunkedLinearCrossEntropyLoss,
        )

    def test_all_ignored_labels_keep_backward_valid(self):
        torch.manual_seed(20260831)
        hidden = torch.randn(5, 1, 4, dtype=torch.float32)
        weight = torch.randn(9, 4, dtype=torch.float32)
        labels = torch.full((1, 5), -100, dtype=torch.long)
        loss, hidden_grad, weight_grad = _run_chunked(
            self.module.ChunkedLinearCrossEntropyLoss,
            hidden,
            weight,
            labels,
            chunk_size=2,
            tail_pad_multiple=0,
            grad_scale=torch.ones_like(labels, dtype=torch.float32),
        )

        self.assertTrue(torch.equal(loss, torch.zeros_like(loss)))
        self.assertTrue(torch.equal(hidden_grad, torch.zeros_like(hidden_grad)))
        self.assertTrue(torch.equal(weight_grad, torch.zeros_like(weight_grad)))


if __name__ == '__main__':
    unittest.main()
