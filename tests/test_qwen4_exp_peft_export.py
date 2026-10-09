# Copyright (c) ModelScope Contributors. All rights reserved.
"""Exercise the PLE conversion method without importing CUDA/Megatron modules.

The method and buffer names are compiled from the production source. Tensor
operations and PP collectives are mocked; this tests conversion control flow,
not distributed checkpoint or vLLM integration.
"""
import ast
import unittest
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

SOURCE = Path(__file__).resolve().parents[1] / 'src/mcore_bridge/model/gpts/qwen4_exp.py'


def _load_ple_bridge():
    tree = ast.parse(SOURCE.read_text())
    bridge = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Qwen4ExpBridge')
    members = [
        node for node in bridge.body if (isinstance(node, ast.FunctionDef) and node.name == '_set_layer_ple') or (
            isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == '_PLE_NGRAM_BUFFERS' for target in node.targets))
    ]
    bridge.bases = []
    bridge.body = members
    module = ast.Module(body=[bridge], type_ignores=[])
    namespace = {'dist': Mock()}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), namespace)
    return namespace['Qwen4ExpBridge'], namespace['dist']


class TestPLEPEFTConversion(unittest.TestCase):

    def setUp(self):
        bridge_class, self.dist = _load_ple_bridge()
        self.bridge = bridge_class()
        self.bridge._get_pp_src_rank = Mock(return_value=0)
        self.bridge._reduce_tensor_pp_group = Mock(side_effect=lambda value, to_mcore: value)
        self.bridge._set_state_dict = Mock()
        self.bridge._iter_ple_table_export = Mock(return_value=sentinel.table_iterator)
        self.bridge._pending_export_iter = sentinel.stale_iterator
        self.bridge._target_device = None
        self.bridge.pp_group = sentinel.pp_group
        self.embedding = SimpleNamespace(cpu_offload=False, fill_table_from_hf=Mock())
        for name in self.bridge._PLE_NGRAM_BUFFERS:
            buffer = Mock()
            buffer.data.clone.return_value = sentinel.buffer_value
            setattr(self.embedding, name, buffer)
        self.layer = SimpleNamespace(ple=SimpleNamespace(ple_embedding=self.embedding))

    def _export(self, peft, saving, offloaded, pp_size):
        self.bridge._peft_format = peft
        self.bridge._is_saving = saving
        self.bridge.pp_size = pp_size
        self.embedding.cpu_offload = offloaded
        state = {'lora_A.weight': sentinel.lora_value}
        self.bridge._set_layer_ple(self.layer, state, False, layer_prefix='model.layers.1.')
        self.assertIs(state['lora_A.weight'], sentinel.lora_value)
        # PLE projections still go through the usual PEFT-aware converter.
        self.assertEqual(self.bridge._set_state_dict.call_count, 6)
        self.assertFalse(self.bridge._converting_ple)
        return state

    def _assert_exported_ngram(self, state, expected):
        keys = {f'ple.ple_embedding.{name}' for name in self.bridge._PLE_NGRAM_BUFFERS}
        self.assertEqual(set(state) - {'lora_A.weight'}, keys if expected else set())
        if expected:
            self.bridge._iter_ple_table_export.assert_called_once_with(self.layer.ple, 0, 'model.layers.1.')
            self.assertIs(self.bridge._pending_export_iter, sentinel.table_iterator)
            for key in keys:
                self.assertIs(state[key], sentinel.buffer_value)
        else:
            self.bridge._iter_ple_table_export.assert_not_called()
            self.assertIsNone(self.bridge._pending_export_iter)
            for name in self.bridge._PLE_NGRAM_BUFFERS:
                getattr(self.embedding, name).data.clone.assert_not_called()

    def test_adapter_export_skips_ngram_even_when_saving(self):
        for saving, offloaded, pp_size in product((False, True), (False, True), (1, 2)):
            with self.subTest(saving=saving, offloaded=offloaded, pp_size=pp_size):
                self.setUp()
                state = self._export(True, saving, offloaded, pp_size)
                self._assert_exported_ngram(state, False)
                self.dist.broadcast_object_list.assert_not_called()

    def test_full_model_save_includes_ngram(self):
        for offloaded, pp_size in product((False, True), (1, 2)):
            with self.subTest(offloaded=offloaded, pp_size=pp_size):
                self.setUp()
                state = self._export(False, True, offloaded, pp_size)
                self._assert_exported_ngram(state, True)

    def test_online_full_export_skips_only_offloaded_ngram(self):
        for offloaded, pp_size in product((False, True), (1, 2)):
            with self.subTest(offloaded=offloaded, pp_size=pp_size):
                self.setUp()
                state = self._export(False, False, offloaded, pp_size)
                self._assert_exported_ngram(state, not offloaded)

    def test_adapter_save_on_pp_stage_without_ple(self):
        for offloaded in (False, True):
            with self.subTest(offloaded=offloaded):
                self.setUp()
                self.bridge._peft_format = True
                self.bridge._is_saving = True
                self.bridge.pp_size = 2
                self.bridge._reduce_tensor_pp_group.side_effect = [True, offloaded]
                state = {}
                self.bridge._set_layer_ple(None, state, False)
                self.assertEqual(state, {})
                self.bridge._iter_ple_table_export.assert_not_called()
                self.dist.broadcast_object_list.assert_not_called()
                self.assertIsNone(self.bridge._pending_export_iter)
                self.assertEqual(self.bridge._set_state_dict.call_count, 6)

    def test_adapter_import_skips_ngram_but_full_import_restores_it(self):
        for peft, offloaded in product((False, True), (False, True)):
            with self.subTest(peft=peft, offloaded=offloaded):
                self.setUp()
                self.bridge._peft_format = peft
                self.bridge._is_saving = False
                self.bridge.pp_size = 2
                self.embedding.cpu_offload = offloaded
                state = {}
                if not peft:
                    for name in self.bridge._PLE_NGRAM_BUFFERS:
                        value = Mock()
                        value.load.return_value.to.return_value = sentinel.loaded_value
                        state[f'ple.ple_embedding.{name}'] = value
                self.bridge._set_layer_ple(self.layer, state, True)
                if peft:
                    self.embedding.fill_table_from_hf.assert_not_called()
                    for name in self.bridge._PLE_NGRAM_BUFFERS:
                        getattr(self.embedding, name).copy_.assert_not_called()
                else:
                    self.embedding.fill_table_from_hf.assert_called_once_with(state)
                    for name in self.bridge._PLE_NGRAM_BUFFERS:
                        getattr(self.embedding, name).copy_.assert_called_once_with(sentinel.loaded_value)
                self.bridge._get_pp_src_rank.assert_not_called()
                self.dist.broadcast_object_list.assert_not_called()
                self.assertIsNone(self.bridge._pending_export_iter)


if __name__ == '__main__':
    unittest.main()
