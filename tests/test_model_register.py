import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mcore_bridge.model.register import ModelLoader


class TestModelLoader(unittest.TestCase):

    @staticmethod
    def _get_loader(transformer_impl):
        loader = ModelLoader.__new__(ModelLoader)
        loader.config = SimpleNamespace(
            transformer_impl=transformer_impl,
            experimental_attention_variant=None,
            normalization='RMSNorm',
            qk_l2_norm=False,
        )
        return loader

    def test_transformer_impl_controls_decoder_layer_spec(self):
        for transformer_impl, expected in [('local', False), ('transformer_engine', True)]:
            with self.subTest(transformer_impl=transformer_impl):
                loader = self._get_loader(transformer_impl)
                with patch('mcore_bridge.model.register.get_gpt_decoder_block_spec') as get_spec:
                    get_spec.return_value = SimpleNamespace(layer_specs=[])
                    loader.get_transformer_layer_spec()
                self.assertEqual(get_spec.call_args.kwargs['use_transformer_engine'], expected)

    def test_transformer_impl_controls_mtp_layer_spec(self):
        for transformer_impl, expected in [('local', False), ('transformer_engine', True)]:
            with self.subTest(transformer_impl=transformer_impl):
                loader = self._get_loader(transformer_impl)
                with patch('mcore_bridge.model.register.get_gpt_mtp_block_spec') as get_spec:
                    get_spec.return_value = None
                    loader.get_mtp_block_spec(SimpleNamespace())
                self.assertEqual(get_spec.call_args.kwargs['use_transformer_engine'], expected)


if __name__ == '__main__':
    unittest.main()
