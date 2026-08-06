import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mcore_bridge.bridge.gpt_bridge import GPTBridge
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


class TestGPTBridge(unittest.TestCase):

    @staticmethod
    def _get_bridge(transformer_impl):
        bridge = GPTBridge.__new__(GPTBridge)
        bridge.config = SimpleNamespace(transformer_impl=transformer_impl, multi_latent_attention=False)
        return bridge

    def test_transformer_impl_controls_attention_layernorm_mapping(self):
        expected_keys = {
            'local': 'input_layernorm.weight',
            'transformer_engine': 'self_attention.linear_qkv.layer_norm_weight',
        }
        layer = SimpleNamespace(self_attention=SimpleNamespace())
        for transformer_impl, expected_key in expected_keys.items():
            with self.subTest(transformer_impl=transformer_impl):
                bridge = self._get_bridge(transformer_impl)
                with patch.object(bridge, '_set_attn_state', return_value={}), \
                        patch.object(bridge, '_set_state_dict') as set_state_dict:
                    bridge._set_layer_attn(layer, {}, 0, True)
                self.assertEqual(set_state_dict.call_args.args[1], expected_key)

    def test_transformer_impl_controls_mlp_layernorm_mapping(self):
        expected_keys = {
            'local': 'pre_mlp_layernorm.weight',
            'transformer_engine': 'mlp.linear_fc1.layer_norm_weight',
        }
        layer = SimpleNamespace(mlp=SimpleNamespace())
        for transformer_impl, expected_key in expected_keys.items():
            with self.subTest(transformer_impl=transformer_impl):
                bridge = self._get_bridge(transformer_impl)
                with patch.object(bridge, '_set_mlp_state', return_value={}), \
                        patch.object(bridge, '_set_state_dict') as set_state_dict:
                    bridge._set_layer_mlp(layer, {}, 0, True)
                self.assertEqual(set_state_dict.call_args.args[1], expected_key)


if __name__ == '__main__':
    unittest.main()
