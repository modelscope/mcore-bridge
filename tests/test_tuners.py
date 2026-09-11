import unittest
from unittest.mock import patch

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

from mcore_bridge.tuners.patcher import dispatch_megatron


class TestMegatronLoraDispatch(unittest.TestCase):

    def test_dispatches_local_parallel_linears(self):
        for linear_cls in (ColumnParallelLinear, RowParallelLinear):
            with self.subTest(linear_cls=linear_cls.__name__):
                target = linear_cls.__new__(linear_cls)
                with patch('mcore_bridge.tuners.patcher.LoraParallelLinear', return_value='adapter') as lora_cls:
                    result = dispatch_megatron(target, 'default')
                self.assertEqual(result, 'adapter')
                self.assertIs(lora_cls.call_args.kwargs['base_layer'], target)


if __name__ == '__main__':
    unittest.main()
