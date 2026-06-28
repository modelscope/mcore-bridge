import transformer_engine
from megatron.core.transformer.spec_utils import build_module

try:
    from megatron.core.transformer.experimental_attention_variant.csa import Compressor as McoreCompressor
    from megatron.core.transformer.experimental_attention_variant.csa import CSAIndexer as McoreCSAIndexer
except ImportError:
    McoreCompressor = object
    McoreCSAIndexer = object


class Compressor(McoreCompressor):

    def __init__(self, config, submodules, *args, **kwargs):
        super().__init__(config, submodules, *args, **kwargs)
        if getattr(config, 'fp8_param', False):
            with transformer_engine.pytorch.fp8_model_init(enabled=False):
                self.linear_wkv = build_module(
                    submodules.linear_wkv,
                    config.hidden_size,
                    self.coff * self.head_dim,
                    config=config,
                    init_method=config.init_method,
                    bias=False,
                    skip_bias_add=False,
                    skip_weight_param_allocation=False,
                    parallel_mode='duplicated',
                )
                self.linear_wgate = build_module(
                    submodules.linear_wgate,
                    config.hidden_size,
                    self.coff * self.head_dim,
                    config=config,
                    init_method=config.init_method,
                    bias=False,
                    skip_bias_add=False,
                    skip_weight_param_allocation=False,
                    parallel_mode='duplicated',
                )


class CSAIndexer(McoreCSAIndexer):

    def __init__(self, config, submodules, *args, **kwargs):
        super().__init__(config, submodules, *args, **kwargs)
        if getattr(config, 'fp8_param', False):
            with transformer_engine.pytorch.fp8_model_init(enabled=False):
                self.linear_weights_proj = build_module(
                    submodules.linear_weights_proj,
                    self.hidden_size,
                    self.index_n_heads,
                    config=config,
                    init_method=config.init_method,
                    bias=False,
                    skip_bias_add=False,
                    skip_weight_param_allocation=False,
                    parallel_mode='duplicated',
                )
