# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import TELinear
from torch import nn
from typing import Tuple


def _duplicated_linear(config, input_size: int, output_size: int) -> TELinear:
    """Replicated projection: TELinear so LoRA can wrap it.

    ``dispatch_megatron`` only wraps TE classes into ``LoraParallelLinear``, which the
    bridge's PEFT export keys off. ``parallel_mode='duplicated'`` because
    ``block_inject_weight``'s output is hc_count (4) and cannot be split across TP=8.
    """
    return TELinear(
        input_size=input_size,
        output_size=output_size,
        parallel_mode='duplicated',
        config=config,
        init_method=config.init_method,
        bias=False,
        skip_bias_add=True,
        skip_weight_param_allocation=False,
    )


class Qwen4ExpTextGroupedRMSNorm(nn.Module):

    def __init__(self, dim: int, group_size: int, eps: float = 1e-6, dtype=None, sequence_parallel: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim, dtype=dtype))
        self.group_size = group_size
        if dim % group_size != 0:
            raise ValueError(f'hidden_size ({dim}) must be divisible by group_size ({group_size}).')
        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(*x.shape[:-1], -1, self.group_size)
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.flatten(-2)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f'{tuple(self.weight.shape)}, eps={self.eps}'


@torch.compile
def _mix_elementwise(down_out: torch.Tensor, hc_count: int) -> torch.Tensor:
    """silu(down/hc) -- the pre-up-projection half of the gate chain."""
    return F.silu(down_out / hc_count)


@torch.compile
def _mix_and_reduce(up_out: torch.Tensor, hyper_input_normed: torch.Tensor, hc_count: int,
                    hidden_size: int) -> torch.Tensor:
    """sigmoid -> unflatten -> multiply -> mean, the post-up-projection half."""
    w = torch.sigmoid(up_out).unflatten(-1, (hc_count, hidden_size))
    return (w * hyper_input_normed.unflatten(-1, (hc_count, hidden_size))).mean(dim=-2)


class Qwen4ExpTextGatedResidual(nn.Module):

    def __init__(self, config, use_combine: bool = True):
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        hc_hidden_size = self.hc_count * self.hidden_size
        # (transformers: Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size,
        #  eps=config.rms_norm_eps); layernorm_epsilon is mcore's rms_norm_eps.)
        self.hc_norm = Qwen4ExpTextGroupedRMSNorm(
            hc_hidden_size, group_size=self.hidden_size, eps=config.layernorm_epsilon, dtype=config.params_dtype)
        self.input_mix_weight_down = _duplicated_linear(config, hc_hidden_size, config.hc_lowrank)
        self.input_mix_weight_up = _duplicated_linear(config, config.hc_lowrank, hc_hidden_size)
        self.block_inject_weight = _duplicated_linear(config, hc_hidden_size, self.hc_count) if use_combine else None
        # mcore-specific: SP grad flag on the replicated weights.
        for param in self.parameters():
            setattr(param, 'sequence_parallel', config.sequence_parallel)

    def forward(self, hyper_input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Mirrors transformers `Qwen4ExpTextGatedResidual.forward` line by line.
        if hyper_input.shape[-1] != self.hc_count * self.hidden_size:
            raise ValueError(f'Expected {self.hc_count * self.hidden_size} hyper-connection features, '
                             f'got {hyper_input.shape[-1]}.')
        hyper_input_normed = self.hc_norm(hyper_input)
        # TELinear returns (output, bias); bias is None here (bias=False). The two
        # linears stay outside the compiled helpers: TE modules do work inside a
        # compiled region (verified), but keeping them out leaves TE's own fused
        # kernels and FP8 bookkeeping untouched and limits the graph to the
        # elementwise ops that actually benefit.
        input_mix_weight = _mix_elementwise(self.input_mix_weight_down(hyper_input_normed)[0], self.hc_count)
        input_mix_weight = self.input_mix_weight_up(input_mix_weight)[0]
        mixed_input = _mix_and_reduce(input_mix_weight, hyper_input_normed, self.hc_count, self.hidden_size)
        if self.block_inject_weight is None:
            return mixed_input
        injection_weights = 2 * torch.sigmoid(self.block_inject_weight(hyper_input_normed)[0] / self.hc_count)
        return mixed_input, hyper_input, injection_weights
