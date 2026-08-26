# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple


class Qwen4ExpTextGroupedRMSNorm(nn.Module):

    def __init__(self, dim: int, group_size: int, eps: float = 1e-6, dtype=None, sequence_parallel: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        self.group_size = group_size
        if dim % group_size != 0:
            raise ValueError(f'hidden_size ({dim}) must be divisible by group_size ({group_size}).')
        # mcore-specific: params_dtype weight + SP grad flag (HF builds fp32).
        if dtype is not None:
            self.weight.data = self.weight.data.to(dtype)
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
        self.input_mix_weight_down = nn.Linear(hc_hidden_size, config.hc_lowrank, bias=False, dtype=config.params_dtype)
        self.input_mix_weight_up = nn.Linear(config.hc_lowrank, hc_hidden_size, bias=False, dtype=config.params_dtype)
        self.block_inject_weight = nn.Linear(
            hc_hidden_size, self.hc_count, bias=False, dtype=config.params_dtype) if use_combine else None
        # mcore-specific: SP grad flag on the replicated weights.
        for param in self.parameters():
            setattr(param, 'sequence_parallel', config.sequence_parallel)

    def forward(self, hyper_input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Mirrors transformers `Qwen4ExpTextGatedResidual.forward` line by line.
        if hyper_input.shape[-1] != self.hc_count * self.hidden_size:
            raise ValueError(f'Expected {self.hc_count * self.hidden_size} hyper-connection features, '
                             f'got {hyper_input.shape[-1]}.')
        hyper_input_normed = self.hc_norm(hyper_input)
        input_mix_weight = F.silu(self.input_mix_weight_down(hyper_input_normed) / self.hc_count)
        input_mix_weight = torch.sigmoid(self.input_mix_weight_up(input_mix_weight))
        input_mix_weight = input_mix_weight.unflatten(-1, (self.hc_count, self.hidden_size))
        mixed_input = (input_mix_weight * hyper_input_normed.unflatten(-1,
                                                                       (self.hc_count, self.hidden_size))).mean(dim=-2)
        if self.block_inject_weight is None:
            return mixed_input
        injection_weights = 2 * torch.sigmoid(self.block_inject_weight(hyper_input_normed) / self.hc_count)
        return mixed_input, hyper_input, injection_weights
