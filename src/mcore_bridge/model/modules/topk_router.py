import torch
from megatron.core.jit import jit_fuser
from megatron.core.transformer.moe.router import TopKRouter as McoreTopKRouter
from typing import Optional


class TopKRouter(McoreTopKRouter):

    @jit_fuser
    def _apply_expert_bias(self, routing_map: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        """
        Update expert bias and tokens_per_expert
        Prevent extra local tokens accumulation on evaluation or activation recomputation
        """
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                if padding_mask is not None:
                    if padding_mask.ndim == 1:
                        padding_mask = padding_mask.unsqueeze(-1)
                    routing_map = routing_map & (~padding_mask)
                self.local_tokens_per_expert += routing_map.sum(dim=0)
