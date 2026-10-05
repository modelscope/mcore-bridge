import copy
import torch
from megatron.core import tensor_parallel
from megatron.core.jit import jit_fuser
from megatron.core.transformer.moe.router import TopKRouter as McoreTopKRouter
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
from typing import Optional

from .router_replay import MaskedRouterReplay


class TopKRouter(McoreTopKRouter):
    """mcore-bridge router extensions kept outside the vendored Megatron-LM tree."""

    def __init__(self, config, *args, **kwargs):
        enable_vl_bias = getattr(config, 'moe_router_enable_vl_bias', False)
        if enable_vl_bias:
            # The fused TE route accepts one shared correction vector, while VL routing
            # selects a correction vector per token. Keep this override local to the router.
            config = copy.copy(config)
            config.moe_router_fusion = False
        super().__init__(config, *args, **kwargs)

        # Stay compatible with a future Megatron version that grows native VL routing.
        self._mcore_has_native_vl_bias = hasattr(self, 'expert_bias_vl')
        if not self._mcore_has_native_vl_bias:
            if enable_vl_bias:
                if self.expert_bias is None:
                    raise ValueError('VL expert bias requires the standard expert bias.')
                self.register_buffer('expert_bias_vl', torch.zeros_like(self.expert_bias, dtype=torch.float32))
            else:
                self.expert_bias_vl = None

        # mcore builds a mask-unaware ``RouterReplay`` when ``moe_enable_routing_replay`` is set; swap in
        # our mask-aware subclass so selective replay (a per-token ``replay_mask``) works. Drop the orphaned
        # mcore instance from the global registry first so it stays 1:1 per layer and ``set_replay_data`` /
        # ``get_recorded_data`` keep lining up. mcore's ``routing()`` forwards ``self.router_replay`` into
        # ``get_replay_topk``, so the subclass is picked up with no further wiring.
        if getattr(self, 'router_replay', None) is not None and not isinstance(self.router_replay, MaskedRouterReplay):
            RouterReplay.global_router_replay_instances.remove(self.router_replay)
            self.router_replay = MaskedRouterReplay()

    def routing(self, logits, *args, **kwargs):
        # The base ``routing`` signature differs across Megatron releases (older versions take
        # only ``logits``), so pass the extra arguments straight through instead of pinning them.
        if self.expert_bias_vl is None or self._mcore_has_native_vl_bias:
            return super().routing(logits, *args, **kwargs)
        input_ids = kwargs.get('input_ids')
        if input_ids is None and len(args) >= 2:
            input_ids = args[1]
        if input_ids is None:
            raise ValueError('input_ids is required when VL expert bias is enabled.')

        seq_length, batch_size = logits.shape[:2]
        image_mask = (input_ids == self.config.image_token_id).transpose(0, 1).contiguous()
        if image_mask.shape != (seq_length, batch_size):
            if (self.config.sequence_parallel and image_mask.shape[1] == batch_size
                    and image_mask.shape[0] % self.config.tensor_model_parallel_size == 0
                    and image_mask.shape[0] // self.config.tensor_model_parallel_size == seq_length):
                image_mask = tensor_parallel.scatter_to_sequence_parallel_region(image_mask)
            else:
                raise ValueError('image token mask cannot be aligned with router logits: '
                                 f'input_ids={tuple(input_ids.shape)}, logits={tuple(logits.shape)}.')

        original_expert_bias = self.expert_bias
        self.expert_bias = torch.where(
            image_mask.reshape(-1, 1),
            self.expert_bias_vl.unsqueeze(0),
            original_expert_bias.unsqueeze(0),
        )
        try:
            return super().routing(logits, *args, **kwargs)
        finally:
            self.expert_bias = original_expert_bias

    def _hash_routing(self, logits: torch.Tensor, input_ids: torch.Tensor):
        # Hash layers pick experts from the fixed ``tid2eid`` table, so mcore's ``routing()`` bypasses
        # ``router_replay`` for them entirely -- leaving R2 RECORD with a gap at every hash layer and a
        # per-layer recorded tensor that no longer lines up. Record the hash decision here (mirroring the
        # learned-router path) so a DeepSeek-V4 checkpoint with hash layers still yields complete R2
        # routing. Under force-load-balancing / force-biased the effective choice is a plain top-k on the
        # logits rather than the table, so record that instead to match what the forward actually used.
        probs, routing_map = super()._hash_routing(logits, input_ids)
        router_replay = getattr(self, 'router_replay', None)
        if router_replay is not None and router_replay.router_replay_action == RouterReplayAction.RECORD:
            if getattr(self.config, 'moe_router_force_load_balancing', False) \
                    or getattr(self.config, 'moe_router_force_biased', None) is not None:
                _, indices = torch.topk(logits, k=self.topk, dim=1)
            else:
                indices = self.tid2eid[input_ids.T.reshape(-1)].long()
            router_replay.record_indices(indices)
        return probs, routing_map

    @jit_fuser
    def _apply_expert_bias(self, routing_map: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        """
        Update expert bias and tokens_per_expert
        Prevent extra local tokens accumulation on evaluation or activation recomputation
        """
        # Older Megatron versions update bias from these counts without checking frozen_expert_bias.
        if self.enable_expert_bias and not getattr(self, 'frozen_expert_bias', False) and torch.is_grad_enabled():
            with torch.no_grad():
                if padding_mask is not None:
                    if padding_mask.ndim == 1:
                        padding_mask = padding_mask.unsqueeze(-1)
                    routing_map = routing_map & (~padding_mask)
                self.local_tokens_per_expert += routing_map.sum(dim=0)
