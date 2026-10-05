# Copyright (c) ModelScope Contributors. All rights reserved.
"""Mask-aware MoE routing replay for the mcore-bridge ``TopKRouter``.

mcore>=0.16 ships a built-in :class:`~megatron.core.transformer.moe.router_replay.RouterReplay` that
records / replays whole-sequence expert choices, but its ``set_target_indices`` takes no
``replay_mask`` and its ``get_replay_topk`` replays *every* token unconditionally. Selective replay
-- only re-route the tokens whose routing actually influences the trained log-probs, and let the
rest recompute natively under the current weights -- needs a per-token mask blended as
``where(mask, replayed, native)``.

Rather than monkeypatch mcore or fork its engine, we subclass it and override exactly the two
methods that touch target indices, so the record / backward-FIFO / static-buffer plumbing is
inherited unchanged. mcore's ``TopKRouter.routing()`` already forwards ``self.router_replay`` into
``topk_routing_with_score_function`` -> ``get_replay_topk``, so once :class:`TopKRouter` installs a
``MaskedRouterReplay`` in place of mcore's instance the mask flows through transparently and the
``get_replay_topk`` override stays drop-in (its signature is identical to the base's).

With ``replay_mask=None`` on every layer this behaves byte-for-byte like mcore's built-in replay
(whole-sequence replay), so the change is backward compatible with the existing R2/R3 wiring.
"""
from typing import Callable, List, Optional, Tuple

import torch

from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction


class MaskedRouterReplay(RouterReplay):
    """mcore ``RouterReplay`` extended with a per-token ``replay_mask`` for selective replay.

    Instances register themselves in ``RouterReplay.global_router_replay_instances`` via the base
    ``__init__``, so every static driver method mcore/twinkle already uses (``set_global_router_
    replay_action`` / ``get_recorded_data`` / ``clear_global_indices`` ...) keeps working on them.
    """

    def __init__(self):
        super().__init__()  # appends self to RouterReplay.global_router_replay_instances
        # Per-token mask selecting which tokens replay their recorded routing (True) versus recompute
        # natively (False). Mirrored into a backward FIFO so activation recompute pops index+mask pairs.
        self.target_replay_mask: Optional[torch.Tensor] = None
        self.replay_backward_mask_list: List[Optional[torch.Tensor]] = []

    def set_target_indices(self, topk_indices: torch.Tensor, replay_mask: Optional[torch.Tensor] = None):
        """Store the replay target plus its per-token mask, keeping the backward FIFO in lockstep.

        ``super()`` sets ``target_topk_idx`` and appends ``topk_indices`` to ``replay_backward_list``;
        we mirror ``replay_mask`` into ``replay_backward_mask_list`` at the same position so that
        REPLAY_BACKWARD pops indices and masks as a pair (they are consumed together in
        ``get_replay_topk``).
        """
        super().set_target_indices(topk_indices)
        self.target_replay_mask = replay_mask
        self.replay_backward_mask_list.append(replay_mask)

    def clear_indices(self):
        super().clear_indices()
        self.target_replay_mask = None
        self.replay_backward_mask_list = []

    @staticmethod
    def set_replay_data(all_layers_topk_indices: List[torch.Tensor], replay_mask: Optional[torch.Tensor] = None):
        """Mask-aware counterpart of ``RouterReplay.set_replay_data``.

        Distributes each layer's target indices together with the shared per-token ``replay_mask``
        (a token either influences the trained log-probs or not -- the same decision holds across
        layers, so one mask serves every layer of this microbatch). ``replay_mask=None`` degrades to
        mcore's whole-sequence behavior.
        """
        instances = RouterReplay.global_router_replay_instances
        if len(all_layers_topk_indices) != len(instances):
            raise ValueError(f'The number of replay tensors ({len(all_layers_topk_indices)}) '
                             f'does not match instances ({len(instances)}).')
        for i, router_instance in enumerate(instances):
            router_instance.set_target_indices(all_layers_topk_indices[i], replay_mask)

    def get_replay_topk(
        self,
        scores: torch.Tensor,
        topk: int,
        num_groups: Optional[int] = None,
        group_topk: Optional[int] = None,
        default_compute_topk: Callable[[torch.Tensor, int, Optional[int], Optional[int]],
                                       Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-k with replay actions, blending replayed and native routes per token under a mask.

        Signature is identical to the base so mcore's ``topk_routing_with_score_function`` calls it
        unchanged. RECORD and the disabled (``else``) path are inherited verbatim; REPLAY_FORWARD /
        REPLAY_BACKWARD add the mask blend: masked-in rows take the replayed indices, masked-out rows
        take freshly computed native indices, so only the tokens that matter are pinned.
        """
        action = self.router_replay_action
        if action == RouterReplayAction.RECORD:
            probs, top_indices = default_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
            self.record_indices(top_indices)
            return probs, top_indices
        if action == RouterReplayAction.REPLAY_FORWARD:
            top_indices, replay_mask = self.target_topk_idx, self.target_replay_mask
        elif action == RouterReplayAction.REPLAY_BACKWARD:
            top_indices = self.replay_backward_list.pop(0)
            replay_mask = self.replay_backward_mask_list.pop(0)
        else:
            return default_compute_topk(scores, topk, num_groups, group_topk)

        if top_indices is None:
            # Nothing to replay for this step (e.g. a layer with no recorded routing): recompute
            # natively rather than crash -- matches the base engine's tolerance for a missing target.
            return default_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

        top_indices = top_indices.to(scores.device)
        if replay_mask is not None:
            _, native_indices = default_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
            replay_mask = replay_mask.to(scores.device)
            if top_indices.shape != native_indices.shape or replay_mask.numel() != scores.shape[0]:
                raise RuntimeError(
                    'Router replay tensors are not aligned: '
                    f'scores={tuple(scores.shape)}, targets={tuple(top_indices.shape)}, '
                    f'native={tuple(native_indices.shape)}, mask={tuple(replay_mask.shape)}')
            top_indices = torch.where(replay_mask.bool().unsqueeze(-1), top_indices, native_indices)
        probs = scores.gather(1, top_indices)
        return probs, top_indices
