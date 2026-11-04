# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import math
import torch
from megatron.core import mpu
from megatron.core.models.hybrid.hybrid_model import HybridModel as McoreHybridModel
from megatron.core.transformer.spec_utils import ModuleSpec
from typing import Optional

from mcore_bridge.config import ModelConfig
from mcore_bridge.utils import split_cp_inputs

from .mtp import get_physical_mtp_depth


class HybridModel(McoreHybridModel):
    """Thin adapter over megatron-core's HybridModel.

    Upstream `HybridModel` already covers embedding, the hybrid layer stack, MTP
    (via `process_mtp_loss`) and the loss/logits tail, so only two things are added
    here:

    1. Translate `ModelConfig` into the upstream constructor signature.
    2. Build `padding_mask`, which upstream takes as a forward argument but never
       computes. Deriving it needs the CP size, the TP size and the current TP rank,
       so it stays on this side rather than leaking into the caller.
    """

    config: ModelConfig
    # These kwargs would otherwise be cleared by the multimodal embedding hook / PP stage.
    extra_forward_keys = ('loss_mask', 'padding_mask')
    _upstream_forward_keys = set(inspect.signature(McoreHybridModel.forward).parameters) - {
        'self', 'input_ids', 'position_ids', 'attention_mask', 'packed_seq_params', 'padding_mask'
    }

    def __init__(
        self,
        config: ModelConfig,
        transformer_layer_spec: ModuleSpec,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: Optional[int] = None,
    ):
        vocab_size = math.ceil(
            config.padded_vocab_size / config.tensor_model_parallel_size) * config.tensor_model_parallel_size
        hybrid_layer_pattern = self._resolve_hybrid_layer_pattern(config)
        super().__init__(
            config,
            transformer_layer_spec,
            vocab_size,
            config.max_position_embeddings,
            hybrid_layer_pattern=hybrid_layer_pattern,
            pre_process=pre_process,
            post_process=post_process,
            share_embeddings_and_output_weights=not config.untie_embeddings_and_output_weights,
            position_embedding_type=config.position_embedding_type,
            rotary_base=config.rotary_base,
            vp_stage=vp_stage,
        )

    @staticmethod
    def _resolve_hybrid_layer_pattern(config: ModelConfig) -> Optional[str]:
        pattern = config.hybrid_layer_pattern
        mtp_pattern = getattr(config, 'mtp_hybrid_override_pattern', None)
        mtp_num_layers = get_physical_mtp_depth(config)
        # Pattern entries describe PHYSICAL MTP layers: repeated (shared-weight)
        # mode collapses the logical depth to one physical layer.
        if (pattern and mtp_pattern and mtp_num_layers and '/' not in pattern):
            pattern = pattern + '/' + '/'.join([mtp_pattern] * mtp_num_layers)
        return pattern

    def _get_padding_mask(self, attention_mask) -> Optional[torch.Tensor]:
        """Mark fully-padded sequence positions, sharded to match the hidden states."""
        if isinstance(attention_mask, dict):
            attention_mask = attention_mask['full_attention']
        if attention_mask is None:
            return None
        padding_mask = ~attention_mask.bool() if attention_mask.dim() == 2 else attention_mask.all(dim=(1, 2))
        if self.config.context_parallel_size > 1:
            padding_mask = split_cp_inputs(padding_mask, None, 1)
        tp_size = self.config.tensor_model_parallel_size
        if self.config.sequence_parallel and tp_size > 1:
            assert padding_mask.shape[1] % tp_size == 0, f'padding_mask.shape: {padding_mask.shape}'
            padding_mask = torch.chunk(padding_mask, tp_size, dim=1)[mpu.get_tensor_model_parallel_rank()]
        return padding_mask.contiguous()

    def forward(self,
                input_ids,
                position_ids,
                attention_mask=None,
                *args,
                packed_seq_params=None,
                padding_mask=None,
                extra_block_kwargs=None,
                **kwargs):
        # MMGPT separates media kwargs from forward controls; Hybrid has no extra_block_kwargs.
        extra_block_kwargs = extra_block_kwargs or {}
        if padding_mask is None:
            padding_mask = extra_block_kwargs.get('padding_mask')
        for key in self.extra_forward_keys:
            if key != 'padding_mask' and key in extra_block_kwargs:
                kwargs.setdefault(key, extra_block_kwargs[key])
        kwargs = {key: value for key, value in kwargs.items() if key in self._upstream_forward_keys}
        if padding_mask is None and packed_seq_params is None:
            padding_mask = self._get_padding_mask(attention_mask)
        return super().forward(
            input_ids,
            position_ids,
            attention_mask,
            *args,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
            **kwargs,
        )

    def get_input_tensor(self):
        return self.decoder.input_tensor
