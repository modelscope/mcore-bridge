# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from contextlib import contextmanager
from typing import Optional

from ..constant import ModelType
from ..gpt_model import GPTModel
from ..modules import TransformerBlock
from ..register import ModelLoader, ModelMeta, register_model

try:
    from megatron.core.transformer.experimental_attention_variant.dsa import (DSAIndexerLossAutoScaler,
                                                                              DSAIndexerLossLoggingHelper, DSAttention,
                                                                              FusedDSAIndexerLoss, unfused_dsa_fn)
except ImportError:
    DSAttention = object


class GlmMoeDsaDSAttention(DSAttention):
    """DSAttention with shared indexer support for GLM 5.2.

    Refer: https://arxiv.org/abs/2603.12201 for more details.
    """

    def __init__(self, config, submodules, layer_number, *args, **kwargs):
        super().__init__(config, submodules, layer_number, *args, **kwargs)
        indexer_types = config.hf_config.indexer_types
        self.skip_topk = False
        if indexer_types is not None:
            layer_idx = layer_number - 1
            if layer_idx < len(indexer_types):
                self.skip_topk = indexer_types[layer_idx] == 'shared'

        if self.skip_topk:
            self.indexer = None

    def _get_float_mask(self, query, key, attention_mask, x, attn_mask_type):
        """Build a FP32 mask with -inf for masked positions."""
        sq = query.size(0)
        skv = key.size(0)
        if attn_mask_type is not None:
            from megatron.core.transformer.enums import AttnMaskType
            assert attn_mask_type == AttnMaskType.causal
            float_mask = torch.triu(
                torch.full((sq, skv), float('-inf'), dtype=torch.float32, device=x.device),
                diagonal=1,
            )
        else:
            b = query.size(1)
            assert attention_mask.shape == (b, 1, sq, skv)
            mask = attention_mask.squeeze()
            float_mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(mask, float('-inf'))
        return float_mask

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        x,
        qr,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
    ):
        shared_topk_indices = getattr(self, '_shared_topk_indices', None)

        if self.skip_topk:
            # Shared layer: reuse topk_indices from previous full layer
            assert shared_topk_indices is not None and 'topk_indices' in shared_topk_indices, (
                f'GLM 5.2 DSA: Layer {self.layer_number} is a "shared" indexer layer but no '
                f'"full" layer precedes it in this PP stage. Please adjust '
                f'`--pipeline_model_parallel_layout` to ensure each PP stage starts with a "full" indexer layer. '
                f'indexer_types: {self.config.hf_config.indexer_types}.')
            topk_indices = shared_topk_indices['topk_indices']
            output = unfused_dsa_fn(query, key, value, topk_indices, self.softmax_scale)
            return output

        # Full layer: compute topk_indices, store for shared layers, then run sparse attention.
        # We override the full forward to avoid double-computing topk_indices.
        x = x.detach()
        qr = qr.detach()
        float_mask = self._get_float_mask(query, key, attention_mask, x, attn_mask_type)

        if self.training and torch.is_grad_enabled():
            q, k, weights = self.indexer.forward_before_topk(x, qr, packed_seq_params)
            indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', 0.0)

            topk_indices, indexer_loss = FusedDSAIndexerLoss.apply(
                q, weights, k, query.detach(), key.detach(), self.softmax_scale,
                self.indexer.index_topk, indexer_loss_coeff, float_mask,
                getattr(self.config, 'dsa_indexer_use_sparse_loss',
                        False), self.indexer.pg_collection, self.config.calculate_per_token_loss)
            if indexer_loss_coeff > 0:
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=indexer_loss,
                    layer_number=self.layer_number,
                    num_layers=max(
                        self.layer_number,
                        self.config.num_layers + (self.config.mtp_num_layers or 0),
                    ),
                )
            output = unfused_dsa_fn(query, key, value, topk_indices, self.softmax_scale)
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
        else:
            _, topk_indices = self.indexer.forward_with_scores(
                x, qr, mask=float_mask, packed_seq_params=packed_seq_params)
            output = unfused_dsa_fn(query, key, value, topk_indices, self.softmax_scale)

        # Store topk_indices for subsequent shared layers (in-place dict mutation)
        if shared_topk_indices is not None:
            shared_topk_indices['topk_indices'] = topk_indices.detach()

        return output


class GlmMoeDsaGPTModel(GPTModel):
    """GPT model for GLM 5.2 with shared DSA indexer support.

    Creates a ``shared_topk_indices`` dict and passes it through
    ``extra_block_kwargs`` so that "full" DSA layers can store their
    topk_indices for reuse by subsequent "shared" layers.
    """

    def forward(self, *args, **kwargs):
        extra_block_kwargs = kwargs.get('extra_block_kwargs') or {}
        extra_block_kwargs['shared_topk_indices'] = {}
        kwargs['extra_block_kwargs'] = extra_block_kwargs
        return super().forward(*args, **kwargs)


@contextmanager
def _shared_topk_indices_context(layer, shared_topk_indices):
    """Temporarily inject shared_topk_indices into the core attention module."""
    core_attn = None
    if shared_topk_indices is not None and hasattr(layer, 'self_attention'):
        _attn = getattr(layer.self_attention, 'core_attention', None)
        if isinstance(_attn, GlmMoeDsaDSAttention):
            core_attn = _attn
            core_attn._shared_topk_indices = shared_topk_indices
    try:
        yield
    finally:
        if core_attn is not None:
            core_attn._shared_topk_indices = None


class GlmMoeDsaTransformerBlock(TransformerBlock):
    """TransformerBlock that routes ``shared_topk_indices`` to DSAttention.

    Pops ``shared_topk_indices`` from kwargs before calling the layer
    (so it doesn't hit ``_forward_attention``'s fixed signature), injects
    it via context manager, and restores it afterward for subsequent layers.
    """

    def _layer_forward(self, layer, hidden_states, **kwargs):
        shared_topk_indices = kwargs.pop('shared_topk_indices', None)
        with _shared_topk_indices_context(layer, shared_topk_indices):
            result = super()._layer_forward(layer, hidden_states, **kwargs)
        # Restore for subsequent layers
        if shared_topk_indices is not None:
            kwargs['shared_topk_indices'] = shared_topk_indices
        return result


class GlmMoeDsaLoader(ModelLoader):
    model_cls = GlmMoeDsaGPTModel
    transformer_block = GlmMoeDsaTransformerBlock

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        transformer_layer_spec = super().get_transformer_layer_spec(vp_stage)

        indexer_types = self.config.hf_config.indexer_types
        if indexer_types is not None:
            for layer_spec in transformer_layer_spec.layer_specs:
                core_attn = layer_spec.submodules.self_attention.submodules.core_attention
                if hasattr(core_attn, 'module') and issubclass(core_attn.module, DSAttention):
                    core_attn.module = GlmMoeDsaDSAttention

        return transformer_layer_spec


register_model(ModelMeta(
    ModelType.glm_moe_dsa,
    ['glm_moe_dsa'],
    loader=GlmMoeDsaLoader,
))
