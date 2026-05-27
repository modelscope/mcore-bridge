# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import transformer_engine
import warnings
from contextlib import nullcontext
from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import te_checkpoint
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import (gather_from_sequence_parallel_region,
                                                    gather_from_tensor_model_parallel_region,
                                                    scatter_to_sequence_parallel_region)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer as _MultiTokenPredictionLayer
from megatron.core.transformer.spec_utils import build_module
from megatron.core.utils import make_viewless_tensor
from typing import Callable, Optional

from mcore_bridge.utils import roll_tensor

from .transformer_block import _checkpoint_flatten, _checkpoint_unflatten

try:
    from megatron.core.typed_torch import apply_module
except ImportError:
    apply_module = None

from mcore_bridge.config import ModelConfig


class MultiTokenPredictionLayer(_MultiTokenPredictionLayer):

    def __init__(self, config: ModelConfig, submodules, *args, **kwargs):
        replace_eh_proj = submodules.eh_proj is not None and config.fp8_param
        if replace_eh_proj:
            eh_proj = submodules.eh_proj
            submodules.eh_proj = IdentityOp
        try:
            super().__init__(config, submodules, *args, **kwargs)
        finally:
            if replace_eh_proj:
                submodules.eh_proj = eh_proj
        self.tp_group = getattr(self, 'tp_group', None)
        if not replace_eh_proj:
            return
        fp8_context = transformer_engine.pytorch.fp8_model_init(enabled=False)
        with fp8_context:
            self.eh_proj = build_module(
                self.submodules.eh_proj,
                self.config.hidden_size * 2,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='mtp_eh_proj',
                tp_group=self.tp_group,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        context: torch.Tensor = None,
        context_mask: torch.Tensor = None,
        rotary_pos_emb: torch.Tensor = None,
        rotary_pos_cos: torch.Tensor = None,
        rotary_pos_sin: torch.Tensor = None,
        attention_bias: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        sequence_len_offset: torch.Tensor = None,
        embedding=None,
        decoder_input=None,
        layer_number: Optional[int] = None,
        **kwargs,
    ):
        assert context is None, 'multi token prediction + cross attention is not yet supported.'
        if layer_number is None:
            layer_number = self.layer_number
        input_ids, position_ids, decoder_input, hidden_states = self._get_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            embedding=embedding,
            packed_seq_params=packed_seq_params,
            hidden_states=hidden_states,
            decoder_input=decoder_input,
        )
        assert not self.transformer_layer.self_attention.config.apply_rope_fusion
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if self.config.position_embedding_type == 'rope' and packed_seq:
            assert position_ids.shape[0] == 1, f'position_ids.shape: {position_ids.shape}'
            if isinstance(rotary_pos_emb, dict):
                for k, v in rotary_pos_emb.items():
                    rotary_pos_emb[k] = v[position_ids[0]]
            else:
                rotary_pos_emb = rotary_pos_emb[position_ids[0]]
        else:
            # mrope or not packed_seq
            if isinstance(rotary_pos_emb, dict):
                for k, v in rotary_pos_emb.items():
                    rotary_pos_emb[k] = torch.roll(v, shifts=-layer_number, dims=0)
            else:
                rotary_pos_emb = torch.roll(rotary_pos_emb, shifts=-layer_number, dims=0)
        if self.config.recompute_granularity == 'full' and self.training:
            hidden_states = self._checkpointed_forward(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                **kwargs,
            )
        else:
            hidden_states = self._proj_and_transformer_layer(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                **kwargs,
            )
        return hidden_states, input_ids, position_ids, decoder_input

    # Code borrowed from NVIDIA/Megatron-LM
    def _checkpointed_forward(
        self,
        hidden_states: torch.Tensor,
        decoder_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward through ``_proj_and_transformer_layer`` with activation
        recomputation.

        Mirrors ``transformer_block._checkpointed_forward``:

        * Non-tensor objects (``attention_bias``, ``inference_params``,
          ``packed_seq_params``) are captured by the ``custom_forward``
          closure; only tensor / ``None`` arguments flow positionally
          through the underlying checkpoint primitive. This is required
          by both backends: ``tensor_parallel.checkpoint`` because its
          ``save_for_backward`` only accepts tensors and ``None``, and
          ``te_checkpoint`` because its reentrant implementation only
          tracks positional tensor inputs as checkpoint inputs (kwarg
          tensors are not represented in the recompute backward path).
        * Quantized recipes (fp8, fp4) route through ``te_checkpoint``;
          everything else uses ``tensor_parallel.checkpoint``.
        * Only ``fp8 + delayed scaling`` needs an outer quantization
          context entered before ``te_checkpoint``; see the
          ``outer_quantization_context`` block below.
        """

        # Variables that don't require gradients can be captured via closure.
        _ckpt_attention_mask = attention_mask
        _ckpt_rotary_pos_emb = rotary_pos_emb
        extra_kwargs_keys = tuple(kwargs.keys())
        _extra_flat_tensors = []
        _extra_schemas = [_checkpoint_flatten(v, _extra_flat_tensors) for v in kwargs.values()]

        def custom_forward(hidden_states, decoder_input, context, context_mask, rotary_pos_cos, rotary_pos_sin,
                           sequence_len_offset, *extra_flat):
            rebuilt = [_checkpoint_unflatten(s, extra_flat) for s in _extra_schemas]
            extra_kwargs = dict(zip(extra_kwargs_keys, rebuilt))
            return self._proj_and_transformer_layer(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=_ckpt_attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=_ckpt_rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                **extra_kwargs,
            )

        # Decide the outer quantization context, matching
        # ``transformer_block._checkpointed_forward``. Only ``fp8 + delayed
        # scaling`` needs an active context at the ``te_checkpoint`` entry
        # point: TE's ``_CheckpointFunction.forward`` samples
        # ``FP8GlobalStateManager.is_fp8_enabled()`` there to gate the
        # phase-1 amax-buffer stash that phase-2 backward looks up via
        # ``global_fp8_buffer_pos_fwd_recompute``. With fp8 only entered
        # *inside* ``_proj_and_transformer_layer``, TE samples fp8 as off,
        # phase-1 skips the stash, and phase-2 raises ``KeyError``.
        # Non-delayed fp8 recipes (MXFP8BlockScaling, Float8CurrentScaling)
        # and fp4 (NVFP4BlockScaling) treat the stash/lookup as a noop, so
        # the inner context entered inside ``_proj_and_transformer_layer``
        # is sufficient.
        if self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed:
            outer_quantization_context = get_fp8_context(self.config)
        else:
            outer_quantization_context = nullcontext()

        def checkpoint_handler():
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            # fp4 quantization is internally implemented via TE's
            # ``fp8_autocast`` (see ``fp4_utils.get_fp4_context``), so
            # quantized recompute on either fp8 or fp4 must go through
            # ``te_checkpoint``. Matches ``transformer_block``'s policy.
            if self.config.fp8 or self.config.fp4:

                return te_checkpoint(
                    custom_forward,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    decoder_input,
                    context,
                    context_mask,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    sequence_len_offset,
                    *_extra_flat_tensors,
                )
            else:
                # tensor_parallel.checkpoint stashes args via autograd's
                # ``save_for_backward``, which only accepts tensors and ``None``.
                # Pass tensor / ``None`` args positionally and capture the
                # non-tensor objects (``attention_bias``, ``inference_params``,
                # ``packed_seq_params``) via the ``custom_forward`` closure.
                return tensor_parallel.checkpoint(
                    custom_forward,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    decoder_input,
                    context,
                    context_mask,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    sequence_len_offset,
                    *_extra_flat_tensors,
                )

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            assert (self.config.recompute_num_layers == 1), 'recompute_num_layers must be 1 for MTP recompute'
            with outer_quantization_context:
                outputs = checkpoint_handler()
        elif self.config.recompute_method == 'block':
            # TODO: implement block-based recompute for MTP
            warnings.warn("recompute_method == 'block' is not supported for MTP yet."
                          ' Skipping recompute.')
            outputs = self._proj_and_transformer_layer(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                **kwargs,
            )
        else:
            raise ValueError('Invalid activation recompute method.')

        return outputs

    def _proj_and_transformer_layer(
        self,
        hidden_states: torch.Tensor,
        decoder_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Concatenates embeddings with hidden states and then applies transformer layer forward.
        """
        padding_mask = kwargs.pop('padding_mask', None)
        if padding_mask is not None:
            kwargs['padding_mask'] = padding_mask
        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # Unlike transformer_block.py which needs to support mixed-precision in
        # different layers,currently MTP only use global fp8 context.
        if self.config.fp8:
            fp8_context = get_fp8_context(self.config)
            transformer_layer_fp8_context = get_fp8_context(self.config)
        else:
            fp8_context = nullcontext()
            transformer_layer_fp8_context = nullcontext()

        # TODO: currently ignoring FP4 in MTP layers because we need more numerical validation
        with rng_context:
            with fp8_context:
                hidden_states = self._concat_embeddings(hidden_states, decoder_input)

            # Use a separate fp8 context for the transformer layer. This is to ensure that when the
            # transformer layer is cudagraphed, the FP8GlobalStateManager.is_first_fp8_module() is
            # True so that the fp8 weight caching can be triggered correctly.
            with transformer_layer_fp8_context:
                if getattr(self, 'mtp_layer_pattern', None) is not None:
                    hidden_states = self.transformer_layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        inference_context=inference_params,
                        packed_seq_params=packed_seq_params,
                        **kwargs,
                    )
                else:
                    # GPT path: single TransformerLayer
                    hidden_states, _ = self.transformer_layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        rotary_pos_cos=rotary_pos_cos,
                        rotary_pos_sin=rotary_pos_sin,
                        attention_bias=attention_bias,
                        inference_params=inference_params,
                        packed_seq_params=packed_seq_params,
                        sequence_len_offset=sequence_len_offset,
                        **kwargs,
                    )

        if not getattr(self, 'mhc_enabled', False):
            hidden_states = self._postprocess(hidden_states)

        return hidden_states

    def _concat_embeddings(self, hidden_states: torch.Tensor, decoder_input: torch.Tensor):
        """
        Concatenate the tokens before sending to transformer layer.
        """
        if getattr(self, 'mhc_enabled', False):
            return super()._concat_embeddings(hidden_states, decoder_input)
        if apply_module is None:
            decoder_input = self.enorm(decoder_input)
            hidden_states = self.hnorm(hidden_states)
        else:
            decoder_input = apply_module(self.enorm)(decoder_input)
            hidden_states = apply_module(self.hnorm)(hidden_states)
        decoder_input = make_viewless_tensor(inp=decoder_input, requires_grad=True, keep_graph=True)
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        # At the (k - 1)-th MTP module, concatenates the i-th token's hidden_states
        # and the (i + K)-th token's embedding, and combine them with linear projection.
        hidden_states = torch.cat((decoder_input, hidden_states), -1)
        if self.config.fp8_param:
            fp8_context = transformer_engine.pytorch.fp8_autocast(enabled=False)
        else:
            fp8_context = nullcontext()
        with fp8_context:
            hidden_states, _ = self.eh_proj(hidden_states)
        # For tensor parallel we need to gather the tensor across the model-parallel
        # ranks after the linear projection. This used to call
        # `all_gather_last_dim_from_tensor_parallel_region`, but that utility reduces
        # the gradient in backward pass and was therefore incorrect in this context.
        # It has been replaced with the correct `gather_from_tensor_model_parallel_region`.
        hidden_states = gather_from_tensor_model_parallel_region(hidden_states, group=self.tp_group)
        # For sequence parallel, scatter after linear_fc and before transformer layer.
        if self.sequence_parallel:
            hidden_states = scatter_to_sequence_parallel_region(hidden_states, group=self.tp_group)
        return hidden_states

    def _get_embeddings(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        embedding: Callable,
        hidden_states: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams] = None,
        decoder_input=None,
    ):
        # Calc logits for the current Multi-Token Prediction (MTP) layers.
        input_ids, _ = roll_tensor(
            input_ids,
            shifts=-1,
            dims=-1,
            cp_group=self.cp_group,
            packed_seq_params=packed_seq_params,
        )
        position_ids, _ = roll_tensor(
            position_ids,
            shifts=-1,
            dims=-1,
            cp_group=self.cp_group,
            packed_seq_params=packed_seq_params,
        )
        if decoder_input is None:
            decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            enable_sp = self.config.sequence_parallel and self.config.tensor_model_parallel_size > 1
            if enable_sp:
                decoder_input = gather_from_sequence_parallel_region(decoder_input, tensor_parallel_output_grad=False)
            decoder_input, _ = roll_tensor(
                decoder_input.transpose(0, 2),
                shifts=-1,
                dims=-1,
                cp_group=self.cp_group,
                packed_seq_params=packed_seq_params,
            )
            decoder_input = decoder_input.transpose(0, 2).contiguous()
            if enable_sp:
                decoder_input = scatter_to_sequence_parallel_region(decoder_input)
        if self.config.mtp_decoder_input_detach:
            decoder_input = decoder_input.detach()
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        return input_ids, position_ids, decoder_input, hidden_states
