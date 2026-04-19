import torch
import transformer_engine
from contextlib import nullcontext
from functools import partial
from megatron.core import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import (gather_from_sequence_parallel_region,
                                                    gather_from_tensor_model_parallel_region,
                                                    scatter_to_sequence_parallel_region)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer as _MultiTokenPredictionLayer
from megatron.core.transformer.spec_utils import build_module
from megatron.core.utils import make_viewless_tensor
from typing import Callable, Optional

try:
    from megatron.core.typed_torch import apply_module
except ImportError:
    apply_module = None

from mcore_bridge.config import ModelConfig


class MultiTokenPredictionLayer(_MultiTokenPredictionLayer):

    def __init__(self, config: ModelConfig, submodules, *args, **kwargs):
        if config.fp8_param:
            eh_proj = submodules.eh_proj
            submodules.eh_proj = IdentityOp
        super().__init__(config, submodules, *args, **kwargs)
        self.tp_group = getattr(self, 'tp_group', None)
        if not config.fp8_param:
            return
        submodules.eh_proj = eh_proj
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
    ):
        assert context is None, 'multi token prediction + cross attention is not yet supported.'
        input_ids, position_ids, decoder_input, hidden_states = self._get_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            embedding=embedding,
            packed_seq_params=packed_seq_params,
            hidden_states=hidden_states,
        )
        assert not self.transformer_layer.self_attention.config.apply_rope_fusion
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if self.config.position_embedding_type == 'rope' and packed_seq:
            assert position_ids.shape[0] == 1, f'position_ids.shape: {position_ids.shape}'
            rotary_pos_emb = rotary_pos_emb[position_ids[0]]
        else:
            # mrope or not packed_seq
            rotary_pos_emb = torch.roll(rotary_pos_emb, shifts=-self.layer_number, dims=0)
        if self.config.recompute_granularity == 'full' and self.training:
            hidden_states = self._checkpointed_forward(
                partial(
                    self._proj_and_transformer_layer,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                ),
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
            )
        return hidden_states, input_ids, position_ids

    def _concat_embeddings(self, hidden_states: torch.Tensor, decoder_input: torch.Tensor):
        """
        Concatenate the tokens before sending to transformer layer.
        """
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
    ):
        from megatron.core.transformer.multi_token_prediction import roll_tensor

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
        # embedding
        if isinstance(embedding, tuple):
            embedding, decoder_input = embedding
        else:
            decoder_input = None
        if decoder_input is None:
            decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            enable_sp = self.config.sequence_parallel and self.config.tensor_model_parallel_size > 1
            if enable_sp:
                decoder_input = gather_from_sequence_parallel_region(decoder_input)
            decoder_input, _ = roll_tensor(
                decoder_input.transpose(0, 2),
                shifts=-self.layer_number,
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
