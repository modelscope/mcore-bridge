# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from contextlib import contextmanager
from megatron.core import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import VocabParallelEmbedding, scatter_to_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec

from mcore_bridge.config import ModelConfig
from mcore_bridge.utils import reconstruct_tensor_cp, split_cp_inputs

from .gpt_model import GPTModel


class MultimodalGPTModel(MegatronModule):
    language_model_cls = GPTModel

    def __init__(self,
                 config: ModelConfig,
                 transformer_layer_spec: ModuleSpec,
                 pre_process: bool = True,
                 post_process: bool = True,
                 *_args,
                 **kwargs):
        super().__init__(config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.language_model = self.language_model_cls(config, transformer_layer_spec, pre_process, post_process, *_args,
                                                      **kwargs)
        self.vp_stage = self.language_model.vp_stage
        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights
        self.model_meta = config.model_meta
        self.visual = None
        if pre_process and self.model_meta.visual_cls is not None:
            self.visual = self.model_meta.visual_cls(config)

    @contextmanager
    def _patch_word_embeddings(self, kwargs):
        origin_forward = VocabParallelEmbedding.forward

        def forward(_self, input_):
            reduce_scatter_embeddings = _self.reduce_scatter_embeddings
            _self.reduce_scatter_embeddings = False
            input_ = torch.masked_fill(input_, input_ < 0, 0)
            res = origin_forward(_self, input_)
            _self.reduce_scatter_embeddings = reduce_scatter_embeddings
            packed_seq_params = kwargs.get('packed_seq_params')
            if self.visual is not None:
                if self.config.language_model_only:
                    res = self.visual.get_inputs_embeds_language_model(res, **kwargs)
                else:
                    res = self.visual.get_inputs_embeds(res, **kwargs)
                kwargs.clear()
                if isinstance(res, dict):
                    # compat dict
                    inputs_embeds = res.pop('inputs_embeds')
                    kwargs.update(res)
                    res = inputs_embeds
            if self.config.context_parallel_size > 1:
                res = split_cp_inputs(res, getattr(packed_seq_params, 'cu_seqlens_q', None), 1)
            if reduce_scatter_embeddings:
                res = res.transpose(0, 1).contiguous()
                res = scatter_to_sequence_parallel_region(res, group=_self.tp_group)
            return res

        VocabParallelEmbedding.forward = forward
        try:
            yield
        finally:
            VocabParallelEmbedding.forward = origin_forward

    # Code borrowed from NVIDIA/Megatron-LM
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        decoder_input: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        mtp_labels: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        # ``mtp_labels`` is named explicitly rather than left to **kwargs: everything not named here
        # ends up in ``extra_block_kwargs``, which is forwarded to the decoder blocks, so an MTP
        # target passed positionally by name would be handed to attention instead of the MTP heads.
        extra_kwargs = {k: kwargs[k] for k in self.language_model.extra_forward_keys}
        # Compatible with legacy mcore-bridge behavior.
        cp_size = self.config.context_parallel_size
        needs_split = cp_size > 1 and input_ids is not None and position_ids.shape[-1] * cp_size == input_ids.shape[-1]
        if decoder_input is not None:
            pass
        elif self.pre_process:
            input_ids_ = input_ids if needs_split else reconstruct_tensor_cp(input_ids, packed_seq_params, dim=1)
            kwargs.update({'input_ids': input_ids_, 'packed_seq_params': packed_seq_params})
            with self._patch_word_embeddings(kwargs):
                decoder_input = self.language_model.embedding(input_ids=input_ids_, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None
            kwargs = {}
        kwargs.update(extra_kwargs)
        if needs_split:
            input_ids = split_cp_inputs(input_ids, getattr(packed_seq_params, 'cu_seqlens_q', None), dim=1)
        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            mtp_labels=mtp_labels,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=kwargs,
        )

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        return self.language_model.set_input_tensor(input_tensor)

    def get_input_tensor(self):
        return self.language_model.get_input_tensor()

    def shared_embedding_or_output_weight(self) -> torch.Tensor:
        return self.language_model.shared_embedding_or_output_weight()
