# Copyright (c) ModelScope Contributors. All rights reserved.
from contextlib import contextmanager
from typing import Optional

import torch
from megatron.core import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import VocabParallelEmbedding, scatter_to_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec

from mcore_bridge.config import ModelConfig
from mcore_bridge.utils import reconstruct_tensor_cp, split_cp_inputs
from .gpt_model import GPTModel


def _pure_text_batch(kwargs: dict, media_input_keys) -> bool:
    """True when every declared media key is None -- the tower would run on fabricated zeros."""
    return all(kwargs.get(key) is None for key in media_input_keys)


def _check_no_media_placeholders(kwargs: dict) -> None:
    """A present, non-zero media-position marker with no media at all is broken input.

    ``mm_token_type_ids`` marks each image (1) / video (2) token position; non-zero entries
    with both ``pixel_values`` and ``pixel_values_videos`` absent mean placeholders without
    media, which the vision path cannot score either. Checked only when the family emits the
    key, so families that do not carry it are unaffected.
    """
    token_types = kwargs.get('mm_token_type_ids')
    if token_types is not None and bool((token_types != 0).any()):
        raise ValueError('mm_token_type_ids carries media placeholders but no pixel_values / '
                         'pixel_values_videos; the batch would train on placeholder embeddings.')


class MultimodalGPTModel(MegatronModule):
    language_model_cls = GPTModel

    #: Set by the training strategy when skipping the tower on all-text micro-batches is safe
    #: under the active parallelism (synchronous DDP only). Off by default: plain inference and
    #: unsafe parallel layouts keep the original fabricated-zero path.
    enable_text_vision_skip = False

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
        # Surface the language model's typed-pipeline payload interface on the wrapper. The PP
        # schedulers locate it with ``get_attr_wrapped_model(chunk, 'pipeline_payload_factory')``,
        # which only descends through ``.module`` wrappers and never reaches ``self.language_model``.
        # Without this a HybridModel backbone that configures a custom cross-stage payload (e.g.
        # DeepSeek-V4.1 CSA2 / single-pass mHC) is not recognised as typed, so both the 1F1B and the
        # interleaved (VPP) schedules fall back to the shape-based ``P2PCommunicator`` -- which calls
        # ``.size()`` on the payload object and crashes. Backbones without a payload (plain
        # ``GPTModel``, GLM's HybridModel adapter) expose ``None`` here and keep the legacy path.
        self.pipeline_payload_factory = getattr(self.language_model, 'pipeline_payload_factory', None)
        self.pipeline_payload_spec = getattr(self.language_model, 'pipeline_payload_spec', None)
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
                elif self.enable_text_vision_skip and _pure_text_batch(kwargs, self.visual.media_input_keys):
                    _check_no_media_placeholders(kwargs)
                    # Keep res: with no media the tower's write-back is the identity, so the
                    # word embeddings are already the final inputs_embeds.
                else:
                    res = self.visual.get_inputs_embeds(res, **kwargs)
                kwargs.clear()
                if isinstance(res, dict):
                    # compat dict
                    inputs_embeds = res.pop('inputs_embeds')
                    kwargs.update(res)
                    res = inputs_embeds
            if self.config.context_parallel_size > 1:
                res = split_cp_inputs(
                    res,
                    getattr(packed_seq_params, 'cu_seqlens_q', None),
                    1,
                    cp_partition_mode=getattr(self.config, 'cp_partition_mode', 'zigzag'))
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
        # Same knob as GPTModel.forward. Left to **kwargs it would be routed into
        # extra_block_kwargs and silently dropped, so a TP caller asking for gathered logits would
        # get the local vocab shard back.
        runtime_gather_output: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        # ``mtp_labels`` is named explicitly rather than left to **kwargs: everything not named here
        # ends up in ``extra_block_kwargs``, which is forwarded to the decoder blocks, so an MTP
        # target passed positionally by name would be handed to attention instead of the MTP heads.
        inference_context = kwargs.pop('inference_context', None)
        extra_kwargs = {k: kwargs[k] for k in self.language_model.extra_forward_keys if k in kwargs}
        # Compatible with legacy mcore-bridge behavior.
        cp_size = self.config.context_parallel_size
        cp_partition_mode = getattr(self.config, 'cp_partition_mode', 'zigzag')
        needs_split = cp_size > 1 and input_ids is not None and position_ids.shape[-1] * cp_size == input_ids.shape[-1]
        if decoder_input is not None:
            pass
        elif self.pre_process:
            input_ids_ = input_ids if needs_split else reconstruct_tensor_cp(
                input_ids, packed_seq_params, dim=1, cp_partition_mode=cp_partition_mode)
            kwargs.update({'input_ids': input_ids_, 'packed_seq_params': packed_seq_params})
            # defect126: mm_token_type_ids must track input_ids' CP transform so get_inputs_embeds
            # sees matching shapes (it validates token_types.shape == input_ids.shape under CP>1).
            _tt = kwargs.get('mm_token_type_ids')
            if _tt is not None and cp_size > 1:
                kwargs['mm_token_type_ids'] = _tt if needs_split else reconstruct_tensor_cp(
                    _tt, packed_seq_params, dim=1, cp_partition_mode=cp_partition_mode)
            with self._patch_word_embeddings(kwargs):
                decoder_input = self.language_model.embedding(input_ids=input_ids_, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None
            kwargs = {}
        kwargs.update(extra_kwargs)
        if needs_split:
            input_ids = split_cp_inputs(
                input_ids, getattr(packed_seq_params, 'cu_seqlens_q', None), dim=1, cp_partition_mode=cp_partition_mode)
        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            mtp_labels=mtp_labels,
            inference_context=inference_context,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=kwargs,
        )

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        return self.language_model.set_input_tensor(input_tensor)

    def get_input_tensor(self):
        return self.language_model.get_input_tensor()

    def shared_embedding_or_output_weight(self) -> torch.Tensor:
        return self.language_model.shared_embedding_or_output_weight()
