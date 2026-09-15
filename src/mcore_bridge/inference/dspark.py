# Copyright (c) ModelScope Contributors. All rights reserved.
"""DeepSeek-V4.1 DSpark adapters for Megatron's dynamic inference API."""

from contextlib import contextmanager

import torch
from megatron.core.inference.communication_utils import broadcast_from_last_pipeline_stage
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher


@contextmanager
def _standard_mtp_compatibility(config):
    """Satisfy legacy MTP-only constructor validation without changing model semantics."""
    sentinel = object()
    original_num_layers = getattr(config, 'mtp_num_layers', sentinel)
    original_repeated = getattr(config, 'mtp_use_repeated_layer', sentinel)
    config.mtp_num_layers = max(getattr(config, 'mtp_num_layers', 0) or 0, 1)
    config.mtp_use_repeated_layer = True
    try:
        yield
    finally:
        if original_num_layers is sentinel:
            delattr(config, 'mtp_num_layers')
        else:
            config.mtp_num_layers = original_num_layers
        if original_repeated is sentinel:
            delattr(config, 'mtp_use_repeated_layer')
        else:
            config.mtp_use_repeated_layer = original_repeated


def _validate_dspark_speculation(model_config, num_speculative_tokens):
    if num_speculative_tokens <= 0:
        return
    block_size = getattr(model_config, 'dspark_block_size', 0)
    if not getattr(model_config, 'dspark_num_layers', None):
        raise ValueError('DSpark speculative decoding requires dspark_num_layers.')
    if not block_size or num_speculative_tokens > block_size:
        raise ValueError(
            f'num_speculative_tokens={num_speculative_tokens} must not exceed '
            f'dspark_block_size={block_size}.')
    if getattr(model_config, 'cuda_graph_impl', None) == 'local':
        raise ValueError('DSpark speculative decoding does not support local CUDA graphs yet.')


class DeepseekV41TextGenerationController(TextGenerationController):
    """Route Megatron's standard speculative loop through the parallel DSpark draft stack."""

    def __init__(self, inference_wrapped_model, tokenizer):
        model_config = inference_wrapped_model.model.config
        inference_config = inference_wrapped_model.inference_context.config
        self._uses_dspark = bool(getattr(model_config, 'dspark_num_layers', None))
        if self._uses_dspark:
            _validate_dspark_speculation(model_config, inference_config.num_speculative_tokens)
            # PR #7224 validates speculative decoding as standard serial MTP. DSpark is
            # separate, so adapt only while the upstream constructor initializes buffers.
            with _standard_mtp_compatibility(model_config):
                super().__init__(inference_wrapped_model, tokenizer)
            self._uses_dspark = True
            self.num_mtp_depths = 0
        else:
            super().__init__(inference_wrapped_model, tokenizer)

    def _compute_dspark_and_sample(self):
        context = self.inference_wrapped_model.inference_context
        active_request_count = context.total_request_count - context.paused_request_count
        speculative_tokens = None

        if self._is_last_pp_stage:
            compute_dspark = getattr(self._unwrapped_model, 'compute_dspark_speculative_tokens', None)
            if compute_dspark is None:
                raise RuntimeError(
                    'DSpark speculative decoding requires compute_dspark_speculative_tokens() '
                    'on the last pipeline stage.')
            if context._nvls_dispatcher:
                NVLSAllGatherVDispatcher.modify_real_token_count_for_mtp(
                    active_request_count * self.model_config.dspark_block_size)
            speculative_tokens = compute_dspark(
                next_token_ids=self._sampled_tokens_cuda[:active_request_count],
                accepted_token_counts=self._accepted_token_counts_per_request[:active_request_count],
                last_accepted_seq_indices=self._last_accepted_seq_indices,
                num_speculative_tokens=self.num_speculative_tokens,
                inference_context=context,
                sample_fn=self._sample_from_logits_2d,
            )
            expected_shape = (self.num_speculative_tokens, active_request_count)
            if tuple(speculative_tokens.shape) != expected_shape:
                raise RuntimeError(
                    f'DSpark returned speculative tokens with shape {tuple(speculative_tokens.shape)}; '
                    f'expected {expected_shape}.')

        if self.model_is_pipeline_parallel:
            speculative_tokens = broadcast_from_last_pipeline_stage(
                [self.num_speculative_tokens, active_request_count],
                dtype=torch.int64,
                tensor=speculative_tokens,
                pp_group=self.pp_group,
            )
        self._sampled_mtp_tokens_cuda[
            :self.num_speculative_tokens, :active_request_count
        ].copy_(speculative_tokens)

    def _compute_serial_mtp_and_sample(self):
        # The upstream event loop invokes this extension point after verification and KV
        # rewind. Reusing it avoids copying Megatron's large scheduling loop.
        if self._uses_dspark:
            return self._compute_dspark_and_sample()
        return super()._compute_serial_mtp_and_sample()


class DeepseekV41DynamicInferenceEngine(DynamicInferenceEngine):
    """Dynamic engine adapter that validates DSpark instead of serial-MTP depth."""

    def __init__(self, controller, context):
        model_config = controller.inference_wrapped_model.model.config
        if getattr(model_config, 'dspark_num_layers', None):
            _validate_dspark_speculation(model_config, context.config.num_speculative_tokens)
            with _standard_mtp_compatibility(model_config):
                super().__init__(controller, context)
        else:
            super().__init__(controller, context)
