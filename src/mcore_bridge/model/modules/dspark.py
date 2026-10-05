# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tensor-parallel building blocks for DeepSeek-V4.1 DSpark."""
import copy
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

try:
    from megatron.core.transformer.hyper_connection import SinglePassMHCState
except ImportError:
    # mHC (SinglePassMHCState) is a dev-only Megatron API; keep the import optional so the package
    # loads on stable releases. Only the DeepSeek-V4.1 mHC path below dereferences it at runtime.
    SinglePassMHCState = None
from torch import nn
from typing import Callable, Optional, Sequence


class DeepseekV41DSparkRMSNorm(nn.Module):
    """RMSNorm matching the fp32 accumulation used by the reference model."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))

    def forward(self, hidden_states: torch.Tensor):
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.square().mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (hidden_states * self.weight.float()).to(dtype)


class DeepseekV41DSparkInput(nn.Module):
    """Project target-layer states and construct the parallel draft-token block.

    DSpark receives target states in Megatron layout ``[s, b, targets * h]`` and
    runs its draft stack on ``[block, b, streams * h]``. The feature projection
    is row-parallel, while sequence-parallel sharding is deliberately disabled:
    the target states are already local to the caller's sequence partition.
    """

    def __init__(self, config):
        super().__init__()
        target_count = len(config.dspark_target_layer_ids or ())
        if target_count == 0:
            raise ValueError('DSpark input projection requires at least one target layer.')
        if config.dspark_block_size <= 0:
            raise ValueError('DSpark block size must be positive.')
        self.hidden_size = config.hidden_size
        self.num_streams = config.num_residual_streams
        self.sequence_parallel = config.sequence_parallel
        self.block_size = config.dspark_block_size
        self.noise_token_id = config.dspark_noise_token_id

        projection_config = copy.copy(config)
        projection_config.sequence_parallel = False
        self.main_proj = RowParallelLinear(
            config.hidden_size * target_count,
            config.hidden_size,
            config=projection_config,
            init_method=config.init_method,
            bias=False,
            input_is_parallel=False,
            skip_bias_add=False,
        )
        device = None if config.use_cpu_initialization else torch.cuda.current_device()
        self.main_norm = DeepseekV41DSparkRMSNorm(
            config.hidden_size,
            config.layernorm_epsilon,
            config.params_dtype,
            device=device,
        )

    def project_main_hidden(self, main_hidden: torch.Tensor):
        if main_hidden.ndim != 3:
            raise ValueError(f'DSpark main hidden states must be [s, b, targets*h], got {tuple(main_hidden.shape)}.')
        if self.sequence_parallel:
            main_hidden = gather_from_sequence_parallel_region(main_hidden)
        main_x, _ = self.main_proj(main_hidden)
        return self.main_norm(main_x)

    def build_draft_hidden(
        self,
        input_ids: torch.Tensor,
        embedding: Callable[[torch.Tensor], torch.Tensor],
    ):
        if input_ids.ndim != 1:
            raise ValueError(f'DSpark input_ids must be [b], got {tuple(input_ids.shape)}.')
        draft_input_ids = input_ids.new_full(
            (self.block_size, input_ids.shape[0]),
            self.noise_token_id,
        )
        draft_input_ids[0] = input_ids
        hidden_states = embedding(draft_input_ids)
        expected = (self.block_size, input_ids.shape[0], self.hidden_size)
        if tuple(hidden_states.shape) != expected:
            raise ValueError(f'DSpark embedding must return {expected}, got {tuple(hidden_states.shape)}.')
        hidden_states = hidden_states.unsqueeze(-2).expand(*hidden_states.shape[:-1], self.num_streams,
                                                           self.hidden_size)
        hidden_states = hidden_states.reshape(
            self.block_size,
            input_ids.shape[0],
            self.num_streams * self.hidden_size,
        )
        return hidden_states, draft_input_ids

    def forward(
        self,
        main_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        embedding: Callable[[torch.Tensor], torch.Tensor],
    ):
        if input_ids.ndim != 1 or input_ids.shape[0] != main_hidden.shape[1]:
            raise ValueError(f'DSpark input_ids must be [b] matching main hidden batch {main_hidden.shape[1]}, '
                             f'got {tuple(input_ids.shape)}.')
        main_x = self.project_main_hidden(main_hidden)
        hidden_states, draft_input_ids = self.build_draft_hidden(input_ids, embedding)
        return hidden_states, main_x, draft_input_ids


class DeepseekV41DSparkMarkovHead(nn.Module):
    """Low-rank Markov logit bias with vocab-row tensor parallelism."""

    def __init__(self, config):
        super().__init__()
        rank = config.dspark_markov_rank
        if rank is None or rank <= 0:
            raise ValueError('DSpark Markov rank must be positive.')
        self.vocab_size = config.padded_vocab_size
        self.embed = VocabParallelEmbedding(
            self.vocab_size,
            rank,
            init_method=config.init_method,
            config=config,
        )
        head_config = copy.copy(config)
        head_config.params_dtype = torch.float32
        self.head = ColumnParallelLinear(
            rank,
            self.vocab_size,
            config=head_config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
        )

    def forward(self, token_ids: torch.Tensor, gather_output: bool = True):
        markov_embed = self.embed(token_ids)
        logits, _ = self.head(markov_embed.float(), runtime_gather_output=gather_output)
        return logits, markov_embed


class DeepseekV41DSparkConfidenceHead(nn.Module):
    """FP32 acceptance-confidence projection from draft and Markov states."""

    def __init__(self, config):
        super().__init__()
        input_size = config.hidden_size + config.dspark_markov_rank
        device = None if config.use_cpu_initialization else torch.cuda.current_device()
        self.proj = nn.Linear(
            input_size,
            1,
            bias=False,
            dtype=torch.float32,
            device=device,
        )

    def forward(self, hidden_states: torch.Tensor, markov_embed: torch.Tensor):
        if hidden_states.shape[:-1] != markov_embed.shape[:-1]:
            raise ValueError('DSpark confidence inputs must have matching leading dimensions, got '
                             f'{tuple(hidden_states.shape)} and {tuple(markov_embed.shape)}.')
        hidden_states = torch.cat((hidden_states, markov_embed), dim=-1)
        return F.linear(hidden_states.float(), self.proj.weight).squeeze(-1)


def dspark_sample(logits: torch.Tensor, temperature: float = 0.0):
    """Sample one DSpark token, matching the reference Gumbel-max path."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / max(temperature, 1e-5)
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probabilities.div(torch.empty_like(probabilities).exponential_()).argmax(dim=-1)


@dataclass
class DeepseekV41DSparkVerification:
    """Verified prefix length and fallback/bonus token for each request."""

    accepted_lengths: torch.Tensor
    accepted_mask: torch.Tensor
    next_tokens: torch.Tensor


def verify_dspark_draft(
    draft_ids: torch.Tensor,
    target_ids: torch.Tensor,
    confidence_logits: Optional[torch.Tensor] = None,
    confidence_threshold: Optional[float] = None,
):
    """Verify proposals in strict prefix order against target-model tokens.

    ``draft_ids`` contains the committed seed followed by ``K`` proposals.
    ``target_ids`` contains ``K`` verification tokens and one bonus token.
    Confidence may shorten a proposal but cannot accept a target mismatch.
    """
    if draft_ids.ndim != 2 or target_ids.ndim != 2:
        raise ValueError('DSpark verification expects rank-2 token tensors.')
    block_size = draft_ids.shape[1] - 1
    if block_size <= 0 or target_ids.shape != (draft_ids.shape[0], block_size + 1):
        raise ValueError(f'Expected draft [b, K+1] and target [b, K+1], got '
                         f'{tuple(draft_ids.shape)} and {tuple(target_ids.shape)}.')
    accepted_mask = draft_ids[:, 1:].eq(target_ids[:, :block_size])
    if confidence_threshold is not None:
        if confidence_logits is None or confidence_logits.shape != accepted_mask.shape:
            raise ValueError('Confidence logits must have shape [b, K] when a threshold is set.')
        accepted_mask &= confidence_logits.sigmoid().ge(confidence_threshold)
    accepted_mask = accepted_mask.cumprod(dim=-1).bool()
    accepted_lengths = accepted_mask.sum(dim=-1)
    next_tokens = target_ids.gather(1, accepted_lengths.unsqueeze(-1)).squeeze(-1)
    return DeepseekV41DSparkVerification(accepted_lengths, accepted_mask, next_tokens)


@dataclass
class DeepseekV41DSparkState:
    """Per-forward target state passed explicitly to every DSpark attention layer."""

    main_hidden: torch.Tensor
    main_rotary_pos_emb: Optional[torch.Tensor] = None
    cache_slots: Optional[torch.Tensor] = None

    def attention_kwargs(self):
        return {
            'dspark_main_hidden': self.main_hidden,
            'dspark_main_rotary_pos_emb': self.main_rotary_pos_emb,
            'dspark_cache_slots': self.cache_slots,
        }

    def recompute_boundary_tensors(self):
        if self.main_rotary_pos_emb is None:
            return (self.main_hidden, )
        return self.main_hidden, self.main_rotary_pos_emb

    def save_for_recompute(self):
        has_rotary = self.main_rotary_pos_emb is not None
        cache_slots = self.cache_slots
        tensors = self.recompute_boundary_tensors()

        def restore(saved_tensors):
            return type(self)(
                main_hidden=saved_tensors[0],
                main_rotary_pos_emb=saved_tensors[1] if has_rotary else None,
                cache_slots=cache_slots,
            )

        return tensors, restore


class DeepseekV41DSparkOutput(nn.Module):
    """Final DSpark norm, main logits, Markov recurrence and confidence."""

    def __init__(self, config):
        super().__init__()
        device = None if config.use_cpu_initialization else torch.cuda.current_device()
        self.block_size = config.dspark_block_size
        self.norm = DeepseekV41DSparkRMSNorm(
            config.hidden_size,
            config.layernorm_epsilon,
            config.params_dtype,
            device=device,
        )
        self.markov_head = DeepseekV41DSparkMarkovHead(config)
        self.confidence_head = DeepseekV41DSparkConfidenceHead(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        output_layer: Callable,
        temperature: float = 0.0,
        sample_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        if hidden_states.ndim != 3 or hidden_states.shape[0] != self.block_size:
            raise ValueError(f'DSpark output hidden states must be [block={self.block_size}, b, h], '
                             f'got {tuple(hidden_states.shape)}.')
        if input_ids.ndim != 1 or input_ids.shape[0] != hidden_states.shape[1]:
            raise ValueError(f'DSpark output input_ids must be [b] matching hidden batch {hidden_states.shape[1]}, '
                             f'got {tuple(input_ids.shape)}.')

        base_logits, _ = output_layer(self.norm(hidden_states), runtime_gather_output=True)
        output_ids = input_ids.new_empty((input_ids.shape[0], self.block_size + 1))
        output_ids[:, 0] = input_ids
        markov_embeds = []
        logits = []
        for index in range(self.block_size):
            logits_bias, markov_embed = self.markov_head(output_ids[:, index])
            step_logits = base_logits[index] + logits_bias
            logits.append(step_logits)
            markov_embeds.append(markov_embed)
            output_ids[:, index + 1] = (
                sample_fn(step_logits) if sample_fn is not None else dspark_sample(step_logits, temperature))
        logits = torch.stack(logits, dim=0)
        markov_embed = torch.stack(markov_embeds, dim=0)
        confidence = self.confidence_head(hidden_states, markov_embed)
        return output_ids, logits.transpose(0, 1).contiguous(), confidence.transpose(0, 1).contiguous()


class DeepseekV41DSparkStack(nn.Module):
    """Orchestrate the dedicated DSpark layers around the TP input/output modules.

    The supplied layers must be mHC-enabled TransformerLayer instances whose
    attention accepts ``dspark_main_hidden`` and ``dspark_main_rotary_pos_emb``.
    Keeping this stack separate prevents V4.1 checkpoint layers under ``mtp.*``
    from entering Megatron's serial MultiTokenPredictionBlock path.
    """

    def __init__(self, config, layers: Sequence[nn.Module]):
        super().__init__()
        if len(layers) != config.dspark_num_layers:
            raise ValueError(f'DSpark requires {config.dspark_num_layers} layers, got {len(layers)}.')
        if not config.mhc_single_pass:
            raise ValueError('DeepSeek-V4.1 DSpark requires single-pass mHC.')
        self.config = config
        self.input = DeepseekV41DSparkInput(config)
        self.layers = nn.ModuleList(layers)
        self.output = DeepseekV41DSparkOutput(config)
        self._request_cache_slots = {}

    def reset_cache(self):
        self._request_cache_slots.clear()
        for layer in self.layers:
            reset = getattr(layer.self_attention, 'reset_dspark_cache', None)
            if reset is not None:
                reset()

    def resolve_cache_slots(self, request_ids: torch.Tensor, live_request_ids: Optional[torch.Tensor] = None):
        """Map scheduler request IDs to stable rows in every DSpark ring cache."""
        request_ids_cpu = request_ids.detach().to(device='cpu', dtype=torch.long).tolist()
        if live_request_ids is not None:
            live_ids = set(live_request_ids.detach().to(device='cpu', dtype=torch.long).tolist())
            self._request_cache_slots = {
                request_id: slot
                for request_id, slot in self._request_cache_slots.items() if request_id in live_ids
            }
        occupied = set(self._request_cache_slots.values())
        for request_id in request_ids_cpu:
            if request_id in self._request_cache_slots:
                continue
            slot = 0
            while slot in occupied:
                slot += 1
            self._request_cache_slots[request_id] = slot
            occupied.add(slot)
        return torch.tensor(
            [self._request_cache_slots[request_id] for request_id in request_ids_cpu],
            dtype=torch.long,
            device=request_ids.device,
        )

    def _seed_main_kv(
        self,
        main_hidden: torch.Tensor,
        main_rotary_pos_emb: Optional[torch.Tensor],
        inference_context=None,
        cache_slots: Optional[torch.Tensor] = None,
    ):
        for layer in self.layers:
            attention = layer.self_attention
            if not hasattr(attention, 'prefill_dspark'):
                raise TypeError(f'{type(attention).__name__} does not implement prefill_dspark().')
            attention.prefill_dspark(
                main_hidden,
                rotary_pos_emb=main_rotary_pos_emb,
                inference_context=inference_context,
                cache_slots=cache_slots,
            )

    def update_main_cache(
        self,
        main_hidden: torch.Tensor,
        main_rotary_pos_emb: torch.Tensor,
        *,
        start_pos,
        cache_slots: torch.Tensor,
        inference_context=None,
    ):
        main_x = self.input.project_main_hidden(main_hidden)
        for layer in self.layers:
            layer.self_attention.prefill_dspark(
                main_x,
                rotary_pos_emb=main_rotary_pos_emb,
                inference_context=inference_context,
                start_pos=start_pos,
                cache_slots=cache_slots,
            )
        return main_x

    def forward(
        self,
        main_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        embedding: Callable[[torch.Tensor], torch.Tensor],
        output_layer: Callable,
        *,
        start_pos: int,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        main_rotary_pos_emb: Optional[torch.Tensor] = None,
        inference_context=None,
        temperature: float = 0.0,
        sample_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        cache_slots: Optional[torch.Tensor] = None,
        prefill_only: Optional[bool] = None,
    ):
        hidden_states, main_x, _ = self.input(main_hidden, input_ids, embedding)
        if prefill_only is None:
            prefill_only = bool(torch.as_tensor(start_pos).eq(0).all().item())
        if prefill_only:
            self._seed_main_kv(main_x, main_rotary_pos_emb, inference_context, cache_slots=cache_slots)
            return None

        mhc_state = SinglePassMHCState()
        dspark_state = DeepseekV41DSparkState(main_x, main_rotary_pos_emb, cache_slots)
        for layer in self.layers:
            hidden_states, _ = layer(
                hidden_states,
                attention_mask=None,
                rotary_pos_emb=rotary_pos_emb,
                inference_context=inference_context,
                sequence_len_offset=start_pos,
                cross_layer_state=dspark_state,
                mhc_state=mhc_state,
            )
        hidden_states = mhc_state.contract(
            hidden_states,
            self.config.num_residual_streams,
            use_fused=self.config.use_fused_mhc,
        )
        return self.output(
            hidden_states,
            input_ids,
            output_layer,
            temperature=temperature,
            sample_fn=sample_fn,
        )
