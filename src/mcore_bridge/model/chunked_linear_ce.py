# Copyright (c) ModelScope Contributors. All rights reserved.
"""Memory-bounded linear cross entropy for long-context Megatron training.

The implementation computes logits only for supervised-token chunks and
recomputes those chunks in backward.  It therefore avoids keeping a full
``[local_tokens, vocab_partition]`` logits tensor alive.  Context parallelism
is naturally supported because each CP rank receives matching local hidden
states and labels; tensor-parallel vocabulary shards are handled with the
usual max/sum reductions.
"""

from __future__ import annotations

import os

import torch


# cuBLAS handles are scoped to the CUDA device/stream and survive across
# optimizer steps. The warmup below avoids recreating the handle on every
# backward call when a long-context job has little cudaMalloc headroom.
_CUBLAS_WARMED_STREAMS: set[tuple[int, int]] = set()


def _get_setting(name: str, legacy_name: str) -> str:
    """Read the canonical setting, falling back to the old experiment name."""

    if name in os.environ:
        return os.environ[name]
    return os.environ.get(legacy_name, '')


def _chunked_linear_ce_debug_enabled() -> bool:
    return _get_setting('CHUNKED_LINEAR_CE_DEBUG', 'LINEAR_CE_DEBUG').strip().lower() not in {
        '',
        '0',
        'false',
        'no',
        'off',
    }


def _chunked_linear_ce_compare_native_enabled() -> bool:
    return _get_setting('CHUNKED_LINEAR_CE_COMPARE_NATIVE', 'LINEAR_CE_COMPARE_NATIVE').strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }


def _chunked_linear_ce_cublas_warmup_every_backward_enabled() -> bool:
    """Retain per-backward cache release for diagnostics."""

    return _get_setting('CHUNKED_LINEAR_CE_CUBLAS_WARMUP_EVERY_BACKWARD',
                        'LINEAR_CE_CUBLAS_WARMUP_EVERY_BACKWARD').strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }


def _chunked_linear_ce_pad_tail_chunk_enabled() -> bool:
    """Use a fixed GEMM row count for the final supervised-token chunk."""

    return _get_setting('CHUNKED_LINEAR_CE_PAD_TAIL_CHUNK', 'LINEAR_CE_PAD_TAIL_CHUNK').strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }


def parse_chunked_linear_ce_tail_pad_multiple(chunk_size: int) -> int:
    """Return the tail-row bucket size (zero leaves the tail dynamic).

    ``CHUNKED_LINEAR_CE_PAD_TAIL_CHUNK=true`` pads every tail to
    ``chunk_size``. A positive ``CHUNKED_LINEAR_CE_TAIL_PAD_MULTIPLE``
    selects finer buckets. The old ``LINEAR_CE_*`` names remain accepted.
    """

    raw_value = _get_setting('CHUNKED_LINEAR_CE_TAIL_PAD_MULTIPLE',
                             'LINEAR_CE_TAIL_PAD_MULTIPLE').strip().lower()
    if raw_value in {'', '0', 'false', 'none', 'off'}:
        return chunk_size if _chunked_linear_ce_pad_tail_chunk_enabled() else 0
    try:
        multiple = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            'CHUNKED_LINEAR_CE_TAIL_PAD_MULTIPLE must be an integer row count, '
            f'e.g. 64 or 128. Got: {raw_value!r}'
        ) from exc
    if multiple <= 0 or multiple > chunk_size or chunk_size % multiple:
        raise ValueError(
            'CHUNKED_LINEAR_CE_TAIL_PAD_MULTIPLE must be a positive divisor of '
            f'CHUNKED_LINEAR_CE_CHUNK_SIZE={chunk_size}. Got: {multiple}'
        )
    return multiple


def parse_linear_ce_tail_pad_multiple(chunk_size: int) -> int:
    """Backward-compatible alias for the original experiment helper."""

    return parse_chunked_linear_ce_tail_pad_multiple(chunk_size)


def _tail_target_rows(valid_rows: int, chunk_size: int, pad_multiple: int) -> int:
    if not pad_multiple or valid_rows >= chunk_size:
        return valid_rows
    return min(chunk_size, ((valid_rows + pad_multiple - 1) // pad_multiple) * pad_multiple)


def _pad_tail_rows(
    hidden_chunk: torch.Tensor,
    target: torch.Tensor,
    target_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a short chunk with numerically inert rows up to ``target_rows``."""

    valid_rows = hidden_chunk.shape[0]
    if valid_rows >= target_rows:
        return hidden_chunk, target
    pad_rows = target_rows - valid_rows
    hidden_chunk = torch.nn.functional.pad(hidden_chunk, (0, 0, 0, pad_rows))
    # Token zero is a valid global target on exactly one TP rank.  Its forward
    # loss is discarded, and its backward scale is forced to zero below, so it
    # contributes neither grad-hidden nor grad-weight on any TP rank.
    target = torch.nn.functional.pad(target, (0, pad_rows), value=0)
    return hidden_chunk, target


def _debug_cuda_memory(tag: str, tensor: torch.Tensor) -> None:
    if not _chunked_linear_ce_debug_enabled() or not tensor.is_cuda:
        return
    free_bytes, total_bytes = torch.cuda.mem_get_info(tensor.device)
    allocated_bytes = torch.cuda.memory_allocated(tensor.device)
    reserved_bytes = torch.cuda.memory_reserved(tensor.device)
    gib = 1024**3
    print(
        '[CHUNKED_LINEAR_CE] '
        f'{tag}: free={free_bytes / gib:.3f} GiB, '
        f'total={total_bytes / gib:.3f} GiB, '
        f'allocated={allocated_bytes / gib:.3f} GiB, '
        f'reserved={reserved_bytes / gib:.3f} GiB',
        flush=True,
    )


def _parse_chunk_size(raw_value) -> int:
    raw_text = str(raw_value).strip().lower()
    if raw_text in {'', '0', 'false', 'none', 'off'}:
        return 0
    multiplier = 1
    if raw_text.endswith('k'):
        multiplier = 1024
        raw_text = raw_text[:-1]
    elif raw_text.endswith('m'):
        multiplier = 1024 * 1024
        raw_text = raw_text[:-1]
    try:
        chunk_size = int(float(raw_text) * multiplier)
    except ValueError as exc:
        raise ValueError(
            'CHUNKED_LINEAR_CE_CHUNK_SIZE must be an integer token count, '
            f'e.g. 2048 or 2k. Got: {raw_value!r}'
        ) from exc
    if chunk_size < 0:
        raise ValueError(f'CHUNKED_LINEAR_CE_CHUNK_SIZE must be >= 0. Got: {chunk_size}')
    return chunk_size


def parse_chunked_linear_ce_chunk_size(config=None) -> int:
    """Read the configured token chunk size; zero disables the feature.

    ``CHUNKED_LINEAR_CE_CHUNK_SIZE`` is the canonical environment variable.
    ``LINEAR_CE_CHUNK_SIZE`` is accepted for compatibility with the original
    V4 experiments. A config value is used when no environment variable is
    present, allowing higher-level launchers to expose the option directly.
    """

    if 'CHUNKED_LINEAR_CE_CHUNK_SIZE' in os.environ:
        raw_value = os.environ['CHUNKED_LINEAR_CE_CHUNK_SIZE']
    elif 'LINEAR_CE_CHUNK_SIZE' in os.environ:
        raw_value = os.environ['LINEAR_CE_CHUNK_SIZE']
    else:
        raw_value = getattr(config, 'chunked_linear_ce_chunk_size', 0)
    return _parse_chunk_size(raw_value)


def parse_linear_ce_chunk_size(config=None) -> int:
    """Backward-compatible alias for older launchers."""

    return parse_chunked_linear_ce_chunk_size(config)


def _tp_group_size(tp_group) -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    if tp_group is None:
        return 1
    return torch.distributed.get_world_size(tp_group)


def _tp_all_reduce(
    tensor: torch.Tensor,
    op: torch.distributed.ReduceOp,
    tp_group,
) -> torch.Tensor:
    if _tp_group_size(tp_group) > 1:
        torch.distributed.all_reduce(tensor, op=op, group=tp_group)
    return tensor


class ChunkedLinearCrossEntropyLoss(torch.autograd.Function):
    """Compute CE only for supervised tokens without full-sequence logits."""

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        output_weight,
        labels,
        tp_group,
        vocab_start_index,
        chunk_size,
        reduce_grad_input,
        tail_pad_multiple,
    ):
        if labels.dim() != 2:
            raise ValueError(f'labels must be [batch, sequence], got {tuple(labels.shape)}')
        if hidden_states.dim() != 3:
            raise ValueError(
                f'hidden_states must be [sequence, batch, hidden], got {tuple(hidden_states.shape)}'
            )
        seq_len, batch_size, hidden_size = hidden_states.shape
        if labels.shape != (batch_size, seq_len):
            raise ValueError(
                'labels shape must match hidden states as [batch, sequence]. '
                f'Got labels={tuple(labels.shape)}, hidden_states={tuple(hidden_states.shape)}'
            )
        if chunk_size <= 0:
            raise ValueError(f'chunk_size must be > 0. Got: {chunk_size}')

        labels_t = labels.transpose(0, 1).contiguous()
        hidden_flat = hidden_states.contiguous().view(seq_len * batch_size, hidden_size)
        target_flat = labels_t.view(-1)
        supervised_indices = torch.nonzero(target_flat != -100, as_tuple=False).flatten()
        partition_vocab_size = output_weight.shape[0]
        vocab_end_index = vocab_start_index + partition_vocab_size
        if supervised_indices.numel() == 0:
            # Preserve a differentiable zero for an all-padding/all-ignored batch.
            # Such a batch is valid after packing or loss-mask filtering, and a
            # detached ``torch.zeros`` would make the caller's backward fail.
            losses_flat = hidden_flat[:, 0].float() * 0.0
        else:
            losses_flat = torch.zeros(
                (seq_len * batch_size,), dtype=torch.float32, device=hidden_states.device
            )

        for chunk_start in range(0, supervised_indices.numel(), chunk_size):
            chunk_end = min(supervised_indices.numel(), chunk_start + chunk_size)
            token_indices = supervised_indices[chunk_start:chunk_end]
            valid_count = token_indices.numel()
            hidden_chunk = hidden_flat.index_select(0, token_indices)
            target = target_flat.index_select(0, token_indices)
            target_count = _tail_target_rows(
                valid_count, chunk_size, tail_pad_multiple
            )
            if target_count > valid_count:
                hidden_chunk, target = _pad_tail_rows(hidden_chunk, target, target_count)
            logits = torch.matmul(hidden_chunk, output_weight.t()).float()

            local_max = logits.max(dim=-1).values
            global_max = _tp_all_reduce(local_max, torch.distributed.ReduceOp.MAX, tp_group)
            exp_logits = torch.exp(logits - global_max.unsqueeze(-1))
            global_sum = _tp_all_reduce(
                exp_logits.sum(dim=-1), torch.distributed.ReduceOp.SUM, tp_group
            )

            target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
            local_target = (target - vocab_start_index).masked_fill(target_mask, 0)
            target_logits = torch.gather(
                logits, dim=-1, index=local_target.unsqueeze(-1)
            ).squeeze(-1)
            target_logits = target_logits.masked_fill(target_mask, 0.0)
            target_logits = _tp_all_reduce(
                target_logits, torch.distributed.ReduceOp.SUM, tp_group
            )

            chunk_loss = torch.log(global_sum) + global_max - target_logits
            losses_flat.index_copy_(0, token_indices, chunk_loss[:valid_count])

        ctx.save_for_backward(hidden_states, output_weight, target_flat, supervised_indices)
        ctx.tp_group = tp_group
        ctx.vocab_start_index = vocab_start_index
        ctx.chunk_size = chunk_size
        ctx.reduce_grad_input = reduce_grad_input
        ctx.tail_pad_multiple = tail_pad_multiple
        return losses_flat.view(seq_len, batch_size).transpose(0, 1).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, output_weight, target_flat, supervised_indices = ctx.saved_tensors
        tp_group = ctx.tp_group
        vocab_start_index = ctx.vocab_start_index
        chunk_size = ctx.chunk_size
        reduce_grad_input = ctx.reduce_grad_input
        tail_pad_multiple = ctx.tail_pad_multiple
        partition_vocab_size = output_weight.shape[0]
        vocab_end_index = vocab_start_index + partition_vocab_size
        seq_len, batch_size, hidden_size = hidden_states.shape

        hidden_flat = hidden_states.contiguous().view(seq_len * batch_size, hidden_size)
        grad_output_flat = grad_output.transpose(0, 1).contiguous().view(-1).float()

        # The first GEMM reached from a Python autograd backward can lazily create a
        # cuBLAS handle.  On the final PP stage, allocating the full hidden/weight
        # gradient buffers first leaves too little non-PyTorch CUDA memory for that
        # handle, producing CUBLAS_STATUS_ALLOC_FAILED even though the chunk itself
        # is bounded.  Exercise the exact BF16 GEMM path before those two large
        # buffers are allocated, then return temporary allocator blocks to CUDA.
        warmup_key = None
        warmup_every_backward = _chunked_linear_ce_cublas_warmup_every_backward_enabled()
        if hidden_states.is_cuda:
            current_stream = torch.cuda.current_stream(hidden_states.device)
            warmup_key = (
                int(hidden_states.device.index),
                int(current_stream.cuda_stream),
            )
        needs_cublas_warmup = (
            warmup_key is not None
            and supervised_indices.numel() > 0
            and (warmup_every_backward or warmup_key not in _CUBLAS_WARMED_STREAMS)
        )
        if needs_cublas_warmup:
            warmup_target_count = min(chunk_size, 512)
            warmup_count = min(supervised_indices.numel(), warmup_target_count)
            warmup_indices = supervised_indices[:warmup_count]
            warmup_hidden = hidden_flat.index_select(0, warmup_indices)
            warmup_padded_count = _tail_target_rows(
                warmup_count, warmup_target_count, tail_pad_multiple
            )
            if warmup_padded_count > warmup_count:
                warmup_hidden = torch.nn.functional.pad(
                    warmup_hidden,
                    (0, 0, 0, warmup_padded_count - warmup_count),
                )
            _debug_cuda_memory('backward pre-cache-release', hidden_states)
            # cuBLAS allocates its handle outside PyTorch's caching allocator.
            # A worker can have substantial cached memory but little visible to
            # cudaMalloc, so release cache before creating the handle.
            torch.cuda.synchronize(hidden_states.device)
            torch.cuda.empty_cache()
            _debug_cuda_memory('backward post-cache-release/pre-cuBLAS-warmup', hidden_states)
            warmup_logits = torch.matmul(warmup_hidden, output_weight.t())
            torch.cuda.synchronize(hidden_states.device)
            _CUBLAS_WARMED_STREAMS.add(warmup_key)
            del warmup_logits, warmup_hidden, warmup_indices
            torch.cuda.empty_cache()
            _debug_cuda_memory('backward post-cuBLAS-warmup', hidden_states)
            if _chunked_linear_ce_debug_enabled():
                print(
                    '[CHUNKED_LINEAR_CE] '
                    f'cuBLAS warmup complete: device={warmup_key[0]} '
                    f'stream={warmup_key[1]} '
                    f'policy={"every-backward" if warmup_every_backward else "once-per-stream"}',
                    flush=True,
                )

        grad_hidden_flat = torch.zeros_like(hidden_flat) if ctx.needs_input_grad[0] else None
        grad_weight = torch.zeros_like(output_weight) if ctx.needs_input_grad[1] else None
        _debug_cuda_memory('backward post-grad-buffer-allocation', hidden_states)

        for chunk_start in range(0, supervised_indices.numel(), chunk_size):
            chunk_end = min(supervised_indices.numel(), chunk_start + chunk_size)
            token_indices = supervised_indices[chunk_start:chunk_end]
            valid_count = token_indices.numel()
            hidden_chunk = hidden_flat.index_select(0, token_indices)
            target = target_flat.index_select(0, token_indices)
            target_count = _tail_target_rows(
                valid_count, chunk_size, tail_pad_multiple
            )
            if target_count > valid_count:
                hidden_chunk, target = _pad_tail_rows(hidden_chunk, target, target_count)
            logits = torch.matmul(hidden_chunk, output_weight.t()).float()

            local_max = logits.max(dim=-1).values
            global_max = _tp_all_reduce(local_max, torch.distributed.ReduceOp.MAX, tp_group)
            exp_logits = torch.exp(logits - global_max.unsqueeze(-1))
            global_sum = _tp_all_reduce(
                exp_logits.sum(dim=-1), torch.distributed.ReduceOp.SUM, tp_group
            )
            grad_logits = exp_logits / global_sum.unsqueeze(-1)

            target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
            local_target = (target - vocab_start_index).masked_fill(target_mask, 0)
            subtract = (~target_mask).to(dtype=grad_logits.dtype).unsqueeze(-1)
            grad_logits.scatter_add_(dim=-1, index=local_target.unsqueeze(-1), src=-subtract)
            grad_scale = grad_output_flat.index_select(0, token_indices)
            if target_count > valid_count:
                grad_scale = torch.nn.functional.pad(
                    grad_scale,
                    (0, target_count - valid_count),
                )
            grad_logits.mul_(grad_scale.unsqueeze(-1))

            # The next two GEMMs do not need the FP32 logits or exp buffers,
            # and model-dtype grad-logits are sufficient for model-dtype
            # gradients. This avoids materializing a second full-vocab buffer
            # for every supervised-token chunk.
            del logits, exp_logits, local_max, global_max, global_sum
            grad_logits_model_dtype = grad_logits.to(dtype=output_weight.dtype)

            if grad_hidden_flat is not None:
                grad_hidden_chunk = torch.matmul(grad_logits_model_dtype, output_weight)
                if reduce_grad_input:
                    _tp_all_reduce(
                        grad_hidden_chunk, torch.distributed.ReduceOp.SUM, tp_group
                    )
                grad_hidden_flat.index_copy_(
                    0,
                    token_indices,
                    grad_hidden_chunk[:valid_count].to(dtype=hidden_states.dtype),
                )

            if grad_weight is not None:
                grad_weight.addmm_(grad_logits_model_dtype.t(), hidden_chunk)

            if chunk_start == 0:
                _debug_cuda_memory('backward post-first-chunk', hidden_states)

        grad_hidden = (
            grad_hidden_flat.view(seq_len, batch_size, hidden_size)
            if grad_hidden_flat is not None
            else None
        )
        return grad_hidden, grad_weight, None, None, None, None, None, None


# Keep the shorter name used by the initial prototype for downstream launchers.
ChunkedLinearCrossEntropy = ChunkedLinearCrossEntropyLoss


def chunked_linear_cross_entropy_loss(
    model,
    hidden_states: torch.Tensor,
    output_weight: torch.Tensor | None,
    labels: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Return per-token CE while respecting Megatron TP/SP layout."""

    if output_weight is None:
        output_weight = model.output_layer.weight
    if output_weight is None:
        raise ValueError('Unable to locate output layer weight for CHUNKED_LINEAR_CE_CHUNK_SIZE.')

    if getattr(model.output_layer, 'sequence_parallel', False):
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

        hidden_states = gather_from_sequence_parallel_region(
            hidden_states,
            tensor_parallel_output_grad=True,
            group=model.pg_collection.tp,
        )
        reduce_grad_input = False
    else:
        reduce_grad_input = _tp_group_size(model.pg_collection.tp) > 1

    vocab_start_index = (
        torch.distributed.get_rank(model.pg_collection.tp) * output_weight.shape[0]
        if _tp_group_size(model.pg_collection.tp) > 1
        else 0
    )

    # Record the exact amount of output-layer GEMM work once per rank. Unlike
    # native LM heads, this implementation projects only labels != -100, so a
    # positions-based FLOPs estimate can substantially overcount sparse-SFT runs.
    shape_call = int(getattr(model, '_chunked_linear_ce_shape_calls', 0)) + 1
    model._chunked_linear_ce_shape_calls = shape_call
    log_every_shape = _get_setting('CHUNKED_LINEAR_CE_LOG_SHAPE_EVERY_CALL',
                                   'LINEAR_CE_LOG_SHAPE_EVERY_CALL').strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }
    if (_chunked_linear_ce_debug_enabled() or log_every_shape) and (shape_call == 1 or log_every_shape):
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        supervised_count = int((labels != -100).sum().item())
        print(
            '[CHUNKED_LINEAR_CE_SHAPE] '
            f'rank={rank} call={shape_call} local_rows={labels.numel()} supervised={supervised_count} '
            f'hidden={hidden_states.shape[-1]} vocab_partition={output_weight.shape[0]} '
            f'chunk_size={chunk_size} '
            f'tail_pad_multiple={parse_chunked_linear_ce_tail_pad_multiple(chunk_size)}',
            flush=True,
        )

    # Diagnostic-only exact-path comparison.  Compute native Megatron CE from
    # the same hidden states/weight/labels without building an autograd graph,
    # release the full logits, and then run the memory-bounded implementation.
    # This mode is intended for short parity probes; production jobs leave it
    # disabled and never materialize native logits.
    native_loss = None
    if _chunked_linear_ce_compare_native_enabled():
        with torch.no_grad():
            native_logits = model._forward_output_layer(
                hidden_states,
                weight=output_weight,
                runtime_gather_output=False,
            )
            native_loss = model.compute_language_model_loss(labels, native_logits).float()
        if hidden_states.is_cuda:
            torch.cuda.synchronize(hidden_states.device)
        del native_logits
        if hidden_states.is_cuda:
            torch.cuda.empty_cache()

    chunked_loss = ChunkedLinearCrossEntropyLoss.apply(
        hidden_states,
        output_weight,
        labels,
        model.pg_collection.tp,
        vocab_start_index,
        chunk_size,
        reduce_grad_input,
        parse_chunked_linear_ce_tail_pad_multiple(chunk_size),
    )

    if native_loss is not None:
        supervised_mask = labels != -100
        supervised_count = int(supervised_mask.sum().item())
        call_index = int(getattr(model, '_chunked_linear_ce_compare_native_calls', 0)) + 1
        model._chunked_linear_ce_compare_native_calls = call_index
        if supervised_count:
            native_values = native_loss[supervised_mask]
            chunked_values = chunked_loss.detach().float()[supervised_mask]
            abs_diff = (native_values - chunked_values).abs()
            stats = {
                'max_abs': abs_diff.max().item(),
                'mean_abs': abs_diff.mean().item(),
                'native_mean': native_values.mean().item(),
                'chunked_mean': chunked_values.mean().item(),
            }
        else:
            stats = {
                'max_abs': 0.0,
                'mean_abs': 0.0,
                'native_mean': 0.0,
                'chunked_mean': 0.0,
            }
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        print(
            '[CHUNKED_LINEAR_CE_COMPARE] '
            f'rank={rank} call={call_index} supervised={supervised_count} '
            f'chunk_size={chunk_size} max_abs={stats["max_abs"]:.9g} '
            f'mean_abs={stats["mean_abs"]:.9g} '
            f'native_mean={stats["native_mean"]:.9g} '
            f'chunked_mean={stats["chunked_mean"]:.9g}',
            flush=True,
        )
        del native_loss

    return chunked_loss
