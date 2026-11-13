# Copyright (c) ModelScope Contributors. All rights reserved.
"""Runtime contracts for the upstream-MTP adaptation layer.

Builds a tiny dense GPT with MTP through the normal mcore-bridge loader and
locks the behaviours that regress silently: per-depth loss/gradient under
independent and repeated weights, packed sequences at depth>=2, ``-100``
masking, precomputed (multimodal) decoder inputs, extras delivery through
recompute, and the checkpoint protocols for padding masks / dict RoPE.

Inputs follow the model contract ``[B, S]`` and ``S`` is much larger than the
prediction depth -- with ``S <= depth`` the rolled loss mask goes all-zero and
deeper heads produce no loss/gradient while a weak ``any(grad)`` assertion
would still pass.
"""
import os
import pytest
import torch
import uuid
from contextlib import contextmanager
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

from mcore_bridge.config import ModelConfig
from mcore_bridge.config.parser import hf_to_mcore_config
from mcore_bridge.model.mtp import MultiTokenPredictionLayer, get_physical_mtp_count, roll_single
from mcore_bridge.model.register import get_mcore_model

VOCAB = 128
HIDDEN = 64
B, S = 2, 16  # S >> max depth (2) so every depth keeps valid tokens after rolling

# A forward pass makes Megatron's attention-backend selection export these
# globally; snapshot/restore them so this file does not leak backend state
# into other test files run in the same process.
_NVTE_ENV = ('NVTE_FLASH_ATTN', 'NVTE_FUSED_ATTN', 'NVTE_UNFUSED_ATTN')


@pytest.fixture(scope='session', autouse=True)
def _distributed_session():
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@contextmanager
def _parallel_context(tp=1, pp=1):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required')
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size != tp * pp:
        pytest.skip(f'requires world size {tp * pp}')
    saved_env = {k: os.environ.get(k) for k in _NVTE_ENV}
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    if world_size == 1:
        init_method = f'file:///tmp/mtp-runtime-{uuid.uuid4().hex}'
        torch.distributed.init_process_group('nccl', init_method=init_method, rank=0, world_size=1)
    elif not torch.distributed.is_initialized():
        torch.distributed.init_process_group('nccl')
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp)
    model_parallel_cuda_manual_seed(123)
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()
        from mcore_bridge.bridge import gpt_bridge
        gpt_bridge.EP_PP_GROUP = gpt_bridge.EP_PP_RANK = gpt_bridge.EP_PP_SIZE = None
        if world_size == 1:
            torch.distributed.destroy_process_group()
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _reset_mtp_tracker():
    """MTPLossLoggingHelper.tracker is module-global and lazily sized on first
    write; clear it so models with different depths do not clash."""
    MTPLossLoggingHelper.tracker.clear()


def _tracked_mtp_losses():
    """Per-depth MTP losses accumulated by the loss helpers during forward.
    The tracker key moved across versions: ``values`` (<=0.18),
    ``loss_values`` (0.19), ``loss_sums`` (dev -- already normalized unless
    calculate_per_token_loss, in which case divide by num_tokens)."""
    tracker = MTPLossLoggingHelper.tracker
    values = tracker.get('values')
    if values is None:
        values = tracker.get('loss_values')
    if values is None:
        values = tracker.get('loss_sums')
        if values is not None and tracker.get('calculate_per_token_loss'):
            values = values / tracker['num_tokens'].clamp(min=1)
    return None if values is None else values.detach().clone()


def _build(depth, recompute=False, repeated=False, bf16=False, hidden=HIDDEN, heads=4, tied=False, detach_heads=False):
    transformers = pytest.importorskip('transformers')
    hf = transformers.LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=hidden,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        max_position_embeddings=128,
        tie_word_embeddings=tied,
    )
    values = hf_to_mcore_config(hf)
    values.update(
        params_dtype=torch.float32,
        pipeline_dtype=torch.float32,
        perform_initialization=True,
        use_cpu_initialization=False,
        mtp_num_layers=depth,
        # determinism for eager-vs-recompute and packed-vs-nonpacked parity
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    if bf16:
        values.update(params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16, bf16=True)
    if repeated:
        values['mtp_shared_weights'] = True
    if detach_heads:
        values['mtp_detach_heads'] = True
    if recompute:
        values['recompute_granularity'] = 'full'
        values['recompute_method'] = 'uniform'
        values['recompute_num_layers'] = 1
    _reset_mtp_tracker()
    return get_mcore_model(ModelConfig(**values))[0].cuda()


def _inputs(batch=B, seq=S, device='cuda'):
    """[B, S] inputs -- the model contract (LanguageModelEmbedding transposes
    [b s h] -> [s b h] internally)."""
    ids = torch.randint(1, VOCAB, (batch, seq), device=device)
    pos = torch.arange(seq, device=device).unsqueeze(0).expand(batch, seq).contiguous()
    return ids, pos


def _run(model, ids, pos, labels, backward=True):
    _reset_mtp_tracker()
    out = model(input_ids=ids, position_ids=pos, attention_mask=None, labels=labels)
    loss = out if torch.is_tensor(out) else out[0]
    loss = loss.float().mean() if loss.dim() > 0 else loss.float()
    depth_losses = _tracked_mtp_losses()
    if not backward:
        return loss.item(), depth_losses, None
    model.zero_grad()
    loss.backward()
    per_layer = _mtp_layer_grad_sums(model)
    return loss.item(), depth_losses, per_layer


def _mtp_layer_grad_sums(model):
    """{physical_layer_index: sum |grad|} over each MTP layer's parameters."""
    sums = {}
    for name, p in model.named_parameters():
        if not name.startswith('mtp.layers.'):
            continue
        idx = int(name.split('.')[2])
        if p.grad is not None:
            sums[idx] = sums.get(idx, 0.0) + p.grad.abs().sum().item()
    return sums


def _assert_every_depth_active(depth, depth_losses, grad_sums, repeated):
    """Every logical depth must produce loss and gradient."""
    assert depth_losses is not None, 'no per-depth MTP losses were tracked'
    assert depth_losses.numel() == depth, f'expected {depth} tracked depths, got {depth_losses.numel()}'
    for d in range(depth):
        val = depth_losses[d].item()
        assert torch.isfinite(depth_losses[d]), f'depth {d + 1} loss not finite: {val}'
        assert val > 0, f'depth {d + 1} produced zero loss (mask rolled empty?)'
    n_phys = 1 if repeated else depth
    assert sorted(grad_sums) == list(range(n_phys)), f'unexpected physical layers with grads: {grad_sums}'
    for idx, total in grad_sums.items():
        assert total > 0, f'physical MTP layer {idx} received no gradient'


@pytest.mark.parametrize('depth', [1, 2])
def test_mtp_forward_backward_per_depth(depth):
    with _parallel_context():
        model = _build(depth=depth)
        ids, pos = _inputs()
        loss, depth_losses, grad_sums = _run(model, ids, pos, ids.clone())
        assert torch.isfinite(torch.tensor(loss))
        _assert_every_depth_active(depth, depth_losses, grad_sums, repeated=False)


def test_mtp_repeated_forward_backward_per_depth():
    with _parallel_context():
        model = _build(depth=2, repeated=True)
        # logical depth 2 collapses to one physical layer / weight set
        assert model.config.mtp_use_repeated_layer is True
        assert len(model.mtp.layers) == 1
        assert get_physical_mtp_count(model) == 1
        ids, pos = _inputs()
        loss, depth_losses, grad_sums = _run(model, ids, pos, ids.clone())
        assert torch.isfinite(torch.tensor(loss))
        # both logical depths active, one physical layer carrying the grads
        _assert_every_depth_active(2, depth_losses, grad_sums, repeated=True)


def test_mtp_full_recompute_per_depth():
    """Full recompute must exercise the bridge checkpoint path without
    changing the math (0.18.x padding_mask threading)."""
    with _parallel_context():
        ids, pos = _inputs()
        labels = ids.clone()
        model_parallel_cuda_manual_seed(123)
        eager = _build(depth=2)
        loss_e, dl_e, gs_e = _run(eager, ids, pos, labels)
        model_parallel_cuda_manual_seed(123)
        recomp = _build(depth=2, recompute=True)
        # same seed -> same init; recompute must not change the math
        loss_r, dl_r, gs_r = _run(recomp, ids, pos, labels)
        assert recomp.config.recompute_granularity == 'full'
        assert torch.isfinite(torch.tensor(loss_r))
        _assert_every_depth_active(2, dl_r, gs_r, repeated=False)
        assert loss_r == pytest.approx(loss_e, rel=1e-5, abs=1e-5), f'recompute changed loss: {loss_e} -> {loss_r}'
        assert torch.allclose(dl_r, dl_e, rtol=1e-4, atol=1e-5), f'recompute changed MTP losses: {dl_e} vs {dl_r}'


def _packed_inputs(lengths, device='cuda'):
    total = sum(lengths)
    ids = torch.randint(1, VOCAB, (1, total), device=device)
    pos = torch.cat([torch.arange(n, device=device) for n in lengths]).unsqueeze(0)
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), device=device, dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format='thd', cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max(lengths), max_seqlen_kv=max(lengths))
    return ids, pos, packed


@pytest.mark.parametrize('depth', [1, 2])
def test_mtp_packed_forward_backward_per_depth(depth):
    """Packed (thd) at depth>=2 must compose single-step rolls; passing the
    cumulative depth to the packed helper trips its ``shifts == -1`` assert."""
    with _parallel_context():
        # packed thd needs a flash/fused backend -> bf16, head_dim 64
        model = _build(depth=depth, bf16=True, hidden=128, heads=2)
        ids, pos, packed = _packed_inputs([8, 8])
        _reset_mtp_tracker()
        out = model(input_ids=ids, position_ids=pos, attention_mask=None, labels=ids.clone(), packed_seq_params=packed)
        loss = out if torch.is_tensor(out) else out[0]
        loss = loss.float().mean() if loss.dim() > 0 else loss.float()
        depth_losses = _tracked_mtp_losses()
        model.zero_grad()
        loss.backward()
        assert torch.isfinite(loss)
        grad_sums = _mtp_layer_grad_sums(model)
        _assert_every_depth_active(depth, depth_losses, grad_sums, repeated=False)


def test_mtp_ignore_index_trailing_tokens():
    """Trailing -100 TOKENS (labels[:, -1], not labels[-1, :]) must be excluded
    from every depth's loss without zeroing the remaining mask."""
    with _parallel_context():
        model = _build(depth=2)
        ids, pos = _inputs()
        labels = ids.clone()
        labels[:, -1] = -100
        loss, depth_losses, grad_sums = _run(model, ids, pos, labels)
        assert torch.isfinite(torch.tensor(loss))
        _assert_every_depth_active(2, depth_losses, grad_sums, repeated=False)


def test_mtp_all_labels_ignored_backward_safe():
    """When every label is -100 the rolled loss mask is all-zero; forward must
    stay finite AND backward must not produce NaN gradients anywhere."""
    with _parallel_context():
        model = _build(depth=2)
        ids, pos = _inputs()
        labels = torch.full_like(ids, -100)
        loss, depth_losses, _ = _run(model, ids, pos, labels, backward=False)
        assert torch.isfinite(torch.tensor(loss)), f'loss became non-finite with all labels ignored: {loss}'
        assert depth_losses is not None and torch.all(depth_losses == 0), \
            f'all-ignored run must track zero MTP loss per depth: {depth_losses}'
        model.zero_grad()
        out = model(input_ids=ids, position_ids=pos, attention_mask=None, labels=labels)
        full = (out if torch.is_tensor(out) else out[0]).float().mean()
        full.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f'non-finite gradient after all-ignored backward: {name}'


def _fake_rope_layer(pos_type):
    from types import SimpleNamespace
    return SimpleNamespace(config=SimpleNamespace(position_embedding_type=pos_type), cp_group=None)


def test_shift_rope_packed_single_step_contract():
    """Packed rope composes depth single-step rolls from the ORIGINAL
    block-entry position_ids; packed MRoPE must shift the table; non-packed
    rolls the original table by the cumulative depth."""
    with _parallel_context():
        S = 16
        table = torch.arange(S * 4, device='cuda', dtype=torch.float32).reshape(S, 4)
        cu = torch.tensor([0, 8, 16], device='cuda', dtype=torch.int32)
        psp = PackedSeqParams(qkv_format='thd', cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=8, max_seqlen_kv=8)
        pos = torch.arange(S, device='cuda').unsqueeze(0)  # [1, S] ORIGINAL

        rolled1 = roll_single(pos, shifts=-1, dims=-1, cp_group=None, packed_seq_params=psp)
        out1 = MultiTokenPredictionLayer._shift_rope(_fake_rope_layer('rope'), table, 1, pos, psp)
        assert torch.equal(out1, table[rolled1[0]])

        # depth 2 from the ORIGINAL: two composed single steps (never -2, the
        # packed helper only supports shifts == -1)
        rolled2 = roll_single(rolled1, shifts=-1, dims=-1, cp_group=None, packed_seq_params=psp)
        out2 = MultiTokenPredictionLayer._shift_rope(_fake_rope_layer('rope'), table, 2, pos, psp)
        assert torch.equal(out2, table[rolled2[0]])
        assert not torch.equal(out2, out1), 'depth 2 must differ from depth 1'

        # packed MRoPE: [3,1,S] position_ids must not defeat the shift
        mrope_pos = torch.arange(S, device='cuda').unsqueeze(0).expand(3, 1, S).contiguous()
        for depth in (1, 2):
            out_m = MultiTokenPredictionLayer._shift_rope(_fake_rope_layer('mrope'), table, depth, mrope_pos, psp)
            assert torch.equal(out_m, torch.roll(table, -depth, dims=0)), \
                f'packed mrope depth {depth} did not shift the table'

        # non-packed: cyclic roll of the original table by cumulative depth
        out_np = MultiTokenPredictionLayer._shift_rope(_fake_rope_layer('rope'), table, 2, pos, None)
        assert torch.equal(out_np, torch.roll(table, -2, dims=0))


def test_precomputed_embedding_skips_vocab_lookup():
    """With a precomputed (multimodal) decoder_input the vocab lookup must be
    skipped entirely -- negative placeholder ids must not reach nn.Embedding
    and the embedding weight must receive no grad."""
    with _parallel_context():
        model = _build(depth=2, bf16=True, hidden=128, heads=2)
        model.train()
        batch, seq = 2, 16
        ids, pos = _inputs(batch, seq)
        ids[:, 3:6] = -1  # image-placeholder ids
        hs = torch.randn(seq, batch, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        pre = torch.randn(seq, batch, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        out = model.mtp(
            input_ids=ids,
            position_ids=pos,
            hidden_states=hs,
            attention_mask=None,
            embedding=model.embedding,
            decoder_input=pre,
            rotary_pos_emb=None)
        loss = out.float().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert pre.grad is not None and pre.grad.abs().sum().item() > 0, \
            'precomputed embedding got no gradient (rolled path not connected?)'
        emb_grad = model.embedding.word_embeddings.weight.grad
        assert emb_grad is None or emb_grad.abs().sum().item() == 0, \
            'vocab embedding was touched despite precomputed decoder_input'


@pytest.mark.parametrize('recompute', [False, True])
def test_bridge_extras_reach_inner_layer(recompute):
    """Keys no upstream forward signature accepts (gemma4 shared_kv_states /
    per_layer_inputs) must be injected into the inner TransformerLayer -- under
    eager AND under full recompute (the checkpoint closure must restore the
    per-microbatch extras during backward)."""
    with _parallel_context():
        model = _build(depth=1, recompute=recompute, bf16=True, hidden=128, heads=2)
        model.train()
        inner = model.mtp.layers[0].mtp_model_layer
        captured = []
        orig_fa = inner._forward_attention

        def spy_fa(*a, _orig=orig_fa, **kw):
            captured.append(dict(kw))
            # consume the extra like Gemma4's attention does so the generic
            # attention below does not TypeError and backward can run
            kw.pop('shared_kv_states', None)
            return _orig(*a, **kw)

        inner._forward_attention = spy_fa
        sentinel = torch.zeros(4, 4, device='cuda')
        ids, pos = _inputs()
        hs = torch.randn(S, B, 128, device='cuda', dtype=torch.bfloat16)
        out = model.mtp(
            input_ids=ids,
            position_ids=pos,
            hidden_states=hs,
            attention_mask=None,
            embedding=model.embedding,
            rotary_pos_emb=None,
            shared_kv_states=sentinel)
        out.float().mean().backward()
        # Eager sees the original object; the recompute replay sees the
        # STAGED DETACHED COPY that CheckpointFunction.detach_variable made
        # (equal values, different identity).
        got = [c for c in captured if c.get('shared_kv_states') is sentinel]
        got_by_value = [
            c for c in captured
            if torch.is_tensor(c.get('shared_kv_states')) and torch.equal(c['shared_kv_states'], sentinel)
        ]
        if recompute:
            assert len(got) == 1, f'eager-phase extras delivery broken: {len(got)} identity captures'
            assert len(got_by_value) == 2, \
                f'extras lost during recompute: {len(captured)} captures, {len(got_by_value)} value matches'
        else:
            assert len(got) == 1, f'extras not delivered eagerly: {len(captured)} captures, {len(got)} with extras'


def test_padding_mask_survives_full_recompute():
    """On versions whose `_checkpointed_forward` lacks the padding_mask
    parameter (0.18.x) the bridge-owned checkpoint path must deliver the ROLLED
    padding mask to the inner layer during forward AND recompute."""
    from mcore_bridge.model.mtp import get_mtp_capabilities
    with _parallel_context():
        model = _build(depth=1, recompute=True)
        model.train()
        inner = model.mtp.layers[0].mtp_model_layer
        seen = []
        orig_fa = inner._forward_attention

        def spy_fa(hidden_states, *a, _orig=orig_fa, **kw):
            seen.append(kw.get('padding_mask', 'ABSENT'))
            return _orig(hidden_states, *a, **kw)

        inner._forward_attention = spy_fa
        ids, pos = _inputs()
        pmask = torch.zeros(B, S, device='cuda', dtype=torch.bool)
        pmask[:, -2:] = True
        hs = torch.randn(S, B, HIDDEN, device='cuda')
        out = model.mtp(
            input_ids=ids,
            position_ids=pos,
            hidden_states=hs,
            attention_mask=None,
            embedding=model.embedding,
            rotary_pos_emb=None,
            padding_mask=pmask)
        out.float().mean().backward()
        tensor_hits = [p for p in seen if torch.is_tensor(p)]
        caps = get_mtp_capabilities()
        if caps.layer_returns_padding_mask:
            # forward + recompute must both deliver a real rolled mask tensor
            assert len(tensor_hits) >= 2, f'padding_mask dropped under recompute: seen={seen}'
            assert tensor_hits[0].shape == pmask.shape


def _detach_heads_supported():
    from mcore_bridge.model.mtp import get_mtp_capabilities
    return get_mtp_capabilities().has_detach_heads


def test_mtp_detach_heads_precomputed():
    """Precomputed decoder input + mtp_detach_heads -- the embedding stub must
    return a real Tensor and the REPLACED precomputed value must honor the
    detach flag, while the hidden-state path keeps training."""
    if not _detach_heads_supported():
        pytest.skip('mtp_detach_heads requires MCore 0.19+/dev')
    with _parallel_context():
        model = _build(depth=2, detach_heads=True, bf16=True, hidden=128, heads=2)
        model.train()
        ids, pos = _inputs()
        ids[:, 3:6] = -1  # placeholder ids must never reach a vocab lookup
        hs = torch.randn(S, B, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        pre = torch.randn(S, B, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        out = model.mtp(
            input_ids=ids,
            position_ids=pos,
            hidden_states=hs,
            attention_mask=None,
            embedding=model.embedding,
            decoder_input=pre,
            rotary_pos_emb=None)
        assert torch.isfinite(out.float()).all()
        out.float().mean().backward()
        assert pre.grad is None or pre.grad.abs().sum().item() == 0, \
            'detach_heads must cut the gradient into the precomputed embedding'
        assert hs.grad is not None and hs.grad.abs().sum().item() > 0, \
            'hidden-state path lost its gradient under detach_heads'


def test_differentiable_extras_recompute_graph_boundary():
    """A NON-LEAF differentiable extra shared by multiple depths must be staged
    as a checkpoint input, not closure-captured -- closure capture drags the
    producer graph through recompute-in-backward and frees it twice."""
    with _parallel_context():

        def run(recompute):
            model_parallel_cuda_manual_seed(123)
            model = _build(depth=2, recompute=recompute, bf16=True, hidden=128, heads=2)
            model.train()
            for li in range(get_physical_mtp_count(model)):
                inner = model.mtp.layers[li].mtp_model_layer
                orig = inner._forward_attention

                def spy(hidden_states, *a, _orig=orig, **kw):
                    ex = kw.pop('shared_kv_states', None)
                    if ex is not None:
                        # consume differentiably so the producer graph matters
                        hidden_states = hidden_states + ex.sum().to(hidden_states.dtype)
                    return _orig(hidden_states, *a, **kw)

                inner._forward_attention = spy
            p = torch.full((2, 2), 2.0, device='cuda', requires_grad=True)
            extra = p.square()  # non-leaf, shared by BOTH depths
            ids, pos = _inputs()
            hs = torch.randn(S, B, 128, device='cuda', dtype=torch.bfloat16)
            out = model.mtp(
                input_ids=ids,
                position_ids=pos,
                hidden_states=hs,
                attention_mask=None,
                embedding=model.embedding,
                rotary_pos_emb=None,
                shared_kv_states=extra)
            loss = out.float().mean()
            loss.backward()
            return loss.item(), p.grad.detach().clone()

        loss_e, grad_e = run(False)
        loss_r, grad_r = run(True)
        assert loss_r == pytest.approx(loss_e, rel=1e-3, abs=1e-3), f'recompute changed loss: {loss_e} vs {loss_r}'
        assert grad_r.abs().sum().item() > 0, 'producer got no gradient under recompute'
        assert torch.allclose(grad_e, grad_r, rtol=2e-2, atol=1e-4), \
            f'producer grad differs under recompute: eager {grad_e.tolist()} vs recompute {grad_r.tolist()}'


def test_dict_rope_checkpoint_protocol():
    """A dict rotary_pos_emb must never reach save_for_backward positionally;
    the bridge checkpoint flattens the dict's tensors into staged inputs,
    recomputes exactly once in backward, and preserves the math."""
    with _parallel_context():
        model = _build(depth=1, recompute=True)
        layer = model.mtp.layers[0]
        hs = torch.randn(S, B, HIDDEN, device='cuda', requires_grad=True)
        dec = torch.randn(S, B, HIDDEN, device='cuda')
        rope = {'full_attention': torch.randn(S, 8, device='cuda')}
        calls = []

        def light_proj(hidden_states, decoder_input, rotary_pos_emb=None, **kw):
            calls.append(rotary_pos_emb)
            return hidden_states * 2 + rotary_pos_emb['full_attention'].sum() * 0

        layer._proj_and_transformer_layer = light_proj
        out = layer._checkpointed_forward(
            hidden_states=hs,
            decoder_input=dec,
            rotary_pos_emb=rope,
            attention_mask=None,
            packed_seq_params=None,
            sequence_len_offset=None)
        out.sum().backward()
        assert len(calls) == 2, f'expected forward + one recompute, got {len(calls)} calls'
        assert all(isinstance(c, dict) and torch.is_tensor(c['full_attention']) for c in calls), \
            f'dict rope was corrupted through the checkpoint: {calls}'
        assert torch.equal(hs.grad, torch.full_like(hs, 2.0)), 'checkpoint changed the gradient math'
