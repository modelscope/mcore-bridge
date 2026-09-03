# Copyright (c) ModelScope Contributors. All rights reserved.
"""Triton kernels for the qwen4_exp PLE host-offload path.

``gather_ple_rows`` reads rows straight out of the CPU-pinned n-gram table via
a raw host pointer, so the GPU pulls only the rows it needs over the coherent
link instead of staging the (~102 GB) table into HBM. Mirrors sglang's
``_gather_ple_embedding_from_pinned_kernel``.

Every kernel here is optional: if triton or CUDA is unavailable the callers in
ple.py fall back to the plain-torch path, which is numerically identical.
"""
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover - triton absent
    HAVE_TRITON = False

if HAVE_TRITON:

    @triton.jit
    def _gather_ple_rows_from_pinned(
        weight_ptr,
        ids_ptr,
        output_ptr,
        embedding_dim,
        row_start,
        row_end,
        BLOCK_D: tl.constexpr,
    ):
        # One program per flattened (token, hash-head) id. Rows outside this TP
        # rank's [row_start, row_end) are written as zero; the caller sums the
        # per-rank results across TP to reassemble the full embedding.
        row_id = tl.program_id(0)
        global_idx = tl.load(ids_ptr + row_id)
        in_range = (global_idx >= row_start) & (global_idx < row_end)
        local_idx = tl.where(in_range, global_idx - row_start, 0)
        offsets = tl.arange(0, BLOCK_D)
        mask = offsets < embedding_dim
        ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
        values = tl.load(ptr + local_idx * embedding_dim + offsets, mask=mask, other=0.0)
        tl.store(
            output_ptr + row_id * embedding_dim + offsets,
            tl.where(in_range, values.to(tl.bfloat16), 0.0),
            mask=mask,
        )


def gather_ple_rows(host_table, ids, row_start, row_end, out=None):
    """Gather n-gram rows from the CPU-pinned table with a triton kernel.

    Returns ``None`` when the fast path is not usable (no triton/CUDA, or the table
    is not bf16), so the caller can fall back to the torch path.

    Args:
        host_table: ``[n_local, embedding_dim]`` bf16 CPU-pinned table partition.
        ids: int64 tensor of any shape, values in ``[0, padded_vocab_size)``.
        row_start / row_end: this rank's global row range.
        out: optional preallocated ``[*ids.shape, embedding_dim]`` bf16 device tensor.
    """
    if not HAVE_TRITON or not torch.cuda.is_available():
        return None
    if host_table.dtype != torch.bfloat16 or ids.device.type != 'cuda':
        return None
    embedding_dim = host_table.shape[-1]

    shape = (*ids.shape, embedding_dim)
    if out is None:
        out = torch.empty(shape, dtype=torch.bfloat16, device=ids.device)
    flat = ids.reshape(-1)
    if flat.numel():
        _gather_ple_rows_from_pinned[(flat.numel(), )](
            host_table.data_ptr(),
            flat.contiguous(),
            out.view(-1, embedding_dim),
            embedding_dim=embedding_dim,
            row_start=row_start,
            row_end=row_end,
            BLOCK_D=triton.next_power_of_2(embedding_dim),
        )
    return out


if HAVE_TRITON:

    @triton.jit(do_not_specialize=['T'])
    def _ple_gate_fwd_kernel(
        key_ptr,  # [T, N*C] pre-norm key projection
        query_ptr,  # [T, N*C] hc state (the PLE query)
        value_ptr,  # [T, C]
        wk_ptr,
        wq_ptr,  # zero-centered grouped-norm weights [N*C]
        gated_ptr,  # fp32 out [T, N*C]
        gate_ptr,
        rstdk_ptr,
        rstdq_ptr,  # fp32 out [T, N]
        T,
        N: tl.constexpr,
        C: tl.constexpr,
        EPS: tl.constexpr,
        SQRTC: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        # Fused: grouped RMSNorm(key) * grouped RMSNorm(query) -> per-group score,
        # gate = sigmoid(sign(s)*sqrt(max(|s|,1e-6))), out = gate * value.
        pid = tl.program_id(0)
        t = pid // N
        c = pid % N
        if t >= T:
            return
        offs = tl.arange(0, BLOCK_C)
        mask = offs < C
        base = t * (N * C) + c * C

        k = tl.load(key_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        q = tl.load(query_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        wk = tl.load(wk_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)
        wq = tl.load(wq_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)

        rk = 1.0 / tl.sqrt(tl.sum(k * k, axis=0) / C + EPS)
        rq = 1.0 / tl.sqrt(tl.sum(q * q, axis=0) / C + EPS)
        kn = k * rk * (1.0 + wk)
        qn = q * rq * (1.0 + wq)

        score = tl.sum(kn * qn, axis=0) / SQRTC
        mag = tl.maximum(tl.abs(score), 1e-6)
        sgn = tl.where(score >= 0, 1.0, -1.0)
        gate = tl.sigmoid(sgn * tl.sqrt(mag))

        v = tl.load(value_ptr + t * C + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(gated_ptr + base + offs, gate * v, mask=mask)
        tl.store(gate_ptr + t * N + c, gate)
        tl.store(rstdk_ptr + t * N + c, rk)
        tl.store(rstdq_ptr + t * N + c, rq)

    @triton.jit(do_not_specialize=['T'])
    def _ple_gate_bwd_kernel(
        dgated_ptr,  # fp32 in [T, N*C]
        key_ptr,
        query_ptr,
        value_ptr,
        wk_ptr,
        wq_ptr,
        gate_ptr,
        rstdk_ptr,
        rstdq_ptr,
        dkey_ptr,
        dquery_ptr,  # out, input dtype [T, N*C]
        dvalue_ptr,  # fp32 out [T, N, C] (host sums over N)
        dwk_partial_ptr,
        dwq_partial_ptr,  # fp32 out [T, N*C] (host sums over T)
        T,
        N: tl.constexpr,
        C: tl.constexpr,
        SQRTC: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)
        t = pid // N
        c = pid % N
        if t >= T:
            return
        offs = tl.arange(0, BLOCK_C)
        mask = offs < C
        base = t * (N * C) + c * C
        sqrtC = SQRTC

        k = tl.load(key_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        q = tl.load(query_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(value_ptr + t * C + offs, mask=mask, other=0.0).to(tl.float32)
        wk = tl.load(wk_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)
        wq = tl.load(wq_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(gate_ptr + t * N + c)
        rk = tl.load(rstdk_ptr + t * N + c)
        rq = tl.load(rstdq_ptr + t * N + c)
        dg_out = tl.load(dgated_ptr + base + offs, mask=mask, other=0.0)

        kn = k * rk * (1.0 + wk)
        qn = q * rq * (1.0 + wq)

        dgate = tl.sum(dg_out * v, axis=0)
        tl.store(dvalue_ptr + (t * N + c) * C + offs, dg_out * g, mask=mask)

        score = tl.sum(kn * qn, axis=0) / sqrtC
        mag = tl.maximum(tl.abs(score), 1e-6)
        du = dgate * g * (1.0 - g)
        ds = tl.where(tl.abs(score) > 1e-6, du / (2.0 * tl.sqrt(mag)), 0.0)

        dkn = qn * (ds / sqrtC)
        dqn = kn * (ds / sqrtC)

        tl.store(dwk_partial_ptr + base + offs, dkn * (k * rk), mask=mask)
        tl.store(dwq_partial_ptr + base + offs, dqn * (q * rq), mask=mask)

        gk = dkn * (1.0 + wk)
        dotk = tl.sum(gk * k, axis=0)
        dk = rk * gk - k * (rk * rk * rk) * (dotk / C)
        gq = dqn * (1.0 + wq)
        dotq = tl.sum(gq * q, axis=0)
        dq = rq * gq - q * (rq * rq * rq) * (dotq / C)

        tl.store(dkey_ptr + base + offs, dk.to(dkey_ptr.dtype.element_ty), mask=mask)
        tl.store(dquery_ptr + base + offs, dq.to(dquery_ptr.dtype.element_ty), mask=mask)

    @triton.jit(do_not_specialize=['T'])
    def _ple_norm_fwd_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        rstd_ptr,
        T,
        N: tl.constexpr,
        C: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        # Grouped zero-centered RMSNorm: out = x * rstd * (1 + w), fp32 out.
        pid = tl.program_id(0)
        t = pid // N
        c = pid % N
        if t >= T:
            return
        offs = tl.arange(0, BLOCK_C)
        mask = offs < C
        base = t * (N * C) + c * C
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)
        r = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / C + EPS)
        tl.store(out_ptr + base + offs, x * r * (1.0 + w), mask=mask)
        tl.store(rstd_ptr + t * N + c, r)

    @triton.jit(do_not_specialize=['T'])
    def _ple_norm_bwd_kernel(
        x_ptr,
        w_ptr,
        rstd_ptr,
        dout_ptr,
        dx_ptr,
        T,
        N: tl.constexpr,
        C: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)
        t = pid // N
        c = pid % N
        if t >= T:
            return
        offs = tl.arange(0, BLOCK_C)
        mask = offs < C
        base = t * (N * C) + c * C
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(rstd_ptr + t * N + c)
        dn = tl.load(dout_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        xh = x * r
        g = dn * (1.0 + w)
        dx = r * (g - xh * (tl.sum(g * xh, axis=0) / C))
        tl.store(dx_ptr + base + offs, dx, mask=mask)

    @triton.jit(do_not_specialize=['T', 'W'])
    def _ple_conv_fwd_kernel(
        normed_ptr,  # fp32 [T, W]
        gated_ptr,  # fp32 [T, W]
        convw_ptr,  # [W, K] depthwise weights
        segstart_ptr,  # int32 [T]
        out_ptr,  # out dtype [T, W]
        conv_ptr,  # fp32 out [T, W] pre-SiLU conv (saved for bwd)
        T,
        W,
        K: tl.constexpr,
        DIL: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        # Causal dilated depthwise conv; rows never read across their segment
        # start. out = gated + silu(conv(normed)).
        t = tl.program_id(0)
        wb = tl.program_id(1)
        if t >= T:
            return
        offs = wb * BLOCK_W + tl.arange(0, BLOCK_W)
        mask = offs < W
        seg_lo = tl.load(segstart_ptr + t)

        acc = tl.zeros([BLOCK_W], dtype=tl.float32)
        for j in tl.static_range(K):
            src = t - (K - 1 - j) * DIL
            wgt = tl.load(convw_ptr + offs * K + j, mask=mask, other=0.0).to(tl.float32)
            if src >= 0:
                ok = src >= seg_lo
                x = tl.load(normed_ptr + src * W + offs, mask=mask & ok, other=0.0)
                acc += wgt * x
        tl.store(conv_ptr + t * W + offs, acc, mask=mask)
        silu = acc * tl.sigmoid(acc)
        gt = tl.load(gated_ptr + t * W + offs, mask=mask, other=0.0)
        tl.store(out_ptr + t * W + offs, (gt + silu).to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit(do_not_specialize=['T', 'W'])
    def _ple_conv_bwd_kernel(
        dout_ptr,  # incoming grad [T, W]
        conv_ptr,  # fp32 [T, W] pre-SiLU
        normed_ptr,  # fp32 [T, W]
        convw_ptr,  # [W, K]
        segstart_ptr,
        segend_ptr,  # int32 [T] (exclusive)
        dnormed_ptr,  # fp32 out [T, W]
        dconvw_ptr,  # fp32 out [W, K] via atomics
        dgated_add_ptr,  # fp32 out [T, W] (dout passthrough for the residual)
        T,
        W,
        K: tl.constexpr,
        DIL: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        t = tl.program_id(0)
        wb = tl.program_id(1)
        if t >= T:
            return
        offs = wb * BLOCK_W + tl.arange(0, BLOCK_W)
        mask = offs < W
        seg_lo = tl.load(segstart_ptr + t)
        seg_hi = tl.load(segend_ptr + t)

        do = tl.load(dout_ptr + t * W + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(dgated_add_ptr + t * W + offs, do, mask=mask)

        cv = tl.load(conv_ptr + t * W + offs, mask=mask, other=0.0)
        sig = tl.sigmoid(cv)
        dconv = do * sig * (1.0 + cv * (1.0 - sig))

        for j in tl.static_range(K):
            src = t - (K - 1 - j) * DIL
            if src >= 0:
                ok = src >= seg_lo
                x = tl.load(normed_ptr + src * W + offs, mask=mask & ok, other=0.0)
                tl.atomic_add(dconvw_ptr + offs * K + j, dconv * x, mask=mask & ok)

        acc = tl.zeros([BLOCK_W], dtype=tl.float32)
        for j in tl.static_range(K):
            dst = t + (K - 1 - j) * DIL
            wgt = tl.load(convw_ptr + offs * K + j, mask=mask, other=0.0).to(tl.float32)
            if dst < T:
                ok = dst < seg_hi
                do2 = tl.load(dout_ptr + dst * W + offs, mask=mask & ok, other=0.0).to(tl.float32)
                cv2 = tl.load(conv_ptr + dst * W + offs, mask=mask & ok, other=0.0)
                sig2 = tl.sigmoid(cv2)
                acc += wgt * do2 * sig2 * (1.0 + cv2 * (1.0 - sig2))
        tl.store(dnormed_ptr + t * W + offs, acc, mask=mask)


if HAVE_TRITON:
    import math as _math

    def _uniform_seg_bounds(total, seq_len, device):
        # The PLE forward runs on padded [rows, seq_len] batches, so every row
        # is its own segment: the dilated conv must not read across rows.
        pos = torch.arange(total, device=device)
        lo = (pos // seq_len * seq_len).to(torch.int32)
        hi = (lo + seq_len).to(torch.int32)
        return lo, hi

    class _PLEGateConv(torch.autograd.Function):

        @staticmethod
        def forward(ctx, hc_state, key, value, wk, wq, wc, conv_w, n, eps, dilation, seq_len):
            T, W = hc_state.shape
            C = W // n
            Kk = conv_w.shape[-1]
            dev = hc_state.device
            block_c = triton.next_power_of_2(C)

            gated = torch.empty(T, W, dtype=torch.float32, device=dev)
            gate = torch.empty(T, n, dtype=torch.float32, device=dev)
            rstdk = torch.empty(T, n, dtype=torch.float32, device=dev)
            rstdq = torch.empty(T, n, dtype=torch.float32, device=dev)
            if T > 0:
                _ple_gate_fwd_kernel[(T * n, )](
                    key,
                    hc_state,
                    value,
                    wk,
                    wq,
                    gated,
                    gate,
                    rstdk,
                    rstdq,
                    T,
                    N=n,
                    C=C,
                    EPS=eps,
                    SQRTC=_math.sqrt(C),
                    BLOCK_C=block_c)

            normed = torch.empty(T, W, dtype=torch.float32, device=dev)
            rstdc = torch.empty(T, n, dtype=torch.float32, device=dev)
            if T > 0:
                _ple_norm_fwd_kernel[(T * n, )](gated, wc, normed, rstdc, T, N=n, C=C, EPS=eps, BLOCK_C=block_c)

            seg_lo, seg_hi = _uniform_seg_bounds(T, seq_len, dev)
            convw2d = conv_w.reshape(W, Kk).contiguous()
            out = torch.empty(T, W, dtype=hc_state.dtype, device=dev)
            conv_pre = torch.empty(T, W, dtype=torch.float32, device=dev)
            BW = 256
            if T > 0:
                _ple_conv_fwd_kernel[(T, triton.cdiv(W, BW))](
                    normed, gated, convw2d, seg_lo, out, conv_pre, T, W, K=Kk, DIL=dilation, BLOCK_W=BW)

            ctx.save_for_backward(hc_state, key, value, wk, wq, wc, convw2d, gate, rstdk, rstdq, rstdc, seg_lo, seg_hi,
                                  conv_pre)
            ctx.dims = (n, eps, dilation, Kk, conv_w.dtype)
            return out

        @staticmethod
        def backward(ctx, dout):
            (hc_state, key, value, wk, wq, wc, convw2d, gate, rstdk, rstdq, rstdc, seg_lo, seg_hi,
             conv_pre) = ctx.saved_tensors
            n, eps, dilation, Kk, conv_w_dtype = ctx.dims
            T, W = hc_state.shape
            C = W // n
            dev = hc_state.device
            dout = dout.contiguous()
            block_c = triton.next_power_of_2(C)

            # Recompute the forward intermediates (the fp32 gated/normed are not
            # saved to keep the recompute footprint small).
            gated = torch.empty(T, W, dtype=torch.float32, device=dev)
            _g = torch.empty(T, n, dtype=torch.float32, device=dev)
            _rk = torch.empty(T, n, dtype=torch.float32, device=dev)
            _rq = torch.empty(T, n, dtype=torch.float32, device=dev)
            if T > 0:
                _ple_gate_fwd_kernel[(T * n, )](
                    key,
                    hc_state,
                    value,
                    wk,
                    wq,
                    gated,
                    _g,
                    _rk,
                    _rq,
                    T,
                    N=n,
                    C=C,
                    EPS=eps,
                    SQRTC=_math.sqrt(C),
                    BLOCK_C=block_c)
            normed = torch.empty(T, W, dtype=torch.float32, device=dev)
            _rc = torch.empty(T, n, dtype=torch.float32, device=dev)
            if T > 0:
                _ple_norm_fwd_kernel[(T * n, )](gated, wc, normed, _rc, T, N=n, C=C, EPS=eps, BLOCK_C=block_c)

            dnormed = torch.empty(T, W, dtype=torch.float32, device=dev)
            dconvw = torch.zeros(W, Kk, dtype=torch.float32, device=dev)
            dgated = torch.empty(T, W, dtype=torch.float32, device=dev)
            BW = 256
            if T > 0:
                _ple_conv_bwd_kernel[(T, triton.cdiv(W, BW))](
                    dout,
                    conv_pre,
                    normed,
                    convw2d,
                    seg_lo,
                    seg_hi,
                    dnormed,
                    dconvw,
                    dgated,
                    T,
                    W,
                    K=Kk,
                    DIL=dilation,
                    BLOCK_W=BW)

            # norm_conv backward: dwc on host, dx via kernel (fp32).
            x_hat = (gated.view(T, n, C) * rstdc.unsqueeze(-1)).view(T, W)
            dwc = (dnormed * x_hat).sum(dim=0).to(wc.dtype)
            dgated_norm = torch.empty(T, W, dtype=torch.float32, device=dev)
            if T > 0:
                _ple_norm_bwd_kernel[(T * n, )](gated, wc, rstdc, dnormed, dgated_norm, T, N=n, C=C, BLOCK_C=block_c)
            dgated += dgated_norm

            dkey = torch.empty_like(key)
            dquery = torch.empty_like(hc_state)
            dvalue_pern = torch.empty(T, n, C, dtype=torch.float32, device=dev)
            dwk_part = torch.empty(T, W, dtype=torch.float32, device=dev)
            dwq_part = torch.empty(T, W, dtype=torch.float32, device=dev)
            if T > 0:
                _ple_gate_bwd_kernel[(T * n, )](
                    dgated,
                    key,
                    hc_state,
                    value,
                    wk,
                    wq,
                    gate,
                    rstdk,
                    rstdq,
                    dkey,
                    dquery,
                    dvalue_pern,
                    dwk_part,
                    dwq_part,
                    T,
                    N=n,
                    C=C,
                    SQRTC=_math.sqrt(C),
                    BLOCK_C=block_c)
            dvalue = dvalue_pern.sum(dim=1).to(value.dtype)
            dwk = dwk_part.sum(dim=0).to(wk.dtype)
            dwq = dwq_part.sum(dim=0).to(wq.dtype)
            dconv_w = dconvw.view(convw2d.shape).view(W, 1, Kk).to(conv_w_dtype)

            return (dquery, dkey, dvalue, dwk, dwq, dwc, dconv_w, None, None, None, None)


def ple_gate_conv_triton(hc_state, key, value, norm_key_w, norm_query_w, norm_conv_w, conv1d_weight, n, eps, dilation,
                         seq_len):
    """Fused PLE increment (gate chain + norm_conv + causal dilated conv + SiLU +
    residual). fp32 accumulation, output dtype = ``hc_state.dtype``.

    Returns ``None`` when the fast path is unavailable so callers fall back.
    """
    if not HAVE_TRITON or not hc_state.is_cuda:
        return None
    return _PLEGateConv.apply(hc_state.contiguous(), key.contiguous(), value.contiguous(), norm_key_w, norm_query_w,
                              norm_conv_w, conv1d_weight, n, eps, dilation, seq_len)
