# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import torch
from torch import nn


class Qwen4ExpTextRMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6, dtype=None, sequence_parallel: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim, dtype=dtype))
        # Replicated across TP; reduce grads across TP when SP is on.
        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        # zero-centered: (1 + w), and Qwen4ExpText casts after scaling
        return (out * (1.0 + self.weight.float())).type_as(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class QSAIndexer(nn.Module):
    # refer: transformers Qwen4ExpTextQSAIndexer
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.index_n_heads = config.indexer_n_heads
        self.index_kv_heads = config.indexer_kv_heads
        self.index_head_dim = config.indexer_head_dim
        self.compress_ratio = config.indexer_compress_ratio
        self.token_budget = config.indexer_budget
        self.block_topk = self.token_budget // self.compress_ratio
        # Replicated projection (reference uses ReplicatedLinear).
        self.index_qk_proj = nn.Linear(
            config.hidden_size, (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            dtype=config.params_dtype)
        self.q_layernorm = Qwen4ExpTextRMSNorm(
            self.index_head_dim,
            eps=config.layernorm_epsilon,
            dtype=config.params_dtype,
            sequence_parallel=config.sequence_parallel)
        self.k_layernorm = Qwen4ExpTextRMSNorm(
            self.index_head_dim,
            eps=config.layernorm_epsilon,
            dtype=config.params_dtype,
            sequence_parallel=config.sequence_parallel)
        setattr(self.index_qk_proj.weight, 'sequence_parallel', config.sequence_parallel)

    def forward(self, *args, **kwargs):
        raise RuntimeError('QSAIndexer performs selection via `select_mask`, not `forward`.')

    @torch.no_grad()
    def select_mask(self, hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Return the QSA selection as a bool mask, or ``None`` when it is a no-op.

        Args:
            hidden_states: ``[s, b, h]`` (mcore layout), pre-attention input --
                the same tensor the reference indexer consumes.
            freqs: mcore rotary frequencies ``[s, 1, 1, rot_dim]``. mcore stores
                angles rather than cos/sin, so they are materialized here the way
                ``_patch_apply_rotary_pos_emb`` does (``cos(freqs) * mscale``),
                keeping the indexer's RoPE identical to the attention's.

        Returns:
            ``[b, 1, s, s]`` bool mask where True marks a *masked-out* key (TE's
            ``arbitrary`` convention), or ``None`` if every visible key is
            selected -- in which case the caller keeps the plain causal path and
            pays nothing.

        Only the causal, unpacked layout is handled; callers must not invoke this
        for packed/THD or context-parallel inputs (see the guard in the layer).
        """
        s, b, _ = hidden_states.shape
        R = self.compress_ratio
        max_blocks = s // R
        # Selection is a no-op while the causal prefix never exceeds the budget:
        # `topk(min(block_topk, num_blocks))` then keeps every block and the tail
        # re-adds the remainder, so the mask would be exactly causal. Skipping it
        # keeps short sequences on TE's fused causal kernel.
        if max_blocks <= self.block_topk:
            return None

        device = hidden_states.device
        # ---- project to indexer q/k ----
        # [s, b, h] -> [s, b, (nh + nkv) * d]
        qk = self.index_qk_proj(hidden_states)
        q, token_k = torch.split(
            qk, [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim], dim=-1)
        # -> [b, s, nh, d] / [b, s, d]
        q = q.view(s, b, self.index_n_heads, self.index_head_dim).permute(1, 0, 2, 3)
        raw_keys = token_k.view(s, b, self.index_kv_heads, self.index_head_dim).permute(1, 0, 2, 3).squeeze(2)
        q = self.q_layernorm(q)

        # ---- materialize cos/sin from mcore freqs ----
        # freqs: [s, 1, 1, rot_dim] -> [s, rot_dim]; mscale mirrors the attention path.
        mscale = getattr(self.config, 'attention_scaling', 1.0) or 1.0
        f = freqs.reshape(freqs.shape[0], -1)[:s]
        cos = (torch.cos(f) * mscale).to(q.dtype)
        sin = (torch.sin(f) * mscale).to(q.dtype)
        rot = cos.shape[-1]

        def apply_rope(t, cos_, sin_):
            t_rope, t_pass = t[..., :rot], t[..., rot:]
            t_rope = (t_rope * cos_) + (_rotate_half(t_rope) * sin_)
            return torch.cat((t_rope, t_pass), dim=-1)

        # queries rotate at their own position: cos [s, rot] -> [1, s, 1, rot]
        q = apply_rope(q, cos[None, :, None, :], sin[None, :, None, :])

        # ---- pool every block once (shared across queries) ----
        usable = max_blocks * R
        key_groups = raw_keys[:, :usable].view(b, max_blocks, R, self.index_head_dim)
        pooled = key_groups.float().mean(dim=2).to(raw_keys.dtype)
        pooled = self.k_layernorm(pooled)
        # blocks rotate at their first token's position
        starts = torch.arange(max_blocks, device=device) * R
        block_keys = apply_rope(pooled, cos[starts][None], sin[starts][None])  # [b, nb, d]

        # ---- score all (query, block) pairs ----
        scores = torch.einsum('bqhd,bkd->bqhk', q.float(), block_keys.float())
        scores = torch.relu(scores).sum(dim=2) / math.sqrt(self.index_head_dim)  # [b, s, nb]

        # ---- restrict to blocks fully inside the causal prefix ----
        n_blocks = (torch.arange(s, device=device) + 1) // R  # [s]
        block_ids = torch.arange(max_blocks, device=device)
        scores = scores.masked_fill((block_ids[None, :] >= n_blocks[:, None])[None], float('-inf'))

        # ---- top-k blocks -> token mask ----
        k = min(self.block_topk, max_blocks)
        top_blocks = scores.topk(k, dim=-1).indices  # [b, s, k]
        keep = top_blocks < n_blocks[None, :, None]  # drop the -inf padding slots
        tok = (top_blocks.unsqueeze(-1) * R + torch.arange(R, device=device)).flatten(-2)
        keep_tok = keep.unsqueeze(-1).expand(-1, -1, -1, R).flatten(-2)

        allowed = torch.zeros((b, s, s + 1), dtype=torch.bool, device=device)
        allowed.scatter_(-1, torch.where(keep_tok, tok, torch.full_like(tok, s)).long(), True)
        allowed = allowed[..., :s]
        # tail: visible tokens after the last complete block are always attended
        pos = torch.arange(s, device=device)
        tail = (pos[None, :] >= (n_blocks * R)[:, None]) & (pos[None, :] <= pos[:, None])
        allowed |= tail[None]
        # causal safety net (the block expansion never crosses it, but keep the
        # invariant explicit so a future layout change fails loudly instead of
        # silently attending to the future)
        allowed &= (pos[None, :] <= pos[:, None])[None]
        return ~allowed.unsqueeze(1)  # True == masked out
