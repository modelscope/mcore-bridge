import torch
import transformer_engine
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.spec_utils import build_module
from typing import Optional, Tuple

try:
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import DSAIndexer as McoreDSAIndexer
except ImportError:
    McoreDSAIndexer = None


class DSAIndexer(McoreDSAIndexer):

    def __init__(self, config, submodules, *args, **kwargs):
        super().__init__(config, submodules, *args, **kwargs)
        if getattr(config, 'fp8_param', False):
            with transformer_engine.pytorch.fp8_model_init(enabled=False):
                self.linear_weights_proj = build_module(
                    submodules.linear_weights_proj,
                    self.hidden_size,
                    self.index_n_heads,
                    config=self.config,
                    init_method=self.config.init_method,
                    bias=False,
                    skip_bias_add=False,
                    skip_weight_param_allocation=False,
                    parallel_mode='duplicated',
                )

    def forward_before_topk(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """All computations before topk."""
        from megatron.core.transformer.experimental_attention_variant.dsa import rotate_activation

        # =========================================
        # Gather inputs if sp is enabled
        # =========================================
        packed_seq_params, rotary_pos_emb = packed_seq_params  # patch
        assert packed_seq_params is None, 'Packed sequence is not supported for DSAttention'

        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
            qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)

        # =========================================
        # Get sequence length and batch size
        # =========================================
        seqlen, bsz, _ = x.size()

        # =========================================
        # q linear and apply rope to q
        # =========================================
        # [seqlen, batch, q_lora_rank] -> [seqlen, batch, index_n_heads * index_head_dim]
        q, _ = self.linear_wq_b(qr)
        # [seqlen, batch, index_n_heads * index_head_dim]
        #   -> [seqlen, batch, index_n_heads, index_head_dim]
        q = q.reshape(seqlen, bsz, self.index_n_heads, self.index_head_dim)
        q = self._apply_rope(q, rotary_pos_emb)  # mscale will be passed in by patch

        # =========================================
        # k linear and apply rope to k
        # =========================================
        # [seqlen, batch, hidden_size] -> [seqlen, batch, index_head_dim]
        k, _ = self.linear_wk(x)
        k = self.k_norm(k)
        # [seqlen, batch, index_head_dim] -> [seqlen, batch, 1, index_head_dim]
        k = k.reshape(seqlen, bsz, 1, self.index_head_dim)
        k = self._apply_rope(k, rotary_pos_emb)
        # [seqlen, batch, 1, index_head_dim] -> [seqlen, batch, index_head_dim]
        k = k.reshape(seqlen, bsz, self.index_head_dim)

        # =========================================
        # Rotate activation
        # =========================================
        q = rotate_activation(q)
        k = rotate_activation(k)

        # =========================================
        # Prepare weights for index scores
        # =========================================
        # [seqlen, batch, hidden_size] -> [seqlen, batch, index_n_heads]
        weights, _ = self.linear_weights_proj(x)
        weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale

        return q, k, weights

    def _apply_rope(self, x: torch.Tensor, rotary_pos_emb: torch.Tensor):
        """Apply RoPE to the input tensor."""
        # x_nope [seqlen, batch, *, index_head_dim - qk_pos_emb_head_dim]
        # x_pe   [seqlen, batch, *, qk_pos_emb_head_dim]
        x_pe, x_nope = torch.split(
            x, [self.index_head_dim - self.qk_pos_emb_head_dim, self.qk_pos_emb_head_dim], dim=-1)
        origin_multi_latent_attention = self.config.multi_latent_attention
        try:
            self.config.multi_latent_attention = self.config.dsa_indexer_rotary_interleaved
            x_pe = apply_rotary_pos_emb(
                x_pe,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=None,
                cp_group=self.pg_collection.cp,
            )
        finally:
            self.config.multi_latent_attention = origin_multi_latent_attention
        # [seqlen, batch, *, index_head_dim]
        x = torch.cat([x_pe, x_nope], dim=-1)
        return x

    def forward_with_scores(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for DSA Indexer that returns both index scores and top-k indices.

        This is used when KL loss is enabled to compare indexer scores with true attention scores.

        Args:
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Attention mask [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            index_scores: Index scores [batch, seqlen, seqlen].
            topk_indices: Top-k indices [batch, seqlen, index_topk].
        """
        try:
            from megatron.core.transformer.experimental_attention_variant.dsa import fused_qk_topk_naive
        except ImportError:
            raise ImportError('fused_qk_topk_naive is not available. Please install "megatron-core>=0.17.0"')
        # [seqlen, batch, index_n_heads * index_head_dim]
        # [seqlen, batch, index_head_dim]
        # [seqlen, batch, index_n_heads]
        q, k, weights = self.forward_before_topk(x, qr, packed_seq_params)

        # [batch, seqlen, seqlen], [batch, seqlen, index_topk]
        index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, self.index_topk, mask)

        return index_scores, topk_indices

    def forward(self,
                x: torch.Tensor,
                qr: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                packed_seq_params: Optional[PackedSeqParams] = None):
        """
        Forward pass for DSA Indexer.

        Args:
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Attention mask [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            topk_indices: Top-k indices for sparse attention [batch, seqlen, index_topk].
        """
        _, topk_indices = self.forward_with_scores(x, qr, mask, packed_seq_params)
        return topk_indices
