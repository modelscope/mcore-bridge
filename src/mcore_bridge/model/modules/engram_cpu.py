# Copyright (c) ModelScope Contributors. All rights reserved.
"""Frozen, EP-sharded Engram tables resident in ordinary host memory.

Only row IDs and selected embeddings cross the host/device boundary. All EP
collectives retain the native device tensors and ordering.
"""
import torch
import torch.nn.functional as F
from megatron.core.models.engram.distributed_embedding import EPShardedEmbeddingTable, EPShardedMultiTableEmbedding


class CPUEmbeddingTable(EPShardedEmbeddingTable):

    def forward(self, local_row_ids):
        if self.weight.requires_grad:
            raise RuntimeError('CPU Engram tables must remain frozen.')
        rows = F.embedding(local_row_ids.to('cpu'), self.weight)
        return rows.to(local_row_ids.device)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'weight'] = self.weight if keep_vars else self.weight.detach()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        key = prefix + 'weight'
        if key in state_dict:
            source = state_dict[key]
            if source.shape != self.weight.shape:
                error_msgs.append(f'{key}: expected {tuple(self.weight.shape)}, got {tuple(source.shape)}')
            else:
                with torch.no_grad():
                    self.weight.copy_(source.to(device='cpu', dtype=self.weight.dtype))
        elif strict:
            missing_keys.append(key)
        # The host tensor deliberately is neither a parameter nor a buffer.
        remaining = {k: v for k, v in state_dict.items() if k != key}
        super()._load_from_state_dict(remaining, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)


class CPUShardedEmbedding(EPShardedMultiTableEmbedding):

    def _lookup_received_requests(self, requests, table_counts):
        rows = super()._lookup_received_requests(requests.to('cpu'), table_counts)
        return rows.to(requests.device)


def retain_tables_on_cpu(embedding):
    """Adopt already CPU-constructed native tables without copying their storage.

    The native sharded_state_dict continues to expose weight, global shape and
    exact EP row offsets. Plain host tensors are excluded from optimizer and
    automatic Module._apply device moves; routing buffers still move normally.
    """
    if any(table.weight.device.type != 'cpu' for table in embedding.tables):
        raise RuntimeError('Construct Engram on CPU before enabling host residency.')
    embedding.__class__ = CPUShardedEmbedding
    for table in embedding.tables:
        weight = table._parameters.pop('weight')
        table.__class__ = CPUEmbeddingTable
        table.weight = weight.detach()
