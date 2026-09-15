# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared constants."""

# Bound the per-collective GPU buffer when a full gathered tensor is streamed
# to CPU (checkpoint save / CPU-offloaded weight sync).
EXPORT_CHUNK_BYTES = 256 << 20
