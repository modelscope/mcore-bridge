# DeepSeek V4.1 DCP to Hugging Face export

`export_checkpoint` reads model records directly from a Megatron PyTorch DCP checkpoint and writes the official unquantized HF tensor layout. It does not construct a Megatron or HF model, initialize distributed process groups, or load optimizer/RNG records. This is useful for models whose complete CPU weights, especially Engram tables, are too large to materialize during a conventional export.

## Supported checkpoints

- DeepSeek V4.1 hybrid attention/MLP layout, BF16 full-parameter training with preserved FP32 parameters.
- TP1 training checkpoints; contiguous PP/EP chunks are reconstructed by global offsets. TP-sharded noncontiguous chunks are rejected.
- `language_model_only=true`, with MTP/DSpark and FP8 training disabled; no adapter or FSDP DTensor checkpoints.
- Checkpoint root containing the Swift `args.json`, `latest_checkpointed_iteration.txt`, and `iter_NNNNNNN/.metadata` from PyTorch `FileSystemWriter`.
- Model keys under `language_model.`, `model.language_model.`, `model0.language_model.`, or a bare backbone prefix. Exactly one supported model namespace is required.
- Base directory containing the original indexed safetensors, config, tokenizer and official `encoding/` assets. Its path must match the recorded training base when one is present.

DCP metadata uses pickle: use only a trusted checkpoint. Tensor records are deserialized on CPU with `torch.load(weights_only=True)`. The exporter deliberately rejects unknown model tensors or incomplete/overlapping coverage instead of guessing their mapping. PyTorch DCP serialization is version-dependent; use the training environment or a compatible PyTorch version.

## Usage

Run one process on a host that can access the source, base model and new output directory. No GPU is required; the installed Bridge runtime dependencies must still be available.

```bash
CUDA_VISIBLE_DEVICES="" python -m mcore_bridge.checkpoint \
  --checkpoint /path/to/checkpoint-70 \
  --base-model /path/to/DeepSeek-V4.1-Flash-1017 \
  --output /path/to/checkpoint-70-hf \
  --preflight

CUDA_VISIBLE_DEVICES="" python -m mcore_bridge.checkpoint \
  --checkpoint /path/to/checkpoint-70 \
  --base-model /path/to/DeepSeek-V4.1-Flash-1017 \
  --output /path/to/checkpoint-70-hf \
  --workers 1
```

The equivalent API can be called by a training framework:

```python
from mcore_bridge import export_checkpoint

report = export_checkpoint(
    checkpoint="/path/to/checkpoint-70",
    base_model="/path/to/DeepSeek-V4.1-Flash-1017",
    output="/path/to/checkpoint-70-hf",
    workers=1,
    max_shard_size=5_000_000_000,
)
```

The preflight returns the mapping plan and total output tensor bytes without creating the output directory. A complete Flash BF16/FP32 artifact can be around 1.5 TB; reserve the reported size plus filesystem overhead. Each writer holds one source DCP record and a bounded finite-check/write block, rather than the full model or concatenated Engram table. Peak memory depends on the largest serialized record and concurrent writers. Unchanged visual tensors are copied separately and briefly held in memory together. Increasing `workers` improves concurrency at the cost of host memory and storage contention.

## Mapping and output

Engram subtables are joined in table-ID order, key/value projections are joined by row, MoE fc1 tensors are split into expert gate/up tensors, and mHC scales retain pre/post/res order. BF16/FP32 source precision is preserved. Base headers verify official keys and logical shapes, including packed FP4 expert storage. Scale and MTP tensors from the quantized base are excluded.

Unchanged visual/aligner/image-marker weights, tokenizer, processor configuration when available, official encoding/inference reference files and generation configuration are copied from the base. The HF config removes quantization and disables MTP/DSpark. No new chat template is synthesized.

The directory includes indexed safetensors, `export-plan.json`, `export-progress.json`, `export-validation.json`, and `_SUCCESS` only after validation. Validation checks complete source/target coverage, shape/dtype and file headers, all language tensor values for NaN/Inf, full SHA256 shard readback, exact visual copies, and independent rectangular DCP/HF comparisons at representative endpoints, attention layers, experts, mHC scales and Engram boundaries.

An existing output directory is preserved and rejected, including a partial export. After interruption, select a new output directory. Conversion checks do not establish inference compatibility or model quality; validate the exported model with the intended serving engine separately.
