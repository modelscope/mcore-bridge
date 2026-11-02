# DeepSeek V4.1 DCP 导出 Hugging Face 格式

`export_checkpoint` 直接读取 Megatron PyTorch DCP 的模型记录，输出官方非量化 HF 权重布局。它不构造完整 Megatron/HF 模型，不初始化分布式进程组，也不加载 optimizer/RNG。对于 Engram 表等超大权重，可避免传统导出路径同时驻留完整模型。

## 支持范围

- DeepSeek V4.1 hybrid attention/MLP 布局，BF16 全参训练，保留需要 FP32 保存的参数。
- TP1 训练 checkpoint；根据全局 offset 重组连续 PP/EP 子分片，拒绝不连续的 TP 分片。
- `language_model_only=true`，关闭 MTP/DSpark 与 FP8 训练；不支持 adapter 和 FSDP DTensor checkpoint。
- 根目录包含 Swift 保存的 `args.json`、`latest_checkpointed_iteration.txt`，以及由 PyTorch `FileSystemWriter` 写入的 `iter_NNNNNNN/.metadata`。
- 模型前缀支持 `language_model.`、`model.language_model.`、`model0.language_model.` 和裸 backbone；只接受一个模型命名空间。
- 底座目录包含官方 safetensors 索引、config、tokenizer 与 `encoding/`。训练参数记录了底座路径时，必须与传入路径一致。

DCP metadata 使用 pickle，仅可读取可信 checkpoint。单个 tensor 记录通过 `torch.load(weights_only=True)` 加载到 CPU。未知模型 tensor、缺失或重叠区间会直接报错，不推测映射。PyTorch DCP 序列化依赖版本，建议使用原训练环境或兼容的 PyTorch。

## 使用方式

在能够访问源 checkpoint、底座与输出盘的机器上单进程运行，无需 GPU；仍需安装 Bridge 的运行依赖。

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

训练框架可直接使用 API：

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

`--preflight` 返回映射计划与输出 tensor 字节数，不创建输出目录。Flash 的完整 BF16/FP32 产物可能约 1.5 TB，需要预留计划大小及文件系统开销。每个 writer 驻留单条源 DCP 记录和有限大小的校验/写入块，不拼接完整 Engram 表。内存峰值取决于最大序列化记录及 writer 数；未训练视觉权重另行复制，会短暂同时驻留。提高 `workers` 会增加主机内存与存储压力。

## 映射、产物与验收

Engram 子表按编号拼接，key/value 投影按行合并，MoE fc1 拆分为各专家 gate/up 权重，mHC scale 保留 pre/post/res 顺序。源 BF16/FP32 精度保持不变；使用底座文件头核对官方 key 和逻辑 shape，包括 packed FP4 专家。排除量化底座的 scale 与 MTP tensor。

未参与训练的视觉/aligner/image marker、tokenizer、可用的 processor 配置、官方编码/推理参考代码和 generation config 从底座复制。HF config 删除量化设置并关闭 MTP/DSpark，不生成新的 chat template。

输出包含 safetensors 及索引、`export-plan.json`、`export-progress.json`、`export-validation.json`。仅验收成功后生成 `_SUCCESS`。检查源/目标完整区间覆盖、shape/dtype/文件头、所有语言权重 NaN/Inf、完整分片 SHA256 回读、视觉逐值一致，以及独立矩形 DCP/HF 抽查：首尾层、代表性 attention/MoE/mHC 与 Engram 边界。

已有输出目录会保留并退出，包含未完成目录。中断后应使用新目录。转换验收不证明推理兼容性或模型质量，需使用目标推理引擎单独验证。
