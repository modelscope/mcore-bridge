# MCore-Bridge: 让 Megatron 训练像 Transformers 一样简单

<!-- <p align="center">
    <br>
    <img src="asset/banner.png"/>
    <br>
<p> -->

<p align="center">
    <b>为最先进的大语言模型提供 Megatron-Core 模型定义</b>
</p>

<p align="center">
<a href="https://modelscope.cn/home">魔搭社区官网</a>
<br>
        中文&nbsp ｜ &nbsp<a href="README.md">English</a>&nbsp
</p>


<p align="center">
<img src="https://img.shields.io/badge/python-3.11-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A52.0-orange.svg">
<a href="https://github.com/NVIDIA/Megatron-LM/"><img src="https://img.shields.io/badge/megatron--core-%E2%89%A50.12-76B900.svg"></a>
<!-- <a href="https://mcore-bridge.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/docs-latest-blue.svg"></a> -->
<a href="https://pypi.org/project/mcore-bridge/"><img src="https://badge.fury.io/py/mcore-bridge.svg"></a>
<a href="https://github.com/modelscope/mcore-bridge/blob/main/LICENSE"><img src="https://img.shields.io/github/license/modelscope/mcore-bridge"></a>
<a href="https://pepy.tech/project/mcore-bridge"><img src="https://pepy.tech/badge/mcore-bridge"></a>
<a href="https://github.com/modelscope/mcore-bridge/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
</p>


<!-- <p align="center">
        <a href="https://mcore-bridge.readthedocs.io/en/latest/">English Documentation</a> &nbsp ｜ &nbsp <a href="https://mcore-bridge.readthedocs.io/zh-cn/latest/">中文文档</a> &nbsp
</p> -->

##  📖 目录
- [用户群](#-用户群)
- [简介](#-简介)
- [新闻](#-新闻)
- [安装](#%EF%B8%8F-安装)
- [快速开始](#-快速开始)
- [模型列表](#-模型列表)
- [License](#-license)

## ☎ 用户群

请扫描下面的二维码来加入我们的交流群：

| 微信群 |
|:-------------------------:|
| <img src="https://raw.githubusercontent.com/modelscope/ms-swift/main/docs/resources/wechat/megatron.png" width="200" height="200"> |

## 📝 简介

## 🎉 新闻
- 🎉 2026.03.30: MCore-Bridge 正式发布！为最先进的大语言模型提供 Megatron-Core 模型定义，让 Megatron 训练像 Transformers 一样简单。

## 🛠️ 安装
使用pip进行安装：
```shell
pip install mcore-bridge -U

# 使用uv
pip install uv
uv pip install mcore-bridge -U --torch-backend=auto
```

从源代码安装：
```shell
# pip install git+https://github.com/modelscope/mcore-bridge.git

git clone https://github.com/modelscope/mcore-bridge.git
cd mcore-bridge
pip install -e .

# 使用uv
uv pip install -e . --torch-backend=auto
```

## 🚀 快速开始

你需要创建以下文件（test.py），然后运行`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 test.py`。以下为使用Mcore-Bridge进行创建模型、权重加载、导出、保存的示例代码。

保存的模型，可以参考[模型卡片的示例代码](https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B)进行推理。

```python
# test env: transformers==5.2.0 megatron-core==0.16.1
import os
import torch
import torch.distributed as dist
from megatron.core import mpu
from modelscope import snapshot_download
from transformers import AutoConfig, AutoProcessor
from mcore_bridge import ModelConfig, get_mcore_model, hf_to_mcore_config

is_rank0 = int(os.getenv('RANK')) == 0
torch.cuda.set_device(f"cuda:{os.getenv('LOCAL_RANK')}")
dist.init_process_group(backend='nccl')
TP, PP, EP, ETP = 2, 2, 2, 1
mpu.initialize_model_parallel(
    tensor_model_parallel_size=TP,
    pipeline_model_parallel_size=PP,
    expert_model_parallel_size=EP,
    expert_tensor_parallel_size=ETP,
)

model_dir = snapshot_download('Qwen/Qwen3.5-35B-A3B')
hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
config_kwargs = hf_to_mcore_config(hf_config)
config = ModelConfig(
    params_dtype=torch.bfloat16,
    tensor_model_parallel_size=TP,
    pipeline_model_parallel_size=PP,
    expert_model_parallel_size=EP,
    expert_tensor_parallel_size=ETP,
    sequence_parallel=True,
    mtp_num_layers=1,
    **config_kwargs)

# 创建模型
mg_models = get_mcore_model(config)

# 加载权重
bridge = config.bridge
bridge.load_weights(mg_models, model_dir)

# 导出权重
for name, parameter in bridge.export_weights(mg_models):
    pass

# 保存权重
output_dir = 'Qwen3.5-35B-A3B-HF'
bridge.save_weights(mg_models, output_dir)
if is_rank0:
    processor.save_pretrained(output_dir)
    hf_config.save_pretrained(output_dir)
```

### 使用Peft

Mcore-Bridge完全兼容使用[Peft](https://github.com/huggingface/peft)进行LoRA训练。以下介绍如何使用peft准备PeftModel，并保存增量权重。

你需要创建以下文件（test.py），然后运行`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 test.py`。

```python
import copy
import os
import torch
import torch.distributed as dist
from megatron.core import mpu
from modelscope import snapshot_download
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoProcessor

from mcore_bridge import ModelConfig, get_mcore_model, hf_to_mcore_config, set_random_seed

is_rank0 = int(os.getenv('RANK')) == 0
torch.cuda.set_device(f"cuda:{os.getenv('LOCAL_RANK')}")
dist.init_process_group(backend='nccl')
TP, PP = 2, 2
mpu.initialize_model_parallel(
    tensor_model_parallel_size=TP,
    pipeline_model_parallel_size=PP,
)
# 为了正确随机初始化模型（全参数/LoRA），你需要设置随机种子
set_random_seed(42)

model_dir = snapshot_download('Qwen/Qwen3.5-4B')
hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
config_kwargs = hf_to_mcore_config(hf_config)
config = ModelConfig(
    params_dtype=torch.bfloat16,
    tensor_model_parallel_size=TP,
    pipeline_model_parallel_size=PP,
    sequence_parallel=True,
    **config_kwargs)

# 创建模型并加载权重
mg_models = get_mcore_model(config)
bridge = config.bridge
bridge.load_weights(mg_models, model_dir)

# 准备PeftModel并加载LoRA权重
# 多模态模型建议使用正则表达式指定target_modules
target_modules = r'^language_model.*\.(in_proj|out_proj|linear_fc1|linear_fc2|linear_qkv|linear_proj)$'
# 存储成safetensors时，需要存储hf对应的target_modules
hf_target_modules = r'^model.language_model.*\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj|gate_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)$'
lora_config = LoraConfig(task_type='CAUSAL_LM', r=8, lora_alpha=32, lora_dropout=0.05, target_modules=target_modules)
peft_models = [get_peft_model(model, lora_config) for model in mg_models]
# 可选
# bridge.load_weights(peft_models, model_dir, peft_format=True)

# 导出LoRA权重
for name, parameter in bridge.export_weights(mg_models, peft_format=True):
    pass

# 保存LoRA权重
output_dir = 'Qwen3.5-4B-LoRA'
bridge.save_weights(mg_models, output_dir, peft_format=True)
if is_rank0:
    hf_lora_config = copy.copy(lora_config)
    hf_lora_config.target_modules = hf_target_modules
    hf_lora_config.save_pretrained(output_dir)
```

使用存储下来的LoRA权重：
```python
from transformers import Qwen3_5ForConditionalGeneration
from modelscope import snapshot_download
from peft import PeftModel

model_dir = snapshot_download('Qwen/Qwen3.5-4B')
model = Qwen3_5ForConditionalGeneration.from_pretrained(model_dir)
peft_model = PeftModel.from_pretrained(model, 'Qwen3.5-4B-LoRA')
```

## ✨ 模型列表

以下为MCore-Bridge支持的模型列表：

| 系列     | model_type                                                   |
| -------- | ------------------------------------------------------------ |
| Qwen     | qwen2, qwen2_moe<br />qwen2_vl, qwen2_5_vl, qwen2_5_omni<br />qwen3, qwen3_moe<br />qwen3_vl, qwen3_vl_moe, qwen3_omni_moe<br />qwen3_next, qwen3_5, qwen3_5_moe |
| DeepSeek | deepseek_v3, deepseek_v32                                    |
| GLM      | glm4, glm4_moe, glm4_moe_lite<br />glm4v, glm4v_moe, <br />glm_moe_dsa |
| MiniMax  | minimax_m2                                                   |
| Kimi     | kimi_k2, kimi_vl                                             |
| InternLM | internlm3, internvl_chat, internvl                           |
| Ovis     | ovis2_5                                                      |
| Llama    | llama, llama4                                                |
| GPT-OSS  | gpt_oss                                                      |
| ERNIE    | ernie4_5, ernie4_5_moe                                       |
| MiMo     | mimo                                                         |
| Dots     | dots1                                                        |
| OLMoE    | olmoe                                                        |


## 🏛 License

本框架使用[Apache License (Version 2.0)](https://github.com/modelscope/mcore-bridge/blob/master/LICENSE)进行许可。模型和数据集请查看原资源页面并遵守对应License。


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/mcore-bridge&type=Date)](https://star-history.com/#modelscope/mcore-bridge&Date)
