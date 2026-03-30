# MCore-Bridge: Making Megatron training as simple as Transformers

<!-- <p align="center">
    <br>
    <img src="asset/banner.png"/>
    <br>
<p> -->

<p align="center">
    <b>Providing Megatron-Core model definitions for state-of-the-art large language models</b>
</p>

<p align="center">
<a href="https://modelscope.cn/home">ModelScope Community Website</a>
<br>
        <a href="README_zh.md">中文</a> &nbsp ｜ &nbsp English &nbsp
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

## 📖 Table of Contents
- [Groups](#-Groups)
- [Introduction](#-introduction)
- [News](#-news)
- [Installation](#%EF%B8%8F-installation)
- [Quick Start](#-quick-Start)
- [Model List](#-Model-List)
- [License](#-License)


## ☎ Groups

You can contact us and communicate with us by adding our group:

| WeChat Group |
|:-------------------------:|
| <img src="https://raw.githubusercontent.com/modelscope/ms-swift/main/docs/resources/wechat/megatron.png" width="200" height="200"> |

## 📝 Introduction

## 🎉 News
- 🎉 2026.03.30: MCore-Bridge is released! Providing Megatron-Core model definitions for state-of-the-art large language models and making Megatron training as simple as Transformers.

## 🛠️ Installation
To install using pip:
```shell
pip install mcore-bridge -U

# Using uv
pip install uv
uv pip install mcore-bridge -U --torch-backend=auto
```

To install from source:
```shell
# pip install git+https://github.com/modelscope/mcore-bridge.git

git clone https://github.com/modelscope/mcore-bridge.git
cd mcore-bridge
pip install -e .

# Using uv
uv pip install -e . --torch-backend=auto
```

## 🚀 Quick Start

You need to create the following file (test.py), then run `CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 test.py`. Below is sample code demonstrating how to use Mcore-Bridge for model creation, weight loading, export, and saving.

The saved model can be used for inference by referring to the [example code in the model card](https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B).

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

# Create model
mg_models = get_mcore_model(config)

# Load weights
bridge = config.bridge
bridge.load_weights(mg_models, model_dir)

# Export weights
for name, parameter in bridge.export_weights(mg_models):
    pass

# Save weights
output_dir = 'Qwen3.5-35B-A3B-HF'
bridge.save_weights(mg_models, output_dir)
if is_rank0:
    processor.save_pretrained(output_dir)
    hf_config.save_pretrained(output_dir)
```

### Using Peft

Mcore-Bridge is fully compatible with [Peft](https://github.com/huggingface/peft) for LoRA training. The following introduces how to use Peft to prepare a PeftModel and save the incremental weights.

You need to create the following file (test.py), then run `CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 test.py`.

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
# To correctly initialize the model randomly (full parameters/LoRA)
# you need to set the random seed.
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

# Create model and load weights
mg_models = get_mcore_model(config)
bridge = config.bridge
bridge.load_weights(mg_models, model_dir)

# Prepare PeftModel and load LoRA weights
# For multimodal models, it is recommended to use regex to specify target_modules
target_modules = r'^language_model.*\.(in_proj|out_proj|linear_fc1|linear_fc2|linear_qkv|linear_proj)$'
# When saving as safetensors, you need to store the corresponding HF target_modules
hf_target_modules = r'^model.language_model.*\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj|gate_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)$'
lora_config = LoraConfig(task_type='CAUSAL_LM', r=8, lora_alpha=32, lora_dropout=0.05, target_modules=target_modules)
peft_models = [get_peft_model(model, lora_config) for model in mg_models]
# Optional
# bridge.load_weights(peft_models, model_dir, peft_format=True)

# Export LoRA weights
for name, parameter in bridge.export_weights(mg_models, peft_format=True):
    pass

# Save LoRA weights
output_dir = 'Qwen3.5-4B-LoRA'
bridge.save_weights(mg_models, output_dir, peft_format=True)
if is_rank0:
    hf_lora_config = copy.copy(lora_config)
    hf_lora_config.target_modules = hf_target_modules
    hf_lora_config.save_pretrained(output_dir)
```

Using the saved LoRA weights:

```python
from transformers import Qwen3_5ForConditionalGeneration
from modelscope import snapshot_download
from peft import PeftModel

model_dir = snapshot_download('Qwen/Qwen3.5-4B')
model = Qwen3_5ForConditionalGeneration.from_pretrained(model_dir)
peft_model = PeftModel.from_pretrained(model, 'Qwen3.5-4B-LoRA')
```

## ✨ Model List

The following is the list of models supported by MCore-Bridge:

| Series     | model_type                                                   |
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

This framework is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/mcore-bridge/blob/master/LICENSE). For models and datasets, please refer to the original resource page and follow the corresponding License.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/mcore-bridge&type=Date)](https://star-history.com/#modelscope/mcore-bridge&Date)
