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
<a href="https://mcore-bridge.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/docs-latest-blue.svg"></a>
<a href="https://pypi.org/project/mcore-bridge/"><img src="https://badge.fury.io/py/mcore-bridge.svg"></a>
<a href="https://github.com/modelscope/mcore-bridge/blob/main/LICENSE"><img src="https://img.shields.io/github/license/modelscope/mcore-bridge"></a>
<a href="https://pepy.tech/project/mcore-bridge"><img src="https://pepy.tech/badge/mcore-bridge"></a>
<a href="https://github.com/modelscope/mcore-bridge/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
</p>


<p align="center">
        <a href="https://mcore-bridge.readthedocs.io/en/latest/">English Documentation</a> &nbsp ｜ &nbsp <a href="https://mcore-bridge.readthedocs.io/zh-cn/latest/">中文文档</a> &nbsp
</p>

## 📖 Table of Contents
- [Groups](#-Groups)
- [Introduction](#-introduction)
- [News](#-news)
- [Installation](#%EF%B8%8F-installation)
- [Quick Start](#-quick-Start)
- [Usage](#-Usage)
- [License](#-License)


## ☎ Groups

You can contact us and communicate with us by adding our group:

WeChat Group |
:-------------------------:
<img src="https://raw.githubusercontent.com/modelscope/ms-swift/main/docs/resources/wechat/megatron.png" width="200" height="200">

## 📝 Introduction

## 🎉 News
- 🎉 2025.03.23: MCore-Bridge is released! Making Megatron training as simple as Transformers, providing Megatron-Core model definitions for state-of-the-art large language models.

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
import os
import torch
import torch.distributed as dist
from megatron.core import mpu
from modelscope import snapshot_download
from transformers import AutoConfig, AutoProcessor
from mcore_bridge import ModelConfig, get_mcore_model, hf_to_mcore_config

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
    processor=processor,
    hf_config=hf_config,
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
processor.save_pretrained(output_dir)
hf_config.save_pretrained(output_dir)
```

## ✨ Usage


## 🏛 License

This framework is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/mcore-bridge/blob/master/LICENSE). For models and datasets, please refer to the original resource page and follow the corresponding License.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/mcore-bridge&type=Date)](https://star-history.com/#modelscope/mcore-bridge&Date)
