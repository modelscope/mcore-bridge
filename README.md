# MCore-Bridge: Making Megatron training as simple as Transformers

<p align="center">
    <br>
    <img src="asset/banner.png"/>
    <br>
<p>

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
<img src="asset/wechat.png" width="200" height="200">

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

```python
import torch

from mcore_bridge import (
    ModelConfig, get_mcore_model, hf_to_mcore_config, get_peft_model
)
from transformers import AutoConfig
from modelscope import snapshot_download
from peft import LoraConfig

model_dir = snapshot_download('Qwen/Qwen3.5-4B')
hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
config_kwargs = hf_to_mcore_config(hf_config)
config = ModelConfig(
    model=model_dir,
    torch_dtype=torch.bfloat16,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=2,
    sequence_parallel=True,
    **config_kwargs)
mg_models = get_mcore_model(config)

# Load weights
bridge = config.bridge
bridge.load_weights(mg_models, model_dir)
# Prepare LoRA
lora_config = LoraConfig(...)
peft_models = [get_peft_model(mg_model, lora_config) for mg_model in mg_models]
print(f'peft_model: {peft_models[0]}')
# Load LoRA (Optional)
# bridge.load_weights(peft_models, 'adapter-path', peft_format=True)

# Export weights
for name, parameter in bridge.export_weights(peft_models, peft_format=True):
    pass
# Save weights
bridge.save_weights(peft_models, 'output/Qwen3.5-4B-lora', peft_format=True)
```

## ✨ Usage


## 🏛 License

This framework is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/mcore-bridge/blob/master/LICENSE). For models and datasets, please refer to the original resource page and follow the corresponding License.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/mcore-bridge&type=Date)](https://star-history.com/#modelscope/mcore-bridge&Date)
