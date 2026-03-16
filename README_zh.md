# MCore-Bridge: 让 Megatron 训练像 Transformers 一样简单

<p align="center">
    <br>
    <img src="asset/banner.png"/>
    <br>
<p>

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
<a href="https://mcore-bridge.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/docs-latest-blue.svg"></a>
<a href="https://pypi.org/project/mcore-bridge/"><img src="https://badge.fury.io/py/mcore-bridge.svg"></a>
<a href="https://github.com/modelscope/mcore-bridge/blob/main/LICENSE"><img src="https://img.shields.io/github/license/modelscope/mcore-bridge"></a>
<a href="https://pepy.tech/project/mcore-bridge"><img src="https://pepy.tech/badge/mcore-bridge"></a>
<a href="https://github.com/modelscope/mcore-bridge/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
</p>


<p align="center">
        <a href="https://mcore-bridge.readthedocs.io/en/latest/">English Documentation</a> &nbsp ｜ &nbsp <a href="https://mcore-bridge.readthedocs.io/zh-cn/latest/">中文文档</a> &nbsp
</p>

##  📖 目录
- [用户群](#-用户群)
- [简介](#-简介)
- [新闻](#-新闻)
- [安装](#%EF%B8%8F-安装)
- [快速开始](#-快速开始)
- [如何使用](#-如何使用)
- [License](#-license)

## ☎ 用户群

请扫描下面的二维码来加入我们的交流群：

微信群 |
:-------------------------:
<img src="asset/wechat.png" width="200" height="200">

## 📝 简介

## 🎉 新闻
- 🎁 2026.03.23: Mcore-Bridge发布，让Megatron训练像transformers一样简单，为最先进的大语言模型提供 Megatron-Core 模型定义。

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

# 加载权重
bridge = config.bridge
bridge.load_weights(mg_models, model_dir)
# 准备LoRA
lora_config = LoraConfig(...)
peft_models = [get_peft_model(mg_model, lora_config) for mg_model in mg_models]
print(f'peft_model: {peft_models[0]}')
# 加载LoRA（可选）
# bridge.load_weights(peft_models, 'adapter-path', peft_format=True)
# 导出权重
for name, parameter in bridge.export_weights(peft_models, peft_format=True):
    pass
# 保存权重
bridge.save_weights(peft_models, 'output/Qwen3.5-4B-lora', peft_format=True)
```

## ✨ 如何使用


## 🏛 License

本框架使用[Apache License (Version 2.0)](https://github.com/modelscope/mcore-brige/blob/master/LICENSE)进行许可。模型和数据集请查看原资源页面并遵守对应License。


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/mcore-brige&type=Date)](https://star-history.com/#modelscope/mcore-brige&Date)
