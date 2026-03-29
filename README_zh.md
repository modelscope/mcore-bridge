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
- [模型列表](#-模型列表)
- [License](#-license)

## ☎ 用户群

请扫描下面的二维码来加入我们的交流群：

| 微信群 |
|:-------------------------:|
| <img src="https://raw.githubusercontent.com/modelscope/ms-swift/main/docs/resources/wechat/megatron.png" width="200" height="200"> |

## 📝 简介

## 🎉 新闻
- 🎁 2026.04.01: MCore-Bridge 正式发布！为最先进的大语言模型提供 Megatron-Core 模型定义，让 Megatron 训练像 Transformers 一样简单。

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
processor.save_pretrained(output_dir)
hf_config.save_pretrained(output_dir)
```

## ✨ 模型列表

以下为MCore-Bridge支持的模型列表：

| 系列     | model_type                                                   |
| -------- | ------------------------------------------------------------ |
| Qwen     | qwen2, qwen2_moe<br />qwen2_vl, qwen2_5_vl, qwen2_5_omni<br />ovis2_5<br />qwen3, qwen3_moe<br />qwen3_vl, qwen3_vl_moe, qwen3_omni_moe<br />qwen3_next, qwen3_5, qwen3_5_moe |
| DeepSeek | deepseek_v3, deepseek_v32                                    |
| GLM      | glm4, glm4_moe, glm4_moe_lite<br />glm4v, glm4v_moe, <br />glm_moe_dsa |
| MiniMax  | minimax_m2                                                   |
| Kimi     | kimi_k2, kimi_vl                                             |
| InternLM | internlm3, internvl_chat, internvl                           |
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
