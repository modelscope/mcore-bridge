# MCore-Bridge: Making Megatron training as simple as Transformers

<!-- <p align="center">
    <br>
    <img src="asset/banner.png"/>
    <br>
<p> -->

<p align="center">
    <b>Providing Megatron-Core model definitions for state-of-the-art large models</b>
</p>

<p align="center">
<a href="https://modelscope.cn">ModelScope</a>
<br>
        <a href="README_zh.md">中文</a> &nbsp ｜ &nbsp English &nbsp
</p>


<p align="center">
<img src="https://img.shields.io/badge/python-3.12-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A52.0-orange.svg">
<a href="https://github.com/NVIDIA/Megatron-LM/"><img src="https://img.shields.io/badge/megatron--core-%E2%89%A50.15-76B900.svg"></a>
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
- [News](#-news)
- [Installation](#%EF%B8%8F-installation)
- [Model List](#-Model-List)
- [Quick Start](#-quick-Start)
- [License](#-License)


## ☎ Groups

You can contact us and communicate with us by adding our group:

| WeChat Group |
|:-------------------------:|
| <img src="https://raw.githubusercontent.com/modelscope/ms-swift/main/docs/resources/wechat/megatron.png" width="200" height="200"> |

## 📝 Introduction

**mcore-bridge** is a large language model and multimodal large model definition library built on the Megatron-Core ecosystem, developed by the ModelScope community. It currently supports 300+ text-only models and 200+ multimodal models, including large language models such as Qwen3-Next, GLM-5.1, DeepSeek-V4, Minimax-2.7, Kimi-K2.5, and GPT-OSS, as well as multimodal large models such as Qwen3.5, Qwen3-Omni, Gemma4, GLM4.6-V, InternVL3.5, and Ovis2.5.

------

**Why Choose mcore-bridge?**

- **Model Coverage**: Supports 300+ text-only large language models and 200+ multimodal large models, with Day 0 support for popular models.
- **Hardware Support**: Compatible with a wide range of hardware platforms, including A10/A100/H100/B200, RTX series, and domestic hardware such as Ascend NPU.
- **Training Methods**: Supports both full-parameter training and LoRA training, with compatibility with the PEFT ecosystem.
- **Parallelism Techniques**: Supports multiple parallelism strategies provided by Megatron-Core, including tensor parallelism, pipeline parallelism, sequence parallelism, context parallelism, expert parallelism, and virtual pipeline parallelism.
- **Multimodal Capabilities**: Supports multimodal FP8 training, MTP, sequence padding-free, and packing features.
- **Task Types**: Supports a variety of task types, including Causal LM, sequence classification, Embedding, and Reranker.
- **Ecosystem Compatibility**: Supports direct loading and saving of LoRA/full-parameter safetensors weights, with compatibility with mainstream inference frameworks such as Transformers, vLLM, and SGLang.

------

**Related Documentation:**

- [Integrating Mcore-Bridge with ms-swift](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Mcore-Bridge.html)
- [Supported Models List](https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html)
- [Custom Megatron Models](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Custom-Model.html)
- [Qwen3.5 Training Best Practices](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_5-Best-Practice.html)


## 🎉 News
- 🎉 2026.03.30: MCore-Bridge is released! Providing Megatron-Core model definitions for state-of-the-art large models and making Megatron training as simple as Transformers.

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

Recommended Runtime Environment:

|        | Range | Recommended | Notes |
|--------------|--------------|-------------|--------------------|
| python       | >=3.10        | 3.12        |                    |
| cuda         |              | cuda12.8/13.0      |                    |
| torch        | >=2.0        | 2.8.0/2.11.0       |                    |
| transformer-engine    | >=2.3       |  2.14.1    |                  |
| apex |         | 0.1 | Optional |
| megatron-core    |   >=0.15,<0.18    | 0.17.0      |                  |
| flash-attn    |        | 2.8.3/3.0.0b1   |     Optional      |
| transformers | >=4.33       | 4.57.6/5.8.1   |                    |
| modelscope   | >=1.23       |             |                    |
| peft         | >=0.11,<0.20 |             |      LoRA          |


## ✨ Model List

The following is the list of models supported by MCore-Bridge:


**text-only large models:**

| Series     | model_type                                            |
| -------- | ------------------------------------------------------------ |
| Qwen     | qwen2, qwen2_moe<br />qwen3, qwen3_moe, qwen3_next |
| DeepSeek | deepseek_v3, deepseek_v32, deepseek_v4                        |
| GLM      | glm4, glm4_moe, glm4_moe_lite<br />glm_moe_dsa |
| MiniMax  | minimax_m2                                                   |
| Kimi     | kimi_k2, kimi_k25                                   |
| Bailing  | bailing_moe, bailing_hybrid                             |
| InternLM | internlm3                           |
| Llama    | llama                                                |
| GPT-OSS  | gpt_oss                                                      |
| Hunyuan  | hy_v3                                                        |
| ERNIE    | ernie4_5, ernie4_5_moe                                       |
| MiMo     | mimo                                                         |
| Dots     | dots1                                                        |
| OLMoE    | olmoe                                                        |

**multimodal large models:**
| Series     | model_type                                            |
| -------- | ------------------------------------------------------------ |
| Qwen     | qwen2_vl, qwen2_5_vl, qwen2_5_omni<br />qwen3_vl, qwen3_vl_moe, qwen3_omni_moe, qwen3_asr<br />qwen3_5, qwen3_5_moe |
| Gemma     | gemma4                                   |
| GLM      | glm4v, glm4v_moe |
| Kimi     | kimi_vl                                   |
| InternVL | internvl_chat, internvl                           |
| Ovis     | ovis2_5                                                      |
| Llama    | llama4                                                |
| Llava    | llava-onevision                                        |



## 🚀 Quick Start

How to use MCore-Bridge for training can be referred to the [ms-swift project](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Mcore-Bridge.html). Here we introduce how to use MCore-Bridge programmatically.

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

### Minimal forward example

Mcore-Bridge integrates seamlessly with the ms-swift template for model training. You can also replace the ms-swift template module with a custom data processing pipeline to suit your own workflow.

The following provides a minimal example demonstrating how to perform a forward pass and compute the loss using a model created with Mcore-Bridge, helping users quickly integrate Mcore-Bridge into other projects.

Create the following file (test.py) and run it with: `CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 test.py`.

```python
import os
import torch
import torch.distributed as dist
from megatron.core import mpu
from modelscope import snapshot_download
from swift import get_processor, get_template
from swift.megatron.utils import get_packed_seq_params, get_padding_to
from swift.utils import to_device

from mcore_bridge import ModelConfig, get_mcore_model, hf_to_mcore_config, set_random_seed

data = {
    'messages': [{
        'role': 'user',
        'content': '<image>describe the image.'
    }, {
        'role':
        'assistant',
        'content':
        'The image depicts a close-up of a kitten with striking features. '
        'The kitten has a white and gray coat with distinct black stripes, '
        'particularly noticeable on its face and ears. Its eyes are large '
        'and expressive, with a captivating blue hue that stands out against '
        "the darker fur around them. The kitten's nose is small and pink, "
        'and it has long, delicate whiskers extending from either side of its mouth. '
        "The background is blurred, drawing attention to the kitten's face and "
        'making it the focal point of the image. The overall impression is '
        'one of cuteness and charm.'
    }],
    'images': ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png']
}


def forward_mg_model(mg_model, template):
    template.use_megatron = True
    template.set_mode('train')
    inputs = template.encode(data, return_length=True)
    mg_inputs = to_device(template.data_collator([inputs], padding_to=get_padding_to(mg_model.config)), 'cuda')
    text_position_ids = mg_inputs.pop('text_position_ids', None)
    if text_position_ids is None:
        text_position_ids = mg_inputs.get('position_ids')
    for key in ['num_samples', 'attention_mask_2d', 'loss_scale']:
        mg_inputs.pop(key, None)
    if template.padding_free:
        mg_inputs['packed_seq_params'] = get_packed_seq_params(text_position_ids)
    mg_inputs['labels'] = torch.roll(mg_inputs['labels'], -1, dims=-1)
    loss = mg_model(**mg_inputs)
    loss_mask = mg_inputs['labels'] != -100
    loss = loss * loss_mask
    return loss.sum() / loss_mask.sum()


torch.cuda.set_device(f"cuda:{os.getenv('LOCAL_RANK')}")
dist.init_process_group(backend='nccl')
TP, PP, EP, ETP = 2, 1, 2, 1
mpu.initialize_model_parallel(
    tensor_model_parallel_size=TP,
    pipeline_model_parallel_size=PP,
    expert_model_parallel_size=EP,
    expert_tensor_parallel_size=ETP,
)
set_random_seed(42)

model_dir = snapshot_download('Qwen/Qwen3.5-35B-A3B')
template = get_template(get_processor(model_dir), padding_free=True)
config_kwargs = hf_to_mcore_config(template.config)
config = ModelConfig(
    params_dtype=torch.bfloat16,
    tensor_model_parallel_size=TP,
    pipeline_model_parallel_size=PP,
    expert_model_parallel_size=EP,
    expert_tensor_parallel_size=ETP,
    sequence_parallel=True,
    mtp_num_layers=1,
    **config_kwargs)

mg_model = get_mcore_model(config)[0]
mg_model.cuda()
config.bridge.load_weights([mg_model], model_dir)
loss = forward_mg_model(mg_model, template)
print(f'loss: {loss}')  # loss: 0.8161308169364929
```


## 🏛 License

This framework is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/mcore-bridge/blob/master/LICENSE). For models and datasets, please refer to the original resource page and follow the corresponding License.
