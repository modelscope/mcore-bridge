"""Isolate the Mamba2 mixer: run HF's and mcore's on identical weights + input.

Loads only one Mamba layer's weights out of the checkpoint, so this is cheap.
    torchrun --nproc_per_node=1 tests/_nemotron_mixerdiff.py
"""
import json
import os

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from safetensors import safe_open

MODEL_PATH = ('/root/.cache/modelscope/hub/models/nv-community/'
              'EA-NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16-07202026')


def load_layer0_mixer():
    index = json.load(open(os.path.join(MODEL_PATH, 'model.safetensors.index.json')))
    weight_map = index['weight_map']
    out = {}
    for key, shard in weight_map.items():
        if key.startswith('backbone.layers.0.mixer.'):
            with safe_open(os.path.join(MODEL_PATH, shard), framework='pt') as f:
                out[key[len('backbone.layers.0.mixer.'):]] = f.get_tensor(key).cuda()
    return out


def main():
    seq_len = int(os.environ.get('SEQ_LEN', 8))
    dist.init_process_group('nccl')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    mpu.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(1234)

    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    w = load_layer0_mixer()
    print('RES loaded mixer keys:', sorted(w))

    torch.manual_seed(0)
    # HF wants [b, s, h]; mcore wants [s, b, h].
    x_bsh = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16, device='cuda')

    # ---- HF mixer ----
    # Go through transformers' dynamic-module loader so the file's relative imports resolve.
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    hf_mixer_cls = get_class_from_dynamic_module(
        'modeling_nemotron_h.NemotronHMamba2Mixer', MODEL_PATH)
    hf_mixer = hf_mixer_cls(hf_config, layer_idx=0).cuda().to(torch.bfloat16).eval()
    missing, unexpected = hf_mixer.load_state_dict(w, strict=False)
    print(f'RES hf_mixer missing={list(missing)} unexpected={list(unexpected)}')
    with torch.no_grad():
        hf_out = hf_mixer(x_bsh).float()
    print(f'RES hf_out {tuple(hf_out.shape)} mean={hf_out.mean():.6f} std={hf_out.std():.6f}')

    # ---- mcore mixer ----
    import mcore_bridge.model.gpts  # noqa: F401
    from mcore_bridge.config.model_config import ModelConfig
    from mcore_bridge.config.parser import hf_to_mcore_config
    overrides = hf_to_mcore_config(hf_config)
    overrides.update(num_layers=1, hybrid_layer_pattern='M', moe_layer_freq='[0]',
                     params_dtype=torch.bfloat16, bf16=True, mtp_num_layers=None)
    cfg = ModelConfig(**overrides)

    from megatron.core.extensions.transformer_engine import (TEColumnParallelLinear,
                                                             TERowParallelLinear)
    from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
    mg_mixer = MambaMixer(
        cfg,
        MambaMixerSubmodules(in_proj=TEColumnParallelLinear, out_proj=TERowParallelLinear),
        d_model=cfg.hidden_size,
        layer_number=1,
        pg_collection=_pg(),
    ).cuda().to(torch.bfloat16).eval()

    mg_sd = {
        'in_proj.weight': w['in_proj.weight'],
        'conv1d_weight': w['conv1d.weight'],
        'conv1d_bias': w['conv1d.bias'],
        'A_log': w['A_log'],
        'D': w['D'],
        'dt_bias': w['dt_bias'],
        'norm.weight': w['norm.weight'],
        'out_proj.weight': w['out_proj.weight'],
    }
    incompat = mg_mixer.load_state_dict(mg_sd, strict=False)
    print(f'RES mg_mixer missing={list(incompat.missing_keys)} '
          f'unexpected={list(incompat.unexpected_keys)}')

    x_sbh = x_bsh.transpose(0, 1).contiguous()
    with torch.no_grad():
        mg_out, mg_bias = mg_mixer(x_sbh)
    mg_out = mg_out.float().transpose(0, 1)  # -> [b, s, h]
    if mg_bias is not None:
        mg_out = mg_out + mg_bias.float()
    print(f'RES mg_out {tuple(mg_out.shape)} mean={mg_out.mean():.6f} std={mg_out.std():.6f}')

    d = (hf_out - mg_out).abs()
    print(f'RES mixer max_abs_diff={d.max():.6f} mean_abs_diff={d.mean():.6f} '
          f'rel={d.max() / hf_out.abs().max():.6f}')

    # ---- Now the FULL block: norm + mixer + residual ----
    # Load layer0's outer norm too.
    import json as _json
    idx = _json.load(open(os.path.join(MODEL_PATH, 'model.safetensors.index.json')))
    nk = 'backbone.layers.0.norm.weight'
    with safe_open(os.path.join(MODEL_PATH, idx['weight_map'][nk]), framework='pt') as f:
        outer_norm_w = f.get_tensor(nk).cuda()

    hf_block_cls = get_class_from_dynamic_module('modeling_nemotron_h.NemotronHBlock', MODEL_PATH)
    hf_block = hf_block_cls(hf_config, layer_idx=0).cuda().to(torch.bfloat16).eval()
    bsd = {f'mixer.{k}': v for k, v in w.items()}
    bsd['norm.weight'] = outer_norm_w
    inc = hf_block.load_state_dict(bsd, strict=False)
    print(f'RES hf_block missing={list(inc.missing_keys)} unexpected={list(inc.unexpected_keys)}')
    with torch.no_grad():
        hf_blk = hf_block(x_bsh)
    hf_blk = (hf_blk[0] if isinstance(hf_blk, tuple) else hf_blk).float()

    from mcore_bridge.model.gpts.nemotron_h import _build_mamba_layer_cls, NemotronHLoader
    loader = NemotronHLoader(cfg)
    mspec = loader._get_mamba_layer_spec()
    from megatron.core.transformer.spec_utils import build_module
    mg_layer = build_module(mspec, config=cfg, layer_number=1,
                           pg_collection=_pg(), vp_stage=None).cuda().to(torch.bfloat16).eval()
    lsd = dict(mg_sd)
    lsd = {f'mixer.{k}': v for k, v in mg_sd.items()}
    lsd['norm.weight'] = outer_norm_w
    inc2 = mg_layer.load_state_dict(lsd, strict=False)
    print(f'RES mg_layer missing={[k for k in inc2.missing_keys if "_extra_state" not in k]} '
          f'unexpected={list(inc2.unexpected_keys)}')
    with torch.no_grad():
        out = mg_layer(x_sbh)
    mg_blk = (out[0] if isinstance(out, tuple) else out).float().transpose(0, 1)
    d2 = (hf_blk - mg_blk).abs()
    print(f'RES BLOCK max_abs_diff={d2.max():.6f} mean_abs_diff={d2.mean():.6f} '
          f'rel={d2.max() / hf_blk.abs().max():.6f}')


def _pg():
    from megatron.core.process_groups_config import ProcessGroupCollection
    return ProcessGroupCollection.use_mpu_process_groups()


if __name__ == '__main__':
    main()
