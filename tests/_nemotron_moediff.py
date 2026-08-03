"""Compare a single 'E' (MoE) block: HF NemotronHBlock vs mcore TransformerLayer.

The Mamba block already matches in isolation (see _nemotron_mixerdiff.py), so this
checks the other layer type. Loads only layer 1's weights.

    torchrun --nproc_per_node=1 tests/_nemotron_moediff.py
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
LAYER = int(os.environ.get('LAYER', 1))  # layer 1 is 'E' in MEMEM*...


def load_layer(prefix):
    index = json.load(open(os.path.join(MODEL_PATH, 'model.safetensors.index.json')))
    out = {}
    for key, shard in index['weight_map'].items():
        if key.startswith(prefix):
            with safe_open(os.path.join(MODEL_PATH, shard), framework='pt') as f:
                out[key[len(prefix):]] = f.get_tensor(key).cuda()
    return out


def _pg():
    from megatron.core.process_groups_config import ProcessGroupCollection
    return ProcessGroupCollection.use_mpu_process_groups()


def main():
    seq_len = int(os.environ.get('SEQ_LEN', 8))
    dist.init_process_group('nccl')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    mpu.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(1234)

    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # NEMOTRONH_ATTENTION_CLASSES is keyed by _attn_implementation; standalone block
    # construction skips the usual model-level default, so set it explicitly.
    hf_config._attn_implementation = os.environ.get('HF_ATTN', 'eager')
    pattern = hf_config.hybrid_override_pattern
    print(f'RES layer{LAYER} type={pattern[LAYER]!r}')

    w = load_layer(f'backbone.layers.{LAYER}.')
    print(f'RES n_weights={len(w)}')

    torch.manual_seed(0)
    if os.environ.get('REAL_EMB'):
        # Use the checkpoint's real embedding rows: their magnitude is far from N(0,1),
        # and Mamba's conv/SSM path is scale sensitive.
        ek = 'backbone.embeddings.weight'
        _idx = json.load(open(os.path.join(MODEL_PATH, 'model.safetensors.index.json')))
        with safe_open(os.path.join(MODEL_PATH, _idx['weight_map'][ek]), framework='pt') as f:
            emb = f.get_tensor(ek).cuda()
        ids = torch.randint(0, emb.shape[0], (1, seq_len), device='cuda')
        x_bsh = emb[ids].to(torch.bfloat16)
        print(f'RES using REAL embedding, std={x_bsh.float().std():.5f}')
    else:
        x_bsh = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16, device='cuda')
        print(f'RES using randn input, std={x_bsh.float().std():.5f}')

    # ---- HF block ----
    hf_block_cls = get_class_from_dynamic_module('modeling_nemotron_h.NemotronHBlock', MODEL_PATH)
    hf_block = hf_block_cls(hf_config, layer_idx=LAYER).cuda().to(torch.bfloat16).eval()
    inc = hf_block.load_state_dict(w, strict=False)
    print(f'RES hf missing={list(inc.missing_keys)[:5]} unexpected={list(inc.unexpected_keys)[:5]}')
    with torch.no_grad():
        hf_out = hf_block(x_bsh)
    hf_out = (hf_out[0] if isinstance(hf_out, tuple) else hf_out).float()
    print(f'RES hf_out mean={hf_out.mean():.6f} std={hf_out.std():.6f}')

    # ---- MCore layer via the real Loader spec ----
    import mcore_bridge.model.gpts  # noqa: F401
    from mcore_bridge.config.model_config import ModelConfig
    from mcore_bridge.config.parser import hf_to_mcore_config
    from mcore_bridge.model.gpts.nemotron_h import NemotronHLoader

    ch = pattern[LAYER]
    overrides = hf_to_mcore_config(hf_config)
    overrides.update(num_layers=1, hybrid_layer_pattern=ch,
                     moe_layer_freq='[1]' if ch == 'E' else '[0]',
                     params_dtype=torch.bfloat16, bf16=True, mtp_num_layers=None)
    from megatron.core.transformer.enums import AttnBackend
    overrides['attention_backend'] = AttnBackend.flash
    cfg = ModelConfig(**overrides)

    loader = NemotronHLoader(cfg)
    spec = loader.get_transformer_layer_spec()
    from megatron.core.transformer.spec_utils import build_module
    mg_layer = build_module(spec.layer_specs[0], config=cfg, layer_number=1,
                            pg_collection=_pg(), vp_stage=None).cuda().to(torch.bfloat16).eval()
    print('RES mg params:', sorted(n for n, _ in mg_layer.named_parameters())[:12])

    # Drive the real Bridge so the mapping under test is exercised.
    class Lazy:
        def __init__(self, t):
            self.t = t

        def load(self):
            return self.t

    sd = {f'backbone.layers.0.{k}': Lazy(v) for k, v in w.items()}
    cfg.bridge._set_layer_state(mg_layer, sd, 'backbone.layers.', 0, True)

    x_sbh = x_bsh.transpose(0, 1).contiguous()
    attn_mask = torch.tril(
        torch.ones((1, 1, seq_len, seq_len), device='cuda', dtype=torch.bool)).logical_not()
    with torch.no_grad():
        out = mg_layer(x_sbh, attention_mask=attn_mask)
    mg_out = (out[0] if isinstance(out, tuple) else out).float().transpose(0, 1)
    print(f'RES mg_out mean={mg_out.mean():.6f} std={mg_out.std():.6f}')

    d = (hf_out - mg_out).abs()
    print(f'RES BLOCK({ch}) max_abs_diff={d.max():.6f} mean_abs_diff={d.mean():.6f} '
          f'rel={d.max() / hf_out.abs().max():.6f}')


if __name__ == '__main__':
    main()
