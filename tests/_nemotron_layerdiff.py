"""Layer-by-layer hidden-state diff between HF and MCore (single process, one load each).

Bisects where the two stacks diverge instead of only comparing final logits.
Both models are built with the same small layer count so this fits comfortably in memory.

    NUM_LAYERS=4 SEQ_LEN=8 torchrun --nproc_per_node=1 tests/_nemotron_layerdiff.py
"""
import os

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

MODEL_PATH = ('/root/.cache/modelscope/hub/models/nv-community/'
              'EA-NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16-07202026')


def _restore_clobbered_weights(hf_model):
    """Undo two HF `_init_weights` bugs that discard trained weights.

    This checkpoint's remote code overwrites weights AFTER they are loaded:

    * `dt_bias`: `module.dt_bias.copy_(inv_dt)` with a random `inv_dt`; the
      `_no_reinit = True` marker is set afterwards and never checked.
    * `out_proj.weight`: because `rescale_prenorm_residual=True`, it runs
      `kaiming_uniform_` then divides by sqrt(num_layers) -- unconditionally.

    Without restoring both, HF is running partly random weights and is useless
    as a numerical reference.
    """
    import json

    from safetensors import safe_open
    index = json.load(open(os.path.join(MODEL_PATH, 'model.safetensors.index.json')))
    weight_map = index['weight_map']
    restored = 0
    for name, param in hf_model.named_parameters():
        key = name.replace('model.backbone', 'backbone')
        if not (key.endswith('mixer.dt_bias') or key.endswith('mixer.out_proj.weight')):
            continue
        if key not in weight_map:
            continue
        with safe_open(os.path.join(MODEL_PATH, weight_map[key]), framework='pt') as f:
            tensor = f.get_tensor(key)
        with torch.no_grad():
            param.copy_(tensor.to(param.device, param.dtype))
        restored += 1
    print(f'RES restored {restored} clobbered weights (HF _init_weights bug)')


def main():
    seq_len = int(os.environ.get('SEQ_LEN', 8))
    n_layers = int(os.environ.get('NUM_LAYERS', 4))

    dist.init_process_group('nccl')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    mpu.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(1234)

    from transformers import AutoConfig, AutoModelForCausalLM

    import mcore_bridge.model.gpts  # noqa: F401
    from mcore_bridge.config.model_config import ModelConfig
    from mcore_bridge.config.parser import hf_to_mcore_config
    from mcore_bridge.model.register import get_mcore_model

    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    hf_config.num_hidden_layers = n_layers
    hf_config.hybrid_override_pattern = hf_config.hybrid_override_pattern[:n_layers]
    hf_config.num_nextn_predict_layers = 0
    pattern = hf_config.hybrid_override_pattern
    print(f'RES pattern={pattern}')

    torch.manual_seed(0)
    input_ids = torch.randint(0, hf_config.vocab_size, (1, seq_len), device='cuda')

    # ---- HF: capture every backbone layer output ----
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, config=hf_config, torch_dtype=torch.bfloat16,
        trust_remote_code=True).cuda().eval()
    _restore_clobbered_weights(hf_model)   # HF _init_weights clobbers dt_bias with random inv_dt
    hf_acts = {}

    hf_ins = {}

    def mk_hook(idx):
        def hook(_mod, inp, out):
            if inp:
                hf_ins[idx] = inp[0].detach().float()
            hf_acts[idx] = (out[0] if isinstance(out, tuple) else out).detach().float()
        return hook

    for i, layer in enumerate(hf_model.backbone.layers):
        layer.register_forward_hook(mk_hook(i))
    hf_emb = {}
    hf_model.backbone.embeddings.register_forward_hook(
        lambda m, i, o: hf_emb.__setitem__(0, o.detach().float()))
    with torch.no_grad():
        hf_model(input_ids)
    del hf_model
    torch.cuda.empty_cache()

    # ---- MCore: same capture ----
    overrides = hf_to_mcore_config(hf_config)
    overrides.update(params_dtype=torch.bfloat16, bf16=True, mtp_num_layers=None)
    from megatron.core.transformer.enums import AttnBackend
    overrides['attention_backend'] = AttnBackend.flash
    cfg = ModelConfig(**overrides)
    models = get_mcore_model(cfg)
    cfg.bridge.load_weights(models, MODEL_PATH)
    mg_model = models[0].cuda().eval()

    mg_acts = {}

    mg_ins = {}

    def mk_hook_mg(idx):
        def hook(_mod, inp, out):
            if inp:
                mg_ins[idx] = inp[0].detach().float()
            t = out[0] if isinstance(out, tuple) else out
            mg_acts[idx] = t.detach().float()
        return hook

    for i, layer in enumerate(mg_model.decoder.layers):
        layer.register_forward_hook(mk_hook_mg(i))
    mg_emb = {}
    mg_model.embedding.register_forward_hook(
        lambda m, i, o: mg_emb.__setitem__(0, o.detach().float()))

    position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0)
    attention_mask = torch.tril(
        torch.ones((1, 1, seq_len, seq_len), device='cuda', dtype=torch.bool)).logical_not()
    with torch.no_grad():
        mg_model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)

    def norm(t):
        """HF is [b, s, h]; MCore is [s, b, h]. Normalize to [s, h]."""
        if t.dim() == 3:
            if t.shape[0] == 1 and t.shape[1] == seq_len:
                return t[0]
            if t.shape[1] == 1 and t.shape[0] == seq_len:
                return t[:, 0]
        return t.reshape(seq_len, -1)

    if 0 in hf_emb and 0 in mg_emb:
        a, b = norm(hf_emb[0]), norm(mg_emb[0])
        print(f'RES embedding max_abs_diff={(a - b).abs().max():.6f}')
    for i in range(n_layers):
        if i not in hf_acts or i not in mg_acts:
            print(f'RES layer{i} MISSING hf={i in hf_acts} mg={i in mg_acts}')
            continue
        a, b = norm(hf_acts[i]), norm(mg_acts[i])
        d = (a - b).abs()
        scale = a.abs().max().clamp(min=1e-6)
        if i in hf_ins and i in mg_ins:
            ia, ib = norm(hf_ins[i]), norm(mg_ins[i])
            di = (ia - ib).abs()
            print(f'RES layer{i} IN  max_abs_diff={di.max():.6f} '
                  f'hf_std={ia.std():.5f} mg_std={ib.std():.5f}')
        print(f'RES layer{i} type={pattern[i]!r} hf_shape={tuple(hf_acts[i].shape)} '
              f'mg_shape={tuple(mg_acts[i].shape)} '
              f'hf_std={a.std():.5f} mg_std={b.std():.5f} '
              f'max_abs_diff={d.max():.6f} mean={d.mean():.6f} rel={d.max() / scale:.6f}')


if __name__ == '__main__':
    main()
