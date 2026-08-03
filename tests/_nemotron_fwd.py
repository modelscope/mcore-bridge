"""Forward-consistency check: HF NemotronHForCausalLM vs converted MCore GPTModel.

The round-trip tests only prove the weight transport is reversible; they say nothing
about whether the MCore model *computes* the same thing. This script loads the real
checkpoint into both stacks and compares logits.

Run (single H20, ~62GB model so keep TP=1 and expect high memory):
    torchrun --nproc_per_node=1 tests/_nemotron_fwd.py
Optionally limit layers for a cheap smoke run:
    NUM_LAYERS=8 torchrun --nproc_per_node=1 tests/_nemotron_fwd.py
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
    seq_len = int(os.environ.get('SEQ_LEN', 16))
    num_layers_override = os.environ.get('NUM_LAYERS')

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
    if num_layers_override:
        n = int(num_layers_override)
        hf_config.num_hidden_layers = n
        hf_config.hybrid_override_pattern = hf_config.hybrid_override_pattern[:n]
    # MTP is not supported by the bridge; disable so both sides match.
    hf_config.num_nextn_predict_layers = 0

    torch.manual_seed(0)
    input_ids = torch.randint(0, hf_config.vocab_size, (1, seq_len), device='cuda')

    # ---- HF reference ----
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, config=hf_config, torch_dtype=torch.bfloat16,
        trust_remote_code=True).cuda().eval()
    # HF BUG WORKAROUND: this checkpoint's `_init_weights` unconditionally does
    # `module.dt_bias.copy_(inv_dt)` with a *random* inv_dt, and only sets
    # `_no_reinit = True` afterwards without ever checking it. So the trained
    # dt_bias from safetensors is discarded and HF runs with random values.
    # Restore it from the checkpoint so the reference is actually the trained model.
    _restore_clobbered_weights(hf_model)
    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits.float()
    del hf_model
    torch.cuda.empty_cache()
    print(f'RES hf_logits {tuple(hf_logits.shape)} '
          f'mean={hf_logits.mean():.5f} std={hf_logits.std():.5f}')

    # ---- MCore under test ----
    overrides = hf_to_mcore_config(hf_config)
    overrides.update(params_dtype=torch.bfloat16, bf16=True, mtp_num_layers=None)
    # The cuDNN fused-attention backend fails to load its sublibrary in this container;
    # flash is equivalent for correctness purposes here.
    backend = os.environ.get('ATTN_BACKEND', 'flash')
    if backend:
        from megatron.core.transformer.enums import AttnBackend
        overrides['attention_backend'] = getattr(AttnBackend, backend)
    cfg = ModelConfig(**overrides)
    models = get_mcore_model(cfg)
    cfg.bridge.load_weights(models, MODEL_PATH)
    mg_model = models[0].cuda().eval()

    position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0)
    attention_mask = torch.tril(
        torch.ones((1, 1, seq_len, seq_len), device='cuda', dtype=torch.bool)).logical_not()
    with torch.no_grad():
        mg_logits = mg_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        ).float()
    if mg_logits.shape[0] == seq_len:  # [s, b, h] -> [b, s, h]
        mg_logits = mg_logits.transpose(0, 1)
    mg_logits = mg_logits[..., :hf_logits.shape[-1]]
    print(f'RES mg_logits {tuple(mg_logits.shape)} '
          f'mean={mg_logits.mean():.5f} std={mg_logits.std():.5f}')

    diff = (hf_logits - mg_logits).abs()
    rel = diff.max() / hf_logits.abs().max()
    hf_top = hf_logits.argmax(-1)
    mg_top = mg_logits.argmax(-1)
    agree = (hf_top == mg_top).float().mean()
    print(f'RES max_abs_diff={diff.max():.6f} mean_abs_diff={diff.mean():.6f} '
          f'rel={rel:.6f} argmax_agree={agree:.4f}')
    print(f'RES hf_top={hf_top.flatten()[:8].tolist()}')
    print(f'RES mg_top={mg_top.flatten()[:8].tolist()}')
    # Where argmax disagrees, check whether it's a near-tie (bf16 noise flipping the
    # order of two nearly-equal logits) rather than a real behavioural difference.
    mism = (hf_top != mg_top).nonzero()
    for pos in mism[:5]:
        b, t = pos.tolist()
        hv, mv = hf_logits[b, t], mg_logits[b, t]
        top2 = hv.topk(2).values
        print(f'RES tie@t={t} hf_top1-top2_gap={float(top2[0] - top2[1]):.5f} '
              f'hf@hf_top={float(hv[hf_top[b, t]]):.5f} hf@mg_top={float(hv[mg_top[b, t]]):.5f} '
              f'delta={float(hv[hf_top[b, t]] - hv[mg_top[b, t]]):.5f}')
    # Rank correlation is the robust check: argmax can flip on ties.
    k = 20
    hf_set = hf_logits.topk(k, -1).indices
    mg_set = mg_logits.topk(k, -1).indices
    overlap = sum(len(set(a.tolist()) & set(b.tolist())) / k
                  for a, b in zip(hf_set.reshape(-1, k), mg_set.reshape(-1, k)))
    overlap /= hf_set.reshape(-1, k).shape[0]
    print(f'RES top{k}_overlap={overlap:.4f}')
    if agree >= 0.9 and rel < 0.05 and overlap > 0.95:
        print('RES FORWARD CONSISTENCY PASS (bf16-level)')
    else:
        print('RES FORWARD CONSISTENCY FAIL')


if __name__ == '__main__':
    main()
