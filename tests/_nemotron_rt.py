"""Nemotron-H round-trip harness: HF -> MCore -> HF must be bit-exact.

Parallel layout is taken from env vars so the same script covers TP/PP/EP:
    TP, PP, EP, ETP  (default 1), PATTERN (default 'ME*')

Run e.g.:
    EP=2 PATTERN='ME*' torchrun --nproc_per_node=2 tests/_nemotron_rt.py
"""
import os

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

MODEL_PATH = ('/root/.cache/modelscope/hub/models/nv-community/'
              'EA-NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16-07202026')
NUM_EXPERTS = 4


class Lazy:
    """Mimics SafetensorLazyLoader's handle: Bridge calls `.load()`."""

    def __init__(self, tensor):
        self.tensor = tensor

    def load(self):
        return self.tensor


def build_hf_state_dict(cfg, mixer, pattern, rand):
    hidden = cfg.hidden_size
    sd = {
        'backbone.embeddings.weight': Lazy(rand(256, hidden)),
        'lm_head.weight': Lazy(rand(256, hidden)),
        'backbone.norm_f.weight': Lazy(rand(hidden)),
    }
    for i, ch in enumerate(pattern):
        p = f'backbone.layers.{i}.'
        sd[f'{p}norm.weight'] = Lazy(rand(hidden))
        if ch == 'M':
            conv_dim = mixer.d_inner + 2 * mixer.ngroups * mixer.d_state
            in_dim = mixer.d_inner * 2 + 2 * mixer.ngroups * mixer.d_state + mixer.nheads
            sd[f'{p}mixer.in_proj.weight'] = Lazy(rand(in_dim, hidden))
            sd[f'{p}mixer.conv1d.weight'] = Lazy(rand(conv_dim, 1, 4))
            sd[f'{p}mixer.conv1d.bias'] = Lazy(rand(conv_dim))
            # A_log/D are fp32 in mcore; dt_bias is bf16.
            sd[f'{p}mixer.A_log'] = Lazy(rand(mixer.nheads, dtype=torch.float32))
            sd[f'{p}mixer.D'] = Lazy(rand(mixer.nheads, dtype=torch.float32))
            sd[f'{p}mixer.dt_bias'] = Lazy(rand(mixer.nheads))
            sd[f'{p}mixer.norm.weight'] = Lazy(rand(mixer.d_inner))
            sd[f'{p}mixer.out_proj.weight'] = Lazy(rand(hidden, mixer.d_inner))
        elif ch == 'E':
            sd[f'{p}mixer.gate.weight'] = Lazy(rand(NUM_EXPERTS, hidden))
            sd[f'{p}mixer.gate.e_score_correction_bias'] = Lazy(
                rand(NUM_EXPERTS, dtype=torch.float32))
            moe_ffn = cfg.moe_ffn_hidden_size
            for e in range(NUM_EXPERTS):
                sd[f'{p}mixer.experts.{e}.up_proj.weight'] = Lazy(rand(moe_ffn, hidden))
                sd[f'{p}mixer.experts.{e}.down_proj.weight'] = Lazy(rand(hidden, moe_ffn))
            shared = cfg.moe_shared_expert_intermediate_size
            sd[f'{p}mixer.shared_experts.up_proj.weight'] = Lazy(rand(shared, hidden))
            sd[f'{p}mixer.shared_experts.down_proj.weight'] = Lazy(rand(hidden, shared))
        elif ch == '*':
            head_dim = cfg.kv_channels
            q = cfg.num_attention_heads * head_dim
            kv = cfg.num_query_groups * head_dim
            sd[f'{p}mixer.q_proj.weight'] = Lazy(rand(q, hidden))
            sd[f'{p}mixer.k_proj.weight'] = Lazy(rand(kv, hidden))
            sd[f'{p}mixer.v_proj.weight'] = Lazy(rand(kv, hidden))
            sd[f'{p}mixer.o_proj.weight'] = Lazy(rand(hidden, q))
        else:
            raise ValueError(f'unsupported pattern char {ch!r}')
    return sd


def main():
    tp = int(os.environ.get('TP', 1))
    pp = int(os.environ.get('PP', 1))
    ep = int(os.environ.get('EP', 1))
    etp = int(os.environ.get('ETP', tp))
    pattern = os.environ.get('PATTERN', 'ME*')

    dist.init_process_group('nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
    )
    model_parallel_cuda_manual_seed(1234)  # MambaMixer uses get_cuda_rng_tracker().fork()

    from transformers import AutoConfig

    import mcore_bridge.model.gpts  # noqa: F401  (triggers registration)
    from mcore_bridge.config.model_config import ModelConfig
    from mcore_bridge.config.parser import hf_to_mcore_config
    from mcore_bridge.model.register import get_mcore_model

    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    overrides = hf_to_mcore_config(hf_config)
    overrides.update(
        num_layers=len(pattern),
        hybrid_layer_pattern=pattern,
        moe_layer_freq='[' + ','.join('1' if c == 'E' else '0' for c in pattern) + ']',
        num_moe_experts=NUM_EXPERTS,
        padded_vocab_size=256,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
        params_dtype=torch.bfloat16,
        bf16=True,
    )
    cfg = ModelConfig(**overrides)

    models = get_mcore_model(cfg)
    bridge = cfg.bridge

    # Any rank holding a Mamba layer can report the mixer dims; they are TP-invariant
    # on the HF side, so derive them from config instead of the (possibly absent) module.
    class _Dims:
        nheads = cfg.mamba_num_heads
        d_inner = cfg.mamba_num_heads * cfg.mamba_head_dim
        ngroups = cfg.mamba_num_groups
        d_state = cfg.mamba_state_dim

    gen = torch.Generator(device='cuda').manual_seed(4321)  # identical data on every rank

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen, dtype=dtype, device='cuda')

    sd = build_hf_state_dict(cfg, _Dims, pattern, rand)
    original = {k: v.load().clone() for k, v in sd.items()}

    for model in models:
        list(bridge._convert([model], sd, '', True, 'Loading: '))

    exported = dict(bridge.export_weights(models, target_device='cuda'))

    if local_rank != 0:
        dist.barrier()
        return

    tag = f'TP{tp}/PP{pp}/EP{ep} pattern={pattern}'
    missing = sorted(set(original) - set(exported))
    extra = sorted(set(exported) - set(original))
    bad = []
    for key, want in original.items():
        got = exported.get(key)
        if got is None:
            continue
        if tuple(got.shape) != tuple(want.shape):
            bad.append((key, 'shape', tuple(want.shape), tuple(got.shape)))
        elif not torch.equal(got.to(want.dtype).cpu(), want.cpu()):
            delta = (got.to(want.dtype).cpu().float() - want.cpu().float()).abs().max()
            bad.append((key, 'value', float(delta)))

    print(f'RES [{tag}] keys={len(exported)} missing={missing} extra={extra} '
          f'mismatches={len(bad)}')
    for item in bad[:10]:
        print('   ', item)
    if not bad and not missing and not extra:
        print(f'RES [{tag}] ROUNDTRIP EXACT PASS')
    else:
        print(f'RES [{tag}] ROUNDTRIP FAILED')
    dist.barrier()


if __name__ == '__main__':
    main()
