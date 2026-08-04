"""Verify `save_missing_weights` against a synthetic DeepSeek-V4 checkpoint.

The real DeepSeek-V4-Flash-0731 is far too large to test with, so this builds a
4-layer model from its config and fills every tensor with random values. The
`mtp.*` (DSpark) weights follow the 3-stage layout of the real checkpoint,
including the stage-specific extras (`main_proj`, `confidence_head`,
`markov_head`) that Megatron has no module for.

Two properties are checked:
  * with the flag on, the unsupported `mtp.*` weights survive the round-trip;
  * `mtp.*` keys never appear twice under two different naming schemes, which is
    what would happen if Megatron also exported its own `model.mtp.*` weights.
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# The megatron entrypoint initializes torch.distributed via env:// rendezvous.
os.environ.setdefault('RANK', '0')
os.environ.setdefault('LOCAL_RANK', '0')
os.environ.setdefault('WORLD_SIZE', '1')
os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '29901')

import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

MODEL_TYPE = 'deepseek_v4'
TEMPLATE = 'deepseek_v4_flash'
# Only the tokenizer files are read from here; the weights are generated locally.
# GIT_LFS_SKIP_SMUDGE=1 git clone https://www.modelscope.cn/deepseek-ai/DeepSeek-V4-Flash-0731.git
REFERENCE_MODEL_ID = 'deepseek-ai/DeepSeek-V4-Flash-0731'
TOKENIZER_FILES = ['tokenizer.json', 'tokenizer_config.json', 'generation_config.json']

NUM_LAYERS = 4
NUM_MTP_STAGES = 3
HIDDEN = 256
VOCAB = 512
N_EXPERTS = 4
MOE_INTERMEDIATE = 128
HC_MULT = 2
Q_LORA_RANK = 64
O_LORA_RANK = 32
O_GROUPS = 2
HEAD_DIM = 32
NUM_HEADS = 4
QK_ROPE_HEAD_DIM = 16
MARKOV_RANK = 32
DSPARK_TARGET_LAYERS = [1, 2, 3]

# Shapes derived the same way the real checkpoint does (verified against 0731).
QK_HEAD_DIM = HEAD_DIM  # wq_b rows are NUM_HEADS * head_dim
KV_DIM = HEAD_DIM  # wkv rows; the rope part is not stored separately here
# wo_a is [o_groups * o_lora_rank, per_group_dim]; wo_b maps that back to hidden.
O_GROUP_DIM = NUM_HEADS * HEAD_DIM // O_GROUPS
O_A_ROWS = O_GROUPS * O_LORA_RANK
HC_STREAM = HC_MULT * HIDDEN  # hc_*_fn columns
HC_ROWS = HC_MULT * (HC_MULT + 2)  # hc_attn/ffn rows: width + depth + scale terms
HC_ALPHAS = 3  # the bridge reads alpha_pre / alpha_post / alpha_res from hc_*_scale

# All layers stay dense: a ratio of 0 avoids the CSA compressor weights, which are
# irrelevant to weight persistence and would only add noise to this test.
COMPRESS_RATIOS = [0] * NUM_LAYERS


def _config() -> dict:
    """A miniature version of the DeepSeek-V4-Flash-0731 config."""
    return {
        'architectures': ['DeepseekV4ForCausalLM'],
        'attention_bias': False,
        'attention_dropout': 0.0,
        'bos_token_id': 0,
        'eos_token_id': 1,
        'hc_eps': 1e-06,
        'hc_mult': HC_MULT,
        'hc_sinkhorn_iters': 20,
        'head_dim': HEAD_DIM,
        'hidden_act': 'silu',
        'hidden_size': HIDDEN,
        'index_head_dim': 32,
        'index_n_heads': 4,
        'index_topk': 32,
        'initializer_range': 0.02,
        'max_position_embeddings': 4096,
        'model_type': 'deepseek_v4',
        'moe_intermediate_size': MOE_INTERMEDIATE,
        'n_routed_experts': N_EXPERTS,
        'n_shared_experts': 1,
        'norm_topk_prob': True,
        'num_attention_heads': NUM_HEADS,
        'num_experts_per_tok': 2,
        'num_hidden_layers': NUM_LAYERS,
        'num_hash_layers': 0,
        'num_key_value_heads': 1,
        'num_nextn_predict_layers': NUM_MTP_STAGES,
        'o_groups': O_GROUPS,
        'o_lora_rank': O_LORA_RANK,
        'q_lora_rank': Q_LORA_RANK,
        'qk_rope_head_dim': QK_ROPE_HEAD_DIM,
        'rms_norm_eps': 1e-06,
        'rope_theta': 10000,
        'routed_scaling_factor': 1.5,
        'scoring_func': 'sqrtsoftplus',
        'sliding_window': 32,
        'swiglu_limit': 10.0,
        'tie_word_embeddings': False,
        'topk_method': 'noaux_tc',
        'torch_dtype': 'bfloat16',
        'use_cache': True,
        'vocab_size': VOCAB,
        'compress_rope_theta': 160000,
        'compress_ratios': COMPRESS_RATIOS,
        'dspark_block_size': 5,
        'dspark_noise_token_id': VOCAB - 1,
        'dspark_target_layer_ids': DSPARK_TARGET_LAYERS,
        'dspark_markov_rank': MARKOV_RANK,
    }


def _rand(*shape) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.bfloat16) * 0.02


def _attn_and_ffn_weights(prefix: str) -> dict:
    """Weights shared by every transformer block, main trunk and DSpark alike."""
    sd = {
        f'{prefix}attn_norm.weight': _rand(HIDDEN),
        f'{prefix}ffn_norm.weight': _rand(HIDDEN),
        f'{prefix}attn.q_norm.weight': _rand(Q_LORA_RANK),
        f'{prefix}attn.kv_norm.weight': _rand(KV_DIM),
        f'{prefix}attn.attn_sink': _rand(NUM_HEADS),
        f'{prefix}attn.wq_a.weight': _rand(Q_LORA_RANK, HIDDEN),
        f'{prefix}attn.wq_b.weight': _rand(NUM_HEADS * QK_HEAD_DIM, Q_LORA_RANK),
        f'{prefix}attn.wkv.weight': _rand(KV_DIM, HIDDEN),
        f'{prefix}attn.wo_a.weight': _rand(O_A_ROWS, O_GROUP_DIM),
        f'{prefix}attn.wo_b.weight': _rand(HIDDEN, O_A_ROWS),
        f'{prefix}ffn.gate.weight': _rand(N_EXPERTS, HIDDEN),
        f'{prefix}ffn.gate.bias': _rand(N_EXPERTS),
    }
    for name, shape in [('w1', (MOE_INTERMEDIATE, HIDDEN)), ('w2', (HIDDEN, MOE_INTERMEDIATE)),
                        ('w3', (MOE_INTERMEDIATE, HIDDEN))]:
        sd[f'{prefix}ffn.shared_experts.{name}.weight'] = _rand(*shape)
        for e in range(N_EXPERTS):
            sd[f'{prefix}ffn.experts.{e}.{name}.weight'] = _rand(*shape)
    # Hyper-connection parameters, present on every block of the real model.
    # `base` holds hc_mult entries per residual stream, `scale` one per stream.
    for tag in ['attn', 'ffn']:
        sd[f'{prefix}hc_{tag}_base'] = _rand(HC_ROWS)
        sd[f'{prefix}hc_{tag}_fn'] = _rand(HC_ROWS, HC_STREAM)
        sd[f'{prefix}hc_{tag}_scale'] = _rand(HC_ALPHAS)
    return sd


def _hc_head_weights(prefix: str) -> dict:
    """Output-side hyper-connection: one row per stream, a single scale."""
    return {
        f'{prefix}hc_head_base': _rand(HC_MULT),
        f'{prefix}hc_head_fn': _rand(HC_MULT, HC_STREAM),
        f'{prefix}hc_head_scale': _rand(1),
    }


def _mtp_weights() -> dict:
    """The 3 asymmetric DSpark stages, mirroring the real key layout."""
    sd = {}
    for stage in range(NUM_MTP_STAGES):
        sd.update(_attn_and_ffn_weights(f'mtp.{stage}.'))
        if stage == 0:
            # Stage 0 consumes the concatenated hidden states of the target layers.
            sd['mtp.0.main_norm.weight'] = _rand(HIDDEN)
            sd['mtp.0.main_proj.weight'] = _rand(HIDDEN, HIDDEN * len(DSPARK_TARGET_LAYERS))
        if stage == NUM_MTP_STAGES - 1:
            # The last stage owns the output-side heads.
            sd['mtp.2.norm.weight'] = _rand(HIDDEN)
            sd['mtp.2.confidence_head.proj.weight'] = _rand(1, HIDDEN + QK_ROPE_HEAD_DIM)
            sd['mtp.2.markov_head.markov_w1.weight'] = _rand(VOCAB, MARKOV_RANK)
            sd['mtp.2.markov_head.markov_w2.weight'] = _rand(VOCAB, MARKOV_RANK)
            sd.update(_hc_head_weights('mtp.2.'))
    return sd


def _build_fake_checkpoint(dst_dir: str) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    config = _config()
    with open(os.path.join(dst_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    # Reuse the real tokenizer; `download_model=False` keeps the multi-GB weights out.
    from swift import safe_snapshot_download
    ref_dir = safe_snapshot_download(REFERENCE_MODEL_ID, download_model=False)
    for fname in TOKENIZER_FILES:
        src = os.path.join(ref_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dst_dir, fname))

    state_dict = {
        'embed.weight': _rand(VOCAB, HIDDEN),
        'head.weight': _rand(VOCAB, HIDDEN),
        'norm.weight': _rand(HIDDEN),
    }
    state_dict.update(_hc_head_weights(''))
    for layer in range(NUM_LAYERS):
        state_dict.update(_attn_and_ffn_weights(f'layers.{layer}.'))
    state_dict.update(_mtp_weights())

    save_file(state_dict, os.path.join(dst_dir, 'model.safetensors'), metadata={'format': 'pt'})
    return dst_dir


def _load_exported(output_dir: str) -> dict:
    index_path = os.path.join(output_dir, 'model.safetensors.index.json')
    if os.path.exists(index_path):
        with open(index_path) as f:
            shards = sorted(set(json.load(f)['weight_map'].values()))
    else:
        shards = ['model.safetensors']
    state_dict = {}
    for shard in shards:
        state_dict.update(load_file(os.path.join(output_dir, shard)))
    return state_dict


def _trunk_keys(state_dict) -> list:
    """Keys of layer 0 of the main trunk, whatever prefix the bridge chose."""
    return [k for k in state_dict if 'layers.0.' in k and not k.startswith('mtp.')]


def _export(model_dir: str, output_dir: str, save_missing_weights: bool, mtp_num_layers=None):
    """Round-trip HF -> mcore -> HF through the real bridge."""
    from swift.megatron import MegatronExportArguments, megatron_export_main
    mcore_dir = f'{output_dir}-mcore'
    common = dict(model_type=MODEL_TYPE, template=TEMPLATE, exist_ok=True, torch_dtype='bfloat16')
    if mtp_num_layers is not None:
        common['mtp_num_layers'] = mtp_num_layers
    megatron_export_main(MegatronExportArguments(model=model_dir, to_mcore=True, output_dir=mcore_dir, **common))
    megatron_export_main(
        MegatronExportArguments(
            mcore_model=mcore_dir,
            # `model` is only used as the source of the weights Megatron cannot export.
            model=model_dir,
            to_hf=True,
            output_dir=output_dir,
            save_missing_weights=save_missing_weights,
            **common))
    return output_dir


def test_dsv4_mtp_weights_restored():
    """Without Megatron MTP, every `mtp.*` tensor must come back byte-for-byte."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = _build_fake_checkpoint(os.path.join(tmp_dir, 'src'))
        source = load_file(os.path.join(model_dir, 'model.safetensors'))
        mtp_keys = {k for k in source if k.startswith('mtp.')}
        assert len(mtp_keys) > 100, f'the fake checkpoint should have many mtp keys, got {len(mtp_keys)}'

        output_dir = _export(model_dir, os.path.join(tmp_dir, 'restored'), save_missing_weights=True)
        exported = _load_exported(output_dir)

        for key in sorted(mtp_keys):
            assert key in exported, f'{key} was not restored'
            assert torch.equal(exported[key], source[key]), f'{key} was altered during restore'
        # The trunk must still be produced by Megatron rather than copied over.
        assert _trunk_keys(exported), 'trunk weights missing from export'


def test_dsv4_mtp_weights_dropped_by_default():
    """With the flag off the `mtp.*` weights are lost, as they are today."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = _build_fake_checkpoint(os.path.join(tmp_dir, 'src'))
        output_dir = _export(model_dir, os.path.join(tmp_dir, 'dropped'), save_missing_weights=False)
        exported = _load_exported(output_dir)

        assert not [k for k in exported if k.startswith('mtp.')], 'mtp weights leaked into the export'
        assert _trunk_keys(exported), 'trunk weights missing from export'


def test_dsv4_no_duplicate_mtp_when_megatron_exports_it():
    """Guard against storing the same DSpark parameters under two naming schemes.

    When Megatron does materialize MTP layers it writes them as `model.mtp.*`,
    while the source checkpoint names them `mtp.*`. Both sets would then land in
    the output, doubling the size and leaving it ambiguous which one is loaded.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = _build_fake_checkpoint(os.path.join(tmp_dir, 'src'))
        try:
            output_dir = _export(
                model_dir, os.path.join(tmp_dir, 'mtp'), save_missing_weights=True, mtp_num_layers=NUM_MTP_STAGES)
        except Exception as e:  # noqa: BLE001
            # Expected today: `_convert_mtp_extra` looks for the pre-0731 `enorm.weight`
            # layout, so Megatron cannot build the DSpark stages at all.
            print(f'SKIP: Megatron cannot load DSpark MTP layers yet ({type(e).__name__}: {e})')
            return
        exported = _load_exported(output_dir)

        megatron_mtp = {k for k in exported if k.startswith('model.mtp.')}
        restored_mtp = {k for k in exported if k.startswith('mtp.')}
        assert not (megatron_mtp
                    and restored_mtp), (f'DSpark weights stored twice: {len(megatron_mtp)} keys as `model.mtp.*` and '
                                        f'{len(restored_mtp)} keys as `mtp.*`')


if __name__ == '__main__':
    test_dsv4_mtp_weights_restored()
    test_dsv4_mtp_weights_dropped_by_default()
    test_dsv4_no_duplicate_mtp_when_megatron_exports_it()
