# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import os
import pytest
import subprocess
import sys
import torch
from transformers import AutoConfig, PretrainedConfig
from types import SimpleNamespace as NS

from mcore_bridge.config.parser import hf_to_mcore_config
from mcore_bridge.model.mm_gpts import nemo_speechlm as speechlm
from mcore_bridge.model.register import MODEL_MAPPING
from mcore_bridge.utils.safetensors import LazyTensor

try:
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    _HAS_NEMO_CONVERSION = bool(get_checkpoint_conversion_mapping('nemo_speechlm'))
except (ImportError, KeyError):
    _HAS_NEMO_CONVERSION = False


def _config():
    llm = AutoConfig.for_model(
        'nemotron_h',
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        layers_block_type=['mamba', 'moe', 'attention'],
        mamba_num_heads=2,
        mamba_head_dim=8,
        n_groups=2,
        ssm_state_size=4,
        n_routed_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        moe_shared_expert_intermediate_size=8,
        rescale_prenorm_residual=True,
        moe_shared_expert_overlap=True,
        tie_word_embeddings=False,
    )
    config = PretrainedConfig()
    config.model_type = 'nemo_speechlm'
    config.text_config = llm
    config.audio_token_id = llm.vocab_size
    config.aux_loss_coeff = 1e-4
    return config


def _enable_mtp(config):
    config.mtp = NS(enabled=True, num_nextn_predict_layers=2, use_repeated_layer=True, hybrid_override_pattern='*E')
    return config


def test_parser_and_registration():
    result = hf_to_mcore_config(_config())
    assert result['llm_model_type'] == 'nemotron_h'
    assert result['hybrid_layer_pattern'] == 'ME*'
    assert result['num_layers'] == 3
    assert result['moe_aux_loss_coeff'] == 1e-4
    assert result['moe_shared_expert_overlap']
    meta = MODEL_MAPPING['nemo_speechlm']
    assert meta.mtp_prefix == 'llm.mtp.layers'
    assert meta.mtp_depth_field == 'mtp.num_nextn_predict_layers'
    assert meta.mtp_repeated_field == 'mtp.use_repeated_layer'
    assert meta.mtp_layers_per_depth == 2


def test_parser_nested_repeated_mtp():
    result = hf_to_mcore_config(_enable_mtp(_config()))
    assert result['hybrid_layer_pattern'] == 'ME*/*E/*E'
    assert result['mtp_hybrid_override_pattern'] == '*E'
    assert result['mtp_use_repeated_layer']


def test_mtp_auto_detection_for_repeated_hybrid(monkeypatch):
    from mcore_bridge.model import mtp_detect

    keys = [
        'llm.model.layers.0.norm.weight',
        'llm.mtp.layers.0.eh_proj.weight',
        'llm.mtp.layers.1.final_layernorm.weight',
    ]
    monkeypatch.setattr(mtp_detect, '_read_weight_keys', lambda _: keys)
    result = mtp_detect.resolve_mtp_config(MODEL_MAPPING['nemo_speechlm'], _enable_mtp(_config()), 'unused', 'auto',
                                           None)
    assert result == (2, True)


def test_runtime_vocab_differs_from_checkpoint_vocab():
    from test_glm5_next import _parallel_context

    hf_config = _config()
    hf_config.text_config = copy.deepcopy(hf_config.text_config)
    hf_config.text_config.vocab_size = 32
    hf_config.audio_token_id = 32
    values = hf_to_mcore_config(hf_config)
    values.update(params_dtype=torch.float32, pipeline_dtype=torch.float32)
    with _parallel_context():
        config = MODEL_MAPPING['nemo_speechlm'].config_cls(**values)
    assert config.padded_vocab_size == 128
    assert config.actual_vocab_size == 32


def test_nemo_expert_lazy_slice_and_pack():
    packed = torch.arange(2 * 3 * 4.).view(2, 3, 4)
    lazy = LazyTensor(tensor=packed)
    torch.testing.assert_close(speechlm._load_nemo_expert(lazy, 1), packed[1].T)

    bridge = speechlm.NemoSpeechLMBridge.__new__(speechlm.NemoSpeechLMBridge)
    bridge.config = NS(num_moe_experts=2)
    state = {}
    for expert in range(2):
        state[f'mixer.experts.{expert}.up_proj.weight'] = packed[expert].T
        state[f'mixer.experts.{expert}.down_proj.weight'] = packed[expert].T + 100
    result = bridge._pack_nemo_experts(state)
    assert set(result) == {'mixer.experts.gate_and_up_projs', 'mixer.experts.down_projs'}
    torch.testing.assert_close(result['mixer.experts.gate_and_up_projs'], packed)
    torch.testing.assert_close(result['mixer.experts.down_projs'], packed + 100)


class _Perception(torch.nn.Module):

    def __init__(self, hidden_size=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.scale = torch.nn.Parameter(torch.tensor(2.))

    def get_audio_features(self, input_features, input_features_mask):
        subsampling_factor = 8
        frames = input_features_mask.sum(-1).to(torch.long)
        lengths = ((frames + subsampling_factor - 1) // subsampling_factor).tolist()
        return torch.cat([
            self.scale * input_features.new_ones((length, self.hidden_size), dtype=self.scale.dtype)
            for length in lengths
        ])


def _visual(hidden_size=3, audio_token_id=9):
    visual = speechlm.NemoSpeechLMVit.__new__(speechlm.NemoSpeechLMVit)
    torch.nn.Module.__init__(visual)
    visual.perception = _Perception(hidden_size)
    visual.audio_token_id = audio_token_id
    visual.config = NS(hf_config=NS(audio_config=NS(asr_config=NS(num_mel_bins=128))))
    return visual


def test_audio_scatter_and_gradient():
    visual = _visual()
    inputs_embeds = torch.zeros(1, 15, 3, requires_grad=True)
    input_ids = torch.tensor([[1] + [9] * 13 + [2]])
    result = visual.get_inputs_embeds(
        inputs_embeds,
        input_ids=input_ids,
        input_features=torch.zeros(1, 104, 128, dtype=torch.float32),
        input_features_mask=torch.ones(1, 104, dtype=torch.bool))
    assert torch.equal(result[0, 0], torch.zeros(3))
    assert torch.equal(result[0, 1:14], torch.full((13, 3), 2.))
    assert torch.equal(result[0, 14], torch.zeros(3))
    result.sum().backward()
    assert visual.perception.scale.grad is not None
    assert visual.perception.scale.grad != 0


def test_text_only_dummy_preserves_values_and_graph():
    visual = _visual()
    inputs_embeds = torch.randn(1, 2, 3, requires_grad=True)
    result = visual.get_inputs_embeds(inputs_embeds, input_ids=torch.tensor([[1, 2]]))
    torch.testing.assert_close(result, inputs_embeds)
    result.sum().backward()
    assert visual.perception.scale.grad is not None


def test_audio_shape_mismatch_fails():
    visual = _visual()
    with pytest.raises(ValueError, match='do not match placeholders'):
        visual.get_inputs_embeds(
            torch.zeros(1, 2, 3),
            input_ids=torch.tensor([[9, 9]]),
            input_features=torch.zeros(1, 3, 128),
            input_features_mask=torch.ones(1, 3, dtype=torch.bool))


@pytest.mark.parametrize('changes,match', [
    ({
        'context_parallel_size': 2
    }, 'context_parallel_size 1'),
    ({
        'mtp_num_layers': 1
    }, 'repeated physical pattern'),
    ({
        'fp8': 'hybrid'
    }, 'FP8/FP4'),
    ({
        'padding_free': True
    }, 'padding_free false'),
])
def test_loader_guards(changes, match):
    config = NS(context_parallel_size=1, mtp_num_layers=0, fp8=None, fp4=None, padding_free=False, packing=False)
    for key, value in changes.items():
        setattr(config, key, value)
    with pytest.raises(NotImplementedError, match=match):
        speechlm.NemoSpeechLMLoader(config)


def test_prepare_model_uses_transformers_perception(monkeypatch):
    visual = speechlm.NemoSpeechLMVit.__new__(speechlm.NemoSpeechLMVit)
    torch.nn.Module.__init__(visual)
    perception = object()
    monkeypatch.setattr(speechlm, '_TransformersPerception', lambda _: perception)
    visual.prepare_model(NS(audio_token_id=7))
    assert visual.perception is perception
    assert visual.audio_token_id == 7


def test_get_hf_meta_model_rejects_legacy_metadata():
    bridge = speechlm.NemoSpeechLMBridge.__new__(speechlm.NemoSpeechLMBridge)
    bridge.config = NS(hf_config=PretrainedConfig())
    with pytest.raises(ValueError, match='legacy NeMo metadata'):
        bridge.get_hf_meta_model()


def test_full_export_requires_source_checkpoint(tmp_path):
    bridge = speechlm.NemoSpeechLMBridge.__new__(speechlm.NemoSpeechLMBridge)
    bridge._source_model_dir = None
    with pytest.raises(ValueError, match='preserve MTP, CTC'):
        bridge.save_weights([], str(tmp_path))
    bridge._source_model_dir = str(tmp_path / 'missing')
    with pytest.raises(ValueError, match='preserve MTP, CTC'):
        bridge.save_weights([], str(tmp_path))


def _build_tiny_model(tp=1, pp=1, ep=1, sequence_parallel=False, mtp=False):
    from mcore_bridge.model.register import get_mcore_model

    hf_config = _config()
    if mtp:
        _enable_mtp(hf_config)
    hf_config.audio_token_id = 31
    if pp > 1:
        hf_config.text_config.num_hidden_layers = 4
        hf_config.text_config.layers_block_type.append('full_attention')
    values = hf_to_mcore_config(hf_config)
    values.update(
        language_model_only=True,
        params_dtype=torch.float32,
        pipeline_dtype=torch.float32,
        perform_initialization=True,
        use_cpu_initialization=False,
        moe_grouped_gemm=True,
        overlap_p2p_comm=False,
        mtp_num_layers=2 if mtp else 0,
        mtp_shared_weights=True if mtp else None,
        moe_router_bias_update_rate=0.,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=1,
        sequence_parallel=sequence_parallel,
    )
    config = MODEL_MAPPING['nemo_speechlm'].config_cls(**values)
    assert config.actual_vocab_size == 32
    assert config.padded_vocab_size % (128 * tp) == 0
    return get_mcore_model(config)[0].cuda(), config


def _export(config, model):
    result = {}
    for key, value in config.bridge._convert([model], {}, '', False, 'Exporting test: '):
        result[key] = value.load() if hasattr(value, 'load') else value
    return result


def test_tiny_model_nemo_roundtrip_and_audio_backward():
    from test_glm5_next import _parallel_context

    from mcore_bridge.model.register import get_mcore_model

    with _parallel_context():
        model, config = _build_tiny_model()
        exported = _export(config, model)
        assert exported['llm.model.embed_tokens.weight'].shape == (32, 16)
        assert exported['llm.lm_head.weight'].shape == (32, 16)
        assert 'llm.model.layers.0.mixer._fp32_params.A_log' in exported
        assert exported['llm.model.layers.1.mixer.experts.gate_and_up_projs'].shape == (2, 16, 8)
        assert exported['llm.model.layers.1.mixer.experts.down_projs'].shape == (2, 8, 16)

        reloaded = get_mcore_model(config)[0].cuda()
        lazy = {key: LazyTensor(tensor=value) for key, value in exported.items()}
        list(config.bridge._convert([reloaded], lazy, '', True, 'Reloading test: '))
        reexported = _export(config, reloaded)
        assert reexported.keys() == exported.keys()
        for key in exported:
            torch.testing.assert_close(reexported[key], exported[key], atol=0, rtol=0)

        model.config.language_model_only = False
        model.visual = _visual(hidden_size=16, audio_token_id=31).cuda()
        input_ids = torch.tensor([[1] + [31] * 13 + [2, 2]], device='cuda')
        position_ids = torch.arange(16, device='cuda').unsqueeze(0)
        attention_mask = torch.triu(torch.ones(1, 1, 16, 16, device='cuda', dtype=torch.bool), diagonal=1)
        logits = model(
            input_ids,
            position_ids,
            attention_mask,
            input_features=torch.zeros(1, 104, 128, device='cuda'),
            input_features_mask=torch.ones(1, 104, dtype=torch.bool, device='cuda'))
        assert logits.shape == (1, 16, config.padded_vocab_size)
        logits.sum().backward()
        assert model.visual.perception.scale.grad is not None
        assert model.visual.perception.scale.grad != 0


def test_tiny_repeated_mtp_roundtrip_and_backward():
    from megatron.core.transformer.moe.moe_logging import destroy_moe_metrics_tracker
    from test_glm5_next import _parallel_context

    from mcore_bridge.model.register import get_mcore_model

    destroy_moe_metrics_tracker()
    with _parallel_context():
        model, config = _build_tiny_model(mtp=True)
        assert config.mtp_num_layers == 2
        assert config.mtp_use_repeated_layer
        assert len(model.language_model.mtp.layers) == 1

        exported = _export(config, model)
        assert 'llm.mtp.layers.0.eh_proj.weight' in exported
        assert 'llm.mtp.layers.1.final_layernorm.weight' in exported
        reloaded = get_mcore_model(config)[0].cuda()
        lazy = {key: LazyTensor(tensor=value) for key, value in exported.items()}
        list(config.bridge._convert([reloaded], lazy, '', True, 'Reloading MTP test: '))
        reexported = _export(config, reloaded)
        assert reexported.keys() == exported.keys()
        for key in exported:
            torch.testing.assert_close(reexported[key], exported[key], atol=0, rtol=0)

        model.train()
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], device='cuda')
        position_ids = torch.arange(6, device='cuda').unsqueeze(0)
        attention_mask = torch.triu(torch.ones(1, 1, 6, 6, device='cuda', dtype=torch.bool), diagonal=1)
        labels = torch.tensor([[2, 3, 4, 5, 6, 2]], device='cuda')
        output = model(input_ids, position_ids, attention_mask, labels=labels)
        loss = output[0] if isinstance(output, tuple) else output
        loss.float().mean().backward()
        assert any(parameter.grad is not None for parameter in model.language_model.mtp.parameters())
        destroy_moe_metrics_tracker()


def test_perception_import_is_strict():
    bridge = speechlm.NemoSpeechLMBridge.__new__(speechlm.NemoSpeechLMBridge)
    bridge._peft_format = False
    module = _Perception()
    with pytest.raises(ValueError, match='Perception checkpoint mismatch'):
        bridge._set_module(module, {}, 'perception.', True)
    state = {
        'perception.scale': LazyTensor(tensor=torch.tensor(3.)),
        'perception.encoder._extra_state': LazyTensor(tensor=torch.tensor(1)),
    }
    assert bridge._set_module(module, state, 'perception.', True) == {}
    assert module.scale.item() == 3


def test_pack_nemo_experts_rejects_cross_layer_prefixes():
    bridge = speechlm.NemoSpeechLMBridge.__new__(speechlm.NemoSpeechLMBridge)
    bridge.config = NS(num_moe_experts=2)
    state = {
        'a.experts.0.up_proj.weight': torch.zeros(2, 2),
        'a.experts.1.up_proj.weight': torch.zeros(2, 2),
        'b.experts.0.up_proj.weight': torch.zeros(2, 2),
        'b.experts.1.up_proj.weight': torch.zeros(2, 2),
    }
    with pytest.raises(ValueError, match='multiple experts prefixes'):
        bridge._pack_nemo_experts(state)


@pytest.mark.skipif(not _HAS_NEMO_CONVERSION, reason='requires Transformers nemo_speechlm conversion mapping')
def test_transformers_perception_key_mapping():
    state = {
        'proj.weight': torch.zeros(3, 2),
        'encoder.asr_encoder.layers.0.attn.w_qkv.weight': torch.zeros(6, 2, 2),
        'encoder.asr_encoder.layers.0.attn.out_proj.weight': torch.zeros(2, 2),
        'encoder.asr_encoder.layers.0.norm1.weight': torch.zeros(2),
        'encoder.asr_encoder.layers.0.ffn.net.0.weight': torch.zeros(2, 2),
        'encoder.asr_encoder.layers.0.ffn.net.3.weight': torch.zeros(2, 2),
        'encoder.asr_encoder.pre_encode.proj.weight': torch.zeros(2, 2),
        'encoder.asr_norm.weight': torch.zeros(2),
        'encoder.diar_norm.weight': torch.zeros(2),
        'encoder.diarization_model.encoder.layers.0.attn.w_qkv.weight': torch.zeros(6, 2, 2),
        'encoder.diarization_model.sortformer_modules.encoder_proj.weight': torch.zeros(2, 2),
        'encoder.diarization_model.sortformer_modules.subpixel_upsample.weight': torch.zeros(2),
        'encoder.diarization_model.sortformer_modules.first_hidden_to_hidden.weight': torch.zeros(2, 2),
        'encoder.diarization_model.sortformer_modules.single_hidden_to_spks.weight': torch.zeros(2, 2),
        'encoder.diarization_model.sortformer_modules.learnable_sil_emb': torch.zeros(2),
        'encoder.diarization_model.sortformer_modules.hidden_to_spks.weight': torch.zeros(1, 2),
    }
    mapped = speechlm._convert_perception_state(state)
    assert 'encoder.asr_encoder.layers.0.self_attn.q_proj.weight' in mapped
    assert 'encoder.diarization_model.model.upsampler.conv.weight' in mapped
    assert 'encoder.diarization_model.silence_embeds' in mapped
    assert 'encoder.diarization_model.sortformer_modules.hidden_to_spks.weight' not in mapped
    inverse = speechlm._convert_perception_state_inverse(mapped)
    expected = set(state) - {'encoder.diarization_model.sortformer_modules.hidden_to_spks.weight'}
    assert set(inverse) == expected
    torch.testing.assert_close(inverse['encoder.asr_encoder.layers.0.attn.w_qkv.weight'],
                               state['encoder.asr_encoder.layers.0.attn.w_qkv.weight'])


@pytest.mark.skipif(not _HAS_NEMO_CONVERSION, reason='requires Transformers nemo_speechlm conversion mapping')
def test_inverse_perception_rejects_unmapped_key():
    with pytest.raises(ValueError, match='not mapped back'):
        speechlm._convert_perception_state_inverse({'unknown.weight': torch.zeros(1)})


@pytest.mark.skipif(not _HAS_NEMO_CONVERSION, reason='requires Transformers nemo_speechlm conversion mapping')
def test_inverse_perception_rejects_non_perception_target(monkeypatch):
    import transformers.core_model_loading as core_model_loading

    monkeypatch.setattr(core_model_loading, 'rename_source_key', lambda *args, **kwargs: ('not_perception', None))
    with pytest.raises(ValueError, match='not mapped back'):
        speechlm._convert_perception_state_inverse({'encoder.unknown.weight': torch.zeros(1)})


def test_full_export_preserves_source_only_weights(tmp_path):
    from safetensors.torch import load_file, save_file
    from test_glm5_next import _parallel_context

    with _parallel_context():
        model, config = _build_tiny_model()
        source_dir = tmp_path / 'source'
        output_dir = tmp_path / 'output'
        source_dir.mkdir()
        source = {key: value.detach().cpu() for key, value in _export(config, model).items()}
        source['llm.mtp.layers.0.eh_proj.weight'] = torch.full((2, 3), 7.)
        source['ctc_timestamp.weight'] = torch.full((2, 3), 8.)
        source['perception.encoder._extra_state'] = torch.tensor([9])
        save_file(source, str(source_dir / 'model.safetensors'))
        config.bridge._source_model_dir = str(source_dir)
        config.bridge.save_weights([model], str(output_dir))
        result = load_file(str(output_dir / 'model.safetensors'))
        for key in ('llm.mtp.layers.0.eh_proj.weight', 'ctc_timestamp.weight', 'perception.encoder._extra_state'):
            torch.testing.assert_close(result[key], source[key])


def _parallel_roundtrip(tp=1, pp=1, ep=1, sequence_parallel=False, mtp=False):
    from mcore_bridge.model.register import get_mcore_model

    model, config = _build_tiny_model(tp=tp, pp=pp, ep=ep, sequence_parallel=sequence_parallel, mtp=mtp)
    exported = _export(config, model)
    assert exported['llm.model.embed_tokens.weight'].shape == (32, 16)
    assert exported['llm.lm_head.weight'].shape == (32, 16)
    reloaded = get_mcore_model(config)[0].cuda()
    lazy = {key: LazyTensor(tensor=value) for key, value in exported.items()}
    list(config.bridge._convert([reloaded], lazy, '', True, 'Reloading test: '))
    reexported = _export(config, reloaded)
    assert reexported.keys() == exported.keys()
    for key in exported:
        torch.testing.assert_close(reexported[key], exported[key], atol=0, rtol=0)
    return model, config


def test_nemo_speechlm_tp2_sp_roundtrip_forward_backward():
    if os.environ.get('NEMO_SPEECHLM_PARALLEL_TEST') != 'tp2':
        pytest.skip('run with NEMO_SPEECHLM_PARALLEL_TEST=tp2 torchrun --nproc-per-node=2')
    from test_glm5_next import _parallel_context

    mtp = os.environ.get('NEMO_SPEECHLM_MTP') == '1'
    with _parallel_context(tp=2):
        model, config = _parallel_roundtrip(tp=2, sequence_parallel=True, mtp=mtp)
        _attach_visual(model, config)
        logits = _test_logits(model, requires_grad=True)
        assert logits.shape == (1, 6, 32)
        logits.sum().backward()
        assert model.visual.perception.scale.grad is not None
        if mtp:
            assert any(p.grad is not None for p in model.language_model.mtp.parameters())


def test_nemo_speechlm_ep2_roundtrip_forward_backward():
    if os.environ.get('NEMO_SPEECHLM_PARALLEL_TEST') != 'ep2':
        pytest.skip('run with NEMO_SPEECHLM_PARALLEL_TEST=ep2 torchrun --nproc-per-node=2')
    from test_glm5_next import _parallel_context

    mtp = os.environ.get('NEMO_SPEECHLM_MTP') == '1'
    with _parallel_context(ep=2):
        model, config = _parallel_roundtrip(ep=2, mtp=mtp)
        _attach_visual(model, config)
        logits = _test_logits(model, requires_grad=True)
        assert logits.shape == (1, 6, 32)
        logits.sum().backward()
        assert model.visual.perception.scale.grad is not None
        if mtp:
            assert any(p.grad is not None for p in model.language_model.mtp.parameters())


def test_nemo_speechlm_pp2_roundtrip_forward_backward():
    if os.environ.get('NEMO_SPEECHLM_PARALLEL_TEST') != 'pp2':
        pytest.skip('run with NEMO_SPEECHLM_PARALLEL_TEST=pp2 torchrun --nproc-per-node=2')
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from test_glm5_next import _parallel_context

    mtp = os.environ.get('NEMO_SPEECHLM_MTP') == '1'
    with _parallel_context(pp=2):
        model, config = _parallel_roundtrip(pp=2, mtp=mtp)
        if model.pre_process:
            _attach_visual(model, config)
        else:
            config.language_model_only = False
        input_ids = torch.tensor([[1] + [31] * 13 + [2, 2]], device='cuda')
        position_ids = torch.arange(16, device='cuda').unsqueeze(0)
        attention_mask = torch.triu(torch.ones(1, 1, 16, 16, device='cuda', dtype=torch.bool), diagonal=1)
        labels = torch.arange(1, 17, device='cuda').unsqueeze(0)

        def forward_step(data_iterator, stage_model):
            ids, positions, mask, target = next(data_iterator)
            output = stage_model(
                ids,
                positions,
                mask,
                labels=target,
                input_features=torch.zeros(1, 104, 128, device='cuda'),
                input_features_mask=torch.ones(1, 104, dtype=torch.bool, device='cuda'))

            def loss_func(tensor):
                loss = tensor.float().mean()
                return loss, {'loss': loss.detach()}

            return output, loss_func

        losses = get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=iter([(input_ids, position_ids, attention_mask, labels)]),
            model=[model],
            num_microbatches=1,
            seq_length=16,
            micro_batch_size=1,
            forward_only=False,
        )
        if parallel_state.is_pipeline_last_stage():
            assert len(losses) == 1 and torch.isfinite(losses[0]['loss'])
            if mtp:
                assert any(p.grad is not None for p in model.language_model.mtp.parameters())
        assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
        if model.pre_process:
            assert model.visual.perception.scale.grad is not None


_LORA_TARGETS = ['linear_qkv', 'linear_proj', 'proj']


def _attach_visual(model, config):
    model.visual = _visual(hidden_size=16, audio_token_id=31).cuda()
    model.visual.perception.proj = torch.nn.Linear(16, 16).cuda()
    model.config.language_model_only = False
    config.language_model_only = False


def _wrap_lora(model):
    from peft import LoraConfig, get_peft_model

    return get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=_LORA_TARGETS, lora_dropout=0.))


def _test_logits(model, requires_grad=False):
    model.eval()
    input_ids = torch.tensor([[1] + [31] * 13 + [2, 2]], device='cuda')
    position_ids = torch.arange(16, device='cuda').unsqueeze(0)
    attention_mask = torch.triu(torch.ones(1, 1, 16, 16, device='cuda', dtype=torch.bool), diagonal=1)
    with torch.set_grad_enabled(requires_grad):
        logits = model(
            input_ids,
            position_ids,
            attention_mask,
            input_features=torch.zeros(1, 104, 128, device='cuda'),
            input_features_mask=torch.ones(1, 104, dtype=torch.bool, device='cuda'),
            runtime_gather_output=True)
    return logits[..., :model.config.actual_vocab_size].cpu()


def _save_lora_fixture(output_dir):
    from safetensors.torch import save_file
    from test_glm5_next import _parallel_context

    with _parallel_context():
        model, config = _build_tiny_model()
        _attach_visual(model, config)
        peft_model = _wrap_lora(model)
        base = {key: value.detach().cpu() for key, value in _export(config, model).items()}
        save_file(base, os.path.join(output_dir, 'base.safetensors'))
        for name, parameter in peft_model.named_parameters():
            assert not ('lora_' in name and '.experts.' in name)
            if 'lora_A' in name or 'lora_B' in name:
                torch.nn.init.normal_(parameter, std=0.1)
        torch.save(_test_logits(peft_model), os.path.join(output_dir, 'expected.pt'))
        config.bridge.save_weights([peft_model], os.path.join(output_dir, 'adapter'), peft_format=True)
        peft_model.peft_config['default'].save_pretrained(os.path.join(output_dir, 'adapter'))


def _check_lora_fixture(output_dir):
    from peft import PeftConfig, get_peft_model
    from safetensors.torch import load_file
    from test_glm5_next import _parallel_context

    with _parallel_context():
        model, config = _build_tiny_model()
        _attach_visual(model, config)
        base = {
            key: LazyTensor(tensor=value)
            for key, value in load_file(os.path.join(output_dir, 'base.safetensors')).items()
        }
        list(config.bridge._convert([model], base, '', True, 'Loading base: '))
        adapter_dir = os.path.join(output_dir, 'adapter')
        peft_model = get_peft_model(model, PeftConfig.from_pretrained(adapter_dir))
        config.bridge.load_weights([peft_model], adapter_dir, peft_format=True)
        actual = _test_logits(peft_model)
        expected = torch.load(os.path.join(output_dir, 'expected.pt'), weights_only=True)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_lora_adapter_save_reload_fresh_process(tmp_path):
    if not torch.cuda.is_available() or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        pytest.skip('requires one CUDA device')
    root = os.path.dirname(os.path.dirname(__file__))
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(
        [os.path.join(root, 'src'), os.path.join(root, 'tests'),
         env.get('PYTHONPATH', '')])
    env['NCCL_DEBUG'] = 'WARN'
    for mode in ('save', 'check'):
        result = subprocess.run([sys.executable, __file__, mode, str(tmp_path)],
                                env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                timeout=180)
        assert result.returncode == 0, result.stdout


if __name__ == '__main__':
    mode, output_dir = sys.argv[1:]
    if mode == 'save':
        _save_lora_fixture(output_dir)
    else:
        _check_lora_fixture(output_dir)
