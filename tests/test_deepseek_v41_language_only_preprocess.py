# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
from types import SimpleNamespace
from unittest.mock import patch

from mcore_bridge.model.gpts.deepseek_v4 import DeepseekV4Bridge
from mcore_bridge.model.mm_gpts.deepseek_v41 import DeepseekV41Bridge


@pytest.fixture
def bridge():
    result = DeepseekV41Bridge.__new__(DeepseekV41Bridge)
    result.config = SimpleNamespace(
        language_model_only=True,
        hf_config=SimpleNamespace(quantization_config=None),
        padded_vocab_size=8,
        actual_vocab_size=6,
        task_type='causal_lm',
    )
    result.hf_embed_key = 'model.embed_tokens.weight'
    result.is_multimodal = True
    result.tp_size = result.etp_size = result.ep_size = result.pp_size = result.ep_pp_size = 1
    result.tp_rank = result.etp_rank = 0
    result._peft_format = False
    result._target_device = None
    result._only_master_rank = False
    return result


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('prefix', ['', 'checkpoint.'])
def test_language_only_loads_embeddings_without_visual_weights(bridge, wrapped, prefix):
    embedding = torch.nn.Embedding(8, 4, dtype=torch.bfloat16)
    backbone = SimpleNamespace(embedding=SimpleNamespace(word_embeddings=embedding))
    model = SimpleNamespace(language_model=backbone, visual=None) if wrapped else backbone
    source = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    state = {f'{prefix}{bridge.hf_embed_key}': SimpleNamespace(load=lambda: source)}
    with patch.object(bridge, '_set_state_dict', wraps=bridge._set_state_dict) as setter:
        result = bridge._convert_pre_process(model, state, prefix, to_mcore=True)
    assert result == {}
    assert setter.call_count == 1
    assert setter.call_args.args[1] == 'embedding.word_embeddings.weight'
    torch.testing.assert_close(embedding.weight.detach()[:6], source, rtol=0, atol=0)
    assert torch.count_nonzero(embedding.weight.detach()[6:]) == 0


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('prefix', ['', 'checkpoint.'])
def test_language_only_exports_embeddings_without_visual_markers(bridge, wrapped, prefix):
    embedding = torch.nn.Embedding(8, 4, dtype=torch.bfloat16)
    embedding.weight.data.copy_(torch.arange(32, dtype=torch.bfloat16).reshape(8, 4))
    backbone = SimpleNamespace(embedding=SimpleNamespace(word_embeddings=embedding))
    model = SimpleNamespace(language_model=backbone, visual=None) if wrapped else backbone
    tensor_factory = torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs['device'] = 'cpu'
        return tensor_factory(*args, **kwargs)

    # Keep the real embedding mapping and export; replace only CUDA control tensors
    # and distributed transport, which are outside this CPU regression.
    with patch('mcore_bridge.bridge.gpt_bridge.torch.tensor', side_effect=cpu_tensor), \
            patch.object(bridge, '_all_gather_tp', side_effect=lambda tensor, *_: tensor), \
            patch.object(bridge, '_broadcast_ep_pp', side_effect=lambda tensor, *_: tensor), \
            patch.object(bridge, '_set_state_dict', wraps=bridge._set_state_dict) as setter:
        result = bridge._convert_pre_process(model, {}, prefix, to_mcore=False)
    assert setter.call_count == 1
    assert set(result) == {f'{prefix}{bridge.hf_embed_key}'}
    torch.testing.assert_close(result[f'{prefix}{bridge.hf_embed_key}'], embedding.weight.detach()[:6], rtol=0, atol=0)


@pytest.mark.parametrize('to_mcore', [False, True])
def test_multimodal_preprocessing_keeps_image_marker_order(bridge, to_mcore):
    bridge.config.language_model_only = False
    model = object()
    state = {}
    parent_result = {}
    with patch.object(DeepseekV4Bridge, '_convert_pre_process', return_value=parent_result) as parent, \
            patch.object(bridge, '_set_state_dict') as setter:
        result = bridge._convert_pre_process(model, state, '', to_mcore)
    parent.assert_called_once_with(model, state, '', to_mcore)
    assert result is parent_result
    target = state if to_mcore else parent_result
    assert [call.args
            for call in setter.call_args_list] == [(model, f'visual.{name}', target, f'model.{name}', to_mcore)
                                                   for name in ('image_start', 'image_end', 'image_newline')]


@pytest.mark.parametrize('first_stage', [False, True])
def test_language_only_conversion_loads_preprocessing_on_first_stage(bridge, first_stage):
    bridge.config.num_layers = 0
    bridge._disable_tqdm = True
    model = SimpleNamespace(vp_stage=None, decoder=SimpleNamespace(layers=[]))
    with patch('mcore_bridge.model.mm_gpts.deepseek_v41.mpu.is_pipeline_first_stage', return_value=first_stage), \
            patch('mcore_bridge.model.mm_gpts.deepseek_v41.mpu.is_pipeline_last_stage', return_value=False), \
            patch.object(bridge, '_convert_hf_state_dict', side_effect=lambda state, _: state), \
            patch.object(bridge, '_convert_additional_layers', return_value=iter(())), \
            patch.object(bridge, '_convert_pre_process', return_value={}) as preprocessing:
        list(bridge._convert([model], {}, '', to_mcore=True))
    assert preprocessing.call_count == int(first_stage)
