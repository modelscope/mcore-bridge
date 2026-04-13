import types

import pytest
import torch
from transformers import PretrainedConfig

from mcore_bridge.config import hf_to_mcore_config
from mcore_bridge.model.gpt_model import GPTModel
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionBlock


class DummyConfig(PretrainedConfig):
    model_type = 'dummy'


class RecorderLayer:

    def __init__(self, tag: int):
        self.tag = tag
        self.calls = []

    def __call__(self, hidden_states, input_ids, position_ids, depth_idx=None, **kwargs):
        self.calls.append(depth_idx)
        return hidden_states + self.tag + depth_idx, input_ids, position_ids


def test_hf_to_mcore_config_reads_mtp_unroll_steps():
    config = DummyConfig()
    config.mtp_unroll_steps = 4

    res = hf_to_mcore_config(config)

    assert res['mtp_unroll_steps'] == 4


def test_mtp_block_reuses_physical_layers_for_unroll_steps():
    layer0 = RecorderLayer(10)
    layer1 = RecorderLayer(20)
    block = types.SimpleNamespace(
        config=types.SimpleNamespace(
            mtp_num_layers=2,
            mtp_unroll_steps=5,
            pipeline_model_parallel_size=1,
            pipeline_model_parallel_layout=None,
        ),
        vp_stage=None,
        layers=[layer0, layer1],
    )
    hidden_states = torch.zeros(2, 1, 1)
    input_ids = torch.zeros(1, 2, dtype=torch.long)
    position_ids = torch.zeros(1, 2, dtype=torch.long)
    attention_mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)

    outputs = MultiTokenPredictionBlock.forward(
        block,
        input_ids=input_ids,
        position_ids=position_ids,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
    )

    assert layer0.calls == [1, 3, 5]
    assert layer1.calls == [2, 4]
    assert outputs.shape[0] == hidden_states.shape[0] * (1 + block.config.mtp_unroll_steps)


def test_multimodal_mtp_reuses_decoder_input_without_detach():
    captured = {}

    class StopMTP(RuntimeError):
        pass

    def mtp(**kwargs):
        captured.update(kwargs.get('extra_block_kwargs') or {})
        raise StopMTP

    model = types.SimpleNamespace(
        post_process=True,
        config=types.SimpleNamespace(
            task_type='causal_lm',
            is_multimodal=True,
            context_parallel_size=1,
            mtp_num_layers=1,
            mtp_unroll_steps=None,
        ),
        training=True,
        share_embeddings_and_output_weights=False,
        mtp_process=True,
        mtp=mtp,
        embedding=object(),
    )
    decoder_input = torch.randn(2, 1, 4, requires_grad=True)

    with pytest.raises(StopMTP):
        GPTModel._postprocess(
            model,
            hidden_states=torch.randn(2, 1, 4),
            input_ids=torch.zeros(1, 2, dtype=torch.long),
            position_ids=torch.zeros(1, 2, dtype=torch.long),
            labels=torch.zeros(1, 2, dtype=torch.long),
            rotary_pos_emb=torch.zeros(2, 1, 4),
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            decoder_input=decoder_input,
            attention_mask=torch.zeros(1, 1, 2, 2, dtype=torch.bool),
        )

    assert captured['base_decoder_input'] is decoder_input
    assert captured['base_decoder_input'].requires_grad
