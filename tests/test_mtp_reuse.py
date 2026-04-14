import pytest
import torch
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionBlock
from types import SimpleNamespace

import mcore_bridge  # noqa: F401
import mcore_bridge.model.gpt_model as gpt_model_mod
from mcore_bridge.model.gpt_model import GPTModel


class RecordingLayer:

    def __init__(self):
        self.depth_history = []

    def __call__(
        self,
        *,
        input_ids,
        position_ids,
        hidden_states,
        attention_mask,
        depth_idx=None,
        **kwargs,
    ):
        self.depth_history.append(depth_idx)
        return hidden_states + depth_idx, input_ids, position_ids


class RecordingOutputLayer:

    def __init__(self):
        self.calls = []
        self.sequence_parallel = False

    def __call__(self, hidden_states, weight=None, runtime_gather_output=None):
        self.calls.append(hidden_states.clone())
        return hidden_states.squeeze(-1).transpose(0, 1), None


def test_mtp_block_reuses_single_physical_layer_across_unroll_steps():
    layer = RecordingLayer()
    block = SimpleNamespace(
        config=SimpleNamespace(
            mtp_num_layers=1,
            mtp_unroll_steps=3,
            pipeline_model_parallel_size=1,
            pipeline_model_parallel_layout=None,
        ),
        vp_stage=None,
        layers=[layer],
    )

    hidden_states = torch.zeros(2, 1, 1)
    input_ids = torch.zeros(1, 2, dtype=torch.long)
    position_ids = torch.zeros(1, 2, dtype=torch.long)

    output = MultiTokenPredictionBlock.forward(
        block,
        input_ids=input_ids,
        position_ids=position_ids,
        hidden_states=hidden_states,
        attention_mask=None,
    )

    assert layer.depth_history == [1, 2, 3]

    chunks = torch.chunk(output, 4, dim=0)
    assert [chunk[0, 0, 0].item() for chunk in chunks] == [0.0, 1.0, 3.0, 6.0]


def test_postprocess_uses_unroll_steps_for_mtp_loss(monkeypatch):
    saved_losses = []
    monkeypatch.setattr(
        gpt_model_mod,
        'roll_tensor',
        lambda tensor, shifts, dims, cp_group=None, packed_seq_params=None: (tensor, tensor.numel()),
    )
    monkeypatch.setattr(
        gpt_model_mod.MTPLossAutoScaler,
        'apply',
        lambda hidden_states, scaled_loss: hidden_states,
    )
    monkeypatch.setattr(
        gpt_model_mod.MTPLossLoggingHelper,
        'save_loss_to_tracker',
        lambda loss, layer_number, total_layers, avg_group=None: saved_losses.append((layer_number, total_layers)),
    )
    monkeypatch.setattr(
        gpt_model_mod.parallel_state,
        'get_data_parallel_group',
        lambda with_context_parallel=True: None,
    )
    monkeypatch.setattr(gpt_model_mod, 'has_config_logger_enabled', lambda config: False)

    output_layer = RecordingOutputLayer()

    def mtp_forward(**kwargs):
        hidden_states = kwargs['hidden_states']
        return torch.cat([hidden_states, hidden_states + 1, hidden_states + 2, hidden_states + 3], dim=0)

    model = SimpleNamespace(
        post_process=True,
        mtp_process=True,
        training=True,
        share_embeddings_and_output_weights=False,
        cp_group=None,
        embedding=lambda *args, **kwargs: None,
        output_layer=output_layer,
        mtp=mtp_forward,
        config=SimpleNamespace(
            task_type='causal_lm',
            is_multimodal=False,
            context_parallel_size=1,
            mtp_num_layers=1,
            mtp_unroll_steps=3,
            decoder_input_detach=True,
            calculate_per_token_loss=False,
            mtp_loss_scaling_factor=0.3,
            sequence_parallel=False,
            tensor_model_parallel_size=1,
        ),
        compute_language_model_loss=lambda labels, logits: logits.float(),
    )

    loss = GPTModel._postprocess(
        model,
        hidden_states=torch.zeros(1, 1, 1),
        input_ids=None,
        position_ids=None,
        labels=torch.ones(1, 1, dtype=torch.long),
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        loss_mask=torch.ones(1, 1, dtype=torch.bool),
        decoder_input=None,
        attention_mask=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        runtime_gather_output=False,
        extra_block_kwargs=None,
        inference_context=None,
    )

    assert loss.shape == (1, 1)
    assert [call[0, 0, 0].item() for call in output_layer.calls] == [1.0, 2.0, 3.0, 0.0]
    assert saved_losses == [(0, 3), (1, 3), (2, 3)]


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
