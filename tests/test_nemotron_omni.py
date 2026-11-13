# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU contracts; no CUDA kernels or real TP/PP collectives.

The wrapper/Hybrid forwards and bridge are real. Embeddings/decoder and vision
features are small CPU doubles. NEMOTRON_OMNI_HF_DIR enables additional tests of
verbatim official projector/EVS code with only the RADIO tower replaced.
"""
import ast
import os
import pytest
import torch
import warnings
from pathlib import Path
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from types import SimpleNamespace as NS

from mcore_bridge.config.parser import hf_to_mcore_config
from mcore_bridge.model.hybrid_model import HybridModel, McoreHybridModel
from mcore_bridge.model.mm_gpt_model import MultimodalGPTModel, VocabParallelEmbedding
from mcore_bridge.model.mm_gpts import nemotron_h_omni as omni
from mcore_bridge.model.register import MODEL_MAPPING


class Projector(torch.nn.Module):
    downsample_ratio = 0.5
    ps_version = 'v2'

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.calls = []

    def forward(self, pixels, tower):
        self.calls.append(('image', pixels.shape))
        n = (pixels.shape[-2] // 4) * (pixels.shape[-1] // 4)
        return (pixels.mean() * self.weight).expand(1, n, 4)

    def forward_video(self, pixels, tower):
        self.calls.append(('video', pixels.shape))
        n = (pixels.shape[-2] // 4) * (pixels.shape[-1] // 4)
        return (pixels.mean() * self.weight).expand((pixels.shape[0] + 1) // 2, n, 4)


def _visual(rate=0.7):
    visual = omni.NemotronH_OmniVit.__new__(omni.NemotronH_OmniVit)
    torch.nn.Module.__init__(visual)
    visual.img_context_token_id, visual.video_pruning_rate, visual.patch_size = 18, rate, 2
    visual.vision_model = NS(dtype=torch.float32)
    visual.vision_projector = Projector()

    def retention(*, video_embeds, thw, spatial_merge_size, q):
        assert q == rate and spatial_merge_size == 1
        assert thw[1:] == (video_embeds.shape[1] // thw[2], thw[2])
        mask = torch.zeros(video_embeds.shape[:2], dtype=torch.bool)
        mask[0] = True
        return mask.flatten()

    visual.compute_retention_mask = retention
    return visual


def _config(**changes):
    config = NS(
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        mtp_num_layers=0,
        fp8=None,
        language_model_only=False,
        cuda_graph_impl='none',
        hf_config=NS(video_pruning_rate=0.7, sound_config=None, llm_config=NS(num_nextn_predict_layers=1)))
    for key, value in changes.items():
        setattr(config, key, value)
    return config


def _model(monkeypatch, visual=None, **changes):
    """Execute actual Omni -> MMGPT -> Hybrid, stopping only at the upstream decoder."""
    config = _config(**changes)
    model = omni.NemotronH_OmniGPTModel.__new__(omni.NemotronH_OmniGPTModel)
    torch.nn.Module.__init__(model)
    model.config, model.pre_process, model.post_process = config, True, True
    model.vp_stage = None
    model.visual = _visual() if visual is None else visual
    lm = HybridModel.__new__(HybridModel)
    torch.nn.Module.__init__(lm)
    lm.config = config
    vocab = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
    torch.nn.Module.__init__(vocab)
    vocab.reduce_scatter_embeddings = False
    vocab.weight = torch.nn.Parameter(torch.randn(128, 4))
    monkeypatch.setattr(VocabParallelEmbedding, 'forward',
                        lambda self, ids: torch.nn.functional.embedding(ids, self.weight))

    class Embedding(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.word_embeddings = vocab

        def forward(self, input_ids, position_ids):
            return self.word_embeddings(input_ids).transpose(0, 1).contiguous()

    lm.embedding = Embedding()
    model.language_model = lm
    model.seen = {}

    def upstream(self, input_ids, position_ids, attention_mask, **kwargs):
        model.seen.update(dict(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, **kwargs))
        decoder_input = kwargs['decoder_input']
        if decoder_input is None:
            return torch.zeros((*input_ids.shape, 4))  # Stand-in for the PP stage's received activation.
        embeds = decoder_input.transpose(0, 1)
        return embeds.square().sum(-1) if kwargs['labels'] is not None else embeds

    monkeypatch.setattr(McoreHybridModel, 'forward', upstream)
    return model


def _batch():
    # Row 0 has one image and a 3-frame 4x8 video; row 1 is a shorter text sample.
    ids = torch.tensor([[10, 18, 18, 18, 18, 18, 30, 31], [40, 41, 42, 43, 0, 0, 0, 0]])
    image_mask = torch.zeros_like(ids, dtype=torch.bool)
    image_mask[0, 1] = True
    video_mask = torch.zeros_like(ids, dtype=torch.bool)
    video_mask[0, 2:6] = True
    return dict(
        input_ids=ids,
        position_ids=torch.arange(8).expand(2, -1),
        attention_mask=torch.tensor([[1] * 8, [1] * 4 + [0] * 4]),
        labels=torch.tensor([[-100, -100, -100, -100, -100, 30, 31, -100], [41, 42, 43, -100, -100, -100, -100, -100]]),
        image_token_mask=image_mask,
        video_token_mask=video_mask,
        pixel_values=[torch.ones(1, 3, 4, 4) * 2],
        pixel_values_videos=[torch.ones(3, 3, 4, 8) * 3])


@pytest.mark.parametrize('tp', [1, 2, 8])
@pytest.mark.parametrize('sp', [False, True])
def test_evs_next_kept_label_map_and_wrapper(monkeypatch, tp, sp):
    monkeypatch.setenv('USE_NEMOTRON_EVS', '1')
    monkeypatch.setattr('mcore_bridge.model.hybrid_model.mpu.get_tensor_model_parallel_rank', lambda: 0)
    model = _model(monkeypatch, tensor_model_parallel_size=tp, sequence_parallel=sp)
    batch = _batch()
    batch['loss_mask'] = batch['labels'] != -100
    batch['loss_scale'] = torch.arange(16).view(2, 8)
    batch['runtime_gather_output'] = True
    result = model(**batch)
    length = (6 + tp - 1) // tp * tp
    assert result.shape == batch['labels'].shape
    assert torch.equal(model.seen['labels'][1, :6], torch.tensor([41, 42, 43, -100, -100, -100]))
    assert torch.equal(model.seen['labels'][0, :6], torch.tensor([-100, -100, -100, 30, 31, -100]))
    assert (model.seen['labels'][:, 6:] == -100).all()
    assert (model.seen['position_ids'][:, 6:] == 0).all()
    assert model.seen['runtime_gather_output'] is True
    assert 'loss_mask' not in model.seen and 'loss_scale' not in model.seen
    padding = model.seen['attention_mask'].all(dim=(1, 2))
    assert padding[0].sum() == length - 6 and padding[1].sum() == length - 4
    assert model.seen['padding_mask'].shape == (2, length // tp if sp else length)
    embeds = model.seen['decoder_input'].transpose(0, 1)
    assert torch.equal(embeds[0, 1], torch.full((4, ), 2.))
    assert torch.equal(embeds[0, 2:4], torch.full((2, 4), 3.))
    (result * (batch['labels'] != -100)).sum().backward()
    assert model.visual.vision_projector.weight.grad != 0
    assert len(model.visual.vision_projector.calls) == 2


@pytest.mark.parametrize('tp', [1, 8])
def test_no_batch_cache_and_inference_tensor(monkeypatch, tp):
    monkeypatch.setenv('USE_NEMOTRON_EVS', '1')
    model = _model(monkeypatch, tensor_model_parallel_size=tp)
    batch = _batch()
    batch.update(labels=None, inference_params=None, inference_context=None)
    first = model(**batch)
    batch['pixel_values_videos'] = [torch.full((3, 3, 4, 8), 7.)]
    second = model(**batch)
    assert isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor)
    assert torch.equal(first[0, 2:4], torch.full((2, 4), 3.))
    assert torch.equal(second[0, 2:4], torch.full((2, 4), 7.))
    assert len(model.visual.vision_projector.calls) == 4
    assert not hasattr(model.visual, 'pruned_video_embeds')
    model.config.hf_config.video_pruning_rate = 0
    third = model(**batch)
    assert third.shape == (2, 8, 4)
    assert torch.equal(third[0, 2:6], torch.full((4, 4), 7.))


@pytest.mark.parametrize('cache_key', ['inference_params', 'inference_context'])
@pytest.mark.parametrize('labels', [False, True])
@pytest.mark.parametrize('tp', [1, 8])
def test_evs_cached_inference_rejected_before_any_vision(monkeypatch, cache_key, labels, tp):
    monkeypatch.setenv('USE_NEMOTRON_EVS', '1')
    model = _model(monkeypatch, tensor_model_parallel_size=tp)
    batch = _batch()
    if not labels:
        batch['labels'] = None
    context = NS(materialize_only_last_token_logits=True, sequence_len_offset=0)
    batch[cache_key] = context
    with pytest.raises(NotImplementedError, match='video_pruning_rate=0.*full forward'):
        model(**batch)
    assert not model.visual.vision_projector.calls  # Even images must not run before this guard.
    assert not model.seen
    assert context.sequence_len_offset == 0


@pytest.mark.parametrize('cache_key', ['inference_params', 'inference_context'])
@pytest.mark.parametrize('media', ['video_rate_zero', 'image_only', 'text_only'])
def test_cache_without_evs_video_is_not_restricted(monkeypatch, cache_key, media):
    monkeypatch.setenv('USE_NEMOTRON_EVS', '1')
    model = _model(monkeypatch)
    batch = _batch()
    batch['labels'] = None
    context = NS(materialize_only_last_token_logits=True, sequence_len_offset=0)
    batch[cache_key] = context
    if media == 'video_rate_zero':
        model.config.hf_config.video_pruning_rate = 0
    else:
        batch['input_ids'][batch['video_token_mask']] = 10
        batch['video_token_mask'].zero_()
        batch['pixel_values_videos'] = []
        if media == 'text_only':
            batch['input_ids'][batch['image_token_mask']] = 10
            batch['image_token_mask'].zero_()
            batch['pixel_values'] = None
    result = model(**batch)
    assert isinstance(result, torch.Tensor) and result.shape == (2, 8, 4)
    assert model.seen[cache_key] is context


def test_plain_text_dummy_zero_gradient(monkeypatch):
    model = _model(monkeypatch)
    ids = torch.tensor([[1, 2]])
    result = model(ids, torch.tensor([[0, 1]]), labels=ids)
    result.sum().backward()
    assert model.visual.vision_projector.calls == [('image', torch.Size([1, 3, 4, 4]))]
    assert model.visual.vision_projector.weight.grad.item() == 0


@pytest.mark.parametrize('mode', ['overlap', 'wrong_dtype', 'missing', 'too_many_features'])
def test_strict_visual_masks(mode):
    visual = _visual()
    batch = _batch()
    if mode == 'overlap':
        batch['image_token_mask'][0, 2] = True
    elif mode == 'wrong_dtype':
        batch['video_token_mask'] = batch['video_token_mask'].long()
    elif mode == 'missing':
        del batch['video_token_mask']
    else:
        batch['pixel_values'].append(torch.ones(1, 3, 4, 4))
    with pytest.raises(ValueError):
        visual.get_inputs_embeds(torch.zeros(2, 8, 4), **batch)


@pytest.mark.parametrize('mask_kind', ['none', '2d', '4d'])
def test_compression_middle_deletion_padding_and_masks(mask_kind):
    ids = torch.tensor([[8, 18, 18, 18, 9, 10]])
    keep = torch.tensor([[True, True, False, False, True, True]])
    video = ids == 18
    image = torch.zeros_like(video)
    attention = None
    if mask_kind == '2d':
        attention = torch.ones_like(ids)
    elif mask_kind == '4d':
        attention = torch.ones(1, 1, 6, 6, dtype=torch.bool).triu(1)
    # In particular, new prediction index 1 targets original token 4: shifted-label index 3.
    result = omni._compress_evs_inputs(ids, torch.arange(6)[None], attention, ids + 100, keep, image, video, 8)
    assert len(result) == 7
    assert torch.equal(result[0], torch.tensor([[8, 18, 9, 10, 0, 0, 0, 0]]))
    assert torch.equal(result[3], torch.tensor([[108, 118, 109, -100, -100, -100, -100, -100]]))
    assert torch.equal(result[6], torch.tensor([[0, 3, 4, -1, -1, -1, -1, -1]]))
    assert result[5].sum() == 1 and result[4].sum() == 0
    assert result[2][:, :, 4:, :].all() and result[2][:, :, :, 4:].all()
    assert torch.equal(result[2][0, 0, :4, :4], torch.ones(4, 4, dtype=torch.bool).triu(1))


@pytest.mark.parametrize('keep_row', [[True, True, True], [True, False, False], [False, False, False]])
def test_compression_keeps_next_token_labels_and_masks_empty_tails(keep_row):
    ids = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[2, 3, -100]])
    keep = torch.tensor([keep_row])
    mask = torch.zeros_like(keep)
    result = omni._compress_evs_inputs(ids, torch.arange(3)[None], None, labels, keep, mask, mask, 1)
    expected = labels if all(keep_row) else torch.tensor([[-100]])
    expected_indices = torch.tensor([[0, 1, -1]]) if all(keep_row) else torch.tensor([[-1]])
    assert torch.equal(result[3], expected)
    assert torch.equal(result[6], expected_indices)
    if not any(keep_row):
        assert result[2].all()
    assert torch.equal(labels, torch.tensor([[2, 3, -100]]))


@pytest.mark.parametrize('pre_process', [True, False])
def test_hybrid_controls_without_media_or_defaults(monkeypatch, pre_process):
    model = _model(monkeypatch)
    model.pre_process = pre_process
    # No supplied decoder_input: exercise both the real embedding-clear and non-first-stage clear.
    controls = dict(loss_mask=torch.ones(1, 2), runtime_gather_output=True, inference_context=object())
    MultimodalGPTModel.forward(model, torch.ones(1, 2, dtype=torch.long), torch.zeros(1, 2, dtype=torch.long),
                               **controls)
    for key, value in controls.items():
        assert model.seen[key] is value
    assert 'extra_block_kwargs' not in model.seen
    model.seen.clear()
    MultimodalGPTModel.forward(model, torch.ones(1, 2, dtype=torch.long), torch.zeros(1, 2, dtype=torch.long))
    assert 'loss_mask' not in model.seen


@pytest.mark.parametrize('field,value,message', [('context_parallel_size', 2, 'context_parallel_size 1'),
                                                 ('mtp_num_layers', 1, 'mtp_num_layers 0'), ('fp8', 'hybrid', 'FP8'),
                                                 ('pipeline_model_parallel_size', 2, 'pipeline_model_parallel_size 1'),
                                                 ('packing', True, 'padding_free false --packing false'),
                                                 ('padding_free', True, 'padding_free false --packing false')])
def test_loader_early_guards(field, value, message, monkeypatch):
    monkeypatch.setenv('USE_NEMOTRON_EVS', '1')
    config = _config(**{field: value})
    with pytest.raises(NotImplementedError, match=message):
        omni.NemotronH_OmniLoader(config)
    assert config.hf_config.llm_config.num_nextn_predict_layers == 1


@pytest.mark.parametrize('version', ['5.0.0', '5.16.0'])
def test_omni_loader_requires_tested_transformers_api(monkeypatch, version):
    from transformers.utils import versions
    original = versions.importlib.metadata.version
    monkeypatch.setattr(versions.importlib.metadata, 'version', lambda name: version
                        if name == 'transformers' else original(name))
    with pytest.raises(ImportError, match='transformers>=5.16.1'):
        omni.NemotronH_OmniLoader(_config())


def test_non_evs_pp_and_packing_are_not_restricted():
    config = _config(pipeline_model_parallel_size=4, packing=True, padding_free=True)
    config.hf_config.video_pruning_rate = 0
    assert omni.NemotronH_OmniLoader(config).config is config


def test_evs_optin_default_off_relaxes_constraints(monkeypatch):
    # Without USE_NEMOTRON_EVS the config's video_pruning_rate is ignored for training: the
    # Loader drops its PP/packing guards and the forward keeps full-length sequences.
    monkeypatch.delenv('USE_NEMOTRON_EVS', raising=False)
    config = _config(pipeline_model_parallel_size=2, packing=True, padding_free=True)
    assert omni.NemotronH_OmniLoader(config).config is config
    assert omni._evs_env_enabled() is False
    model = _model(monkeypatch)
    result = model(**_batch(), packed_seq_params=object())
    assert result.shape == (2, 8)  # Labels present: the upstream stub returns per-token losses.
    assert model.seen['input_ids'].shape == (2, 8)


def test_packed_rejected_before_vision(monkeypatch):
    monkeypatch.setenv('USE_NEMOTRON_EVS', '1')
    model = _model(monkeypatch)
    with pytest.raises(NotImplementedError, match='padding_free false --packing false'):
        model(**_batch(), packed_seq_params=object())
    assert not model.visual.vision_projector.calls


def test_canonical_parser_and_registration():
    cfg = PretrainedConfig()
    cfg.model_type = 'nemotron_h_omni'
    cfg.llm_config = PretrainedConfig(hidden_size=4, num_hidden_layers=1, num_attention_heads=1, moe_latent_size=2)
    cfg.llm_config.model_type = 'nemotron_h'
    result = hf_to_mcore_config(cfg)
    assert result['hf_model_type'] == 'nemotron_h_omni' and result['llm_model_type'] == 'nemotron_h'
    assert result['moe_latent_size'] == 2 and result['is_hybrid_model']
    assert MODEL_MAPPING['nemotron_h_omni'].loader is omni.NemotronH_OmniLoader


class RadioFixture(PreTrainedModel):
    config_class = PretrainedConfig

    @classmethod
    def is_custom_code(cls):
        return False

    def __init__(self):
        cfg = PretrainedConfig()
        cfg.model_type = 'radio'
        super().__init__(cfg)
        self.encoder = torch.nn.Module()
        self.encoder.layer = torch.nn.ModuleList([torch.nn.Module()])
        attention = self.encoder.layer[0].attention = torch.nn.Module()
        attention.attention = torch.nn.Module()
        for name in ('query', 'key', 'value'):
            setattr(attention.attention, name, torch.nn.Linear(4, 4, bias=False))


def _bridge():
    bridge = omni.NemotronH_OmniBridge.__new__(omni.NemotronH_OmniBridge)
    bridge.pp_size, bridge.pp_rank = 1, 0
    bridge._peft_format, bridge._adapter_name, bridge._target_device = False, 'default', 'cpu'
    return bridge


def test_radio_official_conversion_roundtrip(tmp_path):
    radio, bridge = RadioFixture(), _bridge()
    raw = {'vision_model.radio_model.model.blocks.0.attn.qkv.weight': torch.arange(48.).view(12, 4)}
    lazy = {k: NS(load=lambda v=v: v) for k, v in raw.items()}
    bridge._set_module(radio, lazy, 'vision_model.', True)
    output = bridge._set_module(radio, {}, 'vision_model.', False)
    assert output.keys() == raw.keys()
    torch.testing.assert_close(output[next(iter(raw))], raw[next(iter(raw))], rtol=0, atol=0)
    path = tmp_path / 'vision.pt'
    torch.save(output, path)
    reloaded = torch.load(path, weights_only=True)
    fresh = RadioFixture()
    bridge._set_module(fresh, {k: NS(load=lambda v=v: v) for k, v in reloaded.items()}, 'vision_model.', True)
    for key, value in radio.state_dict().items():
        torch.testing.assert_close(fresh.state_dict()[key], value, rtol=0, atol=0)


def test_radio_root_modules_to_save_full_export():
    from peft.utils import ModulesToSaveWrapper
    radio = ModulesToSaveWrapper(RadioFixture(), 'default')
    radio.update('inactive')
    with torch.no_grad():
        radio.modules_to_save.default.encoder.layer[0].attention.attention.query.weight.fill_(7.)
        radio.modules_to_save.inactive.encoder.layer[0].attention.attention.query.weight.fill_(99.)
    state = _bridge()._set_module(radio, {}, 'vision_model.', False)
    key = 'vision_model.radio_model.model.blocks.0.attn.qkv.weight'
    assert set(state) == {key}
    assert torch.equal(state[key][:4], torch.full((4, 4), 7.))


def test_radio_full_export_unmerged_and_merged_lora():
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(RadioFixture(), LoraConfig(target_modules=['query'], r=2, lora_alpha=2))
    radio = model.base_model.model
    query = radio.encoder.layer[0].attention.attention.query
    with torch.no_grad():
        query.lora_A.default.weight.fill_(1.)
        query.lora_B.default.weight.fill_(0.5)
    key = 'vision_model.radio_model.model.blocks.0.attn.qkv.weight'
    bridge = _bridge()
    base = bridge._set_module(radio, {}, 'vision_model.', False)[key].clone()
    query.merge()
    merged = bridge._set_module(radio, {}, 'vision_model.', False)[key].clone()
    torch.testing.assert_close(merged[:4], base[:4] + 1.)
    torch.testing.assert_close(merged[4:], base[4:])
    query.unmerge()
    torch.testing.assert_close(bridge._set_module(radio, {}, 'vision_model.', False)[key], base)


def test_evs_no_deletion_keeps_tensor_result(monkeypatch):
    monkeypatch.setenv('USE_NEMOTRON_EVS', '1')
    visual = _visual()
    visual.compute_retention_mask = lambda **kw: torch.ones(4, dtype=torch.bool)
    model = _model(monkeypatch, visual=visual)
    batch = _batch()
    result = model(**batch)
    assert isinstance(result, torch.Tensor) and result.shape == (2, 8)
    assert len(visual.vision_projector.calls) == 2
    assert model.seen['labels'] is batch['labels']


def test_projector_full_export_wrapper_filtering():
    from peft import LoraConfig, get_peft_model
    projector = torch.nn.Module()
    projector.norm = torch.nn.LayerNorm(4)
    projector.linear1 = torch.nn.Linear(4, 4)
    projector.linear2 = torch.nn.Linear(4, 4)
    model = get_peft_model(projector, LoraConfig(target_modules=['linear1'], modules_to_save=['linear2']))
    bridge = _bridge()
    output = bridge._set_module(model.base_model.model, {}, 'mlp1.', False)
    assert set(output) == {
        'mlp1.0.weight', 'mlp1.0.bias', 'mlp1.1.weight', 'mlp1.1.bias', 'mlp1.3.weight', 'mlp1.3.bias'
    }
    assert torch.equal(output['mlp1.3.weight'], model.base_model.model.linear2.modules_to_save.default.weight)
    bridge._peft_format = True
    bridge._reduce_tensor_pp_group = lambda value, to_mcore: value  # No CUDA scalar for this PP1 boundary test.
    with pytest.raises(NotImplementedError, match='HF visual LoRA'):
        bridge._set_module(model.base_model.model, {}, 'mlp1.', False)
    assert bridge._set_module(torch.nn.Linear(4, 4), {}, 'mlp1.', False) == {}


def test_full_export_uses_base_pp_path(monkeypatch):
    bridge = _bridge()
    bridge.pp_size, bridge.pp_group = 2, object()
    radio = RadioFixture()
    import mcore_bridge.bridge.gpt_bridge as base
    metadata, values = [], []
    tensor = torch.tensor
    monkeypatch.setattr(torch, 'tensor', lambda *args, **kw: tensor(*args, **{**kw, 'device': 'cpu'}))
    monkeypatch.setattr(base.dist, 'all_reduce', lambda *args, **kw: None)
    monkeypatch.setattr(base.dist, 'get_global_rank', lambda *args: 0)

    def broadcast(data, **kw):
        if bridge.pp_rank == 0:
            metadata[:] = data
        else:
            data[:] = metadata

    monkeypatch.setattr(base.dist, 'broadcast_object_list', broadcast)

    def weight(value, key):
        if bridge.pp_rank == 0:
            values.append(value.clone())
            return value, None
        return values.pop(0), None

    bridge._get_weight = weight
    first = bridge._set_module(radio, {}, 'vision_model.', False)
    bridge.pp_rank = 1
    last = bridge._set_module(None, {}, 'vision_model.', False)
    assert first.keys() == last.keys()
    for key in first:
        torch.testing.assert_close(first[key], last[key], rtol=0, atol=0)


def _official_objects():
    directory = os.environ.get('NEMOTRON_OMNI_HF_DIR')
    if not directory:
        pytest.skip('Set NEMOTRON_OMNI_HF_DIR to test official projector/EVS without loading checkpoint weights.')
    source = Path(directory, 'modeling_nemotron_h_omni.py')
    names = {
        'NemotronH_Omni_RMSNorm', 'NemotronH_Omni_Reasoning_V3MLP', 'NemotronH_Omni_Reasoning_V3VisionProjector',
        'compute_retention_mask'
    }
    nodes = [node for node in ast.parse(source.read_text()).body if getattr(node, 'name', '') in names]
    assert len(nodes) == len(names)
    namespace = dict(
        torch=torch,
        nn=torch.nn,
        ACT2FN=ACT2FN,
        warnings=warnings,
        use_kernel_forward_from_hub=lambda *a, **kw: lambda cls: cls)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('frames', [(2, 2), (3, 2), (3, 3)])
@pytest.mark.parametrize('prune', [False, True])
def test_official_projector_clip_isolation_rectangular_grid(frames, prune):
    objects = _official_objects()
    visual = _visual()
    config = NS(
        downsample_ratio=0.5,
        video_temporal_patch_size=2,
        ps_version='v2',
        vit_hidden_size=4,
        projector_hidden_size=8,
        llm_config=NS(hidden_size=4, num_nextn_predict_layers=1),
        vision_config=NS(layer_norm_eps=1e-5))
    visual.vision_projector = objects['NemotronH_Omni_Reasoning_V3VisionProjector'](config)
    visual.compute_retention_mask = objects['compute_retention_mask']

    class Tower(torch.nn.Module):
        patch_size = 2
        dtype = torch.float32
        config = NS(torch_dtype=torch.float32)

        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, pixels, **kw):
            self.calls.append(pixels.clone())
            features = torch.nn.functional.avg_pool2d(pixels.mean(1, keepdim=True), 2)
            return NS(features=features.flatten(2).transpose(1, 2).expand(-1, -1, 4))

    visual.vision_model = Tower()
    clips = [
        torch.arange(frames[0] * 3 * 8 * 12.).view(frames[0], 3, 8, 12),
        torch.arange(frames[1] * 3 * 12 * 8.).view(frames[1], 3, 12, 8) + 10000
    ]
    images = [torch.ones(1, 3, 8, 8), torch.ones(1, 3, 8, 12)]
    combined = visual.encode_media(images, clips, prune=prune)
    assert combined[0].shape == (10, 4)
    independent = [visual.encode_media(None, [clip], prune=prune) for clip in clips]
    torch.testing.assert_close(combined[1], torch.cat([result[1] for result in independent]))
    if prune:
        assert combined[2].sum() == 12  # Independent first 2x3 or 3x2 grids survive.
    else:
        assert combined[1].shape[0] == sum((n + 1) // 2 for n in frames) * 6
    assert visual.vision_projector.vision_final_layernorm is not None
    # First odd clip repeats its own final frame; no subsequent clip contributes channels.
    packed_first = visual.vision_model.calls[2]
    if frames[0] % 2:
        torch.testing.assert_close(packed_first[-1, :3], packed_first[-1, 3:])
