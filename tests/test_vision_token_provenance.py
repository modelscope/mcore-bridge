# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
from types import SimpleNamespace

from mcore_bridge.model.mm_gpts.utils import HuggingFaceVit


class FakeVisual:
    dtype = torch.float32

    def __call__(self, pixels, grid_thw):
        return pixels


@pytest.mark.parametrize('modality', ['image', 'video'])
@pytest.mark.parametrize('explicit_types', [False, True])
def test_input_vision_embedding_preserves_generated_special_token_gradient(modality, explicit_types):
    config = SimpleNamespace(image_token_id=10, video_token_id=11, vision_config=SimpleNamespace(spatial_merge_size=1))
    special = getattr(config, f'{modality}_token_id')
    ids = torch.tensor([[1, special, special if explicit_types else 2]])
    embeddings = torch.randn(1, 3, 4, requires_grad=True)
    features = torch.randn(1, 4, requires_grad=True)
    inputs = {'input_ids': ids, f'{modality}_grid_thw': torch.tensor([[1, 1, 1]])}
    inputs['pixel_values' if modality == 'image' else 'pixel_values_videos'] = features
    if explicit_types:
        inputs['mm_token_type_ids'] = torch.tensor([[0, 1 if modality == 'image' else 2, 0]])
    result = HuggingFaceVit._hf_get_inputs_embeds(embeddings, inputs, FakeVisual(), config)
    torch.testing.assert_close(result[0, 1], features[0], atol=0, rtol=0)
    torch.testing.assert_close(result[0, 2], embeddings[0, 2], atol=0, rtol=0)
    result.sum().backward()
    torch.testing.assert_close(embeddings.grad, torch.tensor([[[1.] * 4, [0.] * 4, [1.] * 4]]), atol=0, rtol=0)
    torch.testing.assert_close(features.grad, torch.ones_like(features), atol=0, rtol=0)


@pytest.mark.parametrize('failure', ['shape', 'token_id', 'count', 'modality', 'missing_pixels'])
def test_invalid_vision_types_are_rejected(failure):
    config = SimpleNamespace(image_token_id=10, video_token_id=11, vision_config=SimpleNamespace(spatial_merge_size=1))
    inputs = {
        'input_ids': torch.tensor([[1, 10, 2]]),
        'mm_token_type_ids': torch.tensor([[0, 1, 0]]),
        'pixel_values': torch.ones(1, 4),
        'image_grid_thw': torch.tensor([[1, 1, 1]]),
    }
    if failure == 'shape':
        inputs['mm_token_type_ids'] = torch.zeros(1, 2)
    elif failure == 'token_id':
        inputs['mm_token_type_ids'][0, 0] = 1
    elif failure == 'count':
        inputs['pixel_values'] = torch.ones(2, 4)
    elif failure == 'missing_pixels':
        inputs.pop('pixel_values')
    else:
        inputs['mm_token_type_ids'][0, 0] = 3
    with pytest.raises((ValueError, RuntimeError)):
        HuggingFaceVit._hf_get_inputs_embeds(torch.ones(1, 3, 4), inputs, FakeVisual(), config)


def test_image_only_config_does_not_require_video_token_id():
    config = SimpleNamespace(image_token_id=10, vision_config=SimpleNamespace(spatial_merge_size=1))
    inputs = {
        'input_ids': torch.tensor([[10, 10]]),
        'mm_token_type_ids': torch.tensor([[1, 0]]),
        'pixel_values': torch.ones(1, 4),
        'image_grid_thw': torch.tensor([[1, 1, 1]]),
    }
    result = HuggingFaceVit._hf_get_inputs_embeds(torch.zeros(1, 2, 4), inputs, FakeVisual(), config)
    torch.testing.assert_close(result, torch.tensor([[[1.] * 4, [0.] * 4]]), atol=0, rtol=0)
