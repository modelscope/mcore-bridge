# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from importlib import import_module
from types import SimpleNamespace
from typing import Optional

from mcore_bridge.bridge import MultimodalGPTBridge
from mcore_bridge.utils import deep_getattr

from ..constant import ModelType
from ..gpts.deepseek_v4 import DeepseekV4Bridge, DeepseekV4GPTModel, DeepseekV4Loader
from ..mm_gpt_model import MultimodalGPTModel
from ..register import ModelMeta, register_model
from .utils import HuggingFaceVit

# Image sentinel token types (must match encoding_dsv4.py / image_processor.py)
IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)


@contextmanager
def _set_default_dtype(dtype):
    """Temporarily set torch default dtype for module construction."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


class DeepseekV4VisionVit(HuggingFaceVit):
    module_mapping = {'model.vision': 'vision', 'model.aligner': 'aligner'}
    _vision_tower = ['vision']
    _aligner = ['aligner']

    def prepare_model(self, hf_config):
        """Instantiate ViT, Aligner, and image special-token params from the HF config."""
        model_dir = hf_config.name_or_path
        inference_dir = os.path.join(model_dir, 'inference')
        if inference_dir not in sys.path:
            sys.path.insert(0, inference_dir)

        vision_mod = import_module('vision')
        ViT = vision_mod.ViT
        Aligner = vision_mod.Aligner

        args = SimpleNamespace(
            vision_patch_size=hf_config.vision_patch_size,
            vision_dim=hf_config.vision_dim,
            vision_n_heads=hf_config.vision_n_heads,
            vision_inter_dim=hf_config.vision_inter_dim,
            vision_n_layers=hf_config.vision_n_layers,
            vision_rope_theta=hf_config.vision_rope_theta,
            vision_downsample_ratio=hf_config.vision_downsample_ratio,
            dim=hf_config.hidden_size,
        )

        dtype = self.config.params_dtype
        with _set_default_dtype(dtype):
            self.vision = ViT(args)
            self.aligner = Aligner(args)

        _orig_gcs = vision_mod.get_vision_cos_sin

        def _gcs(n_h, n_w, dim, theta):
            cos, sin = _orig_gcs(n_h, n_w, dim, theta)
            device = next(self.vision.parameters()).device
            return cos.to(device), sin.to(device)

        vision_mod.get_vision_cos_sin = _gcs

        dim = hf_config.hidden_size
        self.image_start = nn.Parameter(torch.empty(dim, dtype=dtype))
        self.image_end = nn.Parameter(torch.empty(dim, dtype=dtype))
        self.image_pad = nn.Parameter(torch.empty(dim, dtype=dtype))
        self.image_newline = nn.Parameter(torch.empty(dim, dtype=dtype))

        self.dtype = dtype

    def mask_input_ids(self, input_ids):
        """Clamp sentinel token IDs (>= vocab_size) to 0 before embedding lookup."""
        vocab_size = self.hf_config.vocab_size
        return torch.masked_fill(input_ids, input_ids >= vocab_size, 0)

    def _encode_image(self, patches, n_vit_h, n_vit_w):
        """Run ViT + Aligner to produce LLM-space image embeddings."""
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        image_inputs = kwargs.get('image_inputs')
        if not image_inputs:
            return inputs_embeds

        input_ids = kwargs['input_ids']
        vocab_size = self.hf_config.vocab_size

        # Flatten all ImageInput objects across batch samples (order preserved).
        all_images = []
        for sample in image_inputs:
            if sample is not None:
                all_images.extend(sample)
        if not all_images:
            return inputs_embeds

        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        # Special-token embedding lookup: index by sentinel type id.
        params = torch.stack([
            self.image_start,
            self.image_pad,
            self.image_pad,
            self.image_newline,
            self.image_end,
        ]).to(
            device=device, dtype=dtype)

        bsz = input_ids.shape[0] if input_ids.ndim > 1 else 1

        img_idx = 0
        for b in range(bsz):
            row_ids = input_ids[b] if input_ids.ndim > 1 else input_ids

            # Find every IMAGE_START sentinel (vocab_size + IMAGE_START).
            start_mask = row_ids == vocab_size + IMAGE_START
            start_positions = start_mask.nonzero(as_tuple=False).squeeze(-1)

            for start_pos in start_positions:
                if img_idx >= len(all_images):
                    break
                img = all_images[img_idx]
                img_idx += 1

                sp = start_pos.item()
                # Block spans from IMAGE_START through IMAGE_END (inclusive).
                ep = sp
                row_len = row_ids.shape[0]
                while ep < row_len and row_ids[ep].item() >= vocab_size:
                    ep += 1
                if ep == sp:
                    continue

                # Types are encoded in the sentinel token ids themselves.
                types = (row_ids[sp:ep] - vocab_size).to(torch.int64)

                embeds = self._encode_image(
                    img.patches.to(device=device, dtype=self.dtype),
                    img.n_vit_h,
                    img.n_vit_w,
                )[img.perm.to(device)].to(dtype)

                block = params[types].clone()
                image_slots = types == IMAGE
                block[image_slots] = embeds

                # Write block into inputs_embeds ([b, s, h] convention).
                if inputs_embeds.ndim > 2:
                    inputs_embeds[b, sp:ep, :] = block
                else:
                    inputs_embeds[sp:ep] = block

        return inputs_embeds


class DeepseekV4VisionGPTModel(MultimodalGPTModel):
    """MultimodalGPTModel that uses DeepseekV4GPTModel as the language model."""
    language_model_cls = DeepseekV4GPTModel


class DeepseekV4VisionLoader(DeepseekV4Loader):
    """Loader for DeepSeek-V4-Flash-Vision: reuses V4 layer spec, swaps in the MM model."""
    model_cls = DeepseekV4VisionGPTModel

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        # The V4 hybrid attention spec is identical for text and vision variants.
        return DeepseekV4Loader.get_transformer_layer_spec(self, vp_stage)


class DeepseekV4VisionBridge(DeepseekV4Bridge, MultimodalGPTBridge):
    hf_layers_prefix = 'model.layers'
    hf_final_layernorm_key = 'model.norm.weight'

    def _convert_pre_process(self, mg_model, hf_state_dict, hf_prefix: str, to_mcore: bool):
        """Override to also load image_start/end/pad/newline bare parameters."""
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        self._set_word_embeddings(mg_model, hf_state_dict, to_mcore)
        if self.is_multimodal and not self.config.language_model_only:
            for prefix, mg_prefix in self.module_mapping.items():
                mg_module = deep_getattr(mg_model, f'visual.{mg_prefix}')
                hf_state_dict.update(self._set_module(mg_module, hf_state_dict, f'{hf_prefix}{prefix}.', to_mcore))
            # Load the four image special-token embeddings onto the visual module.
            visual = mg_model.visual
            for key in ['image_start', 'image_end', 'image_pad', 'image_newline']:
                hf_key = f'{hf_prefix}model.{key}'
                self._set_state_dict(visual, key, hf_state_dict, hf_key, to_mcore)
        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict


register_model(
    ModelMeta(
        ModelType.deepseek_v4_flash_vision,
        ['deepseek_v4_flash_vision'],
        bridge_cls=DeepseekV4VisionBridge,
        visual_cls=DeepseekV4VisionVit,
        loader=DeepseekV4VisionLoader,
    ))
