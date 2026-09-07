# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from megatron.core import parallel_state

from mcore_bridge.bridge import MultimodalGPTBridge
from mcore_bridge.utils import split_cp_inputs

from ..constant import ModelType
from ..modules import TransformerBlock
from ..register import ModelLoader, ModelMeta, register_model
from .utils import HuggingFaceVit


class Qwen3VLTransformerBlock(TransformerBlock):

    def _layer_forward(self, layer, hidden_states, **kwargs):
        deepstack_visual_embeds = kwargs.pop('deepstack_visual_embeds', None)
        visual_pos_masks = kwargs.pop('visual_pos_masks', None)
        hidden_states, context = super()._layer_forward(layer, hidden_states, **kwargs)
        layer_number = layer.layer_number - 1
        if deepstack_visual_embeds is not None and layer_number in range(len(deepstack_visual_embeds)):
            hidden_states = self._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_number],
            )
        return hidden_states, context

    def forward(self, *args, **kwargs):
        deepstack_visual_embeds = kwargs.get('deepstack_visual_embeds')
        if deepstack_visual_embeds is not None:
            assert len(deepstack_visual_embeds) <= len(
                self.layers), (f'len(deepstack_visual_embeds): {len(deepstack_visual_embeds)}, '
                               f'len(self.layers): {len(self.layers)}.')
        return super().forward(*args, **kwargs)

    def _deepstack_process(self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor,
                           visual_embeds: torch.Tensor):
        if visual_pos_masks is None:
            return hidden_states + visual_embeds.mean() * 0
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states


class Qwen3VL_Vit(HuggingFaceVit):
    module_mapping = {'model.visual': 'visual'}
    _vision_tower = ['visual']
    _aligner = ['visual.merger', 'visual.deepstack_merger_list']

    def prepare_model(self, hf_config):
        hf_model_type = self.config.hf_model_type
        if hf_model_type == 'qwen3_vl':
            from transformers.models.qwen3_vl import Qwen3VLVisionModel as VisionModel
        elif hf_model_type == 'qwen3_vl_moe':
            from transformers.models.qwen3_vl_moe import Qwen3VLMoeVisionModel as VisionModel
        self.visual = VisionModel._from_config(hf_config.vision_config)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        return self._get_inputs_embeds(inputs_embeds, kwargs, self.visual, self.hf_config)

    def _get_inputs_embeds(self, inputs_embeds, inputs, visual, hf_config):
        input_ids = inputs['input_ids']
        packed_seq_params = inputs.get('packed_seq_params')
        pixel_values = inputs.get('pixel_values')
        pixel_values_videos = inputs.get('pixel_values_videos')
        image_grid_thw = inputs.get('image_grid_thw')
        video_grid_thw = inputs.get('video_grid_thw')
        dtype = visual.dtype
        vision_config = HuggingFaceVit._get_vision_config(hf_config)
        if pixel_values is None and pixel_values_videos is None:  # plain-text
            hidden_size = vision_config.in_channels * vision_config.temporal_patch_size * vision_config.patch_size**2
            pixel_values = torch.zeros(16 * 16, hidden_size, dtype=dtype, device=input_ids.device)
            image_grid_thw = input_ids.new_tensor([[1, 16, 16]])
            visual_res = visual(pixel_values, grid_thw=image_grid_thw)
            if hasattr(visual_res, 'pooler_output'):
                image_embeds = visual_res.pooler_output
                deepstack_visual_embeds = visual_res.deepstack_features
            else:
                image_embeds, deepstack_visual_embeds = visual_res
            deepstack_visual_embeds = (torch.stack(deepstack_visual_embeds, dim=0) if deepstack_visual_embeds else None)
            inputs_embeds = inputs_embeds + image_embeds.mean().to(device=inputs_embeds.device) * 0.
            visual_pos_masks = None
        else:
            if pixel_values is None:
                pixel_values_mixed = pixel_values_videos
                grid_thw = video_grid_thw
            elif pixel_values_videos is None:
                pixel_values_mixed = pixel_values
                grid_thw = image_grid_thw
            else:
                pixel_values_mixed = torch.concat([pixel_values, pixel_values_videos], dim=0)
                grid_thw = torch.concat([image_grid_thw, video_grid_thw], dim=0)
            pixel_values_mixed = pixel_values_mixed.type(dtype)
            visual_res = visual(pixel_values_mixed, grid_thw=grid_thw)
            if hasattr(visual_res, 'pooler_output'):
                mixed_embeds = visual_res.pooler_output
                deepstack_visual_embeds = visual_res.deepstack_features
            else:
                mixed_embeds, deepstack_visual_embeds = visual_res
            if pixel_values is None:
                image_embeds = None
                video_embeds = mixed_embeds
            elif pixel_values_videos is None:
                image_embeds = mixed_embeds
                video_embeds = None
            else:
                merge_length = vision_config.spatial_merge_size**2
                image_tokens = (image_grid_thw.prod(dim=-1) // merge_length).sum()
                image_embeds = mixed_embeds[:image_tokens]
                video_embeds = mixed_embeds[image_tokens:]

            image_mask = (input_ids == hf_config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            video_mask = (input_ids == hf_config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            if image_embeds is not None:
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask = image_mask.to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if video_embeds is not None:
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                video_mask = video_mask.to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            image_mask, video_mask = image_mask[..., 0], video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            if image_embeds is not None and video_embeds is not None and deepstack_visual_embeds:
                deepstack_image_embeds = [tensor[:image_tokens] for tensor in deepstack_visual_embeds]
                deepstack_video_embeds = [tensor[image_tokens:] for tensor in deepstack_visual_embeds]
                deepstack_visual_embeds = []
                image_mask_joint = image_mask[visual_pos_masks]
                video_mask_joint = video_mask[visual_pos_masks]
                for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                    embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                    embed_joint[image_mask_joint, :] = img_embed
                    embed_joint[video_mask_joint, :] = vid_embed
                    deepstack_visual_embeds.append(embed_joint)

            deepstack_visual_embeds = (torch.stack(deepstack_visual_embeds, dim=0) if deepstack_visual_embeds else None)
            visual_pos_masks = visual_pos_masks.transpose(0, 1)
            # compat cp
            if self.config.context_parallel_size > 1:
                device = visual_pos_masks.device
                cp_mask = torch.full(visual_pos_masks.shape[:1], -1, dtype=torch.long, device=device)
                cp_mask[visual_pos_masks[:, 0]] = torch.arange(visual_pos_masks.sum(), device=device)
                cu_seqlens = getattr(packed_seq_params, 'cu_seqlens_q', None)
                cp_mask = split_cp_inputs(cp_mask, cu_seqlens, 0)
                visual_pos_masks = split_cp_inputs(visual_pos_masks, cu_seqlens, 0)
                if deepstack_visual_embeds is not None:
                    deepstack_visual_embeds = deepstack_visual_embeds[:, cp_mask[(cp_mask != -1)]]
            # compat sp
            tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            if self.config.sequence_parallel and tp_world_size > 1:
                visual_pos_masks = visual_pos_masks.view(tp_world_size, -1, *visual_pos_masks.shape[1:])
                mask_tokens = visual_pos_masks.sum(dim=(1, 2)).tolist()
                visual_start = 0 if tp_rank == 0 else sum(mask_tokens[:tp_rank])
                visual_end = visual_start + mask_tokens[tp_rank]
                visual_pos_masks = visual_pos_masks[tp_rank]
                if deepstack_visual_embeds is not None:
                    deepstack_visual_embeds = deepstack_visual_embeds[:, visual_start:visual_end]
        return {
            'inputs_embeds': inputs_embeds,
            'visual_pos_masks': visual_pos_masks,
            'deepstack_visual_embeds': deepstack_visual_embeds
        }


class Qwen3VLLoader(ModelLoader):
    transformer_block = Qwen3VLTransformerBlock


register_model(
    ModelMeta(
        ModelType.qwen3_vl,
        ['qwen3_vl', 'qwen3_vl_moe'],
        bridge_cls=MultimodalGPTBridge,
        visual_cls=Qwen3VL_Vit,
        loader=Qwen3VLLoader,
    ))
