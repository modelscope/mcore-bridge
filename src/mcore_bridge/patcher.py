import inspect
import peft
import sys
import torch
from megatron.core import mpu
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.extensions.transformer_engine import TEGroupedLinear, TELinear
from megatron.core.models.common.embeddings import rope_utils
from megatron.core.models.common.embeddings.rotary_pos_embedding import MultimodalRotaryEmbedding
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionBlock, get_mtp_layer_offset
from packaging import version
from peft.tuners.tuners_utils import BaseTuner
from torch import nn
from transformers.utils import is_torch_npu_available
from typing import List, Optional, Tuple

from mcore_bridge.utils import get_logger, is_flash_attn_3_available

logger = get_logger()


def _patch_flash_attn():
    # flash_attention_3
    if is_flash_attn_3_available():
        import flash_attn_interface
        sys.modules['flash_attn_3.flash_attn_interface'] = flash_attn_interface


def _patch_transformer_engine():
    import transformer_engine
    try:
        from transformer_engine.pytorch.attention import apply_rotary_pos_emb
    except ImportError:
        try:
            transformer_engine.pytorch.attention.apply_rotary_pos_emb = (
                transformer_engine.pytorch.attention.rope.apply_rotary_pos_emb)
        except (ImportError, AttributeError):
            logger.warning('Failed to patch apply_rotary_pos_emb.')
    try:
        from transformer_engine.pytorch.attention import _SplitAlongDim
    except ImportError:
        try:
            transformer_engine.pytorch.attention._SplitAlongDim = (transformer_engine.pytorch.utils.SplitAlongDim)
        except (ImportError, AttributeError):
            logger.warning('Failed to patch _SplitAlongDim.')


def _patch_peft_BaseTuner():
    _origin_get_tied_target_modules = BaseTuner._get_tied_target_modules

    def _get_tied_target_modules(self, model: nn.Module) -> List[str]:
        try:
            return _origin_get_tied_target_modules(self, model)
        except AttributeError:
            tied_target_modules = []
            if model.share_embeddings_and_output_weights:
                for target_module in self.targeted_module_names:
                    if target_module.split('.')[-1] in ['output_layer', 'embedding']:
                        tied_target_modules.append(target_module)
            return tied_target_modules

    BaseTuner._get_tied_target_modules = _get_tied_target_modules


def _patch_TEGroupedLinear():

    def sharded_state_dict(
            self,
            prefix: str = '',
            sharded_offsets: Tuple[Tuple[int, int, int]] = (),
            metadata: Optional[dict] = None,
    ):
        return self._sharded_state_dict_grouped(None, prefix, sharded_offsets, metadata)

    TEGroupedLinear.sharded_state_dict = sharded_state_dict


def _patch_peft_ModulesToSaveWrapper():
    if version.parse(peft.__version__) >= version.parse('0.16'):
        from peft.utils import other as peft_module
    else:
        from peft.tuners import tuners_utils as peft_module

    from mcore_bridge.tuners.utils import tuners_sharded_state_dict

    OriginModulesToSaveWrapper = peft_module.ModulesToSaveWrapper

    class ModulesToSaveWrapper(OriginModulesToSaveWrapper):

        def sharded_state_dict(
                self,
                prefix: str = '',
                sharded_offsets: Tuple[Tuple[int, int, int]] = (),
                metadata: Optional[dict] = None,
        ) -> ShardedStateDict:
            sharded_state_dict = tuners_sharded_state_dict(self, prefix, sharded_offsets, metadata)
            if prefix in {'output_layer.', 'language_model.output_layer.'}:
                for k in list(sharded_state_dict.keys()):
                    if '_extra_state' in k:
                        # Old GPT checkpoints only stored the output layer weight key. So we remove the
                        # _extra_state key but check that it doesn't contain any data anyway
                        output_extra_state = sharded_state_dict.pop(k, None)
                        assert not (output_extra_state and output_extra_state.data
                                    ), f'Expected output layer extra state to be empty, got: {output_extra_state}'
                # fix error
                if f'{prefix}modules_to_save.default.weight' in sharded_state_dict:
                    sharded_state_dict[f'{prefix}weight'] = sharded_state_dict[
                        f'{prefix}modules_to_save.default.weight']
            return sharded_state_dict

    peft_module.ModulesToSaveWrapper = ModulesToSaveWrapper
    peft_module.OriginModulesToSaveWrapper = OriginModulesToSaveWrapper


def _patch_TELinear():

    def __repr__(self):
        if is_torch_npu_available():
            # MindSpeed 0.15.x changes some TE debug fields to
            # input_size/output_size. Keep this compatibility on the NPU path
            # only so GPU and older versions retain their original field
            # semantics.
            in_features = getattr(self, 'in_features', getattr(self, 'input_size', None))
            out_features = getattr(self, 'out_features', getattr(self, 'output_size', None))
            use_bias = getattr(self, 'use_bias', getattr(self, 'bias', None) is not None)
            tp_size = getattr(self, 'tp_size', None)
            if tp_size is None:
                parallel_mode = getattr(self, 'parallel_mode', None)
                tp_size = 1 if parallel_mode == 'duplicated' else 'unknown'
        else:
            in_features = self.in_features
            out_features = self.out_features
            use_bias = self.use_bias
            tp_size = self.tp_size
        return (f'{type(self).__name__}(in_features={in_features}, '
                f'out_features={out_features}, bias={use_bias}, TP={tp_size})')

    TELinear.__repr__ = __repr__


def _patch_mrope():
    # Code borrowed from huggingface/transformers
    def apply_interleaved_mrope(freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    # Code borrowed from NVIDIA/Megatron-LM
    def forward(self, position_ids, mrope_section: List[int], mrope_interleaved: bool = False) -> torch.Tensor:
        seq = position_ids.to(device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        # shape (3, bs, dim, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].expand(3, seq.shape[1], -1, 1)
        # shape (3, bs, 1, seq_length)
        seq_expanded = seq[:, :, None, :].float()
        # shape (3, bs, seq_length, dim)
        freqs = (inv_freq_expanded @ seq_expanded).transpose(2, 3)
        if mrope_interleaved:
            freqs = apply_interleaved_mrope(freqs, mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            # first part even vector components, second part odd vector components,
            #  2 * dim in dimension size
            if self.rotary_interleaved:
                emb = torch.cat([m[i % 3] for i, m in enumerate(freqs.split(mrope_section, dim=-1))], dim=-1)
                emb = emb.repeat_interleave(2, dim=-1)
            else:
                emb = torch.cat((freqs, freqs), dim=-1)  # shape (3, bs, seq_length, 2 * dim)
                # generate freqs with mrope_section
                # shape (bs, seq_length, 2 * dim)
                mrope_section = mrope_section * 2
                emb = torch.cat([m[i % 3] for i, m in enumerate(emb.split(mrope_section, dim=-1))], dim=-1)

        # shape (seq_length, bs, 1, 2 * dim)
        emb = emb[..., None, :].transpose(0, 1).contiguous()
        return emb

    MultimodalRotaryEmbedding.forward = forward
    _origin_apply_rotary_pos_emb_thd = rope_utils._apply_rotary_pos_emb_thd

    def _apply_rotary_pos_emb_thd(t: torch.Tensor, cu_seqlens: torch.Tensor, freqs: torch.Tensor, *args,
                                  **kwargs) -> torch.Tensor:
        cp_group = kwargs.pop('cp_group', None)
        if cp_group is not None:
            cp_size = cp_group.size()
        else:
            cp_size = mpu.get_context_parallel_world_size()
            cp_group = mpu.get_context_parallel_group()
        cu_seqlens_for_batched = cu_seqlens // cp_size
        use_batched_rope = (freqs.dim() >= 1 and freqs.shape[0] == cu_seqlens_for_batched[-1]).item()
        # The determination of mla_output_remove_interleaving: a quick solution for identifying deepseek_v4
        # (TODO: refactor)
        if not use_batched_rope and not kwargs.get('mla_output_remove_interleaving', False):
            logger.warning_once('Using non-batched RoPE, which may affect performance.')
            return _origin_apply_rotary_pos_emb_thd(t, cu_seqlens, freqs, *args, cp_group=cp_group, **kwargs)

        kwargs.pop('max_seqlen', None)  # compat megatron-lm dev branch
        return rope_utils._apply_rotary_pos_emb_bshd(t.unsqueeze(1), freqs, *args, **kwargs).squeeze(1)

    rope_utils._apply_rotary_pos_emb_thd = _apply_rotary_pos_emb_thd

    origin_apply_rotary_pos_emb = rope_utils.apply_rotary_pos_emb
    has_mla_rotary_interleaved = 'mla_rotary_interleaved' in inspect.signature(origin_apply_rotary_pos_emb).parameters

    def apply_rotary_pos_emb(
        t: torch.Tensor,
        freqs: torch.Tensor,
        config: TransformerConfig,
        cu_seqlens: Optional[torch.Tensor] = None,
        mscale: float = 1.0,
        cp_group: torch.distributed.ProcessGroup = None,
        mla_rotary_interleaved: Optional[bool] = None,
        **kwargs,
    ):
        if has_mla_rotary_interleaved or mla_rotary_interleaved is not None:
            kwargs['mla_rotary_interleaved'] = mla_rotary_interleaved
        return origin_apply_rotary_pos_emb(
            t, freqs, config, cu_seqlens=cu_seqlens, mscale=mscale, cp_group=cp_group, **kwargs)

    rope_utils.apply_rotary_pos_emb = apply_rotary_pos_emb


def _patch_mtp():

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        # get hidden states from previous mtp stages
        get_offset_kwargs = {} if self.vp_stage is None else {'vp_stage': self.vp_stage}
        mtp_decoder_input = decoder_input = kwargs.pop('decoder_input', None)
        mhc_multistream = kwargs.pop('mhc_multistream', None)

        offset = get_mtp_layer_offset(self.config, **get_offset_kwargs)
        assert offset == 0, 'not support offset'
        hidden_states_list = list(torch.chunk(hidden_states, 1 + offset, dim=0))
        if mhc_multistream is not None:
            # mHC mode: use multi-stream for MTP depth input, contracted for loss list.
            mhc_chunks = list(torch.chunk(mhc_multistream, 1 + offset, dim=0))
            hidden_states = mhc_chunks[offset]
        else:
            hidden_states = hidden_states_list[offset]
        for layer_number in range(self.config.mtp_unroll_steps):
            layer = self.layers[layer_number % len(self.layers)]
            (hidden_states, input_ids, position_ids, decoder_input) = layer(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
                layer_number=layer_number + 1,
                **kwargs,
            )
            if mtp_decoder_input is None:
                decoder_input = None

            if mhc_multistream is not None:
                mhc_chunks.append(hidden_states)
                hidden_states_list.append(layer._postprocess(hidden_states))
            else:
                # append the output hidden states of the current mtp layer
                # to the hidden_states_list
                hidden_states_list.append(hidden_states)

        # concat the hidden states of all mtp layers
        hidden_states = torch.cat(hidden_states_list, dim=0)
        return hidden_states

    MultiTokenPredictionBlock.forward = forward


def apply_patch():
    _patch_flash_attn()
    _patch_transformer_engine()
    # patch peft
    try:
        _patch_peft_BaseTuner()
        _patch_peft_ModulesToSaveWrapper()
    except Exception:
        logger.warning('Failed to patch peft.')
    # patch module
    _patch_TEGroupedLinear()
    _patch_TELinear()
    _patch_mrope()
    _patch_mtp()
    from mcore_bridge import tuners  # apply patch
