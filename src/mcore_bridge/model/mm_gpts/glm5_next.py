# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import torch
import torch.nn as nn
from copy import deepcopy
from transformers import PretrainedConfig
from typing import Optional

from mcore_bridge.bridge import MultimodalGPTBridge
from mcore_bridge.tuners import LoraParallelLinear
from mcore_bridge.utils import get_logger

from ..constant import ModelType
from ..mm_gpt_model import MultimodalGPTModel
from ..register import ModelLoader, ModelMeta, register_model
from .utils import HuggingFaceVit

try:
    from megatron.core.models.hybrid.hybrid_block import HybridStack
    from megatron.core.models.hybrid.hybrid_layer_allocation import parse_hybrid_pattern, select_pipeline_segment
    from megatron.core.transformer.module import mark_keep_in_fp32
    from megatron.core.transformer.multi_token_prediction import \
        MultiTokenPredictionLayer as McoreMultiTokenPredictionLayer

    from ..hybrid_model import HybridModel
except ImportError:
    # Released Megatron has no hybrid stack for this model (0.18 has no mark_keep_in_fp32, 0.16 no
    # models.hybrid). The package must still import for every other model, so fall back to a base
    # that lets the class definitions below succeed; a GLM config is rejected before any of this is
    # used, by require_glm5_hybrid(), which names the missing dev patch.
    HybridModel = HybridStack = McoreMultiTokenPredictionLayer = object
    parse_hybrid_pattern = select_pipeline_segment = mark_keep_in_fp32 = None

logger = get_logger()

# Probes for require_glm5_hybrid(): the hybrid-stack APIs this model is built on, and the
# TransformerConfig fields that only the packaged patch provides.
_HYBRID_API_PROBES = (('megatron.core.models.hybrid.hybrid_model', 'HybridModel'),
                      ('megatron.core.models.hybrid.hybrid_layer_allocation',
                       'select_pipeline_segment'), ('megatron.core.transformer.module', 'mark_keep_in_fp32'))
_PATCH_CONFIG_FIELDS = ('kda_two_stage_gates', 'mhc_norm_eps_inside_sqrt', 'mhc_keep_mappings_in_fp32',
                        'mhc_learned_output_contract', 'dsa_indexer_kpool', 'dsa_indexer_kpool_always_select_tail')


def require_glm5_hybrid():
    """Reject a Megatron that cannot run GLM-5.3, naming the fix rather than just the missing symbol.

    The two causes need different actions: a Megatron that predates the hybrid stack (update it) and
    a Megatron without the patch this package ships (apply it).
    """
    import importlib
    missing_api = []
    for module, symbol in _HYBRID_API_PROBES:
        try:
            found = hasattr(importlib.import_module(module), symbol)
        except ImportError:
            found = False
        if not found:
            missing_api.append(f'{module}.{symbol}')
    if missing_api:
        raise ImportError(f'GLM-5.3 requires a recent Megatron-LM dev, which is missing {missing_api}. '
                          'Update Megatron with:\n'
                          '  pip install -U git+https://github.com/NVIDIA/Megatron-LM.git@dev')

    from megatron.core.transformer.transformer_config import TransformerConfig
    missing_fields = sorted(set(_PATCH_CONFIG_FIELDS) - TransformerConfig.__dataclass_fields__.keys())
    if missing_fields:
        raise ImportError(f'GLM-5.3 requires the Megatron patch shipped with mcore-bridge, which adds '
                          f'{missing_fields}. Apply it with:\n'
                          '  python -m mcore_bridge.tools.apply_megatron_patch')


def glm5_hybrid_layer_mapping(config):
    pattern = config.hybrid_layer_pattern.replace('|', '')
    if len(pattern) != config.num_layers or len(pattern) % 2:
        raise ValueError('The current model needs a hybrid_layer_pattern with two sublayers '
                         '(attention + FFN) per block')
    if any(pattern[i] not in 'KD' or pattern[i + 1] not in '-E' for i in range(0, len(pattern), 2)):
        raise ValueError('The current model needs hybrid_layer_pattern to alternate an attention '
                         'sublayer (K or D) with an FFN sublayer (- or E)')
    return tuple((idx // 2, 'attn' if idx % 2 == 0 else 'ffn', symbol) for idx, symbol in enumerate(pattern))


def _get_physical_cu_seqlens(packed_seq_params):
    if packed_seq_params is None:
        return None
    cu_seqlens = getattr(packed_seq_params, 'cu_seqlens_q_padded', None)
    return packed_seq_params.cu_seqlens_q if cu_seqlens is None else cu_seqlens


class Glm5NextRMSNorm(nn.Module):
    """Mirrors transformers `Glm5NextTextRMSNorm`: the normalized tensor is cast back to the
    input dtype *before* the weight is applied, so the weight multiplies in param dtype."""

    def __init__(self, hidden_size, eps, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(input_dtype)


class Glm5NextHybridRMSNorm(Glm5NextRMSNorm):

    def __init__(self, config, hidden_size, eps):
        super().__init__(hidden_size, eps, dtype=config.params_dtype)
        self.weight.sequence_parallel = config.sequence_parallel


class Glm5NextHybridStack(HybridStack):
    """Outer decoder stack that never hands the pre-contraction multi-stream tensor to MTP.

    GLM-5.3's backbone runs mHC (4 residual streams) but its MTP head is non-mHC: it consumes the
    mean-contracted single stream through a fused ``eh_proj`` (2H -> H), exactly like the HF/vLLM
    draft model. Upstream ``HybridStack.forward`` returns ``(hidden_states, mhc_multistream)`` when
    mHC + MTP are both active, and ``MultiTokenPredictionBlock.forward`` then feeds the multi-stream
    to the MTP layer and re-applies ``_postprocess`` to its output -- which double-norms a non-mHC
    MTP layer whose output is already single-stream and post-final-norm. Dropping the second element
    keeps the block on its single-stream path, where the MTP layer output is used as-is.
    """

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        return out[0] if isinstance(out, tuple) else out


class Glm5NextMultiTokenPredictionLayer(McoreMultiTokenPredictionLayer):
    """GLM-5.3 MTP layer: a non-mHC ``eh_proj`` head over a plain DSA + MoE inner block.

    The backbone sets ``enable_hyper_connections=True``, which would make Megatron build the mHC MTP
    head (per-stream ``e_proj``/``h_proj`` plus an ``hc_head_*`` learned contraction) and wrap the
    inner block's sublayers in hyper-connections. GLM's MTP checkpoint has none of those: it carries
    a fused ``eh_proj``, plain ``enorm``/``hnorm``, a ``shared_head.norm`` final norm, and an inner
    block with standard residuals. So the MTP subtree is built with hyper-connections disabled via a
    shallow config copy that flips only that flag; the backbone keeps the real config. ``mhc_enabled``
    (read once in ``__init__``) then follows the copy, selecting the ``eh_proj`` branch, and the
    nested ``HybridStack`` skips hyper-connection wrapping for its DSA/MoE sublayers.
    """

    def __init__(self, config, *args, **kwargs):
        mtp_config = copy.copy(config)
        mtp_config.enable_hyper_connections = False
        super().__init__(mtp_config, *args, **kwargs)


class Glm5NextHybridModel(HybridModel):
    extra_forward_keys = ()

    @staticmethod
    def _resolve_hybrid_layer_pattern(config):
        mapping = glm5_hybrid_layer_mapping(config)
        pattern = config.hybrid_layer_pattern
        if config.pipeline_model_parallel_layout is not None:
            raise ValueError('The current model splits pipeline stages by hybrid_layer_pattern, so '
                             'pipeline_model_parallel_layout does not apply to it. For an uneven split use '
                             'num_layers_in_first_pipeline_stage / num_layers_in_last_pipeline_stage.')
        if ('|' in pattern or config.num_layers_in_first_pipeline_stage is not None
                or config.num_layers_in_last_pipeline_stage is not None):
            main = pattern
        else:
            stages = config.pipeline_model_parallel_size
            if config.virtual_pipeline_model_parallel_size is not None:
                stages *= config.virtual_pipeline_model_parallel_size
            blocks, extra = divmod(len(mapping) // 2, stages)
            if blocks == 0:
                raise ValueError('The current model needs at least one attention+FFN block per pipeline stage; '
                                 'lower pipeline_model_parallel_size.')
            segments, offset = [], 0
            for stage in range(stages):
                count = 2 * (blocks + int(stage < extra))
                segments.append(pattern[offset:offset + count])
                offset += count
            main = '|'.join(segments)
        # Append the MTP segment (`<main>/<mtp>/<mtp>/...`, one per prediction depth). Megatron's
        # parse_hybrid_pattern splits on '/' and keeps '|' inside the main pattern for PP boundaries;
        # the MTP inner block (DSA + MoE, no hyper-connections) is described by mtp_hybrid_override_pattern.
        mtp_pattern = getattr(config, 'mtp_hybrid_override_pattern', None)
        mtp_num_layers = getattr(config, 'mtp_num_layers', None)
        if mtp_pattern and mtp_num_layers and '/' not in main:
            main = main + '/' + '/'.join([mtp_pattern] * mtp_num_layers)
        return main

    def __init__(self, config, transformer_layer_spec, pre_process=True, post_process=True, vp_stage=None):
        # The PP boundaries have to be validated before super().__init__: an attention/FFN pair must
        # not be split across two stages.
        from megatron.core.process_groups_config import ProcessGroupCollection
        pg = ProcessGroupCollection.use_mpu_process_groups()
        pattern = self._resolve_hybrid_layer_pattern(config)
        segment, offset = select_pipeline_segment(
            parse_hybrid_pattern(pattern).main_pattern,
            pg.pp,
            vp_stage,
            first_stage_layers=config.num_layers_in_first_pipeline_stage,
            last_stage_layers=config.num_layers_in_last_pipeline_stage,
            tp_group=pg.tp,
            dp_cp_group=pg.dp_cp,
        )
        if offset % 2 or len(segment) % 2:
            raise ValueError(
                f'Pipeline stage boundaries of the current model must fall on complete attention+FFN blocks, '
                f'but this stage starts at sublayer {offset} and holds {len(segment)} sublayers; both must be '
                f'even, since one block is two sublayers. num_layers_in_first_pipeline_stage and '
                f'num_layers_in_last_pipeline_stage count sublayers rather than blocks, so pass even values, '
                f'or leave them unset for an even block-aligned split.')
        super().__init__(config, transformer_layer_spec, pre_process, post_process, vp_stage)
        assert len(self.decoder.layers) == len(segment)
        assert not segment or self.decoder.layers[0].layer_number == offset + 1
        if post_process:
            self.decoder.final_norm = Glm5NextHybridRMSNorm(config, config.hidden_size, config.layernorm_epsilon)
        mapping = glm5_hybrid_layer_mapping(config)
        for layer in self.decoder.layers:
            layer.hyper_connection.sinkhorn_eps = config.hc_eps
            layer.hyper_connection.compute_h_eps = config.hc_eps
            self._finalize_hybrid_sublayer(layer.inner_layer, mapping[layer.layer_number - 1][2], config)
        # The MTP head's inner block is a nested HybridStack of raw (non-hyper-connected) sublayers
        # following mtp_hybrid_override_pattern ('DE'); it needs the same DSA/MoE finalization.
        mtp = getattr(self, 'mtp', None)
        if mtp is not None:
            mtp_pattern = getattr(config, 'mtp_hybrid_override_pattern', '') or ''
            for mtp_layer in mtp.layers:
                for symbol, sublayer in zip(mtp_pattern, mtp_layer.mtp_model_layer.layers):
                    self._finalize_hybrid_sublayer(sublayer, symbol, config)

    @staticmethod
    def _finalize_hybrid_sublayer(sublayer, symbol, config):
        if symbol == 'D':
            # HF runs the KPool indexer under no_grad, so it must not be counted in DDP's
            # grad-ready accounting.
            sublayer.self_attention.core_attention.indexer.requires_grad_(False)
        elif symbol == 'E':
            # Experts reduce over expert-DP, and ETP can differ from attention TP even at EP=1.
            # dev's TEGroupedLinear derives `allreduce` from EP alone and does not mark the ETP
            # shard, which would miss expert grad reduction and norm sharding. Corrected here at
            # the GLM boundary so other models keep their defaults.
            for param in sublayer.mlp.experts.parameters():
                param.allreduce = False
                param.tensor_model_parallel = config.expert_tensor_parallel_size > 1
            router = sublayer.mlp.router
            if router.enable_expert_bias:
                # BF16 cannot accumulate large integer counts exactly, and PP changes the number
                # of micro-batches, which would change the router update.
                mark_keep_in_fp32(router.local_tokens_per_expert)
                mark_keep_in_fp32(router.expert_bias)

    def _get_packed_padding_mask(self, packed_seq_params, position_ids):
        # `seq_lens` is the logical length attached by swift's prepare_batch; it is not one of the
        # required upstream fields.
        lengths = getattr(packed_seq_params, 'seq_lens', None)
        if lengths is None:
            return None
        cu = _get_physical_cu_seqlens(packed_seq_params)
        lengths = lengths.to(device=cu.device, dtype=cu.dtype)
        assert 0 < lengths.numel() <= cu.numel() - 1
        cp_size = self.pg_collection.cp.size()
        # Under CP the data pipeline shards position_ids/labels (zigzag) but passes cu_seqlens and
        # seq_lens FULL, so build the mask over the full physical sequence and then shard it with the
        # same zigzag, keeping it token-aligned with the local position_ids. At cp==1 the physical
        # length is just the (unsharded) position count.
        total = int(cu[-1])
        if cp_size == 1:
            assert total == position_ids.shape[-1]
        # The remaining spans are alignment dummy sequences with zero logical length; they must not
        # update the expert bias.
        lengths = torch.nn.functional.pad(lengths, (0, cu.numel() - 1 - lengths.numel()))
        positions = torch.arange(total, device=cu.device)
        segment = torch.bucketize(positions, cu[1:], right=True)
        mask = (positions - cu[segment] >= lengths[segment])[None]
        if cp_size > 1:
            from mcore_bridge.utils import split_cp_inputs
            mask = split_cp_inputs(mask, cu, 1)
        if self.config.sequence_parallel:
            assert mask.shape[1] % self.pg_collection.tp.size() == 0, f'padding_mask.shape: {mask.shape}'
            mask = mask.chunk(self.pg_collection.tp.size(), dim=1)[self.pg_collection.tp.rank()]
        return mask.contiguous()

    def forward(self,
                input_ids,
                position_ids,
                attention_mask=None,
                *args,
                extra_block_kwargs=None,
                inference_params=None,
                inference_context=None,
                packed_seq_params=None,
                padding_mask=None,
                **kwargs):
        if inference_params is not None or inference_context is not None:
            raise NotImplementedError('The current model supports no inference/KV-cache path; use the training forward')
        if extra_block_kwargs:
            raise ValueError(f'Unsupported block inputs for the current model: {sorted(extra_block_kwargs)}')
        if packed_seq_params is not None and padding_mask is None:
            padding_mask = self._get_packed_padding_mask(packed_seq_params, position_ids)
        return super().forward(
            input_ids,
            position_ids,
            attention_mask,
            *args,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
            **kwargs)


class Glm5NextMultimodalHybridModel(MultimodalGPTModel):
    language_model_cls = Glm5NextHybridModel


class Glm5NextVit(HuggingFaceVit):
    module_mapping = {'model.visual': 'visual'}
    _vision_tower = ['visual']
    _aligner = ['visual.merger']

    def prepare_model(self, hf_config: PretrainedConfig):
        from transformers.models.glm5_next import Glm5NextVisionModel
        self.visual = Glm5NextVisionModel._from_config(hf_config.vision_config)

    def prepare_language_model(self, hf_config: PretrainedConfig):
        self.visual = None

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        return self._hf_get_inputs_embeds(inputs_embeds, kwargs, self.visual, self.hf_config)


class Glm5NextBridge(MultimodalGPTBridge):
    """HF <-> MCore weight mapping for GLM-5.3-Flash, the family's only registered type.

    `MultimodalGPTBridge` supplies the composite checkpoint's `model.language_model.*` prefixes and
    keeps `visual.*`; training without the vision tower is `language_model_only`, not a second bridge.

    `_set_layer_attn` is overridden wholesale rather than reusing `_set_mla_attn_state`,
    because the two attention flavours need different mappings (KDA vs NoPE MLA) and the DSA
    indexer is NoPE: routing it through the generic path would apply the DeepSeek-V3
    rope-interleave half-swap that `dsa_indexer_rotary_interleaved` turns on.
    """

    additional_dim0_keys = {
        'q_proj', 'k_proj', 'v_proj', 'beta_proj', 'f_b_proj', 'g_b_proj', 'linear_q_up_proj', 'linear_kv_up_proj'
    }
    additional_dim1_keys = {'o_proj', 'linear_proj'}

    def _set_kda_qkv_lora(self, mg_attn, hf_state_dict, to_mcore: bool):
        """Map fused KDA adapters with shared A and rank-local [Q, K, V] rows in B."""
        proj = None if mg_attn is None else mg_attn.in_proj
        is_lora = self._reduce_tensor_pp_group(isinstance(proj, LoraParallelLinear), to_mcore)
        if not is_lora:
            return
        names = ('q_proj', 'k_proj', 'v_proj')
        a_key = f'in_proj.lora_A.{self._adapter_name}.weight'
        if to_mcore:
            lora_a = [hf_state_dict[f'{name}.lora_A.weight'].load() for name in names]
            if not all(torch.equal(lora_a[0], part) for part in lora_a[1:]):
                raise ValueError('Fused KDA QKV requires identical q_proj/k_proj/v_proj LoRA A weights')
            # Each TP shard stores local Q, then local K, then local V.
            parts = [
                self._split_tp(hf_state_dict[f'{name}.lora_B.weight'].load(), 0, False, is_embedding=False)
                for name in names
            ]
            self._set_weight(proj.lora_A[self._adapter_name].weight, lora_a[0], a_key)
            self._set_weight(proj.lora_B[self._adapter_name].weight, torch.cat(parts, dim=0), None)
        else:
            a = None if proj is None else proj.lora_A[self._adapter_name].weight.data
            a, _ = self._get_weight(a, a_key)
            # Split before gathering: gathering fused B would interleave ranks with Q/K/V.
            parts = (None, ) * 3 if proj is None else proj.lora_B[self._adapter_name].weight.data.chunk(3, dim=0)
            for name, part in zip(names, parts):
                b, _ = self._get_weight(part, f'{name}.lora_B.{self._adapter_name}.weight')
                if a is not None:
                    self._peft_target_modules.add(name)
                    hf_state_dict[f'{name}.lora_A.weight'] = a.clone()
                    hf_state_dict[f'{name}.lora_B.weight'] = b.clone()

    def _set_kda_state(self, mg_attn, hf_state_dict, to_mcore):
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, 'self_attn.')
        else:
            hf_state_dict = {}
        mappings = [('beta_proj.weight', 'b_proj.weight'), ('out_norm.weight', 'o_norm.weight'),
                    ('out_proj.weight', 'o_proj.weight')]
        mappings += [(key, key) for key in ('f_a_proj.weight', 'f_b_proj.weight', 'g_a_proj.weight', 'g_b_proj.weight',
                                            'A_log', 'dt_bias')]
        for mg_key, hf_key in mappings:
            self._set_state_dict(mg_attn, mg_key, hf_state_dict, hf_key, to_mcore)
        if self._peft_format:
            self._set_kda_qkv_lora(mg_attn, hf_state_dict, to_mcore)
            # Frozen QKV/conv weights are absent from an unmerged adapter checkpoint.
            return {} if to_mcore else self._add_prefix(hf_state_dict, 'self_attn.')
        for mg_name, hf_names in (
            ('in_proj', ('q_proj.weight', 'k_proj.weight', 'v_proj.weight')),
            ('conv1d', ('q_conv1d.weight', 'k_conv1d.weight', 'v_conv1d.weight')),
        ):
            param = None if mg_attn is None else getattr(mg_attn, mg_name).weight
            if to_mcore and param is not None:
                parts = [self._split_tp(hf_state_dict[key].load(), 0, False, is_embedding=False) for key in hf_names]
                self._set_weight(param, torch.cat(parts, dim=0), None)
            elif not to_mcore:
                parts = (None, ) * 3 if param is None else param.data.chunk(3, dim=0)
                for key, part in zip(hf_names, parts):
                    weight, _ = self._get_weight(part, 'q_proj.weight')
                    if weight is not None:
                        hf_state_dict[key] = weight
        if to_mcore:
            return {}
        return self._add_prefix(hf_state_dict, 'self_attn.')

    def _set_dsa_state(self, mg_attn, hf_state_dict, to_mcore):
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, 'self_attn.')
        else:
            hf_state_dict = {}
        mappings = (
            ('linear_q_down_proj.weight', 'q_a_proj.weight'),
            ('q_layernorm.weight', 'q_a_layernorm.weight'),
            ('linear_q_up_proj.weight', 'q_b_proj.weight'),
            ('linear_kv_down_proj.weight', 'kv_a_proj_with_mqa.weight'),
            ('kv_layernorm.weight', 'kv_a_layernorm.weight'),
            ('linear_kv_up_proj.weight', 'kv_b_proj.weight'),
            ('linear_proj.weight', 'o_proj.weight'),
        )
        for mg_key, hf_key in mappings:
            self._set_state_dict(mg_attn, mg_key, hf_state_dict, hf_key, to_mcore)
        indexer = None if mg_attn is None else mg_attn.core_attention.indexer
        hf_state_dict.update(self._set_indexer(indexer, hf_state_dict, 'indexer.', to_mcore))
        for mg_key, hf_key in (('index_kpool_compress_ape', 'indexer.index_kpool_compress_ape'),
                               ('index_kpool_compress_gate', 'indexer.index_kpool_compress_gate')):
            self._set_state_dict(indexer, mg_key, hf_state_dict, hf_key, to_mcore)
        if to_mcore:
            return {}
        return self._add_prefix(hf_state_dict, 'self_attn.')

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        mg_attn = None if mg_layer is None else mg_layer.self_attention
        if glm5_hybrid_layer_mapping(self.config)[layer_idx][2] == 'K':
            result = self._set_kda_state(mg_attn, hf_state_dict, to_mcore)
        else:
            result = self._set_dsa_state(mg_attn, hf_state_dict, to_mcore)
        self._set_state_dict(mg_layer, 'input_layernorm.weight', hf_state_dict, self.hf_input_layernorm_key, to_mcore)
        return result

    def _set_final_layernorm(self, lm_model, hf_state_dict, to_mcore):
        self._set_state_dict(lm_model, 'decoder.final_norm.weight', hf_state_dict, self.hf_final_layernorm_key,
                             to_mcore)

    def _set_layer_state(self, mg_layer, hf_state_dict, hf_prefix: str, layer_idx: int, to_mcore: bool):
        hf_idx, branch, symbol = glm5_hybrid_layer_mapping(self.config)[layer_idx]
        layer_prefix = f'{hf_prefix}{hf_idx}.'
        hf_state_dict = self._remove_prefix(hf_state_dict, layer_prefix) if to_mcore else {}
        inner = None if mg_layer is None else mg_layer.inner_layer
        if branch == 'attn':
            hf_state_dict.update(self._set_layer_attn(inner, hf_state_dict, layer_idx, to_mcore))
        else:
            mlp = None if inner is None else inner.mlp
            setter = self._set_moe_state if symbol == 'E' else self._set_mlp_state
            hf_state_dict.update(setter(mlp, hf_state_dict, f'{self.hf_mlp_prefix}.', hf_idx, to_mcore))
            self._set_state_dict(inner, 'pre_mlp_layernorm.weight', hf_state_dict, self.hf_post_attention_layernorm_key,
                                 to_mcore)
        hc = None if mg_layer is None else mg_layer.hyper_connection
        self._set_state_dict(hc, 'mapping_proj.weight', hf_state_dict, f'hc_{branch}_fn', to_mcore)
        self._set_state_dict(hc, 'bias', hf_state_dict, f'hc_{branch}_base', to_mcore)
        if not to_mcore and f'hc_{branch}_fn' in hf_state_dict:
            hf_state_dict[f'hc_{branch}_fn'] = hf_state_dict[f'hc_{branch}_fn'].to(self.config.params_dtype)
        if not self._peft_format and to_mcore and hc is not None:
            alpha = hf_state_dict[f'hc_{branch}_scale'].load()
            for idx, name in enumerate(('alpha_pre', 'alpha_post', 'alpha_res')):
                self._set_weight(getattr(hc, name), alpha[idx:idx + 1], None)
        elif not self._peft_format and not to_mcore:
            alpha = None if hc is None else torch.cat([hc.alpha_pre, hc.alpha_post, hc.alpha_res])
            alpha = self._get_weight(alpha, None)[0]
            if alpha is not None:
                hf_state_dict[f'hc_{branch}_scale'] = alpha
        return {} if to_mcore else self._add_prefix(hf_state_dict, layer_prefix)

    def _filter_mtp_layer(self, hf_state_dict):
        """Drop the extra MTP-only decoder layer (index == num_hidden_layers) from a HF checkpoint.

        Only used when MTP training is OFF: the checkpoint always ships the MTP head, but a run with
        ``mtp_num_layers=0`` builds no MTP module, so those tensors would otherwise be reported as
        unexpected. When MTP is ON they are consumed by ``_convert_mtp_layer`` instead.
        """
        hf_num_layers = self.config.num_layers // 2
        layer_prefix = f'{self.hf_layers_prefix}.{hf_num_layers}.'
        prefixes = (layer_prefix, layer_prefix.removeprefix('model.'), f'layers.{hf_num_layers}.')
        ignored = [key for key in hf_state_dict if key.startswith(prefixes)]
        if ignored:
            logger.warning_once(
                f'Ignoring {len(ignored)} MTP tensors under decoder layer {hf_num_layers}: this run builds no '
                'MTP module (mtp_num_layers=0). Pass --mtp_num_layers 1 to train the MTP head.')
            hf_state_dict = {key: value for key, value in hf_state_dict.items() if not key.startswith(prefixes)}
        return hf_state_dict

    def _convert_hf_state_dict(self, hf_state_dict, to_mcore):
        if to_mcore and not getattr(self.config, 'mtp_num_layers', None):
            hf_state_dict = self._filter_mtp_layer(hf_state_dict)
        return super()._convert_hf_state_dict(hf_state_dict, to_mcore)

    def _convert_mtp_layer(self, lm_model, hf_state_dict, hf_prefix: str, layer_idx: int, to_mcore: bool):
        """Map GLM's MTP head, which lives at decoder-layer index ``num_hidden_layers``.

        ``num_layers`` counts hybrid *sublayers* (two per HF block), so the MTP head's HF index is
        ``num_layers // 2 + layer_idx`` and it shares the backbone's ``model.language_model.layers``
        prefix -- neither the separate-prefix nor the ``+num_layers`` assumptions of the base
        ``_convert_mtp_layer`` hold. The head is non-mHC: ``enorm``/``hnorm``/fused ``eh_proj`` plus a
        ``shared_head.norm`` final norm, over a nested HybridStack whose raw sublayers are a DSA
        attention block (``input_layernorm`` + ``self_attn``) and a MoE FFN block
        (``pre_mlp_layernorm`` + ``mlp``), with no hyper-connection weights.
        """
        mtp_layer = lm_model.mtp.layers[layer_idx] if hasattr(lm_model, 'mtp') else None
        hf_layer_idx = self.config.num_layers // 2 + layer_idx
        layer_prefix = f'{self.hf_layers_prefix}.{hf_layer_idx}.'
        if to_mcore:
            origin_hf_state_dict = hf_state_dict
            hf_state_dict = self._remove_prefix(hf_state_dict, layer_prefix)
            if len(hf_state_dict) == 0:
                logger.info(f'MTP layer {hf_layer_idx} safetensors weights not found, '
                            'this part will be randomly initialized.')
                for param in mtp_layer.parameters():
                    if param.ndim == 2:
                        mtp_layer.config.init_method(param.data)
                return {}
        else:
            origin_hf_state_dict = {}
            hf_state_dict = {}
        # MTP head: enorm / hnorm / fused eh_proj (2H -> H) / final_layernorm (shared_head.norm).
        for mg_key, hf_key in [('enorm.weight', 'enorm.weight'), ('hnorm.weight', 'hnorm.weight'),
                               ('eh_proj.weight', 'eh_proj.weight'),
                               ('final_layernorm.weight', 'shared_head.norm.weight')]:
            self._set_state_dict(mtp_layer, mg_key, hf_state_dict, hf_key, to_mcore)
        # eh_proj is kept in BF16 in the FP8 checkpoint (see modules_to_not_convert); never quantize it.
        self._fp8_skip_modules.update({'eh_proj'})
        # Inner block: nested HybridStack, sublayer 0 = DSA attention, sublayer 1 = MoE FFN.
        inner = None if mtp_layer is None else mtp_layer.mtp_model_layer
        attn_sub = None if inner is None else inner.layers[0]
        ffn_sub = None if inner is None else inner.layers[1]
        attn = None if attn_sub is None else attn_sub.self_attention
        hf_state_dict.update(self._set_dsa_state(attn, hf_state_dict, to_mcore))
        self._set_state_dict(attn_sub, 'input_layernorm.weight', hf_state_dict, self.hf_input_layernorm_key, to_mcore)
        mlp = None if ffn_sub is None else ffn_sub.mlp
        hf_state_dict.update(
            self._set_moe_state(mlp, hf_state_dict, f'{self.hf_mlp_prefix}.', hf_layer_idx, to_mcore, is_mtp=True))
        self._set_state_dict(ffn_sub, 'pre_mlp_layernorm.weight', hf_state_dict, self.hf_post_attention_layernorm_key,
                             to_mcore)
        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, layer_prefix)
            hf_state_dict.update(origin_hf_state_dict)
        return hf_state_dict


class Glm5NextLoader(ModelLoader):

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TELinear
        from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
        from megatron.core.transformer.transformer_config import MLATransformerConfig

        from ..modules import TopKRouter
        config = self.config
        if config.fp8 or config.fp4:
            raise NotImplementedError('The current model is validated for BF16/FP32 only; '
                                      'fp8/fp4 training is not supported')
        if config.dsa_indexer_loss_coeff:
            raise NotImplementedError('The current model has no KPool indexer auxiliary loss; '
                                      'use dsa_indexer_loss_coeff=0')
        if config.context_parallel_size > 1:
            if config.cp_comm_type != 'all_gather':
                logger.warning_once("GLM5-Next under context parallelism requires cp_comm_type='all_gather'; "
                                    "promoting to 'all_gather'.")
                config.cp_comm_type = 'all_gather'
        config.hetereogenous_dist_checkpoint = True
        config.rope_type = MLATransformerConfig.rope_type
        config.rotary_scaling_factor = MLATransformerConfig.rotary_scaling_factor
        config.mscale_all_dim = MLATransformerConfig.mscale_all_dim
        config.cache_mla_latents = MLATransformerConfig.cache_mla_latents
        config.enable_hyper_connections = True
        # GLM-5.3-Flash's MTP head (decoder layer index num_hidden_layers) is a single DSA-attention +
        # MoE-FFN block with plain residuals -- it carries no hyper-connection weights, unlike every
        # backbone block. This describes that inner block for Megatron's hybrid MTP path and is only
        # consumed when mtp_num_layers > 0. Set here rather than in the parser: it is a Megatron
        # TransformerConfig field (not a ModelConfig one), so emitting it from the parser would make
        # ModelConfig(**values) raise a bare TypeError on a Megatron that lacks it -- before
        # __post_init__ could run require_glm5_hybrid() and name the actual fix.
        config.mtp_hybrid_override_pattern = 'DE'
        config.mhc_norm_eps_inside_sqrt = config.mhc_keep_mappings_in_fp32 = True
        config.mhc_learned_output_contract = False
        config.kda_two_stage_gates = True
        config.kda_lower_bound = config.linear_lower_bound
        config.kda_safe_gate = config.linear_lower_bound is not None
        config.linear_num_key_heads = config.linear_num_value_heads = config.linear_num_heads
        config.linear_key_head_dim = config.linear_value_head_dim = config.linear_head_dim
        config.dsa_indexer_kpool = config.index_kpool
        config.dsa_indexer_kpool_always_select_tail = True
        config.dsa_indexer_rotate_activation = False
        config.dsa_indexer_weights_proj_use_quantization = False
        config.dsa_indexer_weights_proj_output_dtype = 'fp32'
        config.dsa_indexer_k_norm_epsilon = 1e-6
        config.dsa_indexer_k_norm_fp32 = True
        config.use_fused_mhc = False
        glm5_hybrid_layer_mapping(config)
        spec = deepcopy(hybrid_stack_spec)
        kda = spec.submodules.kda_layer.submodules
        kda.input_layernorm = Glm5NextHybridRMSNorm
        kda.self_attention.submodules.f_a_proj = kda.self_attention.submodules.g_a_proj = TELinear
        kda.self_attention.submodules.f_b_proj = kda.self_attention.submodules.g_b_proj = TEColumnParallelLinear
        dsa = spec.submodules.dsa_layer.submodules
        dsa.input_layernorm = Glm5NextHybridRMSNorm
        dsa.self_attention.submodules.q_layernorm = Glm5NextHybridRMSNorm
        dsa.self_attention.submodules.kv_layernorm = Glm5NextHybridRMSNorm
        dense = spec.submodules.mlp_layer.submodules
        dense.pre_mlp_layernorm = Glm5NextHybridRMSNorm
        dense.mlp.keywords['submodules'].linear_fc1 = TEColumnParallelLinear
        moe = spec.submodules.moe_layer.submodules
        moe.pre_mlp_layernorm = Glm5NextHybridRMSNorm
        moe.mlp.keywords['submodules'].router = TopKRouter
        # Outer decoder stack drops the multi-stream tensor so MTP stays on its non-mHC path.
        spec.module = Glm5NextHybridStack
        # MTP head (built only when mtp_num_layers > 0): non-mHC eh_proj over a plain DSA + MoE inner
        # block. Glm5NextMultiTokenPredictionLayer disables hyper-connections for the MTP subtree, so
        # only eh_proj/enorm/hnorm/layer_norm are used; the GLM RMSNorm matches the checkpoint's plain
        # (non zero-centered) enorm/hnorm/shared_head.norm.
        mtp_block_spec = getattr(spec.submodules, 'mtp_block_spec', None)
        if mtp_block_spec is not None:
            mtp_layer_spec = mtp_block_spec.submodules.layer_specs[0]
            mtp_layer_spec.module = Glm5NextMultiTokenPredictionLayer
            mtp_sub = mtp_layer_spec.submodules
            mtp_sub.enorm = mtp_sub.hnorm = mtp_sub.layer_norm = Glm5NextHybridRMSNorm
            mtp_sub.eh_proj = TEColumnParallelLinear
        return spec

    def build_model(self, pre_process=True, post_process=True, vp_stage: Optional[int] = None):
        spec = self.get_transformer_layer_spec(vp_stage)
        model_cls = Glm5NextMultimodalHybridModel if self.config.is_multimodal else Glm5NextHybridModel
        model = model_cls(self.config, spec, pre_process, post_process, vp_stage=vp_stage)
        self._set_linear_is_expert(model)
        return model


register_model(
    ModelMeta(
        ModelType.glm5_next,
        ['glm5_next'],
        bridge_cls=Glm5NextBridge,
        visual_cls=Glm5NextVit,
        loader=Glm5NextLoader,
    ))
