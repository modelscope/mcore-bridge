import json
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, VocabParallelEmbedding
from megatron.core.transformer import TransformerConfig
from safetensors.torch import save_file

from mcore_bridge.config.parser import _convert_config
from mcore_bridge.inference import DeepseekV41TextGenerationController
from mcore_bridge.model.gpts import deepseek_v41 as deepseek_v41_module
from mcore_bridge.model.gpts.deepseek_v41 import (
    DeepseekV41Aligner,
    DeepseekV41Bridge,
    DeepseekV41DSparkAttention,
    DeepseekV41Loader,
    DeepseekV41Vision,
    DeepseekV41VisionTransformer,
)
from mcore_bridge.model.modules import engram as engram_adapter
from mcore_bridge.model.modules.dspark import (
    DeepseekV41DSparkConfidenceHead,
    DeepseekV41DSparkInput,
    DeepseekV41DSparkMarkovHead,
    DeepseekV41DSparkOutput,
    DeepseekV41DSparkStack,
    DeepseekV41DSparkState,
    dspark_sample,
    verify_dspark_draft,
)
from mcore_bridge.utils.safetensors import SafetensorLazyLoader


def test_dspark_config_is_kept_separate_from_standard_mtp():
    text_config = SimpleNamespace(
        model_type='deepseek_v41_text',
        num_nextn_predict_layers=3,
        dspark_block_size=5,
        dspark_noise_token_id=128799,
        dspark_target_layer_ids=[37, 38, 39],
        dspark_markov_rank=256,
        dspark_n_routed_experts=128,
        dspark_num_experts_per_tok=3,
    )

    converted = _convert_config(SimpleNamespace(model_type='deepseek_v41', text_config=text_config))

    assert converted['dspark_num_layers'] == 3
    assert converted['dspark_block_size'] == 5
    assert converted['dspark_noise_token_id'] == 128799
    assert converted['dspark_target_layer_ids'] == [37, 38, 39]
    assert converted['dspark_markov_rank'] == 256
    assert converted['dspark_num_experts'] == 128
    assert converted['dspark_router_topk'] == 3
    assert 'mtp_num_layers' not in converted


def test_dspark_input_builds_parallel_noise_block():

    class _Projection(torch.nn.Module):

        def forward(self, hidden_states):
            return hidden_states[..., :2], None

    module = DeepseekV41DSparkInput.__new__(DeepseekV41DSparkInput)
    torch.nn.Module.__init__(module)
    module.hidden_size = 2
    module.num_streams = 3
    module.sequence_parallel = False
    module.block_size = 4
    module.noise_token_id = 7
    module.main_proj = _Projection()
    module.main_norm = torch.nn.Identity()

    main_hidden = torch.arange(2 * 2 * 4, dtype=torch.float32).view(2, 2, 4)
    input_ids = torch.tensor([1, 2])

    def embedding(token_ids):
        return torch.nn.functional.one_hot(token_ids % 2, 2).float()

    hidden_states, main_x, draft_ids = module(main_hidden, input_ids, embedding)

    assert hidden_states.shape == (4, 2, 6)
    assert torch.equal(draft_ids[0], input_ids)
    assert torch.all(draft_ids[1:] == 7)
    torch.testing.assert_close(main_x, main_hidden[..., :2])
    streams = hidden_states.view(4, 2, 3, 2)
    torch.testing.assert_close(streams[:, :, 0], streams[:, :, 1])
    torch.testing.assert_close(streams[:, :, 1], streams[:, :, 2])


def test_dspark_markov_head_returns_full_logits_and_embedding():

    class _Head(torch.nn.Module):

        def forward(self, hidden_states, runtime_gather_output):
            assert runtime_gather_output
            return torch.cat((hidden_states, hidden_states + 10), dim=-1), None

    module = DeepseekV41DSparkMarkovHead.__new__(DeepseekV41DSparkMarkovHead)
    torch.nn.Module.__init__(module)
    module.embed = torch.nn.Embedding.from_pretrained(torch.tensor([[1., 2.], [3., 4.], [5., 6.]]), )
    module.head = _Head()

    logits, embedding = module(torch.tensor([0, 2]))

    torch.testing.assert_close(embedding, torch.tensor([[1., 2.], [5., 6.]]))
    torch.testing.assert_close(logits, torch.tensor([[1., 2., 11., 12.], [5., 6., 15., 16.]]))


def test_dspark_tp_modules_construct_and_run_on_one_rank(tmp_path):
    if dist.is_initialized() and dist.get_world_size() != 1:
        pytest.skip('Single-rank DSpark TP smoke test.')
    if not dist.is_initialized():
        dist.init_process_group(
            'gloo',
            init_method=f'file://{tmp_path}/dspark-dist-init',
            rank=0,
            world_size=1,
        )
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(tensor_model_parallel_size=1)

    config = TransformerConfig(
        num_layers=1,
        hidden_size=4,
        num_attention_heads=1,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
    )
    config.padded_vocab_size = 8
    config.dspark_markov_rank = 2
    config.dspark_target_layer_ids = [0, 1]
    config.dspark_block_size = 3
    config.dspark_noise_token_id = 7
    config.num_residual_streams = 2

    markov = DeepseekV41DSparkMarkovHead(config)
    logits, markov_embed = markov(torch.tensor([0, 7]))
    assert logits.shape == (2, 8)
    assert markov_embed.shape == (2, 2)
    assert logits.dtype == torch.float32

    dspark_input = DeepseekV41DSparkInput(config)
    draft_embedding = VocabParallelEmbedding(
        8,
        4,
        init_method=config.init_method,
        config=config,
    )
    hidden_states, main_x, draft_ids = dspark_input(
        torch.randn(1, 2, 8),
        torch.tensor([1, 2]),
        draft_embedding,
    )
    assert hidden_states.shape == (3, 2, 8)
    assert main_x.shape == (1, 2, 4)
    assert draft_ids.shape == (3, 2)

    output_layer = ColumnParallelLinear(
        4,
        8,
        config=config,
        init_method=config.init_method,
        bias=False,
        gather_output=False,
        skip_bias_add=False,
    )
    dspark_output = DeepseekV41DSparkOutput(config)
    output_ids, logits, confidence = dspark_output(
        hidden_states.view(3, 2, 2, 4).mean(dim=2),
        torch.tensor([1, 2]),
        output_layer,
    )
    assert output_ids.shape == (2, 4)
    assert logits.shape == (2, 3, 8)
    assert confidence.shape == (2, 3)


def test_attach_dspark_freezes_the_draft_stack(tmp_path, monkeypatch):
    """The draft stack has to be attached frozen.

    It is deliberately kept out of the training forward -- it exists so the checkpoint's ``mtp.*``
    weights have somewhere to be loaded into and saved from -- so none of its parameters can ever
    hold anything but a zero gradient. Leaving them trainable costs more than the optimizer state and
    gradient buffers it needlessly allocates: Adam's weight decay is decoupled from the gradient, so
    every step still multiplies them by ``1 - lr * wd`` with nothing pushing back, and the weights the
    stack exists to carry erode over a long run.
    """
    if dist.is_initialized() and dist.get_world_size() != 1:
        pytest.skip('Single-rank DSpark construction test.')
    if not dist.is_initialized():
        dist.init_process_group(
            'gloo',
            init_method=f'file://{tmp_path}/dspark-freeze-init',
            rank=0,
            world_size=1,
        )
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(tensor_model_parallel_size=1)

    config = TransformerConfig(
        num_layers=1,
        hidden_size=4,
        num_attention_heads=1,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
    )
    config.padded_vocab_size = 8
    config.dspark_num_layers = 1
    config.dspark_markov_rank = 2
    config.dspark_target_layer_ids = [0]
    config.dspark_block_size = 3
    config.dspark_noise_token_id = 7
    config.num_residual_streams = 2
    config.mhc_single_pass = True

    class _Layer(torch.nn.Module):
        """Stands in for a draft TransformerLayer: one parameter, and an ``mlp`` to look for a router on."""

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))
            self.mlp = torch.nn.Module()

    class _LanguageModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.trainable = torch.nn.Parameter(torch.ones(2))
            self.config = config
            self.vocab_size = 8
            self.pg_collection = SimpleNamespace(tp=None)

    loader = DeepseekV41Loader.__new__(DeepseekV41Loader)
    loader.config = config
    monkeypatch.setattr(loader, 'get_dspark_layer_spec', lambda: (config, [object()]), raising=False)
    monkeypatch.setattr(loader, '_set_linear_is_expert', lambda module: None, raising=False)
    monkeypatch.setattr(deepseek_v41_module, 'build_module', lambda *args, **kwargs: _Layer())

    language_model = _LanguageModel()
    loader._attach_dspark(language_model, post_process=True)

    trainable = [name for name, p in language_model.dspark.named_parameters() if p.requires_grad]
    assert list(language_model.dspark.parameters()), 'the draft stack was attached with no parameters'
    assert not trainable, trainable
    # this ``language_model`` has no base ``embedding`` to seed drafts from, so the stack built its own
    assert not language_model.dspark_word_embeddings.weight.requires_grad
    # and nothing outside the draft stack was frozen along the way
    assert language_model.trainable.requires_grad


def test_dspark_attention_sink_and_ring_cache():
    attention = DeepseekV41DSparkAttention.__new__(DeepseekV41DSparkAttention)
    torch.nn.Module.__init__(attention)
    attention.window_size = 3
    attention._dspark_window_kv_cache = None
    attention.config = SimpleNamespace(v_head_dim=2)
    attention.core_attention = SimpleNamespace(attn_sink=torch.nn.Parameter(torch.tensor([0.0])))

    query = torch.tensor([[[[1.0, 0.0]]]])
    key_value = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]])
    actual = attention._latent_attention(query, key_value)
    scale = 2**-0.5
    probabilities = torch.softmax(torch.tensor([scale, 0.0, 0.0]), dim=0)[:2]
    expected = torch.tensor([[[[probabilities[0], probabilities[1]]]]])
    torch.testing.assert_close(actual, expected)

    first = torch.arange(5 * 1 * 1 * 2, dtype=torch.float32).view(5, 1, 1, 2)
    cache, valid_lengths = attention._write_main_cache(first, start_pos=0)
    assert torch.equal(valid_lengths, torch.tensor([3]))
    torch.testing.assert_close(cache[0], first[3].squeeze(-2))
    torch.testing.assert_close(cache[1], first[4].squeeze(-2))
    torch.testing.assert_close(cache[2], first[2].squeeze(-2))
    update = torch.tensor([[[[20.0, 21.0]]]])
    cache, valid_lengths = attention._write_main_cache(update, start_pos=5)
    assert torch.equal(valid_lengths, torch.tensor([3]))
    torch.testing.assert_close(cache[2], update[0].squeeze(-2))


def test_dspark_verification_accepts_only_strict_matching_prefix():
    draft_ids = torch.tensor([
        [10, 11, 12, 13],
        [20, 21, 22, 23],
        [30, 31, 32, 33],
    ])
    target_ids = torch.tensor([
        [11, 12, 13, 14],
        [21, 99, 23, 24],
        [98, 32, 33, 34],
    ])

    result = verify_dspark_draft(draft_ids, target_ids)

    assert torch.equal(result.accepted_lengths, torch.tensor([3, 1, 0]))
    assert torch.equal(result.next_tokens, torch.tensor([14, 99, 98]))
    assert torch.equal(result.accepted_mask,
                       torch.tensor([
                           [True, True, True],
                           [True, False, False],
                           [False, False, False],
                       ]))

    confidence = torch.tensor([[10.0, -10.0, 10.0]] * 3)
    result = verify_dspark_draft(draft_ids, target_ids, confidence, confidence_threshold=0.5)
    assert torch.equal(result.accepted_lengths, torch.tensor([1, 1, 0]))


def test_dspark_state_preserves_recompute_inputs():
    main_hidden = torch.randn(2, 1, 4)
    rotary = torch.randn(2, 1, 1, 2)
    cache_slots = torch.tensor([3])
    state = DeepseekV41DSparkState(main_hidden, rotary, cache_slots)

    attention_kwargs = state.attention_kwargs()
    assert attention_kwargs['dspark_main_hidden'] is main_hidden
    assert attention_kwargs['dspark_main_rotary_pos_emb'] is rotary
    assert attention_kwargs['dspark_cache_slots'] is cache_slots
    tensors, restore = state.save_for_recompute()
    restored = restore(tensors)
    assert restored.main_hidden is main_hidden
    assert restored.main_rotary_pos_emb is rotary
    assert restored.cache_slots is cache_slots


def test_dspark_stack_prefill_and_decode_lifecycle():

    class _Input(torch.nn.Module):

        def forward(self, main_hidden, input_ids, embedding):
            hidden = embedding(input_ids).unsqueeze(0).expand(2, -1, -1)
            return hidden, main_hidden[..., :2], None

    class _Attention(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.prefill_args = None

        def prefill_dspark(
            self,
            main_hidden,
            rotary_pos_emb,
            inference_context,
            start_pos=0,
            cache_slots=None,
        ):
            self.prefill_args = (
                main_hidden,
                rotary_pos_emb,
                inference_context,
                start_pos,
                cache_slots,
            )

    class _Layer(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.self_attention = _Attention()
            self.received_main_hidden = None

        def forward(self, hidden_states, **kwargs):
            state = kwargs['cross_layer_state']
            self.received_main_hidden = state.main_hidden
            mhc_state = kwargs['mhc_state']
            mhc_state.pre_mix = hidden_states.new_full(hidden_states.shape[:2] + (2, ), 0.5)
            return hidden_states + 1, None

    class _Output(torch.nn.Module):

        def forward(self, hidden_states, input_ids, output_layer, temperature, sample_fn=None):
            return hidden_states, input_ids, temperature

    stack = DeepseekV41DSparkStack.__new__(DeepseekV41DSparkStack)
    torch.nn.Module.__init__(stack)
    stack.config = SimpleNamespace(num_residual_streams=2, use_fused_mhc=False)
    stack.input = _Input()
    stack.layers = torch.nn.ModuleList([_Layer(), _Layer()])
    stack.output = _Output()
    embedding = torch.nn.Embedding.from_pretrained(torch.arange(10).float().unsqueeze(-1).expand(-1, 4))
    main_hidden = torch.randn(1, 2, 4)
    rotary = torch.randn(1, 1, 1, 2)

    assert stack(
        main_hidden,
        torch.tensor([1, 2]),
        embedding,
        lambda *_args, **_kwargs: None,
        start_pos=0,
        main_rotary_pos_emb=rotary,
    ) is None
    for layer in stack.layers:
        prefill_main, prefill_rotary, prefill_context, prefill_start, prefill_slots = (
            layer.self_attention.prefill_args)
        torch.testing.assert_close(prefill_main, main_hidden[..., :2])
        assert prefill_rotary is rotary
        assert prefill_context is None
        assert prefill_start == 0
        assert prefill_slots is None

    contracted, returned_ids, temperature = stack(
        main_hidden,
        torch.tensor([1, 2]),
        embedding,
        lambda *_args, **_kwargs: None,
        start_pos=1,
        temperature=0.5,
    )
    assert contracted.shape == (2, 2, 2)
    assert torch.equal(returned_ids, torch.tensor([1, 2]))
    assert temperature == 0.5
    for layer in stack.layers:
        assert layer.received_main_hidden is not None


def test_dspark_confidence_head_uses_fp32_projection():
    config = SimpleNamespace(
        hidden_size=3,
        dspark_markov_rank=2,
        use_cpu_initialization=True,
    )
    module = DeepseekV41DSparkConfidenceHead(config)
    module.proj.weight.data.copy_(torch.tensor([[1., 2., 3., 4., 5.]]))
    hidden_states = torch.tensor([[[1., 2., 3.]]], dtype=torch.bfloat16)
    markov_embed = torch.tensor([[[4., 5.]]], dtype=torch.bfloat16)

    actual = module(hidden_states, markov_embed)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, torch.tensor([[55.]]))


def test_dspark_output_applies_markov_recurrence_in_block_order():

    class _OutputLayer(torch.nn.Module):

        def forward(self, hidden_states, runtime_gather_output):
            assert runtime_gather_output
            return hidden_states.new_zeros((*hidden_states.shape[:-1], 5)), None

    class _MarkovHead(torch.nn.Module):

        def forward(self, token_ids):
            logits = torch.nn.functional.one_hot((token_ids + 1) % 5, 5).float() * 10
            return logits, token_ids.float().unsqueeze(-1)

    class _ConfidenceHead(torch.nn.Module):

        def forward(self, hidden_states, markov_embed):
            return hidden_states[..., 0].float() + markov_embed[..., 0]

    module = DeepseekV41DSparkOutput.__new__(DeepseekV41DSparkOutput)
    torch.nn.Module.__init__(module)
    module.block_size = 3
    module.norm = torch.nn.Identity()
    module.markov_head = _MarkovHead()
    module.confidence_head = _ConfidenceHead()
    hidden_states = torch.tensor([
        [[1., 0.], [2., 0.]],
        [[3., 0.], [4., 0.]],
        [[5., 0.], [6., 0.]],
    ])

    output_ids, logits, confidence = module(
        hidden_states,
        torch.tensor([0, 2]),
        _OutputLayer(),
    )

    assert torch.equal(output_ids, torch.tensor([[0, 1, 2, 3], [2, 3, 4, 0]]))
    assert logits.shape == (2, 3, 5)
    torch.testing.assert_close(confidence, torch.tensor([[1., 4., 7.], [4., 7., 10.]]))
    assert torch.equal(dspark_sample(logits, temperature=0), output_ids[:, 1:])


def test_controller_routes_speculative_proposals_to_dspark_provider():
    calls = {}

    class _Model:

        def compute_dspark_speculative_tokens(self, **kwargs):
            calls.update(kwargs)
            return torch.tensor([[7, 8], [9, 10]])

    context = SimpleNamespace(
        total_request_count=2,
        paused_request_count=0,
        _nvls_dispatcher=None,
    )
    controller = DeepseekV41TextGenerationController.__new__(DeepseekV41TextGenerationController)
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    controller._unwrapped_model = _Model()
    controller._is_last_pp_stage = True
    controller.model_is_pipeline_parallel = False
    controller.model_config = SimpleNamespace(dspark_block_size=3)
    controller.num_speculative_tokens = 2
    controller._sampled_tokens_cuda = torch.tensor([5, 6])
    controller._accepted_token_counts_per_request = torch.tensor([1, 0])
    controller._last_accepted_seq_indices = torch.tensor([1, 3])
    controller._sampled_mtp_tokens_cuda = torch.empty(2, 2, dtype=torch.long)
    controller._sample_from_logits_2d = lambda logits: logits.argmax(dim=-1)

    controller._compute_dspark_and_sample()

    assert torch.equal(controller._sampled_mtp_tokens_cuda, torch.tensor([[7, 8], [9, 10]]))
    assert calls['inference_context'] is context
    assert calls['sample_fn'] is controller._sample_from_logits_2d
    assert calls['num_speculative_tokens'] == 2


def test_engram_adapter_remaps_checkpoint_layers_to_megatron_layers(tmp_path):
    if not engram_adapter.has_native_engram():
        pytest.skip('The PR #7224 baseline intentionally has no Engram extension.')
    artifact = tmp_path / 'tokenizer-map.json'
    artifact.write_text(
        json.dumps({
            'format': 'megatron-engram-token-map',
            'version': 1,
            'source_vocab_size': 8,
            'compressed_vocab_size': 8,
            'pad_token_id': 0,
            'compressed_pad_token_id': 0,
            'max_ngram_order': 3,
            'hash_seed': 0,
            'layer_ids': [0, 2],
            'layer_multipliers': {
                '0': [11, 13, 15],
                '2': [17, 19, 21]
            },
            'remap': list(range(8)),
        }))
    config = engram_adapter.build_deepseek_v41_engram_config(
        placement_layer_ids=(1, 3),
        hash_layer_ids=(0, 2),
        excluded_token_ids=(99, ),
        global_vocab_sizes=(17, 19),
        max_ngram_order=3,
        num_hash_heads=1,
        memory_dim=4,
        kernel_size=1,
        hash_seed=0,
        boundary_token_id=0,
        tokenizer_map_path=str(artifact),
    )

    assert config.layer_ids == (1, 3)
    assert config.hash_layer_ids == (0, 2)
    assert config.layer_multipliers == {1: (11, 13, 15), 3: (17, 19, 21)}
    assert set(config.table_sizes_by_layer) == {1, 3}
    assert config.excluded_token_ids == (99, )


def _engram_config_for_validation(tmp_path):
    artifact = tmp_path / 'tokenizer-map.json'
    artifact.write_text(
        json.dumps({
            'format': 'megatron-engram-token-map',
            'version': 1,
            'source_vocab_size': 8,
            'compressed_vocab_size': 8,
            'pad_token_id': 0,
            'compressed_pad_token_id': 0,
            'max_ngram_order': 3,
            'hash_seed': 0,
            'layer_ids': [0],
            'layer_multipliers': {
                '0': [11, 13, 15]
            },
            'remap': list(range(8)),
        }))
    return engram_adapter.build_deepseek_v41_engram_config(
        placement_layer_ids=(1, ),
        hash_layer_ids=(0, ),
        global_vocab_sizes=(17, 19),
        max_ngram_order=3,
        num_hash_heads=1,
        memory_dim=4,
        kernel_size=1,
        hash_seed=0,
        boundary_token_id=0,
        tokenizer_map_path=str(artifact),
    )


def test_engram_config_allows_context_and_virtual_pipeline_but_keeps_the_other_guards(tmp_path):
    if not engram_adapter.has_native_engram():
        pytest.skip('The PR #7224 baseline intentionally has no Engram extension.')
    config = _engram_config_for_validation(tmp_path)
    parallelism = dict(
        context_parallel_size=2,
        tensor_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        sequence_parallel=False,
    )

    # V4.1 hashes the full sequence locally and slices it, so CP no longer has to be 1.
    config._validate_parallelism(SimpleNamespace(**parallelism), None)
    # VPP is now allowed too: Engram.forward is self-contained and layer placement uses the
    # vp_stage-aware global layer_number, so the upstream blanket VPP guard is dropped.
    config._validate_parallelism(SimpleNamespace(**{**parallelism, 'virtual_pipeline_model_parallel_size': 2}), None)
    # ... but only the CP and VPP guards are relaxed; every other parallelism check still fires.
    with pytest.raises(ValueError, match='expert_tensor_parallel_size'):
        config._validate_parallelism(SimpleNamespace(**{**parallelism, 'expert_tensor_parallel_size': 2}), None)


def test_engram_config_allows_packed_sequences_without_losing_the_pipeline_guard(tmp_path):
    if not engram_adapter.has_native_engram():
        pytest.skip('The PR #7224 baseline intentionally has no Engram extension.')
    config = _engram_config_for_validation(tmp_path)
    assert not config.variant_spec.supports_packed_sequences

    config._validate_packed_sequences(SimpleNamespace(pipeline_model_parallel_size=1), packed_sequences=True)
    # The temporary variant override must not leak into the hashing path.
    assert not config.variant_spec.supports_packed_sequences

    with pytest.raises(ValueError, match='pipeline_model_parallel_size > 2'):
        config._validate_packed_sequences(SimpleNamespace(pipeline_model_parallel_size=4), packed_sequences=True)


def test_engram_hash_blocks_suffixes_after_excluded_token():
    hashes = engram_adapter._hash_token_windows(
        token_windows=torch.tensor([[[5, -1, 7]]]),
        tokenizer_remap=None,
        multipliers=torch.tensor([1, 10, 100]),
        table_sizes=torch.tensor([997, 991]),
        max_ngram_order=3,
        num_hash_heads=1,
        boundary_token_id=0,
        invalid_token_id=-1,
    )

    assert torch.equal(hashes, torch.tensor([[[5, 5]]]))


def test_engram_static_inference_cache_matches_full_sequence_hashing():
    module = engram_adapter.DeepseekV41Engram.__new__(engram_adapter.DeepseekV41Engram)
    torch.nn.Module.__init__(module)
    module.engram_config = SimpleNamespace(
        excluded_token_ids=(99, ),
        max_ngram_order=3,
        num_hash_heads=1,
        hash_boundary_token_id=0,
        boundary_token_id=0,
        num_tables=2,
        variant_spec=SimpleNamespace(resets_windows_at_boundary_token=False),
    )
    module.tokenizer_remap = None
    module.hash_multipliers = torch.tensor([11, 13, 15])
    module.table_sizes = torch.tensor([997, 991])
    full_hashes, full_live = module._build_hash_ids(torch.tensor([[1, 2, 3]]))
    context = SimpleNamespace(
        max_batch_size=1,
        max_sequence_length=8,
        batch_size_offset=0,
        sequence_len_offset=0,
        is_static_batching=lambda: True,
    )

    prefill_hashes, prefill_live = module._build_hash_ids(torch.tensor([[1, 2]]), context)
    context.sequence_len_offset = 2
    decode_hashes, decode_live = module._build_hash_ids(torch.tensor([[3]]), context)

    assert torch.equal(torch.cat((prefill_hashes, decode_hashes), dim=1), full_hashes)
    assert torch.equal(torch.cat((prefill_live, decode_live), dim=1), full_live)


def _ngram_hash_kwargs():
    return dict(
        tokenizer_remap=None,
        multipliers=torch.tensor([11, 13, 15]),
        table_sizes=torch.tensor([997, 991]),
        max_ngram_order=3,
        num_hash_heads=1,
        boundary_token_id=0,
        reset_at_boundary=False,
    )


def test_engram_packed_hashes_match_separately_hashed_documents():
    # The DeepSeek variant carries no boundary token in the stream, so cu_seqlens is the only
    # thing that stops an n-gram window from reaching into the previous packed document.
    packed_row = torch.tensor([[5, 6, 7, 8, 9]])
    kwargs = _ngram_hash_kwargs()

    packed = engram_adapter._build_ngram_hashes(packed_row, cu_seqlens=torch.tensor([0, 2, 5]), **kwargs)
    separate = torch.cat(
        (
            engram_adapter._build_ngram_hashes(packed_row[:, :2], **kwargs),
            engram_adapter._build_ngram_hashes(packed_row[:, 2:], **kwargs),
        ),
        dim=1,
    )

    assert torch.equal(packed, separate)
    # Without the reset the second document would mix in tokens 5 and 6.
    assert not torch.equal(packed, engram_adapter._build_ngram_hashes(packed_row, **kwargs))


def test_engram_hashes_are_unchanged_when_cu_seqlens_spans_one_document():
    packed_row = torch.tensor([[5, -1, 7, 8]])
    kwargs = _ngram_hash_kwargs()

    assert torch.equal(
        engram_adapter._build_ngram_hashes(packed_row, cu_seqlens=torch.tensor([0, 4]), **kwargs),
        engram_adapter._build_ngram_hashes(packed_row, **kwargs),
    )


def test_engram_rejects_cu_seqlens_that_does_not_cover_the_row():
    with pytest.raises(ValueError, match='cu_seqlens ends at'):
        engram_adapter._build_ngram_hashes(
            torch.tensor([[5, 6, 7]]), cu_seqlens=torch.tensor([0, 2]), **_ngram_hash_kwargs())


def _bare_engram(context_parallel_size=1, sequence_parallel=False):
    module = engram_adapter.DeepseekV41Engram.__new__(engram_adapter.DeepseekV41Engram)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(context_parallel_size=context_parallel_size, sequence_parallel=sequence_parallel)
    return module


def test_engram_context_parallel_slices_reassemble_the_global_hashes(monkeypatch):
    from mcore_bridge.utils import megatron_utils

    global_hashes = torch.arange(2 * 8 * 3).view(2, 8, 3)
    cp_size = 4
    monkeypatch.setattr(megatron_utils.mpu, 'get_context_parallel_world_size', lambda: cp_size)

    slices = []
    for cp_rank in range(cp_size):
        monkeypatch.setattr(megatron_utils.mpu, 'get_context_parallel_rank', lambda rank=cp_rank: rank)
        slices.append(_bare_engram(cp_size)._slice_for_context_parallel(global_hashes))

    # Contiguous partitioning must hand rank r the block [r * local, (r + 1) * local).
    assert all(item.shape == (2, 2, 3) for item in slices)
    assert torch.equal(torch.cat(slices, dim=1), global_hashes)


def test_contiguous_cp_reconstruct_inverts_the_matching_split(monkeypatch):
    """Multimodal V4.1 splits embeddings/input_ids itself, so both directions must
    honour ``cp_partition_mode``: reconstructing a contiguous shard with the zigzag
    layout silently reorders tokens away from what the DSv4 THD CP forward assumes."""
    from mcore_bridge.utils import megatron_utils

    global_ids = torch.arange(8).view(1, 8)
    cp_size, cp_rank = 2, 1
    local_length = global_ids.shape[1] // cp_size
    monkeypatch.setattr(megatron_utils.mpu, 'get_context_parallel_world_size', lambda: cp_size)
    monkeypatch.setattr(megatron_utils.mpu, 'get_context_parallel_rank', lambda: cp_rank)
    monkeypatch.setattr(megatron_utils.mpu, 'get_context_parallel_group', lambda: 'cp-group')

    shard = megatron_utils.split_cp_inputs(global_ids, None, 1, cp_partition_mode='contiguous')

    def fake_all_gather(output_list, tensor, group=None):
        assert group == 'cp-group'
        assert torch.equal(tensor, shard)
        for rank, buffer in enumerate(output_list):
            buffer.copy_(global_ids[:, rank * local_length:(rank + 1) * local_length])

    monkeypatch.setattr(torch.distributed, 'all_gather', fake_all_gather)
    assert torch.equal(
        megatron_utils.reconstruct_tensor_cp(shard, None, dim=1, cp_partition_mode='contiguous'),
        global_ids,
    )
    # The default zigzag layout must not be applied to a contiguous shard.
    assert not torch.equal(megatron_utils.reconstruct_tensor_cp(shard, None, dim=1), global_ids)


def test_engram_gathers_cp_sharded_input_ids_but_leaves_full_ones_alone(monkeypatch):
    full_ids = torch.arange(8).view(1, 8)
    cp_size, cp_rank = 4, 2
    local_length = 2

    monkeypatch.setattr(engram_adapter.mpu, 'get_context_parallel_group', lambda: 'cp-group')

    def fake_all_gather(output_list, tensor, group=None):
        assert group == 'cp-group'
        assert torch.equal(tensor, full_ids[:, cp_rank * local_length:(cp_rank + 1) * local_length])
        for rank, buffer in enumerate(output_list):
            buffer.copy_(full_ids[:, rank * local_length:(rank + 1) * local_length])

    monkeypatch.setattr(torch.distributed, 'all_gather', fake_all_gather)
    module = _bare_engram(cp_size)

    shard = full_ids[:, cp_rank * local_length:(cp_rank + 1) * local_length]
    assert torch.equal(module._gather_input_ids_for_context_parallel(shard, local_length), full_ids)
    # Multimodal models keep input_ids whole and split the embeddings instead.
    assert module._gather_input_ids_for_context_parallel(full_ids, local_length) is full_ids
    with pytest.raises(ValueError, match='matches neither'):
        module._gather_input_ids_for_context_parallel(full_ids[:, :5], local_length)


def test_engram_cp_local_sequence_length_undoes_the_inner_sp_split(monkeypatch):
    monkeypatch.setattr(engram_adapter, 'get_pg_size', lambda group: 2)
    hidden_states = torch.zeros(4, 1, 8)

    assert _bare_engram(2)._cp_local_sequence_length(hidden_states) == 4
    module = _bare_engram(2, sequence_parallel=True)
    module.tp_group = None
    assert module._cp_local_sequence_length(hidden_states) == 8


def test_engram_layer_spec_uses_bridge_owned_module():
    if not engram_adapter.has_native_engram():
        pytest.skip('The PR #7224 baseline intentionally has no Engram extension.')
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_layer import (
        HyperConnectionTransformerLayer,
        TransformerLayerSubmodules,
    )

    layer_spec = ModuleSpec(
        module=HyperConnectionTransformerLayer,
        submodules=TransformerLayerSubmodules(),
    )
    block_spec = SimpleNamespace(layer_specs=[layer_spec])
    config = SimpleNamespace(layer_ids=(1, ))

    engram_adapter.adapt_deepseek_v41_layer_specs(block_spec, config)

    assert layer_spec.module is engram_adapter.DeepseekV41HyperConnectionTransformerLayer
    assert layer_spec.submodules.engram.module is engram_adapter.DeepseekV41Engram


def test_allow_engram_inference_preserves_input_ids_and_restores_flag():
    config = SimpleNamespace(engram_enabled=True)
    input_ids = torch.tensor([[1, 2]])

    with engram_adapter.allow_engram_inference(config, input_ids, {'marker': 1}) as kwargs:
        assert not config.engram_enabled
        assert kwargs['marker'] == 1
        assert kwargs['input_ids'] is input_ids

    assert config.engram_enabled


class _Table:

    def __init__(self, global_rows, row_start, row_end, dim):
        self.global_num_embeddings = global_rows
        self.row_start = row_start
        self.row_end = row_end
        self.weight = torch.nn.Parameter(torch.empty(row_end - row_start, dim, dtype=torch.bfloat16))

    @property
    def local_num_embeddings(self):
        return self.row_end - self.row_start


def test_safetensor_lazy_loader_reads_only_requested_rows(tmp_path):
    path = tmp_path / 'model.safetensors'
    tensor = torch.arange(40, dtype=torch.float32).view(10, 4)
    save_file({'table': tensor}, path)

    with SafetensorLazyLoader(str(tmp_path)) as loader:
        lazy = loader.get_state_dict()['table']
        sliced = lazy.load_slice(slice(3, 6))

    torch.testing.assert_close(sliced, tensor[3:6])


def test_dspark_bridge_uses_tp_layout_and_official_endpoint_names():
    bridge = DeepseekV41Bridge.__new__(DeepseekV41Bridge)
    bridge.config = SimpleNamespace(task_type='causal_lm', dspark_num_layers=3)
    assert bridge._get_tp_split_dim('input.main_proj.weight') == 1
    assert bridge._get_tp_split_dim('output.markov_head.embed.weight') == 0
    assert bridge._get_tp_split_dim('output.markov_head.head.weight') == 0
    assert bridge._get_tp_split_dim('output.confidence_head.proj.weight') is None

    calls = []

    def record(_module, mg_key, _state, hf_key, to_mcore):
        calls.append((mg_key, hf_key, to_mcore))

    bridge._set_state_dict = record
    result = bridge._set_dspark_endpoints(object(), {}, to_mcore=False)

    assert result == {}
    assert calls == [
        ('input.main_proj.weight', 'main_proj.weight', False),
        ('input.main_norm.weight', 'main_norm.weight', False),
        ('output.norm.weight', 'norm.weight', False),
        ('output.markov_head.embed.weight', 'markov_head.embed.weight', False),
        ('output.markov_head.head.weight', 'markov_head.head.weight', False),
        ('output.confidence_head.proj.weight', 'confidence_head.proj.weight', False),
    ]


def test_dspark_bridge_loads_dedicated_embedding_with_padding_and_tp_shard():
    bridge = DeepseekV41Bridge.__new__(DeepseekV41Bridge)
    bridge.hf_embed_key = 'model.embed.weight'
    bridge.config = SimpleNamespace(padded_vocab_size=8)
    bridge.tp_size = 2
    bridge.tp_rank = 1

    class _Lazy:

        def __init__(self, tensor):
            self.tensor = tensor

        def load(self):
            return self.tensor

    # Six real vocab rows are padded up to padded_vocab_size=8, then split across TP.
    hf_rows = torch.arange(6 * 4, dtype=torch.float32).view(6, 4)
    padded = torch.nn.functional.pad(hf_rows, (0, 0, 0, 2))
    expected = padded.chunk(2, dim=0)[1]

    embedding = SimpleNamespace(weight=torch.zeros(4, 4))
    bridge._load_dspark_word_embeddings(embedding, {'model.embed.weight': _Lazy(hf_rows)})

    torch.testing.assert_close(embedding.weight, expected)


def test_engram_flat_fp8_table_is_dequantized_into_local_prime_shards():
    tables = [_Table(5, 1, 4, 4), _Table(7, 4, 7, 4)]
    engram = SimpleNamespace(
        embedding=SimpleNamespace(tables=tables),
        # HF layer 1 -> doubled-space attention layer_number ``2 * 1 + 1``.
        layer_number=3,
    )
    bridge = DeepseekV41Bridge.__new__(DeepseekV41Bridge)
    bridge.config = SimpleNamespace(engram_num_embeddings=[12], engram_layer_ids=[1])
    bridge._ENGRAM_LOAD_CHUNK_ROWS = 2

    raw = (torch.arange(48, dtype=torch.float32).view(12, 4) % 8).to(torch.float8_e4m3fn)
    scale = torch.tensor([[1.0, 0.5]] * 12, dtype=torch.float32)

    class _Lazy:

        def __init__(self, tensor):
            self.tensor = tensor
            self.slices = []

        def load(self):
            raise AssertionError('full tensor loading is forbidden for Engram tables')

        def load_slice(self, slices):
            self.slices.append(slices)
            return self.tensor[slices]

    lazy_weight, lazy_scale = _Lazy(raw), _Lazy(scale)
    bridge._load_engram_embedding(
        engram,
        {
            'engram.embed.weight': lazy_weight,
            'engram.embed.weight_scale_inv': lazy_scale,
        },
    )

    expected = DeepseekV41Bridge._dequantize_engram_rows(raw, scale).to(torch.bfloat16)
    torch.testing.assert_close(tables[0].weight, expected[1:4])
    torch.testing.assert_close(tables[1].weight, expected[9:12])
    assert [(item.start, item.stop) for item in lazy_weight.slices] == [(1, 3), (3, 4), (9, 11), (11, 12)]
    assert [(item.start, item.stop) for item in lazy_scale.slices] == [(1, 3), (3, 4), (9, 11), (11, 12)]


def test_engram_dense_weights_are_dequantized_and_split_like_official_wkv():

    class _Lazy:

        def __init__(self, tensor):
            self.tensor = tensor

        def load(self):
            return self.tensor

    hidden_size, num_streams, memory_dim = 3, 2, 4
    engram = SimpleNamespace(
        num_streams=num_streams,
        hidden_size=hidden_size,
        engram_config=SimpleNamespace(total_memory_dim=memory_dim),
        key_projection=SimpleNamespace(weight=torch.nn.Parameter(torch.empty(num_streams * hidden_size, memory_dim))),
        value_projection=SimpleNamespace(weight=torch.nn.Parameter(torch.empty(hidden_size, memory_dim))),
        query_norm=SimpleNamespace(weight=torch.nn.Parameter(torch.empty(num_streams * hidden_size))),
        key_norm=SimpleNamespace(weight=torch.nn.Parameter(torch.empty(num_streams * hidden_size))),
    )
    bridge = DeepseekV41Bridge.__new__(DeepseekV41Bridge)
    bridge._load_engram_embedding = lambda *_args: None
    raw_wkv = (torch.arange(36, dtype=torch.float32).view(9, 4) % 8).to(torch.float8_e4m3fn)
    scale = torch.tensor([[1.0, 0.5]] * 9, dtype=torch.float32)
    q_weight = torch.arange(6, dtype=torch.float32).view(2, 3)
    k_weight = q_weight + 10

    bridge._set_layer_engram(
        SimpleNamespace(engram=engram),
        {
            'engram.wkv.weight': _Lazy(raw_wkv),
            'engram.wkv.weight_scale_inv': _Lazy(scale),
            'engram.q_weight': _Lazy(q_weight),
            'engram.k_weight': _Lazy(k_weight),
        },
        to_mcore=True,
    )

    expected = bridge._dequantize_engram_rows(raw_wkv, scale)
    torch.testing.assert_close(engram.key_projection.weight, expected[:6])
    torch.testing.assert_close(engram.value_projection.weight, expected[6:])
    torch.testing.assert_close(engram.query_norm.weight, q_weight.flatten())
    torch.testing.assert_close(engram.key_norm.weight, k_weight.flatten())


def test_vision_and_aligner_match_official_equations():
    torch.manual_seed(7)
    vision_config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=6,
        num_hidden_layers=2,
        patch_size=2,
        rope_theta=10000.0,
        downsample_ratio=2,
    )
    vision = DeepseekV41VisionTransformer(vision_config)
    aligner = DeepseekV41Aligner(vision_config, text_hidden_size=10)
    patches = torch.randn(6, 3, 2, 2)

    actual_vision = vision(patches, n_h=2, n_w=3)
    x = actual_vision.view(2, 3, -1).permute(2, 0, 1)
    x = torch.nn.functional.pad(x, (0, 1, 0, 0))
    unfolded = torch.nn.functional.unfold(x.unsqueeze(0), 2, stride=2).squeeze(0).transpose(0, 1)
    expected = aligner.w2(torch.nn.functional.gelu(aligner.w1(unfolded)))

    actual = aligner(actual_vision, n_h=2, n_w=3)
    assert actual_vision.shape == (6, 8)
    assert actual.shape == (2, 10)
    torch.testing.assert_close(actual, expected)
    assert set(vision.state_dict()) == {
        'patch_embed.proj.weight',
        'patch_embed.proj.bias',
        'blocks.0.norm1.weight',
        'blocks.0.attn.wqkv.weight',
        'blocks.0.attn.wqkv.bias',
        'blocks.0.attn.wo.weight',
        'blocks.0.attn.wo.bias',
        'blocks.0.norm2.weight',
        'blocks.0.mlp.w1.weight',
        'blocks.0.mlp.w2.weight',
        'blocks.1.norm1.weight',
        'blocks.1.attn.wqkv.weight',
        'blocks.1.attn.wqkv.bias',
        'blocks.1.attn.wo.weight',
        'blocks.1.attn.wo.bias',
        'blocks.1.norm2.weight',
        'blocks.1.mlp.w1.weight',
        'blocks.1.mlp.w2.weight',
        'norm.weight',
    }


def test_vision_merges_official_image_span_layout():
    torch.manual_seed(11)
    vision_config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=6,
        num_hidden_layers=1,
        patch_size=2,
        rope_theta=10000.0,
        downsample_ratio=2,
    )
    config = SimpleNamespace(
        hf_config=SimpleNamespace(image_token_id=42, vision_config=vision_config),
        hidden_size=10,
        language_model_only=False,
        params_dtype=torch.float32,
    )
    module = DeepseekV41Vision(config)
    device = module.image_start.device
    input_ids = torch.tensor([[5, 42, 42, 42, 42, 6]], device=device)
    token_types = torch.tensor([[-1, 0, 1, 2, 3, -1]], device=device)
    inputs_embeds = torch.zeros(1, 6, 10, device=device)
    patches = torch.randn(4, 3, 2, 2, device=device)
    grid = torch.tensor([[1, 2, 2]], device=device)

    image_features = module.encode_images(patches, grid)
    actual = module.get_inputs_embeds(
        inputs_embeds,
        input_ids=input_ids,
        pixel_values=patches,
        image_grid_thw=grid,
        image_token_types=token_types,
    )

    torch.testing.assert_close(actual[0, 0], inputs_embeds[0, 0])
    torch.testing.assert_close(actual[0, 1], module.image_start)
    torch.testing.assert_close(actual[0, 2], image_features[0])
    torch.testing.assert_close(actual[0, 3], module.image_newline)
    torch.testing.assert_close(actual[0, 4], module.image_end)
    torch.testing.assert_close(actual[0, 5], inputs_embeds[0, 5])
