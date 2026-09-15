from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.inference.text_generation_controllers.text_generation_controller import TextGenerationController
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, VocabParallelEmbedding
from megatron.core.transformer import TransformerConfig
from safetensors.torch import save_file

from mcore_bridge.config.parser import _convert_config
from mcore_bridge.model.gpts.deepseek_v41 import (
    DeepseekV41Aligner,
    DeepseekV41Bridge,
    DeepseekV41DSparkAttention,
    DeepseekV41GPTModel,
    DeepseekV41Vision,
    DeepseekV41VisionTransformer,
)
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


def test_dspark_target_hidden_averages_mhc_streams():
    hidden = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).view(2, 3, 20)

    actual = DeepseekV41GPTModel._contract_dspark_target_hidden(hidden, num_streams=4)
    expected = hidden.view(2, 3, 4, 5).mean(dim=2)

    assert actual.shape == (2, 3, 5)
    torch.testing.assert_close(actual, expected)


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
    module.embed = torch.nn.Embedding.from_pretrained(
        torch.tensor([[1., 2.], [3., 4.], [5., 6.]]),
    )
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
    assert torch.equal(result.accepted_mask, torch.tensor([
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
            mhc_state.pre_mix = hidden_states.new_full(hidden_states.shape[:2] + (2,), 0.5)
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
            layer.self_attention.prefill_args
        )
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


def test_dspark_model_commits_only_verified_states_before_proposal():
    class _DSpark:

        def __init__(self):
            self.updates = []

        def resolve_cache_slots(self, request_ids, live_request_ids):
            assert torch.equal(request_ids.cpu(), torch.tensor([10, 20]))
            assert torch.equal(live_request_ids.cpu(), torch.tensor([10, 20]))
            return torch.tensor([2, 0], device=request_ids.device)

        def update_main_cache(
            self,
            main_hidden,
            rotary_pos_emb,
            *,
            start_pos,
            cache_slots,
            inference_context,
        ):
            self.updates.append((main_hidden.clone(), start_pos.clone(), cache_slots.clone()))

    model = DeepseekV41GPTModel.__new__(DeepseekV41GPTModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(sequence_parallel=False, dspark_block_size=3)
    model.dspark = _DSpark()
    captured = torch.arange(5 * 4, dtype=torch.float32).view(5, 1, 4)
    model.get_dspark_main_hidden = lambda: captured
    model._dspark_rotary_for_positions = lambda positions: positions.float()
    proposal_args = {}

    def forward_dspark(main_hidden, input_ids, **kwargs):
        proposal_args.update(main_hidden=main_hidden, input_ids=input_ids, **kwargs)
        output_ids = torch.tensor([[31, 32, 33, 34], [41, 42, 43, 44]])
        return output_ids, None, None

    model.forward_dspark = forward_dspark
    context = SimpleNamespace(
        total_request_count=2,
        paused_request_count=0,
        num_decode_requests=1,
        request_query_lengths=torch.tensor([3, 2], dtype=torch.int32),
        request_ids=torch.tensor([10, 20], dtype=torch.int32),
        token_to_position_in_request=torch.tensor([5, 6, 7, 0, 1], dtype=torch.int32),
        using_cuda_graph_this_step=lambda: False,
    )

    proposals = model.compute_dspark_speculative_tokens(
        next_token_ids=torch.tensor([31, 41]),
        accepted_token_counts=torch.tensor([1, 0]),
        last_accepted_seq_indices=torch.tensor([1, 4]),
        num_speculative_tokens=2,
        inference_context=context,
        sample_fn=lambda logits: logits.argmax(dim=-1),
    )

    assert len(model.dspark.updates) == 2
    torch.testing.assert_close(model.dspark.updates[0][0], captured[:2])
    torch.testing.assert_close(model.dspark.updates[1][0], captured[3:5])
    assert model.dspark.updates[0][1].item() == 5
    assert model.dspark.updates[1][1].item() == 0
    assert torch.equal(proposal_args['start_pos'], torch.tensor([6, 1]))
    torch.testing.assert_close(proposal_args['main_hidden'], captured[[1, 4]].transpose(0, 1))
    assert torch.equal(proposals, torch.tensor([[32, 42], [33, 43]]))


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
    controller = TextGenerationController.__new__(TextGenerationController)
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


def test_dspark_word_embeddings_resolver_prefers_base_then_dedicated():
    resolver = DeepseekV41GPTModel._dspark_word_embeddings
    model = DeepseekV41GPTModel.__new__(DeepseekV41GPTModel)
    torch.nn.Module.__init__(model)
    # Neither the base embedding nor a dedicated DSpark embedding is present.
    with pytest.raises(RuntimeError):
        resolver(model)
    # A PP>1 last stage falls back to the dedicated DSpark embedding.
    dedicated = object()
    model.dspark_word_embeddings = dedicated
    assert resolver(model) is dedicated
    # When the base embedding is colocated it always takes priority.
    base = object()
    model.embedding = SimpleNamespace(word_embeddings=base)
    assert resolver(model) is base


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
        layer_number=2,
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
        key_projection=SimpleNamespace(
            weight=torch.nn.Parameter(torch.empty(num_streams * hidden_size, memory_dim))),
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
