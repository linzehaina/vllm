import pytest

from vllm.models.deepseek_v41.ced_prefill import plan_aligned_ced_prefill


@pytest.mark.parametrize("prompt_length", [8192, 32768, 65536, 524288])
def test_only_final_chunk_selects_request_tail(prompt_length):
    replay_tokens = 0
    for computed in range(0, prompt_length, 8192):
        plan = plan_aligned_ced_prefill([prompt_length], [computed], [8192], [True])
        assert plan is not None
        if plan.final_chunk:
            assert computed + plan.token_slice.start == prompt_length - 128
            assert computed + plan.token_slice.stop == prompt_length
            replay_tokens += plan.token_slice.stop - plan.token_slice.start
        else:
            assert computed + 8192 < prompt_length
    assert replay_tokens == 128


@pytest.mark.parametrize(
    "lengths,computed,scheduled,prefilling",
    [
        ([32767], [0], [8192], [True]),
        ([0], [0], [8192], [True]),
        ([32768], [1], [8192], [True]),
        ([32768], [0], [8191], [True]),
        ([32768], [32768], [8192], [True]),
        ([32768, 32768], [32768, 0], [1, 8191], [False, True]),
    ],
)
def test_rejects_unsupported_prefill_instead_of_silent_fallback(
    lengths, computed, scheduled, prefilling
):
    with pytest.raises(ValueError, match="CED"):
        plan_aligned_ced_prefill(lengths, computed, scheduled, prefilling)


def test_decode_does_not_create_replay_plan():
    assert plan_aligned_ced_prefill([32768], [32770], [1], [False]) is None
    assert plan_aligned_ced_prefill([], [], [], []) is None


def test_internal_warmup_is_not_subject_to_serving_alignment(monkeypatch):
    from types import SimpleNamespace

    import torch

    from vllm.config.compilation import CUDAGraphMode
    from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    state = DeepseekV41ModelState.__new__(DeepseekV41ModelState)
    state.ced_prefill_enabled = True
    state.is_warming_up = True
    state.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1)
    )
    batch = SimpleNamespace(
        prefill_len_np=[2],
        num_computed_tokens_np=[0],
        num_scheduled_tokens=[2],
        is_prefilling_np=[True],
        num_tokens_after_padding=2,
    )
    metadata = {}
    monkeypatch.setattr(
        DefaultModelState, "prepare_attn", lambda *args, **kwargs: metadata
    )
    args = (batch, CUDAGraphMode.NONE, (), torch.empty(0), [], None)
    assert state.prepare_attn(*args) is metadata
    assert state.ced_prefill_plan is None
    state.is_warming_up = False
    with pytest.raises(ValueError, match="positive multiple of 8192"):
        state.prepare_attn(*args)


def test_replay_workspace_keeps_global_context_and_truncates_swa():
    import torch

    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

    metadata = DeepseekSparseSWAMetadata(
        block_table=torch.empty(1, 1, dtype=torch.int32),
        slot_mapping=torch.empty(128, dtype=torch.int64),
        block_size=128,
        num_prefills=1,
        prefill_seq_lens_cpu=torch.tensor([32768]),
        prefill_query_lens_cpu=torch.tensor([128]),
        prefill_window_size=128,
        prefill_max_model_len=32768,
        prefill_max_num_batched_tokens=8192,
    )
    normal = metadata.get_prefill_chunk_plan(1, 1, has_compressed=True)
    metadata.prefill_swa_bounded_replay = True
    replay = metadata.get_prefill_chunk_plan(1, 1, has_compressed=True)
    assert normal == [(0, 1, 32768, 32768 + 255)]
    assert replay == [(0, 1, 32768, 32768 + 128)]


@pytest.mark.parametrize("final_chunk", [False, True])
def test_model_replay_preserves_tail_states_and_restores_logits_rows(
    monkeypatch, final_chunk
):
    from contextlib import contextmanager
    from types import SimpleNamespace

    import torch

    from vllm.models.deepseek_v41.ced_prefill import CEDPrefillLayout, CEDPrefillPlan
    from vllm.models.deepseek_v41.nvidia import model as model_module

    calls = []
    in_tail = []
    model = model_module.DeepseekV4Model.__new__(model_module.DeepseekV4Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=2)
    model.use_mega_moe = False
    model.use_sequence_parallel = False
    model.engram_hash = None
    model.start_layer, model.end_layer = 0, 40
    model.aux_hidden_state_layers = set()
    model._mtp_hidden_buffer = None
    model.norm = torch.nn.Identity()
    model.embed_input_ids = lambda ids: ids.float()[:, None].expand(-1, 2).clone()
    model.topk_indices_buffer = torch.arange(8192).view(-1, 1)
    model.candidate_block_buffer = model.topk_indices_buffer.clone() + 100
    monkeypatch.setattr(
        model_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(model_module, "mhc_post_tilelang", lambda hidden, *args: hidden)
    monkeypatch.setattr(model_module, "hc_collapse_triton", lambda hidden, mix: hidden)

    class Layer(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index

        def forward(
            self, hidden, positions, ids, pre, post, mix, residual, *args, **kwargs
        ):
            calls.append((self.index, len(hidden)))
            if self.index == 20:
                assert kwargs["ced_prefill_plan"] is plan
                if final_chunk:
                    token_slice = plan.layout.token_slice
                    hidden = hidden[token_slice]
                    kwargs["ced_context"].enter_context(plan.tail_context())
                    model.topk_indices_buffer[:128, 0] = torch.arange(128) + 1000
                    model.candidate_block_buffer[:128, 0] = torch.arange(128) + 2000
            if self.index >= 21:
                assert in_tail
                assert positions.tolist() == list(range(32640, 32768))
                for state in (pre, post, mix, residual):
                    assert state.shape == (128, 2)
                torch.testing.assert_close(pre, hidden + 4)
            if self.index == 21:
                assert model.topk_indices_buffer[:128, 0].tolist() == list(
                    range(1000, 1128)
                )
                assert model.candidate_block_buffer[:128, 0].tolist() == list(
                    range(2000, 2128)
                )
            return hidden + 1, hidden + 2, hidden + 3, hidden + 4, hidden + 5, None

    model.layers = torch.nn.ModuleList([Layer(index) for index in range(40)])

    @contextmanager
    def tail_context():
        assert calls[-1] == (20, 8192)
        in_tail.append(True)
        try:
            yield
        finally:
            in_tail.pop()

    plan = CEDPrefillPlan(CEDPrefillLayout(final_chunk, 32768), tail_context)
    output = model(
        torch.arange(8192),
        torch.arange(24576, 32768),
        None,
        ced_prefill_plan=plan,
    )
    expected_calls = [(index, 8192) for index in range(21)]
    if final_chunk:
        expected_calls += [(index, 128) for index in range(21, 40)]
        torch.testing.assert_close(
            output[-128:, 0], torch.arange(8064, 8192).float() + 40
        )
        assert not output[:-128].count_nonzero()
    else:
        assert not output.count_nonzero()
    assert output.shape == (8192, 2)
    assert calls == expected_calls
    assert not in_tail


@pytest.mark.parametrize("final_chunk", [False, True])
def test_source_layer_builds_full_normalized_kv_before_tail_queries(
    monkeypatch, final_chunk
):
    from contextlib import ExitStack, contextmanager
    from types import SimpleNamespace

    import torch

    from vllm.models.deepseek_v41.ced_prefill import CEDPrefillLayout, CEDPrefillPlan
    from vllm.models.deepseek_v41.nvidia import model as model_module

    layer = model_module.DeepseekV4DecoderLayer.__new__(
        model_module.DeepseekV4DecoderLayer
    )
    torch.nn.Module.__init__(layer)
    layer.use_sequence_parallel = False
    layer.engram = None
    for name in (
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
        "hc_ffn_base",
        "rms_norm_eps",
        "hc_eps",
        "hc_post_alpha",
        "hc_sinkhorn_iters",
    ):
        setattr(layer, name, None)
    layer.attn_norm = layer.ffn_norm = SimpleNamespace(weight=None, variance_epsilon=0)
    hidden = torch.arange(8192).float().view(-1, 1).expand(-1, 2)
    positions = torch.arange(24576, 32768)
    calls = []
    in_tail = []

    def prepare_attention(hidden, residual, post, mix, *args, pre_mix, **kwargs):
        if not in_tail:
            return residual + 10, post + 10, mix + 10, hidden + 100, pre_mix + 10, None
        for value, offset in ((residual, 11), (post, 12), (mix, 13), (pre_mix, 14)):
            torch.testing.assert_close(value, original_tail + offset)
        return residual, post, mix, hidden, pre_mix, None

    monkeypatch.setattr(model_module, "mhc_shifted_post_pre", prepare_attention)

    class Attention(torch.nn.Module):
        def build_global_kv(self, actual_positions, normalized):
            assert not in_tail
            torch.testing.assert_close(actual_positions, positions)
            torch.testing.assert_close(normalized, hidden + 100)
            calls.append(("kv", len(normalized)))

        def forward(self, actual_positions, normalized, scaling, *, kv_cache_prebuilt):
            assert kv_cache_prebuilt and in_tail
            torch.testing.assert_close(actual_positions, positions[-128:])
            torch.testing.assert_close(normalized, original_tail + 100)
            calls.append(("query", len(normalized)))
            return normalized + 1

    def feed_forward(hidden, ids):
        assert in_tail
        assert ids.tolist() == list(range(8064, 8192))
        calls.append(("ffn", len(hidden)))
        return hidden

    layer.attn = Attention()
    layer.ffn = feed_forward
    original_tail = hidden[-128:]

    @contextmanager
    def tail_context():
        assert calls == [("kv", 8192)]
        in_tail.append(True)
        try:
            yield
        finally:
            in_tail.pop()

    plan = CEDPrefillPlan(CEDPrefillLayout(final_chunk, 32768), tail_context)
    with ExitStack() as context_stack:
        output = layer(
            hidden,
            positions,
            torch.arange(8192),
            hidden + 4,
            hidden + 2,
            hidden + 3,
            hidden + 1,
            ced_prefill_plan=plan,
            ced_context=context_stack,
        )
        if final_chunk:
            assert in_tail
            torch.testing.assert_close(output[0], original_tail + 101)
        else:
            torch.testing.assert_close(output[0], hidden + 100)
    expected = [("kv", 8192)] + ([("query", 128), ("ffn", 128)] if final_chunk else [])
    assert calls == expected
    assert not in_tail
