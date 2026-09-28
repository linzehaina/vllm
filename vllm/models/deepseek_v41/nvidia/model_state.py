# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_dp_group
from vllm.forward_context import set_forward_context
from vllm.models.deepseek_v41.ced_prefill import (
    CED_CHUNK_SIZE,
    CED_REPLAY_SIZE,
    CEDPrefillPlan,
    plan_aligned_ced_prefill,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


@triton.jit
def _gather_lookback_kernel(
    lookback_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    num_reqs,
    DEPTH: tl.constexpr,
    BLOCK_DEPTH: tl.constexpr,
):
    # One program per lookback row; rows past the batch are filled with -1.
    batch_idx = tl.program_id(0)
    in_batch = batch_idx < num_reqs
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx, mask=in_batch, other=0)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)

    offs = tl.arange(0, BLOCK_DEPTH)
    pos = num_computed - 1 - offs
    valid = in_batch & (offs < DEPTH) & (pos >= 0)
    ids = tl.load(
        all_token_ids_ptr + req_state_idx * all_token_ids_stride + pos,
        mask=valid,
        other=-1,
    )
    tl.store(lookback_ptr + batch_idx * DEPTH + offs, ids, mask=offs < DEPTH)


class DeepseekV41ModelState(DefaultModelState):
    """DefaultModelState plus the engram lookback window.

    The engram n-gram hash needs the ids of the ``depth`` tokens preceding
    each request's chunk start (see ``common/engram.py``). The runner keeps
    the full token history on device, so the window is gathered there every
    step: exact for prompt and generated tokens alike, whatever instance
    produced their KV.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        self.ced_prefill_enabled = bool(
            vllm_config.additional_config.get("dsv41_ced_prefill", False)
        )
        self.ced_prefill_plan: CEDPrefillPlan | None = None
        self.ced_input_batch: InputBatch | None = None
        self.ced_prefill_stats = {
            "full_chunks": 0,
            "replay_chunks": 0,
            "full_tokens_per_layer": 0,
            "replay_tokens_per_layer": 0,
        }
        if self.ced_prefill_enabled:
            parallel = vllm_config.parallel_config
            compilation = vllm_config.compilation_config
            text_config = vllm_config.model_config.hf_text_config
            backend = vllm_config.attention_config.backend
            if (
                parallel.tensor_parallel_size != 1
                or parallel.pipeline_parallel_size != 1
                or parallel.prefill_context_parallel_size != 1
                or parallel.decode_context_parallel_size != 1
                or parallel.enable_dbo
                or vllm_config.speculative_config is not None
                or vllm_config.lora_config is not None
                or vllm_config.kv_transfer_config is not None
                or vllm_config.cache_config.enable_prefix_caching
                or vllm_config.scheduler_config.max_num_batched_tokens != CED_CHUNK_SIZE
                or vllm_config.scheduler_config.async_scheduling
                or compilation.mode != 0
                or compilation.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
                or text_config.num_hidden_layers != 40
                or text_config.sliding_window != CED_REPLAY_SIZE
                or text_config.kv_source_layer_ids[-1] != 20
                or text_config.candidate_source_layer_id != 20
                or text_config.compress_ratios[20] != 1
                or 20 not in text_config.index_source_layer_ids
                or str(backend).split(".")[-1] != "FLASHMLA_MEGA_ATTN_DSV41"
                or vllm_config.kernel_config.moe_backend != "deep_gemm_mega_moe"
            ):
                raise ValueError(
                    "CED prototype requires TP1/PP1, 8192-token batches, mode=0, "
                    "synchronous scheduling, FULL_DECODE_ONLY graphs, "
                    "native V4.1 Flash mega backends, "
                    "no prefix cache/MTP/CP/DBO/LoRA/KV transfer"
                )
        depth = model.token_lookback_depth
        self.lookback_token_ids: torch.Tensor | None = None
        if depth > 0:
            # Persistent so a captured graph can read it on replay.
            self.lookback_token_ids = torch.full(
                (self.max_num_reqs, depth), -1, dtype=torch.int32, device=device
            )

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        if self.ced_prefill_enabled and not self.is_warming_up:
            params = new_req_data.sampling_params
            if (
                new_req_data.mm_features
                or new_req_data.prompt_embeds is not None
                or (params is not None and params.prompt_logprobs is not None)
                or new_req_data.num_computed_tokens != 0
            ):
                raise ValueError(
                    "CED prototype requires fresh text-token requests without "
                    "prompt logprobs or cache reuse"
                )
        super().add_request(req_index, new_req_data)

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
    ) -> dict[str, Any]:
        self.ced_prefill_plan = None
        self.ced_input_batch = input_batch
        if (
            self.ced_prefill_enabled
            and not self.is_warming_up
            and not for_capture
            and cudagraph_mode != CUDAGraphMode.FULL
        ):
            layout = None
            error = None
            try:
                layout = plan_aligned_ced_prefill(
                    input_batch.prefill_len_np,
                    input_batch.num_computed_tokens_np,
                    input_batch.num_scheduled_tokens,
                    input_batch.is_prefilling_np,
                )
                if layout is not None and (
                    cudagraph_mode != CUDAGraphMode.NONE
                    or input_batch.num_tokens_after_padding != CED_CHUNK_SIZE
                ):
                    raise ValueError("CED prototype requires unpadded eager prefill")
            except ValueError as exc:
                error = str(exc)
            phase = -1 if error else (1 + int(layout.final_chunk) if layout else 0)
            signature = torch.tensor([phase], dtype=torch.int32, device="cpu")
            dp_size = self.vllm_config.parallel_config.data_parallel_size
            signatures = [signature]
            if dp_size > 1:
                signatures = [torch.empty_like(signature) for _ in range(dp_size)]
                dist.all_gather(signatures, signature, group=get_dp_group().cpu_group)
            phases = [int(value.item()) for value in signatures]
            if -1 in phases or len(set(phases)) != 1:
                raise ValueError(
                    f"CED prototype needs aligned prefill phases on every DP rank; "
                    f"phases={phases}, local_error={error}"
                )
            if layout is not None:
                self.ced_prefill_stats["full_chunks"] += 1
                self.ced_prefill_stats["full_tokens_per_layer"] += CED_CHUNK_SIZE
                if layout.final_chunk:
                    self.ced_prefill_stats["replay_chunks"] += 1
                    self.ced_prefill_stats["replay_tokens_per_layer"] += CED_REPLAY_SIZE

                def tail_context():
                    token_slice = layout.token_slice
                    query_start_cpu = np.array([0, CED_REPLAY_SIZE], dtype=np.int32)
                    tail_batch = replace(
                        input_batch,
                        num_tokens=CED_REPLAY_SIZE,
                        num_tokens_after_padding=CED_REPLAY_SIZE,
                        num_scheduled_tokens=np.array(
                            [CED_REPLAY_SIZE], dtype=np.int32
                        ),
                        query_start_loc=torch.from_numpy(query_start_cpu).to(
                            self.device
                        ),
                        query_start_loc_np=query_start_cpu,
                        num_computed_tokens_np=np.array(
                            [layout.prompt_length - CED_REPLAY_SIZE], dtype=np.int32
                        ),
                        input_ids=input_batch.input_ids[token_slice],
                        positions=input_batch.positions[token_slice],
                        is_padding=input_batch.is_padding[token_slice],
                        max_query_len=CED_REPLAY_SIZE,
                        fast_prefill=None,
                    )
                    tail_slots = slot_mappings[:, token_slice]
                    metadata = super(DeepseekV41ModelState, self).prepare_attn(
                        tail_batch,
                        CUDAGraphMode.NONE,
                        block_tables,
                        tail_slots,
                        attn_groups,
                        kv_cache_config,
                    )
                    for value in metadata.values():
                        if isinstance(value, DeepseekSparseSWAMetadata):
                            value.prefill_swa_bounded_replay = True
                            assert value.prefill_gather_lens is not None
                            value.prefill_gather_lens.fill_(CED_REPLAY_SIZE)
                    return set_forward_context(
                        metadata,
                        self.vllm_config,
                        num_tokens=CED_REPLAY_SIZE,
                        num_tokens_across_dp=torch.full(
                            (dp_size,), CED_REPLAY_SIZE, dtype=torch.int32, device="cpu"
                        ),
                        slot_mapping=build_slot_mappings_by_layer(
                            tail_slots, kv_cache_config
                        ),
                        is_padding=tail_batch.is_padding,
                    )

                self.ced_prefill_plan = CEDPrefillPlan(layout, tail_context)
        return super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
            ubatch_idx,
        )

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if self.ced_input_batch is input_batch and self.ced_prefill_plan is not None:
            model_inputs["ced_prefill_plan"] = self.ced_prefill_plan
        window = self.lookback_token_ids
        if window is None:
            return model_inputs
        all_token_ids = req_states.all_token_ids.gpu
        depth = window.shape[1]
        _gather_lookback_kernel[(window.shape[0],)](
            window,
            input_batch.idx_mapping,
            req_states.num_computed_tokens.gpu,
            all_token_ids,
            all_token_ids.stride(0),
            input_batch.idx_mapping.shape[0],
            DEPTH=depth,
            BLOCK_DEPTH=triton.next_power_of_2(depth),
        )
        model_inputs["lookback_token_ids"] = window
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if self.lookback_token_ids is not None:
            # The captured graph reads this buffer; replays refill it in place.
            self.lookback_token_ids.fill_(-1)
            model_inputs["lookback_token_ids"] = self.lookback_token_ids
        return model_inputs
