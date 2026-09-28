# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout plumbing for DeepSeek V4.1 mega attention.

The kernel's Q and O layouts are produced by permuting ``wq_b`` rows and
``wo_a`` columns once at load, so what has to hold is that the permuted GEMMs
agree with a reference that permutes the activation instead.
"""

import pytest
import torch

from vllm.models.deepseek_v41.common.ops.fused_layout import (
    WV_GROUP_SIZE,
    o_fused_chunk_permutation,
    o_fused_permutation,
    permute_q_to_fused,
    permute_wo_a_,
    permute_wq_b_,
    q_fused_permutation,
)

HEAD_DIM = 512


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bounded_replay", [False, True])
def test_swa_indices_stay_inside_gathered_region(bounded_replay):
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        combine_topk_swa_indices,
    )

    device = "cuda"
    query_length, sequence_length, window = 128, 32768, 128
    gather_length = 128 if bounded_replay else 255
    topk = torch.tensor([0, 1], device=device, dtype=torch.int32).repeat(128, 1)
    indices, lengths = combine_topk_swa_indices(
        topk,
        torch.tensor([0, query_length], device=device, dtype=torch.int32),
        torch.tensor([sequence_length], device=device, dtype=torch.int32),
        torch.tensor([gather_length], device=device, dtype=torch.int32),
        window,
        1,
        2,
        sequence_length + gather_length,
        sequence_length,
    )
    indices, lengths = indices.cpu(), lengths.cpu()
    gather_start = sequence_length - gather_length
    for query_index in range(query_length):
        position = sequence_length - query_length + query_index
        swa_start = max(position - window + 1, gather_start)
        expected = [0, 1] + list(
            range(
                sequence_length + swa_start - gather_start,
                sequence_length + position - gather_start + 1,
            )
        )
        assert lengths[query_index].item() == len(expected)
        assert indices[query_index, : len(expected)].tolist() == expected
        assert (indices[query_index, len(expected) :] == -1).all()


def test_permuted_wq_b_gemm_matches_permuted_activation():
    """Permuting wq_b's rows makes the Q GEMM emit the fused layout directly."""
    torch.manual_seed(0)
    num_heads, q_lora_rank, num_tokens = 16, 64, 5
    weight = torch.randn(num_heads * HEAD_DIM, q_lora_rank)
    # One scale row per weight row, as an MXFP8 wq_b shard carries.
    scale = torch.randint(
        0, 256, (num_heads * HEAD_DIM, q_lora_rank // 32), dtype=torch.uint8
    )
    qr = torch.randn(num_tokens, q_lora_rank)
    orig_weight, orig_scale = weight.clone(), scale.clone()

    standard = (qr @ weight.T).view(num_tokens, num_heads, HEAD_DIM)
    expected = permute_q_to_fused(standard)

    permute_wq_b_(weight, scale, num_heads)
    fused = (qr @ weight.T).view(num_tokens, num_heads, HEAD_DIM)
    torch.testing.assert_close(fused, expected)
    # The scale has to follow its row, or dequant reads another row's scale.
    perm = q_fused_permutation(num_heads, HEAD_DIM)
    torch.testing.assert_close(weight, orig_weight[perm])
    torch.testing.assert_close(scale, orig_scale[perm])


def test_permuted_wo_a_consumes_fused_output():
    """Permuting wo_a's input columns lets it read the kernel's O layout."""
    torch.manual_seed(1)
    out_features, num_tokens = 32, 4
    in_features = WV_GROUP_SIZE * HEAD_DIM
    weight = torch.randn(out_features, in_features)
    scale = torch.randint(0, 256, (out_features, in_features // 32), dtype=torch.uint8)
    standard_o = torch.randn(num_tokens, in_features)
    orig_weight, orig_scale = weight.clone(), scale.clone()

    expected = standard_o @ weight.T
    # The kernel emits the same values in the fused chunk order.
    perm = o_fused_permutation(WV_GROUP_SIZE, HEAD_DIM)
    fused_o = standard_o[:, perm]

    permute_wo_a_(weight, scale, WV_GROUP_SIZE)
    torch.testing.assert_close(weight, orig_weight[:, perm])
    torch.testing.assert_close(
        scale, orig_scale[:, o_fused_chunk_permutation(WV_GROUP_SIZE, HEAD_DIM)]
    )
    # Unlike wq_b, this permutation sits inside the 4096-term reduction, so the
    # summation order changes and the result differs in the last few ulps.
    torch.testing.assert_close(fused_o @ weight.T, expected, rtol=1e-3, atol=1e-3)
