from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass

CED_CHUNK_SIZE = 8192
CED_REPLAY_SIZE = 128
CED_FIRST_REPLAY_LAYER = 20


@dataclass(frozen=True)
class CEDPrefillLayout:
    final_chunk: bool
    prompt_length: int

    @property
    def token_slice(self) -> slice:
        return slice(CED_CHUNK_SIZE - CED_REPLAY_SIZE, CED_CHUNK_SIZE)


@dataclass(frozen=True)
class CEDPrefillPlan:
    layout: CEDPrefillLayout
    tail_context: Callable[[], AbstractContextManager]


def plan_aligned_ced_prefill(
    prompt_lengths: Sequence[int],
    computed_tokens: Sequence[int],
    scheduled_tokens: Sequence[int],
    is_prefilling: Sequence[bool],
) -> CEDPrefillLayout | None:
    """Plan the restricted, single-8192-token-chunk CED prototype."""
    if not any(is_prefilling):
        return None
    if not (
        len(prompt_lengths)
        == len(computed_tokens)
        == len(scheduled_tokens)
        == len(is_prefilling)
        == 1
    ):
        raise ValueError("CED prototype requires one prefill request and no decode")
    prompt_length = int(prompt_lengths[0])
    computed = int(computed_tokens[0])
    scheduled = int(scheduled_tokens[0])
    if prompt_length < CED_CHUNK_SIZE or prompt_length % CED_CHUNK_SIZE:
        raise ValueError("CED prompt length must be a positive multiple of 8192")
    if computed < 0 or computed % CED_CHUNK_SIZE:
        raise ValueError("CED chunk start must be aligned to 8192")
    if scheduled != CED_CHUNK_SIZE or computed + scheduled > prompt_length:
        raise ValueError("CED prototype requires complete 8192-token chunks")
    return CEDPrefillLayout(
        final_chunk=computed + scheduled == prompt_length,
        prompt_length=prompt_length,
    )
