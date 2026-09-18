"""Regression tests for latent accumulation with terminal snapshots.

Latent emissions are per-step chunks, except that a stop-token finish
additionally delivers the full cumulative snapshot. The accumulator must
treat a payload at least as long as everything accumulated so far as a
superseding snapshot; appending it double-counts the hidden states and
breaks the downstream prompt+generated length invariant.
"""

import pytest
import torch

from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.outputs.output_modality import (
    OutputModality,
    TensorAccumulationStrategy,
    get_accumulation_strategy,
)
from vllm_omni.outputs.output_processor import OmniRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeState:
    """Carries just the attributes add_multimodal_tensor needs."""

    def __init__(self):
        self.mm_type = None
        self.mm_accumulated = MultimodalPayload()

    def add(self, tensor):
        OmniRequestState.add_multimodal_tensor(self, {"latent": tensor}, "latent")

    def consolidated(self):
        self.mm_accumulated.consolidate_tensors(get_accumulation_strategy(OutputModality.LATENT))
        return self.mm_accumulated.tensors["latent"]


def test_latent_strategy_is_concat_dim0():
    assert get_accumulation_strategy(OutputModality.LATENT) is TensorAccumulationStrategy.CONCAT_DIM0


def test_per_step_chunks_concatenate():
    state = _FakeState()
    chunks = [torch.randn(1, 8) for _ in range(5)]
    prefill = torch.randn(40, 8)
    state.add(prefill)
    for c in chunks:
        state.add(c)
    out = state.consolidated()
    assert out.shape == (45, 8)
    assert torch.equal(out, torch.cat([prefill, *chunks], dim=0))


def test_terminal_snapshot_supersedes_chunks():
    # Stop-token finish: per-step chunks accumulate, then the final flush
    # delivers the full cumulative snapshot. The snapshot must replace the
    # chunks, not be appended after them.
    state = _FakeState()
    state.add(torch.randn(40, 8))  # prefill hidden states
    for _ in range(10):
        state.add(torch.randn(1, 8))  # decode steps
    final_snapshot = torch.randn(52, 8)  # 40 prompt + 12 generated
    state.add(final_snapshot.clone())
    out = state.consolidated()
    assert out.shape == (52, 8)
    assert torch.equal(out, final_snapshot)


def test_single_snapshot_passthrough():
    state = _FakeState()
    snapshot = torch.randn(4199, 8)
    state.add(snapshot.clone())
    assert torch.equal(state.consolidated(), snapshot)
