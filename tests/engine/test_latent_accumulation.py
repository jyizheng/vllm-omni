"""Regression tests for latent snapshot accumulation (REPLACE semantics).

Latent payloads are cumulative snapshots: every emission carries the full
accumulated hidden states. A request that finishes on a stop token emits
twice (the EOS step and the final flush); concatenating the two snapshots
double-counts and breaks the downstream prompt+generated length invariant.

Note: Uses importlib to load modules directly, bypassing the vllm_omni
package __init__ which requires the vllm base package.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "vllm_omni" / "outputs"


def _load_module(name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_om_mod = _load_module(
    "vllm_omni.outputs.output_modality",
    _OUTPUTS_DIR / "output_modality.py",
)
_load_module(
    "vllm_omni.outputs.utils",
    _OUTPUTS_DIR / "utils.py",
)
_mm_mod = _load_module(
    "vllm_omni.outputs.mm_outputs",
    _OUTPUTS_DIR / "mm_outputs.py",
)

OutputModality = _om_mod.OutputModality
TensorAccumulationStrategy = _om_mod.TensorAccumulationStrategy
get_accumulation_strategy = _om_mod.get_accumulation_strategy
MultimodalPayload = _mm_mod.MultimodalPayload


def test_latent_strategy_is_replace():
    assert get_accumulation_strategy(OutputModality.LATENT) is TensorAccumulationStrategy.REPLACE
    # Audio and image chunked accumulation is unchanged.
    assert get_accumulation_strategy(OutputModality.AUDIO) is TensorAccumulationStrategy.CONCAT_LAST
    assert get_accumulation_strategy(OutputModality.IMAGE) is TensorAccumulationStrategy.CONCAT_DIM0


def test_duplicated_terminal_latent_snapshot_keeps_latest():
    # Simulate a stop-token finish: an EOS-step snapshot (N rows) followed by
    # the final flush snapshot (N+2 rows). The consolidated payload must be
    # exactly the final snapshot, not the concatenation.
    eos_snapshot = MultimodalPayload.from_raw({"latent": torch.randn(2573, 8)}, "latent")
    final_snapshot_tensor = torch.randn(2575, 8)
    final_snapshot = MultimodalPayload.from_raw({"latent": final_snapshot_tensor.clone()}, "latent")

    accumulated = MultimodalPayload()
    accumulated = accumulated.merged_with(eos_snapshot)
    accumulated = accumulated.merged_with(final_snapshot)
    accumulated.consolidate_tensors(get_accumulation_strategy(OutputModality.LATENT))

    latent = accumulated.tensors["latent"]
    assert latent.shape == (2575, 8)
    assert torch.equal(latent, final_snapshot_tensor)


def test_single_latent_snapshot_passthrough():
    snapshot = torch.randn(4199, 8)
    accumulated = MultimodalPayload()
    accumulated = accumulated.merged_with(MultimodalPayload.from_raw({"latent": snapshot.clone()}, "latent"))
    accumulated.consolidate_tensors(get_accumulation_strategy(OutputModality.LATENT))
    assert torch.equal(accumulated.tensors["latent"], snapshot)
