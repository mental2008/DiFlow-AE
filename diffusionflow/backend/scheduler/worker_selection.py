"""
Optional C++ worker-selection helper for the dynamic scheduler.

The coordinator still owns Python request/workflow state. This module only
loads a small C++ scoring kernel for the hot "scan all workers" loop.
"""

import logging
import os
from pathlib import Path
from typing import Sequence, Tuple

logger = logging.getLogger(__name__)

_EXTENSION = None
_EXTENSION_LOAD_ERROR = None


def _load_extension():
    global _EXTENSION, _EXTENSION_LOAD_ERROR

    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_LOAD_ERROR is not None:
        raise _EXTENSION_LOAD_ERROR

    try:
        from torch.utils.cpp_extension import load

        source_path = Path(__file__).with_name("worker_selection_exp.cpp")
        _EXTENSION = load(
            name="diffusionflow_worker_selection_exp",
            sources=[str(source_path)],
            extra_cflags=["-O3", "-std=c++17"],
            verbose=os.environ.get("DIFFUSIONFLOW_WORKER_SELECTION_CPP_VERBOSE")
            == "1",
        )
        return _EXTENSION
    except Exception as exc:  # pragma: no cover - depends on local compiler setup
        _EXTENSION_LOAD_ERROR = exc
        raise


def is_worker_selection_cpp_available() -> bool:
    """Return whether the C++ extension can be loaded in this environment."""
    try:
        _load_extension()
        return True
    except Exception as exc:
        logger.warning("C++ worker-selection extension is unavailable: %s", exc)
        return False


def select_worker_cpp(
    worker_ranks: Sequence[int],
    worker_host_ids: Sequence[int],
    queue_latencies: Sequence[float],
    model_loaded: Sequence[bool],
    worker_latency_threshold: float,
    loading_latency: float,
    tensor_sources: Sequence[Sequence[Tuple[int, int, int]]],
    intra_block_sizes: Sequence[int],
    intra_fetch_overheads_us: Sequence[float],
    inter_block_sizes: Sequence[int],
    inter_fetch_overheads_us: Sequence[float],
) -> Tuple[int, float]:
    """
    Select the minimum-latency worker using the C++ scoring loop.

    Args:
        tensor_sources: One entry per input tensor. Each tensor entry contains
            `(src_worker_rank, src_host_id, tensor_size_bytes)` tuples in the
            same order the Python scheduler would inspect them.

    Returns:
        `(selected_worker_rank, execution_latency_on_selected_worker)`.
        `selected_worker_rank == -1` means no worker passed the latency threshold.
    """
    extension = _load_extension()
    selected_worker, execution_latency = extension.select_worker(
        list(worker_ranks),
        list(worker_host_ids),
        list(queue_latencies),
        list(model_loaded),
        float(worker_latency_threshold),
        float(loading_latency),
        [[tuple(source) for source in tensor_source] for tensor_source in tensor_sources],
        list(intra_block_sizes),
        list(intra_fetch_overheads_us),
        list(inter_block_sizes),
        list(inter_fetch_overheads_us),
    )
    return int(selected_worker), float(execution_latency)
