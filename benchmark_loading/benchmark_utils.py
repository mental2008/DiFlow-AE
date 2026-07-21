import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

from diffusionflow.operators.utils import get_op


DEFAULT_RESULTS_DIR = "benchmark_loading/results"


@dataclass(frozen=True)
class LoadingBenchmarkCase:
    model_name: str
    model_path: Optional[str]
    suite: str


def convert_gpu_name(gpu_name: str) -> str:
    """Convert vendor-specific GPU names into stable path-safe identifiers."""
    normalized_name = gpu_name.strip().lower()
    vendor_aliases = {
        "nvidia corporation": "nvidia",
        "advanced micro devices, inc.": "amd",
        "advanced micro devices": "amd",
        "amd/ati": "amd",
        "intel corporation": "intel",
        "intel(r)": "intel",
    }

    for vendor_name, canonical_vendor_name in vendor_aliases.items():
        normalized_name = normalized_name.replace(vendor_name, canonical_vendor_name)

    normalized_name = normalized_name.replace("(r)", "").replace("(tm)", "")
    normalized_name = normalized_name.replace("\u00ae", "").replace("\u2122", "")
    normalized_name = re.sub(r"[^a-z0-9]+", "_", normalized_name).strip("_")
    return normalized_name or "unknown_gpu"


def get_gpu_info() -> Tuple[str, int, int]:
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")

    gpu_name = torch.cuda.get_device_name()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    gpu_count = torch.cuda.device_count()
    return gpu_name, gpu_memory, gpu_count


def get_result_path(model_name: str, results_dir: str = DEFAULT_RESULTS_DIR) -> str:
    return os.path.join(results_dir, f"{model_name}.json")


def load_existing_result(
    model_name: str, results_dir: str = DEFAULT_RESULTS_DIR
) -> Optional[Dict[str, Any]]:
    result_path = get_result_path(model_name, results_dir)
    if not os.path.exists(result_path):
        return None

    try:
        with open(result_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Ignoring invalid loading benchmark result at {result_path}")
        return None


def result_matches_current_gpu(result: Dict[str, Any]) -> bool:
    gpu_type, gpu_memory, _ = get_gpu_info()
    result_gpu_type = result.get("gpu_type_normalized", result.get("gpu_type", ""))

    try:
        result_gpu_memory = int(result.get("gpu_memory_total", -1))
    except (TypeError, ValueError):
        return False

    return (
        convert_gpu_name(str(result_gpu_type)) == convert_gpu_name(gpu_type)
        and result_gpu_memory == int(gpu_memory)
    )


def save_loading_result(
    result: Dict[str, Any],
    model_name: str,
    results_dir: str = DEFAULT_RESULTS_DIR,
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    result_path = get_result_path(model_name, results_dir)
    print(f"Saving loading benchmark result to {result_path}")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)


def _move_components_to_device(
    model_components: Dict[str, Any], device: str
) -> Tuple[float, int]:
    if device.startswith("cuda"):
        gpu_memory_before = torch.cuda.memory_allocated()
        torch.cuda.synchronize()
    else:
        gpu_memory_before = 0

    start_time = time.time()
    for model in model_components.values():
        if hasattr(model, "to"):
            model.to(device)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        gpu_memory_after = torch.cuda.memory_allocated()
    else:
        gpu_memory_after = 0

    return time.time() - start_time, gpu_memory_after - gpu_memory_before


def _move_components_to_cpu(model_components: Dict[str, Any]) -> None:
    for model in model_components.values():
        if hasattr(model, "to"):
            model.to("cpu")


def run_loading_benchmark(
    case: LoadingBenchmarkCase,
    device: str = "cuda",
    warmup: int = 3,
    repeats: int = 5,
    force_benchmark: bool = False,
    results_dir: str = DEFAULT_RESULTS_DIR,
) -> Dict[str, Any]:
    existing_result = load_existing_result(case.model_name, results_dir)
    if (
        existing_result is not None
        and result_matches_current_gpu(existing_result)
        and not force_benchmark
    ):
        print(
            f"Skipping {case.model_name}: loading result already exists for the "
            f"current GPU. Use --force-benchmark to re-run."
        )
        return existing_result

    gpu_type, gpu_memory, gpu_count = get_gpu_info()
    model_instance = get_op(case.model_name)

    disk_to_mem_start_time = time.time()
    model_components = model_instance.initialize(model_path=case.model_path, device="cpu")
    disk_to_mem_time = time.time() - disk_to_mem_start_time

    for _ in range(warmup):
        _move_components_to_device(model_components, device)
        _move_components_to_cpu(model_components)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    loading_times: List[float] = []
    gpu_memory_required: List[int] = []
    for _ in range(repeats):
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        loading_time, gpu_memory_used = _move_components_to_device(
            model_components, device
        )
        loading_times.append(loading_time)
        gpu_memory_required.append(gpu_memory_used)
        print(
            f"Loading time: {loading_time * 1000:.2f} ms, "
            f"GPU memory required: {gpu_memory_used / (1024**3):.2f} GiB"
        )

        _move_components_to_cpu(model_components)

    result = {
        "gpu_type": gpu_type,
        "gpu_type_normalized": convert_gpu_name(gpu_type),
        "gpu_memory_total": gpu_memory,
        "gpu_count": gpu_count,
        "model_name": case.model_name,
        "model_path": case.model_path,
        "suite": case.suite,
        "loading": {
            "disk_to_host_mem_time": disk_to_mem_time,
            "host_mem_to_gpu_time": sum(loading_times) / len(loading_times),
            "gpu_memory_required": sum(gpu_memory_required) / len(gpu_memory_required),
        },
    }
    save_loading_result(result, case.model_name, results_dir)
    return result


def load_cases_from_yaml(config_path: str) -> List[LoadingBenchmarkCase]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    suite = config["suite"]
    cases = []
    for model_config in config.get("models", []):
        cases.append(
            LoadingBenchmarkCase(
                model_name=model_config["model_name"],
                model_path=model_config.get("model_path"),
                suite=suite,
            )
        )
    return cases
