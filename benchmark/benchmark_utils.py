import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from diffusionflow.operators.utils import get_op


class Metric(Enum):
    BATCH_SIZE = "batch_size"
    HEIGHT = "height"
    WIDTH = "width"
    DISK_TO_HOST_MEM_TIME = "disk_to_host_mem_time"
    HOST_MEM_TO_GPU_TIME = "host_mem_to_gpu_time"
    EXECUTION_TIME = "execution_time"
    GPU_MEMORY_REQUIRED = "gpu_memory_required"
    GPU_MEMORY_USED = "gpu_memory_used"


def convert_enum_to_string(obj):
    """Recursively convert Metric enum values to strings in a dictionary"""
    if isinstance(obj, dict):
        return {
            convert_enum_to_string(k): convert_enum_to_string(v) for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_enum_to_string(item) for item in obj]
    elif isinstance(obj, Metric):
        return obj.value
    else:
        return obj


def get_model_disk_to_host_mem_time(model_config: Dict[str, Any]) -> float:
    """Get model disk to host memory time"""
    if "loading" in model_config:
        return model_config["loading"][Metric.DISK_TO_HOST_MEM_TIME.value]
    return -1


def get_model_host_mem_to_gpu_time(model_config: Dict[str, Any]) -> float:
    """Get model host memory to GPU time"""
    if "loading" in model_config:
        return model_config["loading"][Metric.HOST_MEM_TO_GPU_TIME.value]
    return -1


def get_model_gpu_memory_required(model_config: Dict[str, Any]) -> float:
    """Get model GPU memory required"""
    if "loading" in model_config:
        return model_config["loading"][Metric.GPU_MEMORY_REQUIRED.value]
    return -1


def get_model_gpu_memory_used_for_batch_size(
    model_config: Dict[str, Any], batch_size: int, mode: str = "default"
) -> float:
    """Get model GPU memory used for a given batch size"""
    if "execution" in model_config:
        for execution_list in model_config["execution"]:
            if execution_list["mode"] == mode:
                for result in execution_list["results"]:
                    if result[Metric.BATCH_SIZE.value] == batch_size:
                        return result[Metric.GPU_MEMORY_USED.value]
    return -1


def get_model_execution_time_for_batch_size(
    model_config: Dict[str, Any], batch_size: int, mode: str = "default"
) -> float:
    """Get model execution time for a given batch size"""
    if "execution" in model_config:
        for execution_list in model_config["execution"]:
            if execution_list["mode"] == mode:
                for result in execution_list["results"]:
                    if result[Metric.BATCH_SIZE.value] == batch_size:
                        return result[Metric.EXECUTION_TIME.value]
    return -1


def read_model_configs(config_dir: str) -> Dict[str, Dict[str, Any]]:
    """Read model configs"""
    model_configs = {}
    for file in os.listdir(config_dir):
        if file.endswith(".json"):
            with open(os.path.join(config_dir, file), "r") as f:
                config = json.load(f)
                model_name = config["model_name"]
                model_configs[model_name] = config
    return model_configs


def read_op_latencies(config_dir: str) -> Dict[str, float]:
    """Read op latencies"""
    with open(os.path.join(config_dir, "op_latencies_median.json"), "r") as f:
        return json.load(f)

@dataclass
class BenchmarkConfig:
    mode: str
    batch_sizes: List[int]
    resolutions: Optional[List[Tuple[int, int]]] = None

    @classmethod
    def default(cls):
        return cls(
            mode="default",
            batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256],
            resolutions=None,
        )


def get_gpu_info() -> Tuple[str, float, int]:
    """Get detailed GPU information"""
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")

    gpu_name = torch.cuda.get_device_name()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    gpu_count = torch.cuda.device_count()

    return gpu_name, gpu_memory, gpu_count


def save_benchmark_results(results: Dict, model_name: str):
    """Save benchmark results to a JSON file"""
    base_benchmark_dir = "benchmark/benchmark_results"
    os.makedirs(base_benchmark_dir, exist_ok=True)
    save_path = f"{base_benchmark_dir}/{model_name}.json"
    print(f"Saving benchmark results to {save_path}")
    results = convert_enum_to_string(results)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)


def run_benchmark_with_oom_handling(
    benchmark_func: Callable,
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    warmup: int,
    repeats: int,
    device: str,
    **kwargs,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Run a benchmark function with OOM handling

    Args:
        benchmark_func: Function to benchmark
        model_instance: Model instance
        model_components: Model components
        batch_size: Batch size to test
        config: Benchmark configuration
        **kwargs: Additional arguments for benchmark_func

    Returns:
        Tuple of (latency, throughput, error_message)
    """
    try:
        # Warmup
        for _ in range(warmup):
            benchmark_func(
                model_instance=model_instance,
                model_components=model_components,
                batch_size=batch_size,
                device=device,
                **kwargs,
            )
        # Test
        total_execution_time = 0
        total_gpu_memory_used = 0
        for _ in range(repeats):
            torch.cuda.reset_peak_memory_stats()

            execution_time = benchmark_func(
                model_instance=model_instance,
                model_components=model_components,
                batch_size=batch_size,
                device=device,
                **kwargs,
            )
            total_execution_time += execution_time
            total_gpu_memory_used += torch.cuda.max_memory_allocated()
            print(
                f"Execution time: {execution_time * 1000:.2f} ms, GPU memory used: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GiB"
            )

        avg_execution_time = total_execution_time / repeats  # seconds
        avg_gpu_memory_used = total_gpu_memory_used / repeats  # bytes
        return avg_execution_time, avg_gpu_memory_used, None

    except Exception as e:
        if "out of memory" in str(e):
            if device == "cuda":
                torch.cuda.empty_cache()
            return None, None, "OOM"
        else:
            raise


def run_model_loading_benchmark(
    model_instance: Any,
    model_path: str,
    device: str,
    warmup: int = 3,
    repeats: int = 5,
) -> Dict:
    """Run model loading benchmark"""
    # Model loading
    disk_to_mem_start_time = time.time()
    model_components = model_instance.initialize(model_path=model_path, device="cpu")
    disk_to_mem_end_time = time.time()
    disk_to_mem_time = disk_to_mem_end_time - disk_to_mem_start_time

    # Warmup
    for _ in range(warmup):
        for model in model_components.values():
            if not hasattr(model, "to"):
                continue
            model.to(device)
            model.to("cpu")
    # Test
    print("=== Model loading ===")
    total_loading_time = 0
    total_gpu_memory_used = 0
    for _ in range(repeats):
        for model in model_components.values():
            if not hasattr(model, "to"):
                continue
            gpu_memory_before = torch.cuda.memory_allocated()
            torch.cuda.synchronize()
            start_time = time.time()
            model.to(device)
            torch.cuda.synchronize()
            end_time = time.time()
            loading_time = end_time - start_time
            total_loading_time += loading_time

            gpu_memory_after = torch.cuda.memory_allocated()
            gpu_memory_used = gpu_memory_after - gpu_memory_before
            total_gpu_memory_used += gpu_memory_used
            print(
                f"Loading time: {loading_time * 1000:.2f} ms, GPU memory used: {gpu_memory_used / (1024**3):.2f} GiB"
            )
            model.to("cpu")
    avg_loading_time = total_loading_time / repeats  # seconds
    avg_gpu_memory_used = total_gpu_memory_used / repeats  # bytes

    return model_components, {
        Metric.DISK_TO_HOST_MEM_TIME: disk_to_mem_time,
        Metric.HOST_MEM_TO_GPU_TIME: avg_loading_time,
        Metric.GPU_MEMORY_REQUIRED: avg_gpu_memory_used,
    }


def run_model_execution_benchmark(
    model_instance: Any,
    model_components: Dict[str, Any],
    benchmark_func: Callable,
    config: BenchmarkConfig,
    device: str,
    warmup: int,
    repeats: int,
    **kwargs,
) -> Dict:
    """Run model execution benchmark"""
    # Model execution
    print(f"=== Model execution ({config.mode}) ===")
    if config.resolutions is None:
        results = []
        for batch_size in config.batch_sizes:
            print(f"Batch size: {batch_size}")
            result = run_benchmark_with_oom_handling(
                benchmark_func=benchmark_func,
                model_instance=model_instance,
                model_components=model_components,
                batch_size=batch_size,
                device=device,
                warmup=warmup,
                repeats=repeats,
                **kwargs,
            )
            results.append(
                {
                    Metric.BATCH_SIZE: batch_size,
                    Metric.EXECUTION_TIME: result[0],
                    Metric.GPU_MEMORY_USED: result[1],
                }
            )
    else:
        results = []
        for height, width in config.resolutions:
            for batch_size in config.batch_sizes:
                print(f"Batch size: {batch_size}, Resolution: {height}x{width}")
                result = run_benchmark_with_oom_handling(
                    benchmark_func=benchmark_func,
                    model_instance=model_instance,
                    model_components=model_components,
                    batch_size=batch_size,
                    device=device,
                    warmup=warmup,
                    repeats=repeats,
                    height=height,
                    width=width,
                    **kwargs,
                )
                results.append(
                    {
                        Metric.BATCH_SIZE: batch_size,
                        Metric.HEIGHT: height,
                        Metric.WIDTH: width,
                        Metric.EXECUTION_TIME: result[0],
                        Metric.GPU_MEMORY_USED: result[1],
                    }
                )
    return {
        "mode": config.mode,
        "results": results,
    }


def run_model_benchmark(
    model_name: str,
    model_path: str,
    benchmark_func: List[Callable],
    config: List[BenchmarkConfig],
    device: str = "cuda",
    warmup: int = 3,
    repeats: int = 5,
    **kwargs,
) -> Dict:
    """
    Generic function to run benchmarks for any model

    Args:
        model_name: Name of the model
        benchmark_func: List of functions to benchmark
        config: List of benchmark configurations
        **kwargs: Additional arguments for benchmark_func

    Returns:
        Dictionary of results
    """
    gpu_info = get_gpu_info()
    gpu_type = gpu_info[0]
    gpu_memory = gpu_info[1]

    model_instance = get_op(model_name)

    model_components, loading_results = run_model_loading_benchmark(
        model_instance=model_instance,
        model_path=model_path,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )

    for model in model_components.values():
        if not hasattr(model, "to"):
            continue
        model.to(device)

    execution_results = []
    for func, cfg in zip(benchmark_func, config):
        execution_results.append(
            run_model_execution_benchmark(
                model_name=model_name,
                model_instance=model_instance,
                model_components=model_components,
                benchmark_func=func,
                config=cfg,
                device=device,
                warmup=warmup,
                repeats=repeats,
            )
        )

    return {
        "gpu_type": gpu_type,
        "gpu_memory_total": gpu_memory,
        "model_name": model_name,
        "loading": loading_results,
        "execution": execution_results,
    }


def print_benchmark_header(config: List[BenchmarkConfig], model_name: str = "Model"):
    """Print benchmark header with configuration info"""
    gpu_info = get_gpu_info()
    print(f"Benchmarking {model_name}")
    print(
        f"Device: {gpu_info[0]} ({gpu_info[1] / (1024**3):.2f}GiB, {gpu_info[2]} device(s))"
    )
    print(f"{'=' * 60}")
    for cfg in config:
        print(f"Mode: {cfg.mode}")
        print(f"Batch sizes: {cfg.batch_sizes}")
        if cfg.resolutions is not None:
            print(f"Resolutions: {cfg.resolutions}")
        print(f"{'=' * 60}")
