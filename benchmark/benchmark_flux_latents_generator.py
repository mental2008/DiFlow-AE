import time
from typing import Any, Dict

import torch

from benchmark.benchmark_utils import (
    BenchmarkConfig,
    print_benchmark_header,
    run_model_benchmark,
    save_benchmark_results,
)
from diffusionflow.operators.custom.flux_latents_generator import FluxLatentsGenerator

# Model input shapes
CHANNELS = 16
BATCH_SIZES = [1, 2, 4, 8]
# RESOLUTIONS = [(256, 256), (512, 512), (1024, 1024)]
RESOLUTIONS = [(1024, 1024)]


def benchmark_flux_latents_generator_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    height: int,
    width: int,
    device: str,
    **kwargs
) -> float:
    """Benchmark function for LatentsGenerator model"""
    torch.cuda.synchronize()
    start_time = time.time()
    model_instance.execute(
        model_components=model_components,
        device=device,
        batch_size=batch_size,
        num_channels_latents=CHANNELS,
        height=height,
        width=width,
        seed=42,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    config = BenchmarkConfig(
        mode="default",
        batch_sizes=BATCH_SIZES,
        resolutions=RESOLUTIONS,
    )
    model_name = FluxLatentsGenerator().id
    print_benchmark_header([config], model_name)
    results = run_model_benchmark(
        model_name=model_name,
        model_path=None,
        benchmark_func=[benchmark_flux_latents_generator_func],
        config=[config],
    )
    save_benchmark_results(results, model_name)
