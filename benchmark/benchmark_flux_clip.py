import time
from typing import Any, Dict

import torch

from benchmark.benchmark_utils import (
    BenchmarkConfig,
    print_benchmark_header,
    run_model_benchmark,
    save_benchmark_results,
)
from diffusionflow.operators.models.text_encoders.clip_flux import CLIP_Flux

BATCH_SIZES = [1, 2, 4, 8, 16, 32]
TEST_PROMPTS = [
    "a beautiful landscape",
    "a portrait of a person",
    "an abstract painting",
]


def benchmark_clip_flux_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    device: str,
    **kwargs
) -> float:
    """Benchmark function for CLIP_Flux model"""
    prompts = TEST_PROMPTS * (batch_size // len(TEST_PROMPTS) + 1)
    prompts = prompts[:batch_size]

    torch.cuda.synchronize()
    start_time = time.time()
    model_instance.execute(
        model_components=model_components,
        device=device,
        prompt=prompts,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    model_path = "/project/infattllm/lyangbk/huggingface/FLUX.1-dev"
    model_name = CLIP_Flux().id
    config = BenchmarkConfig(
        mode="default",
        batch_sizes=BATCH_SIZES,
    )
    print_benchmark_header([config], model_name)
    results = run_model_benchmark(
        model_name=model_name,
        model_path=model_path,
        benchmark_func=[benchmark_clip_flux_func],
        config=[config],
    )
    save_benchmark_results(results, model_name)
