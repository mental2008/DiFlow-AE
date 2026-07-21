import time
from typing import Any, Dict

import torch

from benchmark.benchmark_utils import (
    BenchmarkConfig,
    print_benchmark_header,
    run_model_benchmark,
    save_benchmark_results,
)
from diffusionflow.operators.custom.stable_diffusion_15_text_encoder import (
    StableDiffusion15TextEncoder,
)

# Model input shapes
PROMPT_EMBEDS_SHAPE = (77, 768)
BATCH_SIZES = [1, 2, 4, 8, 16, 32]


def benchmark_stable_diffusion_15_text_encoder_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    device: str,
    **kwargs
) -> float:
    """Benchmark function for StableDiffusion15TextEncoder model"""
    prompt_embeds = torch.randn(
        batch_size, *PROMPT_EMBEDS_SHAPE, dtype=torch.float16, device=device
    )

    torch.cuda.synchronize()
    start_time = time.time()
    model_instance.execute(
        model_components=model_components,
        device=device,
        prompt_embeds=prompt_embeds,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    config = BenchmarkConfig(
        mode="default",
        batch_sizes=BATCH_SIZES,
    )
    model_name = StableDiffusion15TextEncoder().id
    print_benchmark_header([config], model_name)
    results = run_model_benchmark(
        model_name=model_name,
        model_path=None,
        benchmark_func=[benchmark_stable_diffusion_15_text_encoder_func],
        config=[config],
    )
    save_benchmark_results(results, model_name)
