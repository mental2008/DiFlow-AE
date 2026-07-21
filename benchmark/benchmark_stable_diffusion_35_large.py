import time
from typing import Any, Dict

import torch

from benchmark.benchmark_utils import (
    BenchmarkConfig,
    print_benchmark_header,
    run_model_benchmark,
    save_benchmark_results,
)
from diffusionflow.operators.models.diffusion_models.stable_diffusion_35_large import (
    StableDiffusion35Large,
)

# Model input shapes
CHANNELS = 16
PROMPT_EMBEDS_SHAPE = (333, 4096)
POOLED_PROMPT_EMBEDS_SHAPE = (2048,)
BATCH_SIZES = [1, 2, 4, 8]
# RESOLUTIONS = [(32, 32), (64, 64), (128, 128)]
RESOLUTIONS = [(128, 128)]


def benchmark_stable_diffusion_35_large_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    height: int,
    width: int,
    device: str,
    **kwargs
) -> float:
    """Benchmark function for StableDiffusion35Large model"""
    latents = torch.randn(
        batch_size, CHANNELS, height, width, dtype=torch.float16, device=device
    )
    timestep = torch.randint(0, 1000, (batch_size,), device=device)
    prompt_embeds = torch.randn(
        batch_size, *PROMPT_EMBEDS_SHAPE, dtype=torch.float16, device=device
    )
    pooled_prompt_embeds = torch.randn(
        batch_size, *POOLED_PROMPT_EMBEDS_SHAPE, dtype=torch.float16, device=device
    )

    torch.cuda.synchronize()
    start_time = time.time()
    model_instance.execute(
        model_components=model_components,
        device=device,
        latents=latents,
        timestep=timestep,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    model_path = "/project/infattllm/lyangbk/huggingface/stable-diffusion-3.5-large"
    model_name = StableDiffusion35Large().id
    config = BenchmarkConfig(
        mode="default",
        batch_sizes=BATCH_SIZES,
        resolutions=RESOLUTIONS,
    )
    print_benchmark_header([config], model_name)
    results = run_model_benchmark(
        model_name=model_name,
        model_path=model_path,
        benchmark_func=[benchmark_stable_diffusion_35_large_func],
        config=[config],
    )
    save_benchmark_results(results, model_name)
