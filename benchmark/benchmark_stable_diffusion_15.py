import time
from typing import Any, Dict

import torch

from benchmark.benchmark_utils import (
    BenchmarkConfig,
    print_benchmark_header,
    run_model_benchmark,
    save_benchmark_results,
)
from diffusionflow.operators.models.diffusion_models.stable_diffusion_15 import (
    StableDiffusion15,
)

# Model input shapes
CHANNELS = 4
PROMPT_EMBEDS_SHAPE = (77, 768)
BATCH_SIZES = [1, 2, 4, 8]
# RESOLUTIONS = [(32, 32), (64, 64), (128, 128)]
RESOLUTIONS = [(512, 512)]
latent_shape = {
    "512x512": (64, 64),
}


def benchmark_stable_diffusion_15_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    height: int,
    width: int,
    device: str,
    **kwargs,
) -> float:
    """Benchmark function for StableDiffusion15 model"""
    latent_height, latent_width = latent_shape[f"{height}x{width}"]
    latents = torch.randn(
        batch_size,
        CHANNELS,
        latent_height,
        latent_width,
        dtype=torch.float16,
        device=device,
    )
    timestep = torch.randint(0, 1000, (batch_size,), device=device)
    prompt_embeds = torch.randn(
        batch_size, *PROMPT_EMBEDS_SHAPE, dtype=torch.float16, device=device
    )

    torch.cuda.synchronize()
    start_time = time.time()
    model_instance.execute(
        model_components=model_components,
        device=device,
        latents=latents,
        timestep=timestep,
        prompt_embeds=prompt_embeds,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    model_path = "/project/infattllm/lyangbk/huggingface/stable-diffusion-v1-5"
    model_name = StableDiffusion15().id
    config = BenchmarkConfig(
        mode="default",
        batch_sizes=BATCH_SIZES,
        resolutions=RESOLUTIONS,
    )
    print_benchmark_header([config], model_name)
    results = run_model_benchmark(
        model_name=model_name,
        model_path=model_path,
        benchmark_func=[benchmark_stable_diffusion_15_func],
        config=[config],
    )
    save_benchmark_results(results, model_name)
