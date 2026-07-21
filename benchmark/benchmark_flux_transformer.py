import time
from typing import Any, Dict

import torch

from benchmark.benchmark_utils import (
    BenchmarkConfig,
    print_benchmark_header,
    run_model_benchmark,
    save_benchmark_results,
)
from diffusionflow.operators.models.diffusion_models.flux_1_dev import (
    Flux1Dev,
)

# Model input shapes for Flux
CHANNELS = 16  # Flux uses 16 channels like SD3
PROMPT_EMBEDS_SHAPE = (512, 4096)  # Flux uses T5 embeddings (512, 4096)
POOLED_PROMPT_EMBEDS_SHAPE = (768,)  # Flux uses CLIP embeddings (768,)
BATCH_SIZES = [1, 2, 4, 8]
RESOLUTIONS = [(1024, 1024)]
latent_shape = {
    "1024x1024": (4096, 64),
}


def benchmark_flux_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    height: int,
    width: int,
    device: str,
    **kwargs,
) -> float:
    """Benchmark function for Flux model"""
    dtype = model_components["transformer"].dtype
    latent_height, latent_width = latent_shape[f"{height}x{width}"]
    latents = torch.randn(
        batch_size, latent_height, latent_width, dtype=dtype, device=device
    )
    timestep = torch.randint(0, 1000, (batch_size,), device=device)
    prompt_embeds = torch.randn(
        batch_size, *PROMPT_EMBEDS_SHAPE, dtype=dtype, device=device
    )
    pooled_prompt_embeds = torch.randn(
        batch_size, *POOLED_PROMPT_EMBEDS_SHAPE, dtype=dtype, device=device
    )

    # Flux-specific inputs
    guidance = torch.full((batch_size,), 3.5, dtype=torch.float32, device=device)

    torch.cuda.synchronize()
    start_time = time.time()
    model_instance.execute(
        model_components=model_components,
        device=device,
        latents=latents,
        timestep=timestep,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        guidance=guidance,
        height=height,
        width=width,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    model_path = "/project/infattllm/lyangbk/huggingface/FLUX.1-dev"
    model_name = Flux1Dev().id
    config = BenchmarkConfig(
        mode="default",
        batch_sizes=BATCH_SIZES,
        resolutions=RESOLUTIONS,
    )
    print_benchmark_header([config], model_name)
    results = run_model_benchmark(
        model_name=model_name,
        model_path=model_path,
        benchmark_func=[benchmark_flux_func],
        config=[config],
    )
    save_benchmark_results(results, model_name)
