import time
from typing import Any, Dict

import torch

from benchmark.benchmark_utils import (
    BenchmarkConfig,
    print_benchmark_header,
    run_model_benchmark,
    save_benchmark_results,
)
from diffusionflow.operators.models.adapters.flux_1_dev_controlnet import (
    Flux1DevControlNetCanny,
    Flux1DevControlNetDepth,
)

# Model input shapes for Flux
CHANNELS = 16  # Flux uses 16 channels (packed latents)
PROMPT_EMBEDS_SHAPE = (512, 4096)  # T5 embeddings for Flux
POOLED_PROMPT_EMBEDS_SHAPE = (768,)  # CLIP pooled embeddings for Flux
BATCH_SIZES = [1, 2, 4, 8]
RESOLUTIONS = [(1024, 1024)]
latent_shape = {
    "1024x1024": (4096, 64),
}


def benchmark_flux_controlnet_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    height: int,
    width: int,
    device: str,
    **kwargs,
) -> float:
    """Benchmark function for Flux ControlNet model"""
    dtype = model_components["controlnet"].dtype
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

    # ControlNet conditioning input
    controlnet_cond = torch.randn(
        batch_size, 3, height, width, dtype=dtype, device=device
    )

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in model_instance.execute(
        model_components=model_components,
        device=device,
        latents=latents,
        timestep=timestep,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        controlnet_cond=controlnet_cond,
        conditioning_scale=0.8,
        guidance=guidance,
        height=height,
        width=width,
    ):
        pass
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    controlnets = [
        (
            Flux1DevControlNetCanny().id,
            "/project/infattllm/lyangbk/huggingface/Xlabs-AI--flux-controlnet-canny-diffusers",
        ),
        (
            Flux1DevControlNetDepth().id,
            "/project/infattllm/lyangbk/huggingface/Xlabs-AI--flux-controlnet-depth-diffusers",
        ),
    ]

    for controlnet_name, controlnet_model_path in controlnets:
        config = BenchmarkConfig(
            mode="default",
            batch_sizes=BATCH_SIZES,
            resolutions=RESOLUTIONS,
        )
        print_benchmark_header([config], controlnet_name)
        results = run_model_benchmark(
            model_name=controlnet_name,
            model_path=controlnet_model_path,
            benchmark_func=[benchmark_flux_controlnet_func],
            config=[config],
        )
        save_benchmark_results(results, controlnet_name)
