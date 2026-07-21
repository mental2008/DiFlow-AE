import time
from typing import Any, Dict

import torch

from benchmark.benchmark_utils import (
    BenchmarkConfig,
    print_benchmark_header,
    run_model_benchmark,
    save_benchmark_results,
)
from diffusionflow.operators.models.adapters.stable_diffusion_xl_controlnet import (
    StableDiffusionXLControlNetCanny,
    StableDiffusionXLControlNetDepth,
)

# Model input shapes for SDXL
CHANNELS = 4  # SDXL uses 4 channels instead of 16
PROMPT_EMBEDS_SHAPE = (77, 2048)
POOLED_PROMPT_EMBEDS_SHAPE = (1280,)
BATCH_SIZES = [1, 2, 4, 8]
# RESOLUTIONS = [(32, 32), (64, 64), (128, 128)]
RESOLUTIONS = [(1024, 1024)]
latent_shape = {
    "1024x1024": (128, 128),
}


def benchmark_stable_diffusion_xl_controlnet_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    height: int,
    width: int,
    device: str,
    **kwargs,
) -> float:
    """Benchmark function for StableDiffusionXLControlNet model"""
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
    pooled_prompt_embeds = torch.randn(
        batch_size, *POOLED_PROMPT_EMBEDS_SHAPE, dtype=torch.float16, device=device
    )
    controlnet_cond = torch.randn(
        batch_size, 3, height, width, dtype=torch.float16, device=device
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
        conditioning_scale=0.5,
        height=height,  # SDXL requires height and width parameters
        width=width,
    ):
        pass
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    controlnets = [
        (
            StableDiffusionXLControlNetCanny().id,
            "/project/infattllm/lyangbk/huggingface/controlnet-canny-sdxl-1.0",
        ),
        (
            StableDiffusionXLControlNetDepth().id,
            "/project/infattllm/lyangbk/huggingface/controlnet-depth-sdxl-1.0",
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
            benchmark_func=[benchmark_stable_diffusion_xl_controlnet_func],
            config=[config],
        )
        save_benchmark_results(results, controlnet_name)
