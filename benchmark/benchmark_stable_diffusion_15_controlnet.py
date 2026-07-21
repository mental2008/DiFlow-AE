import time
from typing import Any, Dict

import torch

from benchmark.benchmark_utils import (
    BenchmarkConfig,
    print_benchmark_header,
    run_model_benchmark,
    save_benchmark_results,
)
from diffusionflow.operators.models.adapters.stable_diffusion_15_controlnet import (
    StableDiffusion15ControlNetCanny,
    StableDiffusion15ControlNetDepth,
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


def benchmark_stable_diffusion_15_controlnet_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    height: int,
    width: int,
    device: str,
    **kwargs,
) -> float:
    """Benchmark function for StableDiffusion15ControlNet model"""
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
        controlnet_cond=controlnet_cond,
        conditioning_scale=0.5,
    ):
        pass
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    controlnets = [
        (
            StableDiffusion15ControlNetCanny().id,
            "/project/infattllm/lyangbk/huggingface/lllyasviel--sd-controlnet-canny",
        ),
        (
            StableDiffusion15ControlNetDepth().id,
            "/project/infattllm/lyangbk/huggingface/lllyasviel--sd-controlnet-depth",
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
            benchmark_func=[benchmark_stable_diffusion_15_controlnet_func],
            config=[config],
        )
        save_benchmark_results(results, controlnet_name)
