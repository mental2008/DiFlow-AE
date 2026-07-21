import time
from typing import Any, Dict

import torch

from benchmark.benchmark_utils import (
    BenchmarkConfig,
    print_benchmark_header,
    run_model_benchmark,
    save_benchmark_results,
)
from diffusionflow.operators.models.autoencoders.stable_diffusion_xl_vae import (
    StableDiffusionXLVAE,
)

# Model input shapes for SDXL VAE
CHANNELS = 4  # SDXL VAE uses 4 channels instead of 16
BATCH_SIZES = [1, 2, 4, 8]
# RESOLUTIONS = [(32, 32), (64, 64), (128, 128)]
RESOLUTIONS = [(1024, 1024)]
latent_shape = {
    "1024_1024": (128, 128),
}


def benchmark_stable_diffusion_xl_vae_prepare_image_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    height: int,
    width: int,
    device: str,
    **kwargs,
) -> float:
    """Benchmark function for StableDiffusionXL VAE model"""
    image = torch.randn(
        batch_size, 3, height, width, dtype=torch.float16, device=device
    )

    torch.cuda.synchronize()
    start_time = time.time()
    model_instance.execute(
        model_components=model_components,
        device=device,
        mode="prepare_image",
        image=image,
        height=height,
        width=width,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


def benchmark_stable_diffusion_xl_vae_decode_latents_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    height: int,
    width: int,
    device: str,
    **kwargs,
) -> float:
    """Benchmark function for StableDiffusionXL VAE model"""
    latent_height, latent_width = latent_shape[f"{height}_{width}"]
    latents = torch.randn(
        batch_size,
        CHANNELS,
        latent_height,
        latent_width,
        dtype=torch.float16,
        device=device,
    )

    torch.cuda.synchronize()
    start_time = time.time()
    model_instance.execute(
        model_components=model_components,
        device=device,
        mode="decode_latents",
        latents=latents,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    model_name = StableDiffusionXLVAE().id
    model_path = "/project/infattllm/lyangbk/huggingface/stable-diffusion-xl-base-1.0"
    prepare_image_config = BenchmarkConfig(
        mode="prepare_image",
        batch_sizes=BATCH_SIZES,
        resolutions=RESOLUTIONS,
    )
    decode_latents_config = BenchmarkConfig(
        mode="decode_latents",
        batch_sizes=BATCH_SIZES,
        resolutions=RESOLUTIONS,
    )
    print_benchmark_header([prepare_image_config, decode_latents_config], model_name)
    results = run_model_benchmark(
        model_name=model_name,
        model_path=model_path,
        benchmark_func=[
            benchmark_stable_diffusion_xl_vae_prepare_image_func,
            benchmark_stable_diffusion_xl_vae_decode_latents_func,
        ],
        config=[prepare_image_config, decode_latents_config],
    )
    save_benchmark_results(results, model_name)
