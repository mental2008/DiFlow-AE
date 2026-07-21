import time
from typing import Any, Dict

import torch

from benchmark.benchmark_utils import (
    BenchmarkConfig,
    print_benchmark_header,
    run_model_benchmark,
    save_benchmark_results,
)
from diffusionflow.operators.models.autoencoders.flux_1_vae import (
    Flux1VAE,
)

# Model input shapes for Flux VAE
# CHANNELS = 4096  # Flux VAE uses 4096 channels in packed format
BATCH_SIZES = [1, 2, 4, 8]
# ! Suyi: the latent shape for a 1024x1024 image with Flux is (1, 4096, 64)
# ! Suyi: to compatible with other model, we set the resolution to (4096, 64), but this is not the actual height and width
RESOLUTIONS = [(1024, 1024)]

latent_shape = {
    "1024_1024": (4096, 64),
}


def benchmark_flux_vae_prepare_image_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    height: int,
    width: int,
    device: str,
    **kwargs,
) -> float:
    """Benchmark function for Flux1VAE model"""
    image = torch.randn(
        batch_size, 3, height, width, dtype=torch.bfloat16, device=device
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


def benchmark_flux_vae_decode_latents_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    height: int,
    width: int,
    device: str,
    **kwargs,
) -> float:
    """Benchmark function for Flux1VAE model"""
    latents = torch.randn(
        batch_size,
        latent_shape[f"{height}_{width}"][0],
        latent_shape[f"{height}_{width}"][1],
        dtype=torch.bfloat16,
        device=device,
    )

    torch.cuda.synchronize()
    start_time = time.time()
    model_instance.execute(
        model_components=model_components,
        device=device,
        mode="decode_latents",
        latents=latents,
        height=height,
        width=width,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    model_path = "/project/infattllm/lyangbk/huggingface/FLUX.1-dev"
    model_name = Flux1VAE().id
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
            benchmark_flux_vae_prepare_image_func,
            benchmark_flux_vae_decode_latents_func,
        ],
        config=[prepare_image_config, decode_latents_config],
    )
    save_benchmark_results(results, model_name)
