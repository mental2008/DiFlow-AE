import time
from typing import Any, Dict

import torch

from benchmark.benchmark_utils import (
    BenchmarkConfig,
    print_benchmark_header,
    run_model_benchmark,
    save_benchmark_results,
)
from diffusionflow.operators.schedulers.euler_discrete_scheduler import (
    EulerDiscreteScheduler,
)

# Model input shapes
CHANNELS = 16
RESOLUTIONS = [(1024, 1024)]
latent_shape = {
    "1024_1024": (128, 128),
}


def benchmark_euler_discrete_scheduler_init_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    device: str,
    **kwargs,
) -> float:
    """Benchmark function for EulerDiscreteScheduler model"""
    torch.cuda.synchronize()
    start_time = time.time()
    model_instance.execute(
        model_components=model_components,
        device=device,
        mode="init",
        num_inference_steps=50,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


def benchmark_euler_discrete_scheduler_scale_model_input_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    height: int,
    width: int,
    device: str,
    **kwargs,
) -> float:
    """Benchmark function for EulerDiscreteScheduler model"""
    latent_height, latent_width = latent_shape[f"{height}_{width}"]
    latents = torch.randn(
        batch_size,
        CHANNELS,
        latent_height,
        latent_width,
        dtype=torch.float16,
        device=device,
    )
    timestep = torch.tensor([981.0], dtype=torch.float16, device=device)

    # reinitialize scheduler to avoid indices out of range
    model_instance.execute(
        model_components=model_components,
        device=device,
        mode="init",
        num_inference_steps=50,
    )

    torch.cuda.synchronize()
    start_time = time.time()
    model_instance.execute(
        model_components=model_components,
        device=device,
        mode="scale_model_input",
        latents=latents,
        timestep=timestep,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


def benchmark_euler_discrete_scheduler_init_noise_sigma_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    height: int,
    width: int,
    device: str,
    **kwargs,
) -> float:
    """Benchmark function for EulerDiscreteScheduler model"""
    latent_height, latent_width = latent_shape[f"{height}_{width}"]
    latents = torch.randn(
        batch_size,
        CHANNELS,
        latent_height,
        latent_width,
        dtype=torch.float16,
        device=device,
    )
    timestep = torch.tensor([981.0], dtype=torch.float16, device=device)

    # reinitialize scheduler to avoid indices out of range
    model_instance.execute(
        model_components=model_components,
        device=device,
        mode="init",
        num_inference_steps=50,
    )

    torch.cuda.synchronize()
    start_time = time.time()
    model_instance.execute(
        model_components=model_components,
        device=device,
        mode="init_noise_sigma",
        latents=latents,
        timestep=timestep,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


def benchmark_euler_discrete_scheduler_step_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    height: int,
    width: int,
    device: str,
    **kwargs,
) -> float:
    """Benchmark function for EulerDiscreteScheduler model"""
    latent_height, latent_width = latent_shape[f"{height}_{width}"]
    latents = torch.randn(
        batch_size,
        CHANNELS,
        latent_height,
        latent_width,
        dtype=torch.float16,
        device=device,
    )
    timestep = torch.tensor([981.0], dtype=torch.float16, device=device)
    noise_pred = torch.randn(
        batch_size,
        CHANNELS,
        latent_height,
        latent_width,
        dtype=torch.float16,
        device=device,
    )

    # reinitialize scheduler to avoid indices out of range
    model_instance.execute(
        model_components=model_components,
        device=device,
        mode="init",
        num_inference_steps=50,
    )

    torch.cuda.synchronize()
    start_time = time.time()
    model_instance.execute(
        model_components=model_components,
        device=device,
        mode="step",
        latents=latents,
        timestep=timestep,
        noise_pred=noise_pred,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


def benchmark_euler_discrete_scheduler_step_classifier_free_guidance_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    height: int,
    width: int,
    device: str,
    **kwargs,
) -> float:
    """Benchmark function for EulerDiscreteScheduler model"""
    latent_height, latent_width = latent_shape[f"{height}_{width}"]
    latents = torch.randn(
        batch_size,
        CHANNELS,
        latent_height,
        latent_width,
        dtype=torch.float16,
        device=device,
    )
    timestep = torch.tensor([981.0], dtype=torch.float16, device=device)
    noise_pred_uncond = torch.randn(
        batch_size,
        CHANNELS,
        latent_height,
        latent_width,
        dtype=torch.float16,
        device=device,
    )
    noise_pred_text = torch.randn(
        batch_size,
        CHANNELS,
        latent_height,
        latent_width,
        dtype=torch.float16,
        device=device,
    )
    guidance_scale = 3.5

    # reinitialize scheduler to avoid indices out of range
    model_instance.execute(
        model_components=model_components,
        device=device,
        mode="init",
        num_inference_steps=50,
        latents=latents,
    )

    torch.cuda.synchronize()
    start_time = time.time()
    model_instance.execute(
        model_components=model_components,
        device=device,
        mode="step_classifier_free_guidance",
        latents=latents,
        timestep=timestep,
        noise_pred_uncond=noise_pred_uncond,
        noise_pred_text=noise_pred_text,
        guidance_scale=guidance_scale,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    model_path = "/project/infattllm/lyangbk/huggingface/stable-diffusion-xl-base-1.0"
    model_name = EulerDiscreteScheduler().id
    init_config = BenchmarkConfig(
        mode="init",
        batch_sizes=[1],
        resolutions=RESOLUTIONS,
    )
    scale_model_input_config = BenchmarkConfig(
        mode="scale_model_input",
        batch_sizes=[1],
        resolutions=RESOLUTIONS,
    )
    init_noise_sigma_config = BenchmarkConfig(
        mode="init_noise_sigma",
        batch_sizes=[1],
        resolutions=RESOLUTIONS,
    )
    step_config = BenchmarkConfig(
        mode="step",
        batch_sizes=[1],
        resolutions=RESOLUTIONS,
    )
    step_classifier_free_guidance_config = BenchmarkConfig(
        mode="step_classifier_free_guidance",
        batch_sizes=[1],
        resolutions=RESOLUTIONS,
    )
    print_benchmark_header(
        [
            init_config,
            scale_model_input_config,
            init_noise_sigma_config,
            step_config,
            step_classifier_free_guidance_config,
        ],
        model_name,
    )
    results = run_model_benchmark(
        model_name=model_name,
        model_path=model_path,
        benchmark_func=[
            benchmark_euler_discrete_scheduler_init_func,
            benchmark_euler_discrete_scheduler_scale_model_input_func,
            benchmark_euler_discrete_scheduler_init_noise_sigma_func,
            benchmark_euler_discrete_scheduler_step_func,
            benchmark_euler_discrete_scheduler_step_classifier_free_guidance_func,
        ],
        config=[
            init_config,
            scale_model_input_config,
            init_noise_sigma_config,
            step_config,
            step_classifier_free_guidance_config,
        ],
    )
    save_benchmark_results(results, model_name)
