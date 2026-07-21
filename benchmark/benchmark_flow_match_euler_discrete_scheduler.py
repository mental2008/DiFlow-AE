import time
from typing import Any, Dict

import torch

from benchmark.benchmark_utils import (
    BenchmarkConfig,
    print_benchmark_header,
    run_model_benchmark,
    save_benchmark_results,
)
from diffusionflow.operators.schedulers.flow_match_euler_discrete_scheduler import (
    FlowMatchEulerDiscreteScheduler,
)

# Model input shapes
CHANNELS = 16
# RESOLUTIONS = [(32, 32), (64, 64), (128, 128)]
RESOLUTIONS = [(1024, 1024)]
latent_shape = {
    "1024x1024": (128, 128),
}


def benchmark_flow_match_euler_discrete_scheduler_init_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    device: str,
    **kwargs,
) -> float:
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


def benchmark_flow_match_euler_discrete_scheduler_step_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    height: int,
    width: int,
    device: str,
    **kwargs,
) -> float:
    """Benchmark function for FlowMatchEulerDiscreteScheduler model"""
    latent_height, latent_width = latent_shape[f"{height}x{width}"]
    latents = torch.randn(
        batch_size,
        CHANNELS,
        latent_height,
        latent_width,
        dtype=torch.float16,
        device=device,
    )
    timestep = torch.tensor([1000.0], dtype=torch.float16, device=device)
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


def benchmark_flow_match_euler_discrete_scheduler_step_classifier_free_guidance_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    height: int,
    width: int,
    device: str,
    **kwargs,
) -> float:
    """Benchmark function for FlowMatchEulerDiscreteScheduler model"""
    latent_height, latent_width = latent_shape[f"{height}x{width}"]
    latents = torch.randn(
        batch_size,
        CHANNELS,
        latent_height,
        latent_width,
        dtype=torch.float16,
        device=device,
    )
    timestep = torch.tensor([1000.0], dtype=torch.float16, device=device)
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
    model_path = (
        "/project/infattllm/lyangbk/huggingface/stable-diffusion-3-medium-diffusers"
    )
    model_name = FlowMatchEulerDiscreteScheduler().id
    init_config = BenchmarkConfig(
        mode="init",
        batch_sizes=[1],
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
        [init_config, step_config, step_classifier_free_guidance_config], model_name
    )
    results = run_model_benchmark(
        model_name=model_name,
        model_path=model_path,
        benchmark_func=[
            benchmark_flow_match_euler_discrete_scheduler_init_func,
            benchmark_flow_match_euler_discrete_scheduler_step_func,
            benchmark_flow_match_euler_discrete_scheduler_step_classifier_free_guidance_func,
        ],
        config=[init_config, step_config, step_classifier_free_guidance_config],
    )
    print(results)
    save_benchmark_results(results, model_name)
