import time
from typing import Any, Dict

import torch

from benchmark.benchmark_utils import (
    BenchmarkConfig,
    print_benchmark_header,
    run_model_benchmark,
    save_benchmark_results,
)
from diffusionflow.operators.custom.stable_diffusion_3_text_encoder import (
    StableDiffusion3TextEncoder,
)

# Model input shapes
PROMPT_EMBEDS_SHAPE = (333, 4096)
POOLED_PROMPT_EMBEDS_SHAPE = (2048,)
CLIP_PROMPT_2_EMBEDS_SHAPE = (333, 4096)
CLIP_POOLED_PROMPT_2_EMBEDS_SHAPE = (2048,)
T5_PROMPT_EMBEDS_SHAPE = (1024, 1024)
BATCH_SIZES = [1, 2, 4, 8, 16, 32]


def benchmark_stable_diffusion_3_text_encoder_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    device: str,
    **kwargs
) -> float:
    """Benchmark function for StableDiffusion3TextEncoder model"""
    clip_prompt_embeds = torch.randn(
        batch_size, *PROMPT_EMBEDS_SHAPE, dtype=torch.float16, device=device
    )
    clip_pooled_prompt_embeds = torch.randn(
        batch_size, *POOLED_PROMPT_EMBEDS_SHAPE, dtype=torch.float16, device=device
    )
    clip_prompt_2_embeds = torch.randn(
        batch_size, *CLIP_PROMPT_2_EMBEDS_SHAPE, dtype=torch.float16, device=device
    )
    clip_pooled_prompt_2_embeds = torch.randn(
        batch_size,
        *CLIP_POOLED_PROMPT_2_EMBEDS_SHAPE,
        dtype=torch.float16,
        device=device
    )
    t5_prompt_embeds = torch.randn(
        batch_size, *T5_PROMPT_EMBEDS_SHAPE, dtype=torch.float16, device=device
    )

    torch.cuda.synchronize()
    start_time = time.time()
    model_instance.execute(
        model_components=model_components,
        device=device,
        clip_prompt_embeds=clip_prompt_embeds,
        clip_pooled_prompt_embeds=clip_pooled_prompt_embeds,
        clip_prompt_2_embeds=clip_prompt_2_embeds,
        clip_pooled_prompt_2_embeds=clip_pooled_prompt_2_embeds,
        t5_prompt_embeds=t5_prompt_embeds,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    config = BenchmarkConfig(
        mode="default",
        batch_sizes=BATCH_SIZES,
    )
    model_name = StableDiffusion3TextEncoder().id
    print_benchmark_header([config], model_name)
    results = run_model_benchmark(
        model_name=model_name,
        model_path=None,
        benchmark_func=[benchmark_stable_diffusion_3_text_encoder_func],
        config=[config],
    )
    save_benchmark_results(results, model_name)
