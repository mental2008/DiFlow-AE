import time
from typing import Any, Dict

import torch

from benchmark.benchmark_utils import (
    BenchmarkConfig,
    print_benchmark_header,
    run_model_benchmark,
    save_benchmark_results,
)
from diffusionflow.operators.custom.stable_diffusion_xl_text_encoder import (
    StableDiffusionXLTextEncoder,
)

# Model input shapes for SDXL text encoder
# CLIP_SDXL_1 output: (batch_size, 77, 768)
CLIP_PROMPT_EMBEDS_SHAPE = (77, 768)
# CLIP_SDXL_2 output: (batch_size, 77, 1280) 
CLIP_PROMPT_2_EMBEDS_SHAPE = (77, 1280)
# CLIP_SDXL_2 pooled output: (batch_size, 1280)
CLIP_POOLED_PROMPT_2_EMBEDS_SHAPE = (1280,)
BATCH_SIZES = [1, 2, 4, 8, 16, 32]


def benchmark_stable_diffusion_xl_text_encoder_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    device: str,
    **kwargs
) -> float:
    """Benchmark function for StableDiffusionXLTextEncoder model"""
    clip_prompt_embeds = torch.randn(
        batch_size, *CLIP_PROMPT_EMBEDS_SHAPE, dtype=torch.float16, device=device
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

    torch.cuda.synchronize()
    start_time = time.time()
    model_instance.execute(
        model_components=model_components,
        device=device,
        clip_prompt_embeds=clip_prompt_embeds,
        clip_prompt_2_embeds=clip_prompt_2_embeds,
        clip_pooled_prompt_2_embeds=clip_pooled_prompt_2_embeds,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    config = BenchmarkConfig(
        mode="default",
        batch_sizes=BATCH_SIZES,
    )
    model_name = StableDiffusionXLTextEncoder().id
    print_benchmark_header([config], model_name)
    results = run_model_benchmark(
        model_name=model_name,
        model_path=None,
        benchmark_func=[benchmark_stable_diffusion_xl_text_encoder_func],
        config=[config],
    )
    save_benchmark_results(results, model_name)
