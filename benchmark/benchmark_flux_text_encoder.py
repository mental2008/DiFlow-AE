import time
from typing import Any, Dict

import torch

from benchmark.benchmark_utils import (
    BenchmarkConfig,
    print_benchmark_header,
    run_model_benchmark,
    save_benchmark_results,
)
from diffusionflow.operators.custom.flux_text_encoder import (
    FluxTextEncoder,
)

# Model input shapes
CLIP_PROMPT_EMBEDS_SHAPE = (768,)
T5_PROMPT_EMBEDS_SHAPE = (512, 4096)
BATCH_SIZES = [1, 2, 4, 8, 16, 32]


def benchmark_flux_text_encoder_func(
    model_instance: Any,
    model_components: Dict[str, Any],
    batch_size: int,
    device: str,
    **kwargs
) -> float:
    """Benchmark function for FluxTextEncoder model"""
    clip_prompt_embeds = torch.randn(
        batch_size, *CLIP_PROMPT_EMBEDS_SHAPE, dtype=torch.float16, device=device
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
    model_name = FluxTextEncoder().id
    print_benchmark_header([config], model_name)
    results = run_model_benchmark(
        model_name=model_name,
        model_path=None,
        benchmark_func=[benchmark_flux_text_encoder_func],
        config=[config],
    )
    save_benchmark_results(results, model_name)
