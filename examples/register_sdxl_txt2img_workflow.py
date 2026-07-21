# python examples/register_sdxl_txt2img_workflow.py
import argparse

from diffusionflow.interface import Workflow, register_workflow
from diffusionflow.interface.node_io import DiffusionModelInputs, SchedulerInputs
from diffusionflow.operators import (
    CLIP_SDXL_1,
    CLIP_SDXL_2,
    Config,
    EulerDiscreteScheduler,
    LatentsGenerator,
    StableDiffusionXL,
    StableDiffusionXLTextEncoder,
    StableDiffusionXLVAE,
)


def create_workflow(model_path: str) -> Workflow:
    workflow = Workflow(name="sdxl_txt2img_workflow")

    # Define model nodes
    latents_generator = LatentsGenerator()
    clip_1 = CLIP_SDXL_1(Config(model_path=model_path))
    clip_2 = CLIP_SDXL_2(Config(model_path=model_path))
    text_encoder = StableDiffusionXLTextEncoder()
    scheduler = EulerDiscreteScheduler(Config(model_path=model_path))
    sdxl = StableDiffusionXL(Config(model_path=model_path))
    vae = StableDiffusionXLVAE(Config(model_path=model_path))

    # Define inputs
    seed = workflow.add_input(name="seed", data_type=int)
    prompt = workflow.add_input(name="prompt", data_type=str)
    height = workflow.add_input(name="height", data_type=int)
    width = workflow.add_input(name="width", data_type=int)
    num_inference_steps = workflow.add_input(name="num_inference_steps", data_type=int)
    num_channels_latents = workflow.add_input(
        name="num_channels_latents", data_type=int
    )

    # Define connections
    latents = latents_generator(
        seed=seed, height=height, width=width, num_channels_latents=num_channels_latents
    )
    clip_prompt_embeds, clip_pooled_prompt_embeds = clip_1(prompt=prompt)
    clip_prompt_2_embeds, clip_pooled_prompt_2_embeds = clip_2(prompt=prompt)

    # SDXL text encoder doesn't use T5 or clip_pooled_prompt_embeds from CLIP_L
    prompt_embeds, pooled_prompt_embeds = text_encoder(
        clip_prompt_embeds=clip_prompt_embeds,
        clip_prompt_2_embeds=clip_prompt_2_embeds,
        clip_pooled_prompt_2_embeds=clip_pooled_prompt_2_embeds,
    )

    denoised_latents = workflow.add_denoise_node(
        model=sdxl,
        scheduler=scheduler,
        base_model_inputs=DiffusionModelInputs(
            latents=latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            height=height,
            width=width,
        ),
        scheduler_inputs=SchedulerInputs(
            num_inference_steps=num_inference_steps,
        ),
    )

    output_img = vae(latents=denoised_latents, mode="decode_latents")

    # Define outputs
    workflow.add_output(output_img, name="output_img")

    return workflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", type=str, default="http://localhost:8000")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/project/infattllm/lyangbk/huggingface/stable-diffusion-xl-base-1.0",
    )
    args = parser.parse_args()
    server_url = args.server_url

    # Register workflow
    service_id = register_workflow(
        workflow=create_workflow(model_path=args.model_path),
        server_url=server_url,
    )
    print(f"Registered workflow with service ID: {service_id}")
