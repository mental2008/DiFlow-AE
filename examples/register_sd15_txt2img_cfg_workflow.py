# python examples/register_sd15_txt2img_cfg_workflow.py
import argparse
import base64
import io

from PIL import Image

from diffusionflow.interface import Workflow, register_workflow
from diffusionflow.interface.node_io import DiffusionModelInputs, SchedulerInputs
from diffusionflow.operators import (
    CLIP_SD15,
    Config,
    LatentsGenerator,
    PNDMScheduler,
    StableDiffusion15,
    StableDiffusion15VAE,
)


def create_workflow(model_path: str) -> Workflow:
    workflow = Workflow(name="sd15_txt2img_cfg_workflow")

    # Define model nodes
    latents_generator = LatentsGenerator()
    clip = CLIP_SD15(Config(model_path=model_path))
    scheduler = PNDMScheduler(Config(model_path=model_path))
    sd15 = StableDiffusion15(Config(model_path=model_path))
    vae = StableDiffusion15VAE(Config(model_path=model_path))

    # Define inputs
    seed = workflow.add_input(name="seed", data_type=int)
    num_channels_latents = workflow.add_input(
        name="num_channels_latents", data_type=int
    )
    prompt = workflow.add_input(name="prompt", data_type=str)
    negative_prompt = workflow.add_input(name="negative_prompt", data_type=str)
    height = workflow.add_input(name="height", data_type=int)
    width = workflow.add_input(name="width", data_type=int)
    num_inference_steps = workflow.add_input(name="num_inference_steps", data_type=int)
    guidance_scale = workflow.add_input(name="guidance_scale", data_type=float)

    # Define connections
    latents = latents_generator(
        seed=seed, height=height, width=width, num_channels_latents=num_channels_latents
    )

    # Text encoding
    prompt_embeds = clip(prompt=prompt)
    negative_prompt_embeds = clip(prompt=negative_prompt)

    denoised_latents = workflow.add_denoise_node(
        model=sd15,
        scheduler=scheduler,
        base_model_inputs=DiffusionModelInputs(
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        ),
        scheduler_inputs=SchedulerInputs(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
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
        default="/project/infattllm/lyangbk/huggingface/stable-diffusion-v1-5",
    )
    args = parser.parse_args()
    server_url = args.server_url

    # Register workflow
    service_id = register_workflow(
        workflow=create_workflow(model_path=args.model_path),
        server_url=server_url,
    )
    print(f"Registered workflow with service ID: {service_id}")
