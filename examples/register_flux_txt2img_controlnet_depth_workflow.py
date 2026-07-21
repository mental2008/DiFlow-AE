# python examples/register_flux_txt2img_controlnet_depth_workflow.py
import argparse
import base64
import io

import torch
from diffusers.utils import load_image
from PIL import Image

from diffusionflow.interface import Workflow, register_workflow
from diffusionflow.interface.node_io import (
    AdapterInputs,
    DiffusionModelInputs,
    SchedulerInputs,
)
from diffusionflow.operators import (
    CLIP_Flux,
    Config,
    Flux1Dev,
    Flux1DevControlNetDepth,
    Flux1VAE,
    FluxFlowMatchEulerDiscreteScheduler,
    FluxLatentsGenerator,
    T5_Flux,
)


def create_workflow(model_path: str, controlnet_model_path: str) -> Workflow:
    workflow = Workflow(name="flux_txt2img_controlnet_depth_workflow")

    # Define model nodes
    latents_generator = FluxLatentsGenerator()
    clip_flux = CLIP_Flux(Config(model_path=model_path))
    t5_flux = T5_Flux(Config(model_path=model_path))
    scheduler = FluxFlowMatchEulerDiscreteScheduler(Config(model_path=model_path))
    flux1_dev = Flux1Dev(Config(model_path=model_path))
    vae = Flux1VAE(Config(model_path=model_path))
    controlnet = Flux1DevControlNetDepth(Config(model_path=controlnet_model_path))

    # Define inputs
    seed = workflow.add_input(name="seed", data_type=int)
    prompt = workflow.add_input(name="prompt", data_type=str)
    height = workflow.add_input(name="height", data_type=int)
    width = workflow.add_input(name="width", data_type=int)
    num_inference_steps = workflow.add_input(name="num_inference_steps", data_type=int)
    # Used for ControlNet
    control_image = workflow.add_input(name="control_image", data_type=Image.Image)
    conditioning_scale = workflow.add_input(name="conditioning_scale", data_type=float)
    # Used for Flux model
    guidance_scale = workflow.add_input(name="guidance_scale", data_type=float)

    # Define connections
    latents = latents_generator(height=height, width=width, seed=seed)

    control_image = vae(
        image=control_image, height=height, width=width, mode="prepare_image"
    )

    clip_prompt_embeds = clip_flux(prompt=prompt)
    t5_prompt_embeds = t5_flux(prompt=prompt)

    denoised_latents = workflow.add_denoise_node(
        model=flux1_dev,
        scheduler=scheduler,
        base_model_inputs=DiffusionModelInputs(
            latents=latents,
            prompt_embeds=t5_prompt_embeds,
            pooled_prompt_embeds=clip_prompt_embeds,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        ),
        adapters=[controlnet],
        adapter_inputs=[
            AdapterInputs(
                controlnet_cond=control_image,
                conditioning_scale=conditioning_scale,
            )
        ],
        scheduler_inputs=SchedulerInputs(
            num_inference_steps=num_inference_steps,
        ),
    )

    output_img = vae(
        latents=denoised_latents, height=height, width=width, mode="decode_latents"
    )

    # Define outputs
    workflow.add_output(output_img, name="output_img")

    return workflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", type=str, default="http://localhost:8000")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/project/infattllm/lyangbk/huggingface/FLUX.1-dev",
    )
    parser.add_argument(
        "--controlnet-model-path",
        type=str,
        default="/project/infattllm/lyangbk/huggingface/Xlabs-AI--flux-controlnet-depth-diffusers",
    )

    args = parser.parse_args()
    server_url = args.server_url

    # Register workflow
    service_id = register_workflow(
        workflow=create_workflow(
            model_path=args.model_path, controlnet_model_path=args.controlnet_model_path
        ),
        server_url=server_url,
    )
    print(f"Registered workflow with service ID: {service_id}")
