# python examples/register_sd15_txt2img_controlnet_depth_workflow.py
import argparse
import base64
import io

from diffusers.utils import load_image
from PIL import Image

from diffusionflow.interface import Workflow, register_workflow
from diffusionflow.interface.node_io import (
    AdapterInputs,
    DiffusionModelInputs,
    SchedulerInputs,
)
from diffusionflow.operators import (
    CLIP_SD15,
    Config,
    LatentsGenerator,
    PNDMScheduler,
    StableDiffusion15,
    StableDiffusion15ControlNetDepth,
    StableDiffusion15VAE,
)


def create_workflow(model_path: str, controlnet_model_path: str) -> Workflow:
    workflow = Workflow(name="sd15_txt2img_controlnet_depth_workflow")

    # Define model nodes
    latents_generator = LatentsGenerator()
    clip_sd15 = CLIP_SD15(Config(model_path=model_path))
    scheduler = PNDMScheduler(Config(model_path=model_path))
    sd15 = StableDiffusion15(Config(model_path=model_path))
    vae = StableDiffusion15VAE(Config(model_path=model_path))
    controlnet = StableDiffusion15ControlNetDepth(
        Config(model_path=controlnet_model_path)
    )

    # Define inputs
    seed = workflow.add_input(name="seed", data_type=int)
    num_channels_latents = workflow.add_input(
        name="num_channels_latents", data_type=int
    )
    prompt = workflow.add_input(name="prompt", data_type=str)
    height = workflow.add_input(name="height", data_type=int)
    width = workflow.add_input(name="width", data_type=int)
    num_inference_steps = workflow.add_input(name="num_inference_steps", data_type=int)
    # Used for ControlNet
    control_image = workflow.add_input(name="control_image", data_type=Image.Image)
    conditioning_scale = workflow.add_input(name="conditioning_scale", data_type=float)

    # Define connections
    latents = latents_generator(
        seed=seed, height=height, width=width, num_channels_latents=num_channels_latents
    )
    control_image = vae(
        image=control_image, height=height, width=width, mode="prepare_image"
    )
    prompt_embeds = clip_sd15(prompt=prompt)

    denoised_latents = workflow.add_denoise_node(
        model=sd15,
        scheduler=scheduler,
        base_model_inputs=DiffusionModelInputs(
            latents=latents,
            prompt_embeds=prompt_embeds,
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
    parser.add_argument(
        "--controlnet-model-path",
        type=str,
        default="/project/infattllm/lyangbk/huggingface/lllyasviel--sd-controlnet-depth",
    )

    args = parser.parse_args()
    server_url = args.server_url

    # Register workflow
    service_id = register_workflow(
        workflow=create_workflow(
            model_path=args.model_path,
            controlnet_model_path=args.controlnet_model_path,
        ),
        server_url=server_url,
    )
    print(f"Registered workflow with service ID: {service_id}")
