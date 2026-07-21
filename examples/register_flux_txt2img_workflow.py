# python examples/register_flux_txt2img_workflow.py
import argparse

from diffusionflow.interface import Workflow, register_workflow
from diffusionflow.interface.node_io import DiffusionModelInputs, SchedulerInputs
from diffusionflow.operators import (
    CLIP_Flux,
    Config,
    Flux1Dev,
    Flux1VAE,
    FluxFlowMatchEulerDiscreteScheduler,
    FluxLatentsGenerator,
    T5_Flux,
)


def create_workflow(model_path: str) -> Workflow:
    workflow = Workflow(name="flux_txt2img_workflow")

    # Define model nodes
    latents_generator = FluxLatentsGenerator()
    clip_flux = CLIP_Flux(Config(model_path=model_path))
    t5_flux = T5_Flux(Config(model_path=model_path))
    scheduler = FluxFlowMatchEulerDiscreteScheduler(Config(model_path=model_path))
    flux = Flux1Dev(Config(model_path=model_path))
    vae = Flux1VAE(Config(model_path=model_path))

    # Define inputs
    seed = workflow.add_input(name="seed", data_type=int)
    prompt = workflow.add_input(name="prompt", data_type=str)
    height = workflow.add_input(name="height", data_type=int)
    width = workflow.add_input(name="width", data_type=int)
    num_inference_steps = workflow.add_input(name="num_inference_steps", data_type=int)
    guidance_scale = workflow.add_input(name="guidance_scale", data_type=float)

    # Define connections
    # Generate latents and latent image IDs
    latents = latents_generator(height=height, width=width, seed=seed)

    # Text encoding
    clip_prompt_embeds = clip_flux(prompt=prompt)
    t5_prompt_embeds = t5_flux(prompt=prompt)

    # Denoising process
    denoised_latents = workflow.add_denoise_node(
        model=flux,
        scheduler=scheduler,
        base_model_inputs=DiffusionModelInputs(
            latents=latents,
            prompt_embeds=t5_prompt_embeds,
            pooled_prompt_embeds=clip_prompt_embeds,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        ),
        scheduler_inputs=SchedulerInputs(
            num_inference_steps=num_inference_steps,
        ),
    )

    # VAE decode
    output_img = vae(
        latents=denoised_latents, mode="decode_latents", height=height, width=width
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
    args = parser.parse_args()
    server_url = args.server_url

    # Register workflow
    service_id = register_workflow(
        workflow=create_workflow(model_path=args.model_path),
        server_url=server_url,
    )
    print(f"Registered workflow with service ID: {service_id}")
