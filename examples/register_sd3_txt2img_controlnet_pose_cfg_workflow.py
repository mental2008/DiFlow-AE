# python examples/register_sd3_txt2img_controlnet_pose_cfg_workflow.py
import argparse

from PIL import Image

from diffusionflow.interface import Workflow, register_workflow
from diffusionflow.interface.node_io import (
    AdapterInputs,
    DiffusionModelInputs,
    SchedulerInputs,
)
from diffusionflow.operators import (
    CLIP_G,
    CLIP_L,
    T5,
    Config,
    FlowMatchEulerDiscreteScheduler,
    LatentsGenerator,
    StableDiffusion3,
    StableDiffusion3ControlNetPose,
    StableDiffusion3TextEncoder,
    StableDiffusion3VAE,
)


def create_workflow(model_path: str, controlnet_model_path: str) -> Workflow:
    workflow = Workflow(name="sd3_txt2img_controlnet_pose_cfg_workflow")

    # Define model nodes
    latents_generator = LatentsGenerator()
    clip_l = CLIP_L(Config(model_path=model_path))
    clip_g = CLIP_G(Config(model_path=model_path))
    t5 = T5(Config(model_path=model_path))
    text_encoder = StableDiffusion3TextEncoder()
    scheduler = FlowMatchEulerDiscreteScheduler(Config(model_path=model_path))
    sd3 = StableDiffusion3(Config(model_path=model_path))
    vae = StableDiffusion3VAE(Config(model_path=model_path))
    controlnet = StableDiffusion3ControlNetPose(
        Config(model_path=controlnet_model_path)
    )

    # Define inputs
    seed = workflow.add_input(name="seed", data_type=int)
    prompt = workflow.add_input(name="prompt", data_type=str)
    negative_prompt = workflow.add_input(name="negative_prompt", data_type=str)
    height = workflow.add_input(name="height", data_type=int)
    width = workflow.add_input(name="width", data_type=int)
    num_inference_steps = workflow.add_input(name="num_inference_steps", data_type=int)
    guidance_scale = workflow.add_input(name="guidance_scale", data_type=float)
    # Used for ControlNet
    control_image = workflow.add_input(name="control_image", data_type=Image.Image)
    conditioning_scale = workflow.add_input(name="conditioning_scale", data_type=float)

    # Define connections
    latents = latents_generator(seed=seed, height=height, width=width)
    control_image = vae(
        image=control_image, height=height, width=width, mode="prepare_image"
    )
    clip_prompt_embeds, clip_pooled_prompt_embeds = clip_l(prompt=prompt)
    clip_negative_prompt_embeds, clip_negative_pooled_prompt_embeds = clip_l(
        prompt=negative_prompt
    )
    clip_prompt_2_embeds, clip_pooled_prompt_2_embeds = clip_g(prompt=prompt)
    clip_negative_prompt_2_embeds, clip_negative_pooled_prompt_2_embeds = clip_g(
        prompt=negative_prompt
    )
    t5_prompt_embeds = t5(prompt=prompt)
    t5_negative_prompt_embeds = t5(prompt=negative_prompt)
    prompt_embeds, pooled_prompt_embeds = text_encoder(
        clip_prompt_embeds=clip_prompt_embeds,
        clip_pooled_prompt_embeds=clip_pooled_prompt_embeds,
        clip_prompt_2_embeds=clip_prompt_2_embeds,
        clip_pooled_prompt_2_embeds=clip_pooled_prompt_2_embeds,
        t5_prompt_embeds=t5_prompt_embeds,
    )
    negative_prompt_embeds, negative_pooled_prompt_embeds = text_encoder(
        clip_prompt_embeds=clip_negative_prompt_embeds,
        clip_pooled_prompt_embeds=clip_negative_pooled_prompt_embeds,
        clip_prompt_2_embeds=clip_negative_prompt_2_embeds,
        clip_pooled_prompt_2_embeds=clip_negative_pooled_prompt_2_embeds,
        t5_prompt_embeds=t5_negative_prompt_embeds,
    )

    denoised_latents = workflow.add_denoise_node(
        model=sd3,
        scheduler=scheduler,
        base_model_inputs=DiffusionModelInputs(
            latents=latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
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
        default="/project/infattllm/lyangbk/huggingface/stable-diffusion-3-medium-diffusers",
    )
    parser.add_argument(
        "--controlnet-model-path",
        type=str,
        default="/project/infattllm/lyangbk/huggingface/sd3-controlnet-pose",
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
