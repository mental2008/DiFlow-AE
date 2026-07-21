from diffusers import (
    FluxPipeline,
)
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel

import torch

def convert_pipeline_name_to_instance(pipeline_name: str, device: str):
    if pipeline_name in ["flux_schnell_txt2img_workflow"]:
        model_path = "./models/FLUX.1-schnell"
        return FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        ).to(device)

    elif pipeline_name in ["flux_schnell_txt2img_controlnet_canny_workflow"]:
        model_path = "./models/FLUX.1-schnell"
        controlnet_model_path = "./models/Xlabs-AI--flux-controlnet-canny-diffusers"
        controlnet = FluxControlNetModel.from_pretrained(
            controlnet_model_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        return FluxControlNetPipeline.from_pretrained(
            model_path,
            controlnet=controlnet,
            torch_dtype=torch.bfloat16
        ).to(device)

    elif pipeline_name in ["flux_schnell_txt2img_controlnet_depth_workflow"]:
        model_path = "./models/FLUX.1-schnell"
        controlnet_model_path = "./models/Xlabs-AI--flux-controlnet-depth-diffusers"
        controlnet = FluxControlNetModel.from_pretrained(
            controlnet_model_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        return FluxControlNetPipeline.from_pretrained(
            model_path,
            controlnet=controlnet,
            torch_dtype=torch.bfloat16
        ).to(device)

    else:
        raise ValueError(f"Pipeline name {pipeline_name} not found")