import logging
from typing import Any, Dict, List, Union

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders import AutoencoderKL
from PIL.Image import Image

from diffusionflow.operators.base import Operator
from diffusionflow.operators.operator_ids import STABLE_DIFFUSION_15_VAE_ID


# Adapted from https://github.com/huggingface/diffusers/blob/3579cd2bb7d2e8f8fa97b198a513f4e02ecccfc1/src/diffusers/pipelines/controlnet_sd3/pipeline_stable_diffusion_3_controlnet.py#L1233
@torch.no_grad()
def _decode_latents(
    vae: AutoencoderKL,
    image_processor: VaeImageProcessor,
    latents: torch.Tensor,
    output_type: str = "pil",
) -> Union[List[Image], np.ndarray]:
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents, return_dict=False)[0]
    image = image_processor.postprocess(
        image, output_type=output_type, do_denormalize=[True] * image.shape[0]
    )

    return image


# Adapted from https://github.com/huggingface/diffusers/blob/3579cd2bb7d2e8f8fa97b198a513f4e02ecccfc1/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L147
class StableDiffusion15VAE(Operator):
    def setup_io(self):
        # Add execution modes with their IO specifications
        self.add_execution_mode(
            "prepare_image",
            inputs={
                "image": Image,
                "width": int,
                "height": int,
            },
            outputs={"image_embedding": torch.Tensor},
        )

        self.add_execution_mode(
            "decode_latents", inputs={"latents": torch.Tensor}, outputs={"image": Image}
        )

    @property
    def id(self) -> str:
        return STABLE_DIFFUSION_15_VAE_ID

    def initialize(
        self, model_path: str, device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.float16,
        ).to(device)

        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
        control_image_processor = VaeImageProcessor(
            vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        return {
            "vae": vae,
            "image_processor": image_processor,
            "control_image_processor": control_image_processor,
        }

    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        mode: str,
        **kwargs,
    ) -> Dict[str, Any]:
        vae = model_components["vae"]

        if mode == "prepare_image":
            control_image_processor = model_components["control_image_processor"]
            image = kwargs["image"]
            width = kwargs["width"]
            height = kwargs["height"]

            # Preprocess image
            if isinstance(image, torch.Tensor):
                pass
            else:
                image = control_image_processor.preprocess(
                    image, height=height, width=width
                )

            image = image.to(device=device, dtype=torch.float16)
            return {"image_embedding": image}

        elif mode == "decode_latents":
            image_processor = model_components["image_processor"]
            latents = kwargs["latents"]
            image = _decode_latents(
                vae=vae, image_processor=image_processor, latents=latents
            )
            return {"image": image}
        else:
            raise ValueError(f"Invalid execution mode: {mode}")
