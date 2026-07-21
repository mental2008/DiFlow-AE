import logging
import os
from typing import Any, Dict, List, Union

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders import AutoencoderKL
from PIL.Image import Image

from diffusionflow.operators.base import Operator
from diffusionflow.operators.operator_ids import STABLE_DIFFUSION_3_VAE_ID
from diffusionflow.operators.utils import test_model_memory_allocation


# Adapted from https://github.com/huggingface/diffusers/blob/3579cd2bb7d2e8f8fa97b198a513f4e02ecccfc1/src/diffusers/pipelines/controlnet_sd3/pipeline_stable_diffusion_3_controlnet.py#L679
@torch.no_grad()
def _prepare_image(
    vae: AutoencoderKL,
    image_processor: VaeImageProcessor,
    image: Image,
    width: int,
    height: int,
    # batch_size,
    # num_images_per_prompt,
    device: Union[str, torch.device],
    # dtype,
) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        pass
    else:
        image = image_processor.preprocess(image, height=height, width=width)

    image_batch_size = image.shape[0]

    # batch_size = 1 # TODO (Lingyun): batch_size should be fed by the input
    # num_images_per_prompt = 1 # TODO (Lingyun): num_images_per_prompt should be fed by the input
    dtype = torch.float16  # TODO (Lingyun): dtype should be fed by the input

    # if image_batch_size == 1:
    #     repeat_by = batch_size
    # else:
    #     # image batch size is the same as prompt batch size
    #     repeat_by = num_images_per_prompt

    # image = image.repeat_interleave(repeat_by, dim=0)

    image = image.to(device=device, dtype=dtype)

    image = vae.encode(image).latent_dist.sample()

    # TODO (Lingyun): Remove this once we have a way to get the shift factor from the VAE
    vae_shift_factor = 0  # instantx sd3 controlnet does not apply shift factor
    # vae_shift_factor = vae.config.shift_factor

    image = (image - vae_shift_factor) * vae.config.scaling_factor

    return image


# Adapted from https://github.com/huggingface/diffusers/blob/3579cd2bb7d2e8f8fa97b198a513f4e02ecccfc1/src/diffusers/pipelines/controlnet_sd3/pipeline_stable_diffusion_3_controlnet.py#L1233
@torch.no_grad()
def _decode_latents(
    vae: AutoencoderKL,
    image_processor: VaeImageProcessor,
    latents: torch.Tensor,
    output_type: str = "pil",
) -> Union[List[Image], np.ndarray]:
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    image = vae.decode(latents, return_dict=False)[0]
    image = image_processor.postprocess(image, output_type=output_type)

    return image


# Adapted from https://github.com/huggingface/diffusers/blob/3579cd2bb7d2e8f8fa97b198a513f4e02ecccfc1/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L147
class StableDiffusion3VAE(Operator):
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
        return STABLE_DIFFUSION_3_VAE_ID

    def initialize(
        self, model_path: Union[str, None], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        if model_path is not None and os.path.exists(model_path):
            vae = AutoencoderKL.from_pretrained(
                model_path,
                subfolder="vae",
                torch_dtype=torch.float16,
            ).to(device)
        else:
            config = self._default_dummy_config()
            vae = AutoencoderKL(**config).to(device=device, dtype=torch.float16)

        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

        return {
            "vae": vae,
            "image_processor": image_processor,
        }

    @staticmethod
    def _default_dummy_config() -> Dict[str, Any]:
        return {
            "act_fn": "silu",
            "block_out_channels": [128, 256, 512, 512],
            "down_block_types": [
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ],
            "force_upcast": True,
            "in_channels": 3,
            "latent_channels": 16,
            "latents_mean": None,
            "latents_std": None,
            "layers_per_block": 2,
            "norm_num_groups": 32,
            "out_channels": 3,
            "sample_size": 1024,
            "scaling_factor": 1.5305,
            "shift_factor": 0.0609,
            "up_block_types": [
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ],
            "use_post_quant_conv": False,
            "use_quant_conv": False,
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
        image_processor = model_components["image_processor"]

        if mode == "prepare_image":
            image = kwargs["image"]
            width = kwargs["width"]
            height = kwargs["height"]
            image = _prepare_image(
                vae=vae,
                image_processor=image_processor,
                image=image,
                width=width,
                height=height,
                device=device,
            )
            return {"image_embedding": image}
        elif mode == "decode_latents":
            latents = kwargs["latents"]
            image = _decode_latents(
                vae=vae, image_processor=image_processor, latents=latents
            )
            return {"image": image}
        else:
            raise ValueError(f"Invalid execution mode: {mode}")


if __name__ == "__main__":
    test_model_memory_allocation(model=StableDiffusion3VAE(), model_path=None)
