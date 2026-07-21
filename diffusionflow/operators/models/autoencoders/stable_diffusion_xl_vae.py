import logging
from typing import Any, Dict, List, Union

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.utils import load_image
from PIL.Image import Image

from diffusionflow.operators.base import Operator
from diffusionflow.operators.operator_ids import STABLE_DIFFUSION_XL_VAE_ID


# Adapted from https://github.com/huggingface/diffusers/blob/3579cd2bb7d2e8f8fa97b198a513f4e02ecccfc1/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L1264
@torch.no_grad()
def _decode_latents(
    vae: AutoencoderKL,
    image_processor: VaeImageProcessor,
    latents: torch.Tensor,
    output_type: str = "pil",
) -> Union[List[Image], np.ndarray]:
    _upcast_vae(vae)
    latents = latents.to(next(iter(vae.post_quant_conv.parameters())).dtype)
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents, return_dict=False)[0]
    vae.to(dtype=torch.float16)
    image = image_processor.postprocess(image, output_type=output_type)
    return image


def _upcast_vae(
    vae: AutoencoderKL,
):
    dtype = vae.dtype
    vae.to(dtype=torch.float32)
    use_torch_2_0_or_xformers = isinstance(
        vae.decoder.mid_block.attentions[0].processor,
        (AttnProcessor2_0,),
    )
    # if xformers or torch_2_0 is used attention block does not need
    # to be in float32 which can save lots of memory
    if use_torch_2_0_or_xformers:
        vae.post_quant_conv.to(dtype)
        vae.decoder.conv_in.to(dtype)
        vae.decoder.mid_block.to(dtype)


# Adapted from https://github.com/huggingface/diffusers/blob/3579cd2bb7d2e8f8fa97b198a513f4e02ecccfc1/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L1264
class StableDiffusionXLVAE(Operator):
    def setup_io(self):
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
        self.add_execution_mode(
            "preprocess_image",
            inputs={"init_image_path": str, "seed": int},
            outputs={"init_latents": torch.Tensor},
        )

    @property
    def id(self) -> str:
        return STABLE_DIFFUSION_XL_VAE_ID

    def initialize(
        self, model_path: str, device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.float16,
        ).to(device)

        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

        image_processor = VaeImageProcessor(
            vae_scale_factor=vae_scale_factor, do_convert_rgb=True
        )

        controlnet_image_processor = VaeImageProcessor(
            vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )

        return {
            "vae": vae,
            "image_processor": image_processor,
            "controlnet_image_processor": controlnet_image_processor,
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
        controlnet_image_processor = model_components["controlnet_image_processor"]

        if mode == "prepare_image":
            image = kwargs["image"]
            width = kwargs["width"]
            height = kwargs["height"]
            image = controlnet_image_processor.preprocess(
                image, height=height, width=width
            ).to(dtype=torch.float32)
            image = image.to(device=device, dtype=torch.float16)
            return {"image_embedding": image}

        elif mode == "decode_latents":
            latents = kwargs["latents"]
            image = _decode_latents(
                vae=vae, image_processor=image_processor, latents=latents
            )
            return {"image": image}

        elif mode == "preprocess_image":
            # to support img2img pipeline of nirvana
            init_image_path = kwargs["init_image_path"]
            image = load_image(init_image_path).convert("RGB")
            seed = kwargs["seed"]
            generator = torch.manual_seed(seed)
            image = image_processor.preprocess(image)
            image = image.to(device=device)

            vae = vae.to(dtype=torch.float32)
            init_latents = vae.encode(image).latent_dist.sample(generator)
            vae = vae.to(dtype=torch.float16)

            init_latents = init_latents.to(dtype=torch.float16)
            init_latents = vae.config.scaling_factor * init_latents
            init_latents = torch.cat([init_latents], dim=0)
            return {"init_latents": init_latents}
        else:
            raise ValueError(f"Invalid execution mode: {mode}")


if __name__ == "__main__":
    vae = StableDiffusionXLVAE()
    model_components = vae.initialize(
        model_path="/project/infattllm/lyangbk/huggingface/stable-diffusion-xl-base-1.0",
        device="cuda",
    )
    init_image_path = "./imgs/sdxl_img2img_init_image.png"
    result = vae.execute(
        model_components,
        device="cuda",
        mode="preprocess_image",
        image_path=init_image_path,
        seed=0,
    )["init_latents"]
    print(result.shape, result.dtype, result.device, result[0][0][0][:10])
    # print(result)
