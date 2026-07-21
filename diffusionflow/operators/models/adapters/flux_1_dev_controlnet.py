import logging
import os
from typing import Any, Dict, Union

import torch
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel

from diffusionflow.operators.flux_utils import (
    prepare_latent_image_ids,
    prepare_text_ids,
)
from diffusionflow.operators.models.adapters.base_adapter import BaseAdapter
from diffusionflow.operators.operator_ids import (
    FLUX_1_DEV_CONTROLNET_CANNY_ID,
    FLUX_1_DEV_CONTROLNET_DEPTH_ID,
)
from diffusionflow.operators.utils import test_model_memory_allocation

logger = logging.getLogger(__name__)


class Flux1DevControlNet(BaseAdapter):
    def setup_io(self):
        super().setup_io()
        self.add_input("controlnet_cond", torch.Tensor)
        self.add_input("conditioning_scale", float)
        # Suyi: using /project/infattllm/lyangbk/huggingface/Xlabs-AI--flux-controlnet-canny-diffusers
        for i in range(2):
            self.add_output(
                "control_block_sample_{}".format(i), torch.Tensor, lazy=True
            )

    @property
    def id(self) -> str:
        raise NotImplementedError("Subclasses must implement model_id")

    def initialize(
        self, model_path: Union[str, None], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        if model_path is not None and os.path.exists(model_path):
            controlnet = FluxControlNetModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to(device)
        else:
            config = self._default_dummy_config()
            controlnet = FluxControlNetModel(**config).to(
                device=device, dtype=torch.bfloat16
            )

        return {"controlnet": controlnet}

    @staticmethod
    def _default_dummy_config() -> Dict[str, Any]:
        return {
            "attention_head_dim": 128,
            "axes_dims_rope": [16, 56, 56],
            "guidance_embeds": True,
            "in_channels": 64,
            "conditioning_embedding_channels": 16,
            "joint_attention_dim": 4096,
            "num_attention_heads": 24,
            "num_layers": 2,
            "num_mode": None,
            "num_single_layers": 0,
            "patch_size": 2,
            "pooled_projection_dim": 768,
        }

    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        **kwargs,
    ) -> Dict[str, Any]:
        controlnet = model_components["controlnet"]

        latents = kwargs["latents"]
        timestep = kwargs["timestep"]

        guidance = kwargs["guidance"]

        prompt_embeds = kwargs["prompt_embeds"]
        pooled_prompt_embeds = kwargs["pooled_prompt_embeds"]

        height = kwargs["height"]
        width = kwargs["width"]
        dtype = latents.dtype
        vae_scale_factor = kwargs.get("vae_scale_factor", 8)

        text_ids = prepare_text_ids(prompt_embeds.shape[1], device, dtype)
        latent_image_ids = prepare_latent_image_ids(
            height // (vae_scale_factor * 2),
            width // (vae_scale_factor * 2),
            device,
            dtype,
        )

        controlnet_cond = kwargs["controlnet_cond"]
        conditioning_scale = kwargs["conditioning_scale"]

        for data in controlnet.yield_controlnet_block_samples(
            hidden_states=latents,
            controlnet_cond=controlnet_cond,
            controlnet_mode=None,
            conditioning_scale=conditioning_scale,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
        ):
            yield data


class Flux1DevControlNetDepth(Flux1DevControlNet):
    @property
    def id(self) -> str:
        return FLUX_1_DEV_CONTROLNET_DEPTH_ID


class Flux1DevControlNetCanny(Flux1DevControlNet):
    @property
    def id(self) -> str:
        return FLUX_1_DEV_CONTROLNET_CANNY_ID


if __name__ == "__main__":
    test_model_memory_allocation(model=Flux1DevControlNetDepth(), model_path=None)
