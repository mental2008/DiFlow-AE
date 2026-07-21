import logging
import time
from typing import Any, Dict, List, Union

import torch
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel

from diffusionflow.operators.models.diffusion_models.base_diffusion_model import (
    BaseDiffusionModel,
)
from diffusionflow.operators.operator_ids import STABLE_DIFFUSION_15_ID

logger = logging.getLogger(__name__)


class StableDiffusion15(BaseDiffusionModel):
    def setup_io(self):
        super().setup_io()
        # ! Suyi: hard coding the length of the list of controlnet input
        for i in range(12):
            self.add_input(
                "down_block_res_sample_{}".format(i), torch.Tensor, lazy=True
            )
        self.add_input("mid_block_res_sample", torch.Tensor, lazy=True)

    @property
    def id(self) -> str:
        return STABLE_DIFFUSION_15_ID

    def initialize(
        self, model_path: str, device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        unet = UNet2DConditionModel.from_pretrained(
            model_path,
            subfolder="unet",
            torch_dtype=torch.float16,
        ).to(device)

        return {"unet": unet}

    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        **kwargs,
    ) -> Dict[str, Any]:
        # Images with 512x512 resolution:
        # latents: (batch_size, 4, 64, 64)
        # prompt embeddings: (batch_size, 77, 768)

        unet = model_components["unet"]

        latents = kwargs["latents"]
        timestep = kwargs["timestep"]

        prompt_embeds = kwargs["prompt_embeds"]

        timestep = timestep.expand(latents.shape[0])

        down_block_res_samples = []
        for i in range(12):
            down_block_res_samples.append(
                kwargs.get("down_block_res_sample_{}".format(i), None)
            )
        if None in down_block_res_samples:
            down_block_res_samples = None

        mid_block_res_sample = kwargs.get("mid_block_res_sample", None)

        noise_pred = unet.stream_forward(
            latents,
            timestep,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            cross_attention_kwargs=None,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            added_cond_kwargs=None,
        )

        return {"noise_pred": noise_pred}
