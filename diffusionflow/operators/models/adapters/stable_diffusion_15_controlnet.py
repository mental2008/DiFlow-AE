from typing import Any, Dict, Union

import torch
from diffusers.models.controlnets.controlnet import ControlNetModel

from diffusionflow.operators.models.adapters.base_adapter import BaseAdapter
from diffusionflow.operators.operator_ids import (
    STABLE_DIFFUSION_15_CONTROLNET_CANNY_ID,
    STABLE_DIFFUSION_15_CONTROLNET_DEPTH_ID,
)


class StableDiffusion15ControlNet(BaseAdapter):
    def setup_io(self):
        super().setup_io()
        self.add_input("controlnet_cond", torch.Tensor)
        self.add_input("conditioning_scale", float)
        # SD15 ControlNet outputs 12 blocks (4 down blocks * 3 layers each)
        for i in range(12):
            self.add_output(
                "down_block_res_sample_{}".format(i), torch.Tensor, lazy=True
            )
        self.add_output("mid_block_res_sample", torch.Tensor, lazy=True)

    @property
    def id(self) -> str:
        return NotImplementedError("Subclasses must implement model_id")

    def initialize(
        self, model_path: str, device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        controlnet = ControlNetModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        ).to(device)

        return {"controlnet": controlnet}

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
        # controlnet_cond: (batch_size, 3, 512, 512)

        controlnet = model_components["controlnet"]

        latents = kwargs["latents"]
        timestep = kwargs["timestep"]
        timestep = timestep.expand(latents.shape[0])

        prompt_embeds = kwargs["prompt_embeds"]

        controlnet_cond = kwargs["controlnet_cond"]
        conditioning_scale = kwargs["conditioning_scale"]

        for data in controlnet.yield_control_block_samples(
            sample=latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
        ):
            yield data


class StableDiffusion15ControlNetCanny(StableDiffusion15ControlNet):
    @property
    def id(self) -> str:
        return STABLE_DIFFUSION_15_CONTROLNET_CANNY_ID


class StableDiffusion15ControlNetDepth(StableDiffusion15ControlNet):
    @property
    def id(self) -> str:
        return STABLE_DIFFUSION_15_CONTROLNET_DEPTH_ID
