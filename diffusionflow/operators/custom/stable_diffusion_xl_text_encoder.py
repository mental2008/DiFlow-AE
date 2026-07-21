import logging
from typing import Any, Dict, Union

import torch

from diffusionflow.operators.base import Operator
from diffusionflow.operators.operator_ids import STABLE_DIFFUSION_XL_TEXT_ENCODER_ID


class StableDiffusionXLTextEncoder(Operator):
    def setup_io(self):
        self.add_input("clip_prompt_embeds", torch.Tensor)
        # self.add_input("clip_pooled_prompt_embeds", torch.Tensor)
        self.add_input("clip_prompt_2_embeds", torch.Tensor)
        self.add_input("clip_pooled_prompt_2_embeds", torch.Tensor)
        self.add_output("prompt_embeds", torch.Tensor)
        self.add_output("pooled_prompt_embeds", torch.Tensor)

    @property
    def id(self) -> str:
        return STABLE_DIFFUSION_XL_TEXT_ENCODER_ID

    # Adapted from https://github.com/huggingface/diffusers/blob/b75b204a584e29ebf4e80a61be11458e9ed56e3e/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L288
    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        **kwargs,
    ) -> Dict[str, Any]:
        # clip_prompt_embeds: (batch_size, 77, 768)
        # clip_prompt_2_embeds: (batch_size, 77, 1280)
        # clip_pooled_prompt_2_embeds: (batch_size, 1280)

        clip_prompt_embeds = kwargs["clip_prompt_embeds"]
        # clip_pooled_prompt_embeds = kwargs["clip_pooled_prompt_embeds"]
        clip_prompt_2_embeds = kwargs["clip_prompt_2_embeds"]
        clip_pooled_prompt_2_embeds = kwargs["clip_pooled_prompt_2_embeds"]

        prompt_embeds = torch.cat([clip_prompt_embeds, clip_prompt_2_embeds], dim=-1)

        pooled_prompt_embeds = clip_pooled_prompt_2_embeds

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }
