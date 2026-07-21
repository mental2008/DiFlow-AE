from typing import Any, Dict, Union

import torch

from diffusionflow.operators.base import Operator
from diffusionflow.operators.operator_ids import STABLE_DIFFUSION_3_TEXT_ENCODER_ID


class StableDiffusion3TextEncoder(Operator):
    def setup_io(self):
        self.add_input("clip_prompt_embeds", torch.Tensor)
        self.add_input("clip_pooled_prompt_embeds", torch.Tensor)
        self.add_input("clip_prompt_2_embeds", torch.Tensor)
        self.add_input("clip_pooled_prompt_2_embeds", torch.Tensor)
        self.add_input("t5_prompt_embeds", torch.Tensor)
        self.add_output("prompt_embeds", torch.Tensor)
        self.add_output("pooled_prompt_embeds", torch.Tensor)

    @property
    def id(self) -> str:
        return STABLE_DIFFUSION_3_TEXT_ENCODER_ID

    # Adapted from https://github.com/huggingface/diffusers/blob/b75b204a584e29ebf4e80a61be11458e9ed56e3e/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L343
    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        **kwargs,
    ) -> Dict[str, Any]:
        clip_prompt_embeds = kwargs["clip_prompt_embeds"]
        clip_pooled_prompt_embeds = kwargs["clip_pooled_prompt_embeds"]
        clip_prompt_embeds_2 = kwargs["clip_prompt_2_embeds"]
        clip_pooled_prompt_2_embeds = kwargs["clip_pooled_prompt_2_embeds"]
        t5_prompt_embeds = kwargs["t5_prompt_embeds"]

        clip_prompt_embeds = torch.cat(
            [clip_prompt_embeds, clip_prompt_embeds_2], dim=-1
        )

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds,
            (0, t5_prompt_embeds.shape[-1] - clip_prompt_embeds.shape[-1]),
        )

        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embeds], dim=-2)
        pooled_prompt_embeds = torch.cat(
            [clip_pooled_prompt_embeds, clip_pooled_prompt_2_embeds], dim=-1
        )

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }
