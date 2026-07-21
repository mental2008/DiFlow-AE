from typing import Any, Dict, Union

import torch

from diffusionflow.operators.base import Operator
from diffusionflow.operators.operator_ids import STABLE_DIFFUSION_15_TEXT_ENCODER_ID


# Deprecated: StableDiffusion15TextEncoder is deprecated and will be removed in the future
class StableDiffusion15TextEncoder(Operator):
    def setup_io(self):
        self.add_input("prompt_embeds", torch.Tensor)
        # self.add_input("negative_prompt_embeds", torch.Tensor)
        self.add_output("prompt_embeds", torch.Tensor)

    @property
    def id(self) -> str:
        return STABLE_DIFFUSION_15_TEXT_ENCODER_ID

    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        **kwargs,
    ) -> Dict[str, Any]:
        prompt_embeds = kwargs["prompt_embeds"]
        # negative_prompt_embeds = kwargs["negative_prompt_embeds"]

        # prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return {
            "prompt_embeds": prompt_embeds,
        }
