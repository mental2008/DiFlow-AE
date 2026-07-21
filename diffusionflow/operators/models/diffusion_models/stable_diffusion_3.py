import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import torch
from diffusers.models.transformers import SD3Transformer2DModel

from diffusionflow.operators.models.diffusion_models.base_diffusion_model import (
    BaseDiffusionModel,
)
from diffusionflow.operators.operator_ids import STABLE_DIFFUSION_3_ID
from diffusionflow.operators.utils import test_model_memory_allocation

logger = logging.getLogger(__name__)


class StableDiffusion3(BaseDiffusionModel):
    def setup_io(self):
        super().setup_io()
        # self.add_input("block_samples", List[torch.Tensor])
        # ! Suyi: hard coding the length of the list of controlnet input
        for i in range(6):
            self.add_input("control_block_sample_{}".format(i), torch.Tensor, lazy=True)

    @property
    def id(self) -> str:
        return STABLE_DIFFUSION_3_ID

    def initialize(
        self, model_path: Union[str, None], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        if model_path is not None and os.path.exists(model_path):
            logger.info(
                "Loading StableDiffusion3 transformer weights from %s", model_path
            )
            transformer = SD3Transformer2DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=torch.float16,
            ).to(device)
        else:
            logger.info(
                "Initializing StableDiffusion3 transformer with dummy SD3Transformer2DModel weights."
            )
            config = self._default_dummy_config()
            transformer = SD3Transformer2DModel(**config).to(
                device=device, dtype=torch.float16
            )

        return {"transformer": transformer}

    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        **kwargs,
    ) -> Dict[str, Any]:
        transformer = model_components["transformer"]

        latents = kwargs["latents"]
        timestep = kwargs["timestep"]

        prompt_embeds = kwargs["prompt_embeds"]
        pooled_prompt_embeds = kwargs["pooled_prompt_embeds"]

        # Get control block samples - they are now wrapped functions from the worker
        block_samples = [
            kwargs.get("control_block_sample_{}".format(i), None) for i in range(6)
        ]

        # Check if any control block samples are available
        if all(sample is None for sample in block_samples):
            block_samples = None

        timestep = timestep.expand(latents.shape[0])

        noise_pred = transformer.stream_forward(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            block_controlnet_hidden_states=block_samples,
            joint_attention_kwargs=None,
        )

        return {"noise_pred": noise_pred}

    @staticmethod
    def _default_dummy_config() -> Dict[str, Any]:
        return {
            "attention_head_dim": 64,
            "caption_projection_dim": 1536,
            "in_channels": 16,
            "joint_attention_dim": 4096,
            "num_attention_heads": 24,
            "num_layers": 24,
            "out_channels": 16,
            "patch_size": 2,
            "pooled_projection_dim": 2048,
            "pos_embed_max_size": 192,
            "sample_size": 128,
        }


if __name__ == "__main__":
    test_model_memory_allocation(model=StableDiffusion3(), model_path=None)
