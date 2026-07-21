import inspect
import os
import time
from typing import Any, Dict, Union

import torch
from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetModel

from diffusionflow.operators.models.adapters.base_adapter import BaseAdapter
from diffusionflow.operators.operator_ids import (
    STABLE_DIFFUSION_3_CONTROLNET_CANNY_ID,
    STABLE_DIFFUSION_3_CONTROLNET_POSE_ID,
)
from diffusionflow.operators.utils import test_model_memory_allocation


class StableDiffusion3ControlNet(BaseAdapter):
    def setup_io(self):
        super().setup_io()
        self.add_input("controlnet_cond", torch.Tensor)
        self.add_input("conditioning_scale", float)
        # ! Suyi: hard coding the length of the list of controlnet output
        for i in range(6):
            self.add_output(
                "control_block_sample_{}".format(i), torch.Tensor, lazy=True
            )

    @property
    def id(self) -> str:
        return NotImplementedError("Subclasses must implement model_id")

    def initialize(
        self, model_path: Union[str, None], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        if model_path is not None and os.path.exists(model_path):
            controlnet = SD3ControlNetModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            ).to(device)
        else:
            config = self._default_dummy_config()
            controlnet = SD3ControlNetModel(**config).to(
                device=device, dtype=torch.float16
            )

        return {"controlnet": controlnet}

    @staticmethod
    def _default_dummy_config() -> Dict[str, Any]:
        return {
            "attention_head_dim": 64,
            "caption_projection_dim": 1536,
            "in_channels": 16,
            "joint_attention_dim": 4096,
            "num_attention_heads": 24,
            "num_layers": 6,
            "out_channels": 16,
            "patch_size": 2,
            "pooled_projection_dim": 2048,
            "pos_embed_max_size": 192,
            "sample_size": 128,
        }

    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        **kwargs,
    ):
        """
        Generator version of execute: yields (name, tensor) for each control block sample.
        """
        controlnet = model_components["controlnet"]

        latents = kwargs["latents"]
        timestep = kwargs["timestep"]

        prompt_embeds = kwargs["prompt_embeds"]
        pooled_prompt_embeds = kwargs["pooled_prompt_embeds"]
        # instantx sd3 controlnet used zero pooled projection
        pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

        timestep = timestep.expand(latents.shape[0])
        controlnet_cond = kwargs["controlnet_cond"]

        conditioning_scale = kwargs["conditioning_scale"]

        for data in controlnet.yield_controlnet_block_samples(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
            joint_attention_kwargs=None,
        ):
            yield data


# if __name__ == "__main__":
#     controlnet = StableDiffusion3ControlNet()
#     model_path = "/project/infattllm/lyangbk/huggingface/sd3-controlnet-canny"
#     device = "cuda"
#     model_components = controlnet.initialize(model_path=model_path, device=device)

#     batch_size, channels, height, width = 1, 16, 128, 128
#     latents = torch.randn(batch_size, channels, height, width, dtype=torch.float16).to(
#         device
#     )
#     timestep = torch.randint(0, 1000, (batch_size,)).to(device)
#     prompt_embeds = torch.randn(batch_size, 333, 4096, dtype=torch.float16).to(device)
#     pooled_prompt_embeds = torch.randn(batch_size, 2048, dtype=torch.float16).to(device)
#     controlnet_cond = torch.randn(
#         batch_size, channels, height, width, dtype=torch.float16
#     ).to(device)
#     conditioning_scale = 0.7

#     start_time = time.time()
#     is_generator = inspect.isgeneratorfunction(controlnet.execute)
#     print(f"is generator function: {is_generator}")
#     end_time = time.time()
#     print(f"Time taken to check is generator function: {end_time - start_time} seconds")

#     result = controlnet.execute(
#         model_components=model_components,
#         device=device,
#         latents=latents,
#         timestep=timestep,
#         prompt_embeds=prompt_embeds,
#         pooled_prompt_embeds=pooled_prompt_embeds,
#         controlnet_cond=controlnet_cond,
#         conditioning_scale=conditioning_scale,
#     )
#     if is_generator:
#         start_time = time.time()
#         for data in result:
#             for name, sample in data.items():
#                 pass
#             end_time = time.time()
#             print(f"Time taken to yield data: {end_time - start_time} seconds")
#             start_time = time.time()
#     else:
#         for name, sample in result.items():
#             print(f"{name}: {sample}")


class StableDiffusion3ControlNetCanny(StableDiffusion3ControlNet):
    @property
    def id(self) -> str:
        return STABLE_DIFFUSION_3_CONTROLNET_CANNY_ID


class StableDiffusion3ControlNetPose(StableDiffusion3ControlNet):
    @property
    def id(self) -> str:
        return STABLE_DIFFUSION_3_CONTROLNET_POSE_ID


if __name__ == "__main__":
    test_model_memory_allocation(
        model=StableDiffusion3ControlNetCanny(), model_path=None
    )
