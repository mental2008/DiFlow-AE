import logging
from typing import Any, Dict, Tuple, Union

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import ControlNetModel

from diffusionflow.operators.models.adapters.base_adapter import BaseAdapter
from diffusionflow.operators.operator_ids import (
    STABLE_DIFFUSION_XL_CONTROLNET_CANNY_ID,
    STABLE_DIFFUSION_XL_CONTROLNET_DEPTH_ID,
)

logger = logging.getLogger(__name__)


def _get_add_time_ids(
    original_size,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    target_size: Tuple[int, int] = (0, 0),
    text_encoder_projection_dim=None,
    dtype=torch.float16,
    device: Union[str, torch.device] = "cuda",
):
    assert original_size is not None, f"original_size is {original_size}"
    assert target_size is not None, f"target_size is {target_size}"
    add_time_ids = list(original_size + crops_coords_top_left + target_size)

    assert (
        text_encoder_projection_dim is not None
    ), f"text_encoder_projection_dim is {text_encoder_projection_dim}"

    # ! Suyi: profile to hard code the unet.config.addition_time_embed_dim
    unet_config_addition_time_embed_dim = 256
    passed_add_embed_dim = (
        unet_config_addition_time_embed_dim * len(add_time_ids)
        + text_encoder_projection_dim
    )
    # expected_add_embed_dim = unet.add_embedding.linear_1.in_features
    # ! Suyi: profile to hard code the unet.add_embedding.linear_1.in_features
    expected_add_embed_dim = 2816

    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`. \
                unet.config.addition_time_embed_dim: {unet_config_addition_time_embed_dim}, \
                text_encoder_projection_dim: {text_encoder_projection_dim}, \
                passed_add_embed_dim: {passed_add_embed_dim}, \
                expected_add_embed_dim: {expected_add_embed_dim}"
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype).to(device)
    return add_time_ids


class StableDiffusionXLControlNet(BaseAdapter):
    def setup_io(self):
        super().setup_io()
        self.add_input("controlnet_cond", torch.Tensor)
        self.add_input("conditioning_scale", float)
        # ! Suyi: hard coding controlnet output
        for i in range(9):
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
        controlnet = model_components["controlnet"]

        latents = kwargs["latents"]
        timestep = kwargs["timestep"]

        prompt_embeds = kwargs["prompt_embeds"]
        pooled_prompt_embeds = kwargs["pooled_prompt_embeds"]

        timestep = timestep.expand(latents.shape[0])

        controlnet_cond = kwargs["controlnet_cond"]
        conditioning_scale = kwargs.get("conditioning_scale", 0.5)

        height = kwargs.get("height", None)
        width = kwargs.get("width", None)
        original_size = (height, width)
        target_size = (height, width)

        text_encoder_projection_dim = pooled_prompt_embeds.shape[-1]
        add_time_ids = _get_add_time_ids(
            original_size=original_size,
            target_size=target_size,
            text_encoder_projection_dim=text_encoder_projection_dim,
            dtype=prompt_embeds.dtype,
            device=device,
        )

        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids.repeat(latents.shape[0], 1),
        }

        for data in controlnet.yield_control_block_samples(
            latents,
            timestep,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
            guess_mode=False,
            added_cond_kwargs=added_cond_kwargs,
        ):
            yield data


class StableDiffusionXLControlNetCanny(StableDiffusionXLControlNet):
    @property
    def id(self) -> str:
        return STABLE_DIFFUSION_XL_CONTROLNET_CANNY_ID


class StableDiffusionXLControlNetDepth(StableDiffusionXLControlNet):
    @property
    def id(self) -> str:
        return STABLE_DIFFUSION_XL_CONTROLNET_DEPTH_ID
