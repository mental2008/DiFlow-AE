import logging
import time
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from diffusers.models import UNet2DConditionModel

from diffusionflow.operators.models.diffusion_models.base_diffusion_model import (
    BaseDiffusionModel,
)
from diffusionflow.operators.operator_ids import STABLE_DIFFUSION_XL_ID

logger = logging.getLogger(__name__)


def _get_add_time_ids(
    unet,
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
    passed_add_embed_dim = (
        # unet.config.addition_time_embed_dim * len(add_time_ids)
        # + text_encoder_projection_dim
        # ! Suyi: profile to hard code the unet.config.addition_time_embed_dim
        256 * len(add_time_ids)
        + text_encoder_projection_dim
    )
    # expected_add_embed_dim = unet.add_embedding.linear_1.in_features
    # ! Suyi: profile to hard code the unet.add_embedding.linear_1.in_features
    expected_add_embed_dim = 2816

    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`. \
                unet.config.addition_time_embed_dim: {unet.config.addition_time_embed_dim}, \
                text_encoder_projection_dim: {text_encoder_projection_dim}, \
                passed_add_embed_dim: {passed_add_embed_dim}, \
                expected_add_embed_dim: {expected_add_embed_dim}"
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype).to(device)
    return add_time_ids


class StableDiffusionXL(BaseDiffusionModel):
    def setup_io(self):
        super().setup_io()
        # ! Suyi: hard coding the length of the list of controlnet input
        for i in range(9):
            self.add_input(
                "down_block_res_sample_{}".format(i), torch.Tensor, lazy=True
            )
        self.add_input("mid_block_res_sample", torch.Tensor, lazy=True)

    @property
    def id(self) -> str:
        return STABLE_DIFFUSION_XL_ID

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

        ### For BAL ###
        if_use_bal = kwargs.get("if_use_bal", False)
        if if_use_bal:
            shm_dict = kwargs["shm_dict"]
            num_lora_model_repos = 1

            status = np.sum(shm_dict["start_loading_flag_np"])

            if status == num_lora_model_repos * 10:
                logging.debug("LoRA loading complete")
        ### For BAL ###

        unet = model_components["unet"]

        latents = kwargs["latents"]
        timestep = kwargs["timestep"]

        prompt_embeds = kwargs["prompt_embeds"]
        pooled_prompt_embeds = kwargs["pooled_prompt_embeds"]

        timestep = timestep.expand(latents.shape[0])

        height = kwargs.get("height", None)
        width = kwargs.get("width", None)
        original_size = (height, width)
        target_size = (height, width)

        text_encoder_projection_dim = pooled_prompt_embeds.shape[-1]
        add_time_ids = _get_add_time_ids(
            unet=unet,
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

        down_block_res_samples = []
        mid_block_res_sample = None
        for i in range(9):
            if "down_block_res_sample_{}".format(i) in kwargs:
                down_block_res_samples.append(
                    kwargs["down_block_res_sample_{}".format(i)]
                )
            else:
                down_block_res_samples.append(None)
        if "mid_block_res_sample" in kwargs:
            mid_block_res_sample = kwargs["mid_block_res_sample"]
        # Suyi: fall back to no ControlNet
        if (
            any(sample is None for sample in down_block_res_samples)
            or mid_block_res_sample is None
        ):
            down_block_res_samples = None
            mid_block_res_sample = None

        # sdxl
        noise_pred = unet.stream_forward(
            latents,
            timestep,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            cross_attention_kwargs=None,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            added_cond_kwargs=added_cond_kwargs,
        )

        return {"noise_pred": noise_pred}


if __name__ == "__main__":
    sdxl = StableDiffusionXL()
    model_components = sdxl.initialize(
        model_path="/project/infattllm/lyangbk/huggingface/stable-diffusion-xl-base-1.0",
        device="cuda",
    )
    result = sdxl.execute(
        model_components=model_components,
        device="cuda",
        latents=torch.randn(1, 4, 1024, 1024).to("cuda"),
        timestep=torch.full((1,), 1000, dtype=torch.float16).to("cuda"),
        prompt_embeds=torch.randn(1, 77, 1024).to("cuda"),
        pooled_prompt_embeds=torch.randn(1, 1024).to("cuda"),
        height=1024,
        width=1024,
    )
    print(result["noise_pred"].shape)
    print(result["noise_pred"][0][0][0][:10])
