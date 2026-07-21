import inspect
import os
import time
from typing import Any, Dict, Union

import torch
from diffusers.models import SD3ControlNetModel
from diffusers.models.embeddings import PatchEmbed

from diffusionflow.operators.models.adapters.base_adapter import BaseAdapter
from diffusionflow.operators.operator_ids import (
    STABLE_DIFFUSION_35_LARGE_CONTROLNET_CANNY_ID,
    STABLE_DIFFUSION_35_LARGE_CONTROLNET_DEPTH_ID,
)
from diffusionflow.operators.utils import test_model_memory_allocation

# Suyi: for the controlnet model: diffusers-internal-dev/sd35-controlnet-depth-8b
NUM_CONTROL_BLOCK_SAMPLES = 19


# Notes: This is for SD3.5 8b controlnet, which shares the pos_embed with the transformer
# we should have handled this in conversion script
def _get_pos_embed_from_transformer(pos_embed_state_dict_path: str = None):
    # transformer.config.sample_size: 128
    # transformer.config.patch_size: 2
    # transformer.config.in_channels: 16
    # transformer.inner_dim: 2432
    # transformer.config.pos_embed_max_size: 192
    pos_embed = PatchEmbed(
        height=128,  # transformer.config.sample_size,
        width=128,  # transformer.config.sample_size,
        patch_size=2,  # transformer.config.patch_size,
        in_channels=16,  # transformer.config.in_channels,
        embed_dim=2432,  # transformer.inner_dim,
        pos_embed_max_size=192,  # transformer.config.pos_embed_max_size,
    )
    if pos_embed_state_dict_path is not None and os.path.exists(
        pos_embed_state_dict_path
    ):
        print(f"Loading pos embed from {pos_embed_state_dict_path}")
        pos_embed.load_state_dict(torch.load(pos_embed_state_dict_path), strict=True)
    return pos_embed


class StableDiffusion35LargeControlNet(BaseAdapter):
    def setup_io(self):
        super().setup_io()
        self.add_input("controlnet_cond", torch.Tensor)
        self.add_input("conditioning_scale", float)
        # ! Suyi: hard coding the length of the list of controlnet output
        for i in range(NUM_CONTROL_BLOCK_SAMPLES):
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

        if (
            hasattr(controlnet.config, "use_pos_embed")
            and controlnet.config.use_pos_embed is False
        ):
            _default_pos_embed_state_dict_path = "/project/infattllm/lyangbk/huggingface/sd3_large_controlnet_pos_embed_state_dict/sd3_large_controlnet_pos_embed_state_dict.pth"
            if os.path.exists(_default_pos_embed_state_dict_path):
                pos_embed = _get_pos_embed_from_transformer(
                    _default_pos_embed_state_dict_path
                )
            else:
                pos_embed = _get_pos_embed_from_transformer()
            controlnet.pos_embed = pos_embed.to(controlnet.dtype).to(controlnet.device)

        return {"controlnet": controlnet}

    @staticmethod
    def _default_dummy_config() -> Dict[str, Any]:
        return {
            "attention_head_dim": 64,
            "caption_projection_dim": 2048,
            "dual_attention_layers": [],
            "extra_conditioning_channels": 0,
            "force_zeros_for_pooled_projection": False,
            "in_channels": 16,
            "joint_attention_dim": None,
            "num_attention_heads": 38,
            "num_layers": 19,
            "out_channels": 16,
            "patch_size": 2,
            "pooled_projection_dim": 2048,
            "pos_embed_max_size": None,
            "pos_embed_type": None,
            "qk_norm": None,
            "sample_size": 128,
            "use_pos_embed": False,
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
        # pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

        timestep = timestep.expand(latents.shape[0])
        controlnet_cond = kwargs["controlnet_cond"]

        conditioning_scale = kwargs["conditioning_scale"]

        for data in controlnet.yield_controlnet_block_samples(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=None,  # SD35 official 8b controlnet does not use encoder_hidden_states
            pooled_projections=pooled_prompt_embeds,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
            joint_attention_kwargs=None,
        ):
            yield data


# if __name__ == "__main__":
#     controlnet = StableDiffusion35LargeControlNet()
#     model_path = "/project/infattllm/lyangbk/huggingface/diffusers-internal-dev--sd35-controlnet-depth-8b"
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


class StableDiffusion35LargeControlNetCanny(StableDiffusion35LargeControlNet):
    @property
    def id(self) -> str:
        return STABLE_DIFFUSION_35_LARGE_CONTROLNET_CANNY_ID


class StableDiffusion35LargeControlNetDepth(StableDiffusion35LargeControlNet):
    @property
    def id(self) -> str:
        return STABLE_DIFFUSION_35_LARGE_CONTROLNET_DEPTH_ID


if __name__ == "__main__":
    # test_model_memory_allocation(
    #     model=StableDiffusion35LargeControlNetDepth(), model_path=None
    # )

    print(
        f"Memory allocated before getting pos embed: {torch.cuda.memory_allocated() / (1024**3)} GiB"
    )
    pos_embed = _get_pos_embed_from_transformer()
    # pos_embed = _get_pos_embed_from_transformer(pos_embed_state_dict_path="/project/infattllm/lyangbk/huggingface/sd3_large_controlnet_pos_embed_state_dict/sd3_large_controlnet_pos_embed_state_dict.pth")
    pos_embed = pos_embed.to(torch.float16).to(torch.device("cuda"))
    print(
        f"Memory allocated after getting pos embed: {torch.cuda.memory_allocated() / (1024**3)} GiB"
    )
