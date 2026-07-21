import logging
from typing import Any, Dict, Union

import torch
from diffusers.utils.torch_utils import randn_tensor

from diffusionflow.operators.base import Operator


class StableDiffusionXLLatentsGenerator(Operator):
    def setup_io(self):
        self.add_input("batch_size", int)
        self.add_input("num_channels_latents", int)
        self.add_input("height", int)
        self.add_input("width", int)
        self.add_input("dtype", torch.dtype)
        self.add_input("seed", int)
        self.add_input("latents", torch.Tensor)
        self.add_output("latents", torch.Tensor)

    @property
    def id(self) -> str:
        return "SDXLLatentsGenerator"

    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        **kwargs,
    ) -> Dict[str, Any]:
        
        batch_size = kwargs.get("batch_size", 1)
        num_channels_latents = kwargs.get("num_channels_latents", 4)
        height = kwargs.get("height", 1024)
        width = kwargs.get("width", 1024)
        dtype = kwargs.get("dtype", torch.float16)
        seed = kwargs.get("seed", 0)
        latents = kwargs.get("latents", None)
        vae_scale_factor = kwargs.get("vae_scale_factor", 8)

        if latents is not None:
            return {"latents": latents.to(device=device, dtype=dtype)}

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // vae_scale_factor,
            int(width) // vae_scale_factor,
        )
        logging.debug(f"[LatentsGenerator] Shape: {shape}")

        generator = torch.manual_seed(seed)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(device, str):
            device = torch.device(device)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return {"latents": latents}

if __name__ == "__main__":
    latents_generator = StableDiffusionXLLatentsGenerator()
    # SDXL
    result = latents_generator.execute(
        model_components={},
        device="cuda",
        batch_size=1,
        num_channels_latents=4,
        height=1024,
        width=1024,
        # init_noise_sigma=13.158469200134277,
    )
    print(result["latents"].shape)
    print(result["latents"][0][0][0][:10])