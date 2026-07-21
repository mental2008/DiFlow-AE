from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import torch

if TYPE_CHECKING:
    from diffusionflow.operators.base import Operator


@dataclass
class Config:
    model_path: str | None = None


def test_model_memory_allocation(
    model: "Operator",
    model_path: Union[str, None] = None,
):
    """
    Helper function to test memory allocation before and after model initialization.

    Args:
        model: Model to test
        model_path: Optional path to model weights (None for dummy weights)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for memory allocation testing")

    device = torch.device("cuda")
    print(f"Device: {device}")

    memory_before = torch.cuda.memory_allocated() / (1024**3)
    print(f"{model.id} - Before initialization: {memory_before:.2f} GiB")

    model_components = model.initialize(model_path=model_path, device=device)

    memory_after = torch.cuda.memory_allocated() / (1024**3)
    print(f"{model.id} - After initialization: {memory_after:.2f} GiB")


def get_op(op: str, model_path: str | None = None):
    if op == "LatentsGenerator":
        from diffusionflow.operators.custom.latents_generator import LatentsGenerator

        return LatentsGenerator()
    elif op == "CLIP_L":
        from diffusionflow.operators.models.text_encoders.clip import CLIP_L

        return CLIP_L(Config(model_path=model_path))
    elif op == "CLIP_G":
        from diffusionflow.operators.models.text_encoders.clip import CLIP_G

        return CLIP_G(Config(model_path=model_path))
    elif op == "T5":
        from diffusionflow.operators.models.text_encoders.t5 import T5

        return T5(Config(model_path=model_path))
    elif op == "StableDiffusion3TextEncoder":
        from diffusionflow.operators.custom.stable_diffusion_3_text_encoder import (
            StableDiffusion3TextEncoder,
        )

        return StableDiffusion3TextEncoder()
    elif op == "StableDiffusion3":
        from diffusionflow.operators.models.diffusion_models.stable_diffusion_3 import (
            StableDiffusion3,
        )

        return StableDiffusion3(Config(model_path=model_path))
    elif op == "StableDiffusion35Large":
        from diffusionflow.operators.models.diffusion_models.stable_diffusion_35_large import (
            StableDiffusion35Large,
        )

        return StableDiffusion35Large(Config(model_path=model_path))
    elif op == "StableDiffusion3VAE":
        from diffusionflow.operators.models.autoencoders.stable_diffusion_3_vae import (
            StableDiffusion3VAE,
        )

        return StableDiffusion3VAE(Config(model_path=model_path))
    elif op == "StableDiffusion35LargeVAE":
        from diffusionflow.operators.models.autoencoders.stable_diffusion_35_large_vae import (
            StableDiffusion35LargeVAE,
        )

        return StableDiffusion35LargeVAE(Config(model_path=model_path))
    elif op == "FlowMatchEulerDiscreteScheduler":
        from diffusionflow.operators.schedulers.flow_match_euler_discrete_scheduler import (
            FlowMatchEulerDiscreteScheduler,
        )

        return FlowMatchEulerDiscreteScheduler(Config(model_path=model_path))

    elif op == "IndexedTensor":
        from diffusionflow.operators.custom.indexed_tensor import IndexedTensor

        return IndexedTensor()

    elif op == "GuidanceTensor":
        from diffusionflow.operators.custom.guidance_tensor import GuidanceTensor

        return GuidanceTensor()

    elif op == "BALTrigger":
        from diffusionflow.operators.custom.bal_trigger import BALTrigger

        return BALTrigger()
    elif op == "BALChecker":
        from diffusionflow.operators.custom.bal_checker import BALChecker

        return BALChecker()

    elif op == "StableDiffusion3ControlNetCanny":
        from diffusionflow.operators.models.adapters.stable_diffusion_3_controlnet import (
            StableDiffusion3ControlNetCanny,
        )

        return StableDiffusion3ControlNetCanny(Config(model_path=model_path))

    elif op == "StableDiffusion35LargeControlNetDepth":
        from diffusionflow.operators.models.adapters.stable_diffusion_35_large_controlnet import (
            StableDiffusion35LargeControlNetDepth,
        )

        return StableDiffusion35LargeControlNetDepth(Config(model_path=model_path))

    elif op == "StableDiffusion35LargeControlNetCanny":
        from diffusionflow.operators.models.adapters.stable_diffusion_35_large_controlnet import (
            StableDiffusion35LargeControlNetCanny,
        )

        return StableDiffusion35LargeControlNetCanny(Config(model_path=model_path))

    elif op == "StableDiffusion3ControlNetPose":
        from diffusionflow.operators.models.adapters.stable_diffusion_3_controlnet import (
            StableDiffusion3ControlNetPose,
        )

        return StableDiffusion3ControlNetPose(Config(model_path=model_path))

    elif op == "StableDiffusion15ControlNet":
        from diffusionflow.operators.models.adapters.stable_diffusion_15_controlnet import (
            StableDiffusion15ControlNet,
        )

        return StableDiffusion15ControlNet(Config(model_path=model_path))

    elif op == "StableDiffusion15ControlNetCanny":
        from diffusionflow.operators.models.adapters.stable_diffusion_15_controlnet import (
            StableDiffusion15ControlNetCanny,
        )

        return StableDiffusion15ControlNetCanny(Config(model_path=model_path))

    elif op == "StableDiffusion15ControlNetDepth":
        from diffusionflow.operators.models.adapters.stable_diffusion_15_controlnet import (
            StableDiffusion15ControlNetDepth,
        )

        return StableDiffusion15ControlNetDepth(Config(model_path=model_path))

    ### SD15
    elif op == "StableDiffusion15TextEncoder":
        from diffusionflow.operators.custom.stable_diffusion_15_text_encoder import (
            StableDiffusion15TextEncoder,
        )

        return StableDiffusion15TextEncoder()
    elif op == "StableDiffusion15":
        from diffusionflow.operators.models.diffusion_models.stable_diffusion_15 import (
            StableDiffusion15,
        )

        return StableDiffusion15(Config(model_path=model_path))
    elif op == "StableDiffusion15VAE":
        from diffusionflow.operators.models.autoencoders.stable_diffusion_15_vae import (
            StableDiffusion15VAE,
        )

        return StableDiffusion15VAE(Config(model_path=model_path))
    elif op == "PNDMScheduler":
        from diffusionflow.operators.schedulers.pndm_scheduler import PNDMScheduler

        return PNDMScheduler(Config(model_path=model_path))
    elif op == "CLIP_SD15":
        from diffusionflow.operators.models.text_encoders.clip_sd15 import CLIP_SD15

        return CLIP_SD15(Config(model_path=model_path))

    ### Flux 1.0 Dev
    elif op == "FluxLatentsGenerator":
        from diffusionflow.operators.custom.flux_latents_generator import (
            FluxLatentsGenerator,
        )

        return FluxLatentsGenerator()
    elif op == "CLIP_Flux":
        from diffusionflow.operators.models.text_encoders.clip_flux import CLIP_Flux

        return CLIP_Flux(Config(model_path=model_path))
    elif op == "T5_Flux":
        from diffusionflow.operators.models.text_encoders.t5_flux import T5_Flux

        return T5_Flux(Config(model_path=model_path))
    elif op == "FluxTextEncoder":
        from diffusionflow.operators.custom.flux_text_encoder import FluxTextEncoder

        return FluxTextEncoder()
    elif op == "Flux1VAE":
        from diffusionflow.operators.models.autoencoders.flux_1_vae import Flux1VAE

        return Flux1VAE(Config(model_path=model_path))
    elif op == "Flux1Dev":
        from diffusionflow.operators.models.diffusion_models.flux_1_dev import Flux1Dev

        return Flux1Dev(Config(model_path=model_path))
    elif op == "Flux1Schnell":
        from diffusionflow.operators.models.diffusion_models.flux_1_schnell import (
            Flux1Schnell,
        )

        return Flux1Schnell(Config(model_path=model_path))
    elif op == "FluxFlowMatchEulerDiscreteScheduler":
        from diffusionflow.operators.schedulers.flux_flow_match_euler_discrete_scheduler import (
            FluxFlowMatchEulerDiscreteScheduler,
        )

        return FluxFlowMatchEulerDiscreteScheduler(Config(model_path=model_path))
    elif op == "FluxSchnellFlowMatchEulerDiscreteScheduler":
        from diffusionflow.operators.schedulers.flux_flow_match_euler_discrete_scheduler import (
            FluxSchnellFlowMatchEulerDiscreteScheduler,
        )

        return FluxSchnellFlowMatchEulerDiscreteScheduler(Config(model_path=model_path))
    elif op == "Flux1DevControlNet":
        from diffusionflow.operators.models.adapters.flux_1_dev_controlnet import (
            Flux1DevControlNet,
        )

        return Flux1DevControlNet(Config(model_path=model_path))

    elif op == "Flux1DevControlNetDepth":
        from diffusionflow.operators.models.adapters.flux_1_dev_controlnet import (
            Flux1DevControlNetDepth,
        )

        return Flux1DevControlNetDepth(Config(model_path=model_path))
    elif op == "Flux1DevControlNetCanny":
        from diffusionflow.operators.models.adapters.flux_1_dev_controlnet import (
            Flux1DevControlNetCanny,
        )

        return Flux1DevControlNetCanny(Config(model_path=model_path))

    ### SDXL
    elif op == "StableDiffusionXLTextEncoder":
        from diffusionflow.operators.custom.stable_diffusion_xl_text_encoder import (
            StableDiffusionXLTextEncoder,
        )

        return StableDiffusionXLTextEncoder()
    elif op == "StableDiffusionXLVAE":
        from diffusionflow.operators.models.autoencoders.stable_diffusion_xl_vae import (
            StableDiffusionXLVAE,
        )

        return StableDiffusionXLVAE(Config(model_path=model_path))
    elif op == "StableDiffusionXL":
        from diffusionflow.operators.models.diffusion_models.stable_diffusion_xl import (
            StableDiffusionXL,
        )

        return StableDiffusionXL(Config(model_path=model_path))
    elif op == "EulerDiscreteScheduler":
        from diffusionflow.operators.schedulers.euler_discrete_scheduler import (
            EulerDiscreteScheduler,
        )

        return EulerDiscreteScheduler(Config(model_path=model_path))
    elif op == "CLIP_SDXL_1":
        from diffusionflow.operators.models.text_encoders.clip_sdxl import CLIP_SDXL_1

        return CLIP_SDXL_1(Config(model_path=model_path))
    elif op == "CLIP_SDXL_2":
        from diffusionflow.operators.models.text_encoders.clip_sdxl import CLIP_SDXL_2

        return CLIP_SDXL_2(Config(model_path=model_path))
    elif op == "StableDiffusionXLControlNet":
        from diffusionflow.operators.models.adapters.stable_diffusion_xl_controlnet import (
            StableDiffusionXLControlNet,
        )

        return StableDiffusionXLControlNet(Config(model_path=model_path))

    elif op == "StableDiffusionXLControlNetCanny":
        from diffusionflow.operators.models.adapters.stable_diffusion_xl_controlnet import (
            StableDiffusionXLControlNetCanny,
        )

        return StableDiffusionXLControlNetCanny(Config(model_path=model_path))

    elif op == "StableDiffusionXLControlNetDepth":
        from diffusionflow.operators.models.adapters.stable_diffusion_xl_controlnet import (
            StableDiffusionXLControlNetDepth,
        )

        return StableDiffusionXLControlNetDepth(Config(model_path=model_path))

    ### Patches
    elif op == "StableDiffusion3LoRA":
        from diffusionflow.operators.models.patches.stable_diffusion_3_lora import (
            StableDiffusion3LoRA,
        )

        return StableDiffusion3LoRA(Config(model_path=model_path))

    raise ValueError(f"Operator with ID {op} not found")
