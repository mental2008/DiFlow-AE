"""
Centralized operator ID constants.

This module contains all operator ID constants used across the codebase.
These constants can be used for comparisons without needing to instantiate operator classes.
"""

# Diffusion Models
STABLE_DIFFUSION_3_ID = "StableDiffusion3"
STABLE_DIFFUSION_15_ID = "StableDiffusion15"
STABLE_DIFFUSION_35_LARGE_ID = "StableDiffusion35Large"
STABLE_DIFFUSION_XL_ID = "StableDiffusionXL"
FLUX_1_SCHNELL_ID = "Flux1Schnell"
FLUX_1_DEV_ID = "Flux1Dev"

# Text Encoders
CLIP_L_ID = "CLIP_L"
CLIP_G_ID = "CLIP_G"
CLIP_SD15_ID = "CLIP_SD15"
CLIP_SDXL_1_ID = "CLIP_SDXL_1"
CLIP_SDXL_2_ID = "CLIP_SDXL_2"
CLIP_FLUX_ID = "CLIP_Flux"
T5_ID = "T5"
T5_FLUX_ID = "T5_Flux"

# Adapters (ControlNet)
STABLE_DIFFUSION_3_CONTROLNET_CANNY_ID = "StableDiffusion3ControlNetCanny"
STABLE_DIFFUSION_3_CONTROLNET_POSE_ID = "StableDiffusion3ControlNetPose"
STABLE_DIFFUSION_15_CONTROLNET_CANNY_ID = "StableDiffusion15ControlNetCanny"
STABLE_DIFFUSION_15_CONTROLNET_DEPTH_ID = "StableDiffusion15ControlNetDepth"
STABLE_DIFFUSION_35_LARGE_CONTROLNET_CANNY_ID = "StableDiffusion35LargeControlNetCanny"
STABLE_DIFFUSION_35_LARGE_CONTROLNET_DEPTH_ID = "StableDiffusion35LargeControlNetDepth"
STABLE_DIFFUSION_XL_CONTROLNET_CANNY_ID = "StableDiffusionXLControlNetCanny"
STABLE_DIFFUSION_XL_CONTROLNET_DEPTH_ID = "StableDiffusionXLControlNetDepth"
FLUX_1_DEV_CONTROLNET_DEPTH_ID = "Flux1DevControlNetDepth"
FLUX_1_DEV_CONTROLNET_CANNY_ID = "Flux1DevControlNetCanny"

# Autoencoders (VAE)
STABLE_DIFFUSION_3_VAE_ID = "StableDiffusion3VAE"
STABLE_DIFFUSION_15_VAE_ID = "StableDiffusion15VAE"
STABLE_DIFFUSION_35_LARGE_VAE_ID = "StableDiffusion35LargeVAE"
STABLE_DIFFUSION_XL_VAE_ID = "StableDiffusionXLVAE"
FLUX_1_VAE_ID = "Flux1VAE"

# Patches (LoRA)
STABLE_DIFFUSION_3_LORA_ID = "StableDiffusion3LoRA"

# Schedulers
EULER_DISCRETE_SCHEDULER_ID = "EulerDiscreteScheduler"
PNDM_SCHEDULER_ID = "PNDMScheduler"
FLOW_MATCH_EULER_DISCRETE_SCHEDULER_ID = "FlowMatchEulerDiscreteScheduler"
FLUX_FLOW_MATCH_EULER_DISCRETE_SCHEDULER_ID = "FluxFlowMatchEulerDiscreteScheduler"
FLUX_SCHNELL_FLOW_MATCH_EULER_DISCRETE_SCHEDULER_ID = (
    "FluxSchnellFlowMatchEulerDiscreteScheduler"
)

# Custom Operators
FLUX_LATENTS_GENERATOR_ID = "FluxLatentsGenerator"
FLUX_TEXT_ENCODER_ID = "FluxTextEncoder"
GUIDANCE_TENSOR_ID = "GuidanceTensor"
INDEXED_TENSOR_ID = "IndexedTensor"
LATENTS_GENERATOR_ID = "LatentsGenerator"
STABLE_DIFFUSION_15_TEXT_ENCODER_ID = "StableDiffusion15TextEncoder"
STABLE_DIFFUSION_3_TEXT_ENCODER_ID = "StableDiffusion3TextEncoder"
STABLE_DIFFUSION_XL_TEXT_ENCODER_ID = "StableDiffusionXLTextEncoder"
