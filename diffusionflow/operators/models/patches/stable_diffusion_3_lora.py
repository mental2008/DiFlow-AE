import logging
from typing import Any, Dict, Union

import torch
from diffusers.loaders import SD3LoraLoaderMixin
from diffusers.loaders.lora_base import _fetch_state_dict
from diffusers.utils import recurse_remove_peft_layers

from diffusionflow.operators.models.patches.base_patch import BasePatch
from diffusionflow.operators.operator_ids import (
    CLIP_G_ID,
    CLIP_L_ID,
    STABLE_DIFFUSION_3_ID,
    STABLE_DIFFUSION_3_LORA_ID,
)

logger = logging.getLogger(__name__)


class StableDiffusion3LoRA(BasePatch):
    @property
    def id(self) -> str:
        return STABLE_DIFFUSION_3_LORA_ID

    # adapted from SD3LoraLoaderMixin.lora_state_dict() and SD3LoraLoaderMixin.load_lora_weights() in diffusers.loaders.lora_pipeline.py
    def initialize(
        self,
        model_path: str,
        device: Union[str, torch.device] = None,
    ) -> Dict[str, Any]:
        # Load the main state dict first which has the LoRA layers for either of
        # transformer and text encoder or both.
        state_dict = _fetch_state_dict(
            pretrained_model_name_or_path_or_dict=model_path,
            weight_name=None,
            use_safetensors=True,
            local_files_only=None,
            cache_dir=None,
            force_download=None,
            proxies=None,
            token=None,
            revision=None,
            subfolder=None,
            user_agent={
                "file_type": "attn_procs_weights",
                "framework": "pytorch",
            },
            allow_pickle=True,
        )[0]

        transformer_state_dict = {
            k: v for k, v in state_dict.items() if "transformer." in k
        }
        text_encoder_state_dict = {
            k: v for k, v in state_dict.items() if "text_encoder." in k
        }
        text_encoder_2_state_dict = {
            k: v for k, v in state_dict.items() if "text_encoder_2." in k
        }

        logger.debug(
            f"got StableDiffusion3LoRA state dict for transformer: {len(transformer_state_dict)}, text_encoder: {len(text_encoder_state_dict)}, text_encoder_2: {len(text_encoder_2_state_dict)}"
        )

        return {
            "transformer": transformer_state_dict,
            "text_encoder": text_encoder_state_dict,
            "text_encoder_2": text_encoder_2_state_dict,
        }

    @torch.no_grad()
    def _patch_on(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        target_model_id: str,
        target_model_components: Dict[str, Any],
    ) -> None:
        """Apply LoRA weights to the target model."""
        logger.debug(f"patching on LoRA weights to the target model {target_model_id}")
        if target_model_id == STABLE_DIFFUSION_3_ID:
            transformer_state_dict = model_components["transformer"]
            logger.debug(f"transformer_state_dict: {len(transformer_state_dict)}")
            if len(transformer_state_dict) > 0:
                logger.debug(f"loading LoRA weights into transformer")
                SD3LoraLoaderMixin.load_lora_into_transformer(
                    state_dict=transformer_state_dict,
                    transformer=target_model_components["transformer"],
                    adapter_name=None,
                    low_cpu_mem_usage=True,
                )
        elif target_model_id == CLIP_L_ID:
            text_encoder_state_dict = model_components["text_encoder"]
            logger.debug(f"text_encoder_state_dict: {len(text_encoder_state_dict)}")
            if len(text_encoder_state_dict) > 0:
                logger.debug(f"loading LoRA weights into text encoder")
                SD3LoraLoaderMixin.load_lora_into_text_encoder(
                    state_dict=text_encoder_state_dict,
                    network_alphas=None,
                    text_encoder=target_model_components["text_encoder"],
                    prefix="text_encoder",
                    lora_scale=1.0,
                    adapter_name=None,
                    low_cpu_mem_usage=True,
                )
        elif target_model_id == CLIP_G_ID:
            text_encoder_2_state_dict = model_components["text_encoder_2"]
            logger.debug(f"text_encoder_2_state_dict: {len(text_encoder_2_state_dict)}")
            if len(text_encoder_2_state_dict) > 0:
                logger.debug(f"loading LoRA weights into text encoder 2")
                SD3LoraLoaderMixin.load_lora_into_text_encoder(
                    state_dict=text_encoder_2_state_dict,
                    network_alphas=None,
                    text_encoder=target_model_components["text_encoder"],
                    prefix="text_encoder_2",
                    lora_scale=1.0,
                    adapter_name=None,
                    low_cpu_mem_usage=True,
                )
        else:
            raise ValueError(
                f"Invalid target model id for the patch StableDiffusion3LoRA: {target_model_id}. Supported target model ids are: {STABLE_DIFFUSION_3_ID}, {CLIP_L_ID}, {CLIP_G_ID}"
            )

    @torch.no_grad()
    def _patch_off(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        target_model_id: str,
        target_model_components: Dict[str, Any],
    ) -> None:
        """Remove LoRA weights from the target model."""
        if target_model_id == STABLE_DIFFUSION_3_ID:
            transformer = target_model_components["transformer"]
            transformer.unload_lora()
        elif target_model_id == CLIP_L_ID or target_model_id == CLIP_G_ID:
            text_encoder = target_model_components["text_encoder"]
            self._remove_text_encoder_monkey_patch(text_encoder=text_encoder)
        else:
            raise ValueError(
                f"Invalid target model id for the patch StableDiffusion3LoRA: {target_model_id}. Supported target model ids are: {STABLE_DIFFUSION_3_ID}, {CLIP_L_ID}, {CLIP_G_ID}"
            )

    @torch.no_grad()
    def _remove_text_encoder_monkey_patch(self, text_encoder):
        recurse_remove_peft_layers(text_encoder)
        if getattr(text_encoder, "peft_config", None) is not None:
            del text_encoder.peft_config
            text_encoder._hf_peft_config_loaded = None
