import os
from abc import abstractmethod
from typing import Any, Dict, List, Union

import torch
from transformers import CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer

from diffusionflow.operators.base import Operator
from diffusionflow.operators.operator_ids import CLIP_G_ID, CLIP_L_ID
from diffusionflow.operators.utils import test_model_memory_allocation


class CLIP(Operator):
    def setup_io(self):
        self.add_input("prompt", Union[str, List[str]])
        self.add_output("prompt_embeds", torch.Tensor)
        self.add_output("pooled_prompt_embeds", torch.Tensor)

    @property
    def id(self) -> str:
        raise NotImplementedError("Subclasses must implement model_id")

    @abstractmethod
    def get_encoder_config(self) -> Dict[str, str]:
        pass

    def initialize(
        self, model_path: Union[str, None], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        config = self.get_encoder_config()

        if model_path is not None and os.path.exists(model_path):
            text_encoder = CLIPTextModelWithProjection.from_pretrained(
                model_path,
                subfolder=config["encoder_path"],
                torch_dtype=torch.float16,
            ).to(device)

            tokenizer = CLIPTokenizer.from_pretrained(
                model_path,
                subfolder=config["tokenizer_path"],
            )
        else:
            text_encoder_config = self._default_text_encoder_dummy_config()
            text_encoder = CLIPTextModelWithProjection(config=text_encoder_config).to(
                device=device, dtype=torch.float16
            )

            default_tokenizer_path = self._default_tokenizer_path()
            tokenizer = CLIPTokenizer.from_pretrained(default_tokenizer_path)

        return {
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

    # Adapted from https://github.com/huggingface/diffusers/blob/b75b204a584e29ebf4e80a61be11458e9ed56e3e/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L288
    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        **kwargs,
    ) -> Dict[str, Any]:
        prompt = kwargs["prompt"]

        clip_skip = kwargs.get("clip_skip", None)  # TODO: should add to the input
        num_images_per_prompt = kwargs.get(
            "num_images_per_prompt", 1
        )  # TODO: should add to the input

        text_encoder = model_components["text_encoder"]
        tokenizer = model_components["tokenizer"]
        max_length = getattr(tokenizer, "model_max_length", 77)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(
                untruncated_ids[:, max_length - 1 : -1]
            )
            print(f"Truncated text: {removed_text}")

        prompt_embeds = text_encoder(
            text_input_ids.to(device), output_hidden_states=True
        )
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(
            batch_size * num_images_per_prompt, -1
        )

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }


class CLIP_L(CLIP):
    @property
    def id(self) -> str:
        return CLIP_L_ID

    def get_encoder_config(self) -> Dict[str, str]:
        return {
            "encoder_path": "text_encoder",
            "tokenizer_path": "tokenizer",
        }

    @staticmethod
    def _default_text_encoder_dummy_config() -> CLIPTextConfig:
        return CLIPTextConfig(
            attention_dropout=0.0,
            bos_token_id=0,
            dropout=0.0,
            eos_token_id=2,
            hidden_act="quick_gelu",
            hidden_size=768,
            initializer_factor=1.0,
            initializer_range=0.02,
            intermediate_size=3072,
            layer_norm_eps=1e-05,
            max_position_embeddings=77,
            model_type="clip_text_model",
            num_attention_heads=12,
            num_hidden_layers=12,
            pad_token_id=1,
            projection_dim=768,
            vocab_size=49408,
        )

    @staticmethod
    def _default_tokenizer_path() -> str:
        return os.path.join(os.path.dirname(__file__), "tokenizers", "clip_l")


class CLIP_G(CLIP):
    @property
    def id(self) -> str:
        return CLIP_G_ID

    def get_encoder_config(self) -> Dict[str, str]:
        return {
            "encoder_path": "text_encoder_2",
            "tokenizer_path": "tokenizer_2",
        }

    @staticmethod
    def _default_text_encoder_dummy_config() -> CLIPTextConfig:
        return CLIPTextConfig(
            attention_dropout=0.0,
            bos_token_id=0,
            dropout=0.0,
            eos_token_id=2,
            hidden_act="gelu",
            hidden_size=1280,
            initializer_factor=1.0,
            initializer_range=0.02,
            intermediate_size=5120,
            layer_norm_eps=1e-05,
            max_position_embeddings=77,
            model_type="clip_text_model",
            num_attention_heads=20,
            num_hidden_layers=32,
            pad_token_id=1,
            projection_dim=1280,
            vocab_size=49408,
        )

    @staticmethod
    def _default_tokenizer_path() -> str:
        return os.path.join(os.path.dirname(__file__), "tokenizers", "clip_g")


if __name__ == "__main__":
    test_model_memory_allocation(model=CLIP_L(), model_path=None)
    test_model_memory_allocation(model=CLIP_G(), model_path=None)
