from abc import abstractmethod
from typing import Any, Dict, List, Union

import torch
from diffusers.loaders.textual_inversion import TextualInversionLoaderMixin
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusionflow.operators.base import Operator
from diffusionflow.operators.operator_ids import CLIP_SDXL_1_ID, CLIP_SDXL_2_ID


class CLIP(Operator, TextualInversionLoaderMixin):
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

    @abstractmethod
    def initialize(
        self, model_path: str, device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        pass

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

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, tokenizer)

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


class CLIP_SDXL_1(CLIP):
    @property
    def id(self) -> str:
        return CLIP_SDXL_1_ID

    def get_encoder_config(self) -> Dict[str, str]:
        return {
            "encoder_path": "text_encoder",
            "tokenizer_path": "tokenizer",
        }

    def initialize(
        self, model_path: str, device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        config = self.get_encoder_config()

        text_encoder = CLIPTextModel.from_pretrained(
            model_path,
            subfolder=config["encoder_path"],
            torch_dtype=torch.float16,
        ).to(device)

        tokenizer = CLIPTokenizer.from_pretrained(
            model_path,
            subfolder=config["tokenizer_path"],
        )

        return {
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }


class CLIP_SDXL_2(CLIP):
    @property
    def id(self) -> str:
        return CLIP_SDXL_2_ID

    def get_encoder_config(self) -> Dict[str, str]:
        return {
            "encoder_path": "text_encoder_2",
            "tokenizer_path": "tokenizer_2",
        }

    def initialize(
        self, model_path: str, device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        config = self.get_encoder_config()

        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            model_path,
            subfolder=config["encoder_path"],
            torch_dtype=torch.float16,
        ).to(device)

        tokenizer = CLIPTokenizer.from_pretrained(
            model_path,
            subfolder=config["tokenizer_path"],
        )

        return {
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }


if __name__ == "__main__":
    clip_sdxl_1 = CLIP_SDXL_1()
    model_components = clip_sdxl_1.initialize(
        model_path="/project/infattllm/lyangbk/huggingface/stable-diffusion-xl-base-1.0",
        device="cuda",
    )
    result = clip_sdxl_1.execute(
        model_components=model_components,
        device="cuda",
        prompt="a photo of an astronaut riding a horse on mars",
    )
    print(result["prompt_embeds"].shape)
    print(result["pooled_prompt_embeds"].shape)

    clip_sdxl_2 = CLIP_SDXL_2()
    model_components = clip_sdxl_2.initialize(
        model_path="/project/infattllm/lyangbk/huggingface/stable-diffusion-xl-base-1.0",
        device="cuda",
    )
    result = clip_sdxl_2.execute(
        model_components=model_components,
        device="cuda",
        prompt="a photo of an astronaut riding a horse on mars",
    )
    print(result["prompt_embeds"].shape)
    print(result["pooled_prompt_embeds"].shape)
