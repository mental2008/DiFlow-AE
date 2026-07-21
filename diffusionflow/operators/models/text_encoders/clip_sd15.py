from abc import abstractmethod
from typing import Any, Dict, List, Union

import torch
from diffusers.loaders.textual_inversion import TextualInversionLoaderMixin
from transformers import CLIPTextModel, CLIPTokenizer

from diffusionflow.operators.base import Operator
from diffusionflow.operators.operator_ids import CLIP_SD15_ID


class CLIP_SD15(Operator, TextualInversionLoaderMixin):
    def setup_io(self):
        self.add_input("prompt", Union[str, List[str]])
        self.add_output("prompt_embeds", torch.Tensor)

    @property
    def id(self) -> str:
        return CLIP_SD15_ID

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

    # Adapted from https://github.com/huggingface/diffusers/blob/b75b204a584e29ebf4e80a61be11458e9ed56e3e/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L288
    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        **kwargs,
    ) -> Dict[str, Any]:
        prompt = kwargs["prompt"]

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, model_components["tokenizer"])

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

        if (
            hasattr(text_encoder.config, "use_attention_mask")
            and text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        if clip_skip is None:
            prompt_embeds = text_encoder(
                text_input_ids.to(device), attention_mask=attention_mask
            )
            prompt_embeds = prompt_embeds[0]
        else:
            prompt_embeds = text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # Access the `hidden_states` first, that contains a tuple of
            # all the hidden states from the encoder layers. Then index into
            # the tuple to access the hidden states from the desired layer.
            prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
            # We also need to apply the final LayerNorm here to not mess with the
            # representations. The `last_hidden_states` that we typically use for
            # obtaining the final prompt representations passes through the LayerNorm
            # layer.
            prompt_embeds = text_encoder.text_model.final_layer_norm(prompt_embeds)

        prompt_embeds_dtype = text_encoder.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        return {
            "prompt_embeds": prompt_embeds,
        }


if __name__ == "__main__":
    text_encoder = CLIP_SD15()
    model_components = text_encoder.initialize(
        "/project/infattllm/lyangbk/huggingface/stable-diffusion-v1-5", "cuda"
    )
    result = text_encoder.execute(
        model_components=model_components,
        device="cuda",
        prompt="a photo of an astronaut riding a horse on mars",
    )
    torch.save(
        result["prompt_embeds"],
        "/home/slida/DiffusionFlow/tmp_tensors/clip_prompt_embeds.pt",
    )
    result = text_encoder.execute(
        model_components=model_components,
        device="cuda",
        prompt="",
    )
    torch.save(
        result["prompt_embeds"],
        "/home/slida/DiffusionFlow/tmp_tensors/clip_negative_prompt_embeds.pt",
    )
