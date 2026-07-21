import os
from typing import Any, Dict, List, Union

import torch
from transformers import T5Config, T5EncoderModel, T5TokenizerFast

from diffusionflow.operators.base import Operator
from diffusionflow.operators.operator_ids import T5_FLUX_ID
from diffusionflow.operators.utils import test_model_memory_allocation


class T5_Flux(Operator):
    def setup_io(self):
        self.add_input("prompt", Union[str, List[str]])
        # [batch_size, 512, 4096]
        self.add_output("prompt_embeds", torch.Tensor, [1, 512, 4096])

    @property
    def id(self) -> str:
        return T5_FLUX_ID

    def initialize(
        self, model_path: Union[str, None], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        if model_path is not None and os.path.exists(model_path):
            text_encoder = T5EncoderModel.from_pretrained(
                model_path,
                subfolder="text_encoder_2",
                torch_dtype=torch.bfloat16,
            ).to(device)

            tokenizer = T5TokenizerFast.from_pretrained(
                model_path,
                subfolder="tokenizer_2",
            )
        else:
            text_encoder = T5EncoderModel(
                config=self._default_text_encoder_dummy_config()
            ).to(device=device, dtype=torch.bfloat16)

            default_tokenizer_path = self._default_tokenizer_path()
            tokenizer = T5TokenizerFast.from_pretrained(default_tokenizer_path)

        return {
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

    @staticmethod
    def _default_text_encoder_dummy_config() -> T5Config:
        return T5Config(
            classifier_dropout=0.0,
            d_ff=10240,
            d_kv=64,
            d_model=4096,
            dropout_rate=0.1,
            eos_token_id=1,
            feed_forward_proj="gated-gelu",
            initializer_factor=1.0,
            is_encoder_decoder=True,
            layer_norm_epsilon=1e-06,
            num_decoder_layers=24,
            num_heads=64,
            num_layers=24,
            pad_token_id=0,
            relative_attention_max_distance=128,
            relative_attention_num_buckets=32,
            use_cache=True,
            vocab_size=32128,
        )

    @staticmethod
    def _default_tokenizer_path() -> str:
        return os.path.join(os.path.dirname(__file__), "tokenizers", "t5_flux")

    # Adapted from https://github.com/huggingface/diffusers/blob/c14057c8dbc32847bac9082bcc0ae00c9a19357d/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L232
    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        **kwargs,
    ) -> Dict[str, Any]:
        prompt = kwargs["prompt"]

        text_encoder = model_components["text_encoder"]
        tokenizer = model_components["tokenizer"]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        num_images_per_prompt = kwargs.get(
            "num_images_per_prompt", 1
        )  # TODO: should add to input
        max_sequence_length = kwargs.get(
            "max_sequence_length", 512
        )  # TODO: should add to input

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
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
                untruncated_ids[:, max_sequence_length - 1 : -1]
            )
            print(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = text_encoder(
            text_input_ids.to(device), output_hidden_states=False
        )[0]

        dtype = text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        return {"prompt_embeds": prompt_embeds}


if __name__ == "__main__":
    # text_encoder = T5_Flux()
    # model_components = text_encoder.initialize(
    #     "/project/infattllm/lyangbk/huggingface/FLUX.1-dev", "cuda"
    # )
    # result = text_encoder.execute(
    #     model_components=model_components,
    #     device="cuda",
    #     prompt="A cat holding a sign that says hello world",
    # )
    # torch.save(
    #     result["prompt_embeds"],
    #     "/home/slida/DiffusionFlow/tmp_tensors/t5_flux_prompt_embeds.pt",
    # )

    test_model_memory_allocation(model=T5_Flux(), model_path=None)
