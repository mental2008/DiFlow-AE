import logging
import os
from typing import Any, Dict, Union

import torch
from diffusers import schedulers
from diffusers.utils.torch_utils import randn_tensor
from overrides import override

from diffusionflow.operators.operator_ids import EULER_DISCRETE_SCHEDULER_ID
from diffusionflow.operators.schedulers.base_scheduler import BaseScheduler

logger = logging.getLogger(__name__)


class EulerDiscreteScheduler(BaseScheduler):
    def setup_io(self):
        super().setup_io()
        # add one more execution mode for SDXL
        self.add_execution_mode(
            "scale_model_input",
            inputs={
                "latents": torch.Tensor,
                "timestep": torch.Tensor,
            },
            outputs={
                "latents": torch.Tensor,
            },
        )

        self.add_execution_mode(
            "init_noise_sigma",
            inputs={
                "latents": torch.Tensor,
                "timestep": torch.Tensor,
            },
            outputs={
                "latents": torch.Tensor,
            },
        )

        self.add_execution_mode(
            "img2img_get_timesteps",
            inputs={
                "num_inference_steps": int,
                "strength": float,
            },
            outputs={
                "timesteps": torch.Tensor,
                "num_inference_steps": int,
                "latents": torch.Tensor,
            },
        )
        self.add_execution_mode(
            "add_noise",
            inputs={
                "init_latents": torch.Tensor,
                "timestep": torch.Tensor,
            },
            outputs={
                "latents": torch.Tensor,
            },
        )

    @property
    def id(self) -> str:
        return EULER_DISCRETE_SCHEDULER_ID

    def initialize(
        self, model_path: str, device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        if model_path is not None and os.path.exists(model_path):
            scheduler = schedulers.EulerDiscreteScheduler.from_pretrained(
                model_path, subfolder="scheduler"
            )
        else:
            dummy_config = self._default_dummy_config()
            scheduler = schedulers.EulerDiscreteScheduler(**dummy_config)

        return {"scheduler": scheduler}

    @staticmethod
    def _default_dummy_config() -> Dict[str, Any]:
        return {
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "interpolation_type": "linear",
            "num_train_timesteps": 1000,
            "prediction_type": "epsilon",
            "sample_max_value": 1.0,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "timestep_spacing": "leading",
            "trained_betas": None,
            "use_karras_sigmas": False,
        }

    @torch.no_grad()
    @override
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        mode: str,
        **kwargs,
    ) -> Dict[str, Any]:
        scheduler = model_components["scheduler"]

        if mode == "init":
            num_inference_steps = kwargs["num_inference_steps"]
            scheduler.set_timesteps(num_inference_steps, device=device)
            return {"timesteps": scheduler.timesteps}
        elif mode == "step":
            latents = kwargs["latents"]
            noise_pred = kwargs["noise_pred"]
            timestep = kwargs["timestep"].item()

            # compute the previous noisy sample x_t -> x_t-1
            output_latents = scheduler.step(
                noise_pred, timestep, latents, return_dict=False
            )[0]

            return {"output_latents": output_latents}
        elif mode == "step_classifier_free_guidance":
            latents = kwargs["latents"]
            timestep = kwargs["timestep"].item()
            noise_pred_uncond = kwargs["noise_pred_uncond"]
            noise_pred_text = kwargs["noise_pred_text"]
            guidance_scale = kwargs["guidance_scale"]

            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            output_latents = scheduler.step(
                noise_pred, timestep, latents, return_dict=False
            )[0]

            return {"output_latents": output_latents}
        elif mode == "scale_model_input":
            if not hasattr(scheduler, "scale_model_input"):
                return {"latents": kwargs["latents"]}
            sample = kwargs["latents"]
            timestep = kwargs["timestep"].item()
            latents = scheduler.scale_model_input(sample, timestep)
            return {"latents": latents}
        elif mode == "init_noise_sigma":
            latents = kwargs["latents"]
            latents = latents * scheduler.init_noise_sigma
            # logger.debug(f"latents: {latents.shape}, {latents[0][0][0][:10]}")
            # logger.debug(f"init_noise_sigma: {scheduler.init_noise_sigma}")
            # logger.debug(f"scheduler.betas: {scheduler.betas[:10]}")
            # logger.debug(f"scheduler.alphas: {scheduler.alphas[:10]}")
            # logger.debug(f"scheduler.alphas_cumprod: {scheduler.alphas_cumprod[:10]}")
            # logger.debug(f"scheduler.sigmas: {scheduler.sigmas[:10]}")
            return {"latents": latents}

        elif mode == "img2img_get_timesteps":
            # Suyi: for the img2img pipeline
            num_inference_steps = kwargs["num_inference_steps"]
            strength = kwargs["strength"]

            scheduler.set_timesteps(num_inference_steps, device=device)

            init_timestep = min(
                int(num_inference_steps * strength), num_inference_steps
            )
            t_start = max(num_inference_steps - init_timestep, 0)

            timesteps = scheduler.timesteps[t_start * scheduler.order :]
            if hasattr(scheduler, "set_begin_index"):
                scheduler.set_begin_index(t_start * scheduler.order)

            init_latents = kwargs["latents"]

            seed = kwargs.get("seed", 0)

            generator = torch.manual_seed(seed)
            shape = init_latents.shape
            dtype = init_latents.dtype

            if isinstance(device, str):
                device = torch.device(device)

            # ! Suyi: hard coding for batch size 1
            timestep = timesteps[:1].repeat(init_latents.shape[0])

            logger.debug(f"init_latents.shape: {init_latents.shape}")

            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = scheduler.add_noise(init_latents, noise, timestep)

            # logger.debug(f"noise: {noise[0][0][0][:10]}")
            # logger.debug(f"latents after add_noise: {latents[0][0][0][:10]}")

            return {
                "timesteps": timesteps,
                "num_inference_steps": num_inference_steps - t_start,
                "latents": latents,
            }

        # elif mode == "add_noise":
        #     # Suyi: for the img2img pipeline
        #     # init_latents = kwargs["init_latents"]
        #     init_latents = kwargs["latents"]
        #     seed = kwargs.get("seed", 0)
        #     timestep = kwargs["timesteps"][:1].repeat(init_latents.shape[0])

        #     generator = torch.manual_seed(seed)
        #     shape = init_latents.shape
        #     dtype = init_latents.dtype

        #     if isinstance(device, str):
        #         device = torch.device(device)

        #     noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        #     init_latents = scheduler.add_noise(init_latents, noise, timestep)
        #     return {"latents": init_latents}

        else:
            raise ValueError(f"Invalid execution mode: {self.execution_mode}")


if __name__ == "__main__":
    scheduler = EulerDiscreteScheduler()
    model_components = scheduler.initialize(
        "/project/infattllm/lyangbk/huggingface/stable-diffusion-xl-base-1.0", "cuda"
    )
    print(model_components["scheduler"].init_noise_sigma)
    result = scheduler.execute(
        model_components=model_components,
        device="cuda",
        mode="init",
        num_inference_steps=50,
    )
    print(result)
