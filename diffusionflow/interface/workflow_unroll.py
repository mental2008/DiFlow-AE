import logging
from copy import deepcopy
from typing import Any, Dict

from diffusionflow.interface.node_io import NodeIO, SourceType
from diffusionflow.interface.workflow import Workflow
from diffusionflow.interface.workflow_context import WorkflowContext
from diffusionflow.operators.custom.guidance_tensor import GuidanceTensor
from diffusionflow.operators.custom.indexed_tensor import IndexedTensor

logger = logging.getLogger(__name__)


def run_adapters(
    adapters,
    adapter_inputs,
    diffusion_model_id,
    latents,
    timestep,
    prompt_embeds,
    pooled_prompt_embeds,
    **kwargs,
):
    adapter_block_samples = None
    # Run adapters if present
    if adapters:
        if (
            diffusion_model_id == "StableDiffusion3"
            or diffusion_model_id == "StableDiffusion35Large"
        ):
            adapter_block_samples = []
            for adapter, adapter_input in zip(adapters, adapter_inputs):
                samples = adapter(
                    latents=latents,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    controlnet_cond=adapter_input.controlnet_cond,
                    conditioning_scale=adapter_input.conditioning_scale,
                )
                control_block_samples = {}
                for control_block_sample_idx in range(len(samples)):
                    control_block_samples[
                        "control_block_sample_{}".format(control_block_sample_idx)
                    ] = samples[control_block_sample_idx]
                adapter_block_samples.append(control_block_samples)
        elif diffusion_model_id == "StableDiffusionXL":
            adapter_block_samples = []
            for adapter, adapter_input in zip(adapters, adapter_inputs):
                samples = adapter(
                    latents=latents,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    controlnet_cond=adapter_input.controlnet_cond,
                    conditioning_scale=adapter_input.conditioning_scale,
                    height=kwargs["height"],
                    width=kwargs["width"],
                )
                control_block_samples = {}
                for control_block_sample_idx in range(len(samples) - 1):
                    control_block_samples[
                        "down_block_res_sample_{}".format(control_block_sample_idx)
                    ] = samples[control_block_sample_idx]
                control_block_samples["mid_block_res_sample"] = samples[-1]
                adapter_block_samples.append(control_block_samples)
        elif diffusion_model_id == "StableDiffusion15":
            adapter_block_samples = []
            for adapter, adapter_input in zip(adapters, adapter_inputs):
                samples = adapter(
                    latents=latents,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    controlnet_cond=adapter_input.controlnet_cond,
                    conditioning_scale=adapter_input.conditioning_scale,
                )
                control_block_samples = {}
                for control_block_sample_idx in range(len(samples) - 1):
                    control_block_samples[
                        "down_block_res_sample_{}".format(control_block_sample_idx)
                    ] = samples[control_block_sample_idx]
                control_block_samples["mid_block_res_sample"] = samples[-1]
                adapter_block_samples.append(control_block_samples)
        elif diffusion_model_id == "Flux1Dev" or diffusion_model_id == "Flux1Schnell":
            adapter_block_samples = []
            for adapter, adapter_input in zip(adapters, adapter_inputs):
                samples = adapter(
                    latents=latents,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    controlnet_cond=adapter_input.controlnet_cond,
                    conditioning_scale=adapter_input.conditioning_scale,
                    guidance=kwargs["guidance"],
                    height=kwargs["height"],
                    width=kwargs["width"],
                )
                control_block_samples = {}
                for control_block_sample_idx in range(len(samples)):
                    control_block_samples[
                        "control_block_sample_{}".format(control_block_sample_idx)
                    ] = samples[control_block_sample_idx]
                adapter_block_samples.append(control_block_samples)

    return adapter_block_samples


def prepare_model_kwargs(
    diffusion_model_id,
    latents,
    timestep,
    prompt_embeds,
    pooled_prompt_embeds,
    adapter_block_samples,
    **kwargs,
):
    model_kwargs = {
        "latents": latents,
        "timestep": timestep,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
    }

    # Add adapter-specific parameters if available
    if (
        diffusion_model_id == "StableDiffusion3"
        or diffusion_model_id == "StableDiffusion35Large"
    ):
        if adapter_block_samples is not None:
            # TODO: support multiple adapters
            # ! Suyi: hard coding controlnet input
            model_kwargs.update(adapter_block_samples[0])
    elif diffusion_model_id == "Flux1Dev" or diffusion_model_id == "Flux1Schnell":
        # ! Suyi: these are for Flux 1.0 Dev
        model_kwargs.update(
            {
                "guidance": kwargs["guidance"],
                "height": kwargs["height"],
                "width": kwargs["width"],
            }
        )
        if adapter_block_samples is not None:
            # TODO: support multiple adapters
            # ! Suyi: hard coding controlnet input for Flux
            model_kwargs.update(adapter_block_samples[0])
    elif diffusion_model_id == "StableDiffusionXL":
        if adapter_block_samples is not None:
            # TODO: support multiple adapters
            # ! Suyi: hard coding controlnet input
            model_kwargs.update(adapter_block_samples[0])
        model_kwargs.update(
            {
                "height": kwargs["height"],
                "width": kwargs["width"],
            }
        )
    elif diffusion_model_id == "StableDiffusion15":
        if adapter_block_samples is not None:
            # TODO: support multiple adapters
            # ! Suyi: hard coding controlnet input for SD15
            model_kwargs.update(adapter_block_samples[0])
    else:
        raise ValueError(f"Invalid diffusion model: {diffusion_model_id}")

    return model_kwargs


def unroll_workflow(workflow: Workflow, inputs: Dict[str, Any]) -> Workflow:
    unrolled_workflow = deepcopy(workflow)

    # Suyi: this is to determine whether we are running the nirvana;
    # which incur differences in latent initialization and etc.
    run_nirvana = "nirvana" in unrolled_workflow.name

    WorkflowContext.set_current_workflow(unrolled_workflow)

    # Unroll each denoise node into a sequence of workflow nodes
    for denoise_node in unrolled_workflow.denoise_nodes:
        # Add scheduler initialization node
        assert (
            denoise_node.scheduler_inputs.num_inference_steps.name in inputs
        ), f"Number of inference steps not found in inputs for node {denoise_node.name}"
        num_inference_steps = inputs[
            denoise_node.scheduler_inputs.num_inference_steps.name
        ]

        latents = denoise_node.base_model_inputs.latents

        guidance_scale = (
            inputs[denoise_node.scheduler_inputs.guidance_scale.name]
            if denoise_node.scheduler_inputs.guidance_scale is not None
            else 1.0
        )
        do_classifier_free_guidance = guidance_scale > 1.0
        cfg_guidance_scale = denoise_node.scheduler_inputs.guidance_scale

        scheduler = denoise_node.scheduler

        if run_nirvana and denoise_node.model.id == "StableDiffusionXL":
            # Suyi: for the sdxl img2img pipeline, only support sdxl img2img now
            timesteps, num_inference_steps, latents = scheduler(
                num_inference_steps=denoise_node.scheduler_inputs.num_inference_steps,
                strength=denoise_node.scheduler_inputs.strength,
                latents=latents,
                mode="img2img_get_timesteps",
            )

            num_inference_steps = inputs[
                denoise_node.scheduler_inputs.num_inference_steps.name
            ]
            # init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            # t_start = max(num_inference_steps - init_timestep, 0)
            init_timestep = min(
                int(
                    num_inference_steps
                    * inputs[denoise_node.scheduler_inputs.strength.name]
                ),
                num_inference_steps,
            )
            num_inference_steps = num_inference_steps - max(
                num_inference_steps - init_timestep, 0
            )

        elif not run_nirvana:
            timesteps = scheduler(
                num_inference_steps=denoise_node.scheduler_inputs.num_inference_steps,
                latents=latents,  # Suyi: Flux need latents[1] for scheduler init
                mode="init",
            )
        else:
            raise ValueError(f"Invalid diffusion model: {denoise_node.model.id}")

        # ! Suyi: SDXL requires latent scaling before denoising; add args timesteps to ensure init is executed before
        if not run_nirvana and denoise_node.model.id == "StableDiffusionXL":
            latents = scheduler(
                latents=latents, timesteps=timesteps, mode="init_noise_sigma"
            )

        indexed_tensor_op = IndexedTensor()

        # Prepare parameters for adapters
        guidance_tensor_op = GuidanceTensor()
        if denoise_node.base_model_inputs.guidance_scale is not None:
            guidance_tensor = guidance_tensor_op(
                guidance_scale=denoise_node.base_model_inputs.guidance_scale
            )
        else:
            guidance_tensor = None
        height = denoise_node.base_model_inputs.height
        width = denoise_node.base_model_inputs.width

        # Add node for each iteration
        for i in range(num_inference_steps):
            index_io = NodeIO(
                name=f"{denoise_node.name}_timestep_{i}",
                data_type=int,
                source_type=SourceType.INPUT,
            )
            timestep = indexed_tensor_op(tensor=timesteps, index=index_io)
            inputs[index_io.name] = i

            diffusion_model = denoise_node.model

            # ! Suyi: save original latents for scheduler step; for sd15, sd3, flux, this is the same as latents; for sdxl, it takes the scaled latents
            original_latents_for_scheduler_step = latents
            # ! Suyi: some models require latent scaling before denoising, such as SDXL
            if diffusion_model.id == "StableDiffusionXL":
                latents = scheduler(
                    latents=latents,
                    timestep=timestep,
                    mode="scale_model_input",
                )

            if do_classifier_free_guidance:
                prompt_embeds = denoise_node.base_model_inputs.prompt_embeds
                pooled_prompt_embeds = (
                    denoise_node.base_model_inputs.pooled_prompt_embeds
                )
                negative_prompt_embeds = (
                    denoise_node.base_model_inputs.negative_prompt_embeds
                )
                negative_pooled_prompt_embeds = (
                    denoise_node.base_model_inputs.negative_pooled_prompt_embeds
                )

                # Run adapters if present for unconditioned using run_adapters
                adapter_block_samples_uncond = run_adapters(
                    adapters=denoise_node.adapters,
                    adapter_inputs=denoise_node.adapter_inputs,
                    diffusion_model_id=diffusion_model.id,
                    latents=latents,
                    timestep=timestep,
                    prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    height=height,
                    width=width,
                    guidance=guidance_tensor,
                )

                model_kwargs_uncond = prepare_model_kwargs(
                    diffusion_model_id=diffusion_model.id,
                    latents=latents,
                    timestep=timestep,
                    prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    adapter_block_samples=adapter_block_samples_uncond,
                    height=height,
                    width=width,
                    guidance=guidance_tensor,
                )
                noise_pred_uncond = diffusion_model(**model_kwargs_uncond)

                # Run adapters if present for conditioned using run_adapters
                adapter_block_samples_text = run_adapters(
                    adapters=denoise_node.adapters,
                    adapter_inputs=denoise_node.adapter_inputs,
                    diffusion_model_id=diffusion_model.id,
                    latents=latents,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    height=height,
                    width=width,
                    guidance=guidance_tensor,
                )

                model_kwargs_text = prepare_model_kwargs(
                    diffusion_model_id=diffusion_model.id,
                    latents=latents,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    adapter_block_samples=adapter_block_samples_text,
                    height=height,
                    width=width,
                    guidance=guidance_tensor,
                )

                noise_pred_text = diffusion_model(**model_kwargs_text)

                latents = scheduler(
                    latents=latents,
                    timestep=timestep,
                    noise_pred_uncond=noise_pred_uncond,
                    noise_pred_text=noise_pred_text,
                    guidance_scale=cfg_guidance_scale,
                    mode="step_classifier_free_guidance",
                )
            else:
                prompt_embeds = denoise_node.base_model_inputs.prompt_embeds
                pooled_prompt_embeds = (
                    denoise_node.base_model_inputs.pooled_prompt_embeds
                )

                adapter_block_samples = run_adapters(
                    adapters=denoise_node.adapters,
                    adapter_inputs=denoise_node.adapter_inputs,
                    diffusion_model_id=diffusion_model.id,
                    latents=latents,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    height=height,
                    width=width,
                    guidance=guidance_tensor,
                )

                model_kwargs = prepare_model_kwargs(
                    diffusion_model_id=diffusion_model.id,
                    latents=latents,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    adapter_block_samples=adapter_block_samples,
                    height=height,
                    width=width,
                    guidance=guidance_tensor,
                )

                noise_pred = diffusion_model(**model_kwargs)

                latents = scheduler(
                    latents=original_latents_for_scheduler_step,
                    timestep=timestep,
                    noise_pred=noise_pred,
                    mode="step",
                )

            if i == num_inference_steps - 1:
                for node in unrolled_workflow.workflow_nodes:
                    for input_name in node.get_inputs().keys():
                        input_io = node.get_inputs()[input_name]
                        if (
                            input_io is not None
                            and input_io.name == denoise_node.denoised_latents.name
                        ):
                            # replace the previous denoised latents with the final latents
                            node.set_input(input_name, latents)

    # Clear the denoise nodes
    unrolled_workflow.denoise_nodes = []

    # logger.debug(f"Unrolled workflow: {unrolled_workflow}")
    return unrolled_workflow
