import base64
import io
from typing import Any, Dict, Optional

import numpy as np
from diffusers.utils import load_image
from PIL import Image

from diffusionflow.interface.request import InferenceRequest


def decode_image(img_str: str) -> Image.Image:
    """Decode a base64-encoded image string to a PIL Image.

    Args:
        img_str: Base64-encoded image string

    Returns:
        PIL Image object
    """
    return Image.open(io.BytesIO(base64.b64decode(img_str)))


def encode_control_image(image_path: str) -> str:
    """Load and encode a control image to base64.

    Args:
        image_path: Path to the control image file

    Returns:
        Base64-encoded image string
    """
    control_image = load_image(image_path)
    control_image = control_image.convert("RGB")
    buffered = io.BytesIO()
    control_image.save(buffered, format="PNG")
    buffered.seek(0)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def process_response(response: Dict[str, Any]) -> None:
    """Process the inference response based on the output keys.

    Args:
        response: Response dictionary from the inference API

    The response is expected to contain "status" which is either "success" or "failure".
    If "status" is "success", the response will also contain "results".
    If "status" is "failure", the response will contain "error".
    """
    # The response should contain a "status" field.
    # When status == "success": response also contains a "results" field.
    # When status == "failure": response also contains an "error" field.
    status = response.get("status")
    if status == "success":
        results = response["results"]

        # Check for text embeddings (e.g., from t5_workflow)
        if "text_embed" in results:
            data = np.array(results["text_embed"])
            print(f"text_embed.shape: {data.shape}")
            print(f"The first 5 elements of text_embed: {data[:5]}")
        # Check for image outputs
        elif "output_img" in results:
            if isinstance(results["output_img"], list):
                for idx, img_str in enumerate(results["output_img"]):
                    img = decode_image(img_str)
                    print(f"output_img_{idx}.shape: {img.size}")
                    img.save(f"output_img_{idx}.png")
            else:
                img = decode_image(results["output_img"])
                print(f"output_img.shape: {img.size}")
                img.save(f"output_img_0.png")
        else:
            raise ValueError(
                f"Unknown response format. Available keys: {list(results.keys())}"
            )
    elif status == "failure":
        # If failure, there should be an "error" field in the response.
        error_msg = response.get("error", "No error message provided")
        raise RuntimeError(f"Inference failed: {error_msg}")
    else:
        raise ValueError(f"Invalid response: {response}")


def _create_inference_request(
    inputs: Dict[str, Any], timeout: Optional[float], profiled_latency: Optional[float]
) -> InferenceRequest:
    """Helper function to create InferenceRequest with optional timeout"""
    return InferenceRequest(inputs=inputs, timeout=timeout, profiled_latency=profiled_latency)


def generate_a_single_request(
    service_id: str, timeout: Optional[float] = None, 
    profiled_latency: Optional[float] = None,
) -> InferenceRequest:
    """
    Generate a single request for a given service/workflow.

    Args:
        service_id: The workflow/service identifier
        timeout: Optional timeout in seconds. If None, no timeout is set.

    Returns:
        InferenceRequest containing inputs and optionally timeout
    """

    if service_id == "t5_workflow":
        return _create_inference_request(
            {
                "text_prompt": "Anime style illustration of a girl wearing a suit.",
            },
            timeout,
            profiled_latency
        )

    elif (
        service_id == "sd3_txt2img_workflow"
        or service_id == "sd3_txt2img_lora_workflow"
    ):
        return _create_inference_request(
            {
                "prompt": "Anime style illustration of a girl wearing a suit.",
                "num_inference_steps": 28,
                "seed": 666,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 0.0,
            },
            timeout,
            profiled_latency
        )

    elif service_id == "sd3_txt2img_cfg_workflow":
        return _create_inference_request(
            {
                "prompt": "Anime style illustration of a girl wearing a suit.",
                "negative_prompt": "NSFW, nude, naked, porn, ugly",
                "num_inference_steps": 28,
                "seed": 666,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 7.0,
            },
            timeout,
            profiled_latency
        )

    elif service_id == "sd3_txt2img_lora_cfg_workflow":
        return _create_inference_request(
            {
                "prompt": "Anime style illustration of a girl wearing a suit., yarn art style",
                "negative_prompt": "NSFW, nude, naked, porn, ugly",
                "num_inference_steps": 28,
                "seed": 666,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 7.0,
            },
            timeout,
            profiled_latency
        )

    elif service_id == "sd3_txt2img_controlnet_canny_workflow":
        control_image_b64 = encode_control_image("imgs/canny.jpg")
        return _create_inference_request(
            {
                "prompt": "Anime style illustration of a girl wearing a suit.",
                "num_inference_steps": 28,
                "seed": 666,
                "height": 1024,
                "width": 1024,
                "control_image": control_image_b64,
                "conditioning_scale": 0.7,
                "guidance_scale": 0.0,
            },
            timeout,
            profiled_latency
        )

    elif service_id == "sd3_txt2img_controlnet_canny_cfg_workflow":
        control_image_b64 = encode_control_image("imgs/canny.jpg")
        return _create_inference_request(
            {
                "prompt": "Anime style illustration of a girl wearing a suit.",
                "negative_prompt": "NSFW, nude, naked, porn, ugly",
                "num_inference_steps": 28,
                "seed": 666,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 7.0,
                "control_image": control_image_b64,
                "conditioning_scale": 0.7,
            },
            timeout,
            profiled_latency
        )

    elif service_id == "sd3_txt2img_controlnet_pose_workflow":
        # TODO @ Suyi: change to pose
        control_image_b64 = encode_control_image("imgs/canny.jpg")
        return _create_inference_request(
            {
                "prompt": "Anime style illustration of a girl wearing a suit.",
                "num_inference_steps": 28,
                "seed": 666,
                "height": 1024,
                "width": 1024,
                "control_image": control_image_b64,
                "conditioning_scale": 0.7,
                "guidance_scale": 0.0,
            },
            timeout,
            profiled_latency
        )

    elif service_id == "sd3_txt2img_controlnet_pose_cfg_workflow":
        # TODO @ Suyi: change to pose
        control_image_b64 = encode_control_image("imgs/canny.jpg")
        return _create_inference_request(
            {
                "prompt": "Anime style illustration of a girl wearing a suit.",
                "negative_prompt": "NSFW, nude, naked, porn, ugly",
                "num_inference_steps": 28,
                "seed": 666,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 7.0,
                "control_image": control_image_b64,
                "conditioning_scale": 0.7,
            },
            timeout,
            profiled_latency
        )

    elif service_id == "sd15_txt2img_workflow":
        return _create_inference_request(
            {
                "prompt": "a photo of an astronaut riding a horse on mars",
                "negative_prompt": "",
                "num_inference_steps": 50,
                "seed": 0,
                "height": 512,
                "width": 512,
                "guidance_scale": 0.0,
                "num_channels_latents": 4,
            },
            timeout,
            profiled_latency
        )

    elif service_id == "sd15_txt2img_cfg_workflow":
        return _create_inference_request(
            {
                "prompt": "a photo of an astronaut riding a horse on mars",
                "negative_prompt": "ugly, blurry, low quality",
                "num_channels_latents": 4,
                "num_inference_steps": 50,
                "seed": 0,
                "height": 512,
                "width": 512,
                "guidance_scale": 7.5,
                "num_channels_latents": 4,
            },
            timeout,
            profiled_latency
        )

    elif (
        service_id == "sd15_txt2img_controlnet_canny_workflow"
        or service_id == "sd15_txt2img_controlnet_depth_workflow"
    ):
        # Load and encode the control image
        if service_id == "sd15_txt2img_controlnet_canny_workflow":
            control_image_b64 = encode_control_image("imgs/canny.jpg")
        elif service_id == "sd15_txt2img_controlnet_depth_workflow":
            control_image_b64 = encode_control_image("imgs/depth.jpg")
        return _create_inference_request(
            {
                "prompt": "futuristic-looking woman",
                "negative_prompt": "",
                "num_channels_latents": 4,
                "num_inference_steps": 20,
                "seed": 0,
                "control_image": control_image_b64,
                "height": 512,
                "width": 512,
                "conditioning_scale": 1.0,
            },
            timeout,
            profiled_latency
        )

    elif (
        service_id == "sd15_txt2img_controlnet_canny_cfg_workflow"
        or service_id == "sd15_txt2img_controlnet_depth_cfg_workflow"
    ):
        # Load and encode the control image
        if service_id == "sd15_txt2img_controlnet_canny_cfg_workflow":
            control_image_b64 = encode_control_image("imgs/canny.jpg")
        elif service_id == "sd15_txt2img_controlnet_depth_cfg_workflow":
            control_image_b64 = encode_control_image("imgs/depth.jpg")
        return _create_inference_request(
            {
                "prompt": "futuristic-looking woman",
                "negative_prompt": "ugly, blurry, low quality, deformed",
                "num_channels_latents": 4,
                "num_inference_steps": 20,
                "seed": 0,
                "control_image": control_image_b64,
                "height": 512,
                "width": 512,
                "conditioning_scale": 1.0,
                "guidance_scale": 7.5,
            },
            timeout,
            profiled_latency
        )

    elif service_id == "flux_txt2img_workflow":
        return _create_inference_request(
            {
                "prompt": "A cat holding a sign that says hello world",
                "num_inference_steps": 50,
                "seed": 0,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 3.5,
            },
            timeout,
            profiled_latency
        )

    elif service_id == "flux_txt2img_cfg_workflow":
        return _create_inference_request(
            {
                "prompt": "A cat holding a sign that says hello world",
                "negative_prompt": "ugly, blurry, low quality",
                "cfg_guidance_scale": 7.0,
                "num_inference_steps": 50,
                "seed": 0,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 7.5,
            },
            timeout,
            profiled_latency
        )

    elif (
        service_id == "flux_txt2img_controlnet_canny_workflow"
        or service_id == "flux_txt2img_controlnet_depth_workflow"
    ):
        # Load and encode the control image
        if service_id == "flux_txt2img_controlnet_canny_workflow":
            control_image_b64 = encode_control_image("imgs/flux_canny_image.png")
        elif service_id == "flux_txt2img_controlnet_depth_workflow":
            control_image_b64 = encode_control_image("imgs/flux_depth_image.png")

        return _create_inference_request(
            {
                "prompt": "A portrait of a lovely shorthair golden-shaded cat, sitting on a windowsill, capturing every fur detail.",
                "control_image": control_image_b64,
                "conditioning_scale": 0.7,  # originally named controlnet_conditioning_scale
                "guidance_scale": 3.5,
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 50,
                "seed": 0,
            },
            timeout,
            profiled_latency
        )

    elif (
        service_id == "flux_txt2img_controlnet_canny_cfg_workflow"
        or service_id == "flux_txt2img_controlnet_depth_cfg_workflow"
    ):
        # Load and encode the control image
        if service_id == "flux_txt2img_controlnet_canny_cfg_workflow":
            control_image_b64 = encode_control_image("imgs/flux_canny_image.png")
        elif service_id == "flux_txt2img_controlnet_depth_cfg_workflow":
            control_image_b64 = encode_control_image("imgs/flux_depth_image.png")

        return _create_inference_request(
            {
                "prompt": "A portrait of a lovely shorthair golden-shaded cat, sitting on a windowsill, capturing every fur detail.",
                "negative_prompt": "ugly, blurry, low quality, deformed, dark, text",
                "cfg_guidance_scale": 7.0,
                "control_image": control_image_b64,
                "conditioning_scale": 0.7,  # originally named controlnet_conditioning_scale
                "guidance_scale": 3.5,
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 50,
                "seed": 0,
            },
            timeout,
            profiled_latency
        )

    elif service_id == "sdxl_txt2img_workflow":
        return _create_inference_request(
            {
                "prompt": "a photo of an astronaut riding a horse on mars",
                "num_inference_steps": 50,
                "seed": 0,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 1.0,
                "num_channels_latents": 4,  # sdxl requires 4 channels for latents
            },
            timeout,
            profiled_latency
        )

    elif service_id == "sdxl_txt2img_cfg_workflow":
        return _create_inference_request(
            {
                "prompt": "a photo of an astronaut riding a horse on mars",
                "negative_prompt": "ugly, blurry, low quality",
                "cfg_guidance_scale": 7.0,
                "num_inference_steps": 50,
                "seed": 0,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 7.5,
                "num_channels_latents": 4,
            },
            timeout,
            profiled_latency
        )

    elif (
        service_id == "sdxl_txt2img_controlnet_canny_workflow"
        or service_id == "sdxl_txt2img_controlnet_depth_workflow"
    ):
        # Load and encode the control image
        if service_id == "sdxl_txt2img_controlnet_canny_workflow":
            control_image_b64 = encode_control_image("imgs/sdxl_canny_image.png")
        elif service_id == "sdxl_txt2img_controlnet_depth_workflow":
            control_image_b64 = encode_control_image("imgs/sdxl_depth_image.png")
        return _create_inference_request(
            {
                "prompt": "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting",
                "num_inference_steps": 50,
                "seed": 666,
                "height": 1024,
                "width": 1024,
                "control_image": control_image_b64,
                "conditioning_scale": 0.5,
                "num_channels_latents": 4,  # sdxl requires 4 channels for latents
            },
            timeout,
            profiled_latency
        )

    elif (
        service_id == "sdxl_txt2img_controlnet_canny_cfg_workflow"
        or service_id == "sdxl_txt2img_controlnet_depth_cfg_workflow"
    ):
        # Load and encode the control image
        if service_id == "sdxl_txt2img_controlnet_canny_cfg_workflow":
            control_image_b64 = encode_control_image("imgs/sdxl_canny_image.png")
        elif service_id == "sdxl_txt2img_controlnet_depth_cfg_workflow":
            control_image_b64 = encode_control_image("imgs/sdxl_depth_image.png")
        return _create_inference_request(
            {
                "prompt": "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting",
                "negative_prompt": "ugly, blurry, low quality, deformed, dark",
                "cfg_guidance_scale": 7.0,
                "num_inference_steps": 50,
                "seed": 666,
                "height": 1024,
                "width": 1024,
                "control_image": control_image_b64,
                "conditioning_scale": 0.5,
                "num_channels_latents": 4,  # sdxl requires 4 channels for latents
                "guidance_scale": 7.5,
            },
            timeout,
            profiled_latency
        )

    elif service_id == "sdxl_txt2img_nirvana_workflow":
        return _create_inference_request(
            {
                "prompt": "a photo of an astronaut riding a horse on mars",
                "num_inference_steps": 50,
                "seed": 0,
                "height": 1024,
                "width": 1024,
                "strength": 1.0 - 10 / 50.0,
                "init_image_path": "imgs/sdxl_img2img_init_image.png",
            },
            timeout,
            profiled_latency
        )

    elif service_id == "sdxl_txt2img_bal_workflow":
        return _create_inference_request(
            {
                "prompt": "a photo of an astronaut riding a horse on mars",
                "num_inference_steps": 50,
                "seed": 0,
                "height": 1024,
                "width": 1024,
                "num_channels_latents": 4,  # sdxl requires 4 channels for latents
            },
            timeout,
            profiled_latency
        )

    elif service_id == "flux_schnell_txt2img_workflow":
        return _create_inference_request(
            {
                "prompt": "A cat holding a sign that says hello world",
                "num_inference_steps": 4,
                "seed": 0,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 0.0,
            },
            timeout,
            profiled_latency
        )

    elif service_id == "flux_schnell_txt2img_cfg_workflow":
        return _create_inference_request(
            {
                "prompt": "A cat holding a sign that says hello world",
                "negative_prompt": "ugly, blurry, low quality, deformed, text",
                "cfg_guidance_scale": 7.0,
                "num_inference_steps": 4,
                "seed": 0,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 0.0,
            },
            timeout,
            profiled_latency
        )

    elif (
        service_id == "flux_schnell_txt2img_controlnet_canny_cfg_workflow"
        or service_id == "flux_schnell_txt2img_controlnet_depth_cfg_workflow"
    ):
        # Load and encode the control image
        if service_id == "flux_schnell_txt2img_controlnet_canny_cfg_workflow":
            control_image_b64 = encode_control_image("imgs/flux_canny_image.png")
        elif service_id == "flux_schnell_txt2img_controlnet_depth_cfg_workflow":
            control_image_b64 = encode_control_image("imgs/flux_depth_image.png")

        return _create_inference_request(
            {
                "prompt": "A portrait of a lovely shorthair golden-shaded cat, sitting on a windowsill, capturing every fur detail.",
                "negative_prompt": "ugly, blurry, low quality, deformed, dark, text",
                "cfg_guidance_scale": 7.0,
                "control_image": control_image_b64,
                "conditioning_scale": 0.7,  # originally named controlnet_conditioning_scale
                "guidance_scale": 0.0,
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 4,
                "seed": 0,
            },
            timeout,
            profiled_latency
        )

    elif (
        service_id == "flux_schnell_txt2img_controlnet_canny_workflow"
        or service_id == "flux_schnell_txt2img_controlnet_depth_workflow"
    ):
        # Load and encode the control image
        if service_id == "flux_schnell_txt2img_controlnet_canny_workflow":
            control_image_b64 = encode_control_image("imgs/flux_canny_image.png")
        elif service_id == "flux_schnell_txt2img_controlnet_depth_workflow":
            control_image_b64 = encode_control_image("imgs/flux_depth_image.png")

        return _create_inference_request(
            {
                "prompt": "A portrait of a lovely shorthair golden-shaded cat, sitting on a windowsill, capturing every fur detail.",
                "control_image": control_image_b64,
                "conditioning_scale": 0.7,  # originally named controlnet_conditioning_scale
                "guidance_scale": 0.0,
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 4,
                "seed": 0,
            },
            timeout,
            profiled_latency
        )

    elif service_id == "sd35_large_txt2img_workflow":
        return _create_inference_request(
            {
                "prompt": "A capybara holding a sign that reads Hello World",
                "num_inference_steps": 28,
                "seed": 0,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 1.0,
            },
            timeout,
            profiled_latency
        )

    elif service_id == "sd35_large_txt2img_cfg_workflow":
        return _create_inference_request(
            {
                "prompt": "A capybara holding a sign that reads Hello World",
                "negative_prompt": "NSFW, nude, naked, porn, ugly",
                "num_inference_steps": 28,
                "seed": 0,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 7.0,
            },
            timeout,
            profiled_latency
        )

    elif (
        service_id == "sd35_large_txt2img_controlnet_depth_workflow"
        or service_id == "sd35_large_txt2img_controlnet_canny_workflow"
    ):
        # Load and encode the control image
        if service_id == "sd35_large_txt2img_controlnet_depth_workflow":
            control_image_b64 = encode_control_image("imgs/flux_depth_image.png")
        elif service_id == "sd35_large_txt2img_controlnet_canny_workflow":
            control_image_b64 = encode_control_image("imgs/flux_canny_image.png")
        return _create_inference_request(
            {
                "prompt": "A portrait of a lovely shorthair golden-shaded cat, sitting on a windowsill, capturing every fur detail.",
                "num_inference_steps": 40,
                "seed": 0,
                "height": 1024,
                "width": 1024,
                "control_image": control_image_b64,
                "conditioning_scale": 0.7,
                "guidance_scale": 1.0,
            },
            timeout,
            profiled_latency
        )

    elif (
        service_id == "sd35_large_txt2img_controlnet_canny_cfg_workflow"
        or service_id == "sd35_large_txt2img_controlnet_depth_cfg_workflow"
    ):
        # Load and encode the control image
        if service_id == "sd35_large_txt2img_controlnet_canny_cfg_workflow":
            control_image_b64 = encode_control_image("imgs/flux_canny_image.png")
        elif service_id == "sd35_large_txt2img_controlnet_depth_cfg_workflow":
            control_image_b64 = encode_control_image("imgs/flux_depth_image.png")
        return _create_inference_request(
            {
                "prompt": "A portrait of a lovely shorthair golden-shaded cat, sitting on a windowsill, capturing every fur detail.",
                "negative_prompt": "NSFW, nude, naked, porn, ugly",
                "num_inference_steps": 40,
                "seed": 0,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 7.0,
                "control_image": control_image_b64,
                "conditioning_scale": 0.7,
            },
            timeout,
            profiled_latency
        )

    else:
        raise ValueError(f"Invalid service_id: {service_id}")