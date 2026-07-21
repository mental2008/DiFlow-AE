# python examples/run_workflow.py --service-id <service_id>
import argparse
import base64
import io
import time

import numpy as np
from diffusers.utils import load_image
from PIL import Image

from diffusionflow.interface import run_inference


def decode_image(img_str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(img_str)))


def process_response(response, store_images):
    if response["status"] == "success":
        results = response["results"]
        if store_images:
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
        raise ValueError(f"Invalid response: {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--service-id",
        type=str,
        required=True,
        choices=[
            # SD35 Large
            "sd35_large_txt2img_workflow",
            "sd35_large_txt2img_cfg_workflow",
            "sd35_large_txt2img_controlnet_depth_workflow",
            "sd35_large_txt2img_controlnet_canny_workflow",
            "sd35_large_txt2img_controlnet_depth_cfg_workflow",
            "sd35_large_txt2img_controlnet_canny_cfg_workflow",
        ],
    )
    parser.add_argument("--server-url", type=str, default="http://localhost:8000")
    parser.add_argument("--store-images", action="store_true", default=False)
    args = parser.parse_args()

    service_id = args.service_id
    server_url = args.server_url
    store_images = args.store_images

    start_time = time.time()

    if service_id == "sd35_large_txt2img_workflow":
        response = run_inference(
            service_id,
            inputs={
                "prompt": "A capybara holding a sign that reads Hello World",
                "num_inference_steps": 28,
                "seed": 0,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 1.0,
            },
            server_url=server_url,
        )

        process_response(response, store_images)

    elif service_id == "sd35_large_txt2img_cfg_workflow":
        response = run_inference(
            service_id,
            inputs={
                "prompt": "A capybara holding a sign that reads Hello World",
                "negative_prompt": "NSFW, nude, naked, porn, ugly",
                "num_inference_steps": 28,
                "seed": 0,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 7.0,
            },
            server_url=server_url,
        )

        process_response(response, store_images)

    elif (
        service_id == "sd35_large_txt2img_controlnet_depth_workflow"
        or service_id == "sd35_large_txt2img_controlnet_canny_workflow"
    ):
        # Load and encode the control image
        if service_id == "sd35_large_txt2img_controlnet_depth_workflow":
            control_image = load_image("imgs/flux_depth_image.png")
        elif service_id == "sd35_large_txt2img_controlnet_canny_workflow":
            control_image = load_image("imgs/flux_canny_image.png")
        control_image = control_image.convert("RGB")
        buffered = io.BytesIO()
        control_image.save(buffered, format="PNG")
        buffered.seek(0)
        control_image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        start_time = time.time()
        response = run_inference(
            service_id,
            inputs={
                "prompt": "A portrait of a lovely shorthair golden-shaded cat, sitting on a windowsill, capturing every fur detail.",
                "num_inference_steps": 20,
                "seed": 0,
                "height": 1024,
                "width": 1024,
                "control_image": control_image_b64,
                "conditioning_scale": 0.7,
                "guidance_scale": 1.0,
            },
            server_url=server_url,
        )

        process_response(response, store_images)

    elif (
        service_id == "sd35_large_txt2img_controlnet_canny_cfg_workflow"
        or service_id == "sd35_large_txt2img_controlnet_depth_cfg_workflow"
    ):
        # Load and encode the control image
        if service_id == "sd35_large_txt2img_controlnet_canny_cfg_workflow":
            control_image = load_image("imgs/flux_canny_image.png")
        elif service_id == "sd35_large_txt2img_controlnet_depth_cfg_workflow":
            control_image = load_image("imgs/flux_depth_image.png")
        control_image = control_image.convert("RGB")
        buffered = io.BytesIO()
        control_image.save(buffered, format="PNG")
        buffered.seek(0)
        control_image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response = run_inference(
            service_id,
            inputs={
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
            server_url=server_url,
        )

        process_response(response, store_images)

    else:
        raise ValueError(f"Invalid service ID: {service_id}")

    end_time = time.time()
    print(f"For service {service_id}, Time taken: {end_time - start_time} seconds")
