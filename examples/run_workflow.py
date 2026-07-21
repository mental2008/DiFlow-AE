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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--service-id",
        type=str,
        required=True,
        choices=[
            "t5_workflow",
            "t5_func_workflow",
            # SD3
            "sd3_txt2img_workflow",
            "sd3_txt2img_cfg_workflow",
            "sd3_txt2img_controlnet_canny_workflow",
            "sd3_txt2img_controlnet_canny_cfg_workflow",
            "sd3_txt2img_controlnet_pose_workflow",
            "sd3_txt2img_controlnet_pose_cfg_workflow",
            # SD1.5
            "sd15_txt2img_workflow",
            "sd15_txt2img_cfg_workflow",
            "sd15_txt2img_controlnet_canny_workflow",
            "sd15_txt2img_controlnet_canny_cfg_workflow",
            "sd15_txt2img_controlnet_depth_workflow",
            "sd15_txt2img_controlnet_depth_cfg_workflow",
            # SDXL
            "sdxl_txt2img_workflow",
            "sdxl_txt2img_cfg_workflow",
            "sdxl_txt2img_controlnet_canny_workflow",
            "sdxl_txt2img_controlnet_canny_cfg_workflow",
            "sdxl_txt2img_controlnet_depth_workflow",
            "sdxl_txt2img_controlnet_depth_cfg_workflow",
            "sdxl_txt2img_bal_workflow",
            # FLUX
            "flux_txt2img_workflow",
            "flux_txt2img_cfg_workflow",
            "flux_txt2img_controlnet_canny_workflow",
            "flux_txt2img_controlnet_canny_cfg_workflow",
            "flux_txt2img_controlnet_depth_workflow",
            "flux_txt2img_controlnet_depth_cfg_workflow",
        ],
    )
    parser.add_argument("--server-url", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    service_id = args.service_id
    server_url = args.server_url

    start_time = time.time()

    if service_id == "t5_workflow" or service_id == "t5_func_workflow":
        response = run_inference(
            service_id,
            inputs={
                "text_prompt": "Anime style illustration of a girl wearing a suit.",
            },
            server_url=server_url,
        )
        if response["status"] == "success":
            results = response["results"]
            data = np.array(results["text_embed"])
            print(f"text_embed.shape: {data.shape}")
            print(f"The first 5 elements of text_embed: {data[:5]}")

    elif service_id == "sd3_txt2img_workflow":
        response = run_inference(
            service_id,
            inputs={
                "prompt": "Anime style illustration of a girl wearing a suit.",
                "num_inference_steps": 28,
                "seed": 666,
                "height": 1024,
                "width": 1024,
            },
            server_url=server_url,
        )
        if response["status"] == "success":
            results = response["results"]
            if isinstance(results["output_img"], list):
                for idx, img_str in enumerate(results["output_img"]):
                    img = decode_image(img_str)
                    print(f"output_img_{idx}.shape: {img.size}")
                    img.save(f"output_img_{idx}.png")
            else:
                img = decode_image(results["output_img"])
                print(f"output_img.shape: {img.size}")
                img.save(f"output_img_0.png")

    elif service_id == "sd3_txt2img_cfg_workflow":
        response = run_inference(
            service_id,
            inputs={
                "prompt": "Anime style illustration of a girl wearing a suit.",
                "negative_prompt": "NSFW, nude, naked, porn, ugly",
                "num_inference_steps": 28,
                "seed": 666,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 7.0,
            },
            server_url=server_url,
        )

        if response["status"] == "success":
            results = response["results"]
            if isinstance(results["output_img"], list):
                for idx, img_str in enumerate(results["output_img"]):
                    img = decode_image(img_str)
                    print(f"output_img_{idx}.shape: {img.size}")
                    img.save(f"output_img_{idx}.png")
            else:
                img = decode_image(results["output_img"])
                print(f"output_img.shape: {img.size}")
                img.save(f"output_img_0.png")

    elif service_id == "sd3_txt2img_controlnet_canny_workflow":
        # Load and encode the control image
        control_image = load_image("imgs/canny.jpg")
        control_image = control_image.convert("RGB")
        buffered = io.BytesIO()
        control_image.save(buffered, format="PNG")
        buffered.seek(0)
        control_image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response = run_inference(
            service_id,
            inputs={
                "prompt": "Anime style illustration of a girl wearing a suit.",
                "num_inference_steps": 28,
                "seed": 666,
                "height": 1024,
                "width": 1024,
                "control_image": control_image_b64,
                "conditioning_scale": 0.7,
            },
            server_url=server_url,
        )
        if response["status"] == "success":
            results = response["results"]
            if isinstance(results["output_img"], list):
                for idx, img_str in enumerate(results["output_img"]):
                    img = decode_image(img_str)
                    print(f"output_img_{idx}.shape: {img.size}")
                    img.save(f"output_img_{idx}.png")
            else:
                img = decode_image(results["output_img"])
                print(f"output_img.shape: {img.size}")
                img.save(f"output_img_0.png")

    elif service_id == "sd3_txt2img_controlnet_canny_cfg_workflow":
        # Load and encode the control image
        control_image = load_image("imgs/canny.jpg")
        control_image = control_image.convert("RGB")
        buffered = io.BytesIO()
        control_image.save(buffered, format="PNG")
        buffered.seek(0)
        control_image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response = run_inference(
            service_id,
            inputs={
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
            server_url=server_url,
        )

        if response["status"] == "success":
            results = response["results"]
            if isinstance(results["output_img"], list):
                for idx, img_str in enumerate(results["output_img"]):
                    img = decode_image(img_str)
                    print(f"output_img_{idx}.shape: {img.size}")
                    img.save(f"output_img_{idx}.png")
            else:
                img = decode_image(results["output_img"])
                print(f"output_img.shape: {img.size}")
                img.save(f"output_img_0.png")

    elif service_id == "sd3_txt2img_controlnet_pose_workflow":
        # Load and encode the control image
        control_image = load_image("imgs/canny.jpg")  # TODO @ Suyi: change to pose
        control_image = control_image.convert("RGB")
        buffered = io.BytesIO()
        control_image.save(buffered, format="PNG")
        buffered.seek(0)
        control_image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response = run_inference(
            service_id,
            inputs={
                "prompt": "Anime style illustration of a girl wearing a suit.",
                "num_inference_steps": 28,
                "seed": 666,
                "height": 1024,
                "width": 1024,
                "control_image": control_image_b64,
                "conditioning_scale": 0.7,
            },
            server_url=server_url,
        )

        if response["status"] == "success":
            results = response["results"]
            if isinstance(results["output_img"], list):
                for idx, img_str in enumerate(results["output_img"]):
                    img = decode_image(img_str)
                    print(f"output_img_{idx}.shape: {img.size}")
                    img.save(f"output_img_{idx}.png")
            else:
                img = decode_image(results["output_img"])
                print(f"output_img.shape: {img.size}")
                img.save(f"output_img_0.png")

    elif service_id == "sd3_txt2img_controlnet_pose_cfg_workflow":
        # Load and encode the control image
        control_image = load_image("imgs/canny.jpg")  # TODO @ Suyi: change to pose
        control_image = control_image.convert("RGB")
        buffered = io.BytesIO()
        control_image.save(buffered, format="PNG")
        buffered.seek(0)
        control_image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response = run_inference(
            service_id,
            inputs={
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
            server_url=server_url,
        )

        if response["status"] == "success":
            results = response["results"]
            if isinstance(results["output_img"], list):
                for idx, img_str in enumerate(results["output_img"]):
                    img = decode_image(img_str)
                    print(f"output_img_{idx}.shape: {img.size}")
                    img.save(f"output_img_{idx}.png")
            else:
                img = decode_image(results["output_img"])
                print(f"output_img.shape: {img.size}")
                img.save(f"output_img_0.png")

    elif service_id == "sd15_txt2img_workflow":
        response = run_inference(
            service_id,
            inputs={
                "prompt": "a photo of an astronaut riding a horse on mars",
                "negative_prompt": "",
                "num_channels_latents": 4,
                "num_inference_steps": 50,
                "seed": 0,
                "height": 512,
                "width": 512,
            },
            server_url=server_url,
        )
        if response["status"] == "success":
            results = response["results"]
            if isinstance(results["output_img"], list):
                for idx, img_str in enumerate(results["output_img"]):
                    img = decode_image(img_str)
                    print(f"output_img_{idx}.shape: {img.size}")
                    img.save(f"output_img_{idx}.png")
            else:
                img = decode_image(results["output_img"])
                print(f"output_img.shape: {img.size}")
                img.save(f"output_img_0.png")

    elif service_id == "sd15_txt2img_cfg_workflow":
        response = run_inference(
            service_id,
            inputs={
                "prompt": "a photo of an astronaut riding a horse on mars",
                "negative_prompt": "ugly, blurry, low quality",
                "num_channels_latents": 4,
                "num_inference_steps": 50,
                "seed": 0,
                "height": 512,
                "width": 512,
                "guidance_scale": 7.5,
            },
            server_url=server_url,
        )
        if response["status"] == "success":
            results = response["results"]
            if isinstance(results["output_img"], list):
                for idx, img_str in enumerate(results["output_img"]):
                    img = decode_image(img_str)
                    print(f"output_img_{idx}.shape: {img.size}")
                    img.save(f"output_img_{idx}.png")
            else:
                img = decode_image(results["output_img"])
                print(f"output_img.shape: {img.size}")
                img.save(f"output_img_0.png")

    elif (
        service_id == "sd15_txt2img_controlnet_canny_workflow"
        or service_id == "sd15_txt2img_controlnet_depth_workflow"
    ):
        # Load and encode the control image
        if service_id == "sd15_txt2img_controlnet_canny_workflow":
            control_image = load_image("imgs/canny.jpg")
        elif service_id == "sd15_txt2img_controlnet_depth_workflow":
            control_image = load_image("imgs/depth.jpg")
        control_image = control_image.convert("RGB")
        buffered = io.BytesIO()
        control_image.save(buffered, format="PNG")
        buffered.seek(0)
        control_image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response = run_inference(
            service_id,
            inputs={
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
            server_url=server_url,
        )
        if response["status"] == "success":
            results = response["results"]
            if isinstance(results["output_img"], list):
                for idx, img_str in enumerate(results["output_img"]):
                    img = decode_image(img_str)
                    print(f"output_img_{idx}.shape: {img.size}")
                    img.save(f"output_img_{idx}.png")
            else:
                img = decode_image(results["output_img"])
                print(f"output_img.shape: {img.size}")
                img.save(f"output_img_0.png")

    elif (
        service_id == "sd15_txt2img_controlnet_canny_cfg_workflow"
        or service_id == "sd15_txt2img_controlnet_depth_cfg_workflow"
    ):
        # Load and encode the control image
        if service_id == "sd15_txt2img_controlnet_canny_cfg_workflow":
            control_image = load_image("imgs/canny.jpg")
        elif service_id == "sd15_txt2img_controlnet_depth_cfg_workflow":
            control_image = load_image("imgs/depth.jpg")
        control_image = control_image.convert("RGB")
        buffered = io.BytesIO()
        control_image.save(buffered, format="PNG")
        buffered.seek(0)
        control_image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response = run_inference(
            service_id,
            inputs={
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
            server_url=server_url,
        )
        if response["status"] == "success":
            results = response["results"]
            if isinstance(results["output_img"], list):
                for idx, img_str in enumerate(results["output_img"]):
                    img = decode_image(img_str)
                    print(f"output_img_{idx}.shape: {img.size}")
                    img.save(f"output_img_{idx}.png")
            else:
                img = decode_image(results["output_img"])
                print(f"output_img.shape: {img.size}")
                img.save(f"output_img_0.png")

    elif service_id == "flux_txt2img_workflow":
        response = run_inference(
            service_id,
            inputs={
                "prompt": "A cat holding a sign that says hello world",
                "num_inference_steps": 50,
                "seed": 0,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 3.5,
            },
            server_url=server_url,
        )
        if response["status"] == "success":
            results = response["results"]
            if isinstance(results["output_img"], list):
                for idx, img_str in enumerate(results["output_img"]):
                    img = decode_image(img_str)
                    print(f"output_img_{idx}.shape: {img.size}")
                    img.save(f"output_img_{idx}.png")
            else:
                img = decode_image(results["output_img"])
                print(f"output_img.shape: {img.size}")
                img.save(f"output_img_0.png")

    elif service_id == "flux_txt2img_cfg_workflow":
        response = run_inference(
            service_id,
            inputs={
                "prompt": "A cat holding a sign that says hello world",
                "negative_prompt": "ugly, blurry, low quality, deformed, text",
                "cfg_guidance_scale": 7.0,
                "num_inference_steps": 50,
                "seed": 0,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 3.5,
            },
            server_url=server_url,
        )
        if response["status"] == "success":
            results = response["results"]
            if isinstance(results["output_img"], list):
                for idx, img_str in enumerate(results["output_img"]):
                    img = decode_image(img_str)
                    print(f"output_img_{idx}.shape: {img.size}")
                    img.save(f"output_img_{idx}.png")
            else:
                img = decode_image(results["output_img"])
                print(f"output_img.shape: {img.size}")
                img.save(f"output_img_0.png")

    elif (
        service_id == "flux_txt2img_controlnet_canny_workflow"
        or service_id == "flux_txt2img_controlnet_depth_workflow"
    ):

        # Load and encode the control image
        if service_id == "flux_txt2img_controlnet_canny_workflow":
            control_image = load_image("imgs/flux_canny_image.png")
        elif service_id == "flux_txt2img_controlnet_depth_workflow":
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
                "control_image": control_image_b64,
                "conditioning_scale": 0.7,  # originally named controlnet_conditioning_scale
                "guidance_scale": 3.5,
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 50,
                "seed": 0,
            },
            server_url=server_url,
        )
        if response["status"] == "success":
            results = response["results"]
            if isinstance(results["output_img"], list):
                for idx, img_str in enumerate(results["output_img"]):
                    img = decode_image(img_str)
                    print(f"output_img_{idx}.shape: {img.size}")
                    img.save(f"output_img_{idx}.png")
            else:
                img = decode_image(results["output_img"])
                print(f"output_img.shape: {img.size}")
                img.save(f"output_img_0.png")

    elif (
        service_id == "flux_txt2img_controlnet_canny_cfg_workflow"
        or service_id == "flux_txt2img_controlnet_depth_cfg_workflow"
    ):

        # Load and encode the control image
        if service_id == "flux_txt2img_controlnet_canny_cfg_workflow":
            control_image = load_image("imgs/flux_canny_image.png")
        elif service_id == "flux_txt2img_controlnet_depth_cfg_workflow":
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
            server_url=server_url,
        )
        if response["status"] == "success":
            results = response["results"]
            if isinstance(results["output_img"], list):
                for idx, img_str in enumerate(results["output_img"]):
                    img = decode_image(img_str)
                    print(f"output_img_{idx}.shape: {img.size}")
                    img.save(f"output_img_{idx}.png")
            else:
                img = decode_image(results["output_img"])
                print(f"output_img.shape: {img.size}")
                img.save(f"output_img_0.png")

    elif service_id == "sdxl_txt2img_workflow":
        response = run_inference(
            service_id,
            inputs={
                "prompt": "a photo of an astronaut riding a horse on mars",
                "num_inference_steps": 50,
                "seed": 0,
                "height": 1024,
                "width": 1024,
                "num_channels_latents": 4,  # sdxl requires 4 channels for latents
            },
            server_url=server_url,
        )
        if response["status"] == "success":
            results = response["results"]
            if isinstance(results["output_img"], list):
                for idx, img_str in enumerate(results["output_img"]):
                    img = decode_image(img_str)
                    print(f"output_img_{idx}.shape: {img.size}")
                    img.save(f"output_img_{idx}.png")
            else:
                img = decode_image(results["output_img"])
                print(f"output_img.shape: {img.size}")
                img.save(f"output_img_0.png")

    elif service_id == "sdxl_txt2img_cfg_workflow":
        response = run_inference(
            service_id,
            inputs={
                "prompt": "a photo of an astronaut riding a horse on mars",
                "negative_prompt": "ugly, blurry, low quality, deformed",
                "num_inference_steps": 50,
                "seed": 0,
                "height": 1024,
                "width": 1024,
                "num_channels_latents": 4,  # sdxl requires 4 channels for latents
                "guidance_scale": 7.5,
            },
            server_url=server_url,
        )
        if response["status"] == "success":
            results = response["results"]
            if isinstance(results["output_img"], list):
                for idx, img_str in enumerate(results["output_img"]):
                    img = decode_image(img_str)
                    print(f"output_img_{idx}.shape: {img.size}")
                    img.save(f"output_img_{idx}.png")
            else:
                img = decode_image(results["output_img"])
                print(f"output_img.shape: {img.size}")
                img.save(f"output_img_0.png")

    elif (
        service_id == "sdxl_txt2img_controlnet_canny_workflow"
        or service_id == "sdxl_txt2img_controlnet_depth_workflow"
    ):
        # Load and encode the control image
        if service_id == "sdxl_txt2img_controlnet_canny_workflow":
            control_image = load_image("imgs/sdxl_canny_image.png")
        elif service_id == "sdxl_txt2img_controlnet_depth_workflow":
            control_image = load_image("imgs/sdxl_depth_image.png")
        control_image = control_image.convert("RGB")
        buffered = io.BytesIO()
        control_image.save(buffered, format="PNG")
        buffered.seek(0)
        control_image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response = run_inference(
            service_id,
            inputs={
                "prompt": "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting",
                "num_inference_steps": 50,
                "seed": 666,
                "height": 1024,
                "width": 1024,
                "control_image": control_image_b64,
                "conditioning_scale": 0.5,
                "num_channels_latents": 4,  # sdxl requires 4 channels for latents
            },
            server_url=server_url,
        )
        if response["status"] == "success":
            results = response["results"]
            if isinstance(results["output_img"], list):
                for idx, img_str in enumerate(results["output_img"]):
                    img = decode_image(img_str)
                    print(f"output_img_{idx}.shape: {img.size}")
                    img.save(f"output_img_{idx}.png")
            else:
                img = decode_image(results["output_img"])
                print(f"output_img.shape: {img.size}")
                img.save(f"output_img_0.png")

    elif (
        service_id == "sdxl_txt2img_controlnet_canny_cfg_workflow"
        or service_id == "sdxl_txt2img_controlnet_depth_cfg_workflow"
    ):
        # Load and encode the control image
        if service_id == "sdxl_txt2img_controlnet_canny_cfg_workflow":
            control_image = load_image("imgs/sdxl_canny_image.png")
        elif service_id == "sdxl_txt2img_controlnet_depth_cfg_workflow":
            control_image = load_image("imgs/sdxl_depth_image.png")
        control_image = control_image.convert("RGB")
        buffered = io.BytesIO()
        control_image.save(buffered, format="PNG")
        buffered.seek(0)
        control_image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response = run_inference(
            service_id,
            inputs={
                "prompt": "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting",
                "negative_prompt": "ugly, blurry, low quality, deformed, dark",
                "num_inference_steps": 50,
                "seed": 666,
                "height": 1024,
                "width": 1024,
                "control_image": control_image_b64,
                "conditioning_scale": 0.5,
                "num_channels_latents": 4,  # sdxl requires 4 channels for latents
                "guidance_scale": 7.5,
            },
            server_url=server_url,
        )
        if response["status"] == "success":
            results = response["results"]
            if isinstance(results["output_img"], list):
                for idx, img_str in enumerate(results["output_img"]):
                    img = decode_image(img_str)
                    print(f"output_img_{idx}.shape: {img.size}")
                    img.save(f"output_img_{idx}.png")
            else:
                img = decode_image(results["output_img"])
                print(f"output_img.shape: {img.size}")
                img.save(f"output_img_0.png")

    elif service_id == "sdxl_txt2img_bal_workflow":
        response = run_inference(
            service_id,
            inputs={
                "prompt": "a photo of an astronaut riding a horse on mars",
                "num_inference_steps": 50,
                "seed": 0,
                "height": 1024,
                "width": 1024,
                "num_channels_latents": 4,  # sdxl requires 4 channels for latents
            },
            server_url=server_url,
        )
        if response["status"] == "success":
            results = response["results"]
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
        raise ValueError(f"Invalid service ID: {service_id}")

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
