import torch

# from diffusers.models.controlnet_flux import FluxControlNetModel
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.utils import load_image

base_model = "./models/FLUX.1-schnell"
controlnet_model = (
    "./models/Xlabs-AI--flux-controlnet-canny-diffusers"
)
controlnet = FluxControlNetModel.from_pretrained(
    controlnet_model,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
)

pipe = FluxControlNetPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

# load the control image
control_image = load_image("./imgs/flux_canny_image.png")

prompt = "A portrait of a lovely shorthair golden-shaded cat, sitting on a windowsill, capturing every fur detail."

image = pipe(
    prompt,
    control_image=control_image,
    negative_prompt="NSFW, nude, naked, porn, ugly",
    true_cfg_scale=7.0,
    controlnet_conditioning_scale=0.7,
    guidance_scale=3.5,
    height=1024,
    width=1024,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

image.save("flux-schnell-canny-cfg.png")
