import torch
from diffusers import FluxPipeline

model_path = "./models/FLUX.1-schnell"
pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
pipe = pipe.to("cuda")

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
image.save("./ae-results/verify_correctness/diffusers_flux_schnell.png")
