import torch
from diffusers import FluxPipeline
import time

model_path = "./models/FLUX.1-schnell"
pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
pipe = pipe.to("cuda")

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    negative_prompt="ugly, blurry, low quality, deformed, text",
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0),
    true_cfg_scale=7.0,
).images[0]



start_time = time.time()
for i in range(10):
    image = pipe(
        prompt,
        negative_prompt="ugly, blurry, low quality, deformed, text",
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0),
        true_cfg_scale=7.0,
    ).images[0]
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
