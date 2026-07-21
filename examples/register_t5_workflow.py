# python examples/register_t5_workflow.py
import argparse

from diffusionflow.interface import Workflow, register_workflow
from diffusionflow.operators import T5, Config


def create_workflow(model_path: str) -> Workflow:
    workflow = Workflow(name="t5_workflow")

    # Define model nodes
    t5 = T5(Config(model_path=model_path))

    # Define inputs
    prompt = workflow.add_input(name="text_prompt", data_type=str)

    # Define connections
    text_embed = t5(prompt=prompt)

    # Define outputs
    workflow.add_output(text_embed, name="text_embed")

    return workflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", type=str, default="http://localhost:8000")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/project/infattllm/lyangbk/huggingface/stable-diffusion-3-medium-diffusers",
    )
    args = parser.parse_args()
    server_url = args.server_url

    # Register workflow
    service_id = register_workflow(
        workflow=create_workflow(model_path=args.model_path),
        server_url=server_url,
    )
    print(f"Registered workflow with service ID: {service_id}")
