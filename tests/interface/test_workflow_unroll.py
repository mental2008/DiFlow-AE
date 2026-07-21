import json
import unittest

from diffusionflow.interface.workflow import Workflow
from diffusionflow.interface.workflow_unroll import unroll_workflow

WORKFLOW_JSON = """
{
  "name": "sd3_txt2img_workflow",
  "nodes": [
    {
      "name": "LatentsGenerator_9736c1e2-68e3-4e2c-9773-03b3185d4ba0",
      "id": "LatentsGenerator",
      "model_path": null,
      "mode": "default",
      "inputs": {
        "seed": {
          "name": "seed",
          "data_type": "int",
          "source_type": "input",
          "source_node": null
        },
        "height": {
          "name": "height",
          "data_type": "int",
          "source_type": "input",
          "source_node": null
        },
        "width": {
          "name": "width",
          "data_type": "int",
          "source_type": "input",
          "source_node": null
        }
      },
      "outputs": {
        "latents": {
          "name": "LatentsGenerator_9736c1e2-68e3-4e2c-9773-03b3185d4ba0:latents",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "LatentsGenerator_9736c1e2-68e3-4e2c-9773-03b3185d4ba0"
        }
      }
    },
    {
      "name": "CLIP_L_3f3fee47-201d-4a4a-8ef9-0868be657759",
      "id": "CLIP_L",
      "model_path": "/project/infattllm/lyangbk/huggingface/stable-diffusion-3-medium-diffusers",
      "mode": "default",
      "inputs": {
        "prompt": {
          "name": "prompt",
          "data_type": "str",
          "source_type": "input",
          "source_node": null
        }
      },
      "outputs": {
        "prompt_embeds": {
          "name": "CLIP_L_3f3fee47-201d-4a4a-8ef9-0868be657759:prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "CLIP_L_3f3fee47-201d-4a4a-8ef9-0868be657759"
        },
        "pooled_prompt_embeds": {
          "name": "CLIP_L_3f3fee47-201d-4a4a-8ef9-0868be657759:pooled_prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "CLIP_L_3f3fee47-201d-4a4a-8ef9-0868be657759"
        }
      }
    },
    {
      "name": "CLIP_L_a3af072f-9956-4e03-91e3-a04f86c2cbf0",
      "id": "CLIP_L",
      "model_path": "/project/infattllm/lyangbk/huggingface/stable-diffusion-3-medium-diffusers",
      "mode": "default",
      "inputs": {
        "prompt": {
          "name": "negative_prompt",
          "data_type": "str",
          "source_type": "input",
          "source_node": null
        }
      },
      "outputs": {
        "prompt_embeds": {
          "name": "CLIP_L_a3af072f-9956-4e03-91e3-a04f86c2cbf0:prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "CLIP_L_a3af072f-9956-4e03-91e3-a04f86c2cbf0"
        },
        "pooled_prompt_embeds": {
          "name": "CLIP_L_a3af072f-9956-4e03-91e3-a04f86c2cbf0:pooled_prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "CLIP_L_a3af072f-9956-4e03-91e3-a04f86c2cbf0"
        }
      }
    },
    {
      "name": "CLIP_G_c55a913c-0428-4466-84f8-33b316ac7c68",
      "id": "CLIP_G",
      "model_path": "/project/infattllm/lyangbk/huggingface/stable-diffusion-3-medium-diffusers",
      "mode": "default",
      "inputs": {
        "prompt": {
          "name": "prompt",
          "data_type": "str",
          "source_type": "input",
          "source_node": null
        }
      },
      "outputs": {
        "prompt_embeds": {
          "name": "CLIP_G_c55a913c-0428-4466-84f8-33b316ac7c68:prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "CLIP_G_c55a913c-0428-4466-84f8-33b316ac7c68"
        },
        "pooled_prompt_embeds": {
          "name": "CLIP_G_c55a913c-0428-4466-84f8-33b316ac7c68:pooled_prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "CLIP_G_c55a913c-0428-4466-84f8-33b316ac7c68"
        }
      }
    },
    {
      "name": "CLIP_G_3c5afea8-67ea-4f14-b746-56a7c62bb562",
      "id": "CLIP_G",
      "model_path": "/project/infattllm/lyangbk/huggingface/stable-diffusion-3-medium-diffusers",
      "mode": "default",
      "inputs": {
        "prompt": {
          "name": "negative_prompt",
          "data_type": "str",
          "source_type": "input",
          "source_node": null
        }
      },
      "outputs": {
        "prompt_embeds": {
          "name": "CLIP_G_3c5afea8-67ea-4f14-b746-56a7c62bb562:prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "CLIP_G_3c5afea8-67ea-4f14-b746-56a7c62bb562"
        },
        "pooled_prompt_embeds": {
          "name": "CLIP_G_3c5afea8-67ea-4f14-b746-56a7c62bb562:pooled_prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "CLIP_G_3c5afea8-67ea-4f14-b746-56a7c62bb562"
        }
      }
    },
    {
      "name": "T5_5779b958-4d56-4aa8-86d1-1d432081add3",
      "id": "T5",
      "model_path": "/project/infattllm/lyangbk/huggingface/stable-diffusion-3-medium-diffusers",
      "mode": "default",
      "inputs": {
        "prompt": {
          "name": "prompt",
          "data_type": "str",
          "source_type": "input",
          "source_node": null
        }
      },
      "outputs": {
        "prompt_embeds": {
          "name": "T5_5779b958-4d56-4aa8-86d1-1d432081add3:prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "T5_5779b958-4d56-4aa8-86d1-1d432081add3"
        }
      }
    },
    {
      "name": "T5_a277fa31-7400-4e21-bd5e-efec1c7cef07",
      "id": "T5",
      "model_path": "/project/infattllm/lyangbk/huggingface/stable-diffusion-3-medium-diffusers",
      "mode": "default",
      "inputs": {
        "prompt": {
          "name": "negative_prompt",
          "data_type": "str",
          "source_type": "input",
          "source_node": null
        }
      },
      "outputs": {
        "prompt_embeds": {
          "name": "T5_a277fa31-7400-4e21-bd5e-efec1c7cef07:prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "T5_a277fa31-7400-4e21-bd5e-efec1c7cef07"
        }
      }
    },
    {
      "name": "StableDiffusion3TextEncoder_2d374bff-c412-4bbc-a270-339a10234186",
      "id": "StableDiffusion3TextEncoder",
      "model_path": null,
      "mode": "default",
      "inputs": {
        "clip_prompt_embeds": {
          "name": "CLIP_L_3f3fee47-201d-4a4a-8ef9-0868be657759:prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "CLIP_L_3f3fee47-201d-4a4a-8ef9-0868be657759"
        },
        "clip_pooled_prompt_embeds": {
          "name": "CLIP_L_3f3fee47-201d-4a4a-8ef9-0868be657759:pooled_prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "CLIP_L_3f3fee47-201d-4a4a-8ef9-0868be657759"
        },
        "clip_prompt_2_embeds": {
          "name": "CLIP_G_c55a913c-0428-4466-84f8-33b316ac7c68:prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "CLIP_G_c55a913c-0428-4466-84f8-33b316ac7c68"
        },
        "clip_pooled_prompt_2_embeds": {
          "name": "CLIP_G_c55a913c-0428-4466-84f8-33b316ac7c68:pooled_prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "CLIP_G_c55a913c-0428-4466-84f8-33b316ac7c68"
        },
        "t5_prompt_embeds": {
          "name": "T5_5779b958-4d56-4aa8-86d1-1d432081add3:prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "T5_5779b958-4d56-4aa8-86d1-1d432081add3"
        }
      },
      "outputs": {
        "prompt_embeds": {
          "name": "StableDiffusion3TextEncoder_2d374bff-c412-4bbc-a270-339a10234186:prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "StableDiffusion3TextEncoder_2d374bff-c412-4bbc-a270-339a10234186"
        },
        "pooled_prompt_embeds": {
          "name": "StableDiffusion3TextEncoder_2d374bff-c412-4bbc-a270-339a10234186:pooled_prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "StableDiffusion3TextEncoder_2d374bff-c412-4bbc-a270-339a10234186"
        }
      }
    },
    {
      "name": "StableDiffusion3TextEncoder_77826707-fa99-4223-9709-fa693b878720",
      "id": "StableDiffusion3TextEncoder",
      "model_path": null,
      "mode": "default",
      "inputs": {
        "clip_prompt_embeds": {
          "name": "CLIP_L_a3af072f-9956-4e03-91e3-a04f86c2cbf0:prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "CLIP_L_a3af072f-9956-4e03-91e3-a04f86c2cbf0"
        },
        "clip_pooled_prompt_embeds": {
          "name": "CLIP_L_a3af072f-9956-4e03-91e3-a04f86c2cbf0:pooled_prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "CLIP_L_a3af072f-9956-4e03-91e3-a04f86c2cbf0"
        },
        "clip_prompt_2_embeds": {
          "name": "CLIP_G_3c5afea8-67ea-4f14-b746-56a7c62bb562:prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "CLIP_G_3c5afea8-67ea-4f14-b746-56a7c62bb562"
        },
        "clip_pooled_prompt_2_embeds": {
          "name": "CLIP_G_3c5afea8-67ea-4f14-b746-56a7c62bb562:pooled_prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "CLIP_G_3c5afea8-67ea-4f14-b746-56a7c62bb562"
        },
        "t5_prompt_embeds": {
          "name": "T5_a277fa31-7400-4e21-bd5e-efec1c7cef07:prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "T5_a277fa31-7400-4e21-bd5e-efec1c7cef07"
        }
      },
      "outputs": {
        "prompt_embeds": {
          "name": "StableDiffusion3TextEncoder_77826707-fa99-4223-9709-fa693b878720:prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "StableDiffusion3TextEncoder_77826707-fa99-4223-9709-fa693b878720"
        },
        "pooled_prompt_embeds": {
          "name": "StableDiffusion3TextEncoder_77826707-fa99-4223-9709-fa693b878720:pooled_prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "StableDiffusion3TextEncoder_77826707-fa99-4223-9709-fa693b878720"
        }
      }
    },
    {
      "name": "StableDiffusion3VAE_b3ca477d-2e9c-4f17-95f7-99455b4efe54",
      "id": "StableDiffusion3VAE",
      "model_path": "/project/infattllm/lyangbk/huggingface/stable-diffusion-3-medium-diffusers",
      "mode": "decode_latents",
      "inputs": {
        "latents": {
          "name": "StableDiffusion3_Scheduler_82d2bb92-992f-42f9-91a1-c0d1ec5ad4c4:denoised_latents",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "StableDiffusion3_Scheduler_82d2bb92-992f-42f9-91a1-c0d1ec5ad4c4"
        }
      },
      "outputs": {
        "image": {
          "name": "StableDiffusion3VAE_b3ca477d-2e9c-4f17-95f7-99455b4efe54:image",
          "data_type": "Image",
          "source_type": "node",
          "source_node": "StableDiffusion3VAE_b3ca477d-2e9c-4f17-95f7-99455b4efe54"
        }
      }
    }
  ],
  "outputs": {
    "StableDiffusion3VAE_b3ca477d-2e9c-4f17-95f7-99455b4efe54:image": "output_img"
  },
  "denoise_nodes": [
    {
      "name": "StableDiffusion3_Scheduler_82d2bb92-992f-42f9-91a1-c0d1ec5ad4c4",
      "model": "StableDiffusion3",
      "model_path": "/project/infattllm/lyangbk/huggingface/stable-diffusion-3-medium-diffusers",
      "scheduler": "Scheduler",
      "scheduler_path": "/project/infattllm/lyangbk/huggingface/stable-diffusion-3-medium-diffusers",
      "base_model_inputs": {
        "latents": {
          "name": "LatentsGenerator_9736c1e2-68e3-4e2c-9773-03b3185d4ba0:latents",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "LatentsGenerator_9736c1e2-68e3-4e2c-9773-03b3185d4ba0"
        },
        "prompt_embeds": {
          "name": "StableDiffusion3TextEncoder_2d374bff-c412-4bbc-a270-339a10234186:prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "StableDiffusion3TextEncoder_2d374bff-c412-4bbc-a270-339a10234186"
        },
        "pooled_prompt_embeds": {
          "name": "StableDiffusion3TextEncoder_2d374bff-c412-4bbc-a270-339a10234186:pooled_prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "StableDiffusion3TextEncoder_2d374bff-c412-4bbc-a270-339a10234186"
        },
        "negative_prompt_embeds": {
          "name": "StableDiffusion3TextEncoder_77826707-fa99-4223-9709-fa693b878720:prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "StableDiffusion3TextEncoder_77826707-fa99-4223-9709-fa693b878720"
        },
        "negative_pooled_prompt_embeds": {
          "name": "StableDiffusion3TextEncoder_77826707-fa99-4223-9709-fa693b878720:pooled_prompt_embeds",
          "data_type": "Tensor",
          "source_type": "node",
          "source_node": "StableDiffusion3TextEncoder_77826707-fa99-4223-9709-fa693b878720"
        }
      },
      "scheduler_inputs": {
        "num_inference_steps": {
          "name": "num_inference_steps",
          "data_type": "int",
          "source_type": "input",
          "source_node": null
        },
        "guidance_scale": {
          "name": "guidance_scale",
          "data_type": "float",
          "source_type": "input",
          "source_node": null
        }
      },
      "denoised_latents": {
        "name": "StableDiffusion3_Scheduler_82d2bb92-992f-42f9-91a1-c0d1ec5ad4c4:denoised_latents",
        "data_type": "Tensor",
        "source_type": "node",
        "source_node": "StableDiffusion3_Scheduler_82d2bb92-992f-42f9-91a1-c0d1ec5ad4c4"
      }
    }
  ]
}
"""


class TestWorkflowUnroll(unittest.TestCase):
    def setUp(self):
        # Create workflow from JSON
        workflow_dict = json.loads(WORKFLOW_JSON)
        self.workflow = Workflow.from_dict(workflow_dict)

        # Define test inputs
        self.test_inputs = {
            "prompt": "a photo of a cat",
            "negative_prompt": "blurry, low quality",
            "num_inference_steps": 2,
            "guidance_scale": 7.5,
            "seed": 42,
            "height": 1024,
            "width": 1024,
        }

    def test_unroll_workflow(self):
        unrolled_workflow = unroll_workflow(
            workflow=self.workflow,
            inputs=self.test_inputs,
        )

        with open("unrolled_workflow.json", "w") as f:
            json.dump(unrolled_workflow.to_dict(), f)


if __name__ == "__main__":
    unittest.main()
