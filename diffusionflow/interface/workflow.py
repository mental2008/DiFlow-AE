import functools
import json
import time
import traceback
from typing import Any, Dict, List, Optional

import aiohttp
import requests
import logging

from diffusionflow.interface.denoise_node import DenoiseNode
from diffusionflow.interface.node_io import (
    AdapterInputs,
    DiffusionModelInputs,
    NodeIO,
    SchedulerInputs,
    SourceType,
)
from diffusionflow.interface.request import InferenceRequest
from diffusionflow.interface.workflow_context import WorkflowContext
from diffusionflow.interface.workflow_node import WorkflowNode
from diffusionflow.operators.models.adapters.base_adapter import BaseAdapter
from diffusionflow.operators.models.diffusion_models.base_diffusion_model import (
    BaseDiffusionModel,
)
from diffusionflow.operators.schedulers.base_scheduler import BaseScheduler


class Workflow:
    def __init__(self, name: str):
        self.name = name
        self.workflow_nodes: List[WorkflowNode] = []
        self.inputs: Dict[str, NodeIO] = {}
        self.outputs: Dict[str, str] = {}
        self.denoise_nodes: List[DenoiseNode] = []
        WorkflowContext.set_current_workflow(self)

    def __repr__(self):
        return f"""
        Workflow(
            name={self.name},
            workflow_nodes={self.workflow_nodes},
            inputs={self.inputs},
            outputs={self.outputs},
            denoise_nodes={self.denoise_nodes},
        )
        """

    def add_workflow_node(self, workflow_node: WorkflowNode) -> None:
        self.workflow_nodes.append(workflow_node)

    def add_input(self, name: str, data_type: type) -> NodeIO:
        self.inputs[name] = NodeIO(
            name=name, data_type=data_type, source_type=SourceType.INPUT
        )
        return self.inputs[name]

    def add_output(self, node_output: NodeIO, name: str) -> None:
        self.outputs[node_output.name] = name

    def add_denoise_node(
        self,
        model: BaseDiffusionModel,
        scheduler: BaseScheduler,
        base_model_inputs: DiffusionModelInputs,
        scheduler_inputs: SchedulerInputs,
        adapters: Optional[List[BaseAdapter]] = None,
        adapter_inputs: Optional[List[AdapterInputs]] = None,
    ) -> NodeIO:
        denoise_node = DenoiseNode(
            model=model,
            scheduler=scheduler,
            base_model_inputs=base_model_inputs,
            adapters=adapters,
            adapter_inputs=adapter_inputs,
            scheduler_inputs=scheduler_inputs,
        )
        self.denoise_nodes.append(denoise_node)
        return denoise_node.denoised_latents

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [node.to_dict() for node in self.workflow_nodes],
            "outputs": {
                node_output: output_name
                for node_output, output_name in self.outputs.items()
            },
            "denoise_nodes": [
                denoise_node.to_dict() for denoise_node in self.denoise_nodes
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, workflow_dict: Dict[str, Any]) -> "Workflow":
        try:
            workflow = cls(name=workflow_dict["name"])

            # Add workflow nodes
            for node_dict in workflow_dict["nodes"]:
                workflow_node = WorkflowNode.from_dict(node_dict)

                for _, input_io in workflow_node.get_inputs().items():
                    if input_io.source_type == SourceType.INPUT:
                        workflow.add_input(input_io.name, input_io.data_type)

                for node_output, output_name in workflow_dict["outputs"].items():
                    for output_io in workflow_node.get_outputs().values():
                        if output_io.name == node_output:
                            workflow.add_output(output_io, output_name)

                workflow.add_workflow_node(workflow_node)

            # Add denoise nodes
            for denoise_node_dict in workflow_dict["denoise_nodes"]:
                denoise_node = DenoiseNode.from_dict(denoise_node_dict)

                workflow.denoise_nodes.append(denoise_node)

            return workflow
        except Exception as e:
            print(f"Error creating workflow from dict: {e}")
            raise e


def register_workflow(
    workflow: Workflow,
    server_url: str = "http://localhost:8000",
    service_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Register a workflow directly with the backend service

    Args:
        workflow: The workflow to register
        server_url: The URL of the backend service
        service_config: Optional service configuration

    Returns:
        The service ID of the registered workflow
    """
    # Convert workflow to JSON
    workflow_json = workflow.to_json()
    print(f"Registering workflow: {workflow_json}")

    # Send workflow to backend service
    response = requests.post(
        f"{server_url}/api/workflow/register",
        json={
            "workflow": workflow_json,
            "service_config": service_config or {},
        },
    )

    if response.status_code != 200:
        raise Exception(f"Failed to register workflow: {response.text}")
    return response.json()["service_id"]


def run_inference(
    service_id: str, inputs: Dict[str, Any], server_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """Run inference on a registered workflow"""
    response = requests.post(
        f"{server_url}/api/workflow/{service_id}/inference", json={"inputs": inputs}
    )

    if response.status_code != 200:
        raise Exception(f"Failed to run inference: {response.text}")
    return response.json()

async def run_inference_async(
    service_id: str,
    request: InferenceRequest,
    session: aiohttp.ClientSession,
    server_url: str = "http://localhost:8000",
) -> Dict[str, Any]:
    """Run inference on a registered workflow asynchronously

    Args:
        service_id: The workflow/service identifier
        request: InferenceRequest instance containing inputs and optionally timeout
        session: aiohttp session
        server_url: Server URL

    Returns:
        Dict containing 'response_json' and 'latency' keys
    """
    try:
        start_time = time.time()

        payload = request.model_dump(exclude_none=True)

        logging.debug(
            f"Starting inference for service {service_id} with request: {payload}"
        )

        async with session.post(
            f"{server_url}/api/workflow/{service_id}/inference",
            json=payload,
        ) as response:
            result = await response.json()
            end_time = time.time()
            latency = end_time - start_time
            logging.debug(
                f"Completed inference for service {service_id} in {latency:.2f}s"
            )
            return {"response_json": result, "latency": latency}
    except Exception as e:
        logging.error(f"Error in inference for service {service_id}: {e}")
        logging.error(traceback.format_exc())
        raise e