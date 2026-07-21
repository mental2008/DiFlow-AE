import argparse
import asyncio
import json
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from diffusionflow.backend.coordinator import Coordinator, ExecutionTimeoutError
from diffusionflow.backend.scheduler import SchedulingPolicy
from diffusionflow.interface.request import InferenceRequest
from diffusionflow.interface.workflow import Workflow


class WorkflowService:
    def __init__(
        self,
        worker_hostnames: List[str],
        scheduling_policy: SchedulingPolicy,
        base_port: int,
        preload_models_config: str,
        model_batch_config: str,
        enable_early_abort: bool = False,
        op_latencies_config_dir: Optional[str] = None,
    ):
        # self.dist_config = dist_config
        self.workflows: Dict[str, Workflow] = {}
        self.coordinator = Coordinator(
            worker_hostnames=worker_hostnames,
            scheduling_policy=scheduling_policy,
            base_port=base_port,
            preload_models_config=preload_models_config,
            model_batch_config=model_batch_config,
            enable_early_abort=enable_early_abort,
            op_latencies_config_dir=op_latencies_config_dir,
        )

    async def startup(self):
        """Initialize the distributed system on service startup"""
        print(f"Starting workflow service")

        # Wait for all workers to be ready before accepting requests
        await self.coordinator.wait_for_workers_ready()

        # Run the scheduler
        await self.coordinator.run_scheduler()

        print(f"Workflow service ready")

    def register_workflow(
        self, workflow_dict: Dict[str, Any], service_config: Dict
    ) -> str:
        service_id = f"{workflow_dict['name']}"
        print(f"Registering workflow: {service_id}")
        if service_id in self.workflows:
            print(f"Workflow {service_id} already registered")
            return

        # Convert the raw dict to a Workflow
        workflow = Workflow.from_dict(workflow_dict)
        # print(f"{workflow}")
        self.workflows[service_id] = {"workflow": workflow, "config": service_config}

        return service_id

    async def run_inference(
        self, service_id: str, inputs: Dict[str, Any], slo_slack: Optional[float] = None
    ) -> Dict[str, Any]:
        if service_id not in self.workflows:
            raise ValueError(f"Workflow {service_id} not found")

        workflow = self.workflows[service_id]["workflow"]
        # print(f"Running inference for workflow: {service_id}")

        # Generate unique request ID
        request_id = str(uuid.uuid4())

        # Execute workflow asynchronously
        return await self.coordinator.execute_workflow(
            request_id, workflow, inputs, slo_slack=slo_slack
        )

    async def shutdown(self):
        """Cleanup on service shutdown"""
        print("Shutting down workflow service...")
        self.coordinator.cleanup()
        print("Workflow service shutdown complete")


class WorkflowRegistration(BaseModel):
    workflow: str
    service_config: Dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle"""
    # Initialize service
    await workflow_service.startup()

    yield

    # Cleanup on shutdown
    await workflow_service.shutdown()


app = FastAPI(lifespan=lifespan)


@app.post("/api/workflow/register")
async def register_workflow(registration: WorkflowRegistration):
    try:
        workflow_dict = json.loads(registration.workflow)
        service_id = workflow_service.register_workflow(
            workflow_dict, registration.service_config
        )
        return {
            "status": "success",
            "service_id": service_id,
            "message": f"Workflow '{service_id}' registered successfully",
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/workflow/{service_id}/inference")
async def run_inference(service_id: str, request: InferenceRequest):
    try:
        if request.timeout is None or request.profiled_latency is None:
            slo_slack = None
        else:
            slo_slack = request.timeout - request.profiled_latency

        results = await workflow_service.run_inference(
            service_id, request.inputs, slo_slack=slo_slack
        )
        return {"status": "success", "results": results}
    except ExecutionTimeoutError as e:
        # Early abort is an admission-control decision, not an internal error.
        return {"status": "rejected", "error": str(e)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--base-port", type=int, default=14000)
    parser.add_argument("--hostfile", type=str, default="hostfile")
    parser.add_argument(
        "--scheduling-policy",
        type=str,
        default="dynamic",
        choices=["exclusive", "random", "dynamic"],
    )
    parser.add_argument(
        "--preload-models-config",
        type=str,
        default="configs/preload_models.yaml",
        help="Path to preload models YAML config file",
    )
    parser.add_argument(
        "--model-batch-config",
        type=str,
        default="configs/model_batch.json",
        help="Path to model batch JSON config file",
    )
    parser.add_argument(
        "--enable-early-abort",
        action="store_true",
        help="Reject requests early when estimated inflight work exceeds SLO slack",
    )
    parser.add_argument(
        "--op-latencies-config-dir",
        type=str,
        default=Coordinator.OP_LATENCIES_DIR,
        help="Directory containing op_latencies_median.json for early abort",
    )
    args = parser.parse_args()

    hostfile = args.hostfile
    with open(hostfile, "r") as f:
        worker_hostnames = [line.strip() for line in f.readlines()]
    print(f"Worker hostnames: {worker_hostnames}")

    # Initialize the service
    workflow_service = WorkflowService(
        worker_hostnames=worker_hostnames,
        scheduling_policy=SchedulingPolicy(args.scheduling_policy),
        base_port=args.base_port,
        preload_models_config=args.preload_models_config,
        model_batch_config=args.model_batch_config,
        enable_early_abort=args.enable_early_abort,
        op_latencies_config_dir=args.op_latencies_config_dir,
    )

    # Run the FastAPI server
    try:
        config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            loop="asyncio",
            timeout_keep_alive=30,
            timeout_graceful_shutdown=30,
        )
        server = uvicorn.Server(config)
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        asyncio.run(workflow_service.shutdown())
    except Exception as e:
        print(f"Error during server execution: {e}")
        asyncio.run(workflow_service.shutdown())
