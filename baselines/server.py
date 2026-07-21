import argparse
import asyncio
import json
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import yaml
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from coordinator import Coordinator, ExecutionTimeoutError


class WorkflowService:
    def __init__(self, worker_hostnames: List[str], baseline_config: Dict[str, Any], base_port: int):
        # self.dist_config = dist_config
        self.workflows: Dict[str, str] = {}
        self.coordinator = Coordinator(worker_hostnames=worker_hostnames, baseline_config=baseline_config, base_port=base_port)

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

        self.workflows[service_id] = service_id

        return service_id

    async def run_inference(
        self, service_id: str, inputs: Dict[str, Any], timeout: float = None
    ) -> Dict[str, Any]:
        # if service_id not in self.workflows:
        #     raise ValueError(f"Workflow {service_id} not found")

        # pipeline_name = self.workflows[service_id]
        pipeline_name = service_id
        print(f"Running inference for workflow: {service_id}")

        # Generate unique request ID
        request_id = str(uuid.uuid4())

        # Execute workflow asynchronously
        return await self.coordinator.execute_workflow(request_id, pipeline_name, inputs, timeout=timeout)

    async def shutdown(self):
        """Cleanup on service shutdown"""
        print("Shutting down workflow service...")
        self.coordinator.cleanup()
        print("Workflow service shutdown complete")


class WorkflowRegistration(BaseModel):
    pipeline_name: str


class InferenceRequest(BaseModel):
    inputs: Dict[str, Any]
    timeout: Optional[float] = None  # Maximum execution time in seconds (optional)


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
        results = await workflow_service.run_inference(
            service_id, request.inputs, timeout=request.timeout
        )
        return {"status": "success", "results": results}
    except ExecutionTimeoutError as e:
        # Request exceeded timeout
        return {"status": "failure", "error": str(e)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--hostfile", type=str, default="hostfile")
    parser.add_argument("--base-port", type=int, default=14000)
    parser.add_argument("--baseline-config", type=str, default="./baseline_configs/basic.yml")
    args = parser.parse_args()

    ### parse baseline config ###
    with open(args.baseline_config, "r") as f:
        baseline_config = yaml.safe_load(f)
    print(f"Baseline config: {baseline_config}")

    hostfile = args.hostfile
    with open(hostfile, "r") as f:
        worker_hostnames = [line.strip() for line in f.readlines()]
    print(f"Worker hostnames: {worker_hostnames}")

    # Initialize the service
    workflow_service = WorkflowService(worker_hostnames=worker_hostnames, baseline_config=baseline_config, base_port=args.base_port)

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