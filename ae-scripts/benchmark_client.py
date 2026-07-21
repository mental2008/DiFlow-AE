import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Tuple

import aiohttp

from benchmark.client_utils import generate_a_single_request
from diffusionflow.interface.request import InferenceRequest
from diffusionflow.interface.workflow import run_inference, run_inference_async
from workflow_timeouts import DEFAULT_WORKFLOW_TIMEOUTS

flux_schnell_workflows = [
    "flux_schnell_txt2img_cfg_workflow",
    "flux_schnell_txt2img_controlnet_canny_cfg_workflow",
    "flux_schnell_txt2img_controlnet_depth_cfg_workflow",
]


def setup_logging(log_file: str = None):
    """Set up logging to write to both file and console (or console only if log_file is None)"""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler (only if log_file is provided)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def load_trace_file(trace_file: str) -> List[Tuple[str, float]]:
    """
    Load trace data from file.

    Format: Each line contains "tenant_label interval"
    Example:
        A 10.5
        B 2.3
        A 5.1

    Returns:
        List of (tenant_label, interval) tuples
    """
    trace_entries = []

    try:
        with open(trace_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                parts = line.split()
                if len(parts) != 2:
                    logging.warning(f"Skipping invalid line {line_num}: {line}")
                    continue

                tenant_label = parts[0]
                try:
                    interval = float(parts[1])
                    trace_entries.append((tenant_label, interval))
                except ValueError:
                    logging.warning(
                        f"Skipping line {line_num} with invalid interval: {line}"
                    )
                    continue

        logging.info(f"Loaded {len(trace_entries)} trace entries from {trace_file}")
        return trace_entries

    except FileNotFoundError:
        logging.error(f"Error: File {trace_file} not found")
        sys.exit(1)


async def send_requests_async(
    trace_entries: List[Tuple[str, float]],
    workflows: List[str],
    pre_created_requests: Dict[str, Dict[str, Any]],
    server_url: str = "http://localhost:8000",
    dry_run: bool = False,
    baseline: bool = False,
):
    """Send requests asynchronously following the trace pattern with specified intervals

    Args:
        trace_entries: List of (tenant_label, interval) tuples
        workflows: List of workflow names
        pre_created_requests: Dict mapping workflow names to request data
        server_url: Server URL
        dry_run: If True, only log requests without sending them
        baseline: If True, remove _cfg from workflow names to get service_id
    """
    all_tasks = []
    first_request_time = None
    last_request_time = None

    def get_service_id(workflow_name: str) -> str:
        """Get the actual service_id from workflow_name, applying baseline mode if needed"""
        if baseline and "_cfg_workflow" in workflow_name:
            # Remove _cfg from the workflow name (e.g., sd35_large_txt2img_cfg_workflow -> sd35_large_txt2img_workflow)
            return workflow_name.replace("_cfg_workflow", "_workflow")
        return workflow_name

    if dry_run:
        logging.info("=" * 80)
        logging.info("DRY RUN MODE: Checking generated requests without sending them")
        logging.info("=" * 80)

        for req_idx, (tenant_label, interval) in enumerate(trace_entries):
            # Calculate workflow index from tenant label (A=0, B=1, C=2, etc.)
            workflow_idx = ord(tenant_label) - ord("A")

            # Validate workflow index
            if workflow_idx < 0 or workflow_idx >= len(workflows):
                logging.warning(f"Invalid tenant label: {tenant_label}, skipping")
                continue

            workflow_name = workflows[workflow_idx]

            # Get the pre-created request for this workflow
            if workflow_name not in pre_created_requests:
                logging.warning(
                    f"Workflow {workflow_name} not found in pre_created_requests, skipping"
                )
                continue

            request = pre_created_requests[workflow_name]

            service_id = get_service_id(workflow_name)

            logging.info(f"\n--- Request {req_idx} ---")
            logging.info(f"Tenant label: {tenant_label}")
            logging.info(f"Workflow: {workflow_name}")
            logging.info(f"Service ID: {service_id}")
            logging.info(f"Interval before next request: {interval}s")
            logging.info(f"Request payload:")

            # Format the request nicely
            # request is an InferenceRequest object, use model_dump() to convert to dict
            request_dict = request.model_dump(exclude_none=True)
            request_payload = {
                "service_id": service_id,
                "inputs": request_dict.get("inputs", {}),
            }
            # Include timeout in payload if it exists
            if request.timeout is not None:
                request_payload["timeout"] = request.timeout

            # Truncate base64 images for readability
            formatted_payload = json.loads(json.dumps(request_payload))
            if "inputs" in formatted_payload:
                for key, value in formatted_payload["inputs"].items():
                    if (
                        isinstance(value, str)
                        and len(value) > 100
                        and key in ["control_image", "image"]
                    ):
                        formatted_payload["inputs"][
                            key
                        ] = f"<base64_image_data: {len(value)} chars>"

            logging.info(json.dumps(formatted_payload, indent=2))

        logging.info("\n" + "=" * 80)
        logging.info(f"Total requests that would be sent: {len(trace_entries)}")
        logging.info("=" * 80)
        return []

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Helper function to wrap request with logging
        async def run_request_with_logging(
            req_idx: int,
            tenant_label: str,
            workflow_name: str,
            service_id: str,
            request: InferenceRequest,
        ):
            """Run a request and log its completion or failure"""
            try:
                result = await run_inference_async(
                    service_id=service_id,
                    request=request,
                    session=session,
                    server_url=server_url,
                )

                # Check if the response indicates success or failure
                response_data = result.get("response_json", {})
                latency = result.get("latency", 0)

                # Store timeout in result for later processing
                result["timeout"] = request.timeout

                # Check if latency exceeds timeout
                if request.timeout is not None and latency > request.timeout:
                    logging.info(
                        f"[FAILED] Request {req_idx} (tenant {tenant_label}, workflow {workflow_name}) "
                        f"exceeded timeout ({request.timeout:.2f}s) with latency {latency:.2f}s"
                    )
                else:
                    status = response_data.get("status")
                    if status == "success":
                        logging.info(
                            f"[SUCCESS] Request {req_idx} (tenant {tenant_label}, workflow {workflow_name}) "
                            f"completed successfully in {latency:.2f}s"
                        )
                    elif status == "rejected":
                        logging.info(
                            f"[REJECTED] Request {req_idx} (tenant {tenant_label}, workflow {workflow_name}) "
                            f"was rejected in {latency:.2f}s"
                        )
                    elif status == "failure":
                        # Extract error message from response (following client_utils.py pattern)
                        error_message = response_data.get(
                            "error", "No error message provided"
                        )
                        logging.info(
                            f"[FAILED] Request {req_idx} (tenant {tenant_label}, workflow {workflow_name}) "
                            f"failed with status 'failure' in {latency:.2f}s. Error: {error_message}"
                        )
                    else:
                        print(result)
                        # Status is neither "success" nor "failure"
                        logging.warning(
                            f"[WARNING] Request {req_idx} (tenant {tenant_label}, workflow {workflow_name}) "
                            f"completed but status is '{status}' (expected 'success' or 'failure') in {latency:.2f}s. Response: {response_data}"
                        )

                return result
            except asyncio.TimeoutError as e:
                logging.error(
                    f"[FAILED] Request {req_idx} (tenant {tenant_label}, workflow {workflow_name}) "
                    f"timed out: {e}"
                )
                raise
            except aiohttp.ClientError as e:
                logging.error(
                    f"[FAILED] Request {req_idx} (tenant {tenant_label}, workflow {workflow_name}) "
                    f"failed with client error: {e}"
                )
                raise
            except Exception as e:
                logging.error(
                    f"[FAILED] Request {req_idx} (tenant {tenant_label}, workflow {workflow_name}) "
                    f"failed with exception: {type(e).__name__}: {e}"
                )
                logging.debug(f"Full traceback for request {req_idx}:", exc_info=True)
                raise

        for req_idx, (tenant_label, interval) in enumerate(trace_entries):
            # Calculate workflow index from tenant label (A=0, B=1, C=2, etc.)
            workflow_idx = ord(tenant_label) - ord("A")

            # Validate workflow index
            if workflow_idx < 0 or workflow_idx >= len(workflows):
                logging.warning(f"Invalid tenant label: {tenant_label}, skipping")
                continue

            workflow_name = workflows[workflow_idx]

            # Get the pre-created request for this workflow
            if workflow_name not in pre_created_requests:
                logging.warning(
                    f"Workflow {workflow_name} not found in pre_created_requests, skipping"
                )
                continue

            request = pre_created_requests[workflow_name]

            current_time = time.time()
            if first_request_time is None:
                first_request_time = current_time
            last_request_time = current_time

            service_id = get_service_id(workflow_name)

            logging.info(
                f"Sending request {req_idx} with tenant {tenant_label}, workflow {workflow_name}, service_id {service_id}, datetime {datetime.now()}"
            )
            # Compute service_id based on baseline mode (request already contains inputs and optionally timeout)
            task = asyncio.create_task(
                run_request_with_logging(
                    req_idx=req_idx,
                    tenant_label=tenant_label,
                    workflow_name=workflow_name,
                    service_id=service_id,
                    request=request,
                )
            )
            all_tasks.append(task)

            # Wait for the specified interval before sending the next request
            # (except for the last request)
            if req_idx < len(trace_entries) - 1:
                await asyncio.sleep(interval)

        # Wait for all requests to complete and collect responses
        responses = await asyncio.gather(*all_tasks, return_exceptions=True)

    ### Log response details ###
    logging.info(f"Received {len(responses)} responses")

    # Calculate SLO attainment
    successful_requests = 0
    rejected_requests = 0
    failed_requests = 0
    total_requests = len(responses)

    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            failed_requests += 1
            # Error was already logged in run_request_with_logging, but log summary here
            error_type = type(response).__name__
            error_msg = str(response)
            logging.error(f"Response {i} failed with {error_type}: {error_msg}")
            continue

        if not isinstance(response, dict):
            failed_requests += 1
            logging.warning(f"Response {i} has unexpected type: {type(response)}")
            continue

        if "response_json" in response:
            logging.debug(f"Response {i}: {response['response_json']}")
            # Check if the response indicates success or failure
            response_data = response["response_json"]
            latency = response.get("latency", 0)
            timeout = response.get("timeout")
            
            # Check if latency exceeds timeout first
            if timeout is not None and latency > timeout:
                failed_requests += 1
                logging.info(
                    f"Response {i} exceeded timeout ({timeout:.2f}s) with latency {latency:.2f}s"
                )
            else:
                status = response_data.get("status")
                if status == "success":
                    successful_requests += 1
                elif status == "failure":
                    failed_requests += 1
                    # Extract error message from response (following client_utils.py pattern)
                    error_message = response_data.get("error", "No error message provided")
                    logging.info(
                        f"Response {i} has status 'failure'. Error: {error_message}"
                    )
                elif status == "rejected":
                    rejected_requests += 1
                    logging.info(
                        f"Response {i} has status 'rejected'. Message: {response_data.get('message', 'No message provided')}"
                    )
                else:
                    # Status is neither "success" nor "failure"
                    failed_requests += 1
                    logging.warning(
                        f"Response {i} has unexpected status '{status}' (expected 'success' or 'failure'): {response_data}"
                    )
        else:
            failed_requests += 1
            logging.warning(f"Response {i} missing response_json: {response}")

    # Calculate SLO attainment
    slo_attainment = (
        (successful_requests / total_requests * 100) if total_requests > 0 else 0
    )

    ### Log latency statistics ###
    latencies = [
        r["latency"] for r in responses if isinstance(r, dict) and "latency" in r
    ]
    logging.info(f"Request Latencies: {latencies}")
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        logging.info(f"========== Latency Statistics:")
        logging.info(f"Average latency: {avg_latency:.2f} seconds")
        logging.info(f"Min latency: {min(latencies):.2f} seconds")
        logging.info(f"Max latency: {max(latencies):.2f} seconds")

        if first_request_time is not None and last_request_time is not None:
            time_interval = last_request_time - first_request_time
            effective_rps = total_requests / time_interval if time_interval > 0 else 0
            logging.info(f"Effective RPS: {effective_rps:.2f} requests/second")
            logging.info(f"Effective RPM: {effective_rps * 60:.2f} requests/minute")
            logging.info(f"Total requests: {total_requests}")
            logging.info(f"Successful requests: {successful_requests}")
            logging.info(f"Rejected requests: {rejected_requests}")
            logging.info(f"Failed requests: {failed_requests}")
            logging.info(f"SLO Attainment: {slo_attainment:.2f}%")
            logging.info(f"Time interval: {time_interval:.2f} seconds")

        logging.info(f"==========")

    return responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", type=str, default="http://localhost:8000")
    parser.add_argument(
        "--trace-file",
        type=str,
        default="generated_trace.txt",
        help="Path to the generated trace file",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use baseline mode (removes cfg from workflow names)",
    )
    parser.add_argument(
        "--warmup", action="store_true", help="Run warmup (only console logging)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Log directory name (e.g., diflow, shepherd, diffusers, clockwork). Log file will be saved to ./ae-results/client_logs/{log_dir}/benchmark_client_{timestamp}.log",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check generated requests without sending them",
    )
    parser.add_argument(
        "--slo-scale",
        type=float,
        default=1.0,
        help="SLO scale factor to multiply timeout values (default: 1.0)",
    )
    parser.add_argument(
        "--workflows",
        type=str,
        default="sd35_large",
        choices=[
            "sd3_family",
            "flux_family",
            "sd3_medium",
            "sd35_large",
            "flux_dev",
            "flux_schnell",
        ],
        help="Workflows to use (default: sd35_large)",
    )
    args = parser.parse_args()

    server_url = args.server_url
    trace_file = args.trace_file
    baseline = args.baseline
    warmup = args.warmup
    log_dir_arg = args.log_dir

    # Set up logging
    # If warmup is enabled, don't save to log file (only console)
    # Otherwise, use log_dir to determine log file path
    log_file = None
    if not warmup:
        if log_dir_arg is not None:
            log_dir = f"./ae-results/client_logs/{log_dir_arg}"
            os.makedirs(log_dir, exist_ok=True)
            log_file = f"{log_dir}/benchmark_client_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        else:
            # Default behavior: use ./ae-results/client_logs root
            log_dir = "./ae-results/client_logs"
            os.makedirs(log_dir, exist_ok=True)
            log_file = f"{log_dir}/benchmark_client_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = setup_logging(log_file)
    logging.info("=" * 80)
    logging.info("Benchmark Client Settings:")
    logging.info(f"  Trace file: {trace_file}")
    logging.info(f"  Baseline mode: {baseline}")
    logging.info(f"  SLO scale: {args.slo_scale}")
    logging.info(f"  Workflows: {args.workflows}")
    logging.info(f"  Warmup: {warmup}")
    logging.info(f"  Server URL: {server_url}")
    if log_file:
        logging.info(f"  Log file: {log_file}")
    else:
        logging.info(f"  Log file: (warmup mode - console only)")
    logging.info("=" * 80)

    # Load trace file
    # Try trace_file as-is first (absolute or relative to CWD)
    # If not found, try relative to script directory
    if os.path.isabs(trace_file) or os.path.exists(trace_file):
        trace_file_path = trace_file
    else:
        trace_file_path = os.path.join(os.path.dirname(__file__), trace_file)
    logging.info(f"Loading trace from: {trace_file_path}")
    trace_entries = load_trace_file(trace_file_path)

    if args.workflows == "sd3_family":
        workflows = sd3_family_workflows
    elif args.workflows == "flux_family":
        workflows = flux_family_workflows
    elif args.workflows == "sd3_medium":
        workflows = sd3_medium_workflows
    elif args.workflows == "sd35_large":
        workflows = sd35_large_workflows
    elif args.workflows == "flux_dev":
        workflows = flux_dev_workflows
    elif args.workflows == "flux_schnell":
        workflows = flux_schnell_workflows
    else:
        raise ValueError(f"Invalid workflows: {args.workflows}")

    logging.info(f"Selected workflows: {workflows}")

    # Get SLO scale from args
    slo_scale = args.slo_scale

    # Pre-create requests for all workflows with timeout configuration
    pre_created_requests = {}
    for workflow_name in workflows:
        # Get timeout for this workflow (if configured) and apply SLO scale
        workflow_timeout = DEFAULT_WORKFLOW_TIMEOUTS.get(workflow_name)
        final_timeout = None
        if workflow_timeout is not None:
            final_timeout = workflow_timeout * slo_scale
        pre_created_requests[workflow_name] = generate_a_single_request(
            workflow_name, timeout=final_timeout, profiled_latency=workflow_timeout
        )
        logging.debug(
            f"Pre-created request for {workflow_name} with timeout={final_timeout} (base={workflow_timeout}, slo_scale={slo_scale})"
        )

    if baseline:
        logging.info(
            "Baseline mode enabled: service_id will be computed by removing _cfg from workflow names"
        )

    # Log trace summary
    tenant_counts = {}
    for tenant_label, _ in trace_entries:
        tenant_counts[tenant_label] = tenant_counts.get(tenant_label, 0) + 1
    logging.info(f"Trace summary: {len(trace_entries)} total requests")
    for tenant_label in sorted(tenant_counts.keys()):
        logging.info(f"  Tenant {tenant_label}: {tenant_counts[tenant_label]} requests")

    # Run the async function
    start_time = time.time()
    response_list = asyncio.run(
        send_requests_async(
            trace_entries=trace_entries,
            workflows=workflows,
            pre_created_requests=pre_created_requests,
            server_url=server_url,
            dry_run=args.dry_run,
            baseline=baseline,
        )
    )
    end_time = time.time()
    if not args.dry_run:
        logging.info(
            f"Time taken to complete {len(trace_entries)} requests: {end_time - start_time:.2f} seconds"
        )