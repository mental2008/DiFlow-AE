import asyncio
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple

import zmq
from collections import defaultdict

@dataclass
class Task:
    request_id: str
    pipeline_name: str
    inputs: Dict[str, Any] = None


class ExecutionTimeoutError(Exception):
    """Exception raised when a request exceeds its timeout."""

    pass


class WorkerStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"


class Coordinator:
    def __init__(self, worker_hostnames: List[str], baseline_config: Dict[str, Any] = None, base_port: int = 14000):
        self.worker_hostnames = worker_hostnames
        self.baseline_config = baseline_config
        self.baseline_name = baseline_config["baseline_name"]
        self.base_port = base_port

        # Setup logging
        self._setup_logging()

        # ZMQ setup
        self.context = zmq.Context()
        self.task_sockets: Dict[str, zmq.Socket] = {}  # For sending tasks
        self.result_sockets: Dict[str, zmq.Socket] = {}  # For receiving results

        # Coordinator setup
        self.all_workers_info = (
            self._gather_all_workers_info()
        )  # Maps worker_id to (global_rank, hostname, task_port, result_port)
        self._setup_coordinator_sockets()

        # Task tracking
        self.task_queue: asyncio.Queue[Task] = (
            asyncio.Queue()
        )  # Tasks ready to be scheduled
        ### Suyi: in diffusionflow, map WorkflowNode name -> worker_id
        ### Suyi: here, we map pipeline_name + request_id -> worker_id
        self.active_tasks: Dict[str, str] = {}
        # map request_id -> Task
        self.request_tasks: Dict[str, Task] = {}
        # map request_id -> results
        self.request_results: Dict[str, Dict[str, Any]] = {}
        # map request_id -> completion event
        self.request_completed: Dict[str, asyncio.Event] = {}
        # map request_id -> arrival time
        self.request_arrival_times: Dict[str, float] = {}
        # map request_id -> timeout requirement in seconds (None if no timeout)
        self.request_timeouts: Dict[str, float] = {}
        # map request_id -> failure reason (if request failed)
        self.request_failures: Dict[str, str] = {}

        # Worker availability tracking
        self.worker_status: Dict[str, Dict[str, Any]] = {}
        for worker_id, _ in self.all_workers_info.items():
            self.worker_status[worker_id] = {
                "status": WorkerStatus.IDLE,
                "task": None,
                "last_ping": time.time(),
            }

        # Scheduler
        self._scheduler_task = None

        # Status
        self.is_running = True

        ### build pipeline_placements ###
        # self.pipeline_placements is a dictionary that maps pipeline_name to a list of global ranks
        self.pipeline_placements = defaultdict(list)
        for pipeline_name, pipeline_placements in self.baseline_config["pipelines"].items():
            for cur_pipeline_placement in pipeline_placements:
                self.pipeline_placements[pipeline_name].append(cur_pipeline_placement["rank"])

        self.logger.info(f"pipeline_placements: {self.pipeline_placements}")

    def _setup_logging(self):
        """Setup logging for the coordinator"""
        log_dir = os.environ.get("DIFFUSIONFLOW_LOG_DIR", "logs")

        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Create a unique log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{log_dir}/coordinator_{timestamp}.log"

        log_level_str = os.environ.get("LOGLEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)

        # Setup the logger
        self.logger = logging.getLogger("coordinator")
        self.logger.setLevel(log_level)
        self.logger.propagate = False

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)

        # Console handler that uses sys.stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Configure the root logger to use the same handlers
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        self.logger.info(f"Initialized logging for coordinator, log file: {log_file}")

    async def wait_for_workers_ready(self, timeout_seconds: int = 60):
        """Wait for all workers to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if not self.is_running:
                raise RuntimeError("Coordinator was stopped before workers were ready")
            if self._check_workers_ready():
                return
            await asyncio.sleep(1)
        raise TimeoutError("Workers failed to initialize within timeout period")

    def _check_workers_ready(self) -> bool:
        """Check if all workers are ready"""
        try:
            # Send ping to all workers
            self.logger.info(f"self.task_sockets.items(): {self.task_sockets.items()}")
            for worker_id, socket in self.task_sockets.items():
                socket.send_json({"type": "ping"})
                self.logger.info(f"Sent ping to {worker_id}")

            # Wait for responses
            self.logger.info(f"self.result_sockets.values(): {self.result_sockets.values()}")
            for socket in self.result_sockets.values():
                response = socket.recv_json()
                if response.get("type") != "pong":
                    return False
            return True
        except zmq.ZMQError:
            return False

    def _setup_coordinator_sockets(self):
        """Setup ZMQ sockets for the coordinator"""
        for worker_id, worker_info in self.all_workers_info.items():

            task_socket = self.context.socket(zmq.PUSH)
            result_socket = self.context.socket(zmq.PULL)

            task_socket.connect(
                f"tcp://{worker_info['hostname']}:{worker_info['task_port']}"
            )
            result_socket.connect(
                f"tcp://{worker_info['hostname']}:{worker_info['result_port']}"
            )
            self.logger.info(
                f"Coordinator connected to {worker_id}, {worker_info['hostname']}:{worker_info['task_port']} and {worker_info['hostname']}:{worker_info['result_port']}"
            )

            self.task_sockets[worker_id] = task_socket
            self.result_sockets[worker_id] = result_socket

    def _gather_all_workers_info(self) -> Dict[str, Dict[str, Any]]:
        """Gather information about all workers across all nodes"""
        workers_info = {}
        for rank, hostname in enumerate(self.worker_hostnames):
            worker_id = f"worker_{rank}"
            global_rank = rank
            task_port = self.base_port + global_rank * 2
            result_port = task_port + 1

            self.logger.info(
                f"Worker: {worker_id}, global_rank: {global_rank}, "
                f"task_port: {task_port}, result_port: {result_port}"
            )
            workers_info[worker_id] = {
                "global_rank": global_rank,  # Note: nvshmem pe is the same as global rank
                "hostname": hostname,
                "task_port": task_port,
                "result_port": result_port,
            }
        return workers_info


    async def run_scheduler(self):
        """Run the scheduler that processes the request queue"""
        self._scheduler_task = asyncio.create_task(self._process_task_queue_loop())

    def _check_timeout(self, request_id: str) -> bool:
        """Check if request has exceeded its timeout. Returns True if timeout is exceeded."""
        if (
            request_id not in self.request_timeouts
            or self.request_timeouts[request_id] is None
        ):
            return False

        if request_id not in self.request_arrival_times:
            return False

        elapsed_time = time.time() - self.request_arrival_times[request_id]
        timeout = self.request_timeouts[request_id]

        if elapsed_time > timeout:
            self.logger.warning(
                f"Request {request_id} exceeded timeout: {elapsed_time:.2f}s > {timeout:.2f}s"
            )
            return True
        return False

    async def _process_task_queue_loop(self):
        """Continuously process the task queue"""

        while self.is_running:
            try:
                task = await self.task_queue.get()

                if self._check_timeout(task.request_id):
                    await self._fail_request_timeout(task.request_id)
                    self.logger.debug(
                        f"Skipping task from timed out request {task.request_id}"
                    )

                    self.request_completed[task.request_id].set()
                    continue

                # self.logger.info(f"Processing task {task.pipeline_name}")
                # self.logger.debug(f"Task inputs: {task.inputs}")

                # Find an available worker
                available_worker_id = None
                for worker_id, worker_info in self.all_workers_info.items():
                    if self.worker_status[worker_id]["status"] == WorkerStatus.IDLE:
                        if worker_info["global_rank"] in self.pipeline_placements[task.pipeline_name]:
                            available_worker_id = worker_id
                            break

                if available_worker_id is None:
                    # No workers available yet
                    # Put the task back to the queue
                    await self.task_queue.put(task)
                    await asyncio.sleep(0.00001)  # 10 us
                    continue

                self.worker_status[available_worker_id]["status"] = WorkerStatus.BUSY
                self.worker_status[available_worker_id]["task"] = task

                # Schedule task
                asyncio.create_task(
                    self._schedule_task(task, available_worker_id)
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                traceback.print_exc()
                print(f"Error processing task queue: {e}")
                await asyncio.sleep(0.00001)  # 10 us

    async def _schedule_task(self, task: Task, worker_id: str):
        """Schedule a task to a specific worker"""
        try:
            task_message = {"type": "task", "data": {}}
            self.logger.debug(
                f"Scheduling task for node {task.pipeline_name} on worker {worker_id}"
            )
            task_message["pipeline_name"] = task.pipeline_name
            task_message["data"] = {
                "request_id": task.request_id,
                "pipeline_name": task.pipeline_name,
                "inputs": task.inputs,
            }

            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.task_sockets[worker_id].send_json(task_message)
            )
            self.active_tasks[f"{task.pipeline_name}_{task.request_id}"] = worker_id

            asyncio.create_task(self._gather_result_for_task(task, worker_id))
        except Exception as e:
            print(f"Error scheduling task: {e}")
            traceback.print_exc()
            raise e

    async def _gather_result_for_task(
        self, task: Task, worker_id: str, timeout: int = 6000
    ):
        """Gather results for a task"""
        start_time = time.time()
        max_wait_time = 300  # Note: 5 minutes maximum wait time

        result_socket = self.result_sockets[worker_id]
        poller = zmq.Poller()
        poller.register(result_socket, zmq.POLLIN)

        while True:
            if time.time() - start_time > max_wait_time:
                raise TimeoutError(
                    f"Timeout waiting for result for task {task.pipeline_name} on worker {worker_id}"
                )

            # Poll for result in a non-blocking way
            socks = dict(
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: poller.poll(timeout=timeout)
                )
            )

            if result_socket in socks:
                # Receive response in a non-blocking way
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: result_socket.recv_json()
                )
                self.logger.info(
                    f"{datetime.now()} Received response from worker {worker_id}"
                )

                if response.get("type") == "pong":
                    continue
                elif response.get("type") == "completed":
                    # Update worker status
                    self.worker_status[worker_id]["status"] = WorkerStatus.IDLE
                    self.worker_status[worker_id]["task"] = None

                    response_data = response["data"]
                    request_id = response_data["request_id"]

                    # Check if request has timed out before processing results
                    if self._check_timeout(request_id):
                        await self._fail_request_timeout(request_id)
                        self.logger.warning(
                            f"Request {request_id} completed but exceeded timeout, marking as failed"
                        )
                    else:
                        img_str_list = response_data["img_str_list"]

                        self.request_results[request_id] = {
                            "output_img": img_str_list
                        }

                    self.request_completed[request_id].set()

                    # Clean up task tracking
                    del self.active_tasks[f"{task.pipeline_name}_{task.request_id}"]

                    return

            await asyncio.sleep(0.00001)  # 10 us

    async def _fail_request_timeout(self, request_id: str):
        """Fail a request due to timeout violation"""
        elapsed_time = time.time() - self.request_arrival_times.get(
            request_id, time.time()
        )
        timeout = self.request_timeouts.get(request_id, "N/A")
        failure_msg = (
            f"Request {request_id} exceeded timeout: {elapsed_time:.2f}s > {timeout}s"
        )
        self.logger.error(
            f"Failing request {request_id} due to timeout violation: {elapsed_time:.2f}s > {timeout}s"
        )

        # Store failure reason
        self.request_failures[request_id] = failure_msg

    async def execute_workflow(
        self, request_id: str, pipeline_name: str, inputs: Dict[str, Any], timeout: float = None
    ) -> Dict[str, Any]:
        """Execute a workflow asynchronously"""
        workflow_start_time = time.time()
        
        self.request_arrival_times[request_id] = workflow_start_time
        self.request_timeouts[request_id] = timeout

        task = Task(
            request_id=request_id,
            pipeline_name=pipeline_name,
            inputs=inputs,
        )

        await self.task_queue.put(task)
        # Store tasks for this request
        self.request_tasks[request_id] = task
        self.request_completed[request_id] = asyncio.Event()

        self.logger.debug(f"Added task {pipeline_name} to task queue for request {request_id}")

        # Wait for request to complete
        await self.request_completed[request_id].wait()
        self.logger.info(f"Request {request_id} is complete")

        # Check if request failed
        is_request_failed = request_id in self.request_failures
        failure_msg = None
        if is_request_failed:
            failure_msg = self.request_failures[request_id]

        # Get results (may be empty if request failed)
        results = self.request_results.get(request_id, {})

        # Clean up
        del self.request_tasks[request_id]
        if request_id in self.request_results:
            del self.request_results[request_id]
        del self.request_completed[request_id]
        if request_id in self.request_arrival_times:
            del self.request_arrival_times[request_id]
        if request_id in self.request_timeouts:
            del self.request_timeouts[request_id]
        if request_id in self.request_failures:
            del self.request_failures[request_id]

        workflow_end_time = time.time()
        self.logger.info(
            f"Time taken to execute request {request_id} for workflow {pipeline_name}: {workflow_end_time - workflow_start_time} seconds"
        )

        if is_request_failed:
            raise ExecutionTimeoutError(failure_msg)

        return results

    def cleanup(self):
        """Cleanup coordinator resources"""
        self.is_running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()

        # Stop all workers first with timeout
        for worker_id, socket in self.task_sockets.items():
            try:
                # Use non-blocking send with retry
                for _ in range(3):
                    try:
                        socket.send_json({"type": "stop"}, zmq.NOBLOCK)
                        break
                    except zmq.Again:
                        time.sleep(0.1)
            except Exception as e:
                print(f"Error sending stop signal to {worker_id}: {e}")

        # Give workers time to process stop signal
        time.sleep(3)

        # Send stop signal to all workers
        for worker_id, socket in list(self.task_sockets.items()):
            try:
                socket.close(linger=1000)  # 1 second linger
                del self.task_sockets[worker_id]
            except Exception as e:
                print(f"Error closing task socket for {worker_id}: {e}")

        for worker_id, socket in list(self.result_sockets.items()):
            try:
                socket.close(linger=1000)
                del self.result_sockets[worker_id]
            except Exception as e:
                print(f"Error closing result socket for {worker_id}: {e}")