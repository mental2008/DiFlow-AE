"""
Dynamic scheduler implementation for DYNAMIC scheduling policy.

This scheduler uses latency estimation and worker load balancing
to intelligently select workers for task groups.
"""

import logging
import math
import random
from typing import Any, Dict, List

import pandas as pd

from benchmark.benchmark_utils import (
    get_model_execution_time_for_batch_size,
    get_model_host_mem_to_gpu_time,
)
from diffusionflow.operators.schedulers.base_scheduler import BaseScheduler

from .base_scheduler import Scheduler
from .task import Task, get_task_id
from .utils import SchedulingPolicy, have_same_patches, next_power_of_2

logger = logging.getLogger(__name__)

# Data engine profiling file paths
BENCHMARK_DIR = "benchmark/benchmark_results"
DATA_ENGINE_PROFILING_1N2G_FILE = f"{BENCHMARK_DIR}/overhead_data_1n2g.csv"
DATA_ENGINE_PROFILING_2N1G_FILE = f"{BENCHMARK_DIR}/overhead_data_2n1g.csv"

# Map dtype string to bytes per element
DTYPE_BYTES_MAP = {
    "torch.float32": 4,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.half": 2,
    "torch.int64": 8,
    "torch.int32": 4,
    "torch.int16": 2,
    "torch.int8": 1,
    "torch.uint8": 1,
    "torch.bool": 1,
}


class DynamicScheduler(Scheduler):
    """Scheduler implementation for DYNAMIC scheduling policy."""

    def __init__(
        self,
        all_workers_info: Dict[int, Dict[str, Any]],
        model_configs: Dict[str, Any] = None,
    ):
        super().__init__(SchedulingPolicy.DYNAMIC, all_workers_info)
        self.model_configs = model_configs or {}

        self.data_engine_profiling_intra_node = pd.read_csv(
            DATA_ENGINE_PROFILING_1N2G_FILE
        )
        self.data_engine_profiling_inter_node = pd.read_csv(
            DATA_ENGINE_PROFILING_2N1G_FILE
        )

        # In order to avoid selecting a worker that is too busy.
        # The current threshold is 10 ms.
        self.worker_latency_threshold = 0.01

        logger.info(f"Initialized dynamic scheduler with all workers info: {self.all_workers_info}")

    def _initialize_custom_worker_status(self):
        """Initialize worker status for dynamic scheduling."""
        for worker_rank, _ in self.all_workers_info.items():
            self.worker_status[worker_rank]["estimated_latency"] = {}

    def _calc_latency(
        self,
        task_group: List[Task],
        dst_worker_rank: int,
        loaded_models: Dict[str, Any] = None,
    ) -> float:
        """Calculate the latency of a task group"""
        workflow_node = task_group[0].workflow_node
        batch_size = len(task_group)

        model_name = workflow_node.op.id
        logger.debug(
            f"Calculating latency for {model_name} at worker {dst_worker_rank}"
        )

        ignore_model_names = ["IndexedTensor", "GuidanceTensor"]
        if isinstance(workflow_node.op, BaseScheduler):
            ignore_model_names.append(workflow_node.op.id)

        if (
            model_name not in ignore_model_names
            and model_name not in self.model_configs
        ):
            logger.warning(f"No model config found for model {model_name}")
            return float("inf")

        model_config = (
            self.model_configs[model_name]
            if model_name not in ignore_model_names
            else None
        )

        # Calculate tensor transfer latency based on data engine profiling (intra-node and inter-node)
        tensor_transfer_latency = 0
        dst_hostname = self.all_workers_info[dst_worker_rank]["hostname"]
        for task in task_group:
            # Use node_input_locations to determine the source workers for each input
            for input_name, worker_tensorinfo_dict in task.node_input_locations.items():
                possible_source_workers = list(worker_tensorinfo_dict.keys())

                # Prefer source worker in the same node as dest_worker
                selected_src_worker_rank = None
                for sw in possible_source_workers:
                    if self.all_workers_info[sw]["hostname"] == dst_hostname:
                        selected_src_worker_rank = sw
                        break
                if selected_src_worker_rank is None:
                    # Fallback: pick the first available source worker
                    selected_src_worker_rank = possible_source_workers[0]

                if selected_src_worker_rank == dst_worker_rank:
                    # Skip if the source worker is the same as the dest worker.
                    continue

                # Determine if intra-node or inter-node
                src_hostname = self.all_workers_info[selected_src_worker_rank][
                    "hostname"
                ]
                intra_node = src_hostname == dst_hostname

                # Calculate tensor size (in bytes from num_elements * dtype_size)
                tensor_info = worker_tensorinfo_dict.get(selected_src_worker_rank, None)
                dtype = tensor_info["dtype"]
                size = tensor_info["size"]
                dtype_str = str(dtype).lower()
                if dtype_str not in DTYPE_BYTES_MAP:
                    raise ValueError(f"Unknown dtype: {dtype_str}")
                dtype_bytes = DTYPE_BYTES_MAP[dtype_str]

                def _num_elements_for_size(size):
                    num_elements = 1
                    for dim in size:
                        num_elements *= dim
                    return num_elements

                tensor_size_bytes = _num_elements_for_size(size) * dtype_bytes

                # Find the closest block size in profiling
                if intra_node:
                    profiling_df = self.data_engine_profiling_intra_node
                else:
                    profiling_df = self.data_engine_profiling_inter_node

                # Find the row with the closest block_size_bytes >= tensor_size_bytes
                profiling_row = profiling_df[
                    profiling_df["block_size_bytes"] >= tensor_size_bytes
                ]
                if not profiling_row.empty:
                    profiling_row = profiling_row.iloc[0]
                else:
                    # Use the largest available block size
                    profiling_row = profiling_df.iloc[-1]

                # create_overhead = profiling_row["create_overhead_microseconds"]
                # Here we only consider the fetch overhead, as the create overhead is negligible.
                fetch_overhead = profiling_row["fetch_overhead_microseconds"]
                transfer_latency = fetch_overhead / 1e6  # convert to seconds

                tensor_transfer_latency += transfer_latency

        # Calculate loading latency
        loading_latency = 0
        if model_name not in ignore_model_names and (
            loaded_models is not None and model_name not in loaded_models
        ):
            loading_latency = get_model_host_mem_to_gpu_time(model_config)
            if loading_latency == -1:
                loading_latency = float("inf")
                logger.warning(f"No loading time found for model {model_name}")

        # Calculate execution latency
        execution_latency = 0
        if model_name not in ignore_model_names:
            adjusted_batch_size = next_power_of_2(batch_size)
            execution_latency = get_model_execution_time_for_batch_size(
                model_config, batch_size=adjusted_batch_size, mode=workflow_node.mode
            )
            if execution_latency == -1:
                execution_latency = float("inf")
                logger.warning(
                    f"No execution time found for model {model_name} and batch size {adjusted_batch_size}"
                )

        total_latency = tensor_transfer_latency + loading_latency + execution_latency
        logger.debug(
            f"Calculated latency for {model_name} at worker {dst_worker_rank}: tensor_transfer_latency={tensor_transfer_latency:.4f} seconds, loading_latency={loading_latency:.4f} seconds, execution_latency={execution_latency:.4f} seconds, total_latency={total_latency:.4f} seconds"
        )

        return total_latency

    async def check_worker_availability(self, task: Task) -> bool:
        """Check if any worker is available for dynamic scheduling."""
        async with self.worker_status_lock:
            for worker_rank, status in self.worker_status.items():
                queue_latency = sum(status["estimated_latency"].values())
                if queue_latency <= self.worker_latency_threshold:
                    return True
            return False

    async def select_worker_for_task_group(
        self,
        task_group: List[Task],
    ) -> int:
        """Select a worker for dynamic scheduling based on latency estimation."""
        async with self.worker_status_lock:
            selected_worker_rank = None
            min_overall_latency = float("inf")
            execution_latency_on_selected_worker = float("inf")
            is_scheduler_op = isinstance(task_group[0].workflow_node.op, BaseScheduler)

            worker_ranks = list(self.all_workers_info.keys())
            # random.shuffle(worker_ranks)
            for worker_rank in worker_ranks:
                status = self.worker_status[worker_rank]
                models = status["active_models"]
                queue_latency = sum(status["estimated_latency"].values())
                if queue_latency > self.worker_latency_threshold:
                    # This worker is too busy, skip it.
                    continue

                execution_latency = self._calc_latency(task_group, worker_rank, models)

                estimated_overall_latency = queue_latency + execution_latency
                if estimated_overall_latency < min_overall_latency:
                    min_overall_latency = estimated_overall_latency
                    execution_latency_on_selected_worker = execution_latency
                    selected_worker_rank = worker_rank
                    # break

            if selected_worker_rank is None:
                # No worker is available, select a random worker.
                logger.warning("No worker is available, selecting a random worker.")
                selected_worker_rank = random.choice(list(self.all_workers_info.keys()))

            # For scheduler operations, we need to calculate the execution latency on the selected worker.
            if is_scheduler_op:
                execution_latency_on_selected_worker = self._calc_latency(
                    task_group, selected_worker_rank
                )

            task_id = get_task_id(task_group)
            logger.debug(
                f"Adding task {task_id} to estimated latency of worker {selected_worker_rank}"
            )
            logger.debug(
                f"Estimated latency of worker {selected_worker_rank}: {self.worker_status[selected_worker_rank]['estimated_latency']}"
            )
            self.worker_status[selected_worker_rank]["estimated_latency"][
                task_id
            ] = execution_latency_on_selected_worker

            return selected_worker_rank

    async def update_custom_worker_status_after_completion(
        self,
        worker_rank: int,
        task_id: str,
    ):
        """Update worker status after task completion for dynamic scheduling."""
        async with self.worker_status_lock:
            status = self.worker_status[worker_rank]
            logger.debug(
                f"Removing task {task_id} from estimated latency of worker {worker_rank}"
            )
            logger.debug(
                f"Estimated latency of worker {worker_rank}: {status['estimated_latency']}"
            )
            if task_id in status["estimated_latency"]:
                del status["estimated_latency"][task_id]

    async def cleanup_request(self, request_id: str):
        """Clean up request-specific data for dynamic scheduling."""
        # No request-specific cleanup needed for dynamic scheduling
        pass

    def can_group_tasks(self, task1: Task, task2: Task) -> bool:
        """Check if two tasks can be grouped together for dynamic scheduling."""
        if task1.workflow_node.op.id != task2.workflow_node.op.id:
            return False

        if task1.workflow_node.mode != task2.workflow_node.mode:
            return False

        # Check if patches are the same
        if not have_same_patches(task1.workflow_node.op, task2.workflow_node.op):
            return False

        if task1.workflow_node.op.id == "IndexedTensor":
            if task1.request_id != task2.request_id:
                return False

            if task1.input_map["tensor"] != task2.input_map["tensor"]:
                return False

            return True

        # If one of the tasks has lazy inputs, they cannot be grouped together.
        if len(task1.lazy_inputs) > 0 or len(task2.lazy_inputs) > 0:
            return False

        # For dynamic scheduling, we can be more flexible with grouping
        # Allow grouping across different requests for better resource utilization
        return True

    async def select_tasks_for_grouping(
        self, ready_tasks: List[Task], model_batch_configs: Dict[str, Dict[str, int]]
    ) -> List[Task]:
        """Select tasks for grouping using dynamic scheduling logic."""
        task_group = []

        # Pop the first task from ready_tasks
        first_task = ready_tasks.pop(0)
        task_group.append(first_task)
        batch_size = 1
        batch_mode = "throughput_mode"

        # Dynamic scheduling logic for batch mode selection
        num_available_workers = 0

        async with self.worker_status_lock:
            for worker_rank, status in self.worker_status.items():
                queue_latency = sum(status["estimated_latency"].values())
                if queue_latency <= self.worker_latency_threshold:
                    num_available_workers += 1

        num_required_workers_latency_mode = 0
        task_count_dict = {}
        for task in ready_tasks:
            if task.workflow_node.op.id not in task_count_dict:
                task_count_dict[task.workflow_node.op.id] = 0
            task_count_dict[task.workflow_node.op.id] += 1
        for op_id, count in task_count_dict.items():
            if op_id not in model_batch_configs:
                num_required_workers_latency_mode += count
            else:
                num_required_workers_latency_mode += math.ceil(
                    count / model_batch_configs[op_id]["latency_mode"]
                )

        if num_required_workers_latency_mode <= num_available_workers:
            batch_mode = "latency_mode"

        logger.debug(f"batch_mode for {first_task.workflow_node.name}: {batch_mode}")

        if first_task.workflow_node.op.id in model_batch_configs:
            batch_size = model_batch_configs[first_task.workflow_node.op.id][batch_mode]
        elif first_task.workflow_node.op.id == "IndexedTensor":
            batch_size = float("inf")

        logger.debug(f"batch_size for {first_task.workflow_node.name}: {batch_size}")

        # Iterate over a copy of ready_tasks to allow removal
        i = 0
        while len(task_group) < batch_size and i < len(ready_tasks):
            candidate_task = ready_tasks[i]
            if self.can_group_tasks(task_group[0], candidate_task):
                task_group.append(ready_tasks.pop(i))
                # Do not increment i, as the list has shifted
            else:
                i += 1

        return task_group
