import argparse
import base64
import io
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, Sequence

import yaml
import torch
import zmq

from mpi4py import MPI
from PIL import Image

from baseline_utils import convert_pipeline_name_to_instance

import logging
logging.getLogger("diffusers.pipelines").setLevel(logging.ERROR)

# a rough estimate of the GPU memory size for each pipeline
# without CFG
PIPELINE_GPU_MEMORY_SIZE = {
    "sdxl_txt2img_workflow": 17,
    # w/o CFG: 19, w/ CFG: 20
    "sdxl_txt2img_controlnet_canny_workflow": 20,
    # w/o CFG: 19, w/ CFG: 20
    "sdxl_txt2img_controlnet_depth_workflow": 20,
    # sd1.5
    "sd15_txt2img_workflow": 5,
    "sd15_txt2img_controlnet_canny_workflow": 6,
    "sd15_txt2img_controlnet_depth_workflow": 6,
    # sd3-medium
    "sd3_txt2img_workflow": 23,
    # w/o CFG: 23, w/ CFG: 25
    "sd3_txt2img_controlnet_canny_workflow": 25,
    # w/o CFG: 23, w/ CFG: 25
    "sd3_txt2img_controlnet_pose_workflow": 25,
    # sd3.5-large
    "sd35_large_txt2img_workflow": 35,
    "sd35_large_txt2img_controlnet_depth_workflow": 40,
    "sd35_large_txt2img_controlnet_canny_workflow": 40,
    # flux dev
    "flux_txt2img_workflow": 38,
    "flux_txt2img_controlnet_canny_workflow": 40,
    "flux_txt2img_controlnet_depth_workflow": 40,
    # flux schnell
    "flux_schnell_txt2img_workflow": 38,
    "flux_schnell_txt2img_controlnet_canny_workflow": 40,
    "flux_schnell_txt2img_controlnet_depth_workflow": 40,
}

PIPELINE_GPU_MEMORY_SIZE_CFG = {
    "sdxl_txt2img_controlnet_canny_cfg_workflow": 20,
    "sdxl_txt2img_controlnet_depth_cfg_workflow": 20,
    "sd3_txt2img_controlnet_canny_cfg_workflow": 25,
    "sd3_txt2img_controlnet_pose_cfg_workflow": 25,
}

class DistributedWorker:
    IMAGE_RESULT_DIR = "image_results"

    def __init__(
        self,
        local_rank: int,
        global_rank: int,
        hostname: str,
        baseline_config: Dict[str, Any],
        base_port: int = 14000,
    ):
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.hostname = hostname
        self.base_port = base_port

        # ZMQ setup
        self.context = zmq.Context()
        self.task_socket = self.context.socket(zmq.PULL)
        self.result_socket = self.context.socket(zmq.PUSH)

        # Setup logging
        self._setup_logging()

        # Ports for this worker
        self.task_port = self._get_task_port()
        self.result_port = self._get_result_port()


        assert torch.cuda.is_available()
        self.device = f"cuda:{self.local_rank}"
        self.max_gpu_memory_fraction = 0.90

        # Load pipeline plan config and assign pipelines to workers
        self.running_pipelines = {}
        self.pipeline_reference_timestamp = {}

        # Get actual GPU memory size from the device
        self.gpu_memory_size = self._get_gpu_memory_size() * self.max_gpu_memory_fraction

        self.baseline_config = baseline_config
        self.baseline_name = baseline_config["baseline_name"]

        self.logger.info(f"baseline_config: {self.baseline_config}")
        for pipeline_name, pipeline_placements in self.baseline_config["pipelines"].items():
            for cur_pipeline_placement in pipeline_placements:
                if cur_pipeline_placement["rank"] == self.global_rank:
                    self.logger.info(f"pipeline_name: {pipeline_name}, gpu_rank: {cur_pipeline_placement['rank']}")
                    self.running_pipelines[pipeline_name] = convert_pipeline_name_to_instance(pipeline_name, device=self.device)

                    # on demand loading, move to host memory
                    if self.baseline_name == "clockwork":
                        self.pipeline_reference_timestamp[pipeline_name] = time.time()
                        self.running_pipelines[pipeline_name] = self.running_pipelines[pipeline_name].to("cpu")

        # cache some pipelines to the GPU
        if self.baseline_name == "clockwork":
            cur_gpu_memory_usage = 0
            for pipeline_name in self.running_pipelines:
                if cur_gpu_memory_usage + PIPELINE_GPU_MEMORY_SIZE[pipeline_name] > self.gpu_memory_size:
                    self.logger.info(f"Not enough memory, skipping loading pipeline {pipeline_name}")
                    continue
                self.running_pipelines[pipeline_name] = self.running_pipelines[pipeline_name].to(self.device)
                cur_gpu_memory_usage += PIPELINE_GPU_MEMORY_SIZE[pipeline_name]

        # pipeline_name = "sd3_txt2img_workflow"
        # self.running_pipelines[pipeline_name] = convert_pipeline_name_to_instance(pipeline_name, device=self.device)

        self.running = True

    def _setup_logging(self):
        """Setup logging for this worker"""
        log_dir = os.environ.get("DIFFUSIONFLOW_LOG_DIR", "logs")

        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Create a unique log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{log_dir}/worker_{self.global_rank}_{timestamp}.log"

        log_level_str = os.environ.get("LOGLEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)

        # Setup the logger
        self.logger = logging.getLogger(f"worker_{self.global_rank}")
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

        self.logger.info(
            f"Initialized logging for worker {self.global_rank}, log file: {log_file}"
        )

    def _get_task_port(self) -> int:
        """Calculate unique task port for this worker"""
        return self.base_port + self.global_rank * 2

    def _get_result_port(self) -> int:
        """Calculate unique result port for this worker"""
        return self._get_task_port() + 1

    def setup(self):
        # Setup ZMQ connection
        self._setup_zmq_connection()

        # Set GPU device for this worker
        torch.cuda.set_device(self.local_rank)

    def _setup_zmq_connection(self):
        self.logger.info(
            f"Worker {self.global_rank} => Task Port: {self.task_port}, Result Port: {self.result_port}"
        )

        self.task_socket.bind(f"tcp://*:{self.task_port}")
        self.result_socket.bind(f"tcp://*:{self.result_port}")

    def _deserialize_image(self, image_b64: str) -> Image.Image:
        try:
            img_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
            return image
        except Exception as e:
            raise ValueError(
                f"Failed to decode image for input image: {str(e)}"
            )

    def _process_task(self, message: Dict[str, Any]):
        if self.logger.level == logging.DEBUG:
            torch.cuda.synchronize(self.local_rank)
            start_time = time.time()

        #### Execute the workflow ####
        ### Find the pipeline and load it
        pipeline_name = message["data"]["pipeline_name"]

        if self.baseline_name == "basic" or self.baseline_name == "shepherd":
            pipeline = self.running_pipelines[pipeline_name]
        elif self.baseline_name == "clockwork":
            # if the pipeline is not on the GPU
            if str(self.running_pipelines[pipeline_name].device) != self.device:
                pipelines_on_gpu = [ item for item in self.running_pipelines if str(self.running_pipelines[item].device) == self.device ]
                cur_gpu_memory_usage = sum([ PIPELINE_GPU_MEMORY_SIZE[item] for item in pipelines_on_gpu ])
                # if the pipeline is not on the GPU, and there is not enough memory, offload the least recently used pipeline to host memory
                while cur_gpu_memory_usage + PIPELINE_GPU_MEMORY_SIZE[pipeline_name] > self.gpu_memory_size:
                    self.logger.debug(f"Not enough memory, offloading the least recently used pipeline to host memory")
                    last_reference_timestamp = float('inf')
                    pipeline_name_to_offload = None
                    for item in pipelines_on_gpu:
                        if self.pipeline_reference_timestamp[item] < last_reference_timestamp:
                            last_reference_timestamp = self.pipeline_reference_timestamp[item]
                            pipeline_name_to_offload = item
                    assert pipeline_name_to_offload is not None, f"pipeline_name_to_offload is None"

                    clockwork_offloading_start = time.time()
                    self.logger.debug(f"Offloading pipeline {pipeline_name_to_offload} to host memory")
                    self.running_pipelines[pipeline_name_to_offload] = self.running_pipelines[pipeline_name_to_offload].to("cpu")
                    torch.cuda.empty_cache()
                    pipelines_on_gpu.remove(pipeline_name_to_offload)
                    cur_gpu_memory_usage -= PIPELINE_GPU_MEMORY_SIZE[pipeline_name_to_offload]
                    clockwork_offloading_end = time.time()
                    # torch.cuda.empty_cache()
                    self.logger.info(f"Clockwork offloading latency: {clockwork_offloading_end - clockwork_offloading_start}")

                # load the pipeline to the GPU
                clockwork_loading_start = time.time()
                try:
                    self.running_pipelines[pipeline_name] = self.running_pipelines[pipeline_name].to(self.device)
                    clockwork_loading_end = time.time()
                except Exception as e:
                    self.logger.error(f"Error loading pipeline {pipeline_name} to GPU: {e}. cur_gpu_memory_usage: {cur_gpu_memory_usage}, needed: {PIPELINE_GPU_MEMORY_SIZE[pipeline_name]}")
                    torch.cuda.empty_cache()
                    raise e
                self.logger.info(f"Clockwork loading latency: {clockwork_loading_end - clockwork_loading_start}")

            self.pipeline_reference_timestamp[pipeline_name] = time.time()
            pipeline = self.running_pipelines[pipeline_name]
        else:
            raise ValueError(f"Invalid scheduling baseline: {self.baseline_name}")

        # pipeline = self.running_pipelines[pipeline_name]

        ### Pasre the inputs
        # negative prompt: flux, flux_controlnet, sd3, sd3_controlnet, sd15, sd15_controlnet, sdxl, sdxl_controlnet
        # height, width: flux, flux_controlnet, sd3, sd3_controlnet, sd15, sd15_controlnet, sdxl, sdxl_controlnet
        # true_cfg_scale: flux, flux_controlnet
        # controlnet_conditioning_scale: flux_controlnet, sd3_controlnet, sd15_controlnet, sdxl_controlnet
        # image: sdxl_controlnet

        inputs = message["data"]["inputs"]
        prompt = inputs["prompt"]
        num_inference_steps = inputs["num_inference_steps"]
        guidance_scale = inputs["guidance_scale"]
        height = inputs["height"]
        width = inputs["width"]
        sd_generator = torch.manual_seed(inputs["seed"])

        negative_prompt = inputs.get("negative_prompt", None)

        # process image inputs
        control_image = inputs.get("control_image", None) # for controlnet pipelines other than sd15
        if control_image is not None:
            control_image = self._deserialize_image(control_image)
        # image = inputs.get("image", None) # for sd15 controlnet pipelines
        # if image is not None:
        #     image = self._deserialize_image(image)
        conditioning_scale = inputs.get("conditioning_scale", None)  # for controlnet pipelines

        true_cfg_scale = inputs.get("cfg_guidance_scale", None)  # for flux pipelines

        #### Run the pipeline ####
        inference_start_time = time.time()
        self.logger.info(f"Running pipeline {pipeline_name} with for request_id: {message['data']['request_id']}")
        pipeline_inputs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "generator": sd_generator,
        }
        if true_cfg_scale is not None:
            pipeline_inputs["true_cfg_scale"] = true_cfg_scale
        if control_image is not None:
            if "flux" in pipeline_name or "sd3" in pipeline_name:
                pipeline_inputs["control_image"] = control_image
            if "sdxl" in pipeline_name or "sd15" in pipeline_name:
                pipeline_inputs["image"] = control_image
        if conditioning_scale is not None:
            pipeline_inputs["controlnet_conditioning_scale"] = conditioning_scale
        images = pipeline(**pipeline_inputs).images
        assert isinstance(images, list) and all(isinstance(img, Image.Image) for img in images), f"image is not an instance of Image.Image"
        #### Execute the workflow ####

        #### Serialize the images ####
        img_str_list = []
        for image in images:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            img_str_list.append(img_str)
        inference_end_time = time.time()
        inference_latency = inference_end_time - inference_start_time
        #### Serialize the images ####

        self.logger.info(f"Sending result for request_id: {message['data']['request_id']}")

        response = {
            "type": "completed",
            "data": {
                "request_id": message["data"]["request_id"],
                "img_str_list": img_str_list,
                "inference_latency": inference_latency,
            }
        }

        self.result_socket.send_json(response)

        # # on demand loading, offload to host memory after inference
        # if self.baseline_name == "clockwork":
        #     self.running_pipelines[pipeline_name] = pipeline.to("cpu")
            
    def run(self):
        """Main worker loop"""
        self.setup()
        self.logger.info(
            f"[DistributedWorker] Initialized => Hostname: {self.hostname}, "
            f"Local Rank: {self.local_rank}, Global Rank: {self.global_rank}"
        )

        poller = zmq.Poller()
        poller.register(self.task_socket, zmq.POLLIN)

        while True:
            try:
                # Use poller with timeout instead of blocking recv
                socks = dict(poller.poll(timeout=1000))  # 1 second timeout
                if self.task_socket not in socks:
                    continue

                message = self.task_socket.recv_json()
                self.logger.debug(f"Received message: {message.get('type')}")

                if message.get("type") == "stop":  # Poison pill
                    self.logger.info("Received stop signal, initiating shutdown...")
                    break
                elif message.get("type") == "ping":  # Health check
                    self.result_socket.send_json({"type": "pong"})
                    continue
                elif message.get("type") == "clear_cache":
                    # TODO @ Lingyun: Clear the cache of tensor_map
                    self.result_socket.send_json({"type": "cache_cleared"})
                    continue
                elif message.get("type") == "task":
                    self._process_task(message)
                else:
                    self.logger.warning(
                        f"Unknown message type: {message.get('type')}"
                    )

            except Exception as e:
                traceback.print_exc()
                self.logger.error(f"Error in worker loop: {e}")
                continue

        self.cleanup()

    def _get_gpu_memory_size(self) -> int:
        """Get GPU memory size in GB for the current device"""
        if not torch.cuda.is_available():
            return 0

        device_idx = int(self.device.split(":")[-1])
        torch.cuda.synchronize(device_idx)

        # Get total memory in bytes and convert to GB
        total_memory_bytes = torch.cuda.get_device_properties(device_idx).total_memory
        total_memory_gb = total_memory_bytes / (1024**3)  # Convert bytes to GB

        # Apply the memory fraction to get usable memory
        usable_memory_gb = total_memory_gb * self.max_gpu_memory_fraction

        self.logger.info(f"GPU {device_idx} total memory: {total_memory_gb:.2f} GB, usable: {usable_memory_gb:.2f} GB")

        return int(usable_memory_gb)

    def _get_gpu_memory_info(self, device: str) -> Dict[str, Any]:
        """Get GPU memory information in bytes"""
        if not torch.cuda.is_available():
            return {"free": 0, "total": 0, "used": 0}

        device_idx = int(device.split(":")[-1])
        torch.cuda.synchronize(device_idx)

        total_memory = torch.cuda.get_device_properties(device_idx).total_memory
        reserved_memory = torch.cuda.memory_reserved(device_idx)
        allocated_memory = torch.cuda.memory_allocated(device_idx)
        free_memory = total_memory - reserved_memory

        return {"free": free_memory, "total": total_memory, "used": allocated_memory}

    def _load_model(self, model_id: str, model_path: Optional[str] = None):
        """Load a model and cache it"""
        if model_id in self.models:
            return

        model = get_op(model_id)
        model_components = None
        if model_path is not None:
            model_components = model.initialize(model_path, self.device)

        self.logger.debug(f"Loading model: {model_id}, model_path: {model_path}, device: {self.device}")
        self.models[model_id] = {
            "model": model,
            "model_components": model_components,
        }

    def _unload_model(self, model_id: str):
        """Unload a model and free its memory"""
        if model_id not in self.models:
            return

        self.logger.info(f"Unloading model: {model_id}")

        del self.models[model_id]
        torch.cuda.empty_cache()

    def cleanup(self):
        try:
            if self.running:  # Add check to prevent double cleanup
                self.running = False
                self.logger.info(f"Worker {self.global_rank} is stopping")

                # """Unload all models and free their memory"""
                # for model_id in list(self.models.keys()):
                #     self._unload_model(model_id)
                del self.running_pipelines

                # Cleanup ZMQ resources
                self.task_socket.setsockopt(zmq.LINGER, 1000)  # 1 second timeout
                self.result_socket.setsockopt(zmq.LINGER, 1000)

                self.task_socket.close()
                self.result_socket.close()
                self.context.term()

                self.logger.info("Cleanup completed successfully")
                # Close all handlers
                for handler in self.logger.handlers[:]:
                    handler.close()
                    self.logger.removeHandler(handler)
        except Exception as e:
            self.logger.error(f"Error during worker cleanup: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", type=str, default="./baseline_configs/basic.yml")
    parser.add_argument("--base-port", type=int, default=14000)
    args = parser.parse_args()

    ### parse baseline config ###
    with open(args.baseline_config, "r") as f:
        baseline_config = yaml.safe_load(f)

    global_rank = MPI.COMM_WORLD.Get_rank()
    global_size = MPI.COMM_WORLD.Get_size()
    local_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local_comm.Get_rank()
    local_size = local_comm.Get_size()
    hostname = MPI.Get_processor_name()
    print(
        f"Worker {global_rank}/{global_size} (local rank {local_rank}/{local_size}) starting on {hostname}..."
    )

    worker = DistributedWorker(
        local_rank=local_rank,
        global_rank=global_rank,
        hostname=hostname,
        baseline_config=baseline_config,
        base_port=args.base_port,
    )
    print(f"Worker {global_rank} initialized")
    worker.run()