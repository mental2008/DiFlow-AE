import copy
import logging
import os
import time
from typing import Any, Dict, Union

import torch
from diffusers import schedulers

from diffusionflow.operators.operator_ids import (
    FLOW_MATCH_EULER_DISCRETE_SCHEDULER_ID,
)
from diffusionflow.operators.schedulers.base_scheduler import BaseScheduler

logger = logging.getLogger(__name__)


class FlowMatchEulerDiscreteScheduler(BaseScheduler):
    @property
    def id(self) -> str:
        return FLOW_MATCH_EULER_DISCRETE_SCHEDULER_ID

    def initialize(
        self, model_path: Union[str, None], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        if model_path is not None and os.path.exists(model_path):
            scheduler = schedulers.FlowMatchEulerDiscreteScheduler.from_pretrained(
                model_path, subfolder="scheduler"
            )
        else:
            dummy_config = self._default_dummy_config()
            scheduler = schedulers.FlowMatchEulerDiscreteScheduler(**dummy_config)

        return {"scheduler": scheduler}

    @staticmethod
    def _default_dummy_config() -> Dict[str, Any]:
        return {"num_train_timesteps": 1000, "shift": 3.0}


if __name__ == "__main__":
    model_path = (
        "/project/infattllm/lyangbk/huggingface/stable-diffusion-3-medium-diffusers"
    )

    init_start = time.time()
    scheduler = FlowMatchEulerDiscreteScheduler().initialize(model_path, "cpu")
    init_end = time.time()
    print(f"Time taken to initialize the scheduler: {init_end - init_start} seconds")

    for _ in range(10):
        copy_start = time.time()
        scheduler = copy.deepcopy(scheduler)
        copy_end = time.time()
        print(f"Time taken to duplicate the scheduler: {copy_end - copy_start} seconds")
