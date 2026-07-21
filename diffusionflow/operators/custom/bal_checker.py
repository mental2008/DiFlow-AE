from typing import Any, Dict, Union

import torch
import numpy as np

from diffusionflow.operators.base import Operator
import logging
import time

class BALChecker(Operator):
    def setup_io(self):
        # self.add_input("start_loading_flag_shm", str)
        self.add_output("is_loading_complete", bool)
        
    @property
    def id(self) -> str:
        return "BALChecker"

    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        **kwargs,
    ) -> Dict[str, Any]:
        
        if_use_bal = kwargs.get("if_use_bal", False)
        assert if_use_bal, "if_use_bal should be True"
        shm_dict = kwargs["shm_dict"]
        num_lora_model_repos = 1
        
        status = np.sum(shm_dict["start_loading_flag_np"])

        if status == num_lora_model_repos * 10:
            logging.debug("LoRA loading complete")
            return {"is_loading_complete": True}
        else:
            time.sleep(0.1)
            logging.debug("LoRA loading not complete")
            return {"is_loading_complete": False}