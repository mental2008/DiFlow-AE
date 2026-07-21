from typing import Any, Dict, Union

import torch
import numpy as np

from diffusionflow.operators.base import Operator


class BALTrigger(Operator):
    def setup_io(self):
        # self.add_input("start_loading_flag_shm", str)
        self.add_output("bal_trigger", str)

    @property
    def id(self) -> str:
        return "BALTrigger"

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

        # Suyi: currently support only one loader
        shm_dict["start_loading_flag_np"][0] = 1

        return {"bal_trigger": "bal_trigger"}

        
        