from abc import abstractmethod
from typing import List

import torch

from diffusionflow.operators.base import Operator


class BaseAdapter(Operator):
    @abstractmethod
    def id(self) -> str:
        pass

    def setup_io(self):
        self.add_input("latents", torch.Tensor)
        self.add_input("timestep", torch.Tensor)
        self.add_input("prompt_embeds", torch.Tensor)
        self.add_input("pooled_prompt_embeds", torch.Tensor)
        # self.add_output("block_samples", List[torch.Tensor])
