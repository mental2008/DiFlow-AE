from abc import abstractmethod

import torch

from diffusionflow.operators.base import Operator


class BaseDiffusionModel(Operator):
    @abstractmethod
    def id(self) -> str:
        pass

    def setup_io(self):
        self.add_input("latents", torch.Tensor)
        self.add_input("timestep", torch.Tensor)
        self.add_input("prompt_embeds", torch.Tensor)
        self.add_input("pooled_prompt_embeds", torch.Tensor)
        self.add_output("noise_pred", torch.Tensor)
