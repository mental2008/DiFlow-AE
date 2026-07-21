import uuid
from typing import List, Optional

import torch

from diffusionflow.interface.node_io import (
    AdapterInputs,
    DiffusionModelInputs,
    NodeIO,
    SchedulerInputs,
    SourceType,
)
from diffusionflow.operators.base import Operator
from diffusionflow.operators.models.adapters.base_adapter import BaseAdapter
from diffusionflow.operators.models.diffusion_models.base_diffusion_model import (
    BaseDiffusionModel,
)
from diffusionflow.operators.schedulers.base_scheduler import BaseScheduler
from diffusionflow.operators.utils import get_op


class DenoiseNode:
    def __init__(
        self,
        model: BaseDiffusionModel,
        scheduler: BaseScheduler,
        base_model_inputs: DiffusionModelInputs,
        scheduler_inputs: SchedulerInputs,
        denoised_latents: NodeIO = None,
        adapters: Optional[List[BaseAdapter]] = None,
        adapter_inputs: Optional[List[AdapterInputs]] = None,
        name: str = None,
    ):
        self.name = (
            f"{model.id}_{scheduler.id}_{uuid.uuid4()}" if name is None else name
        )
        self.model = model
        self.scheduler = scheduler
        self.base_model_inputs = base_model_inputs
        self.scheduler_inputs = scheduler_inputs
        self.adapters = adapters if adapters is not None else []
        self.adapter_inputs = adapter_inputs if adapter_inputs is not None else []

        if len(self.adapters) != len(self.adapter_inputs):
            raise ValueError(
                f"Number of adapters ({len(self.adapters)}) must match number of adapter inputs ({len(self.adapter_inputs)})"
            )

        self.denoised_latents = (
            NodeIO(
                name=f"{self.name}:denoised_latents",
                data_type=torch.Tensor,
                source_type=SourceType.NODE,
                source_node=self.name,
            )
            if denoised_latents is None
            else denoised_latents
        )

    def __repr__(self):
        return f"""
        DenoiseNode(
            model={self.model},
            scheduler={self.scheduler},
            base_model_inputs={self.base_model_inputs},
            scheduler_inputs={self.scheduler_inputs},
            adapters={self.adapters},
            adapter_inputs={self.adapter_inputs},
            denoised_latents={self.denoised_latents}
        )
        """

    def to_dict(self):
        return {
            "name": self.name,
            "model": self.model.to_dict(),
            "scheduler": self.scheduler.id,
            "scheduler_path": (
                self.scheduler.config.model_path
                if self.scheduler.config is not None
                else None
            ),
            "base_model_inputs": self.base_model_inputs.to_dict(),
            "scheduler_inputs": self.scheduler_inputs.to_dict(),
            "adapters": (
                [
                    {
                        "id": adapter.id,
                        "path": (
                            adapter.config.model_path
                            if adapter.config is not None
                            else None
                        ),
                    }
                    for adapter in self.adapters
                ]
                if self.adapters
                else None
            ),
            "adapter_inputs": (
                [inputs.to_dict() for inputs in self.adapter_inputs]
                if self.adapter_inputs
                else None
            ),
            "denoised_latents": self.denoised_latents.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        adapters = None
        adapter_inputs = None

        if data.get("adapters"):
            adapters = [
                get_op(adapter_data["id"], adapter_data["path"])
                for adapter_data in data["adapters"]
            ]

        if data.get("adapter_inputs"):
            adapter_inputs = [
                AdapterInputs.from_dict(inputs_data)
                for inputs_data in data["adapter_inputs"]
            ]

        return cls(
            name=data["name"],
            model=Operator.from_dict(data["model"]),
            scheduler=get_op(data["scheduler"], data["scheduler_path"]),
            base_model_inputs=DiffusionModelInputs.from_dict(data["base_model_inputs"]),
            scheduler_inputs=SchedulerInputs.from_dict(data["scheduler_inputs"]),
            denoised_latents=NodeIO.from_dict(data["denoised_latents"]),
            adapters=adapters,
            adapter_inputs=adapter_inputs,
        )
