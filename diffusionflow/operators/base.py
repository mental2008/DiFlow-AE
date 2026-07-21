from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch

from diffusionflow.interface.node_io import NodeIO
from diffusionflow.operators.utils import Config

if TYPE_CHECKING:
    from diffusionflow.operators.models.patches.base_patch import BasePatch


class Operator(ABC):
    def __init__(self, config: Config = None):
        self.config = config
        self._inputs: Dict[str, NodeIO] = {}
        self._outputs: Dict[str, NodeIO] = {}
        self._patches: List["BasePatch"] = []
        self.setup_io()

    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    def setup_io(self):
        """Define input and output specifications"""
        pass

    def add_input(
        self,
        name: str,
        data_type: type,
        size: Optional[list[int]] = None,
        lazy: bool = False,
    ):
        self._inputs[name] = NodeIO(
            name=name, data_type=data_type, size=size, lazy=lazy
        )

    def add_output(
        self,
        name: str,
        data_type: type,
        size: Optional[list[int]] = None,
        lazy: bool = False,
    ):
        self._outputs[name] = NodeIO(
            name=name, data_type=data_type, size=size, lazy=lazy
        )

    def add_patch(self, patch: "BasePatch"):
        if patch in self._patches:
            raise ValueError(f"Patch {patch.id} already added to operator {self.id}.")
        self._patches.append(patch)

    def get_inputs(self) -> Dict[str, NodeIO]:
        return self._inputs

    def get_outputs(self) -> Dict[str, NodeIO]:
        return self._outputs

    def get_patches(self) -> List["BasePatch"]:
        return self._patches

    def add_execution_mode(
        self, mode: str, inputs: Dict[str, type], outputs: Dict[str, type]
    ):
        if not hasattr(self, "_execution_modes"):
            self._execution_modes = {}

        mode_spec = {
            "inputs": {name: NodeIO(name, dtype) for name, dtype in inputs.items()},
            "outputs": {name: NodeIO(name, dtype) for name, dtype in outputs.items()},
        }
        self._execution_modes[mode] = mode_spec

    def get_execution_modes(self) -> Dict[str, Dict[str, Dict[str, NodeIO]]]:
        """Get all available execution modes and their IO specifications"""
        return getattr(self, "_execution_modes", {})

    def initialize(
        self, model_path: str, device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        """Default implementation for models that don't require initialization"""
        return {}

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        pass

    def __call__(self, **inputs):
        from diffusionflow.interface.workflow_context import WorkflowContext
        from diffusionflow.interface.workflow_node import WorkflowNode

        workflow = WorkflowContext.get_current_workflow()
        if workflow is None:
            raise RuntimeError("No active workflow context")

        # Remove None values from inputs
        inputs = {k: v for k, v in inputs.items() if v is not None}

        mode = inputs.pop("mode", "default")

        workflow_node = WorkflowNode(op=self, inputs=inputs, mode=mode)

        workflow.add_workflow_node(workflow_node)

        output_list = list(workflow_node.get_outputs().values())
        if len(output_list) == 1:
            return output_list[0]
        return output_list

    def to_dict(self) -> Dict[str, Any]:
        """Serialize operator to dictionary with base_model and patches structure."""
        return {
            "base_model": {
                "model_id": self.id,
                "model_path": (
                    self.config.model_path if self.config is not None else None
                ),
            },
            "patches": [
                {
                    "model_id": patch.id,
                    "model_path": (
                        patch.config.model_path if patch.config is not None else None
                    ),
                }
                for patch in self._patches
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Operator":
        """Reconstruct operator from dictionary with base_model and patches structure."""
        from diffusionflow.operators.utils import get_op

        base_model = data["base_model"]
        model_id = base_model["model_id"]
        model_path = base_model.get("model_path")

        # Create the base operator
        op = get_op(model_id, model_path)

        # Add patches to the operator if they exist
        patches = data.get("patches", [])
        for patch_data in patches:
            patch_id = patch_data["model_id"]
            patch_path = patch_data.get("model_path")
            patch_op = get_op(patch_id, patch_path)
            op.add_patch(patch_op)

        return op
