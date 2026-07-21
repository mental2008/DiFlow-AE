from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from diffusionflow.interface.workflow import Workflow


class WorkflowContext:
    _current_workflow: Optional["Workflow"] = None

    @classmethod
    def get_current_workflow(cls) -> Optional["Workflow"]:
        return cls._current_workflow

    @classmethod
    def set_current_workflow(cls, workflow: Optional["Workflow"]) -> None:
        cls._current_workflow = workflow


@contextmanager
def workflow_context(workflow: "Workflow"):
    previous_workflow = WorkflowContext.get_current_workflow()
    WorkflowContext.set_current_workflow(workflow)
    try:
        yield workflow
    finally:
        WorkflowContext.set_current_workflow(previous_workflow)
