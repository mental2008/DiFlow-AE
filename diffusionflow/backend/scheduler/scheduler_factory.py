"""
Factory module for creating scheduler instances.

This module provides a factory function to create the appropriate scheduler
based on the scheduling policy.
"""

import logging
from typing import Any, Dict

from .base_scheduler import Scheduler
from .dynamic_scheduler import DynamicScheduler
from .exclusive_scheduler import ExclusiveScheduler
from .random_scheduler import RandomScheduler
from .utils import SchedulingPolicy


def create_scheduler(
    scheduling_policy: SchedulingPolicy,
    all_workers_info: Dict[int, Dict[str, Any]],
    model_configs: Dict[str, Any] = None,
) -> Scheduler:
    """
    Factory function to create the appropriate scheduler based on scheduling policy.

    Args:
        scheduling_policy: The scheduling policy to use
        all_workers_info: Information about all workers
        logger: Logger instance
        model_configs: Model configurations for dynamic scheduling

    Returns:
        Scheduler: The appropriate scheduler implementation
    """
    if scheduling_policy == SchedulingPolicy.EXCLUSIVE:
        return ExclusiveScheduler(all_workers_info)
    elif scheduling_policy == SchedulingPolicy.RANDOM:
        return RandomScheduler(all_workers_info)
    elif scheduling_policy == SchedulingPolicy.DYNAMIC:
        return DynamicScheduler(all_workers_info, model_configs)
    else:
        raise ValueError(f"Invalid scheduling policy: {scheduling_policy}")
