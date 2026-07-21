import os

from benchmark_loading.benchmark_utils import load_cases_from_yaml
from benchmark_loading.registry import register_suite


CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "configs", "flux1.yaml"
)

register_suite("flux1", load_cases_from_yaml(CONFIG_PATH))
