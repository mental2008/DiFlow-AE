from typing import Dict, Iterable, List, Optional

from benchmark_loading.benchmark_utils import LoadingBenchmarkCase


_SUITES: Dict[str, List[LoadingBenchmarkCase]] = {}


def register_suite(suite_name: str, cases: Iterable[LoadingBenchmarkCase]) -> None:
    _SUITES[suite_name] = list(cases)


def load_builtin_suites() -> None:
    # Importing suite modules registers their cases.
    import benchmark_loading.suites.flux1  # noqa: F401
    import benchmark_loading.suites.stable_diffusion_3  # noqa: F401
    import benchmark_loading.suites.stable_diffusion_xl  # noqa: F401


def list_suites() -> List[str]:
    return sorted(_SUITES)


def list_cases() -> List[LoadingBenchmarkCase]:
    cases: List[LoadingBenchmarkCase] = []
    for suite_name in list_suites():
        cases.extend(_SUITES[suite_name])
    return cases


def select_cases(
    suite_name: Optional[str] = None,
    model_names: Optional[List[str]] = None,
) -> List[LoadingBenchmarkCase]:
    selected_cases = _SUITES.get(suite_name, []) if suite_name else list_cases()
    if not model_names:
        return selected_cases

    requested = set(model_names)
    return [case for case in selected_cases if case.model_name in requested]
