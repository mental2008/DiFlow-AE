import argparse
from typing import List

from benchmark_loading.benchmark_utils import DEFAULT_RESULTS_DIR, run_loading_benchmark
from benchmark_loading.registry import (
    list_cases,
    list_suites,
    load_builtin_suites,
    select_cases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run loading-only model benchmarks.")
    parser.add_argument("--suite", help="Benchmark suite to run, e.g. flux1.")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model ID to run. Can be provided multiple times.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all registered loading benchmark cases.",
    )
    parser.add_argument("--list", action="store_true", help="List available cases.")
    parser.add_argument("--device", default="cuda", help="Target device for loading.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations.")
    parser.add_argument("--repeats", type=int, default=5, help="Measured iterations.")
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Root directory for loading benchmark results.",
    )
    parser.add_argument(
        "--force-benchmark",
        action="store_true",
        help="Re-run even if a matching current-GPU result exists.",
    )
    return parser.parse_args()


def _print_cases() -> None:
    print("Available loading benchmark suites:")
    for suite_name in list_suites():
        print(f"  {suite_name}")

    print("\nAvailable loading benchmark cases:")
    for case in list_cases():
        print(f"  {case.suite}: {case.model_name} ({case.model_path})")


def _validate_selection(args: argparse.Namespace) -> List:
    if args.list:
        return []

    if not args.all and args.suite is None and not args.models:
        raise SystemExit("Please pass --suite, --model, --all, or --list.")

    cases = select_cases(suite_name=args.suite, model_names=args.models)
    if not cases:
        raise SystemExit("No loading benchmark cases matched the selection.")
    return cases


def main() -> None:
    args = parse_args()
    load_builtin_suites()

    if args.list:
        _print_cases()
        return

    for case in _validate_selection(args):
        print(f"=== Loading benchmark: {case.model_name} ({case.suite}) ===")
        run_loading_benchmark(
            case=case,
            device=args.device,
            warmup=args.warmup,
            repeats=args.repeats,
            force_benchmark=args.force_benchmark,
            results_dir=args.results_dir,
        )


if __name__ == "__main__":
    main()
