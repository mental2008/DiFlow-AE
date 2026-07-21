import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "ae-results" / "client_logs"

EXPERIMENTS = [
    (
        "with_admission_control",
        RESULTS_DIR / "diflow-with-admission-control" / "benchmark_client_*.log",
    ),
    (
        "without_admission_control",
        RESULTS_DIR / "diflow-wo-admission-control" / "benchmark_client_*.log",
    ),
]

PATTERNS = {
    "avg_latency": r"Average latency:\s*([0-9.]+) seconds",
    "total": r"Total requests:\s*(\d+)",
    "success": r"Successful requests:\s*(\d+)",
    "rejected": r"Rejected requests:\s*(\d+)",
    "failed": r"Failed requests:\s*(\d+)",
    "slo": r"SLO Attainment:\s*([0-9.]+)%",
}


def latest_log(paths_or_patterns):
    logs = []

    for item in paths_or_patterns:
        path = Path(item)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if path.is_file():
            logs.append(path)
        else:
            logs.extend(path.parent.glob(path.name))

    if not logs:
        raise FileNotFoundError(f"No logs match: {paths_or_patterns}")

    return max(logs, key=lambda path: path.name)


def parse_log(log_path):
    text = log_path.read_text(encoding="utf-8")
    result = {"log": log_path}

    for name, pattern in PATTERNS.items():
        match = re.search(pattern, text)
        result[name] = match.group(1) if match else "N/A"

    return result


def print_result(name, result):
    print(f"{name}:")
    print(f"  SLO Attainment: {result['slo']}%")
    print(f"  Total requests: {result['total']}")
    print(f"  Successful requests: {result['success']}")
    print(f"  Rejected requests: {result['rejected']}")
    print(f"  Failed requests: {result['failed']}")
    print(f"  Log: {result['log'].relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_path = latest_log(sys.argv[1:])
        print_result("selected_log", parse_log(log_path))
    else:
        for experiment_name, log_pattern in EXPERIMENTS:
            log_path = latest_log([log_pattern])
            print_result(experiment_name, parse_log(log_path))