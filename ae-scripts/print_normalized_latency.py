import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


RESULTS_DIR = Path(__file__).resolve().parents[1] / "ae-results" / "parallel_speedup"

LOG_FILES = {
    "one_gpu": RESULTS_DIR / "flux_schnell_one_gpu.log",
    "two_gpu": RESULTS_DIR / "flux_schnell_two_gpu.log",
}

# The first run often includes startup/warmup work. Set to False to include every run.
SKIP_FIRST_THREE_RUNS = True

TIME_PATTERN = re.compile(
    r"For service (?P<workflow>[^,]+), Time taken: (?P<seconds>[0-9.eE+-]+) seconds"
)


def parse_workflow_times(log_path):
    workflow_times = defaultdict(list)

    with log_path.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            match = TIME_PATTERN.search(line)
            if not match:
                continue

            workflow = match.group("workflow")
            seconds = float(match.group("seconds"))
            workflow_times[workflow].append(seconds)

    return workflow_times


def average_latency(times):
    measured_times = times[1:] if SKIP_FIRST_THREE_RUNS and len(times) > 1 else times
    return mean(measured_times)


def collect_latencies():
    latencies = {}

    for gpu_name, log_path in LOG_FILES.items():
        workflow_times = parse_workflow_times(log_path)
        latencies[gpu_name] = {
            workflow: average_latency(times) for workflow, times in workflow_times.items()
        }

    return latencies


def print_normalized_latencies(latencies):
    workflows = sorted(latencies["one_gpu"])

    print(
        f"{'workflow':<55} "
        f"{'one_gpu_latency':>16} "
        f"{'one_gpu_norm':>12} "
        f"{'two_gpu_latency':>16} "
        f"{'two_gpu_norm':>12}"
    )

    for workflow in workflows:
        one_gpu_latency = latencies["one_gpu"][workflow]
        two_gpu_latency = latencies["two_gpu"].get(workflow)

        one_gpu_norm = one_gpu_latency / one_gpu_latency
        two_gpu_norm = (
            two_gpu_latency / one_gpu_latency if two_gpu_latency is not None else None
        )

        print(
            f"{workflow:<55} "
            f"{one_gpu_latency:>16.6f} "
            f"{one_gpu_norm:>12.6f} "
            f"{format_latency(two_gpu_latency):>16} "
            f"{format_latency(two_gpu_norm):>12}"
        )


def format_latency(value):
    if value is None:
        return "N/A"

    return f"{value:.6f}"


if __name__ == "__main__":
    print_normalized_latencies(collect_latencies())
