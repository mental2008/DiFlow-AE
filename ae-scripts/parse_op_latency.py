import argparse
import glob
import os
import re
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional
import numpy as np
import json

TIME_TAKEN_RE = re.compile(
    r"Time taken to (?P<op>.+?):\s*(?P<latency>[0-9]*\.?[0-9]+)\s*seconds"
)


def _extract_operator_name(op_field: str) -> Optional[str]:
    """
    Given the text that appears between "Time taken to" and the latency,
    extract a short operator name.

    Examples
    --------
    - "process task group (StableDiffusion3_8c8a4630-..., req_id), (...)" -> "StableDiffusion3"
    - "execute model (StableDiffusion3)" -> "StableDiffusion3"
    """
    # We rely on the first parenthesized group.
    if "(" not in op_field or ")" not in op_field:
        return None

    inner = op_field.split("(", 1)[1].split(")", 1)[0]
    # For "StableDiffusion3_8c8a4630-..., req_id" keep only the first item.
    first_item = inner.split(",", 1)[0].strip()
    if not first_item:
        return None

    # Strip any uuid / suffix after the first underscore.
    base = first_item.split("_", 1)[0].strip()
    return base or None


def parse_worker_logs(
    log_dir: str,
    *,
    op_substring: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Parse worker logs in `log_dir` and return a dict mapping
    operator/operation name -> list of latencies in seconds.

    By default, it picks up every line containing:

        Time taken to <op>: <latency> seconds

    If `op_substring` is provided (e.g. "process"), only operations whose
    name contains that substring will be kept.
    """
    op_latencies: DefaultDict[str, List[float]] = defaultdict(list)

    # Match all worker log files like worker_0_*.log, worker_1_*.log, etc.
    pattern = os.path.join(log_dir, "worker_*.log")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, "r") as fr:
                for line in fr:
                    if "Time taken to process " not in line:
                        continue
                    m = TIME_TAKEN_RE.search(line)
                    if not m:
                        continue

                    op_field = m.group("op").strip()
                    op = _extract_operator_name(op_field)
                    if op is None:
                        continue

                    if op_substring is not None and op_substring not in op:
                        continue

                    latency = float(m.group("latency"))
                    op_latencies[op].append(latency)
        except OSError:
            # Skip unreadable files
            continue

    # record median
    op_latencies_median = {}
    for op, vals in op_latencies.items():
        op_latencies_median[op] = np.median(vals)

    return op_latencies_median


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse DiffusionFlow worker logs and extract operator latencies "
            "from 'Time taken to ...' lines."
        )
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory containing worker_*.log files.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="If set, print a simple summary (count, min, max, avg) per op.",
    )

    args = parser.parse_args()

    op_latencies = parse_worker_logs(args.log_dir)

    if args.print_summary:
        from statistics import mean

        for op, vals in sorted(op_latencies.items()):
            if not vals:
                continue
            print(
                f"{op!r}: count={len(vals)}, "
                f"min={min(vals):.6f}s, max={max(vals):.6f}s, avg={mean(vals):.6f}s"
            )
    else:
        # Just print the raw dict; you can import and use programmatically instead.
        # This simple repr is often good enough for quick inspection.
        print(op_latencies)
    
    save_dir = "./configs/"
    with open(os.path.join(save_dir, "op_latencies_median.json"), "w") as fw:
        json.dump(op_latencies, fw)


if __name__ == "__main__":
    main()