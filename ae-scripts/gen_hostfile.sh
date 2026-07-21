#!/usr/bin/env bash

set -euo pipefail

output_file="${1:-worker_hostfile}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Error: nvidia-smi was not found; cannot detect GPU count." >&2
    exit 1
fi

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ "$num_gpus" -eq 0 ]; then
    echo "Error: no GPUs detected by nvidia-smi." >&2
    exit 1
fi

: > "$output_file"
for _ in $(seq 1 "$num_gpus"); do
    echo "localhost" >> "$output_file"
done

echo "Generated $output_file with $num_gpus localhost entries."
