#!/usr/bin/env bash
set -euo pipefail

# Experiment 1
# Run from any directory: this moves to the repository root first.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SERVER_READY_TIMEOUT=300

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <baseline_name>"
    echo "Example: $0 shepherd"
    exit 1
fi

BASELINE_NAME="$1"
BASELINE_CONFIG="./baselines/baseline_configs/${BASELINE_NAME}.yml"
HOSTFILE="worker_hostfile"
NUM_WORKERS=$(awk 'NF > 0 { count++ } END { print count + 0 }' "${HOSTFILE}")
BASE_PORT=12500
SERVER_PORT=7777
SERVER_URL="http://localhost:${SERVER_PORT}"

WORKFLOWS="flux_schnell"
SLO_SCALE=2.0
LOG_DIR="${BASELINE_NAME}"
SERVER_LOG_DIR="logs/run_baseline"
SERVER_LOG="${SERVER_LOG_DIR}/${BASELINE_NAME}_server.log"

# Edit this list to add/remove request rates.
REQUEST_RATES=(0.5 1.0 2.0 3.0)

show_recent_log() {
    local log_file="$1"
    local lines="$2"

    tail -n "$lines" "$log_file" >&2 || true
}

wait_for_server() {
    echo "Waiting for server readiness..."
    local deadline=$((SECONDS + SERVER_READY_TIMEOUT))

    while ! grep -q "Uvicorn running on" "$SERVER_LOG" 2>/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server exited before becoming ready. Last log lines:" >&2
            show_recent_log "$SERVER_LOG" 120
            exit 1
        fi

        if (( SECONDS >= deadline )); then
            echo "Timed out waiting for server readiness after ${SERVER_READY_TIMEOUT}s." >&2
            echo "Last server log lines:" >&2
            show_recent_log "$SERVER_LOG" 120
            exit 1
        fi

        sleep 1
    done

    echo "Server is ready."
}

cleanup() {
    echo "Stopping server and worker..."
    kill "${SERVER_PID:-}" "${WORKER_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting ${BASELINE_NAME} worker..."
mpirun -n "${NUM_WORKERS}" --bind-to core --map-by slot:pe=12 \
    --hostfile "${HOSTFILE}" \
    python3 ./baselines/worker.py \
    --base-port "${BASE_PORT}" \
    --baseline-config "${BASELINE_CONFIG}" &
WORKER_PID=$!

echo "Waiting for worker to start..."
sleep 10

mkdir -p "$SERVER_LOG_DIR"
echo "Starting ${BASELINE_NAME} server; log: $SERVER_LOG"
python3 ./baselines/server.py \
    --hostfile "${HOSTFILE}" \
    --port "${SERVER_PORT}" \
    --base-port "${BASE_PORT}" \
    --baseline-config "${BASELINE_CONFIG}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server

echo "Running warmup..."
PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" python3 ae-scripts/benchmark_client.py \
    --server-url "${SERVER_URL}" \
    --trace-file "ae-scripts/traces/generated_trace_n3_rs0.5_st50000_t120_seed666.txt" \
    --workflows "${WORKFLOWS}" \
    --slo-scale "${SLO_SCALE}" \
    --log-dir "${LOG_DIR}" \
    --baseline

for rate in "${REQUEST_RATES[@]}"; do
    echo "Running request rate ${rate}..."
    PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" python3 ae-scripts/benchmark_client.py \
        --server-url "${SERVER_URL}" \
        --trace-file "ae-scripts/traces/generated_trace_n3_rs${rate}_st50000_t120_seed666.txt" \
        --workflows "${WORKFLOWS}" \
        --slo-scale "${SLO_SCALE}" \
        --log-dir "${LOG_DIR}" \
        --baseline
done

sleep 10
echo "Experiment finished."