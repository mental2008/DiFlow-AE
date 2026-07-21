#!/usr/bin/env bash
set -euo pipefail

# Experiment 1
# Run from any directory: this moves to the repository root first.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SERVER_READY_TIMEOUT=300

HOSTFILE="worker_hostfile"
NUM_WORKERS=$(awk 'NF > 0 { count++ } END { print count + 0 }' "${HOSTFILE}")
BASE_PORT=12500
SERVER_PORT=7777
SERVER_URL="http://localhost:${SERVER_PORT}"

WORKFLOWS="flux_schnell"
SLO_SCALE=2.0
LOG_DIR="diflow"

PREFETCH_MODELS_CONFIG="./configs/prefetch_flux_models.yaml"
PRELOAD_MODELS_CONFIG="configs/preload_flux_models.yaml"
SERVER_LOG_DIR="logs/run_diflow"
SERVER_LOG="${SERVER_LOG_DIR}/server.log"

# Edit this list to add/remove request rates.
REQUEST_RATES=(1.0 2.0 4.0 6.0 8.0)

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

echo "Starting DiFlow worker..."
env LOGLEVEL=INFO mpirun -n "${NUM_WORKERS}" --bind-to core --map-by slot:pe=8 \
    --hostfile "${HOSTFILE}" \
    python3 diffusionflow/backend/worker.py \
    --base-port "${BASE_PORT}" \
    --prefetch-models-config "${PREFETCH_MODELS_CONFIG}" &
WORKER_PID=$!

echo "Waiting for worker to start..."
sleep 10

mkdir -p "$SERVER_LOG_DIR"
echo "Starting DiFlow server; log: $SERVER_LOG"
LOGLEVEL=INFO python3 -m diffusionflow.backend.server \
    --hostfile "${HOSTFILE}" \
    --port "${SERVER_PORT}" \
    --base-port "${BASE_PORT}" \
    --preload-models-config "${PRELOAD_MODELS_CONFIG}" \
    --enable-early-abort >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server

echo "Registering workflows..."
bash ae-scripts/register_workflows.sh \
    --workflows "${WORKFLOWS}" \
    --server-url "${SERVER_URL}"

echo "Running warmup..."
PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" python3 ae-scripts/benchmark_client.py \
    --server-url "${SERVER_URL}" \
    --trace-file "ae-scripts/traces/generated_trace_n3_rs0.5_st50000_t120_seed666.txt" \
    --workflows "${WORKFLOWS}" \
    --slo-scale "${SLO_SCALE}" \
    --log-dir "${LOG_DIR}" \
    --warmup

for rate in "${REQUEST_RATES[@]}"; do
    echo "Running request rate ${rate}..."
    PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" python3 ae-scripts/benchmark_client.py \
        --server-url "${SERVER_URL}" \
        --trace-file "ae-scripts/traces/generated_trace_n3_rs${rate}_st50000_t120_seed666.txt" \
        --workflows "${WORKFLOWS}" \
        --slo-scale "${SLO_SCALE}" \
        --log-dir "${LOG_DIR}"
done

echo "Experiment finished."
