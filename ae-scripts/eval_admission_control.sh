#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" 

SERVER_READY_TIMEOUT=300
LOG_DIR="logs/eval_admission_control"

HOSTFILE="worker_hostfile"
NUM_WORKERS=$(awk 'NF > 0 { count++ } END { print count + 0 }' "${HOSTFILE}")

WORKER_PID=""
SERVER_PID=""
WORKER_LOG=""
SERVER_LOG=""

show_recent_log() {
  local log_file="$1"
  local lines="$2"

  tail -n "$lines" "$log_file" >&2 || true
}

stop_services() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Stopping server (pid $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
  fi

  if [[ -n "$WORKER_PID" ]] && kill -0 "$WORKER_PID" 2>/dev/null; then
    echo "Stopping worker (pid $WORKER_PID)..."
    kill "$WORKER_PID" 2>/dev/null || true
  fi

  if [[ -n "$SERVER_PID" ]]; then
    wait "$SERVER_PID" 2>/dev/null || true
  fi

  if [[ -n "$WORKER_PID" ]]; then
    wait "$WORKER_PID" 2>/dev/null || true
  fi

  WORKER_PID=""
  SERVER_PID=""
}

cleanup() {
  local exit_code=$?

  stop_services
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

print_experiment() {
  local title="$1"

  echo
  echo "============================================================"
  echo "$title"
  echo "============================================================"
}

ensure_worker_running() {
  sleep 2

  if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    echo "Worker exited early. Last log lines:" >&2
    show_recent_log "$WORKER_LOG" 80
    exit 1
  fi
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

start_profile_services() {
  WORKER_LOG="$LOG_DIR/profile_operator_latency_worker.log"
  SERVER_LOG="$LOG_DIR/profile_operator_latency_server.log"

  echo "Starting worker; log: $WORKER_LOG"
  env LOGLEVEL=DEBUG mpirun -n "${NUM_WORKERS}" --bind-to core --map-by slot:pe=12 --hostfile "${HOSTFILE}" python3 diffusionflow/backend/worker.py --base-port 12500 --prefetch-models-config ./configs/prefetch_flux_models.yaml >"$WORKER_LOG" 2>&1 &
  WORKER_PID=$!
  ensure_worker_running

  echo "Starting server; log: $SERVER_LOG"
  LOGLEVEL=DEBUG python3 -m diffusionflow.backend.server --hostfile "${HOSTFILE}" --port 7777 --base-port 12500 --preload-models-config configs/preload_flux_models.yaml >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  wait_for_server
}

start_without_admission_control_services() {
  WORKER_LOG="$LOG_DIR/without_admission_control_worker.log"
  SERVER_LOG="$LOG_DIR/without_admission_control_server.log"

  echo "Starting worker; log: $WORKER_LOG"
  env LOGLEVEL=INFO mpirun -n "${NUM_WORKERS}" --bind-to core --map-by slot:pe=12 --hostfile "${HOSTFILE}" python3 diffusionflow/backend/worker.py --base-port 12500 --prefetch-models-config ./configs/prefetch_flux_models.yaml >"$WORKER_LOG" 2>&1 &
  WORKER_PID=$!
  ensure_worker_running

  echo "Starting server; log: $SERVER_LOG"
  LOGLEVEL=INFO python3 -m diffusionflow.backend.server --hostfile "${HOSTFILE}" --port 7777 --base-port 12500 --preload-models-config configs/preload_flux_models.yaml >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  wait_for_server
}

start_with_admission_control_services() {
  WORKER_LOG="$LOG_DIR/with_admission_control_worker.log"
  SERVER_LOG="$LOG_DIR/with_admission_control_server.log"

  echo "Starting worker; log: $WORKER_LOG"
  env LOGLEVEL=INFO mpirun -n "${NUM_WORKERS}" --bind-to core --map-by slot:pe=12 --hostfile "${HOSTFILE}" python3 diffusionflow/backend/worker.py --base-port 12500 --prefetch-models-config ./configs/prefetch_flux_models.yaml >"$WORKER_LOG" 2>&1 &
  WORKER_PID=$!
  ensure_worker_running

  echo "Starting server; log: $SERVER_LOG"
  LOGLEVEL=INFO python3 -m diffusionflow.backend.server --hostfile "${HOSTFILE}" --port 7777 --base-port 12500 --preload-models-config configs/preload_flux_models.yaml --enable-early-abort >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  wait_for_server
}

run_timeout_measurement() {
  print_experiment "First experiment: measure the timeouts to setup SLO"
  python3 ae-scripts/measure_flux_schnell_workflow_timeouts.py
}

run_operator_latency_profile() {
  print_experiment "Second experiment: profile the operator latency"
  start_profile_services
  bash ae-scripts/register_workflows.sh --workflows flux_schnell --server-url http://localhost:7777
  python3 ae-scripts/run_flux_schnell_warmup.py
  python3 ae-scripts/parse_op_latency.py
  stop_services
}

run_without_admission_control() {
  print_experiment "Third experiment: evaluate without admission control"
  start_without_admission_control_services
  bash ae-scripts/register_workflows.sh --workflows flux_schnell --server-url http://localhost:7777
  python3 ae-scripts/run_flux_schnell_warmup.py
  PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" python3 ae-scripts/benchmark_client.py --server-url http://localhost:7777 --trace-file ae-scripts/traces/generated_trace_n3_rs4.0_st50000_t120_seed666.txt --workflows flux_schnell --slo-scale 2.0 --log-dir diflow-wo-admission-control
  stop_services
}

run_with_admission_control() {
  print_experiment "Fourth experiment: evaluate with admission control"
  start_with_admission_control_services
  bash ae-scripts/register_workflows.sh --workflows flux_schnell --server-url http://localhost:7777
  python3 ae-scripts/run_flux_schnell_warmup.py
  PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" python3 ae-scripts/benchmark_client.py --server-url http://localhost:7777 --trace-file ae-scripts/traces/generated_trace_n3_rs4.0_st50000_t120_seed666.txt --workflows flux_schnell --slo-scale 2.0 --log-dir diflow-with-admission-control
  stop_services
}

mkdir -p "$LOG_DIR"

run_timeout_measurement
run_operator_latency_profile
run_without_admission_control
run_with_admission_control

python3 ae-scripts/print_admission_control_results.py

echo
echo "All admission control experiments completed."
