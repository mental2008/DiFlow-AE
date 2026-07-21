#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Edit these values for a different one-off run.
HOSTFILE="worker_hostfile_1gpu"
NPROC=1
PE=12
BASE_PORT=12500
SERVER_PORT=7777
SERVER_HOST="0.0.0.0"
SERVER_URL="http://localhost:${SERVER_PORT}"
LOGLEVEL="DEBUG"
SERVER_READY_TIMEOUT=300

REGISTER_SCRIPT="examples/register_flux_schnell_txt2img_workflow.py"
MODEL_PATH="./models/FLUX.1-schnell"
RUN_SCRIPT="examples/run_flux_schnell_workflow.py"
SERVICE_ID="flux_schnell_txt2img_workflow"
OUTPUT_PATH="./ae-results/verify_correctness/diflow_flux_schnell.png"

LOG_DIR="logs/run_flux_schnell"
WORKER_LOG="$LOG_DIR/worker.log"
SERVER_LOG="$LOG_DIR/server.log"

WORKER_PID=""
SERVER_PID=""

cleanup() {
  local exit_code=$?

  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Stopping server (pid $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
  fi

  if [[ -n "$WORKER_PID" ]] && kill -0 "$WORKER_PID" 2>/dev/null; then
    echo "Stopping worker (pid $WORKER_PID)..."
    kill "$WORKER_PID" 2>/dev/null || true
  fi

  wait "$SERVER_PID" 2>/dev/null || true
  wait "$WORKER_PID" 2>/dev/null || true
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

show_recent_log() {
  local log_file="$1"
  local lines="$2"
  tail -n "$lines" "$log_file" >&2 || true
}

require_hostfile() {
  if [[ ! -f "$HOSTFILE" ]]; then
    echo "Hostfile not found: $HOSTFILE" >&2
    exit 1
  fi
}

start_worker() {
  echo "Starting worker; log: $WORKER_LOG"
  env LOGLEVEL="$LOGLEVEL" mpirun -n "$NPROC" --bind-to core --map-by "slot:pe=${PE}" \
    --hostfile "$HOSTFILE" \
    python3 diffusionflow/backend/worker.py --base-port "$BASE_PORT" \
    >"$WORKER_LOG" 2>&1 &
  WORKER_PID=$!

  sleep 2
  if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    echo "Worker exited early. Last log lines:" >&2
    show_recent_log "$WORKER_LOG" 80
    exit 1
  fi
}

start_server() {
  echo "Starting server; log: $SERVER_LOG"
  LOGLEVEL="$LOGLEVEL" python3 -m diffusionflow.backend.server \
    --host "$SERVER_HOST" \
    --hostfile "$HOSTFILE" \
    --port "$SERVER_PORT" \
    --base-port "$BASE_PORT" \
    >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
}

wait_for_server() {
  echo "Waiting for server readiness on $SERVER_URL ..."
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
}

run_workflow() {
  echo "Registering workflow with $REGISTER_SCRIPT"
  python3 "$REGISTER_SCRIPT" --model-path "$MODEL_PATH" --server-url "$SERVER_URL"

  echo "Running workflow $SERVICE_ID"
  python3 "$RUN_SCRIPT" \
    --service-id "$SERVICE_ID" \
    --server-url "$SERVER_URL" \
    --output-path "$OUTPUT_PATH" \
    --store-images
}

require_hostfile
mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_PATH")"
start_worker
start_server
wait_for_server
echo "Server is ready."
run_workflow
echo "Done. Output written to $OUTPUT_PATH"
