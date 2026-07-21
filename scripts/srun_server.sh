#!/bin/bash

# Default values
LOG_DIR="logs"
SERVER_PORT=6666
DEBUG_MODE=0

# Store the current working directory as the project path
PROJECT_PATH="$(pwd)"

# Create log directory
mkdir -p "$LOG_DIR"
echo "Log directory: $LOG_DIR"

echo "=== DiffusionFlow Job Submission ==="
echo "Server port: $SERVER_PORT"
echo "Log directory: $LOG_DIR"
echo ""

echo "Submitting server job..."
SERVER_JOB_ID=$(sbatch --job-name=df-cpu \
                       --nodes=1 \
                       --ntasks=1 \
                       --cpus-per-task=8 \
                       --partition=cpu \
                       --account=infattllm \
                       --output="server_%j.out" \
                       --error="server_%j.err" \
                       --export=ALL,DIFFUSIONFLOW_LOG_DIR="$LOG_DIR" \
                       --wrap="cd $PROJECT_PATH && bash scripts/run_server.sh $SERVER_PORT $DEBUG_MODE $LOG_DIR" | grep -o '[0-9]*')
if [ -z "$SERVER_JOB_ID" ] ; then
  echo "Error: Server job submission failed."
  exit 1
fi
echo "Server job submitted with ID: $SERVER_JOB_ID"

echo ""
echo "=== Job Submission Complete ==="
echo "Server job ID: $SERVER_JOB_ID"
echo "Log directory: $LOG_DIR"
echo ""
echo "To monitor jobs:"
echo "  squeue -j $SERVER_JOB_ID"
echo ""
echo "To view logs:"
echo "  tail -f $LOG_DIR/server_$SERVER_JOB_ID.out"
echo ""
echo "To cancel jobs:"
echo "  scancel $SERVER_JOB_ID"
