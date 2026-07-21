#!/bin/bash

# Default values
NODES=1
GPUS_PER_NODE=1
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
SERVER_PORT=6666
DEBUG_MODE=0

# Store the current working directory as the project path
PROJECT_PATH="$(pwd)"

# Parse command line arguments
while getopts "N:G:p:l:hd" opt; do
    case $opt in
        N) NODES="$OPTARG";;
        G) GPUS_PER_NODE="$OPTARG";;
        p) SERVER_PORT="$OPTARG";;
        l) LOG_DIR="$OPTARG";;
        d) DEBUG_MODE=1;;
        h) echo "Usage: $0 [-N nodes] [-G gpus_per_node] [-p server_port] [-l log_dir] [-d]"
           echo "  -N: Number of nodes for workers (default: 1)"
           echo "  -G: Number of GPUs per node (default: 1)"
           echo "  -p: Server port (default: 6666)"
           echo "  -l: Log directory (default: logs/YYYYMMDD_HHMMSS)"
           echo "  -d: Enable DEBUG mode (default: off)"
           exit 0;;
        \?) echo "Invalid option -$OPTARG" >&2
            exit 1;;
    esac
done

# Create log directory
mkdir -p "$LOG_DIR"
echo "Log directory: $LOG_DIR"

echo "=== DiffusionFlow Job Submission ==="
echo "Nodes: $NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Server port: $SERVER_PORT"
echo "Log directory: $LOG_DIR"
echo ""

echo "Submitting worker job..."
WORKER_JOB_ID=$(sbatch --job-name=df-gpu \
                       --nodes=$NODES \
                       --ntasks-per-node=$GPUS_PER_NODE \
                       --cpus-per-gpu=24 \
                       --gpus-per-node=$GPUS_PER_NODE \
                       --partition=normal \
                       --account=infattllm \
                       --exclude=dgx-15 \
                       --output="$LOG_DIR/workers_%j.out" \
                       --error="$LOG_DIR/workers_%j.err" \
                       --export=ALL,DIFFUSIONFLOW_LOG_DIR="$LOG_DIR" \
                       --wrap="cd $PROJECT_PATH && bash scripts/run_worker.sh $DEBUG_MODE $LOG_DIR" | grep -o '[0-9]*')
if [ -z "$WORKER_JOB_ID" ]; then
  echo "Error: Worker job submission failed. Exiting."
  exit 1
fi
echo "Worker job submitted with ID: $WORKER_JOB_ID"
sleep 5 # Wait for the worker job to start

echo "Submitting server job..."
SERVER_JOB_ID=$(sbatch --job-name=df-cpu \
                       --nodes=1 \
                       --ntasks=1 \
                       --cpus-per-task=8 \
                       --partition=cpu \
                       --account=infattllm \
                       --output="$LOG_DIR/server_%j.out" \
                       --error="$LOG_DIR/server_%j.err" \
                       --export=ALL,DIFFUSIONFLOW_LOG_DIR="$LOG_DIR" \
                       --wrap="cd $PROJECT_PATH && bash scripts/run_server.sh $SERVER_PORT $DEBUG_MODE $LOG_DIR" | grep -o '[0-9]*')
if [ -z "$SERVER_JOB_ID" ] ; then
  echo "Error: Server job submission failed. Cancelling worker job $WORKER_JOB_ID."
  scancel $WORKER_JOB_ID
  exit 1
fi
echo "Server job submitted with ID: $SERVER_JOB_ID"

echo ""
echo "=== Job Submission Complete ==="
echo "Server job ID: $SERVER_JOB_ID"
echo "Worker job ID: $WORKER_JOB_ID"
echo "Log directory: $LOG_DIR"
echo ""
echo "To monitor jobs:"
echo "  squeue -j $SERVER_JOB_ID,$WORKER_JOB_ID"
echo ""
echo "To view logs:"
echo "  tail -f $LOG_DIR/server_$SERVER_JOB_ID.out"
echo "  tail -f $LOG_DIR/workers_$WORKER_JOB_ID.out"
echo ""
echo "To cancel jobs:"
echo "  scancel $SERVER_JOB_ID $WORKER_JOB_ID"
