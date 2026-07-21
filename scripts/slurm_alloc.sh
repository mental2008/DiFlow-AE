#! /bin/bash

# Default values
NODES=1
GPUS_PER_NODE=1

# Parse command line arguments
while getopts "N:G:h" opt; do
    case $opt in
        N) NODES="$OPTARG";;
        G) GPUS_PER_NODE="$OPTARG";;
        h) echo "Usage: $0 [-N nodes] [-G gpus_per_node]"
           echo "  -N: Number of nodes (default: 1)"
           echo "  -G: Number of GPUs per node (default: 1)"
           exit 0;;
        \?) echo "Invalid option -$OPTARG" >&2
            exit 1;;
    esac
done

salloc --nodes=$NODES --ntasks-per-node=$GPUS_PER_NODE --gpus-per-node=$GPUS_PER_NODE --cpus-per-gpu=24 \
    --time=08:00:00 --partition=normal --account=infattllm --job-name=df-gpu -x dgx-23,dgx-09

# salloc --nodes=$NODES --ntasks-per-node=$GPUS_PER_NODE --gpus-per-node=$GPUS_PER_NODE --cpus-per-gpu=24 \
#     --time=08:00:00 --partition=preempt --account=infattllm --job-name=df-gpu -x dgx-23
