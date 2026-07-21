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
           echo "  -N: Number of nodes (default: 2)"
           echo "  -G: Number of GPUs per node (default: 2)"
           exit 0;;
        \?) echo "Invalid option -$OPTARG" >&2
            exit 1;;
    esac
done

# lsdisttrain, infattllm
srun --nodes=1 -n $GPUS_PER_NODE --cpus-per-gpu=12 --gres=gpu:$GPUS_PER_NODE --pty --account infattllm bash
