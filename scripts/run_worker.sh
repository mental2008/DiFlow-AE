#!/bin/bash

# Activate conda environment
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate df

# Set LD_LIBRARY_PATH for NVIDIA libraries
export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

DEBUG_MODE=0
if [ ! -z "$1" ]; then
  DEBUG_MODE=$1
fi

LOG_DIR="logs"
if [ ! -z "$2" ]; then
  LOG_DIR=$2
fi

WORKER_HOSTFILE="$LOG_DIR/worker_hostfile"

# Generate hostfile from SLURM allocation
if [ -n "$SLURM_NODELIST" ]; then
  scontrol show hostnames $SLURM_NODELIST | sort > "$WORKER_HOSTFILE"
  echo "Hostfile generated from SLURM allocation at $WORKER_HOSTFILE:"
  cat "$WORKER_HOSTFILE"
else
  echo "Error: SLURM_NODELIST not set. Cannot generate hostfile. Exiting."
  exit 1
fi

num=$(cat "$WORKER_HOSTFILE" | wc -l)

# mpirun -n 4 --hostfile hostfile --mca btl_base_verbose 100 python diffusionflow/backend/worker.py
# mpirun -n $num --hostfile hostfile python diffusionflow/backend/worker.py
if [ "$DEBUG_MODE" -eq 1 ]; then
  env LOGLEVEL=DEBUG mpirun -n $num --bind-to core --map-by slot:pe=12 --hostfile "$WORKER_HOSTFILE" python diffusionflow/backend/worker.py
else
  mpirun -n $num --bind-to core --map-by slot:pe=12 --hostfile "$WORKER_HOSTFILE" python diffusionflow/backend/worker.py
fi
# env LOGLEVEL=DEBUG mpirun -n $num --bind-to core --map-by slot:pe=12 --hostfile hostfile nsys profile python diffusionflow/backend/worker.py
