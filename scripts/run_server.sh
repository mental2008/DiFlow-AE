#!/bin/bash

# Activate conda environment
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate df

# Set LD_LIBRARY_PATH for NVIDIA libraries
export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# Default port
PORT=6666
DEBUG_MODE=0
LOG_DIR="logs"

# Parse optional port, debug, and log directory arguments
if [ ! -z "$1" ]; then
  PORT=$1
fi
if [ ! -z "$2" ]; then
  DEBUG_MODE=$2
fi
if [ ! -z "$3" ]; then
  LOG_DIR=$3
fi

HOSTFILE="worker_hostfile"
SERVER_HOSTFILE="server_hostfile"

# Generate server_hostfile with this server's hostname
hostname > "$SERVER_HOSTFILE"
echo "Server hostfile generated at $SERVER_HOSTFILE:"
cat "$SERVER_HOSTFILE"

if [ -f "$HOSTFILE" ]; then
  echo "Server using hostfile at $HOSTFILE:"
  cat "$HOSTFILE"
else
  echo "Warning: Hostfile $HOSTFILE does not exist."
fi

if [ "$DEBUG_MODE" -eq 1 ]; then
  LOGLEVEL=DEBUG python diffusionflow/backend/server.py --hostfile "$HOSTFILE" --port $PORT
else
  python diffusionflow/backend/server.py --hostfile "$HOSTFILE" --port $PORT
fi
