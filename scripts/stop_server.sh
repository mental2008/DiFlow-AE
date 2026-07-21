#!/bin/bash

graceful_shutdown() {
    local pid=$1
    local name=$2

    echo "Sending SIGTERM to $name (PID: $pid)..."
    kill -TERM $pid

    # Wait for up to 30 seconds for graceful shutdown
    COUNTER=0
    while ps -p $pid > /dev/null && [ $COUNTER -lt 30 ]; do
        echo "Waiting for $name to shutdown gracefully..."
        sleep 1
        ((COUNTER++))
    done

    # If process is still running after timeout, force kill
    if ps -p $pid > /dev/null; then
        echo "$name did not shutdown gracefully, forcing kill..."
        kill -9 $pid
        sleep 1
    else
        echo "$name stopped gracefully"
    fi
}

cd diffusionflow/backend

# Check if server is running
if [ -f server.pid ]; then
    SERVER_PID=$(cat server.pid)
    if ps -p $SERVER_PID > /dev/null; then
        graceful_shutdown $SERVER_PID "main server"

        if [ -f worker.pid ]; then
            WORKER_PIDS=$(cat worker.pid)
            for WORKER_PID in $WORKER_PIDS; do
                if ps -p $WORKER_PID > /dev/null; then
                    graceful_shutdown $WORKER_PID "worker process"
                fi
            done
        fi

        rm -f server.pid worker.pid
    else
        echo "Server not running (stale PID file)"
        rm -f server.pid worker.pid
    fi
else
    echo "Server not running (no PID file found)"
fi
