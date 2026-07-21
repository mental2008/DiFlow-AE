#!/bin/bash

# Parse command line arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <setting>"
    echo "Settings: s1, s2 (8 workers) or s3 (16 workers)"
    exit 1
fi

setting=$1

# Set number of workers and baseline config based on setting
case $setting in
    s1)
        num_workers=8
        baseline_config="submodules/baselines/baseline_configs/basic_S1.yml"
        ;;
    s2)
        num_workers=8
        baseline_config="submodules/baselines/baseline_configs/basic_S2.yml"
        ;;
    s3)
        num_workers=16
        baseline_config="submodules/baselines/baseline_configs/basic_S3.yml"
        ;;
    *)
        echo "Invalid setting: $setting"
        echo "Valid settings: s1, s2, s3"
        exit 1
        ;;
esac

echo "Running with setting: $setting, workers: $num_workers, config: $baseline_config"

mpirun -n $num_workers --bind-to core --map-by slot:pe=12 --hostfile worker_hostfile python submodules/baselines/worker.py --baseline-config $baseline_config --base-port 12000
