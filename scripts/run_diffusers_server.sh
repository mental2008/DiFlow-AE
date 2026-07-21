#!/bin/bash

# Parse command line arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <setting>"
    echo "Settings: s1, s2, s3"
    exit 1
fi

setting=$1

# Set baseline config based on setting
case $setting in
    s1)
        baseline_config="submodules/baselines/baseline_configs/basic_S1.yml"
        ;;
    s2)
        baseline_config="submodules/baselines/baseline_configs/basic_S2.yml"
        ;;
    s3)
        baseline_config="submodules/baselines/baseline_configs/basic_S3.yml"
        ;;
    *)
        echo "Invalid setting: $setting"
        echo "Valid settings: s1, s2, s3"
        exit 1
        ;;
esac

echo "Running server with setting: $setting, config: $baseline_config"

python submodules/baselines/server.py --hostfile worker_hostfile --port 6666 --base-port 12000 --baseline-config $baseline_config
