#!/bin/bash

# Parse command line arguments
setting=""
while getopts "s:" opt; do
    case $opt in
        s)
            setting="$OPTARG"
            ;;
        \?)
            echo "Invalid option: -$OPTARG"
            echo "Usage: $0 -s <setting>"
            echo "Settings: sd3, flux, s1, s2, s3"
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument"
            echo "Usage: $0 -s <setting>"
            echo "Settings: sd3, flux, s1, s2, s3"
            exit 1
            ;;
    esac
done

if [ -z "$setting" ]; then
    echo "Usage: $0 -s <setting>"
    echo "Settings: sd3, flux, s1, s2, s3"
    exit 1
fi

# Set baseline config based on setting
case $setting in
    sd3)
        baseline_config="submodules/baselines/baseline_configs/clockwork_sd3_gpu_1.yml"
        ;;
    flux)
        baseline_config="submodules/baselines/baseline_configs/clockwork_flux_gpu_1.yml"
        ;;
    1GPU_test)
        baseline_config="submodules/baselines/baseline_configs/clockwork_sd3_flux_1GPU_test.yml"
        ;;
    s1)
        baseline_config="submodules/baselines/baseline_configs/clockwork_S1.yml"
        ;;
    s2)
        baseline_config="submodules/baselines/baseline_configs/clockwork_S2.yml"
        ;;
    s3)
        baseline_config="submodules/baselines/baseline_configs/clockwork_S3.yml"
        ;;
    *)
        echo "Invalid setting: $setting"
        echo "Valid settings: sd3, flux, s1, s2, s3"
        exit 1
        ;;
esac

echo "Running server with setting: $setting, config: $baseline_config"

python submodules/baselines/server.py --hostfile worker_hostfile --port 6666 --base-port 12000 --baseline-config $baseline_config
