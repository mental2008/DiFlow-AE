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

num_workers=$(wc -l < worker_hostfile)
echo "Parsed num_workers from worker_hostfile: $num_workers"

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
        num_workers=8
        baseline_config="submodules/baselines/baseline_configs/clockwork_S1.yml"
        ;;
    s2)
        num_workers=8
        baseline_config="submodules/baselines/baseline_configs/clockwork_S2.yml"
        ;;
    s3)
        num_workers=16
        baseline_config="submodules/baselines/baseline_configs/clockwork_S3.yml"
        ;;
    *)
        echo "Invalid setting: $setting"
        echo "Valid settings: sd3, flux, s1, s2, s3"
        exit 1
        ;;
esac

echo "Running with setting: $setting, workers: $num_workers, config: $baseline_config"

mpirun -n $num_workers --bind-to core --map-by slot:pe=12 --hostfile worker_hostfile python submodules/baselines/worker.py --baseline-config $baseline_config --base-port 12000
