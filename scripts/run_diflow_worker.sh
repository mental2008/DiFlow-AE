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

# Parse num_workers from worker_hostfile
num_workers=$(wc -l < worker_hostfile)
echo "Parsed num_workers from worker_hostfile: $num_workers"

# Set prefetch_models_config based on setting
case $setting in
    sd3)
        prefetch_models_config="configs/prefetch_sd3_models.yaml"
        ;;
    flux)
        prefetch_models_config="configs/prefetch_flux_models.yaml"
        ;;
    s1|s2|s3)
        prefetch_models_config="configs/prefetch_models.yaml"
        ;;
    *)
        echo "Invalid setting: $setting"
        echo "Valid settings: sd3, flux, s1, s2, s3"
        exit 1
        ;;
esac

echo "Running with setting: $setting, workers: $num_workers, prefetch_models_config: $prefetch_models_config"

mpirun -n $num_workers --bind-to core --map-by slot:pe=12 --hostfile worker_hostfile python diffusionflow/backend/worker.py --base-port 12000 --prefetch-models-config $prefetch_models_config
