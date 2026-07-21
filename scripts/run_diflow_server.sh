#!/bin/bash

# Parse command line arguments
setting=""
scheduling_policy="dynamic"
while getopts "s:p:" opt; do
    case $opt in
        s)
            setting="$OPTARG"
            ;;
        p)
            scheduling_policy="$OPTARG"
            ;;
        \?)
            echo "Invalid option: -$OPTARG"
            echo "Usage: $0 -s <setting> [-p <scheduling_policy>]"
            echo "Settings: sd3, flux, s1, s2, s3"
            echo "Scheduling policies: exclusive, random, dynamic (default: dynamic)"
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument"
            echo "Usage: $0 -s <setting> [-p <scheduling_policy>]"
            echo "Settings: sd3, flux, s1, s2, s3"
            echo "Scheduling policies: exclusive, random, dynamic (default: dynamic)"
            exit 1
            ;;
    esac
done

if [ -z "$setting" ]; then
    echo "Usage: $0 -s <setting> [-p <scheduling_policy>]"
    echo "Settings: sd3, flux, s1, s2, s3"
    echo "Scheduling policies: exclusive, random, dynamic (default: dynamic)"
    exit 1
fi

# Set number of workers based on setting
case $setting in
    sd3)
        preload_models_config="configs/preload_sd3_models.yaml"
        ;;
    flux)
        preload_models_config="configs/preload_flux_models.yaml"
        ;;
    s1|s2)
        preload_models_config="configs/preload_models.yaml"
        ;;
    s3)
        preload_models_config="configs/preload_models.yaml"
        ;;
    *)
        echo "Invalid setting: $setting"
        echo "Valid settings: sd3, flux, s1, s2, s3"
        exit 1
        ;;
esac

echo "Running with setting: $setting, preload_models_config: $preload_models_config, scheduling_policy: $scheduling_policy"

python diffusionflow/backend/server.py --hostfile worker_hostfile --port 6666 --base-port 12000 --preload-models-config $preload_models_config --scheduling-policy $scheduling_policy
