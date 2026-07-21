#!/bin/bash

# Default values
SERVER_URL="http://localhost:7777"
WORKFLOWS="sd3_family"
MODEL_PATH="./models/FLUX.1-schnell"
CONTROLNET_CANNY_MODEL_PATH="./models/Xlabs-AI--flux-controlnet-canny-diffusers"
CONTROLNET_DEPTH_MODEL_PATH="./models/Xlabs-AI--flux-controlnet-depth-diffusers"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --server-url URL    Server URL (default: http://localhost:7777)"
    echo "  --workflows WORKFLOWS     Workflows to register: sd3_medium, sd35_large, flux_dev, flux_schnell, sd3_family, flux_family, or all (default: all)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Workflows:"
    echo "  sd3_medium:   SD3-medium workflows (txt2img CFG, ControlNet Canny CFG, ControlNet Pose CFG)"
    echo "  sd35_large:   SD35-large workflows (txt2img CFG, ControlNet Canny CFG, ControlNet Depth CFG)"
    echo "  flux_dev:    FLUX.1-dev workflows (txt2img CFG, ControlNet Canny CFG, ControlNet Depth CFG)"
    echo "  flux_schnell: FLUX.1-schnell workflows (txt2img CFG, ControlNet Canny CFG, ControlNet Depth CFG)"
    echo "  sd3_family:  Both SD3-medium and SD35-large workflows"
    echo "  flux_family: Both FLUX.1-dev and FLUX.1-schnell workflows"
    echo "  all:         All workflows (SD3-medium, SD35-large, and FLUX)"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --server-url)
            SERVER_URL="$2"
            shift 2
            ;;
        --workflows)
            WORKFLOWS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Debug: Print variable values before validation
echo "DEBUG: WORKFLOWS='$WORKFLOWS', SERVER_URL='$SERVER_URL'"

# Validate workflows
if [[ "$WORKFLOWS" != "sd3_medium" && "$WORKFLOWS" != "sd35_large" && "$WORKFLOWS" != "flux_dev" && "$WORKFLOWS" != "flux_schnell" && "$WORKFLOWS" != "sd3_family" && "$WORKFLOWS" != "flux_family" && "$WORKFLOWS" != "all" ]]; then
    echo "Error: Invalid workflows '$WORKFLOWS'. Must be sd3_medium, sd35_large, flux_dev, flux_schnell, sd3_family, flux_family, or all."
    usage
fi

echo "Registering workflows: $WORKFLOWS"
echo "Server URL: $SERVER_URL"
echo ""


# Function to register FLUX.1-schnell workflows
register_flux_schnell() {
    echo "######################## FLUX.1-schnell ########################"
    echo "Registering FLUX.1-schnell txt2img CFG workflow..."
    python3 examples/register_flux_schnell_txt2img_cfg_workflow.py --server-url "$SERVER_URL" --model-path "$MODEL_PATH" 
    echo "Registering FLUX.1-schnell ControlNet Canny CFG workflow..."
    python3 examples/register_flux_schnell_txt2img_controlnet_canny_cfg_workflow.py --server-url "$SERVER_URL" --model-path "$MODEL_PATH" --controlnet-model-path "$CONTROLNET_CANNY_MODEL_PATH"
    echo "Registering FLUX.1-schnell ControlNet Depth CFG workflow..."
    python3 examples/register_flux_schnell_txt2img_controlnet_depth_cfg_workflow.py --server-url "$SERVER_URL" --model-path "$MODEL_PATH" --controlnet-model-path "$CONTROLNET_DEPTH_MODEL_PATH"
}


# Register workflows based on selection
case "$WORKFLOWS" in
    "flux_schnell")
        register_flux_schnell
        ;;
    "all")
        register_sd3_family
        echo ""
        register_flux_family
        ;;
esac

echo ""
echo "Workflow registration completed for: $WORKFLOWS"