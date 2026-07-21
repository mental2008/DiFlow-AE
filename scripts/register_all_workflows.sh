#!/bin/bash

# Default values
SERVER_URL="http://localhost:8000"
SETTING="s1"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --server-url URL    Server URL (default: http://localhost:8000)"
    echo "  --setting SETTING   Workflow setting: s1, s2, or s3 (default: s1)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Settings:"
    echo "  s1: Basic workflows (SD3, FLUX, SDXL, SD15)"
    echo "  s2: S1 + CFG versions"
    echo "  s3: S2 + ControlNet versions"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --server-url)
            SERVER_URL="$2"
            shift 2
            ;;
        --setting)
            SETTING="$2"
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

# Validate setting
if [[ "$SETTING" != "s1" && "$SETTING" != "s2" && "$SETTING" != "s3" ]]; then
    echo "Error: Invalid setting '$SETTING'. Must be s1, s2, or s3."
    usage
fi

echo "Registering workflows with setting: $SETTING"
echo "Server URL: $SERVER_URL"
echo ""

# Function to register S1 workflows (basic workflows)
register_s1() {
    echo "######################## S1: Basic Workflows ########################"
    # SD3
    echo "Registering SD3 workflow..."
    python examples/register_sd3_txt2img_workflow.py --server-url "$SERVER_URL"

    # FLUX
    echo "Registering FLUX workflow..."
    python examples/register_flux_txt2img_workflow.py --server-url "$SERVER_URL"

    # SDXL
    echo "Registering SDXL workflow..."
    python examples/register_sdxl_txt2img_workflow.py --server-url "$SERVER_URL"

    # SD15
    echo "Registering SD15 workflow..."
    python examples/register_sd15_txt2img_workflow.py --server-url "$SERVER_URL"
}

# Function to register S2 workflows (S1 + CFG versions)
register_s2() {
    register_s1
    echo ""
    echo "######################## S2: Adding CFG Versions ########################"

    # SD3 CFG
    echo "Registering SD3 CFG workflow..."
    python examples/register_sd3_txt2img_cfg_workflow.py --server-url "$SERVER_URL"

    # FLUX CFG
    echo "Registering FLUX CFG workflow..."
    python examples/register_flux_txt2img_cfg_workflow.py --server-url "$SERVER_URL"

    # SDXL CFG
    echo "Registering SDXL CFG workflow..."
    python examples/register_sdxl_txt2img_cfg_workflow.py --server-url "$SERVER_URL"

    # SD15 CFG
    echo "Registering SD15 CFG workflow..."
    python examples/register_sd15_txt2img_cfg_workflow.py --server-url "$SERVER_URL"
}

# Function to register S3 workflows (S2 + ControlNet versions)
register_s3() {
    register_s2
    echo ""
    echo "######################## S3: Adding ControlNet Versions ########################"

    # SD3 ControlNet workflows
    echo "Registering SD3 ControlNet workflows..."
    python examples/register_sd3_txt2img_controlnet_canny_workflow.py --server-url "$SERVER_URL"
    python examples/register_sd3_txt2img_controlnet_canny_cfg_workflow.py --server-url "$SERVER_URL"
    python examples/register_sd3_txt2img_controlnet_pose_workflow.py --server-url "$SERVER_URL"
    python examples/register_sd3_txt2img_controlnet_pose_cfg_workflow.py --server-url "$SERVER_URL"

    # FLUX ControlNet workflows
    echo "Registering FLUX ControlNet workflows..."
    python examples/register_flux_txt2img_controlnet_canny_workflow.py --server-url "$SERVER_URL"
    python examples/register_flux_txt2img_controlnet_canny_cfg_workflow.py --server-url "$SERVER_URL"
    python examples/register_flux_txt2img_controlnet_depth_workflow.py --server-url "$SERVER_URL"
    python examples/register_flux_txt2img_controlnet_depth_cfg_workflow.py --server-url "$SERVER_URL"

    # SDXL ControlNet workflows
    echo "Registering SDXL ControlNet workflows..."
    python examples/register_sdxl_txt2img_controlnet_canny_workflow.py --server-url "$SERVER_URL"
    python examples/register_sdxl_txt2img_controlnet_canny_cfg_workflow.py --server-url "$SERVER_URL"
    python examples/register_sdxl_txt2img_controlnet_depth_workflow.py --server-url "$SERVER_URL"
    python examples/register_sdxl_txt2img_controlnet_depth_cfg_workflow.py --server-url "$SERVER_URL"

    # SD15 ControlNet workflows
    echo "Registering SD15 ControlNet workflows..."
    python examples/register_sd15_txt2img_controlnet_canny_workflow.py --server-url "$SERVER_URL"
    python examples/register_sd15_txt2img_controlnet_canny_cfg_workflow.py --server-url "$SERVER_URL"
    python examples/register_sd15_txt2img_controlnet_depth_workflow.py --server-url "$SERVER_URL"
    python examples/register_sd15_txt2img_controlnet_depth_cfg_workflow.py --server-url "$SERVER_URL"
}

# Register workflows based on setting
case "$SETTING" in
    "s1")
        register_s1
        ;;
    "s2")
        register_s2
        ;;
    "s3")
        register_s3
        ;;
esac

echo ""
echo "Workflow registration completed for setting: $SETTING"
