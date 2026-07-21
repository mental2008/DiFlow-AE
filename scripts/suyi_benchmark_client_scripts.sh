# sd3
echo "Running sd3_txt2img_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd3_txt2img_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd3_txt2img_workflow

echo "Running sd3_txt2img_cfg_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd3_txt2img_cfg_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd3_txt2img_cfg_workflow

echo "Running sd3_txt2img_controlnet_canny_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd3_txt2img_controlnet_canny_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd3_txt2img_controlnet_canny_workflow

echo "Running sd3_txt2img_controlnet_canny_cfg_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd3_txt2img_controlnet_canny_cfg_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd3_txt2img_controlnet_canny_cfg_workflow

echo "Running sd3_txt2img_controlnet_pose_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd3_txt2img_controlnet_pose_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd3_txt2img_controlnet_pose_workflow

echo "Running sd3_txt2img_controlnet_pose_cfg_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd3_txt2img_controlnet_pose_cfg_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd3_txt2img_controlnet_pose_cfg_workflow

# sd35 large
echo "Running sd35_large_txt2img_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd35_large_txt2img_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd35_large_txt2img_workflow

echo "Running sd35_large_txt2img_cfg_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd35_large_txt2img_cfg_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd35_large_txt2img_cfg_workflow

echo "Running sd35_large_txt2img_controlnet_canny_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd35_large_txt2img_controlnet_canny_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd35_large_txt2img_controlnet_canny_workflow

echo "Running sd35_large_txt2img_controlnet_canny_cfg_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd35_large_txt2img_controlnet_canny_cfg_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd35_large_txt2img_controlnet_canny_cfg_workflow

echo "Running sd35_large_txt2img_controlnet_depth_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd35_large_txt2img_controlnet_depth_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd35_large_txt2img_controlnet_depth_workflow

echo "Running sd35_large_txt2img_controlnet_depth_cfg_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd35_large_txt2img_controlnet_depth_cfg_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id sd35_large_txt2img_controlnet_depth_cfg_workflow

# flux
echo "Running flux_txt2img_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_txt2img_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_txt2img_workflow

echo "Running flux_txt2img_cfg_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_txt2img_cfg_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_txt2img_cfg_workflow

echo "Running flux_txt2img_controlnet_canny_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_txt2img_controlnet_canny_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_txt2img_controlnet_canny_workflow

echo "Running flux_txt2img_controlnet_canny_cfg_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_txt2img_controlnet_canny_cfg_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_txt2img_controlnet_canny_cfg_workflow

echo "Running flux_txt2img_controlnet_depth_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_txt2img_controlnet_depth_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_txt2img_controlnet_depth_workflow

echo "Running flux_txt2img_controlnet_depth_cfg_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_txt2img_controlnet_depth_cfg_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_txt2img_controlnet_depth_cfg_workflow

# flux schnell
echo "Running flux_schnell_txt2img_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_schnell_txt2img_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_schnell_txt2img_workflow

echo "Running flux_schnell_txt2img_cfg_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_schnell_txt2img_cfg_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_schnell_txt2img_cfg_workflow

echo "Running flux_schnell_txt2img_controlnet_canny_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_schnell_txt2img_controlnet_canny_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_schnell_txt2img_controlnet_canny_workflow

echo "Running flux_schnell_txt2img_controlnet_canny_cfg_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_schnell_txt2img_controlnet_canny_cfg_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_schnell_txt2img_controlnet_canny_cfg_workflow

echo "Running flux_schnell_txt2img_controlnet_depth_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_schnell_txt2img_controlnet_depth_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_schnell_txt2img_controlnet_depth_workflow

echo "Running flux_schnell_txt2img_controlnet_depth_cfg_workflow"
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_schnell_txt2img_controlnet_depth_cfg_workflow
python benchmark/benchmark_client.py --server-url http://slogin-02:6666 --rps 1 --count 1 --baseline --service-id flux_schnell_txt2img_controlnet_depth_cfg_workflow
