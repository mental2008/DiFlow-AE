import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

CMD = [
    "python",
    "examples/run_workflow.py",
    "--service-id", 
    "sd3_txt2img_controlnet_canny_cfg_workflow",
    # "sd3_txt2img_controlnet_pose_cfg_workflow",
    # "sd3_txt2img_cfg_workflow",
    "--server-url", "http://localhost:7777",
]

NUM_REQUESTS_CONCURRENT = 2


def run_one(idx: int) -> int:
    print(f"Starting request {idx}, time: {datetime.now()}")
    result = subprocess.run(CMD, text=True)
    print(f"Request {idx} finished with code {result.returncode}")
    return result.returncode


if __name__ == "__main__":
    NUM_REQUESTS_SERIAL = 1
    print(f"Serial {NUM_REQUESTS_SERIAL} requests")
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(run_one, i) for i in range(1, NUM_REQUESTS_SERIAL + 1)]
        for f in futures:
            f.result()

    print(f"Concurrent {NUM_REQUESTS_CONCURRENT} requests")
    with ThreadPoolExecutor(max_workers=NUM_REQUESTS_CONCURRENT) as executor:
        futures = [executor.submit(run_one, i) for i in range(1, NUM_REQUESTS_CONCURRENT + 1)]
        for f in futures:
            f.result()