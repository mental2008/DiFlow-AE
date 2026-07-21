import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

CMD_SD35_LARGE_CFG = [
    "python",
    "examples/run_sd3_workflow.py",
    "--service-id", 
    "sd35_large_txt2img_cfg_workflow",
    "--server-url", "http://localhost:7777",
]

CMD_SD35_LARGE_CONTROLNET_Canny_CFG = [
    "python",
    "examples/run_sd3_workflow.py",
    "--service-id", 
    "sd35_large_txt2img_controlnet_canny_cfg_workflow",
    "--server-url", "http://localhost:7777",
]

CMD_SD35_LARGE_CONTROLNET_DEPTH_CFG = [
    "python",
    "examples/run_sd3_workflow.py",
    "--service-id", 
    "sd35_large_txt2img_controlnet_depth_cfg_workflow",
    "--server-url", "http://localhost:7777",
]

def run_one(CMD, idx: int) -> int:
    print(f"Starting request {idx}, time: {datetime.now()}")
    result = subprocess.run(CMD, text=True)
    print(f"Request {idx} finished with code {result.returncode}")
    return result.returncode


if __name__ == "__main__":
    NUM_REQUESTS_SERIAL = 2
    NUM_REQUESTS_CONCURRENT = 4
    
    print(f"Serial {NUM_REQUESTS_SERIAL} requests", CMD_SD35_LARGE_CFG)
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(run_one, CMD_SD35_LARGE_CFG, i) for i in range(1, NUM_REQUESTS_SERIAL + 1)]
        for f in futures:
            f.result()
    
    print(f"Serial {NUM_REQUESTS_SERIAL} requests", CMD_SD35_LARGE_CONTROLNET_Canny_CFG)
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(run_one, CMD_SD35_LARGE_CONTROLNET_Canny_CFG, i) for i in range(1, NUM_REQUESTS_SERIAL + 1)]
        for f in futures:
            f.result()
    
    print(f"Serial {NUM_REQUESTS_SERIAL} requests", CMD_SD35_LARGE_CONTROLNET_DEPTH_CFG)
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(run_one, CMD_SD35_LARGE_CONTROLNET_DEPTH_CFG, i) for i in range(1, NUM_REQUESTS_SERIAL + 1)]
        for f in futures:
            f.result()

    print(f"Concurrent {NUM_REQUESTS_CONCURRENT} requests")
    with ThreadPoolExecutor(max_workers=NUM_REQUESTS_CONCURRENT) as executor:
        futures = [executor.submit(run_one, CMD_SD35_LARGE_CFG, i) for i in range(1, NUM_REQUESTS_CONCURRENT + 1)]
        for f in futures:
            f.result()

    print(f"Concurrent {NUM_REQUESTS_CONCURRENT} requests")
    with ThreadPoolExecutor(max_workers=NUM_REQUESTS_CONCURRENT) as executor:
        futures = [executor.submit(run_one, CMD_SD35_LARGE_CONTROLNET_Canny_CFG, i) for i in range(1, NUM_REQUESTS_CONCURRENT + 1)]
        for f in futures:
            f.result()

    print(f"Concurrent {NUM_REQUESTS_CONCURRENT} requests")
    with ThreadPoolExecutor(max_workers=NUM_REQUESTS_CONCURRENT) as executor:
        futures = [executor.submit(run_one, CMD_SD35_LARGE_CONTROLNET_DEPTH_CFG, i) for i in range(1, NUM_REQUESTS_CONCURRENT + 1)]
        for f in futures:
            f.result()