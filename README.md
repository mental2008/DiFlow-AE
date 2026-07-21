# DiffusionFlow

Let's reinvent diffusion workflow!

## Installation

```bash
# To avoid environmental inconsistency, build on a node with GPUs
# (e.g. on SuperPOD: srun --nodes=1 -n 1 --gres=gpu:1 --account=infattllm --pty bash)

# Load the NVHPC suite (MPI + NVSHMEM, needed to build mpi4py and the data_engine extension)
$ module load "nvhpc-hpcx-cuda12/23.11"
# A CUDA toolkit different from the one shipped with NVHPC is required
# (e.g. conda install cuda-toolkit=12.4, or point CUDA_HOME at an existing install)

# Install necessary dependencies
$ pip install -r requirements.txt
# Install mpi4py (build from source against the loaded Open MPI; the prebuilt wheel is MPICH-based)
$ CFLAGS=-noswitcherror pip install --no-binary=mpi4py mpi4py
# Install diffusers in editable mode
$ pip install -e submodules/diffusers/
# Install DiffusionFlow in editable mode. This also builds the vendored
# diffusionflow.backend.data_engine NVSHMEM extension from csrc/; GCC must be used for
# compatibility with LibTorch. NVSHMEM_DIR / MPI_DIR env vars override the
# default NVHPC 23.11 paths, and DFLOW_SKIP_DATA_ENGINE=1 skips the extension on
# machines without NVSHMEM/MPI.
$ CC=gcc CXX=g++ pip install -e . --no-build-isolation

# If you hit libnvJitLink-related import errors afterwards, the NVHPC module
# shadows the pip-installed one; fix the search path with:
$ export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
```

## Run the Backend Service

### Single-GPU Setting

```bash
$ bash scripts/run_server.sh
```

<!-- ### Distributed Setting

```bash
# Master Node
$ python server.py --config dist_config.yaml --node-rank 0
# Worker Node
$ python server.py --config dist_config.yaml --node-rank 1
``` -->

## Examples

### Basic Stable Diffusion 3 Text-to-Image Workflow

```bash
# Register the workflow
$ python examples/basic_sd3_txt2img_workflow.py --action register
# Run the inference
$ python examples/basic_sd3_txt2img_workflow.py --action inference --service-id sd3_txt2img_workflow
```

## Testing

```bash
$ bash scripts/run_all_unittests.sh
```

## Code Formatting

```bash
$ bash scripts/format.sh
```
