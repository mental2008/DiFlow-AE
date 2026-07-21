# DiffusionFlow container image
#
# Targets H100/H20 (sm_90), A100 (sm_80) and RTX 4090 (sm_89) with a single
# fat binary for the vendored diffusionflow.backend.data_engine NVSHMEM extension.
#
#   docker build -t diffusionflow:latest .
#
# NVSHMEM/MPI notes:
# - NVSHMEM comes from the official nvidia-nvshmem-cu12 wheel (headers, host
#   lib, device lib and bootstrap plugins). Its MPI bootstrap links
#   libmpi.so.40, which matches Ubuntu 22.04's Open MPI 4.1.
# - mpi4py is built from source against the same Open MPI.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python3-venv \
        git ninja-build zsh vim \
        openmpi-bin libopenmpi-dev \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# zsh shell setup for interactive container sessions.
RUN git clone --depth=1 https://github.com/ohmyzsh/ohmyzsh.git /root/.oh-my-zsh \
    && git clone --depth=1 https://github.com/zsh-users/zsh-autosuggestions.git \
        /root/.oh-my-zsh/custom/plugins/zsh-autosuggestions \
    && git clone --depth=1 https://github.com/zsh-users/zsh-syntax-highlighting.git \
        /root/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting \
    && cp /root/.oh-my-zsh/templates/zshrc.zsh-template /root/.zshrc \
    && sed -i 's/^plugins=(git)$/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' /root/.zshrc

# Torch first (biggest layer, changes least often). torch 2.5.1 on PyPI is the
# cu124 build, matching the base image.
RUN pip install --no-cache-dir torch==2.5.1

WORKDIR /workspace/DiFlow-AE

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NVSHMEM (wheels ship only versioned .so files; the linker needs the
# unversioned names)
RUN pip install --no-cache-dir nvidia-nvshmem-cu12==3.7.1 \
    && cd /usr/local/lib/python3.10/dist-packages/nvidia/nvshmem/lib \
    && ln -s libnvshmem_host.so.3 libnvshmem_host.so \
    && ln -s nvshmem_bootstrap_mpi.so.3 nvshmem_bootstrap_mpi.so

# mpi4py from source against Open MPI (the prebuilt wheel is MPICH-based)
RUN pip install --no-cache-dir --no-binary=mpi4py mpi4py

COPY . .

# Diffusers fork (vendored as a submodule). Fresh clones have an empty
# submodule dir (and the .gitmodules URL needs an SSH key), so fetch it over
# https at the pinned commit if it is not checked out.
RUN if [ ! -f submodules/diffusers/setup.py ]; then \
        rm -rf submodules/diffusers \
        && git clone https://github.com/mental2008/diffusers.git submodules/diffusers \
        && git -C submodules/diffusers checkout 1c3482238417884e430bbb23a44a508b9d1461c2 \
        && rm -rf submodules/diffusers/.git; \
    fi \
    && pip install --no-cache-dir -e submodules/diffusers/

# DiffusionFlow + the diffusionflow.backend.data_engine._data_engine extension as a fat binary:
# sm_80 (A100), sm_89 (RTX 4090), sm_90 (H100/H20) + PTX for newer archs.
ENV NVSHMEM_DIR=/usr/local/lib/python3.10/dist-packages/nvidia/nvshmem
ENV MPI_DIR=/usr/lib/x86_64-linux-gnu/openmpi
RUN TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0+PTX" \
    CC=gcc CXX=g++ \
    pip install --no-cache-dir -e . --no-build-isolation

# Open MPI refuses to run as root without these
ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

CMD ["/bin/zsh"]
