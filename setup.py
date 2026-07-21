import os

import setuptools

# Building the data_engine NVSHMEM extension requires torch, CUDA, NVSHMEM and MPI
# (install with `pip install -e . --no-build-isolation` so the environment's
# torch is used). Set DFLOW_SKIP_DATA_ENGINE=1 to install without the extension
# (diffusionflow.backend.data_engine will then be unusable at runtime).
if os.getenv("DFLOW_SKIP_DATA_ENGINE", "0") == "1":
    ext_modules = []
    cmdclass = {}
else:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    nvshmem_dir = os.getenv(
        "NVSHMEM_DIR",
        "/cm/shared/apps/nvhpc/23.11/Linux_x86_64/23.11/comm_libs/nvshmem",
    )
    mpi_dir = os.getenv(
        "MPI_DIR",
        "/cm/shared/apps/nvhpc/23.11/Linux_x86_64/23.11/comm_libs/12.3/hpcx/hpcx-2.16/ompi",
    )
    assert nvshmem_dir is not None and os.path.exists(
        nvshmem_dir
    ), "Failed to find NVSHMEM"
    assert mpi_dir is not None and os.path.exists(mpi_dir), "Failed to find MPI"
    print(f"NVSHMEM directory: {nvshmem_dir}")
    print(f"MPI directory: {mpi_dir}")

    # Target architectures for the extension. Defaults to Hopper (the original
    # target); override via TORCH_CUDA_ARCH_LIST, e.g. "8.0;8.9;9.0" to also
    # cover A100 and RTX 4090.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    cxx_flags = [
        "-O3",
        # Only the MPI C API is used; skipping the C++ bindings avoids a
        # dependency on libmpi_cxx.so (which HPC-X folds into libmpi but
        # stock Open MPI ships separately)
        "-DOMPI_SKIP_MPICXX=1",
        "-Wno-deprecated-declarations",
        "-Wno-unused-variable",
        "-Wno-sign-compare",
        "-Wno-reorder",
        "-Wno-attributes",
    ]
    nvcc_flags = [
        "-O3",
        "-DOMPI_SKIP_MPICXX=1",
        "-Xcompiler",
        "-O3",
        "-rdc=true",
        "--ptxas-options=--register-usage-level=10",
    ]
    include_dirs = [
        f"{os.path.dirname(os.path.abspath(__file__))}/csrc",
        f"{nvshmem_dir}/include",
        f"{mpi_dir}/include",
    ]
    sources = [
        "csrc/data_engine/allocator/buddy_allocator.cpp",
        "csrc/data_engine/allocator/paged_allocator.cpp",
        "csrc/data_engine/allocator/small_object_allocator.cpp",
        "csrc/data_engine/allocator/complicated_allocator.cpp",
        "csrc/data_engine/engine/backends/nvshmem/nvshmem_backend.cpp",
    ]
    library_dirs = [f"{nvshmem_dir}/lib", f"{mpi_dir}/lib"]

    # Disable DLTO (default by PyTorch)
    nvcc_dlink = [
        "-dlink",
        f"-L{nvshmem_dir}/lib",
        "-lnvshmem_host",
        "-lnvshmem_device",
        f"-L{mpi_dir}/lib",
        "-lmpi",
    ]
    extra_link_args = [
        "-l:libnvshmem_host.so",
        "-l:libnvshmem_device.a",
        "-l:nvshmem_bootstrap_mpi.so",
        f"-Wl,-rpath,{nvshmem_dir}/lib",
        "-l:libmpi.so",
        f"-Wl,-rpath,{mpi_dir}/lib",
    ]
    extra_compile_args = {
        "cxx": cxx_flags,
        "nvcc": nvcc_flags,
        "nvcc_dlink": nvcc_dlink,
    }

    ext_modules = [
        CUDAExtension(
            name="diffusionflow.backend.data_engine._data_engine",
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            sources=sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setuptools.setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
