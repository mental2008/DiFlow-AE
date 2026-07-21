#!/bin/bash
#SBATCH --job-name=fty               # Job name
#SBATCH --nodes=2
#SBATCH --ntasks=2                     # Number of nodes
#SBATCH --cpus-per-gpu=16
#SBATCH --gres=gpu:1                  # Number of GPUs
#SBATCH --account=infattllm           # Account name
#SBATCH --partition=preempt
#SBATCH --propagate=STACK
# #SBATCH --exclude=dgx-48,dgx-42,dgx-43,dgx-39  # Exclude these nodes
#SBATCH -o logs/%j/slurm.out
#SBATCH -e logs/%j/slurm.err

# Create log directory for this run
log_dir="${SLURM_SUBMIT_DIR}/logs/${SLURM_JOB_ID}"
mkdir -p $log_dir

# Load modules and set up environment
module load nvhpc-hpcx-cuda12/23.11 Anaconda3 slurm

srun hostname | sort >$log_dir/hostfile
N=$(wc -l $log_dir/hostfile | awk '{print $1}')
export NVSHMEM_MPI_SUPPORT=1

nvidia-smi >> $log_dir/nvidia-smi.log
numactl -H >> $log_dir/numactl.log

script_dir=$SLURM_SUBMIT_DIR # Use the directory from which the job was submitted

# Set up conda environment
eval "$(conda shell.bash hook)"
conda activate dflow

export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

mpirun -n $N --hostfile $log_dir/hostfile --bind-to core --map-by slot:pe=8 --report-bindings python ${script_dir}/bw_test.py --log-dir $log_dir --hostfile $log_dir/hostfile