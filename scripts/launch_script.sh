bash scripts/slurm_alloc.sh

$ scontrol show hostnames $SLURM_NODELIST | sort > worker_hostfile
$ env LOGLEVEL=DEBUG mpirun -n 1 --bind-to core --map-by slot:pe=12 --hostfile worker_hostfile python diffusionflow/backend/worker.py

$ LOGLEVEL=DEBUG python diffusionflow/backend/server.py --hostfile worker_hostfile

$ python examples/register_sd3_txt2img_workflow.py
$ python examples/run_workflow.py --service-id sd3_txt2img_workflow

$ python examples/register_sd15_txt2img_workflow.py
$ python examples/run_workflow.py --service-id sd15_txt2img_workflow


python examples/register_sd3_txt2img_controlnet_workflow.py
python examples/run_workflow.py --service-id sd3_txt2img_controlnet_workflow