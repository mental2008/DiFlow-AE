#! /bin/bash

srun hostname | sort > worker_hostfile && cat worker_hostfile
