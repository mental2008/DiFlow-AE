module load slurm
current_time=$(date +%s)

target_time=$(date -d "23:04:00" +%s)

if [ $current_time -gt $target_time ]; then
        target_time=$(date -d "tomorrow 09:15:00" +%s)
fi

sleep_time=$((target_time - current_time))

echo $sleep_time

echo start_waiting for GPU resources...

sleep $sleep_time

bash ./scripts/slurm_alloc.sh -N 1 -G 1