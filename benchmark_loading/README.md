# Loading Benchmarks

This directory contains loading-only benchmarks. It intentionally does not write
execution latency data, so its results are separate from `benchmark/benchmark_results`.

## Commands

List available suites and models:

```bash
python -m benchmark_loading.cli --list
```

Run one suite:

```bash
python -m benchmark_loading.cli --suite flux1
```

Run one model:

```bash
python -m benchmark_loading.cli --model Flux1Dev
```

Run all registered loading benchmarks:

```bash
python -m benchmark_loading.cli --all
```

Force a re-run even when a result already exists for the current GPU:

```bash
python -m benchmark_loading.cli --suite flux1 --force-benchmark
```

## Results

Results are written under the normalized current GPU name:

```text
benchmark_loading/results/
  nvidia_h800/
    Flux1Dev.json
```

Each result contains only GPU metadata and loading metrics:

```json
{
  "gpu_type": "NVIDIA H800",
  "gpu_type_normalized": "nvidia_h800",
  "gpu_memory_total": 85028634624,
  "gpu_count": 8,
  "model_name": "Flux1Dev",
  "model_path": "/path/to/model",
  "suite": "flux1",
  "loading": {
    "disk_to_host_mem_time": 0.36,
    "host_mem_to_gpu_time": 2.94,
    "gpu_memory_required": 23802855936.0
  }
}
```

## Adding A Model

For normal loading-only profiling, add an entry to the relevant YAML file in
`benchmark_loading/configs/`:

```yaml
- model_name: NewModelId
  model_path: /path/to/huggingface/model
```

If a new model family needs custom discovery or special loading behavior, add a
new suite module in `benchmark_loading/suites/` and import it from
`benchmark_loading/registry.py`.
