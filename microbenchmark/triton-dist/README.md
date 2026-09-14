# Triton-distributed AllToAll benchmark

This directory runs the Triton-distributed AllToAll benchmark with the original and patched NVSHMEM Libfabric transport libraries.

## Setup

```bash
./setup_env.sh
```

Stage the original and patched NVSHMEM transport libraries under `../../src/nic-ordering/` before running.

## Run

```bash
sbatch -A <account> run.sh
M_LIST="1024 65536" bash run.sh
```

Logs include the active transport library hash and the Triton-distributed commit.
