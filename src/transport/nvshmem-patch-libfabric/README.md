# NVSHMEM Libfabric patch

Base version is NVSHMEM v3.4.5-0.

The patch replaces the blocking Libfabric fence path with a pending fence flag. The next signal atomic is submitted with `FI_FENCE`.

## Build

```bash
module load PrgEnv-gnu cudatoolkit cmake
./build_patched_nvshmem.sh
```

The script builds and stages both `original` and `patched` transport libraries.

## Switch transport

```bash
./swap_so.sh original
./swap_so.sh patched
```
