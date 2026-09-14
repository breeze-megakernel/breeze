# WRITE_WITH_IMM microbenchmark

This benchmark compares sender-side signaling with receiver-side `WRITE_WITH_IMM` notification paths.

## Build

```bash
./build.sh
USE_LIBFABRIC=1 ./build.sh
```

The first command builds the verbs backend. The second builds the Libfabric backend.

## Run

```bash
./run.sh ~/hostfile
mpirun ... ./writeimm_bench --n 96 --msg 262144 --hca mlx5_ib0
```

GDRCopy is used when available. Use `--relaxed-ordering` only for ordering checks, not for write-based signal performance runs.
