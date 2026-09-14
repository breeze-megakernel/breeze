# Put-with-Signal microbenchmark

This benchmark compares three submission patterns under increasing concurrency.

| Mode | Pattern |
|---|---|
| `pipelined` | nonblocking PUTs followed by one quiet |
| `coupled` | Put-with-Signal per transfer |
| `decoupled` | nonblocking PUTs followed by one fence and the signals |

## Build

```bash
./build.sh
```

## Run

```bash
sbatch -A <account> run.sh vanilla
sbatch -A <account> run.sh breeze
```

Direct invocation is also supported.

```bash
srun -n 32 -N 8 --gpus-per-node=4 ./put_signal_bench [N] [msg] [warmup] [iters] [nblocks] [add|set]
```
