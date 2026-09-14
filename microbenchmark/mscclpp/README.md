# MSCCL++ ordering microbenchmark

This benchmark compares pipelined PUTs, in-QP signaling, and per-transfer draining with MSCCL++ PortChannel.

Requires IB verbs.

## Build

```bash
./build_mscclpp.sh
./build.sh
```

## Run

```bash
./run.sh ~/hostfile
./run.sh ~/hostfile 0
```

The second command disables the staging copy.
