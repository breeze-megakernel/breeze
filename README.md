# Breeze

This repository contains implementation code and microbenchmarks for GPU-initiated RDMA signaling.

## Layout

```text
src/
  protocol/                  device-side grouped signaling API
  transport/                 NVSHMEM transport patches
    nvshmem-patch-libfabric/ Libfabric patch for NVSHMEM
    nvshmem-patch-ibrc/      IBRC patch for NVSHMEM
microbenchmark/
  put-fence-signal/          Put-with-Signal benchmark
  mscclpp/                   MSCCL++ in-QP ordering benchmark
  writeimm/                  WRITE_WITH_IMM benchmark
  triton-dist/               Triton-distributed AllToAll benchmark
bootstrap_node.sh            setup script for InfiniBand cloud nodes
```

Each directory contains its own build and run instructions.

## Platforms

| | Perlmutter | InfiniBand cloud |
|---|---|---|
| GPU | A100 | H100 |
| NIC | Slingshot-11 / Cassini | ConnectX-7 |
| Transport | Libfabric | IBRC |

`put-fence-signal`, `writeimm`, and `triton-dist` support both platforms. `mscclpp` requires IB verbs.

## Dependencies

| Dependency | Version |
|---|---|
| NVSHMEM | v3.4.5-0 or v3.5.21-0 |
| CUDA | 12.x |
| MSCCL++ | pinned in `microbenchmark/mscclpp/build_mscclpp.sh` |
| Triton-distributed | pinned in `microbenchmark/triton-dist/setup_env.sh` |
