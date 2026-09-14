# NVSHMEM IBRC patch

Base version is NVSHMEM v3.5.21-0.

The patch pins operations for each peer to one QP and applies `IBV_SEND_FENCE` to the next signal atomic.

## Build

```bash
./build_patched_nvshmem.sh
```

## Switch transport

```bash
sudo ./swap_so.sh original
sudo ./swap_so.sh patched
```

The patched library also supports a runtime toggle.

```bash
export NVSHMEM_IB_PEERHASH=1
export NVSHMEM_IB_PEERHASH=0
```
