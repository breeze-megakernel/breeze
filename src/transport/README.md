# NIC-side ordering

This directory contains NVSHMEM transport patches that replace a blocking proxy fence with NIC-side ordering.

| Directory | NVSHMEM base | Transport |
|---|---|---|
| `nvshmem-patch-libfabric/` | v3.4.5-0 | Libfabric |
| `nvshmem-patch-ibrc/` | v3.5.21-0 | IBRC |

Each subdirectory contains the patch, a build script, and a script for switching between the original and patched transport libraries.

## Libfabric

The patch records a pending fence and applies `FI_FENCE` to the next signal atomic.

## IBRC

The patch pins operations for each peer to one QP and applies `IBV_SEND_FENCE` to the next signal atomic.
