#!/bin/bash
# run_all2all_sweep.sh. inner per-node runner: launches Triton-distributed's
# AllToAll test across the M sweep. Invoked once per node by run.sh via srun;
# do not run directly (torchrun handles the per-GPU fan-out).
#
# The torch_ms column in the output is PyTorch's all_to_all (NCCL-backed).
# this is the AllToAll sweep's NCCL baseline; triton_ms is the GPU-initiated kernel.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source ~/.bashrc
conda activate "${TRITON_DIST_ENV:-$SCRATCH/envs/triton-dist}"

TRITON_DIST_SRC=${TRITON_DIST_SRC:-$SCRIPT_DIR/triton-distributed}

# ── NVSHMEM: use the transport staged by ../nvshmem-patch (run.sh swaps
#    original/patched before calling this script) ──────────────────────────
export NVSHMEM_HOME=${NVSHMEM_HOME:-$HOME/nvshmem}
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NVSHMEM_SYMMETRIC_SIZE=12000000000
export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_USE_IB=0
export NVSHMEM_LIBFABRIC_SUPPORT=1

# ── CXI provider tuning (Perlmutter/Slingshot); part of the measured
#    configuration. changing these changes the numbers ────────────────────
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$SCRATCH/.triton/cache}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-23456}

cd "$TRITON_DIST_SRC"

# Full sweep behind the AllToAll sweep. Override for a quick check:
#   M_LIST="1024 65536" bash run_all2all_sweep.sh
M_LIST=${M_LIST:-"128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152"}

# TODO(release): confirm -M/-N semantics in test_all_to_all.py and reconcile
# with the the AllToAll sweep caption (H=2048) before publishing.
for M in $M_LIST; do
  echo "==============================="
  echo "  M=$M"
  echo "==============================="
  torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node=4 \
    --node_rank="$SLURM_NODEID" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    python/triton_dist/test/nvidia/test_all_to_all.py \
    -M "$M" -N 32 -G 128 --topk 8 --bench_iters 100 --rounds 1
  echo ""
done
