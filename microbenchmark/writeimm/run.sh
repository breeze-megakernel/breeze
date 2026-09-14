#!/bin/bash
# run.sh. full writeimm sweep (msg_size × concurrency), the notification sweep.
# Runs on both platforms; the launcher is picked automatically:
#   - inside a Slurm allocation (Perlmutter, libfabric/CXI build): srun
#   - otherwise (InfiniBand cloud, verbs build): mpirun + hostfile
#
# Usage:
#   # Perlmutter: salloc -N 4 -C gpu --gpus-per-node=4 -A <acct> ... ; then
#   ./run.sh                       # all 5 modes
#   ./run.sh 27                    # modes bitmask (27 skips nic_fence)
#   # Cloud:
#   ./run.sh 31 ~/hostfile
#
# Modes bitmask: 1=write_only 2=coupled 4=nic_fence 8=imm_gdr 16=imm_host.
#
# This bench is NVSHMEM-free: it talks libfabric or verbs directly, so
# there is no .so variant to record. the binary hash and the transport
# line the bench prints are the provenance.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODES="${1:-31}"
HOSTFILE="${2:-$HOME/hostfile}"
BIN="${SCRIPT_DIR}/writeimm_bench"

MSG_SIZES=(4096 32768 262144 1048576 4194304)
N_LIST=(1 2 4 8 16 32 64 96)

if [ -n "${SLURM_JOB_ID:-}" ]; then
    # ── Perlmutter / Slingshot (libfabric build) ──
    LAUNCH_DESC="srun, ${SLURM_NNODES} nodes"
    GPUS_PER_NODE=${GPUS_PER_NODE:-4}
    NP=$((SLURM_NNODES * GPUS_PER_NODE))
    export FI_CXI_DEFAULT_CQ_SIZE=${FI_CXI_DEFAULT_CQ_SIZE:-131072}
    export FI_HMEM_CUDA_USE_GDRCOPY=${FI_HMEM_CUDA_USE_GDRCOPY:-1}
    export MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-1}
    launch() {
        srun -n "$NP" -N "$SLURM_NNODES" --gpus-per-node="$GPUS_PER_NODE" \
            "$BIN" "$@" --local-size "$GPUS_PER_NODE"
    }
else
    # ── InfiniBand cloud (verbs build) ──
    LAUNCH_DESC="mpirun, hostfile=$HOSTFILE"
    LOCAL_SIZE=${LOCAL_SIZE:-8}
    NP=${NP:-16}
    HCA=${HCA:-mlx5_ib0}         # find yours: ibv_devinfo | grep hca_id
    launch() {
        mpirun --allow-run-as-root -np "$NP" \
            --hostfile "$HOSTFILE" \
            --map-by "ppr:${LOCAL_SIZE}:node" --bind-to none \
            -x CUDA_HOME -x LD_LIBRARY_PATH -x PATH \
            "$BIN" "$@" --hca "$HCA" --local-size "$LOCAL_SIZE"
    }
fi

OUT="writeimm_sweep_$(date +%Y%m%d_%H%M%S).log"
{
    echo "# date:     $(date -Is)"
    echo "# bench:    $(sha256sum "$BIN" | cut -d' ' -f1)"
    echo "# launch:   $LAUNCH_DESC (np=$NP)"
    echo "# modes:    $MODES"
} | tee "$OUT"

for MSG in "${MSG_SIZES[@]}"; do
    for N in "${N_LIST[@]}"; do
        echo "===== MSG=$MSG N=$N =====" | tee -a "$OUT"
        launch --n "$N" --msg "$MSG" --modes "$MODES" --no-probe \
            2>&1 | tee -a "$OUT"
    done
done

echo "Done. Results: $OUT" | tee -a "$OUT"
echo "The notification hop probe was skipped per-run (--no-probe); run once"
echo "standalone without --no-probe for the host->GPU hop measurement."
