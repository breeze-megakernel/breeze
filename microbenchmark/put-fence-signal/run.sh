#!/bin/bash
# Sweep Put-with-Signal across message size and concurrency.
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 8
#SBATCH --gpus-per-node=4
#SBATCH -t 01:00:00
#SBATCH -J put_signal_sweep
#SBATCH -o put_signal_sweep_%j.out
#SBATCH -e put_signal_sweep_%j.err

set -euo pipefail

VARIANT=${1:-vanilla}          # vanilla | breeze
BIN=./put_signal_bench
WARMUP=20
ITERS=100

# Signal op is passed EXPLICITLY per transport rather than relying on the
# binary's default. On Libfabric, FI_FENCE defers the flagged request until
# prior operations to the peer COMPLETE (delivery-complete), so a SET
# (write-based) signal is sound and matches the measured configuration.
# (The add-vs-set distinction matters on IBRC; see nvshmem-patch/README.md.)
SIGNAL_OP=set

NVSHMEM_HOME=${NVSHMEM_HOME:-$HOME/nvshmem}
TRANSPORT_SO=$NVSHMEM_HOME/lib/nvshmem_transport_libfabric.so.3.0.0

case "$VARIANT" in
    vanilla) bash ./switch_nvshmem.sh original ;;
    breeze)  bash ./switch_nvshmem.sh patched  ;;
    *) echo "unknown variant '$VARIANT' (use vanilla|breeze)" >&2; exit 1 ;;
esac

CONCURRENCY_LIST=(1 2 4 8 16 32 64 96 128)

# Message sizes (bytes)
MSG_SIZES=(
    4096        # 4KB
    32768       # 32KB
    262144      # 256KB
    1048576     # 1MB
    4194304     # 4MB
)

NODES=$SLURM_NNODES
GPUS_PER_NODE=4
TOTAL_GPUS=$((NODES * GPUS_PER_NODE))

OUT=put_signal_2d_sweep_${NODES}nodes_${VARIANT}_$(date +%Y%m%d_%H%M%S).txt

# ── Provenance header: this block is what lets a reader trust the file ──
{
    echo "# variant:    $VARIANT"
    echo "# signal_op:  $SIGNAL_OP"
    echo "# date:       $(date -Is)"
    echo "# job:        ${SLURM_JOB_ID:-n/a} on ${NODES} nodes x ${GPUS_PER_NODE} GPUs"
    echo "# transport:  $TRANSPORT_SO"
    echo "# sha256:     $(sha256sum "$TRANSPORT_SO" | cut -d' ' -f1)"
    echo "# bench:      $(sha256sum "$BIN" | cut -d' ' -f1)"
    env | grep '^NVSHMEM' | sed 's/^/# env:        /' || true
    echo "Starting 2D sweep..."
} | tee "$OUT"

for MSG in "${MSG_SIZES[@]}"; do
    echo "" | tee -a "$OUT"
    echo "===== MSG_SIZE=$MSG =====" | tee -a "$OUT"

    for C in "${CONCURRENCY_LIST[@]}"; do
        echo ">>> C=$C" | tee -a "$OUT"

        srun -n "$TOTAL_GPUS" -N "$NODES" --gpus-per-node="$GPUS_PER_NODE" \
            "$BIN" "$C" "$MSG" "$WARMUP" "$ITERS" "$C" "$SIGNAL_OP" \
            | tee -a "$OUT"
    done
done

echo "Done. Results: $OUT" | tee -a "$OUT"
