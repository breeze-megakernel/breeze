#!/bin/bash
# Run Triton-distributed AllToAll with original and patched NVSHMEM transports.
#SBATCH -N 4
#SBATCH --gpus-per-node=4
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -o triton_dist_bench_%j.out
#SBATCH -e triton_dist_bench_%j.err

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NNODES=${SLURM_NNODES:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)}
LOGDIR=${LOGDIR:-$SCRIPT_DIR/logs}
mkdir -p "$LOGDIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PREFIX="triton-dist-${NNODES}nodes-a2a"

SWITCH=${SWITCH:-$SCRIPT_DIR/../nvshmem-patch/swap_so.sh}
SWEEP=${SWEEP:-$SCRIPT_DIR/run_all2all_sweep.sh}
TRITON_DIST_SRC=${TRITON_DIST_SRC:-$SCRIPT_DIR/triton-distributed}

# ── Environment ───────────────────────────────────────────────────────────
source ~/.bashrc
conda activate "${TRITON_DIST_ENV:-$SCRATCH/envs/triton-dist}"

export NVSHMEM_HOME=${NVSHMEM_HOME:-$HOME/nvshmem}
export NVSHMEM_DIR=$NVSHMEM_HOME
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_DISABLE_CUDA_VMM=1
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$SCRATCH/.triton/cache}
export CC=${CC:-gcc-12}
export CXX=${CXX:-g++-12}

TRANSPORT_SO=$NVSHMEM_HOME/lib/nvshmem_transport_libfabric.so.3.0.0

banner() { echo "════════════════════════════════════════════════════════════"; }

banner
echo "  Triton-distributed AllToAll benchmark"
echo "  Nodes: ${NNODES} (${SLURM_JOB_NODELIST:-interactive})"
echo "  Job: ${SLURM_JOB_ID:-n/a}   Date: $(date -Is)"
echo "  Triton-distributed: $(git -C "$TRITON_DIST_SRC" rev-parse HEAD 2>/dev/null || echo 'source dir missing')"
banner

run_variant() {
    local variant=$1 label=$2
    echo ""
    echo "━━━ ${label} ━━━"
    bash "$SWITCH" "$variant"
    local log="${LOGDIR}/${PREFIX}-${variant}-${TIMESTAMP}.log"
    {
        echo "# variant:   $variant"
        echo "# transport: $TRANSPORT_SO"
        echo "# sha256:    $(sha256sum "$TRANSPORT_SO" | cut -d' ' -f1)"
    } | tee "$log"
    srun --nodes="${NNODES}" --ntasks-per-node=1 bash "$SWEEP" 2>&1 | tee -a "$log"
    echo "$log"
}

ORIG_LOG=$(run_variant original "Original NVSHMEM (fence = quiet)" | tail -1)
PATCH_LOG=$(run_variant patched  "Patched NVSHMEM (fence = FI_FENCE)" | tail -1)

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
banner
echo "  Results summary (${NNODES} nodes)"
banner
for pair in "Original:${ORIG_LOG}" "Patched:${PATCH_LOG}"; do
    echo ""
    echo "--- ${pair%%:*} ---"
    grep -E "^  M=|Avg" "${pair#*:}" 2>/dev/null || echo "(no results)"
done

# ── CSV for plotting ──────────────────────────────────────────────────────
CSV="${LOGDIR}/${PREFIX}-summary-${TIMESTAMP}.csv"
echo "nodes,variant,M,torch_ms,triton_ms" > "$CSV"
for variant in original patched; do
    LOG=$([ "$variant" = original ] && echo "$ORIG_LOG" || echo "$PATCH_LOG")
    current_m=""
    while IFS= read -r line; do
        if [[ "$line" =~ ^"  M=" ]]; then
            current_m=${line#  M=}
        elif [[ "$line" =~ "Avg" ]] && [ -n "$current_m" ]; then
            torch_ms=$(echo "$line" | awk -F'|' '{print $3}' | tr -d ' ')
            triton_ms=$(echo "$line" | awk -F'|' '{print $4}' | tr -d ' ')
            echo "${NNODES},${variant},${current_m},${torch_ms},${triton_ms}" >> "$CSV"
            current_m=""
        fi
    done < "$LOG"
done
echo ""
echo "CSV saved to: $CSV"
