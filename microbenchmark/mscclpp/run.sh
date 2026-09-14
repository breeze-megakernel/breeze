#!/bin/bash
# run.sh. full 2D sweep (msg_size × concurrency) for the throughput sweep.
#
# Usage: ./run.sh [hostfile] [staging]
#   hostfile: default ~/hostfile (two nodes, slots=8 each)
#   staging:  1 (default) = int4 staging copy per transfer, matching the
#             NVSHMEM bench; 0 = disable, to quantify staging cost.
#
# One invocation produces everything: the binary internally sweeps
# {4KB..4MB} × {1..128} for all three modes and emits CSV.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOSTFILE="${1:-$HOME/hostfile}"
STAGING="${2:-1}"
BIN="${SCRIPT_DIR}/mscclpp_bench"
MSCCLPP_SRC="${SCRIPT_DIR}/mscclpp"
NP="${NP:-16}"
PPN="${PPN:-8}"

export LD_LIBRARY_PATH="${SCRIPT_DIR}/mscclpp-install/lib:${LD_LIBRARY_PATH:-}"

OUT="mscclpp_sweep_$(date +%Y%m%d_%H%M%S).log"

# ── Provenance header ──
{
    echo "# date:      $(date -Is)"
    echo "# mscclpp:   $(git -C "$MSCCLPP_SRC" rev-parse HEAD 2>/dev/null || echo 'source dir missing')"
    echo "# bench:     $(sha256sum "$BIN" | cut -d' ' -f1)"
    echo "# staging:   $STAGING"
    echo "# hostfile:  $HOSTFILE ($NP ranks, $PPN per node)"
} | tee "$OUT"

mpirun -np "$NP" \
    --hostfile "$HOSTFILE" \
    --map-by "ppr:${PPN}:node" --bind-to none \
    -x LD_LIBRARY_PATH -x PATH -x UCX_LOG_LEVEL=error \
    "$BIN" 128 4194304 20 100 "$STAGING" \
    2>&1 | tee -a "$OUT"

echo "Done. Results: $OUT"
echo "Extract CSV:   grep '^[0-9]' $OUT > mscclpp.csv"
