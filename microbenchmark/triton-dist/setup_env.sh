#!/bin/bash
# setup_env.sh. create the Triton-distributed environment (one-time).
#
# TODO(release): two placeholders below must be filled from the original
# measurement environment before publishing:
#   1. TRITON_DIST_COMMIT. `git -C ~/Triton-distributed rev-parse HEAD`
#   2. the install recipe. replace the block marked INSTALL with the exact
#      steps used (Triton-distributed's own install script, pip install -e,
#      pinned torch version, etc.), ideally cross-checked against
#      `conda env export` of the working env.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRITON_DIST_SRC=${TRITON_DIST_SRC:-$SCRIPT_DIR/triton-distributed}
TRITON_DIST_ENV=${TRITON_DIST_ENV:-$SCRATCH/envs/triton-dist}
TRITON_DIST_COMMIT=${TRITON_DIST_COMMIT:-CHANGE-ME}   # <-- pin

[ "$TRITON_DIST_COMMIT" = "CHANGE-ME" ] && {
    echo "ERROR: set TRITON_DIST_COMMIT to the measured commit (see header)"; exit 1; }

# ── Pinned clone (no application changes: the artifact runs upstream
#    Triton-distributed unmodified; only the NVSHMEM .so differs) ──────────
if [ ! -d "$TRITON_DIST_SRC" ]; then
    git clone https://github.com/ByteDance-Seed/Triton-distributed "$TRITON_DIST_SRC"
fi
git -C "$TRITON_DIST_SRC" checkout "$TRITON_DIST_COMMIT"
echo "Triton-distributed @ $(git -C "$TRITON_DIST_SRC" rev-parse HEAD)"

# ── Conda env ──────────────────────────────────────────────────────────────
if [ ! -d "$TRITON_DIST_ENV" ]; then
    conda create -y -p "$TRITON_DIST_ENV" python=3.11   # match measured env
fi
conda activate "$TRITON_DIST_ENV"

# ── INSTALL (replace with the exact measured recipe) ──────────────────────
# pip install torch==<version> --index-url ...
# cd "$TRITON_DIST_SRC" && <its install procedure against $NVSHMEM_HOME>
echo "ERROR: fill in the INSTALL block with the measured recipe"; exit 1
