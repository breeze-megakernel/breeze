#!/bin/bash
# Create the Triton-distributed environment.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRITON_DIST_SRC=${TRITON_DIST_SRC:-$SCRIPT_DIR/triton-distributed}
TRITON_DIST_ENV=${TRITON_DIST_ENV:-$SCRATCH/envs/triton-dist}

# Commit used for evaluation.
TRITON_DIST_COMMIT=${TRITON_DIST_COMMIT:-d9f94a33233089faca2b9fa391ee0b5f729ad355}

# Clone and pin Triton-distributed.
if [ ! -d "$TRITON_DIST_SRC" ]; then
    git clone https://github.com/ByteDance-Seed/Triton-distributed "$TRITON_DIST_SRC"
fi

git -C "$TRITON_DIST_SRC" fetch --quiet origin "$TRITON_DIST_COMMIT" 2>/dev/null || true
git -C "$TRITON_DIST_SRC" checkout "$TRITON_DIST_COMMIT"
echo "Triton-distributed @ $(git -C "$TRITON_DIST_SRC" rev-parse HEAD)"

# Create and activate the Conda environment.
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [ ! -d "$TRITON_DIST_ENV" ]; then
    conda create -y -p "$TRITON_DIST_ENV" python=3.11
fi

conda activate "$TRITON_DIST_ENV"

# Install dependencies.
# pip install torch==<version> --index-url ...
# cd "$TRITON_DIST_SRC" && <install against $NVSHMEM_HOME>

echo "ERROR: fill in the install block"
exit 1
