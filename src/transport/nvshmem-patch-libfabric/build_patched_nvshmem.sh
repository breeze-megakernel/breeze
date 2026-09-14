#!/bin/bash
# Build NVSHMEM v3.4.5 for Libfabric.
# twice (stock, then patched) and stage both transport .so variants
# swap_so.sh toggles the active variant.
#
# Usage:
#   module load PrgEnv-gnu cudatoolkit cmake
#   ./build_patched_nvshmem.sh          # installs to $NVSHMEM_HOME (~/nvshmem)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NVSHMEM_HOME=${NVSHMEM_HOME:-$HOME/nvshmem}
NVSHMEM_TAG="v3.4.5-0"                      # the Libfabric patch base
SRC=${SRC:-$SCRIPT_DIR/nvshmem}
PATCH="$SCRIPT_DIR/breeze-libfabric-v3.4.5.patch"
CUDA_HOME=${CUDA_HOME:?load the cudatoolkit module or set CUDA_HOME}
MPI_HOME=${MPI_HOME:-${CRAY_MPICH_DIR:?load cray-mpich or set MPI_HOME}}
LIBFABRIC_HOME=${LIBFABRIC_HOME:-${LIBFABRIC_DIR:-$(ls -d /opt/cray/libfabric/* 2>/dev/null | tail -1)}}

# TODO(verify): replace with the exact cmake flags of the measured build if
# they differ. these are the standard Perlmutter/libfabric NVSHMEM options.
CMAKE_FLAGS=(
    -DCMAKE_INSTALL_PREFIX="$NVSHMEM_HOME"
    -DCUDA_HOME="$CUDA_HOME"
    -DNVSHMEM_LIBFABRIC_SUPPORT=1
    -DLIBFABRIC_HOME="$LIBFABRIC_HOME"
    -DNVSHMEM_MPI_SUPPORT=1
    -DMPI_HOME="$MPI_HOME"
    -DNVSHMEM_PMIX_SUPPORT=0
    -DNVSHMEM_IBGDA_SUPPORT=0
    -DNVSHMEM_IBRC_SUPPORT=0
    -DNVSHMEM_BUILD_TESTS=0
    -DNVSHMEM_BUILD_EXAMPLES=0
)

if [ ! -d "$SRC" ]; then
    git clone --branch "$NVSHMEM_TAG" --depth 1 https://github.com/NVIDIA/nvshmem "$SRC"
fi
git -C "$SRC" checkout -- .                 # pristine before any build
echo "NVSHMEM @ $(git -C "$SRC" describe --tags)"

# ── Pass 1: stock build ──
cmake -S "$SRC" -B "$SRC/build" "${CMAKE_FLAGS[@]}"
cmake --build "$SRC/build" -j"$(nproc)"
cmake --install "$SRC/build"

SO=$(ls "$NVSHMEM_HOME"/lib/nvshmem_transport_libfabric.so.*.*.* | head -1)
cp "$SO" "${SO}.original"
echo "stock transport staged: ${SO}.original"

# Pass 2: apply the patch, rebuild the transport, and stage the patched .so
( cd "$SRC" && patch -p1 < "$PATCH" )
cmake --build "$SRC/build" -j"$(nproc)"
cmake --install "$SRC/build"
cp "$SO" "${SO}.patched"
git -C "$SRC" checkout -- .                 # leave the tree pristine
echo "patched transport staged: ${SO}.patched"

# Default active transport is original. Swap explicitly for patched runs.
cp "${SO}.original" "$SO"
echo ""
echo "Installed to $NVSHMEM_HOME. Variants:"
for v in original patched; do
    echo "  $v: $(sha256sum "${SO}.$v" | cut -d' ' -f1)"
done
echo "Toggle with: ./swap_so.sh [original|patched]"
