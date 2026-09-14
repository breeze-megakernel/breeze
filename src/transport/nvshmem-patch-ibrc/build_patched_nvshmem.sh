#!/bin/bash
# build_patched_nvshmem.sh. Build the patched IBRC transport
# transport .so from NVSHMEM v3.5.21 source and stage it next to the
# stock deb-installed one (bootstrap_node.sh installs the deb).
#
# Run on ONE node, then copy the staged .so files to the others (or run
# everywhere). /usr/lib is node-local, unlike Perlmutter's shared home.
#
# Usage: ./build_patched_nvshmem.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NVSHMEM_TAG="v3.5.21-0"                     # the IBRC patch base
SRC=${SRC:-$SCRIPT_DIR/nvshmem}
PATCH="$SCRIPT_DIR/breeze-ibrc-v3.5.21.patch"
NVSHMEM_LIB_HOME=${NVSHMEM_LIB_HOME:-/usr/lib/x86_64-linux-gnu/nvshmem/12}
CUDA_HOME=${CUDA_HOME:-$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)}
STAGE="$SCRIPT_DIR/staged"

# TODO(verify): replace with the exact cmake flags of the measured build if
# they differ. these are the standard verbs/IBRC NVSHMEM options.
CMAKE_FLAGS=(
    -DCUDA_HOME="$CUDA_HOME"
    -DNVSHMEM_IBRC_SUPPORT=1
    -DNVSHMEM_IBGDA_SUPPORT=1
    -DNVSHMEM_LIBFABRIC_SUPPORT=0
    -DNVSHMEM_MPI_SUPPORT=1
    -DNVSHMEM_BUILD_TESTS=0
    -DNVSHMEM_BUILD_EXAMPLES=0
)

if [ ! -d "$SRC" ]; then
    git clone --branch "$NVSHMEM_TAG" --depth 1 https://github.com/NVIDIA/nvshmem "$SRC"
fi
git -C "$SRC" checkout -- .
( cd "$SRC" && patch -p1 < "$PATCH" )
echo "NVSHMEM @ $(git -C "$SRC" describe --tags) + breeze-ibrc patch"

cmake -S "$SRC" -B "$SRC/build" "${CMAKE_FLAGS[@]}"
cmake --build "$SRC/build" -j"$(nproc)"
git -C "$SRC" checkout -- .

# ── Stage: patched from the build tree, original from the deb install ──
mkdir -p "$STAGE"
BUILT=$(find "$SRC/build" -name 'nvshmem_transport_ibrc.so.*' -type f | head -1)
[ -n "$BUILT" ] || { echo "built transport .so not found under $SRC/build" >&2; exit 1; }
SONAME=$(basename "$BUILT")
cp "$BUILT" "$STAGE/${SONAME}.patched"
cp "$NVSHMEM_LIB_HOME/$SONAME" "$STAGE/${SONAME}.original"

echo ""
echo "Staged in $STAGE:"
for v in original patched; do
    echo "  $v: $(sha256sum "$STAGE/${SONAME}.$v" | cut -d' ' -f1)"
done
echo ""
echo "Activate (needs sudo, on EVERY node):  sudo ./swap_so.sh [original|patched]"
echo "Runtime A/B without swapping: the patched .so honors NVSHMEM_IB_PEERHASH=0/1."
