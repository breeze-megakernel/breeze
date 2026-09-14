#!/bin/bash
# build_mscclpp.sh. clone and build MSCCL++ at the pinned commit.
#
# This benchmark requires IB verbs (ConnectX); it does not run on
# Slingshot/CXI, so there is no Perlmutter build path.
#
# Usage: ./build_mscclpp.sh          # GPU_ARCH=90 (H100) by default
#        GPU_ARCH=80 ./build_mscclpp.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MSCCLPP_SRC="${SCRIPT_DIR}/mscclpp"
MSCCLPP_INSTALL="${SCRIPT_DIR}/mscclpp-install"
GPU_ARCH="${GPU_ARCH:-90}"

MSCCLPP_COMMIT="${MSCCLPP_COMMIT:-52f659dac390c9820bdc4db0577faeffe61a7370}"

if [ ! -d "$MSCCLPP_SRC" ]; then
    echo "Cloning MSCCL++ @ ${MSCCLPP_COMMIT}..."
    git clone https://github.com/microsoft/mscclpp.git "$MSCCLPP_SRC"
fi
git -C "$MSCCLPP_SRC" checkout "$MSCCLPP_COMMIT"
echo "MSCCL++ @ $(git -C "$MSCCLPP_SRC" rev-parse HEAD)"

echo "Building MSCCL++ (USE_IB=ON, sm_${GPU_ARCH})..."
mkdir -p "$MSCCLPP_SRC/build"
cd "$MSCCLPP_SRC/build"

cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$MSCCLPP_INSTALL" \
      -DMSCCLPP_USE_IB=ON \
      -DMSCCLPP_GPU_ARCHS="$GPU_ARCH" \
      -DMSCCLPP_BUILD_PYTHON_BINDINGS=OFF \
      -DMSCCLPP_BUILD_TESTS=OFF \
      ..

make -j"$(nproc)"
make install

echo ""
echo "=== MSCCL++ ${MSCCLPP_COMMIT} installed to: $MSCCLPP_INSTALL ==="
ls -lh "$MSCCLPP_INSTALL/lib/"libmscclpp* 2>/dev/null
