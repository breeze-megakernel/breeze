#!/bin/bash
# build.sh. build the MSCCL++ putWithSignal microbenchmark.
# Prereq: ./build_mscclpp.sh
#
# Usage: ./build.sh                      # H100 (sm_90), OpenMPI via mpicxx
#        GPU_ARCH=80 ./build.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MSCCLPP_INSTALL="${SCRIPT_DIR}/mscclpp-install"
GPU_ARCH="${GPU_ARCH:-90}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="${CUDA_HOME}/bin/nvcc"

if [ ! -d "$MSCCLPP_INSTALL/include/mscclpp" ]; then
    echo "ERROR: MSCCL++ not found at $MSCCLPP_INSTALL. run ./build_mscclpp.sh first"
    exit 1
fi

MPI_CFLAGS="$(mpicxx --showme:compile 2>/dev/null || echo "-I/usr/include/mpi")"
MPI_LDFLAGS="$(mpicxx --showme:link 2>/dev/null || echo "-lmpi")"

echo "NVCC=$NVCC  GPU_ARCH=sm_${GPU_ARCH}  MSCCLPP=$MSCCLPP_INSTALL"

"$NVCC" -std=c++17 \
    -gencode arch=compute_${GPU_ARCH},code=sm_${GPU_ARCH} \
    -O3 --expt-relaxed-constexpr \
    -I"${MSCCLPP_INSTALL}/include" \
    -I"${CUDA_HOME}/include" \
    ${MPI_CFLAGS} \
    -o "${SCRIPT_DIR}/mscclpp_bench" \
    "${SCRIPT_DIR}/mscclpp_bench.cu" \
    -L"${MSCCLPP_INSTALL}/lib" -lmscclpp \
    -L"${CUDA_HOME}/lib64" -lcudart \
    ${MPI_LDFLAGS} \
    -Xlinker -rpath -Xlinker "${MSCCLPP_INSTALL}/lib"

echo "Built: mscclpp_bench"
