#!/bin/bash
# build.sh. build the put vs signal microbenchmark.
#
# Perlmutter (A100, default):
#   module load PrgEnv-gnu cudatoolkit
#   NVSHMEM_HOME=$HOME/nvshmem ./build.sh
#
# H100 / InfiniBand cloud:
#   GPU_ARCH=90 NVSHMEM_HOME=/usr/lib/x86_64-linux-gnu/nvshmem/12 \
#   MPI_HOME=/usr/mpi/gcc/openmpi-4.1.7rc1 MPI_LIB=-lmpi ./build.sh
#
# All locations are overridable via env; defaults target Perlmutter with
# the standard modules loaded (cudatoolkit exports CUDA_HOME, cray-mpich
# exports CRAY_MPICH_DIR).
set -euo pipefail

CUDA_HOME=${CUDA_HOME:?load the cudatoolkit module or set CUDA_HOME}
MPI_HOME=${MPI_HOME:-${CRAY_MPICH_DIR:?load cray-mpich or set MPI_HOME}}
NVSHMEM_HOME=${NVSHMEM_HOME:-$HOME/nvshmem}
GPU_ARCH=${GPU_ARCH:-80}            # 80 = A100 (Perlmutter), 90 = H100
HOST_CXX=${HOST_CXX:-g++}           # PrgEnv-gnu g++ matches cray-mpich's ABI

# The MPI library name tracks the vendor and toolchain and has changed
# across NERSC maintenances (cray-mpich 9.0.x: libmpi_gnu_123, 9.1.x:
# libmpi_gnu). Detect rather than assume; override with MPI_LIB for
# OpenMPI (-lmpi) or unusual layouts.
if [ -z "${MPI_LIB:-}" ]; then
    if   [ -e "$MPI_HOME/lib/libmpi_gnu_123.so" ]; then MPI_LIB=-lmpi_gnu_123
    elif [ -e "$MPI_HOME/lib/libmpi_gnu.so" ];     then MPI_LIB=-lmpi_gnu
    else                                                 MPI_LIB=-lmpi
    fi
fi

CUDACXX=${CUDACXX:-$CUDA_HOME/bin/nvcc}

echo "CUDA_HOME=$CUDA_HOME"
echo "MPI_HOME=$MPI_HOME"
echo "NVSHMEM_HOME=$NVSHMEM_HOME"
echo "GPU_ARCH=sm_$GPU_ARCH  HOST_CXX=$HOST_CXX  MPI_LIB=$MPI_LIB"

${CUDACXX} -o put_signal_bench put_signal_bench.cu \
    -I"${NVSHMEM_HOME}"/include \
    -I"${MPI_HOME}"/include \
    -L"${NVSHMEM_HOME}"/lib -lnvshmem_host -lnvshmem_device \
    -L"${MPI_HOME}"/lib ${MPI_LIB} \
    -L"${CUDA_HOME}"/lib64/stubs -lcuda \
    -lcudart \
    -rdc=true \
    -gencode arch=compute_${GPU_ARCH},code=sm_${GPU_ARCH} \
    -Xlinker -rpath="${NVSHMEM_HOME}"/lib \
    -Xlinker -rpath="${MPI_HOME}"/lib \
    -Xlinker --allow-shlib-undefined \
    --compiler-bindir="${HOST_CXX}"

echo "Built: put_signal_bench"
echo ""
echo "Run examples:"
echo "  # Default: 96 transfers, 2MB each, 107 blocks (matches FlashMoE 4-node)"
echo "  srun -n 2 -N 2 --gpus-per-node=1 ./put_signal_bench"
echo ""
echo "  # Custom: 128 transfers, 2MB, 20 warmup, 100 iters, signal op explicit"
echo "  srun -n 8 -N 2 --gpus-per-node=4 ./put_signal_bench 96 2097152 20 100 96 set"
