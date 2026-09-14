#!/bin/bash
# build_writeimm_bench.sh. WRITE-with-IMM vs PUT-WITH-SIGNAL characterization
#
# Backends, auto-selected:
#   libfabric   Slingshot-11 / Cassini (Perlmutter, CXI provider)
#   libibverbs  ConnectX-7 / InfiniBand
# Force with: TRANSPORT=verbs|fabric ./build_writeimm_bench.sh
set -euo pipefail

# ---------------------------------------------------------------------------
# Platform
# ---------------------------------------------------------------------------
if [ -n "${CRAY_MPICH_DIR:-}" ] || [ -d /opt/cray/pe/mpich ]; then
    CRAY=1
    MPI_HOME=${MPI_HOME:-${CRAY_MPICH_DIR:-$(ls -d /opt/cray/pe/mpich/*/ofi/gnu/* 2>/dev/null | tail -1)}}
    if   [ -e "$MPI_HOME/lib/libmpi_gnu_123.so" ]; then MPI_LIB=${MPI_LIB:--lmpi_gnu_123}
    elif [ -e "$MPI_HOME/lib/libmpi_gnu.so" ];     then MPI_LIB=${MPI_LIB:--lmpi_gnu}
    else                                                 MPI_LIB=${MPI_LIB:--lmpi}
    fi
    CUDA_HOME=${CUDA_HOME:-$(ls -d /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/12.* 2>/dev/null | tail -1)}
    ARCH=${ARCH:-80}
    DEFAULT_TRANSPORT=fabric
    echo "Cray MPICH detected: ${MPI_HOME}"
else
    CRAY=0
    MPI_HOME=${MPI_HOME:-/usr/mpi/gcc/openmpi-4.1.7a1}
    MPI_LIB=${MPI_LIB:--lmpi}
    CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
    ARCH=${ARCH:-90}
    DEFAULT_TRANSPORT=verbs
    echo "MPI: ${MPI_HOME}"
fi

TRANSPORT=${TRANSPORT:-$DEFAULT_TRANSPORT}
NVCC=${CUDA_HOME}/bin/nvcc

[ -f "${MPI_HOME}/include/mpi.h" ] || {
    echo "ERROR: mpi.h not under ${MPI_HOME}/include"
    echo "  module load cray-mpich && export MPI_HOME=\$CRAY_MPICH_DIR"
    exit 1
}

# ---------------------------------------------------------------------------
# Host compiler.
#
# The conda toolchain's sysroot has an older glibc than the one Cray MPICH and
# libfabric were built against, which produces a wall of
# 'undefined reference to X@GLIBC_2.3x' at link time. Two ways out:
#   (a) gcc-native, which matches the system glibc -- preferred
#   (b) conda g++ plus --allow-shlib-undefined, so undefined symbols coming
#       from shared libraries are deferred to runtime. This is what
#       build_put_signal_bench.sh already does.
# ---------------------------------------------------------------------------
if [ -n "${HOST_CXX:-}" ]; then
    :
elif [ -n "${CRAY_GCC_PREFIX:-}" ] && [ -x "${CRAY_GCC_PREFIX}/bin/g++" ]; then
    HOST_CXX=${CRAY_GCC_PREFIX}/bin/g++
elif [ -x /opt/cray/pe/gcc-native/12/bin/g++ ]; then
    HOST_CXX=/opt/cray/pe/gcc-native/12/bin/g++
elif command -v x86_64-conda-linux-gnu-c++ >/dev/null 2>&1; then
    HOST_CXX=$(command -v x86_64-conda-linux-gnu-c++)
else
    HOST_CXX=$(command -v g++)
fi
echo "Host compiler: ${HOST_CXX}"
case "$HOST_CXX" in
    *conda*) echo "  (conda toolchain: deferring shared-library symbols to runtime)";;
esac

# Quiet the 'needed by ... not found' warnings where we can find the libs.
RPATH_LINK=""
add_rpath_link() { [ -d "$1" ] && RPATH_LINK="${RPATH_LINK} -Xlinker -rpath-link=$1" || true; }
for d in ${CRAY_PMI_PREFIX:-}/lib /opt/cray/pe/pmi/*/lib \
         /opt/cray/pe/lib64 /opt/cray/libfabric/*/lib64 \
         "$(dirname "$HOST_CXX")/../lib" /usr/lib64; do
    add_rpath_link "$d"
done

# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
if [ "$TRANSPORT" = "fabric" ]; then
    LIBFABRIC_HOME=${LIBFABRIC_HOME:-${LIBFABRIC_DIR:-/opt/cray/libfabric/1.22.0}}
    [ -f "${LIBFABRIC_HOME}/include/rdma/fabric.h" ] || {
        echo "ERROR: libfabric headers not under ${LIBFABRIC_HOME}/include"
        echo "  module load libfabric && export LIBFABRIC_HOME=\$LIBFABRIC_DIR"
        exit 1
    }
    TRANSPORT_FLAGS="-DUSE_LIBFABRIC -I${LIBFABRIC_HOME}/include -L${LIBFABRIC_HOME}/lib64 -lfabric"
    TRANSPORT_RPATH="-Xlinker -rpath=${LIBFABRIC_HOME}/lib64"
    echo "Transport: libfabric (CXI)"
    command -v fi_info >/dev/null 2>&1 && ! fi_info -p cxi >/dev/null 2>&1 && \
        echo "WARNING: no CXI provider visible from this node." || true
else
    [ -f /usr/include/infiniband/verbs.h ] || {
        echo "ERROR: libibverbs headers not found (install rdma-core-devel)."
        exit 1
    }
    TRANSPORT_FLAGS="-libverbs"
    TRANSPORT_RPATH=""
    echo "Transport: libibverbs"
fi

# ---------------------------------------------------------------------------
# gdrcopy. search the usual Perlmutter locations
# ---------------------------------------------------------------------------
GDR_FLAGS=""
if [ -z "${GDRCOPY_HOME:-}" ]; then
    for c in /usr /usr/local/gdrcopy "$HOME/gdrcopy" "$HOME/nvshmem" \
             ${NVSHMEM_HOME:-/nonexistent} ${GDRCOPY_DIR:-/nonexistent} \
             /global/common/software/nersc9/gdrcopy/*; do
        [ -f "$c/include/gdrapi.h" ] && { GDRCOPY_HOME=$c; break; }
    done
fi
if [ -n "${GDRCOPY_HOME:-}" ] && [ -f "${GDRCOPY_HOME}/include/gdrapi.h" ]; then
    GDRLIB=${GDRCOPY_HOME}/lib64; [ -d "$GDRLIB" ] || GDRLIB=${GDRCOPY_HOME}/lib
    echo "gdrcopy: ${GDRCOPY_HOME} (imm_gdr enabled)"
    GDR_FLAGS="-DUSE_GDRCOPY -I${GDRCOPY_HOME}/include -L${GDRLIB} -lgdrapi -Xlinker -rpath=${GDRLIB}"
else
    echo "gdrcopy: not found -> imm_gdr unavailable, only imm_host measured."
    echo "  Try: find \$HOME /usr -name gdrapi.h 2>/dev/null | head"
    echo "  Without it the WriteImm receive path is measured only in its"
    echo "  slower host-memory-flag variant, which overstates its cost."
fi

# ---------------------------------------------------------------------------
${NVCC} -O3 -std=c++17 -o writeimm_bench writeimm_bench.cu \
    -I${MPI_HOME}/include \
    -L${MPI_HOME}/lib ${MPI_LIB} \
    ${TRANSPORT_FLAGS} \
    -lcudart -lpthread \
    ${GDR_FLAGS} \
    -gencode arch=compute_${ARCH},code=sm_${ARCH} \
    -Xcompiler -fPIC -Xcompiler -fpermissive \
    -Xlinker --allow-shlib-undefined \
    -Xlinker -rpath=${MPI_HOME}/lib ${TRANSPORT_RPATH} ${RPATH_LINK} \
    --compiler-bindir=${HOST_CXX}

echo ""
echo "Built: writeimm_bench (${TRANSPORT})"
echo ""
if [ "$TRANSPORT" = "fabric" ]; then
cat <<'NOTES'
Perlmutter run (4 GPUs/node):
  srun -n 16 -N 4 --gpus-per-node=4 ./writeimm_bench \
       --n 96 --msg 262144 --local-size 4

  export FI_CXI_DEFAULT_CQ_SIZE=131072
  export FI_HMEM_CUDA_USE_GDRCOPY=1
  export MPICH_GPU_SUPPORT_ENABLED=1

If fi_mr_regattr fails, check GPU registration support:
  fi_info -p cxi -v | grep -i hmem
NOTES
else
cat <<'NOTES'
ConnectX-7 run:
  srun -n 16 -N 2 --gpus-per-node=8 ./writeimm_bench \
       --n 96 --msg 262144 --hca mlx5_0 --local-size 8

  lsmod | grep -E 'nvidia_peermem|gdrdrv'
  nvidia-smi topo -m
NOTES
fi