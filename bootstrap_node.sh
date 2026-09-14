#!/bin/bash
# ─────────────────────────────────────────────────────────────
# bootstrap_node.sh. Set up a bare GPU node (InfiniBand cloud platform)
#
# Run this on EACH node. It installs:
#   1. System packages (build tools, IB drivers, etc.)
#   2. OpenMPI + ninja
#   3. GDRCopy (check only)
#   4. NVSHMEM 3.5.21 via local repo deb (the IBRC patch base. see
#      src/nic-ordering/nvshmem-patch-ibrc/)
#   5. cuBLASDx / MathDx
#   6. Environment file
#   7. Hostfile
#
# Usage:
#   NODE_RANK=0 PEER_IPS="<node1-ip>" ./bootstrap_node.sh
#   NODE_RANK=1 PEER_IPS="<node0-ip>" ./bootstrap_node.sh
#
# Environment overrides:
#   CUDA_HOME          (default: auto-detect /usr/local/cuda-*)
#   GPUS_PER_NODE      (default: 8)
#   WORKSPACE          (default: ~/)
#   SKIP_SYSTEM_PKGS   (default: 0, set 1 to skip apt)
#   SKIP_NVSHMEM       (default: 0)
#   SKIP_MATHDX        (default: 0)
# ─────────────────────────────────────────────────────────────
set -euxo pipefail

# ── Resolve real user home (works whether invoked via sudo or not) ──
if [ -n "${SUDO_USER:-}" ]; then
    REAL_HOME=$(eval echo "~${SUDO_USER}")
else
    REAL_HOME="$HOME"
fi

# ── Configuration ────────────────────────────────────────────

NODE_RANK=${NODE_RANK:-0}
PEER_IPS=${PEER_IPS:-""}              # comma-separated list of peer IPs
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
WORKSPACE=${WORKSPACE:-${REAL_HOME}}

SKIP_SYSTEM_PKGS=${SKIP_SYSTEM_PKGS:-0}
SKIP_NVSHMEM=${SKIP_NVSHMEM:-0}
SKIP_MATHDX=${SKIP_MATHDX:-0}

# Auto-detect CUDA_HOME if not set
if [ -z "${CUDA_HOME:-}" ]; then
    CUDA_HOME=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)
    if [ -z "$CUDA_HOME" ] && [ -d /usr/local/cuda ]; then
        CUDA_HOME=/usr/local/cuda
    fi
fi

NVSHMEM_VERSION="3.5.21"
NVSHMEM_DEB_URL="https://developer.download.nvidia.com/compute/nvshmem/${NVSHMEM_VERSION}/local_installers/nvshmem-local-repo-ubuntu2404-${NVSHMEM_VERSION}_${NVSHMEM_VERSION}-1_amd64.deb"
NVSHMEM_LIB_HOME=/usr/lib/x86_64-linux-gnu/nvshmem/12

MATHDX_VERSION="25.12.1"
MATHDX_TARBALL="nvidia-mathdx-${MATHDX_VERSION}-cuda12.tar.gz"
MATHDX_URL="https://developer.nvidia.com/downloads/compute/cublasdx/redist/cublasdx/cuda12/${MATHDX_TARBALL}"
MATHDX_DIR="${REAL_HOME}/.local/nvidia-mathdx-${MATHDX_VERSION}-cuda12"
MATHDX_ROOT="${MATHDX_DIR}/nvidia/mathdx/25.12"

echo "══════════════════════════════════════════════════════════"
echo "  Node Bootstrap (InfiniBand cloud platform)"
echo "  Node rank   : ${NODE_RANK}"
echo "  Hostname    : $(hostname)"
echo "  Real user   : ${SUDO_USER:-$(whoami)}"
echo "  Real home   : ${REAL_HOME}"
echo "  Peer IPs    : ${PEER_IPS:-none}"
echo "  CUDA_HOME   : ${CUDA_HOME:-NOT FOUND}"
echo "  GPUs/node   : ${GPUS_PER_NODE}"
echo "══════════════════════════════════════════════════════════"


# ═══════════════════════════════════════════════════════════════
# 1. System packages (needs sudo)
# ═══════════════════════════════════════════════════════════════

if [ "$SKIP_SYSTEM_PKGS" = "0" ]; then
    echo ">>> Installing system packages"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y \
        build-essential cmake ninja-build \
        git wget curl zip unzip \
        openssh-server openssh-client \
        libibverbs-dev librdmacm-dev rdma-core \
        ibverbs-utils infiniband-diags perftest \
        libnuma-dev numactl \
        autoconf automake libtool pkg-config \
        python3 python3-pip python3-yaml
else
    echo ">>> Skipping system packages (SKIP_SYSTEM_PKGS=1)"
fi


# ═══════════════════════════════════════════════════════════════
# 2. Verify CUDA
# ═══════════════════════════════════════════════════════════════

echo ">>> Checking CUDA"
test -x "${CUDA_HOME}/bin/nvcc"
${CUDA_HOME}/bin/nvcc --version
nvidia-smi -L


# ═══════════════════════════════════════════════════════════════
# 3. OpenMPI (needs sudo)
# ═══════════════════════════════════════════════════════════════

echo ">>> Checking OpenMPI"
if ! command -v mpirun &>/dev/null; then
    echo ">>> Installing OpenMPI"
    apt-get update
    apt-get install -y openmpi-bin libopenmpi-dev
fi
mpirun --version


# ═══════════════════════════════════════════════════════════════
# 4. GDRCopy + InfiniBand checks (needs sudo for build/install)
# ═══════════════════════════════════════════════════════════════

echo ">>> Checking / installing GDRCopy"
if ldconfig -p | grep -q libgdrapi; then
    echo ">>> GDRCopy already installed"
    ldconfig -p | grep libgdrapi
else
    echo ">>> Building GDRCopy from source"
    cd /tmp
    if [ ! -d gdrcopy ]; then
        git clone https://github.com/NVIDIA/gdrcopy.git
    fi
    cd gdrcopy
    make -j"$(nproc)" prefix=/usr/local lib lib_install
    ldconfig
    ldconfig -p | grep libgdrapi
    cd /tmp
fi

echo ">>> Checking InfiniBand"
ibv_devinfo || echo "WARNING: No IB devices found"

echo ">>> Checking nvidia_peermem"
lsmod | grep -E "nvidia_peermem|nv_peer_mem" || modprobe nvidia_peermem || echo "WARNING: nvidia_peermem not available"


# ═══════════════════════════════════════════════════════════════
# 5. NVSHMEM via local repo deb (needs sudo)
# ═══════════════════════════════════════════════════════════════

if [ "$SKIP_NVSHMEM" = "0" ]; then
    if [ -d "${NVSHMEM_LIB_HOME}" ] && ls "${NVSHMEM_LIB_HOME}"/libnvshmem* &>/dev/null; then
        echo ">>> NVSHMEM already installed at ${NVSHMEM_LIB_HOME}"
        ls -la "${NVSHMEM_LIB_HOME}"/libnvshmem*
    else
        echo ">>> Installing NVSHMEM ${NVSHMEM_VERSION}"
        cd /tmp
        wget "${NVSHMEM_DEB_URL}"
        dpkg -i "nvshmem-local-repo-ubuntu2404-${NVSHMEM_VERSION}_${NVSHMEM_VERSION}-1_amd64.deb"
        cp /var/nvshmem-local-repo-ubuntu2404-${NVSHMEM_VERSION}/nvshmem-*-keyring.gpg /usr/share/keyrings/
        apt-get update
        apt-get install -y \
            nvshmem-cuda-12=${NVSHMEM_VERSION}-1 \
            libnvshmem3-cuda-12=${NVSHMEM_VERSION}-1 \
            libnvshmem3-dev-cuda-12=${NVSHMEM_VERSION}-1 \
            libnvshmem3-static-cuda-12=${NVSHMEM_VERSION}-1
        ls -la "${NVSHMEM_LIB_HOME}"/libnvshmem*
        rm -f "/tmp/nvshmem-local-repo-ubuntu2404-${NVSHMEM_VERSION}_${NVSHMEM_VERSION}-1_amd64.deb"
    fi
else
    echo ">>> Skipping NVSHMEM install (SKIP_NVSHMEM=1)"
fi

# Quick NVSHMEM sanity: check perftest binaries exist
ls /usr/bin/nvshmem_12/perftest/host/init/malloc || echo "WARNING: NVSHMEM perftest binaries not found"


# ═══════════════════════════════════════════════════════════════
# 6. cuBLASDx / MathDx (install to user home, no sudo needed)
# ═══════════════════════════════════════════════════════════════

if [ "$SKIP_MATHDX" = "0" ]; then
    if [ -d "${MATHDX_ROOT}" ]; then
        echo ">>> MathDx already present at ${MATHDX_ROOT}"
    else
        echo ">>> Downloading and extracting MathDx ${MATHDX_VERSION}"
        mkdir -p "${REAL_HOME}/.local"
        cd "${REAL_HOME}/.local"
        wget "${MATHDX_URL}"
        tar xvf "${MATHDX_TARBALL}"
        rm -f "${MATHDX_TARBALL}"
        ls -la "${MATHDX_ROOT}"
        cd -
    fi
    # Fix ownership if running under sudo
    if [ -n "${SUDO_USER:-}" ]; then
        chown -R "${SUDO_USER}:$(id -gn ${SUDO_USER})" "${REAL_HOME}/.local"
    fi
else
    echo ">>> Skipping MathDx install (SKIP_MATHDX=1)"
fi


# ═══════════════════════════════════════════════════════════════
# 7. Hostfile (user-owned)
# ═══════════════════════════════════════════════════════════════

echo ">>> Generating hostfile"
HOSTFILE="${WORKSPACE}/hostfile"

MY_IP=$(hostname -I | awk '{print $1}')
echo "${MY_IP} slots=${GPUS_PER_NODE}" > "${HOSTFILE}"

if [ -n "${PEER_IPS}" ]; then
    IFS=',' read -ra PEERS <<< "${PEER_IPS}"
    for peer in "${PEERS[@]}"; do
        [ -n "$peer" ] && echo "${peer} slots=${GPUS_PER_NODE}" >> "${HOSTFILE}"
    done
fi

echo ">>> Hostfile contents:"
cat "${HOSTFILE}"


# ═══════════════════════════════════════════════════════════════
# 8. Environment file (user-owned)
# ═══════════════════════════════════════════════════════════════

echo ">>> Writing environment file"

NODES=$(grep -c 'slots=' "${HOSTFILE}" 2>/dev/null || echo 1)

ENV_FILE="${WORKSPACE}/breeze_env.sh"
cat > "${ENV_FILE}" <<ENVEOF
#!/bin/bash
# Source this file: source ${ENV_FILE}

# CUDA
export CUDA_HOME=${CUDA_HOME}
export PATH=\${CUDA_HOME}/bin:\${PATH}
export LD_LIBRARY_PATH=\${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH:-}

# NVSHMEM
export NVSHMEM_LIB_HOME=${NVSHMEM_LIB_HOME}
export LD_LIBRARY_PATH=\${NVSHMEM_LIB_HOME}:\${LD_LIBRARY_PATH}
export CMAKE_PREFIX_PATH=\${NVSHMEM_LIB_HOME}:\${CMAKE_PREFIX_PATH:-}

# NVSHMEM runtime. critical for multi-node
export NVSHMEM_BOOTSTRAP=MPI
export NVSHMEM_SYMMETRIC_SIZE=\${NVSHMEM_SYMMETRIC_SIZE:-17179869184}  # 16GB default

# Submission path: 1 = IBGDA (GPU-submitted reference point),
#                  0 = proxy-submitted IBRC.
# Every IBRC experiment must export 0 explicitly (the run scripts do).
export NVSHMEM_IB_ENABLE_IBGDA=\${NVSHMEM_IB_ENABLE_IBGDA:-1}

# For write-based signals on IBRC,
# carrying a packed payload), which the NIC fence only orders when PCIe
# relaxed ordering is disabled (the relaxed-ordering requirement footnote).
export NVSHMEM_IB_ENABLE_RELAXED_ORDERING=\${NVSHMEM_IB_ENABLE_RELAXED_ORDERING:-0}

# Multi-QP configuration used in the IBRC evaluation.
# NVSHMEM's default is 1, at which the patched fence path never engages.
export NVSHMEM_IB_NUM_RC_PER_DEVICE=\${NVSHMEM_IB_NUM_RC_PER_DEVICE:-4}

# MathDx / cuBLASDx
export MATHDX_ROOT=${MATHDX_ROOT}
export CMAKE_PREFIX_PATH=\${MATHDX_ROOT}:\${CMAKE_PREFIX_PATH}

# NCCL
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_DEBUG=WARN

# Cluster shape
export BREEZE_GPUS_PER_NODE=${GPUS_PER_NODE}
export BREEZE_SCRATCH=${WORKSPACE}
export BREEZE_HOSTFILE=${HOSTFILE}
export BREEZE_NODES=${NODES}
ENVEOF

chmod +x "${ENV_FILE}"

# ── Fix ownership of generated files if running under sudo ──
if [ -n "${SUDO_USER:-}" ]; then
    chown "${SUDO_USER}:$(id -gn ${SUDO_USER})" "${ENV_FILE}" "${HOSTFILE}"
fi

echo ">>> Environment file written to: ${ENV_FILE}"
cat "${ENV_FILE}"


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Bootstrap complete for node ${NODE_RANK}"
echo "══════════════════════════════════════════════════════════"
echo ""
echo "  All user files are in: ${REAL_HOME}/"
echo "  (hostfile, breeze_env.sh. owned by ${SUDO_USER:-$(whoami)}, no sudo needed)"
echo ""
echo "  Next steps:"
echo "    1. Run this script on ALL other nodes"
echo "    2. Set up passwordless SSH between nodes"
echo "    3. Source the env file (no sudo!):"
echo "         source ${ENV_FILE}"
echo "    4. Verify NVSHMEM with perftest:"
echo "         NVSHMEM_DEBUG=INFO mpirun --allow-run-as-root -np 2 --hostfile ${HOSTFILE} \\"
echo "           --map-by ppr:1:node --bind-to none \\"
echo "           -x CUDA_HOME -x LD_LIBRARY_PATH \\"
echo "           -x NVSHMEM_BOOTSTRAP -x NVSHMEM_SYMMETRIC_SIZE \\"
echo "           -x NVSHMEM_IB_ENABLE_IBGDA -x NVSHMEM_IB_ENABLE_RELAXED_ORDERING \\"
echo "           -x NVSHMEM_IB_NUM_RC_PER_DEVICE -x NVSHMEM_DEBUG \\"
echo "           /usr/bin/nvshmem_12/perftest/host/pt-to-pt/bw"
echo "    5. Build the microbenchmarks:"
echo "         cd microbenchmark/put-fence-signal && GPU_ARCH=90 \\"
echo "           NVSHMEM_HOME=${NVSHMEM_LIB_HOME} MPI_LIB=-lmpi ./build.sh"
echo ""
