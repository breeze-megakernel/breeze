#!/bin/bash
# swap_so.sh [original|patched]. activate a staged IBRC transport .so.
# /usr/lib is node-local: run on EVERY node (needs sudo). Prefer the
# runtime toggle where possible: the patched .so behaves stock under
# NVSHMEM_IB_PEERHASH=0, so one swap to 'patched' + the env var covers
# both conditions without further sudo.
set -euo pipefail
VARIANT=${1:-patched}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NVSHMEM_LIB_HOME=${NVSHMEM_LIB_HOME:-/usr/lib/x86_64-linux-gnu/nvshmem/12}
STAGE="$SCRIPT_DIR/staged"

SONAME=$(ls "$STAGE" | grep -o 'nvshmem_transport_ibrc\.so\.[0-9.]*' | sort -u | head -1)
[ -n "$SONAME" ] || { echo "nothing staged. run build_patched_nvshmem.sh" >&2; exit 1; }
[ -e "$STAGE/${SONAME}.${VARIANT}" ] || { echo "missing $STAGE/${SONAME}.${VARIANT}" >&2; exit 1; }

cp "$STAGE/${SONAME}.${VARIANT}" "$NVSHMEM_LIB_HOME/$SONAME"
echo "$(hostname): switched to ${VARIANT}  ($(sha256sum "$NVSHMEM_LIB_HOME/$SONAME" | cut -d' ' -f1))"
