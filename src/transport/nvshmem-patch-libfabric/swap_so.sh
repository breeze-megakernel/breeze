#!/bin/bash
# swap_so.sh [original|patched]. toggle the active Libfabric transport.
# $NVSHMEM_HOME/lib is on Perlmutter's shared home FS: one copy switches
# every node. The staged .original/.patched variants are created by
# build_patched_nvshmem.sh.
set -euo pipefail
VARIANT=${1:-patched}
NVSHMEM_HOME=${NVSHMEM_HOME:-$HOME/nvshmem}
SO=$(ls "$NVSHMEM_HOME"/lib/nvshmem_transport_libfabric.so.*.*.* 2>/dev/null \
     | grep -v '\.original$\|\.patched$' | head -1)
[ -n "$SO" ] || { echo "no transport .so under $NVSHMEM_HOME/lib" >&2; exit 1; }
[ -e "${SO}.${VARIANT}" ] || { echo "missing ${SO}.${VARIANT}. run build_patched_nvshmem.sh" >&2; exit 1; }
cp "${SO}.${VARIANT}" "$SO"
echo "Switched to: ${VARIANT}  ($(sha256sum "$SO" | cut -d' ' -f1))"
