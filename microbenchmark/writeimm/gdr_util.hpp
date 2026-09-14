// gdr_util.hpp. BAR1 mapping of GPU memory so a host thread can store
// notification flags directly into HBM.
//
// This is the realistic receive path for a WRITE-WITH-IMM design: the CQE
// lands in host memory, the proxy decodes it, and the flag has to be moved
// into GPU memory where a megakernel subscriber CTA can poll it with a
// single load. gdrcopy is what production implementations use for this hop
// (fabric-lib, UEP); cudaMemcpy would add kernel-launch-scale latency and
// is not a fair representation.
//
// Build without -DUSE_GDRCOPY to get a stub: the imm_gdr mode then reports
// as unavailable, and only imm_host (flag in host memory, GPU polls over
// PCIe) is measured.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef USE_GDRCOPY
#include <gdrapi.h>
#endif

class GdrFlags {
 public:
  bool available() const { return ok_; }

  // gdrcopy pins and maps at 64 KB GPU-page granularity: both the address
  // and the length must be aligned, or gdr_map fails with EINVAL. dptr
  // should already be 64 KB aligned; the containing page is used otherwise.
  static const unsigned long kGpuPage = 1ul << 16;

  bool init(void *dptr, size_t len) {
#ifdef USE_GDRCOPY
    const unsigned long addr = reinterpret_cast<unsigned long>(dptr);
    const unsigned long base = addr & ~(kGpuPage - 1);
    const size_t span = (addr - base) + len;
    const size_t rounded = (span + kGpuPage - 1) & ~(kGpuPage - 1);

    g_ = gdr_open();
    if (!g_) {
      fprintf(stderr, "[warn] gdr_open failed (is gdrdrv loaded?)\n");
      return false;
    }
    int rc = gdr_pin_buffer(g_, base, rounded, 0, 0, &mh_);
    if (rc != 0) {
      fprintf(stderr, "[warn] gdr_pin_buffer failed: rc=%d (%s)\n", rc,
              strerror(rc > 0 ? rc : -rc));
      gdr_close(g_);
      g_ = nullptr;
      return false;
    }
    rc = gdr_map(g_, mh_, &bar_, rounded);
    if (rc != 0) {
      fprintf(stderr,
              "[warn] gdr_map failed: rc=%d (%s) addr=0x%lx len=%zu\n", rc,
              strerror(rc > 0 ? rc : -rc), base, rounded);
      gdr_unpin_buffer(g_, mh_);
      gdr_close(g_);
      g_ = nullptr;
      return false;
    }
    gdr_info_t info {};
    if (gdr_get_info(g_, mh_, &info) != 0) {
      fprintf(stderr, "[warn] gdr_get_info failed\n");
      return false;
    }
    // The mapping starts at the aligned base; correct for the offset.
    off_ = addr - info.va;
    len_ = rounded;
    ok_ = true;
    return true;
#else
    (void)dptr;
    (void)len;
    return false;
#endif
  }

  // Store one 64-bit flag at slot `idx` into GPU memory.
  // gdr_copy_to_mapping issues a write-combined store to BAR1 and flushes.
  inline void store(size_t idx, uint64_t val) {
#ifdef USE_GDRCOPY
    if (!ok_) return;
    uint8_t *dst = static_cast<uint8_t *>(bar_) + off_ + idx * sizeof(uint64_t);
    gdr_copy_to_mapping(mh_, dst, &val, sizeof(val));
#else
    (void)idx;
    (void)val;
#endif
  }

  void destroy() {
#ifdef USE_GDRCOPY
    if (!ok_) return;
    gdr_unmap(g_, mh_, bar_, len_);
    gdr_unpin_buffer(g_, mh_);
    gdr_close(g_);
    ok_ = false;
#endif
  }

 private:
  bool ok_ = false;
#ifdef USE_GDRCOPY
  gdr_t g_ = nullptr;
  gdr_mh_t mh_ {};
  void *bar_ = nullptr;
  size_t len_ = 0;
  size_t off_ = 0;
#endif
};