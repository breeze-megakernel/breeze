// fabric_util.hpp. libfabric backend (HPE Slingshot-11 / Cassini, CXI provider).
//
// Same interface as verbs_util.hpp; drop-in via -DUSE_LIBFABRIC.
//
// Three CXI properties make this cleaner than the verbs path, all confirmed
// by `fi_info -p cxi -v`:
//
//   cq_data_size: 8   64-bit remote CQ data, so the immediate can carry a
//                     full signal value (verbs gives 32 bits).
//   FI_RMA_EVENT      fi_writedata raises a target completion WITHOUT
//                     consuming a posted receive buffer, so there is no
//                     receive-WQE pool to manage and no RNR retries.
//   FI_FENCE          orders against prior operations to the SAME TARGET
//                     ENDPOINT, not merely the same connection, so the
//                     per-peer grouping needs no queue pinning.
//
// The receive path still ends at a completion queue in host memory. That is
// the hop this benchmark measures.
//
// Useful environment on Perlmutter:
//   export FI_CXI_RDZV_THRESHOLD=...     # rendezvous cutover, affects large msgs
//   export FI_CXI_DEFAULT_CQ_SIZE=131072
//   export FI_HMEM_CUDA_USE_GDRCOPY=1

#pragma once

#include <cuda_runtime.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>

#include <ctime>

#include "rdma_common.hpp"

#define FCHK(expr, msg)                                                        \
  do {                                                                         \
    int rc_ = (int)(expr);                                                     \
    if (rc_) {                                                                 \
      fprintf(stderr, "[%s:%d] FATAL: %s: %s\n", __FILE__, __LINE__, (msg),    \
              fi_strerror(-rc_));                                              \
      MPI_Abort(MPI_COMM_WORLD, 1);                                            \
    }                                                                          \
  } while (0)

class RdmaCtx {
 public:
  fi_info *info = nullptr;
  fid_fabric *fabric = nullptr;
  fid_domain *domain = nullptr;
  fid_ep *ep = nullptr;
  fid_av *av = nullptr;
  fid_cq *txcq = nullptr;
  fid_cq *rxcq = nullptr;
  std::vector<fi_addr_t> peers;

  int rank = 0, nranks = 0;
  int qps_per_peer = 1;  // accepted for interface parity; CXI needs one ep
  int cuda_dev = 0;
  bool mr_endpoint = false, mr_prov_key = false, mr_virt_addr = true;
  bool mr_raw = false;

  static const char *backend() { return "libfabric"; }

  // ------------------------------------------------------------------
  void open(const char *want_dev, int rank_, int nranks_, int qps_per_peer_) {
    rank = rank_;
    nranks = nranks_;
    qps_per_peer = qps_per_peer_;
    cudaGetDevice(&cuda_dev);

    fi_info *hints = fi_allocinfo();
    VCHK(hints, "fi_allocinfo failed");
    hints->ep_attr->type = FI_EP_RDM;
    // FI_FENCE must be requested explicitly (CXI advertises it, but a
    // provider only returns what is asked for) and mode C depends on it.
    // FI_MSG/FI_SEND/FI_RECV are requested because some providers gate the
    // remote-CQ-data path on the messaging capability even for RMA writes.
    hints->caps = FI_RMA | FI_WRITE | FI_REMOTE_WRITE | FI_RMA_EVENT |
                  FI_HMEM | FI_FENCE | FI_MSG | FI_SEND | FI_RECV;
    hints->mode = FI_CONTEXT | FI_CONTEXT2;
    hints->domain_attr->mr_mode = FI_MR_ENDPOINT | FI_MR_ALLOCATED |
                                  FI_MR_PROV_KEY | FI_MR_VIRT_ADDR | FI_MR_HMEM;
    hints->domain_attr->threading = FI_THREAD_SAFE;
    hints->domain_attr->resource_mgmt = FI_RM_ENABLED;
    hints->tx_attr->op_flags = 0;
    hints->fabric_attr->prov_name = strdup("cxi");
    if (want_dev && *want_dev) hints->domain_attr->name = strdup(want_dev);

    FCHK(fi_getinfo(FI_VERSION(1, 15), nullptr, nullptr, 0, hints, &info),
         "fi_getinfo(cxi) failed -- is the CXI provider present?");
    fi_freeinfo(hints);

    const uint64_t m = info->domain_attr->mr_mode;
    mr_endpoint = m & FI_MR_ENDPOINT;
    mr_prov_key = m & FI_MR_PROV_KEY;
    mr_virt_addr = m & FI_MR_VIRT_ADDR;
    mr_raw = m & FI_MR_RAW;
    VCHK(!mr_raw, "FI_MR_RAW not supported by this benchmark");
    VCHK(info->domain_attr->cq_data_size >= 4,
         "provider reports no remote CQ data; WRITE-with-IMM not expressible");
    VCHK(info->tx_attr->inject_size >= sizeof(uint64_t),
         "inject_size < 8: the inline flag path (post_flag) would be unsafe");

    if (rank == 0) {
      printf("[ofi] provider=%s domain=%s\n", info->fabric_attr->prov_name,
             info->domain_attr->name);
      printf("[ofi] mr_mode=0x%llx (endpoint=%d prov_key=%d virt_addr=%d)\n",
             (unsigned long long)m, (int)mr_endpoint, (int)mr_prov_key,
             (int)mr_virt_addr);
      printf("[ofi] cq_data=%zu inject=%zu  RMA_EVENT=%d HMEM=%d FENCE=%d\n",
             info->domain_attr->cq_data_size, info->tx_attr->inject_size,
             (int)!!(info->caps & FI_RMA_EVENT), (int)!!(info->caps & FI_HMEM),
             (int)!!(info->caps & FI_FENCE));
      if (!(info->caps & FI_RMA_EVENT))
        fprintf(stderr,
                "[ofi] WARNING: FI_RMA_EVENT not negotiated; fi_writedata will "
                "not raise target completions and the imm modes will time "
                "out.\n");
      if (!(info->caps & FI_FENCE))
        fprintf(stderr,
                "[ofi] WARNING: FI_FENCE not negotiated; mode C cannot enforce "
                "NIC-side ordering and its results are meaningless.\n");
      if (!mr_virt_addr)
        printf("[ofi] FI_MR_VIRT_ADDR clear: remote addresses are region "
               "offsets; publishing base 0.\n");
    }

    FCHK(fi_fabric(info->fabric_attr, &fabric, nullptr), "fi_fabric failed");
    FCHK(fi_domain(fabric, info, &domain, nullptr), "fi_domain failed");
  }

  const char *dev_name() const { return info->domain_attr->name; }
  const char *link_name() const { return info->fabric_attr->prov_name; }

  // ------------------------------------------------------------------
  void create_queues(int send_depth, int recv_depth) {
    fi_cq_attr ca {};
    ca.format = FI_CQ_FORMAT_DATA;  // needed to read the 64-bit immediate
    ca.wait_obj = FI_WAIT_NONE;
    ca.size = send_depth;
    FCHK(fi_cq_open(domain, &ca, &txcq, nullptr), "fi_cq_open(tx) failed");
    ca.size = recv_depth;
    FCHK(fi_cq_open(domain, &ca, &rxcq, nullptr), "fi_cq_open(rx) failed");

    fi_av_attr aa {};
    aa.type = FI_AV_TABLE;
    aa.count = nranks;
    FCHK(fi_av_open(domain, &aa, &av, nullptr), "fi_av_open failed");

    FCHK(fi_endpoint(domain, info, &ep, nullptr), "fi_endpoint failed");
    // Selective completion is the analogue of IBV_SEND_SIGNALED: only
    // requests carrying FI_COMPLETION produce a tx CQ entry.
    FCHK(fi_ep_bind(ep, &txcq->fid, FI_TRANSMIT | FI_SELECTIVE_COMPLETION),
         "fi_ep_bind(txcq) failed");
    FCHK(fi_ep_bind(ep, &rxcq->fid, FI_RECV), "fi_ep_bind(rxcq) failed");
    FCHK(fi_ep_bind(ep, &av->fid, 0), "fi_ep_bind(av) failed");
    FCHK(fi_enable(ep), "fi_enable failed");
    ctx_ring_.assign((size_t)send_depth + 8, fi_context2{});  // +probes headroom
  }

  // remote_event is accepted for interface parity but needs no per-MR
  // action here. fi_mr_bind() takes only a counter (fid_cntr) or an
  // endpoint (fid_ep) -- binding a CQ to an MR is invalid in OFI. Target
  // completions for fi_writedata come from the ENDPOINT's receive CQ,
  // enabled by FI_RMA_EVENT in the endpoint capabilities together with the
  // fi_ep_bind(rxcq, FI_RECV) performed in create_queues().
  //
  // relaxed_ordering has no MR-level equivalent on CXI; it is a NIC/PCIe
  // setting exposed through FI_CXI_* environment variables.
  MemRegion reg_mr(void *ptr, size_t len, bool relaxed_ordering,
                   bool /*remote_event*/) {
    if (relaxed_ordering && rank == 0)
      fprintf(stderr,
              "[warn] --relaxed-ordering has no MR-level equivalent on CXI; "
              "see FI_CXI_* PCIe ordering knobs\n");

    iovec iov {ptr, len};
    fi_mr_attr attr {};
    attr.mr_iov = &iov;
    attr.iov_count = 1;
    attr.access = FI_WRITE | FI_REMOTE_WRITE;  // match the negotiated caps
    attr.requested_key = mr_prov_key ? 0 : next_key_++;
    // Detect rather than assume: the capability probe re-runs entirely in
    // host memory to rule out FI_HMEM as the cause of an RMA rejection.
    cudaPointerAttributes pa {};
    const bool is_dev = (cudaPointerGetAttributes(&pa, ptr) == cudaSuccess &&
                         (pa.type == cudaMemoryTypeDevice ||
                          pa.type == cudaMemoryTypeManaged));
    cudaGetLastError();  // clear the error set for unregistered host pointers
    if (is_dev) {
      attr.iface = FI_HMEM_CUDA;
      attr.device.cuda = cuda_dev;
    } else {
      attr.iface = FI_HMEM_SYSTEM;
    }

    fid_mr *mr = nullptr;
    FCHK(fi_mr_regattr(domain, &attr, 0, &mr), "fi_mr_regattr failed");

    if (mr_endpoint) {
      FCHK(fi_mr_bind(mr, &ep->fid, 0), "fi_mr_bind(ep) failed");
      FCHK(fi_mr_enable(mr), "fi_mr_enable failed");
    }

    MemRegion m;
    m.desc = fi_mr_desc(mr);
    m.key = fi_mr_key(mr);
    m.impl = mr;
    return m;
  }

  // With FI_MR_VIRT_ADDR clear the target names memory by offset into the
  // region rather than by virtual address, so publish base 0 and let the
  // caller's usual "base + offset" arithmetic yield the offset directly.
  std::vector<MrDesc> exchange_mr(const MemRegion &m, void *addr) {
    MrDesc mine {mr_virt_addr ? (uint64_t)addr : 0ull, m.key};
    std::vector<MrDesc> all(nranks);
    MPI_Allgather(&mine, sizeof(MrDesc), MPI_BYTE, all.data(), sizeof(MrDesc),
                  MPI_BYTE, MPI_COMM_WORLD);
    return all;
  }

  void dereg_mr(const MemRegion &m) {
    if (m.impl) fi_close(&((fid_mr *)m.impl)->fid);
  }

  // ------------------------------------------------------------------
  void connect() {
    size_t len = 0;
    fi_getname(&ep->fid, nullptr, &len);  // returns -FI_ETOOSMALL, sets len
    VCHK(len > 0 && len <= 256, "unexpected fi_getname address length");

    std::vector<char> mine(len), all((size_t)len * nranks);
    FCHK(fi_getname(&ep->fid, mine.data(), &len), "fi_getname failed");
    MPI_Allgather(mine.data(), (int)len, MPI_BYTE, all.data(), (int)len,
                  MPI_BYTE, MPI_COMM_WORLD);

    peers.assign(nranks, FI_ADDR_UNSPEC);
    int n = fi_av_insert(av, all.data(), nranks, peers.data(), 0, nullptr);
    VCHK(n == nranks, "fi_av_insert did not insert all peers");
  }

  // Does this provider actually carry remote CQ data on a one-sided write?
  // Answered once, with an 8-byte loopback write to our own region, rather
  // than discovered mid-benchmark. CXI advertises cq_data_size but may only
  // implement it on the messaging path.
  bool probe_imm(const MemRegion &lmr, uint64_t laddr, const MrDesc &self,
                 const char *label = "RMA write") {
    iovec iov {(void *)laddr, 8};
    fi_rma_iov rma {self.addr, 8, self.key};
    void *desc = lmr.desc;

    fi_msg_rma msg {};
    msg.msg_iov = &iov;
    msg.desc = &desc;
    msg.iov_count = 1;
    msg.addr = peers[rank];
    msg.rma_iov = &rma;
    msg.rma_iov_count = 1;
    msg.context = next_ctx();
    msg.data = 0xABCDu;

    ssize_t r;
    for (;;) {
      r = fi_writemsg(ep, &msg, FI_REMOTE_CQ_DATA | FI_COMPLETION);
      if (r != -FI_EAGAIN) break;
      progress();
    }
    if (r == 0) {
      poll_send(1);
      return probe_delivered(label, "fi_writemsg");
    }

    if (r == -FI_EBADFLAGS) {
      for (;;) {
        r = fi_writedata(ep, (void *)laddr, 8, desc, 0xABCDu, peers[rank],
                         self.addr, self.key, next_ctx());
        if (r != -FI_EAGAIN) break;
        progress();
      }
      if (r == 0) {
        imm_via_writedata_ = true;
        return probe_delivered(label, "fi_writedata");
      }
    }

    imm_supported_ = false;
    if (rank == 0)
      fprintf(stderr,
              "[ofi] remote CQ data on %s: UNAVAILABLE\n"
              "      fi_writemsg(FI_REMOTE_CQ_DATA) -> %s\n"
              "      fi_writedata()                 -> %s\n"
              "      WRITE-with-IMM is not expressible for one-sided writes "
              "on this provider; the imm modes will be skipped. Note that an "
              "advertised cq_data_size may apply only to the messaging path "
              "(fi_senddata / tagged), not to RMA.\n",
              label, fi_strerror(FI_EBADFLAGS), fi_strerror((int)-r));
    return false;
  }

  bool imm_available() const { return imm_supported_; }

  // Discriminating control for the result above: does this provider carry
  // remote CQ data on the MESSAGING path? If yes, remote CQ data is
  // supported and configured correctly, and its absence on RMA writes is a
  // property of the provider's RMA implementation rather than of our setup.
  // Loopback send to ourselves; bounded so it cannot hang.
  bool probe_msg_imm() {
    char rbuf[64] = {0};
    char sbuf[8] = {0};
    // FI_MR_LOCAL is not in mr_mode, so send/recv buffers need no descriptor.
    ssize_t r = fi_recv(ep, rbuf, sizeof(rbuf), nullptr, FI_ADDR_UNSPEC,
                        next_ctx());  // FI_DIRECTED_RECV not negotiated
    if (r) {
      if (rank == 0)
        printf("[ofi] remote CQ data on message: fi_recv -> %s\n",
               fi_strerror((int)-r));
      return false;
    }
    for (;;) {
      r = fi_senddata(ep, sbuf, sizeof(sbuf), nullptr, 0xBEEFu, peers[rank],
                      next_ctx());
      if (r != -FI_EAGAIN) break;
      progress();
    }
    if (r) {
      if (rank == 0)
        printf("[ofi] remote CQ data on message: fi_senddata -> %s\n",
               fi_strerror((int)-r));
      return false;
    }

    const double deadline = mono_s() + 2.0;
    while (mono_s() < deadline) {
      fi_cq_data_entry e[8];
      ssize_t n = fi_cq_read(rxcq, e, 8);
      if (n == -FI_EAGAIN) continue;
      if (n < 0) break;
      for (ssize_t i = 0; i < n; i++) {
        if ((e[i].flags & FI_REMOTE_CQ_DATA) && e[i].data == 0xBEEFu) {
          if (rank == 0)
            printf("[ofi] remote CQ data on message: available "
                   "(fi_senddata delivered 0x%llx)\n",
                   (unsigned long long)e[i].data);
          return true;
        }
      }
    }
    if (rank == 0)
      printf("[ofi] remote CQ data on message: no completion within 2s\n");
    return false;
  }

  // FI_RMA_EVENT completions do not consume receive buffers.
  void post_recvs(int /*n*/) {}

  int qp_of(int /*peer*/) const { return 0; }

  // ------------------------------------------------------------------
  void post_write(int peer, int, const MemRegion &lmr, uint64_t laddr,
                  uint64_t raddr, uint64_t rkey, uint32_t len, bool signaled,
                  uint64_t id) {
    do_write(peer, lmr.desc, laddr, raddr, rkey, len, signaled, false, 0, false,
             false, id);
  }

  void post_write_imm(int peer, int, const MemRegion &lmr, uint64_t laddr,
                      uint64_t raddr, uint64_t rkey, uint32_t len, uint64_t imm,
                      bool signaled, uint64_t id) {
    do_write(peer, lmr.desc, laddr, raddr, rkey, len, signaled, true, imm, false,
             false, id);
  }

  // FI_INJECT copies flag_val_ out at post time; the single shared source
  // slot is safe ONLY because of that (inject_size >= 8 asserted at open).
  // Per-slot flag values later (dsg payload flags) must keep FI_INJECT or
  // switch to a per-slot source slab.
  void post_flag(int peer, int, uint64_t value, uint64_t raddr, uint64_t rkey,
                 bool fence, bool signaled) {
    flag_val_ = value;
    do_write(peer, nullptr, (uint64_t)&flag_val_, raddr, rkey, sizeof(uint64_t),
             signaled, false, 0, true, fence, 0);
  }

  // Drain until every operation that requested a completion has produced
  // one. Callers no longer count completions themselves, which also lets
  // the fi_writedata fallback below post operations that never complete.
  void poll_send_all() {
    // Completions already reaped while making progress on a full transmit
    // queue still count; polling for them again would hang.
    const int need = pending_ - drained_;
    if (need > 0) poll_send(need);
    pending_ = 0;
    drained_ = 0;
  }

  void poll_send(int n) {
    fi_cq_data_entry e[32];
    int got = 0;
    while (got < n) {
      ssize_t r = fi_cq_read(txcq, e, 32);
      if (r > 0) {
        got += (int)r;
      } else if (r == -FI_EAGAIN) {
        continue;
      } else if (r == -FI_EAVAIL) {
        report_cq_err(txcq, "tx");
      } else {
        FCHK(-r, "fi_cq_read(tx) failed");
      }
    }
  }

  int poll_recv(uint64_t *imm_out, int max) {
    fi_cq_data_entry e[32];
    if (max > 32) max = 32;
    ssize_t r = fi_cq_read(rxcq, e, max);
    if (r == -FI_EAGAIN) return 0;
    if (r == -FI_EAVAIL) {
      report_cq_err(rxcq, "rx");
      return 0;
    }
    if (r < 0) FCHK(-r, "fi_cq_read(rx) failed");
    int n = 0;
    for (ssize_t i = 0; i < r; i++)
      if (e[i].flags & FI_REMOTE_CQ_DATA) imm_out[n++] = e[i].data;
    return n;
  }

  void destroy() {
    if (ep) fi_close(&ep->fid);
    if (av) fi_close(&av->fid);
    if (txcq) fi_close(&txcq->fid);
    if (rxcq) fi_close(&rxcq->fid);
    if (domain) fi_close(&domain->fid);
    if (fabric) fi_close(&fabric->fid);
    if (info) fi_freeinfo(info);
  }

 private:
  uint64_t flag_val_ = 0;
  uint64_t next_key_ = 1;
  int pending_ = 0;
  int drained_ = 0;   // completions consumed by the EAGAIN progress path
  bool imm_via_writedata_ = false;
  bool imm_supported_ = false;
  // FI_CONTEXT/FI_CONTEXT2 mode: every in-flight operation needs its OWN
  // context struct (the provider may use it as scratch until completion).
  // A ring sized to the send queue makes reuse safe by construction: a slot
  // cannot be reissued until the queue has drained the operation using it.
  std::vector<fi_context2> ctx_ring_;
  size_t ctx_cur_ = 0;
  void *next_ctx() { return &ctx_ring_[ctx_cur_++ % ctx_ring_.size()]; }

  void do_write(int peer, void *desc, uint64_t laddr, uint64_t raddr,
                uint64_t rkey, uint32_t len, bool signaled, bool with_imm,
                uint64_t imm, bool inject, bool fence, uint64_t id) {
    iovec iov {(void *)laddr, len};
    fi_rma_iov rma {};
    // With FI_MR_VIRT_ADDR the target address is the peer's virtual address;
    // otherwise it is an offset into the region.
    rma.addr = raddr;
    rma.len = len;
    rma.key = rkey;

    fi_msg_rma msg {};
    msg.msg_iov = &iov;
    msg.desc = &desc;
    msg.iov_count = 1;
    msg.addr = peers[peer];
    msg.rma_iov = &rma;
    msg.rma_iov_count = 1;
    msg.context = next_ctx();
    msg.data = imm;

    uint64_t flags = 0;
    if (signaled) flags |= FI_COMPLETION;
    if (with_imm) flags |= FI_REMOTE_CQ_DATA;
    if (inject) flags |= FI_INJECT;
    if (fence) flags |= FI_FENCE;

    if (!(with_imm && imm_via_writedata_)) {
      for (;;) {
        ssize_t r = fi_writemsg(ep, &msg, flags);
        if (r == 0) {
          if (signaled) pending_++;
          return;
        }
        if (r == -FI_EAGAIN) {
          progress();  // TX queue full: reap completions, then retry
          continue;
        }
        if (r == -FI_EBADFLAGS && with_imm) {
          // Provider accepts FI_COMPLETION/FI_INJECT/FI_FENCE but rejects
          // FI_REMOTE_CQ_DATA on the generic message path. Fall back to the
          // dedicated call, which carries the immediate implicitly.
          imm_via_writedata_ = true;
          if (rank == 0)
            fprintf(stderr,
                    "[ofi] fi_writemsg rejects FI_REMOTE_CQ_DATA; using "
                    "fi_writedata(). Under FI_SELECTIVE_COMPLETION that call "
                    "raises no transmit completion, so send_us for the imm "
                    "modes covers submission only. spread_us is unaffected.\n");
          break;
        }
        FCHK(-(int)r, "fi_writemsg failed");
      }
    }

    // fi_writedata takes no flags, so it produces no transmit completion
    // under selective completion; deliberately not counted in pending_.
    if (with_imm && !imm_supported_) return;  // probe already found it absent
    for (;;) {
      ssize_t r = fi_writedata(ep, (void *)laddr, len, desc, imm, peers[peer],
                               rma.addr, rma.key, next_ctx());
      if (r == 0) return;
      if (r == -FI_EAGAIN) {
        progress();
        continue;
      }
      FCHK(-(int)r, "fi_writedata failed");
    }
    (void)id;
  }

  // Acceptance is not delivery: wait (bounded) for the loopback probe's
  // remote CQ data to actually appear on the rx CQ, and consume it so no
  // stray completion is left for the first imm mode to pop.
  bool probe_delivered(const char *label, const char *via) {
    const double deadline = mono_s() + 2.0;
    while (mono_s() < deadline) {
      fi_cq_data_entry e[8];
      ssize_t n = fi_cq_read(rxcq, e, 8);
      if (n == -FI_EAGAIN || n == 0) continue;
      if (n < 0) break;
      for (ssize_t i = 0; i < n; i++) {
        if ((e[i].flags & FI_REMOTE_CQ_DATA) && e[i].data == 0xABCDu) {
          imm_supported_ = true;
          if (rank == 0)
            printf("[ofi] remote CQ data on %s: %s (delivered)\n", label, via);
          return true;
        }
      }
    }
    imm_supported_ = false;
    if (rank == 0)
      fprintf(stderr,
              "[ofi] remote CQ data on %s: %s accepted but no target "
              "completion within 2s\n", label, via);
    return false;
  }

  static double mono_s() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
  }

  void progress() {
    fi_cq_data_entry e[8];
    ssize_t r = fi_cq_read(txcq, e, 8);
    if (r > 0) drained_ += (int)r;
  }

  void report_cq_err(fid_cq *cq, const char *which) {
    fi_cq_err_entry err {};
    fi_cq_readerr(cq, &err, 0);
    fprintf(stderr, "[rank %d] %s CQ error: %s (prov: %s)\n", rank, which,
            fi_strerror(err.err),
            fi_cq_strerror(cq, err.prov_errno, err.err_data, nullptr, 0));
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
};