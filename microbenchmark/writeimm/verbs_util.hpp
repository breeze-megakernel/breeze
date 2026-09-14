// verbs_util.hpp. libibverbs backend (ConnectX-7 / InfiniBand).
//
// RC QPs, full mesh across MPI ranks, GPUDirect-registered MRs, connection
// parameters exchanged via MPI_Allgather. Minimal by design: no error
// recovery, no reconnection.
//
// Note vs the libfabric backend: IBV_SEND_FENCE orders only within a QP, so
// a peer's payload writes and its flag write must land on the same QP
// (peer-hash, see qp_of()). WRITE_WITH_IMM also consumes a receive WQE at
// the responder, hence the SRQ.

#pragma once

#include <arpa/inet.h>  // ntohl/htonl for imm_data byte order

#include <infiniband/verbs.h>

#include "rdma_common.hpp"

struct QpDesc {
  uint32_t qpn;
  uint32_t psn;
  uint32_t lid;
  uint8_t gid[16];
  uint8_t link_layer;
  uint8_t pad[3];
};

class RdmaCtx {
 public:
  ibv_context *ctx = nullptr;
  ibv_pd *pd = nullptr;
  ibv_cq *send_cq = nullptr;
  ibv_cq *recv_cq = nullptr;
  ibv_srq *srq = nullptr;

  int rank = 0, nranks = 0;
  int qps_per_peer = 1;
  int gid_index = 3;
  uint8_t port_num = 1;
  ibv_port_attr port_attr {};
  std::vector<ibv_qp *> qps;

  static const char *backend() { return "libibverbs"; }

  // ------------------------------------------------------------------
  void open(const char *want_dev, int rank_, int nranks_, int qps_per_peer_) {
    rank = rank_;
    nranks = nranks_;
    qps_per_peer = qps_per_peer_;
    if (const char *e = getenv("BENCH_GID_INDEX")) gid_index = atoi(e);

    int ndev = 0;
    ibv_device **devs = ibv_get_device_list(&ndev);
    VCHK(devs && ndev > 0,
         "no verbs devices (Slingshot/Cassini has none; build -DUSE_LIBFABRIC)");

    ibv_device *chosen = nullptr;
    if (want_dev && *want_dev) {
      for (int i = 0; i < ndev; i++)
        if (!strcmp(ibv_get_device_name(devs[i]), want_dev)) chosen = devs[i];
      VCHK(chosen, "requested HCA not present (check --hca)");
    } else {
      for (int i = 0; i < ndev && !chosen; i++) {
        ibv_context *c = ibv_open_device(devs[i]);
        if (!c) continue;
        ibv_port_attr pa {};
        if (!ibv_query_port(c, port_num, &pa) && pa.state == IBV_PORT_ACTIVE)
          chosen = devs[i];
        ibv_close_device(c);
      }
      VCHK(chosen, "no device with an ACTIVE port");
    }

    ctx = ibv_open_device(chosen);
    VCHK(ctx, "ibv_open_device failed");
    ibv_free_device_list(devs);
    VCHK(!ibv_query_port(ctx, port_num, &port_attr), "ibv_query_port");
    pd = ibv_alloc_pd(ctx);
    VCHK(pd, "ibv_alloc_pd failed");
  }

  const char *dev_name() const { return ibv_get_device_name(ctx->device); }
  const char *link_name() const {
    return port_attr.link_layer == IBV_LINK_LAYER_ETHERNET ? "RoCE" : "IB";
  }

  // ------------------------------------------------------------------
  // remote_event is unused here: WRITE_WITH_IMM always raises a target
  // completion, at the cost of consuming a receive WQE.
  MemRegion reg_mr(void *ptr, size_t len, bool relaxed_ordering,
                   bool /*remote_event*/) {
    unsigned f = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                 IBV_ACCESS_REMOTE_READ;
#ifdef IBV_ACCESS_RELAXED_ORDERING
    if (relaxed_ordering) f |= IBV_ACCESS_RELAXED_ORDERING;
#else
    if (relaxed_ordering && rank == 0)
      fprintf(stderr, "[warn] rdma-core lacks IBV_ACCESS_RELAXED_ORDERING\n");
#endif
    ibv_mr *mr = ibv_reg_mr(pd, ptr, len, f);
    VCHK(mr, "ibv_reg_mr failed (GPU memory needs nvidia-peermem loaded)");
    MemRegion m;
    m.desc = (void *)(uintptr_t)mr->lkey;
    m.key = mr->rkey;
    m.impl = mr;
    return m;
  }

  std::vector<MrDesc> exchange_mr(const MemRegion &m, void *addr) {
    MrDesc mine {(uint64_t)addr, m.key};  // verbs always names by address
    std::vector<MrDesc> all(nranks);
    MPI_Allgather(&mine, sizeof(MrDesc), MPI_BYTE, all.data(), sizeof(MrDesc),
                  MPI_BYTE, MPI_COMM_WORLD);
    return all;
  }

  void dereg_mr(const MemRegion &m) {
    if (m.impl) ibv_dereg_mr((ibv_mr *)m.impl);
  }

  // ------------------------------------------------------------------
  void create_queues(int send_depth, int recv_depth) {
    send_cq = ibv_create_cq(ctx, send_depth, nullptr, nullptr, 0);
    VCHK(send_cq, "ibv_create_cq(send) failed");
    recv_cq = ibv_create_cq(ctx, recv_depth, nullptr, nullptr, 0);
    VCHK(recv_cq, "ibv_create_cq(recv) failed");

    ibv_srq_init_attr sa {};
    sa.attr.max_wr = recv_depth;
    sa.attr.max_sge = 1;
    srq = ibv_create_srq(pd, &sa);
    VCHK(srq, "ibv_create_srq failed");

    qps.assign((size_t)nranks * qps_per_peer, nullptr);
    for (int p = 0; p < nranks; p++) {
      if (p == rank) continue;
      for (int q = 0; q < qps_per_peer; q++) {
        ibv_qp_init_attr ia {};
        ia.send_cq = send_cq;
        ia.recv_cq = recv_cq;
        ia.srq = srq;
        ia.qp_type = IBV_QPT_RC;
        ia.cap.max_send_wr = send_depth;
        ia.cap.max_send_sge = 1;
        ia.cap.max_recv_sge = 1;
        ia.cap.max_inline_data = 64;
        ibv_qp *qp = ibv_create_qp(pd, &ia);
        VCHK(qp, "ibv_create_qp failed (lower send depth?)");
        qps[(size_t)p * qps_per_peer + q] = qp;
        to_init(qp);
      }
    }
  }

  // Peer-hash: a peer's payload writes and its flag must share a QP, or
  // IBV_SEND_FENCE does not order them.
  int qp_of(int peer) const { return peer % qps_per_peer; }
  ibv_qp *qp_for(int peer, int q) const {
    return qps[(size_t)peer * qps_per_peer + q];
  }

  void connect() {
    const size_t per_rank = (size_t)nranks * qps_per_peer;
    std::vector<QpDesc> mine(per_rank), all(per_rank * nranks);
    ibv_gid gid {};
    if (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET)
      VCHK(!ibv_query_gid(ctx, port_num, gid_index, &gid),
           "ibv_query_gid failed (set BENCH_GID_INDEX)");

    for (int p = 0; p < nranks; p++)
      for (int q = 0; q < qps_per_peer; q++) {
        QpDesc &d = mine[(size_t)p * qps_per_peer + q];
        ibv_qp *qp = qps[(size_t)p * qps_per_peer + q];
        d.qpn = qp ? qp->qp_num : 0;
        d.psn = 0x1000 + (uint32_t)(rank * 131 + p * 17 + q);
        d.lid = port_attr.lid;
        d.link_layer = port_attr.link_layer;
        memcpy(d.gid, &gid, 16);
      }

    MPI_Allgather(mine.data(), per_rank * sizeof(QpDesc), MPI_BYTE, all.data(),
                  per_rank * sizeof(QpDesc), MPI_BYTE, MPI_COMM_WORLD);

    for (int p = 0; p < nranks; p++) {
      if (p == rank) continue;
      for (int q = 0; q < qps_per_peer; q++) {
        const QpDesc &rem =
            all[(size_t)p * per_rank + (size_t)rank * qps_per_peer + q];
        const QpDesc &loc = mine[(size_t)p * qps_per_peer + q];
        to_rtr(qps[(size_t)p * qps_per_peer + q], rem);
        to_rts(qps[(size_t)p * qps_per_peer + q], loc.psn);
      }
    }
  }

  // RDMA_WRITE_WITH_IMM is mandatory in the IB spec; no probe needed.
  bool probe_imm(const MemRegion &, uint64_t, const MrDesc &,
                 const char * = nullptr) {
    return true;
  }
  bool imm_available() const { return true; }
  bool probe_msg_imm() { return true; }  // IB carries imm on SEND and WRITE

  void post_recvs(int n) {
    for (int i = 0; i < n; i++) {
      ibv_recv_wr wr {};
      ibv_recv_wr *bad = nullptr;
      wr.num_sge = 0;
      VCHK(!ibv_post_srq_recv(srq, &wr, &bad), "ibv_post_srq_recv failed");
    }
  }

  // ------------------------------------------------------------------
  void post_write(int peer, int q, const MemRegion &lmr, uint64_t laddr,
                  uint64_t raddr, uint64_t rkey, uint32_t len, bool signaled,
                  uint64_t id) {
    do_send(peer, q, lmr, laddr, raddr, rkey, len, signaled, false, 0, false,
            false, id);
  }

  void post_write_imm(int peer, int q, const MemRegion &lmr, uint64_t laddr,
                      uint64_t raddr, uint64_t rkey, uint32_t len, uint64_t imm,
                      bool signaled, uint64_t id) {
    do_send(peer, q, lmr, laddr, raddr, rkey, len, signaled, true, imm, false,
            false, id);
  }

  // 8-byte inline flag write. fence -> IBV_SEND_FENCE, which defers this
  // request at the NIC until all prior requests on this QP complete.
  // IBV_SEND_INLINE copies flag_val_ into the WQE at post time; the single
  // shared source slot is safe ONLY because of that (max_inline_data >= 8).
  // If per-slot flag values are added later (dsg payload flags), keep INLINE
  // or switch to a per-slot source slab.
  void post_flag(int peer, int q, uint64_t value, uint64_t raddr, uint64_t rkey,
                 bool fence, bool signaled) {
    flag_val_ = value;
    MemRegion none;
    do_send(peer, q, none, (uint64_t)&flag_val_, raddr, rkey,
            sizeof(uint64_t), signaled, false, 0, true, fence, 0);
  }

  void poll_send_all() {
    poll_send(pending_);
    pending_ = 0;
  }

  void poll_send(int n) {
    ibv_wc wc[32];
    int got = 0;
    while (got < n) {
      int r = ibv_poll_cq(send_cq, 32, wc);
      VCHK(r >= 0, "ibv_poll_cq(send) failed");
      for (int i = 0; i < r; i++)
        if (wc[i].status != IBV_WC_SUCCESS) {
          fprintf(stderr, "[rank %d] send WC error: %s\n", rank,
                  ibv_wc_status_str(wc[i].status));
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
      got += r;
    }
  }

  int poll_recv(uint64_t *imm_out, int max) {
    ibv_wc wc[32];
    if (max > 32) max = 32;
    int r = ibv_poll_cq(recv_cq, max, wc);
    if (r <= 0) return r;
    int n = 0;
    for (int i = 0; i < r; i++) {
      if (wc[i].status != IBV_WC_SUCCESS) {
        fprintf(stderr, "recv WC error: %s\n", ibv_wc_status_str(wc[i].status));
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      if (wc[i].wc_flags & IBV_WC_WITH_IMM) imm_out[n++] = ntohl(wc[i].imm_data);
    }
    // Contract: the caller (recv_progress in writeimm_bench.cu) reposts what
    // it consumed; reposting here as well would double-post past SRQ depth.
    return n;
  }

  void destroy() {
    for (auto *qp : qps)
      if (qp) ibv_destroy_qp(qp);
    if (srq) ibv_destroy_srq(srq);
    if (send_cq) ibv_destroy_cq(send_cq);
    if (recv_cq) ibv_destroy_cq(recv_cq);
    if (pd) ibv_dealloc_pd(pd);
    if (ctx) ibv_close_device(ctx);
  }

 private:
  uint64_t flag_val_ = 0;
  int pending_ = 0;

  void do_send(int peer, int q, const MemRegion &lmr, uint64_t laddr,
               uint64_t raddr, uint64_t rkey, uint32_t len, bool signaled,
               bool with_imm, uint64_t imm, bool inl, bool fence, uint64_t id) {
    ibv_sge sge {};
    sge.addr = laddr;
    sge.length = len;
    sge.lkey = inl ? 0 : (uint32_t)(uintptr_t)lmr.desc;

    ibv_send_wr wr {};
    ibv_send_wr *bad = nullptr;
    wr.wr_id = id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = with_imm ? IBV_WR_RDMA_WRITE_WITH_IMM : IBV_WR_RDMA_WRITE;
    wr.send_flags = 0;
    if (signaled) wr.send_flags |= IBV_SEND_SIGNALED;
    if (inl) wr.send_flags |= IBV_SEND_INLINE;
    if (fence) wr.send_flags |= IBV_SEND_FENCE;
    if (with_imm) wr.imm_data = htonl((uint32_t)imm);
    wr.wr.rdma.remote_addr = raddr;
    wr.wr.rdma.rkey = (uint32_t)rkey;
    VCHK(!ibv_post_send(qp_for(peer, q), &wr, &bad), "ibv_post_send failed");
    if (signaled) pending_++;
  }

  void to_init(ibv_qp *qp) {
    ibv_qp_attr a {};
    a.qp_state = IBV_QPS_INIT;
    a.port_num = port_num;
    a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                        IBV_ACCESS_REMOTE_READ;
    VCHK(!ibv_modify_qp(qp, &a,
                        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                            IBV_QP_ACCESS_FLAGS),
         "modify_qp -> INIT failed");
  }

  void to_rtr(ibv_qp *qp, const QpDesc &rem) {
    ibv_qp_attr a {};
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu =
        port_attr.active_mtu < IBV_MTU_4096 ? port_attr.active_mtu : IBV_MTU_4096;
    a.dest_qp_num = rem.qpn;
    a.rq_psn = rem.psn;
    a.max_dest_rd_atomic = 16;
    a.min_rnr_timer = 12;
    a.ah_attr.port_num = port_num;
    if (rem.link_layer == IBV_LINK_LAYER_ETHERNET) {
      a.ah_attr.is_global = 1;
      memcpy(&a.ah_attr.grh.dgid, rem.gid, 16);
      a.ah_attr.grh.sgid_index = (uint8_t)gid_index;
      a.ah_attr.grh.hop_limit = 255;
    } else {
      a.ah_attr.dlid = (uint16_t)rem.lid;
    }
    VCHK(!ibv_modify_qp(qp, &a,
                        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                            IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                            IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER),
         "modify_qp -> RTR failed");
  }

  void to_rts(ibv_qp *qp, uint32_t psn) {
    ibv_qp_attr a {};
    a.qp_state = IBV_QPS_RTS;
    a.timeout = 14;
    a.retry_cnt = 7;
    a.rnr_retry = 7;
    a.sq_psn = psn;
    a.max_rd_atomic = 16;
    VCHK(!ibv_modify_qp(qp, &a,
                        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                            IBV_QP_MAX_QP_RD_ATOMIC),
         "modify_qp -> RTS failed");
  }
};