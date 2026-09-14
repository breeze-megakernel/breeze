/*
 * WRITE_WITH_IMM notification microbenchmark.
 *
 * Modes
 *   A   write_only
 *   B   coupled
 *   C   nic_fence
 *   D1  imm_gdr
 *   D2  imm_host
 */

#include <cuda_runtime.h>
#include <mpi.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "gdr_util.hpp"
#ifdef USE_LIBFABRIC
#include "fabric_util.hpp"   // Slingshot-11 / Cassini (CXI)
#else
#include "verbs_util.hpp"    // ConnectX-7 / InfiniBand
#endif

#define CCHK(call)                                                             \
  do {                                                                         \
    cudaError_t e_ = (call);                                                    \
    if (e_ != cudaSuccess) {                                                    \
      fprintf(stderr, "[%s:%d] CUDA: %s\n", __FILE__, __LINE__,                \
              cudaGetErrorString(e_));                                          \
      MPI_Abort(MPI_COMM_WORLD, 1);                                            \
    }                                                                          \
  } while (0)

static const unsigned long long kRecvFill = 0xA5A5A5A5A5A5A5A5ull;
static const unsigned long long kSendFill = 0x5A5A5A5A5A5A5A5Aull;

// Shared between stamp_kernel (writer) and recv_wait_kernel (checker): the
// two must sample identical interior offsets or the check goes blind (F4).
static const int kInteriorSamples = 64;
__host__ __device__ __forceinline__ size_t interior_off(int j, size_t nw) {
  return 1 + (size_t)j * (nw - 2) / kInteriorSamples;
}

// ============================================================================
// Device code
// ============================================================================

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void fill_kernel(unsigned long long *buf, size_t nwords,
                            unsigned long long word) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < nwords; i += stride) buf[i] = word;
}

// Stamp the epoch into the first word, last word, and the same interior
// sample points the receiver checks (F4). The last word is the one that
// exposes a payload/notification ordering window.
__global__ void stamp_kernel(unsigned char *send_base, size_t msg_size, int n,
                             unsigned long long epoch) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long long *p =
      reinterpret_cast<unsigned long long *>(send_base + (size_t)i * msg_size);
  size_t nw = msg_size / sizeof(unsigned long long);
  p[0] = epoch;
  p[nw - 1] = epoch;
  if (nw > 2) {
    for (int j = 0; j < kInteriorSamples; j++) p[interior_off(j, nw)] = epoch;
  }
}

// One block per expected flag, so out-of-order arrivals are observed in the
// order they actually happen.
__global__ void recv_wait_kernel(volatile unsigned long long *flags,
                                 const int *expect_idx, int n_expect,
                                 unsigned long long epoch,
                                 unsigned long long *arrive_ns,
                                 unsigned long long *data_arrive_ns,
                                 unsigned long long *data_minmax_ns,
                                 const unsigned char *recv_base,
                                 size_t msg_size, int check_payload,
                                 unsigned int *integrity_fail,
                                 unsigned long long *minmax_ns,
                                 unsigned long long timeout_ns,
                                 unsigned int *timed_out) {
  int b = blockIdx.x;
  if (b >= n_expect) return;
  const int idx = expect_idx[b];

  unsigned long long start = gtime();

  // Phase 1: detect payload arrival (last word of tile stamped with epoch)
  if (threadIdx.x == 0) {
    const size_t _nw = msg_size / sizeof(unsigned long long);
    const unsigned long long *_pay = reinterpret_cast<const unsigned long long *>(
        recv_base + (size_t)idx * msg_size);
    unsigned long long t_data;
    for (;;) {
      if (__ldcg(_pay + _nw - 1) == epoch) { t_data = gtime(); break; }
      unsigned long long now = gtime();
      if (now - start > timeout_ns) { atomicExch(timed_out, 1u); t_data = now; break; }
      __nanosleep(64);
    }
    data_arrive_ns[b] = t_data;
    atomicMin(data_minmax_ns, t_data);
    atomicMax(data_minmax_ns + 1, t_data);
  }

  // Phase 2: detect notification flag arrival
  if (threadIdx.x == 0) {
    unsigned long long t;
    for (;;) {
      if (flags[idx] == epoch) {
        t = gtime();
        break;
      }
      unsigned long long now = gtime();
      if (now - start > timeout_ns) {
        atomicExch(timed_out, 1u);
        t = now;
        break;
      }
      __nanosleep(64);
    }
    arrive_ns[b] = t;
    atomicMin(minmax_ns, t);
    atomicMax(minmax_ns + 1, t);
  }
  __syncthreads();

  if (!check_payload) return;

  // __ldcg: read through L2, bypassing L1, so a stale-cache explanation for
  // any 'bad' count is excluded and the finding is attributable to the
  // wire / PCIe path (F4).
  const unsigned long long *p = reinterpret_cast<const unsigned long long *>(
      recv_base + (size_t)idx * msg_size);
  const size_t nw = msg_size / sizeof(unsigned long long);
  unsigned int bad = 0;

  if (threadIdx.x == 0) {
    if (__ldcg(p) != epoch) bad++;
    if (__ldcg(p + nw - 1) != epoch) bad++;  // the interesting failure
  }
  // Interior samples now compare against the current epoch (stamped by the
  // sender at the same offsets), so the check stays live on every iteration.
  for (int j = threadIdx.x; j < kInteriorSamples && nw > 2; j += blockDim.x) {
    if (__ldcg(p + interior_off(j, nw)) != epoch) {
      bad++;
      break;
    }
  }
  if (bad) atomicAdd(integrity_fail, bad);
}

// Persistent single-thread kernel for the local notification-latency probe.
__global__ void notify_probe_kernel(volatile unsigned long long *flag,
                                    volatile unsigned int *ack, int iters,
                                    unsigned long long timeout_ns) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  for (int e = 1; e <= iters; e++) {
    unsigned long long start = gtime();
    while (*flag != (unsigned long long)e) {
      if (gtime() - start > timeout_ns) return;
    }
    __threadfence_system();
    *ack = (unsigned int)e;
  }
}

// ============================================================================
// Host: configuration and transfer plan
// ============================================================================

enum Mode {
  MODE_WRITE_ONLY = 0,
  MODE_COUPLED,
  MODE_NIC_FENCE,
  MODE_IMM_GDR,
  MODE_IMM_HOST,
  MODE_COUNT
};

static const char *kModeName[MODE_COUNT] = {
    "write_only (ceiling)", "coupled    (vanilla)", "nic_fence  (fence flag)",
    "imm_gdr    (CQ->HBM)", "imm_host   (CQ->host)"};

struct Config {
  int n = 96;
  size_t msg = 256 * 1024;
  int warmup = 20;
  int iters = 100;
  int group_size = 0;  // 0 = one group per destination peer
  int qps_per_peer = 1;
  int local_size = 8;  // GPUs per node; peers on my node are excluded
  int check_payload = 1;
  int relaxed_ordering = 0;
  int notify_probe = 1;
  int recv_core = -1;  // pin the progress thread here; -1 = unpinned (S3)
  int mode_mask = (1 << MODE_COUNT) - 1;
  const char *hca = "";
};

static double now_us() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

// Remote peers of rank r: every rank not on r's node. Intra-node traffic
// would go over NVLink in a real megakernel and never touch the proxy.
static std::vector<int> remote_peers_of(int r, int nranks, int local_size) {
  std::vector<int> v;
  const int my_node = r / local_size;
  for (int p = 0; p < nranks; p++)
    if (p / local_size != my_node) v.push_back(p);
  return v;
}

struct Plan {
  std::vector<int> peer;  // [i] destination rank of outbound transfer i
  std::vector<int> slot;  // [i] slot index within that peer's stream
  int max_slots = 0;
  std::vector<int> expect_idx;    // receiver flag indices I should observe
  std::vector<int> incoming_cnt;  // [src_rank] transfers src sends to me
};

static Plan build_plan(const Config &cfg, int rank, int nranks) {
  Plan pl;
  auto mine = remote_peers_of(rank, nranks, cfg.local_size);
  const int R = (int)mine.size();
  if (R == 0) {
    fprintf(stderr, "no remote peers: need more than one node\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  pl.max_slots = (cfg.n + R - 1) / R;

  pl.peer.resize(cfg.n);
  pl.slot.resize(cfg.n);
  for (int i = 0; i < cfg.n; i++) {
    pl.peer[i] = mine[i % R];  // interleaved: faithful to CTA issue order
    pl.slot[i] = i / R;
  }

  pl.incoming_cnt.assign(nranks, 0);
  for (int s = 0; s < nranks; s++) {
    if (s == rank) continue;
    auto theirs = remote_peers_of(s, nranks, cfg.local_size);
    const int Rs = (int)theirs.size();
    if (Rs == 0) continue;
    for (int i = 0; i < cfg.n; i++) {
      if (theirs[i % Rs] == rank) {
        pl.expect_idx.push_back(s * pl.max_slots + (i / Rs));
        pl.incoming_cnt[s]++;
      }
    }
  }
  std::sort(pl.expect_idx.begin(), pl.expect_idx.end());
  return pl;
}

static uint32_t pack_imm(unsigned long long epoch, int src, int slot) {
  return ((uint32_t)(epoch & 0xFF) << 24) | ((uint32_t)(src & 0xFF) << 16) |
         (uint32_t)(slot & 0xFFFF);
}

// ============================================================================
// Receiver progress thread (WRITE_WITH_IMM modes only)
//
// The IMMCOUNTER of fabric-lib / the control buffer of UEP: decode imm_data
// from the completion queue, count arrivals per group, and only when a group
// is complete push flags into GPU-visible memory.
//
// F2: a group is keyed by (source, slot / ng) and never spans sources.
// ============================================================================

struct RecvThreadArgs {
  RdmaCtx *rc = nullptr;
  std::atomic<bool> quit {false};
  std::atomic<bool> reset_req {false};  // S2: cleared by this thread only
  std::atomic<unsigned long long> epoch {0};
  int notify_group = 1;
  int groups_per_src = 1;  // ceil(max_slots / notify_group)
  int max_slots = 0;
  int nranks = 0;
  const int *incoming_cnt = nullptr;
  bool use_gdr = false;
  GdrFlags *gdr = nullptr;
  volatile unsigned long long *host_flags = nullptr;  // imm_host backend
  std::vector<int> counters;                          // [nranks*groups_per_src]
  std::atomic<long> cqe_seen {0};
  std::atomic<long> stale_seen {0};  // S1: wrong-epoch completions
};

static void recv_progress(RecvThreadArgs *a) {
  uint64_t imm[32];
  while (!a->quit.load(std::memory_order_relaxed)) {
    // S2: reset counters here, in the only thread that touches them, so a
    // timed-out iteration cannot poison the next one.
    if (a->reset_req.exchange(false, std::memory_order_relaxed))
      std::fill(a->counters.begin(), a->counters.end(), 0);

    const int n = a->rc->poll_recv(imm, 32);
    if (n <= 0) continue;

    // F1: every WRITE_WITH_IMM consumed one posted receive; replace them or
    // the queue exhausts after ~recv_depth transfers and the NIC RNR-stalls.
    // (Contract: poll_recv must NOT also repost; no-op on libfabric.)
    a->rc->post_recvs(n);

    const unsigned long long ep = a->epoch.load(std::memory_order_relaxed);

    for (int i = 0; i < n; i++) {
      // S1: drop completions from a previous epoch instead of applying them.
      const int ep_lo = (int)((imm[i] >> 24) & 0xFF);
      if (ep_lo != (int)(ep & 0xFF)) {
        a->stale_seen.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      const int src = (int)((imm[i] >> 16) & 0xFF);
      const int slot = (int)(imm[i] & 0xFFFF);
      const int ng = a->notify_group;

      // F2: group id local to this source; groups never span sources.
      const int g_in_src = slot / ng;
      const int gid = src * a->groups_per_src + g_in_src;
      const int lo = g_in_src * ng;
      const int avail = a->incoming_cnt[src] - lo;
      const int need = std::min(ng, std::max(avail, 0));

      if (need > 0 && ++a->counters[gid] >= need) {
        a->counters[gid] = 0;
        // Group complete: publish every member's flag into GPU-visible
        // memory. This is the hop a PUT-WITH-SIGNAL design does not pay.
        const int base = src * a->max_slots + lo;
        for (int k = 0; k < need; k++) {
          if (a->use_gdr)
            a->gdr->store(base + k, ep);
          else
            a->host_flags[base + k] = ep;
        }
        if (!a->use_gdr) __sync_synchronize();
      }
    }
    a->cqe_seen.fetch_add(n, std::memory_order_relaxed);
  }
}

// ============================================================================
// Local notification-latency probe
//
// No network involved. Measures the hop a WRITE_WITH_IMM design must pay and
// a PUT-WITH-SIGNAL design does not: getting a flag from the host into
// memory a GPU thread can poll, and back. Reported as a ROUND TRIP; the two
// halves are asymmetric (WC store down vs mapped-memory poll up), so do not
// halve it when quoting one-way numbers. quote the RTT.
// ============================================================================

static void probe_notify(int rank, GdrFlags *gdr,
                         unsigned long long *d_flag,
                         volatile unsigned long long *h_flag_host,
                         volatile unsigned long long *h_flag_dev,
                         volatile unsigned int *h_ack_host,
                         volatile unsigned int *h_ack_dev, bool use_gdr,
                         const char *label) {
  const int iters = 2000;
  CCHK(cudaMemset(d_flag, 0, sizeof(unsigned long long)));
  *h_flag_host = 0;
  *h_ack_host = 0;

  volatile unsigned long long *poll_target =
      use_gdr ? reinterpret_cast<volatile unsigned long long *>(d_flag)
              : h_flag_dev;

  notify_probe_kernel<<<1, 1>>>(poll_target, h_ack_dev, iters,
                                5ull * 1000000000ull);

  std::vector<double> s;
  s.reserve(iters);
  for (int e = 1; e <= iters; e++) {
    const double t0 = now_us();
    if (use_gdr)
      gdr->store(0, (unsigned long long)e);
    else {
      *h_flag_host = (unsigned long long)e;
      __sync_synchronize();
    }
    double t1;
    for (;;) {
      if (*h_ack_host == (unsigned int)e) {
        t1 = now_us();
        break;
      }
      if (now_us() - t0 > 1e6) {
        t1 = now_us();
        break;
      }
    }
    if (e > 200) s.push_back(t1 - t0);  // drop warmup
  }
  CCHK(cudaDeviceSynchronize());

  std::sort(s.begin(), s.end());
  if (rank == 0 && !s.empty()) {
    printf("  %-22s p50 %6.2f us   p99 %6.2f us   min %6.2f us   (round trip)\n",
           label, s[s.size() / 2], s[(size_t)(s.size() * 0.99)], s.front());
  }
}

// ============================================================================
// Per-iteration send paths
// ============================================================================

struct SendCtx {
  RdmaCtx *rc;
  const Config *cfg;
  const Plan *pl;
  MemRegion send_mr;
  uint64_t send_base;
  const std::vector<MrDesc> *recv_mr;
  const std::vector<MrDesc> *flag_mr;
  int rank;
  int max_slots;
};

// Signal every k-th write so the send queue drains without a completion per WR.
static const int kSigEvery = 32;

static void post_payload_writes(SendCtx &s, unsigned long long epoch,
                                bool with_imm) {
  const Config &cfg = *s.cfg;
  const Plan &pl = *s.pl;

  for (int i = 0; i < cfg.n; i++) {
    const int p = pl.peer[i];
    const int q = s.rc->qp_of(p);  // contract: pure peer-hash (see header)
    const int ridx = s.rank * s.max_slots + pl.slot[i];
    const uint64_t laddr = s.send_base + (uint64_t)i * cfg.msg;
    const uint64_t raddr = (*s.recv_mr)[p].addr + (uint64_t)ridx * cfg.msg;

    const bool sig = ((i + 1) % kSigEvery == 0) || (i == cfg.n - 1);

    if (with_imm) {
      s.rc->post_write_imm(p, q, s.send_mr, laddr, raddr, (*s.recv_mr)[p].key,
                           (uint32_t)cfg.msg,
                           pack_imm(epoch, s.rank, pl.slot[i]), sig, i);
    } else {
      s.rc->post_write(p, q, s.send_mr, laddr, raddr, (*s.recv_mr)[p].key,
                       (uint32_t)cfg.msg, sig, i);
    }
  }
}

// Mode B: the vanilla proxy expansion, PUT -> FENCE -> SIGNAL per transfer.
// The proxy is FIFO and blocks on each fence, so at most one PUT is in
// flight at a time; posting signaled and draining one CQE reproduces that.
static void run_coupled(SendCtx &s, unsigned long long epoch) {
  const Config &cfg = *s.cfg;
  const Plan &pl = *s.pl;

  for (int i = 0; i < cfg.n; i++) {
    const int p = pl.peer[i];
    const int q = s.rc->qp_of(p);
    const int ridx = s.rank * s.max_slots + pl.slot[i];

    s.rc->post_write(p, q, s.send_mr, s.send_base + (uint64_t)i * cfg.msg,
                     (*s.recv_mr)[p].addr + (uint64_t)ridx * cfg.msg,
                     (*s.recv_mr)[p].key, (uint32_t)cfg.msg, true, i);
    s.rc->poll_send_all();  // the drain

    s.rc->post_flag(p, q, epoch,
                    (*s.flag_mr)[p].addr +
                        (uint64_t)ridx * sizeof(unsigned long long),
                    (*s.flag_mr)[p].key, /*fence=*/false, /*signaled=*/true);
    s.rc->poll_send_all();
  }
}

// Mode C: all payload writes first, then flag writes; the first flag write of
// each group carries a NIC fence flag (IBV_SEND_FENCE / FI_FENCE) and the
// NIC absorbs the ordering instead of the proxy blocking on a drain.
// F3: correctness of the single fenced flag relies on the qp_of() peer-hash
// contract in the header. all of a peer's payload writes share the flag's QP.
static void run_nic_fence(SendCtx &s, unsigned long long epoch) {
  const Config &cfg = *s.cfg;
  const Plan &pl = *s.pl;

  post_payload_writes(s, epoch, /*with_imm=*/false);

  std::vector<std::vector<int>> by_peer(s.rc->nranks);
  for (int i = 0; i < cfg.n; i++) by_peer[pl.peer[i]].push_back(i);

  for (int p = 0; p < s.rc->nranks; p++) {
    auto &v = by_peer[p];
    if (v.empty()) continue;
    const int q = s.rc->qp_of(p);
    const int ng = cfg.group_size > 0 ? cfg.group_size : (int)v.size();

    for (size_t k = 0; k < v.size(); k++) {
      const bool first_of_group = (k % ng == 0);
      const bool last = (k + 1 == v.size());
      const int ridx = s.rank * s.max_slots + pl.slot[v[k]];
      s.rc->post_flag(p, q, epoch,
                      (*s.flag_mr)[p].addr +
                          (uint64_t)ridx * sizeof(unsigned long long),
                      (*s.flag_mr)[p].key, first_of_group, last);
    }
  }
  s.rc->poll_send_all();
}

// ============================================================================
// main
// ============================================================================

static void usage() {
  printf(
      "writeimm_bench [options]\n"
      "  --n N              transfers per rank            (default 96)\n"
      "  --msg BYTES        message size                  (default 262144)\n"
      "  --warmup N         warmup iterations             (default 20)\n"
      "  --iters N          timed iterations              (default 100)\n"
      "  --group-size N     fences/IMMCOUNTER width; 0=per-peer (default 0)\n"
      "  --qps-per-peer N   RC QPs per peer               (default 1)\n"
      "  --local-size N     GPUs per node                 (default 8)\n"
      "  --recv-core N      pin the recv progress thread to core N\n"
      "  --hca NAME         HCA to use, e.g. mlx5_0\n"
      "  --relaxed-ordering enable IBV_ACCESS_RELAXED_ORDERING\n"
      "  --no-check         skip payload integrity checking\n"
      "  --no-probe         skip the local notification probe\n"
      "  --modes MASK       bitmask of modes to run       (default 31)\n");
}

int main(int argc, char **argv) {
  Config cfg;
  for (int i = 1; i < argc; i++) {
    auto next = [&](void) { return (i + 1 < argc) ? argv[++i] : nullptr; };
    if (!strcmp(argv[i], "--n")) cfg.n = atoi(next());
    else if (!strcmp(argv[i], "--msg")) cfg.msg = (size_t)atoll(next());
    else if (!strcmp(argv[i], "--warmup")) cfg.warmup = atoi(next());
    else if (!strcmp(argv[i], "--iters")) cfg.iters = atoi(next());
    else if (!strcmp(argv[i], "--group-size")) cfg.group_size = atoi(next());
    else if (!strcmp(argv[i], "--qps-per-peer")) cfg.qps_per_peer = atoi(next());
    else if (!strcmp(argv[i], "--local-size")) cfg.local_size = atoi(next());
    else if (!strcmp(argv[i], "--recv-core")) cfg.recv_core = atoi(next());
    else if (!strcmp(argv[i], "--hca")) cfg.hca = next();
    else if (!strcmp(argv[i], "--relaxed-ordering")) cfg.relaxed_ordering = 1;
    else if (!strcmp(argv[i], "--no-check")) cfg.check_payload = 0;
    else if (!strcmp(argv[i], "--no-probe")) cfg.notify_probe = 0;
    else if (!strcmp(argv[i], "--modes")) cfg.mode_mask = atoi(next());
    else { usage(); return 1; }
  }

  MPI_Init(&argc, &argv);
  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  int ndev = 0;
  CCHK(cudaGetDeviceCount(&ndev));
  CCHK(cudaSetDevice(rank % ndev));

  Plan pl = build_plan(cfg, rank, nranks);
  const int n_expect = (int)pl.expect_idx.size();
  const size_t nslots = (size_t)nranks * pl.max_slots;

  if (rank == 0) {
    printf("writeimm_bench: ranks=%d n=%d msg=%zuB (%.1f KB) group=%d "
           "qps/peer=%d local=%d relaxed_ordering=%d\n",
           nranks, cfg.n, cfg.msg, cfg.msg / 1024.0,
           cfg.group_size ? cfg.group_size : -1, cfg.qps_per_peer,
           cfg.local_size, cfg.relaxed_ordering);
    printf("  slots/peer=%d  recv slots=%zu  expected inbound=%d\n",
           pl.max_slots, nslots, n_expect);
    if (cfg.qps_per_peer > 1)
      printf("  NOTE: qps_per_peer>1. mode C fence correctness relies on\n"
             "        qp_of() peer-hash pinning (see backend contract).\n");
    printf("\n");
  }

  // ---- buffers -------------------------------------------------------------
  unsigned char *d_send = nullptr, *d_recv = nullptr;
  unsigned long long *d_flags = nullptr, *d_flags_raw = nullptr;
  CCHK(cudaMalloc(&d_send, (size_t)cfg.n * cfg.msg));
  CCHK(cudaMalloc(&d_recv, nslots * cfg.msg));
  const size_t kGpuPage = 1ull << 16;
  const size_t flags_bytes = nslots * sizeof(unsigned long long);
  const size_t flags_span = (flags_bytes + kGpuPage - 1) & ~(kGpuPage - 1);
  CCHK(cudaMalloc(&d_flags_raw, flags_span + kGpuPage));
  d_flags = (unsigned long long *)(((uintptr_t)d_flags_raw + kGpuPage - 1) &
                                   ~(uintptr_t)(kGpuPage - 1));
  CCHK(cudaMemset(d_flags, 0, flags_bytes));

  fill_kernel<<<256, 256>>>((unsigned long long *)d_send,
                            (size_t)cfg.n * cfg.msg / 8, kSendFill);
  fill_kernel<<<256, 256>>>((unsigned long long *)d_recv, nslots * cfg.msg / 8,
                            kRecvFill);
  CCHK(cudaDeviceSynchronize());

  // Host-mapped flags for the imm_host backend.
  unsigned long long *h_flags = nullptr;
  CCHK(cudaHostAlloc(&h_flags, nslots * sizeof(unsigned long long),
                     cudaHostAllocMapped));
  memset(h_flags, 0, nslots * sizeof(unsigned long long));
  unsigned long long *h_flags_dev = nullptr;
  CCHK(cudaHostGetDevicePointer(&h_flags_dev, h_flags, 0));

  unsigned long long *h_probe = nullptr;
  unsigned int *h_ack = nullptr;
  CCHK(cudaHostAlloc(&h_probe, 64, cudaHostAllocMapped));
  CCHK(cudaHostAlloc(&h_ack, 64, cudaHostAllocMapped));
  memset(h_probe, 0, 64);
  memset((void *)h_ack, 0, 64);
  unsigned long long *h_probe_dev = nullptr;
  unsigned int *h_ack_dev = nullptr;
  CCHK(cudaHostGetDevicePointer(&h_probe_dev, h_probe, 0));
  CCHK(cudaHostGetDevicePointer(&h_ack_dev, h_ack, 0));

  int *d_expect = nullptr;
  unsigned long long *d_arrive = nullptr, *d_minmax = nullptr;
  unsigned int *d_bad = nullptr, *d_timeout = nullptr;
  CCHK(cudaMalloc(&d_expect, std::max(1, n_expect) * sizeof(int)));
  CCHK(cudaMalloc(&d_arrive, std::max(1, n_expect) * sizeof(unsigned long long)));
  unsigned long long *d_data_arrive = nullptr, *d_data_minmax = nullptr;
  CCHK(cudaMalloc(&d_data_arrive, std::max(1, n_expect) * sizeof(unsigned long long)));
  CCHK(cudaMalloc(&d_data_minmax, 2 * sizeof(unsigned long long)));
  CCHK(cudaMalloc(&d_minmax, 2 * sizeof(unsigned long long)));
  CCHK(cudaMalloc(&d_bad, sizeof(unsigned int)));
  CCHK(cudaMalloc(&d_timeout, sizeof(unsigned int)));
  if (n_expect)
    CCHK(cudaMemcpy(d_expect, pl.expect_idx.data(), n_expect * sizeof(int),
                    cudaMemcpyHostToDevice));

  // ---- RDMA ---------------------------------------------------------------
  RdmaCtx rc;
  rc.open(cfg.hca, rank, nranks, cfg.qps_per_peer);
  const int send_depth = std::max(1024, 4 * cfg.n);
  const int recv_depth = std::max(4096, 8 * cfg.n);
  rc.create_queues(send_depth, recv_depth);

  MemRegion mr_send =
      rc.reg_mr(d_send, (size_t)cfg.n * cfg.msg, cfg.relaxed_ordering, false);
  MemRegion mr_recv =
      rc.reg_mr(d_recv, nslots * cfg.msg, cfg.relaxed_ordering, true);
  MemRegion mr_flag = rc.reg_mr(d_flags, nslots * sizeof(unsigned long long),
                                cfg.relaxed_ordering, false);

  auto recv_desc = rc.exchange_mr(mr_recv, d_recv);
  auto flag_desc = rc.exchange_mr(mr_flag, d_flags);
  rc.connect();
  rc.post_recvs(recv_depth - 16);
  MPI_Barrier(MPI_COMM_WORLD);

  const bool imm_ok =
      rc.probe_imm(mr_send, (uint64_t)d_send, recv_desc[rank], "RMA write");
  if (!imm_ok) {
    rc.probe_msg_imm();
    void *h_src = nullptr, *h_dst = nullptr;
    CCHK(cudaHostAlloc(&h_src, 4096, cudaHostAllocDefault));
    CCHK(cudaHostAlloc(&h_dst, 4096, cudaHostAllocDefault));
    memset(h_src, 0, 4096);
    memset(h_dst, 0, 4096);
    MemRegion hs = rc.reg_mr(h_src, 4096, false, false);
    MemRegion hd = rc.reg_mr(h_dst, 4096, false, true);
    auto hd_desc = rc.exchange_mr(hd, h_dst);
    rc.probe_imm(hs, (uint64_t)h_src, hd_desc[rank], "RMA write, host memory");
    rc.dereg_mr(hs);
    rc.dereg_mr(hd);
    cudaFreeHost(h_src);
    cudaFreeHost(h_dst);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0)
    printf("transport=%s  device=%s  link=%s\n\n", RdmaCtx::backend(),
           rc.dev_name(), rc.link_name());

  GdrFlags gdr;
  const bool gdr_ok = gdr.init(d_flags, flags_span);

  // ---- local notification probe -------------------------------------------
  if (cfg.notify_probe) {
    if (rank == 0) printf("Notification hop (local, no network):\n");
    if (gdr_ok)
      probe_notify(rank, &gdr, d_flags, h_probe, h_probe_dev, h_ack, h_ack_dev,
                   true, "host -> HBM (gdrcopy)");
    else if (rank == 0)
      printf("  %-22s unavailable (build with -DUSE_GDRCOPY)\n",
             "host -> HBM (gdrcopy)");
    probe_notify(rank, &gdr, d_flags, h_probe, h_probe_dev, h_ack, h_ack_dev,
                 false, "host -> host (PCIe poll)");
    if (rank == 0) printf("\n");
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // ---- receiver progress thread -------------------------------------------
  RecvThreadArgs rt;
  rt.rc = &rc;
  rt.max_slots = pl.max_slots;
  rt.nranks = nranks;
  rt.incoming_cnt = pl.incoming_cnt.data();
  rt.gdr = &gdr;
  rt.host_flags = h_flags;
  std::thread rthread;

  SendCtx sc {&rc,        &cfg,       &pl,        mr_send,
              (uint64_t)d_send, &recv_desc, &flag_desc, rank,
              pl.max_slots};

  // ---- run ----------------------------------------------------------------
  if (rank == 0) {
    printf("%-24s %10s %10s %12s %10s %8s\n", "mode", "submit_us", "send_us",
           "spread_us", "spread_p99", "bad");
    printf("---------------------------------------------------------------"
           "-------------------\n");
  }

  unsigned long long epoch = 0;

  for (int m = 0; m < MODE_COUNT; m++) {
    if (!(cfg.mode_mask & (1 << m))) continue;
    const bool imm_mode = (m == MODE_IMM_GDR || m == MODE_IMM_HOST);
    if (imm_mode && !imm_ok) {
      if (rank == 0)
        printf("%-24s %10s  (no remote CQ data on RMA write)\n", kModeName[m],
               "n/a");
      continue;
    }
    if (m == MODE_IMM_GDR && !gdr_ok) {
      if (rank == 0)
        printf("%-24s %10s\n", kModeName[m], "(no gdrcopy)");
      continue;
    }

    if (imm_mode) {
      rt.use_gdr = (m == MODE_IMM_GDR);
      rt.notify_group = cfg.group_size > 0 ? cfg.group_size : pl.max_slots;
      rt.groups_per_src =
          (pl.max_slots + rt.notify_group - 1) / rt.notify_group;   // F2
      rt.counters.assign((size_t)nranks * rt.groups_per_src, 0);    // F2
      rt.stale_seen = 0;
      rt.cqe_seen = 0;
      rt.quit = false;
      rt.reset_req = false;
      rthread = std::thread(recv_progress, &rt);
      if (cfg.recv_core >= 0) {                                     // S3
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(cfg.recv_core, &cs);
        pthread_setaffinity_np(rthread.native_handle(), sizeof(cs), &cs);
      }
    }

    std::vector<double> v_submit, v_send, v_spread;
    unsigned int bad_total = 0;
    std::vector<double> all_xfer_deltas;   // data arrival dispersion (send-side)
    std::vector<double> all_notify_deltas; // flag - data per tile (recv-side overhead)
    all_xfer_deltas.reserve(cfg.iters * n_expect);
    all_notify_deltas.reserve(cfg.iters * n_expect);

    for (int it = 0; it < cfg.warmup + cfg.iters; it++) {
      epoch++;
      const bool timed = (it >= cfg.warmup);
      rt.epoch.store(epoch, std::memory_order_relaxed);

      stamp_kernel<<<(cfg.n + 127) / 128, 128>>>(d_send, cfg.msg, cfg.n, epoch);
      const unsigned long long mm_init[2] = {~0ull, 0ull};
      CCHK(cudaMemcpy(d_minmax, mm_init, sizeof(mm_init), cudaMemcpyHostToDevice));
      CCHK(cudaMemcpy(d_data_minmax, mm_init, sizeof(mm_init), cudaMemcpyHostToDevice));
      CCHK(cudaMemset(d_bad, 0, sizeof(unsigned int)));
      CCHK(cudaMemset(d_timeout, 0, sizeof(unsigned int)));
      CCHK(cudaDeviceSynchronize());

      volatile unsigned long long *poll_flags =
          (m == MODE_IMM_HOST) ? h_flags_dev : d_flags;

      if (m != MODE_WRITE_ONLY && n_expect > 0) {
        recv_wait_kernel<<<n_expect, 128>>>(
            poll_flags, d_expect, n_expect, epoch, d_arrive,
            d_data_arrive, d_data_minmax,
            d_recv, cfg.msg,
            cfg.check_payload, d_bad, d_minmax, 10ull * 1000000000ull,
            d_timeout);
      }

      MPI_Barrier(MPI_COMM_WORLD);

      const double t0 = now_us();
      switch (m) {
        case MODE_WRITE_ONLY:
          post_payload_writes(sc, epoch, false);
          break;
        case MODE_COUPLED:
          run_coupled(sc, epoch);
          break;
        case MODE_NIC_FENCE:
          run_nic_fence(sc, epoch);
          break;
        default:
          post_payload_writes(sc, epoch, true);
          break;
      }
      const double t1 = now_us();
      rc.poll_send_all();
      const double t2 = now_us();

      CCHK(cudaDeviceSynchronize());

      unsigned long long mm[2];
      unsigned int bad = 0, to = 0;
      CCHK(cudaMemcpy(mm, d_minmax, sizeof(mm), cudaMemcpyDeviceToHost));
      CCHK(cudaMemcpy(&bad, d_bad, sizeof(bad), cudaMemcpyDeviceToHost));
      CCHK(cudaMemcpy(&to, d_timeout, sizeof(to), cudaMemcpyDeviceToHost));

      if (to) {
        if (rank == 0 && timed)
          fprintf(stderr, "[warn] receiver timed out (mode %d, iter %d)\n", m,
                  it);
        if (imm_mode) rt.reset_req.store(true);  // S2
      }

      if (timed) {
        v_submit.push_back(t1 - t0);
        v_send.push_back(t2 - t0);
        if (m != MODE_WRITE_ONLY && n_expect > 0 && !to) {
          v_spread.push_back((mm[1] - mm[0]) / 1000.0);
          std::vector<unsigned long long> h_arrive(n_expect), h_data_arrive(n_expect);
          unsigned long long data_mm[2];
          CCHK(cudaMemcpy(h_arrive.data(), d_arrive,
                          n_expect * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
          CCHK(cudaMemcpy(h_data_arrive.data(), d_data_arrive,
                          n_expect * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
          CCHK(cudaMemcpy(data_mm, d_data_minmax, sizeof(data_mm),
                          cudaMemcpyDeviceToHost));
          for (int j = 0; j < n_expect; j++) {
            all_xfer_deltas.push_back((double)(h_data_arrive[j] - data_mm[0]) / 1000.0);
            all_notify_deltas.push_back((double)(h_arrive[j] - h_data_arrive[j]) / 1000.0);
          }
        }
        bad_total += bad;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    if (imm_mode) {
      rt.quit = true;
      rthread.join();
      const long stale = rt.stale_seen.load();
      if (rank == 0 && stale > 0)
        fprintf(stderr, "[note] mode %s: %ld stale-epoch completions dropped\n",
                kModeName[m], stale);
      if (rank == 0 && rt.cqe_seen.load() == 0)
        fprintf(stderr,
                "[warn] mode %s saw zero remote completions. The transport is "
                "not delivering FI_REMOTE_CQ_DATA / imm_data to the target; "
                "check that FI_RMA_EVENT was negotiated.\n",
                kModeName[m]);
    }

    auto med = [](std::vector<double> &x) {
      if (x.empty()) return 0.0;
      std::sort(x.begin(), x.end());
      return x[x.size() / 2];
    };
    auto p99 = [](std::vector<double> &x) {
      if (x.empty()) return 0.0;
      return x[std::min(x.size() - 1, (size_t)(x.size() * 0.99))];
    };

    double lsub = med(v_submit), lsnd = med(v_send);
    double lspr = med(v_spread), lspr99 = p99(v_spread);
    double gsub, gsnd, gspr, gspr99;
    unsigned int gbad;
    MPI_Reduce(&lsub, &gsub, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&lsnd, &gsnd, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&lspr, &gspr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&lspr99, &gspr99, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bad_total, &gbad, 1, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);

    // Per-tile breakdown CDFs
    auto pcts = [](std::vector<double> &v, double o[5]) {
      if (v.empty()) { for(int i=0;i<5;i++) o[i]=0; return; }
      std::sort(v.begin(), v.end());
      size_t s = v.size();
      o[0]=v[s/2]; o[1]=v[std::min(s-1,(size_t)(s*0.90))];
      o[2]=v[std::min(s-1,(size_t)(s*0.95))];
      o[3]=v[std::min(s-1,(size_t)(s*0.99))]; o[4]=v.back();
    };
    double lx[5], ln[5];
    pcts(all_xfer_deltas, lx);
    pcts(all_notify_deltas, ln);
    double gx[5], gn[5];
    MPI_Reduce(lx, gx, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(ln, gn, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      printf("%-24s %10.1f %10.1f %12.1f %10.1f %8u\n", kModeName[m], gsub,
             gsnd, gspr, gspr99, gbad);
      if (!all_xfer_deltas.empty()) {
        printf("  xfer  (data dispersion, us): p50=%7.1f p90=%7.1f p95=%7.1f p99=%7.1f max=%7.1f\n",
               gx[0], gx[1], gx[2], gx[3], gx[4]);
        printf("  notify(flag-data,       us): p50=%7.1f p90=%7.1f p95=%7.1f p99=%7.1f max=%7.1f\n",
               gn[0], gn[1], gn[2], gn[3], gn[4]);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // ---- teardown -----------------------------------------------------------
  gdr.destroy();
  rc.dereg_mr(mr_send);
  rc.dereg_mr(mr_recv);
  rc.dereg_mr(mr_flag);
  rc.destroy();
  cudaFree(d_send);
  cudaFree(d_recv);
  cudaFree(d_flags_raw);
  cudaFreeHost(h_flags);
  cudaFreeHost(h_probe);
  cudaFreeHost((void *)h_ack);
  MPI_Finalize();
  return 0;
}
