// rdma_common.hpp. types shared by the verbs and libfabric backends.
#pragma once

#include <mpi.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define VCHK(cond, msg)                                                        \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "[%s:%d] FATAL: %s (errno=%d: %s)\n", __FILE__,          \
              __LINE__, (msg), errno, strerror(errno));                        \
      MPI_Abort(MPI_COMM_WORLD, 1);                                            \
    }                                                                          \
  } while (0)

// Remote memory region descriptor, exchanged between peers.
// key is 64-bit: CXI provider keys do not fit in verbs' 32-bit rkey.
struct MrDesc {
  uint64_t addr;
  uint64_t key;
};

// Local handle to a registered region.
//   verbs:     desc holds lkey (cast), impl is ibv_mr*
//   libfabric: desc is fi_mr_desc(), impl is fid_mr*
struct MemRegion {
  void *desc = nullptr;
  uint64_t key = 0;
  void *impl = nullptr;
};

// Both backends expose a class named RdmaCtx with this interface:
//
//   void open(const char* dev, int rank, int nranks, int qps_per_peer);
//   const char* dev_name() const;
//   const char* link_name() const;
//   void create_queues(int send_depth, int recv_depth);
//   MemRegion reg_mr(void* ptr, size_t len, bool relaxed, bool remote_event);
//   static std::vector<MrDesc> exchange_mr(const MemRegion&, void*, int);
//   void connect();
//   void post_recvs(int n);                       // no-op on libfabric
//   void post_write(peer,q,lmr,laddr,raddr,rkey,len,signaled,id);
//   void post_write_imm(peer,q,lmr,laddr,raddr,rkey,len,imm,signaled,id);
//   void post_flag(peer,q,value,raddr,rkey,fence,signaled);
//   void poll_send(int n);
//   int  poll_recv(uint64_t* imm_out, int max);   // returns count
//   void destroy();