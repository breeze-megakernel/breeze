/*
 * Copyright (c) 2026
 * SPDX-License-Identifier: Apache-2.0
 *
 * Grouped signaling for NVSHMEM.
 * Each transfer issues a nonblocking PUT and stores its signal descriptor.
 * The last arrival issues one ordering point followed by all stored signals.
 *
 * A group must not be reused until the previous group has completed.
 * 
 */

#ifndef DSG_CUH
#define DSG_CUH

#include <vector>

#include <cuda_runtime.h>
#include <nvshmem.h>

#include <cuda/atomic>
#include <cuda/std/utility>

namespace dsg
{
  enum class Flush { Carrier, Leader };

  template <class Backend>
  struct Group {
    uint32_t slots;     // slot allocator for banked signals
    uint32_t done;      // completed registrations
    uint32_t expected;  // registrations per epoch (== capacity of sigs)
    uint32_t pad;
    typename Backend::Signal* sigs; // [expected], device-resident
  };

  // Issue one ordering point, send stored signals, then reset the group.
  template <class Backend>
  __device__ __forceinline__
  void flushNow(Group<Backend>* __restrict__ const& g) {
    Backend::orderPoint();
    const auto k = g->expected;
    for (uint32_t i = 0; i < k; ++i) {
      Backend::signal(g->sigs[i]);
    }
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device>{g->slots}
      .store(0, cuda::memory_order_relaxed);
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device>{g->done}
      .store(0, cuda::memory_order_relaxed);
  }

  // Store a signal descriptor and count arrivals.
  // The acq_rel update publishes descriptors before the final flush.
  template <class Backend>
  __device__ __forceinline__
  bool arrive(Group<Backend>* __restrict__ const& g,
              const typename Backend::Signal& sig) {
    const auto slot =
      cuda::atomic_ref<uint32_t, cuda::thread_scope_device>{g->slots}
        .fetch_add(1, cuda::memory_order_relaxed);
    g->sigs[slot] = sig;
    return cuda::atomic_ref<uint32_t, cuda::thread_scope_device>{g->done}
      .fetch_add(1, cuda::memory_order_acq_rel) + 1 == g->expected;
  }

  // Issue the data transfer and defer its signal.
  template <class Backend, Flush policy = Flush::Carrier, typename... PutArgs>
  __device__ __forceinline__
  void putSignalNBI(Group<Backend>* __restrict__ const& g,
                    const typename Backend::Signal& sig,
                    PutArgs&&... put) {
    Backend::putNBI(cuda::std::forward<PutArgs>(put)...);
    const bool last = arrive(g, sig);
    if constexpr (policy == Flush::Carrier) {
      if (last) { flushNow(g); }
    }
  }

  // Register a signal without a data transfer.
  template <class Backend, Flush policy = Flush::Carrier>
  __device__ __forceinline__
  void signalNBI(Group<Backend>* __restrict__ const& g,
                 const typename Backend::Signal& sig) {
    const bool last = arrive(g, sig);
    if constexpr (policy == Flush::Carrier) {
      if (last) { flushNow(g); }
    }
  }

  // Wait for all group members, then flush.
  template <class Backend>
  __device__ __forceinline__
  void flushWait(Group<Backend>* __restrict__ const& g) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> d{g->done};
    while (d.load(cuda::memory_order_acquire) < g->expected) {}
    flushNow(g);
  }

  // NVSHMEM backend
  struct Nvshmem {
    struct Signal {
      uint64_t* addr; // symmetric address of the flag at the destination
      uint64_t val;
      int op;         // NVSHMEM_SIGNAL_SET / NVSHMEM_SIGNAL_ADD
      int pe;         // NVSHMEM PE id
    };
    template <typename Dst, typename Src>
    static __device__ __forceinline__
    void putNBI(Dst* __restrict__ const dst, const Src* __restrict__ const src,
                const size_t& bytes, const int& pe) {
      nvshmem_putmem_nbi(dst, src, bytes, pe);
    }
    static __device__ __forceinline__
    void orderPoint() {
      // Stock NVSHMEM drains the proxy. The patched transport uses NIC ordering.
      nvshmem_fence();
    }
    static __device__ __forceinline__
    void signal(const Signal& s) {
      nvshmemx_signal_op(s.addr, s.val, s.op, s.pe);
    }
  };

  // Initialize one group per PE. dSlab holds world * expected descriptors.
  template <class Backend>
  __host__ __forceinline__
  void initGroups(Group<Backend>* const dGroups,
                  typename Backend::Signal* const dSlab,
                  const int world, const uint32_t expected,
                  cudaStream_t stream = nullptr) {
    std::vector<Group<Backend>> h(world);
    for (int p = 0; p < world; ++p) {
      h[p] = {0U, 0U, expected, 0U,
              dSlab + static_cast<size_t>(p) * expected};
    }
    cudaMemcpyAsync(dGroups, h.data(), world * sizeof(Group<Backend>),
                    cudaMemcpyHostToDevice, stream);
  }
} // namespace dsg
#endif // DSG_CUH
