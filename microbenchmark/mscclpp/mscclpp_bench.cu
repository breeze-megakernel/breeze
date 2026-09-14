/*
 * MSCCL++ PortChannel signaling microbenchmark.
 *
 * Modes
 *   A  pipelined PUTs followed by flush
 *   B  putWithSignal with in-QP ordering
 *   C  putWithSignalAndFlush per transfer
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <memory>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include <mscclpp/core.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <mscclpp/gpu_utils.hpp>

using namespace mscclpp;

#define CUDACHECK(cmd) do {                                          \
    cudaError_t e = (cmd);                                           \
    if (e != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(e));          \
        exit(1);                                                     \
    }                                                                \
} while(0)

#define MPICHECK(cmd) do {                                           \
    int e = (cmd);                                                   \
    if (e != MPI_SUCCESS) {                                          \
        fprintf(stderr, "MPI error %s:%d: %d\n",                    \
                __FILE__, __LINE__, e);                               \
        exit(1);                                                     \
    }                                                                \
} while(0)

// ═════════════════════════════════════════════════════════════════════
// Device-side: channel handles in constant memory (max 128 channels)
// ═════════════════════════════════════════════════════════════════════

#define MAX_CHANNELS 128
__device__ DeviceHandle<PortChannel> d_channels[MAX_CHANNELS];

// Staging copy identical to the NVSHMEM bench: all threads copy the
// transfer's payload from src region [i] into staging region [N+i],
// then thread 0 issues the RDMA. localBuf holds 2*N*msg_size.
__device__ __forceinline__ void stage_transfer(char *localBuf, size_t msg_size,
                                               int i, int N, int tid, int nthreads) {
    int4 *src4 = (int4 *)(localBuf + (size_t)i * msg_size);
    int4 *dst_staging = (int4 *)(localBuf + (size_t)(N + i) * msg_size);
    size_t elems = msg_size / sizeof(int4);
    for (size_t k = tid; k < elems; k += nthreads) {
        dst_staging[k] = src4[k];
    }
}

// ═════════════════════════════════════════════════════════════════════
// Mode A: N × put + flush(all peers) (pipelined puts, no signals)
// ═════════════════════════════════════════════════════════════════════
__global__ void kernel_put_flush(char *localBuf, size_t msg_size, int N,
                                 int nPeers, int doStage) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    for (int i = bid; i < N; i += gridDim.x) {
        uint64_t offset = (uint64_t)i * msg_size;

        if (doStage) {
            stage_transfer(localBuf, msg_size, i, N, tid, blockDim.x);
            __syncthreads();
        }

        if (tid == 0) {
            d_channels[i].put(offset, offset, msg_size);
        }
    }

    // Drain EVERY distinct peer channel. matches nvshmem_quiet, which
    // drains the whole proxy, not one connection. Channels 0..nPeers-1
    // cover all connections used (transfer i -> channel i % nPeers).
    if (bid == 0 && tid == 0) {
        int nc = (nPeers < N) ? nPeers : N;
        for (int p = 0; p < nc; p++) d_channels[p].flush();
    }
}

// ═════════════════════════════════════════════════════════════════════
// Mode B: N × putWithSignal + flush(all peers) (in-QP ordering)
// ═════════════════════════════════════════════════════════════════════
__global__ void kernel_putWithSignal(char *localBuf, size_t msg_size, int N,
                                     int nPeers, int doStage) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    for (int i = bid; i < N; i += gridDim.x) {
        uint64_t offset = (uint64_t)i * msg_size;

        if (doStage) {
            stage_transfer(localBuf, msg_size, i, N, tid, blockDim.x);
            __syncthreads();
        }

        if (tid == 0) {
            // Single FIFO push: proxy posts WRITE + ATOMIC on same QP.
            // In-QP ordering guarantees data arrives before signal.
            // No proxy drain, no fence. this is the key MSCCL++ mechanism.
            d_channels[i].putWithSignal(offset, offset, msg_size);
        }
    }

    if (bid == 0 && tid == 0) {
        int nc = (nPeers < N) ? nPeers : N;
        for (int p = 0; p < nc; p++) d_channels[p].flush();
    }
}

// ═════════════════════════════════════════════════════════════════════
// Mode C: N × putWithSignalAndFlush (per-transfer CQ drain)
// This approximates NVSHMEM vanilla behavior: each transfer
// includes a CQ drain before the proxy can proceed to the next.
// ═════════════════════════════════════════════════════════════════════
__global__ void kernel_putWithSignalAndFlush(char *localBuf, size_t msg_size, int N,
                                             int nPeers, int doStage) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    for (int i = bid; i < N; i += gridDim.x) {
        uint64_t offset = (uint64_t)i * msg_size;

        if (doStage) {
            stage_transfer(localBuf, msg_size, i, N, tid, blockDim.x);
            __syncthreads();
        }

        if (tid == 0) {
            // Each call pushes TriggerData|TriggerFlag|TriggerSync.
            // The TriggerSync causes the GPU to spin-wait until the proxy
            // has drained the CQ. equivalent to NVSHMEM's per-signal fence.
            d_channels[i].putWithSignalAndFlush(offset, offset, msg_size);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// Host. single-iteration kernel launches with host-side timing
// (matches ../put-fence-signal/put_signal_bench.cu pattern)
// ═════════════════════════════════════════════════════════════════════

typedef void (*bench_kernel_t)(char*, size_t, int, int, int);

float run_bench(bench_kernel_t kernel, int nblocks, int threads,
                char *localBuf, size_t msg_size, int N, int nPeers, int doStage,
                int warmup, int iters, cudaStream_t stream) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        kernel<<<nblocks, threads, 0, stream>>>(localBuf, msg_size, N, nPeers, doStage);
        CUDACHECK(cudaStreamSynchronize(stream));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Timed
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    CUDACHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; i++) {
        kernel<<<nblocks, threads, 0, stream>>>(localBuf, msg_size, N, nPeers, doStage);
    }
    CUDACHECK(cudaEventRecord(stop, stream));
    CUDACHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDACHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));

    // Max across all PEs (bottleneck)
    float all_ms = 0;
    MPI_Allreduce(&ms, &all_ms, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    return all_ms / iters;
}

// ═════════════════════════════════════════════════════════════════════
// MSCCL++ setup (what NVSHMEM does automatically behind nvshmem_init)
// ═════════════════════════════════════════════════════════════════════

// Map local GPU index → IB transport (IB0..IB7)
Transport getIBTransport(int localGpuIdx) {
    static const Transport ibs[] = {
        Transport::IB0, Transport::IB1, Transport::IB2, Transport::IB3,
        Transport::IB4, Transport::IB5, Transport::IB6, Transport::IB7,
    };
    int nIb = getIBDeviceCount();
    if (nIb == 0) {
        fprintf(stderr, "No IB devices found\n");
        exit(1);
    }
    return ibs[localGpuIdx % nIb];
}

int main(int argc, char **argv) {
    // ── Defaults (match the NVSHMEM bench) ──
    int N        = 96;
    size_t msg_size = 2 * 1024 * 1024;
    int warmup   = 20;
    int iters    = 100;
    int threads  = 256;
    int doStage  = 1;   // staging parity with the NVSHMEM bench; 0 disables

    if (argc > 1) N        = atoi(argv[1]);
    if (argc > 2) msg_size = (size_t)atol(argv[2]);
    if (argc > 3) warmup   = atoi(argv[3]);
    if (argc > 4) iters    = atoi(argv[4]);
    if (argc > 5) doStage  = atoi(argv[5]);

    if (N > MAX_CHANNELS) {
        fprintf(stderr, "N=%d exceeds MAX_CHANNELS=%d (d_channels is fixed size)\n",
                N, MAX_CHANNELS);
        return 1;
    }

    // ── MPI init ──
    MPICHECK(MPI_Init(&argc, &argv));
    int rank, worldSize;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));

    // ── Local rank / GPU ──
    int localRank = 0;
    {
        MPI_Comm localComm;
        MPICHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                      rank, MPI_INFO_NULL, &localComm));
        MPICHECK(MPI_Comm_rank(localComm, &localRank));
        MPI_Comm_free(&localComm);
    }
    CUDACHECK(cudaSetDevice(localRank));

    int nGpusPerNode = 0;
    CUDACHECK(cudaGetDeviceCount(&nGpusPerNode));
    int nNodes = worldSize / nGpusPerNode;
    int thisNode = rank / nGpusPerNode;

    if (rank == 0) {
        printf("mscclpp_bench: N=%d, msg_size=%zuB (%.1fMB), "
               "warmup=%d, iters=%d, npes=%d, nodes=%d, staging=%s\n",
               N, msg_size, (double)msg_size / (1024*1024),
               warmup, iters, worldSize, nNodes, doStage ? "on" : "off");
        printf("Total data per iteration: %.1f MB\n\n",
               (double)N * msg_size / (1024*1024));
    }

    // ── Buffer allocation ──
    // sendBuf holds 2*N regions: [0..N) payload, [N..2N) staging target,
    // matching the NVSHMEM bench's local_buf layout.
    size_t bufSize = (size_t)N * msg_size;
    void *sendBuf = nullptr, *recvBuf = nullptr;
    CUDACHECK(cudaMalloc(&sendBuf, 2 * bufSize));
    CUDACHECK(cudaMalloc(&recvBuf, bufSize));
    CUDACHECK(cudaMemset(sendBuf, rank + 1, 2 * bufSize));
    CUDACHECK(cudaMemset(recvBuf, 0, bufSize));

    // ── MSCCL++ bootstrap ──
    // (In NVSHMEM this is hidden inside nvshmem_init)
    auto bootstrap = std::make_shared<TcpBootstrap>(rank, worldSize);
    UniqueId uniqueId;
    if (rank == 0) uniqueId = TcpBootstrap::createUniqueId();
    MPICHECK(MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));
    bootstrap->initialize(uniqueId);

    // ── Communicator ──
    Communicator comm(bootstrap);

    // ── Register memory with IB transport ──
    // Only the payload half of sendBuf is the RDMA source; register it all
    // anyway (registration size does not affect the timed path).
    Transport ibTransport = getIBTransport(localRank);
    RegisteredMemory sendRegMem = comm.registerMemory(sendBuf, bufSize, ibTransport);
    RegisteredMemory recvRegMem = comm.registerMemory(recvBuf, bufSize, ibTransport);

    // ── Connect to remote peers (skip same-node peers, IB only) ──
    // Build dest PE list: round-robin to remote PEs (same as NVSHMEM bench)
    std::vector<int> remotePeers;
    for (int r = 0; r < worldSize; r++) {
        if (r == rank) continue;
        if (r / nGpusPerNode == thisNode) continue;  // skip same-node (CudaIpc)
        remotePeers.push_back(r);
    }
    int nRemotePeers = remotePeers.size();
    if (nRemotePeers == 0) {
        if (rank == 0) fprintf(stderr, "No remote peers (need multi-node)\n");
        MPI_Finalize();
        return 1;
    }

    // Each rank connects to every remote rank, sends its recvBuf handle,
    // receives the remote rank's recvBuf handle.
    std::vector<std::shared_future<Connection>> connFutures;
    std::vector<std::shared_future<RegisteredMemory>> remoteMemFutures;
    for (int r : remotePeers) {
        connFutures.push_back(comm.connect(ibTransport, r));
        comm.sendMemory(recvRegMem, r);
        remoteMemFutures.push_back(comm.recvMemory(r));
    }

    std::vector<Connection> connections;
    for (auto& f : connFutures) connections.push_back(f.get());
    std::vector<RegisteredMemory> remoteRecvMems;
    for (auto& f : remoteMemFutures) remoteRecvMems.push_back(f.get());

    // ── ProxyService + PortChannels ──
    // One PortChannel per remote peer (each has its own QP).
    // Transfers are assigned to channels round-robin (same as NVSHMEM dest_pes).
    auto proxyService = std::make_shared<ProxyService>();

    struct PeerChannel {
        SemaphoreId semId;
        MemoryId dstMemId;
        MemoryId srcMemId;
    };
    std::vector<PeerChannel> peerChannels;

    for (int p = 0; p < nRemotePeers; p++) {
        SemaphoreId semId = proxyService->buildAndAddSemaphore(comm, connections[p]);
        MemoryId dstMemId = proxyService->addMemory(remoteRecvMems[p]);
        MemoryId srcMemId = proxyService->addMemory(sendRegMem);
        peerChannels.push_back({semId, dstMemId, srcMemId});
    }

    // Build N device handles: transfer i uses peer channel (i % nRemotePeers)
    // This matches the NVSHMEM pattern: h_dest[i] = (mype + 1 + (i % (npes-1))) % npes
    std::vector<DeviceHandle<PortChannel>> allHandles(N);
    for (int i = 0; i < N; i++) {
        int p = i % nRemotePeers;
        auto pc = proxyService->portChannel(
            peerChannels[p].semId, peerChannels[p].dstMemId, peerChannels[p].srcMemId);
        allHandles[i] = deviceHandle(pc);
    }

    proxyService->startProxy();

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    // ── Benchmark ────────────────────────────────────────────────────────
    const char *modeNames[] = {
        "put + flush           (pipelined)",
        "putWithSignal + flush (in-QP ord)",
        "putWithSignalAndFlush (per-drain)",
    };
    bench_kernel_t kernels[] = {
        kernel_put_flush,
        kernel_putWithSignal,
        kernel_putWithSignalAndFlush,
    };

    // Upload channel handles for N transfers
    auto uploadChannels = [&](int Ns) {
        CUDACHECK(cudaMemcpyToSymbol(d_channels, allHandles.data(),
                                     sizeof(DeviceHandle<PortChannel>) * Ns));
    };

    char *localBuf = (char *)sendBuf;

    // ── Fixed-N benchmark ──
    {
        uploadChannels(N);
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            printf("%-45s  %12s  %10s\n", "Mode", "ms/iter", "GB/s");
            printf("──────────────────────────────────────────────"
                   "─────────────────────────\n");
        }

        for (int m = 0; m < 3; m++) {
            MPI_Barrier(MPI_COMM_WORLD);
            int nblocks = N;  // one block per transfer
            float per_iter_ms = run_bench(kernels[m], nblocks, threads,
                                          localBuf, msg_size, N, nRemotePeers,
                                          doStage, warmup, iters, stream);
            double total_bytes = (double)N * msg_size;
            double bw = (total_bytes / (per_iter_ms / 1000.0)) / (1024.0*1024*1024);
            if (rank == 0) {
                printf("%-45s  %9.3f ms  %7.2f GB/s\n",
                       modeNames[m], per_iter_ms, bw);
            }
        }
    }

    // ── Sweep: N transfers at fixed msg_size ──
    {
        int N_sweep[] = {1, 2, 4, 8, 16, 32, 64, 96, 128};
        int n_sweep = sizeof(N_sweep) / sizeof(N_sweep[0]);

        if (rank == 0) {
            printf("\n=== Sweep: N transfers at msg_size=%zuB (%.1fMB), %d nodes ===\n",
                   msg_size, (double)msg_size/(1024*1024), nNodes);
            printf("%-6s  %15s  %15s  %15s  %15s  %15s  %15s\n",
                   "N",
                   "pipelined(ms)", "pipe(GB/s)",
                   "inQP(ms)", "inQP(GB/s)",
                   "perDrain(ms)", "drain(GB/s)");
        }

        for (int ni = 0; ni < n_sweep; ni++) {
            int Ns = N_sweep[ni];
            if (Ns > N) break;
            uploadChannels(Ns);

            float times[3];
            for (int m = 0; m < 3; m++) {
                MPI_Barrier(MPI_COMM_WORLD);
                int nblocks = Ns;
                times[m] = run_bench(kernels[m], nblocks, threads,
                                     localBuf, msg_size, Ns, nRemotePeers,
                                     doStage, warmup, iters, stream);
            }

            if (rank == 0) {
                double bytes = (double)Ns * msg_size;
                printf("%-6d  %12.3f ms  %12.2f GB/s"
                       "  %12.3f ms  %12.2f GB/s"
                       "  %12.3f ms  %12.2f GB/s\n",
                       Ns,
                       times[0], (bytes/(times[0]/1000.0))/(1024.0*1024*1024),
                       times[1], (bytes/(times[1]/1000.0))/(1024.0*1024*1024),
                       times[2], (bytes/(times[2]/1000.0))/(1024.0*1024*1024));
            }
        }
    }

    // ── 2D sweep: msg_size × concurrency (the throughput sweep-style plot) ──
    {
        size_t msgSweep[] = {4096, 32768, 262144, 1048576, 4194304};
        int nMsgSweep = sizeof(msgSweep) / sizeof(msgSweep[0]);
        int N_sweep[] = {1, 2, 4, 8, 16, 32, 64, 96, 128};
        int n_sweep = sizeof(N_sweep) / sizeof(N_sweep[0]);

        if (rank == 0) {
            printf("\n=== 2D Sweep: msg_size × concurrency, %d nodes ===\n", nNodes);
            printf("# CSV output for plotting (the throughput sweep)\n");
            printf("# staging=%s\n", doStage ? "on" : "off");
            printf("msg_bytes,concurrency,nodes,"
                   "put_ms,put_gbps,"
                   "putWithSignal_ms,pws_gbps,signaling_efficiency,"
                   "putWithSigFlush_ms,pwsf_gbps,pwsf_efficiency\n");
        }

        for (int mi = 0; mi < nMsgSweep; mi++) {
            size_t M = msgSweep[mi];
            for (int ni = 0; ni < n_sweep; ni++) {
                int Ns = N_sweep[ni];
                if (Ns > N) break;
                if ((size_t)Ns * M > bufSize) continue;  // skip if buffer too small

                uploadChannels(Ns);
                float times[3];
                for (int m = 0; m < 3; m++) {
                    MPI_Barrier(MPI_COMM_WORLD);
                    times[m] = run_bench(kernels[m], Ns, threads,
                                         localBuf, M, Ns, nRemotePeers,
                                         doStage, warmup, iters, stream);
                }

                if (rank == 0) {
                    double bytes = (double)Ns * M;
                    double bw0 = (bytes/(times[0]/1000.0))/(1024.0*1024*1024);
                    double bw1 = (bytes/(times[1]/1000.0))/(1024.0*1024*1024);
                    double bw2 = (bytes/(times[2]/1000.0))/(1024.0*1024*1024);
                    // Signaling efficiency = put_only_time / signaled_time
                    // (>1 shouldn't happen; <1 means signal adds overhead)
                    double eff1 = times[0] / times[1];
                    double eff2 = times[0] / times[2];
                    printf("%zu,%d,%d,"
                           "%.4f,%.3f,"
                           "%.4f,%.3f,%.4f,"
                           "%.4f,%.3f,%.4f\n",
                           M, Ns, nNodes,
                           times[0], bw0,
                           times[1], bw1, eff1,
                           times[2], bw2, eff2);
                    fflush(stdout);
                }
            }
        }
    }

    // ── Cleanup ──
    proxyService->stopProxy();
    CUDACHECK(cudaStreamDestroy(stream));
    CUDACHECK(cudaFree(sendBuf));
    CUDACHECK(cudaFree(recvBuf));
    MPICHECK(MPI_Finalize());
    return 0;
}

