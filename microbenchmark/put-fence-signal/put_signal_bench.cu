/*
 * Put-with-Signal microbenchmark.
 *
 * Modes
 *   A  nonblocking PUTs followed by one quiet
 *   B  Put-with-Signal per transfer
 *   C  nonblocking PUTs followed by one fence and the signals
 *
 * On IBRC, SIGNAL_SET requires NVSHMEM_IB_ENABLE_RELAXED_ORDERING=0.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

// ═══════════════════════════════════════════════════════════════════════════
// Mode A: N × put_nbi + 1 quiet (pipelined, no per-transfer signal)
// ═══════════════════════════════════════════════════════════════════════════
__global__ void kernel_put_nbi_quiet(
    char *local_buf,
    char *remote_buf,
    uint64_t *signals,
    const int *dest_pes,
    size_t msg_size,
    int N,
    int npes, int mype,
    int sig_op)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int nblocks = gridDim.x;

    for (int i = bid; i < N; i += nblocks) {
        int peer = dest_pes[i];
        char *src = local_buf + (size_t)i * msg_size;
        char *dst = remote_buf + (size_t)i * msg_size;

        // Stage data (simulate memory traffic)
        int4 *src4 = (int4 *)src;
        int4 *dst_staging = (int4 *)(local_buf + (size_t)(N + i) * msg_size);
        size_t elems = msg_size / sizeof(int4);
        for (size_t k = tid; k < elems; k += blockDim.x) {
            dst_staging[k] = src4[k];
        }
        __syncthreads();

        if (tid == 0) {
            nvshmem_putmem_nbi(dst, src, msg_size, peer);
        }
    }

    // Single quiet at the end. only one thread globally
    if (tid == 0 && bid == 0) {
        nvshmem_quiet();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Mode B: N × putmem_signal (blocking, coupled put+fence+signal+quiet)
// ═══════════════════════════════════════════════════════════════════════════
__global__ void kernel_putmem_signal(
    char *local_buf,
    char *remote_buf,
    uint64_t *signals,
    const int *dest_pes,
    size_t msg_size,
    int N,
    int npes, int mype,
    int sig_op)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int nblocks = gridDim.x;

    for (int i = bid; i < N; i += nblocks) {
        int peer = dest_pes[i];
        char *src = local_buf + (size_t)i * msg_size;
        char *dst = remote_buf + (size_t)i * msg_size;

        int4 *src4 = (int4 *)src;
        int4 *dst_staging = (int4 *)(local_buf + (size_t)(N + i) * msg_size);
        size_t elems = msg_size / sizeof(int4);
        for (size_t k = tid; k < elems; k += blockDim.x) {
            dst_staging[k] = src4[k];
        }
        __syncthreads();

        if (tid == 0) {
            uint64_t sig_val = 1ULL;
            nvshmem_putmem_signal(dst, src, msg_size, signals + i,
                                  sig_val, sig_op, peer);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Mode C: N × put_nbi + 1 fence + N × signal_op (decoupled)
// ═══════════════════════════════════════════════════════════════════════════
__global__ void kernel_decoupled(
    char *local_buf,
    char *remote_buf,
    uint64_t *signals,
    const int *dest_pes,
    size_t msg_size,
    int N,
    int npes, int mype,
    int sig_op)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int nblocks = gridDim.x;

    // Phase 1: All blocks issue put_nbi
    for (int i = bid; i < N; i += nblocks) {
        int peer = dest_pes[i];
        char *src = local_buf + (size_t)i * msg_size;
        char *dst = remote_buf + (size_t)i * msg_size;

        int4 *src4 = (int4 *)src;
        int4 *dst_staging = (int4 *)(local_buf + (size_t)(N + i) * msg_size);
        size_t elems = msg_size / sizeof(int4);
        for (size_t k = tid; k < elems; k += blockDim.x) {
            dst_staging[k] = src4[k];
        }
        __syncthreads();

        if (tid == 0) {
            nvshmem_putmem_nbi(dst, src, msg_size, peer);
        }
    }

    // Phase 2: Block 0 does fence + all signal_ops
    if (bid == 0 && tid == 0) {
        nvshmem_fence();
        for (int i = 0; i < N; i++) {
            int peer = dest_pes[i];
            uint64_t sig_val = 1ULL;
            nvshmemx_signal_op(signals + i, sig_val, sig_op, peer);
        }
        nvshmem_quiet();  // ensure all puts + signals actually complete
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Host. single-iteration kernel launches, host-side timing
// ═══════════════════════════════════════════════════════════════════════════

typedef void (*bench_kernel_t)(char*, char*, uint64_t*, const int*, size_t, int, int, int, int);

float run_bench(bench_kernel_t kernel, int nblocks, int threads,
                char *local_buf, char *remote_buf, uint64_t *signals,
                int *d_dest, size_t msg_size, int N, int npes, int mype,
                int warmup, int iters, int sig_op, cudaStream_t stream)
{
    // Warmup
    for (int i = 0; i < warmup; i++) {
        kernel<<<nblocks, threads, 0, stream>>>(
            local_buf, remote_buf, signals, d_dest,
            msg_size, N, npes, mype, sig_op);
        cudaStreamSynchronize(stream);
        nvshmem_barrier_all();
    }

    // Reset signals
    cudaMemsetAsync(signals, 0, N * sizeof(uint64_t), stream);
    cudaStreamSynchronize(stream);
    nvshmem_barrier_all();

    // Timed
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; i++) {
        kernel<<<nblocks, threads, 0, stream>>>(
            local_buf, remote_buf, signals, d_dest,
            msg_size, N, npes, mype, sig_op);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Get max across PEs
    float all_ms = 0;
    MPI_Allreduce(&ms, &all_ms, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    return all_ms / iters;
}

int main(int argc, char **argv)
{
    int N = 96;
    size_t msg_size = 2 * 1024 * 1024;
    int warmup = 20;
    int iters = 100;
    int nblocks = 107;
    int threads = 256;
    int sig_op = NVSHMEM_SIGNAL_ADD;   // default: atomic signal, safe under
                                       // relaxed ordering (the relaxed-ordering requirement, L3)
    const char *sig_op_name = "add";

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) msg_size = (size_t)atol(argv[2]);
    if (argc > 3) warmup = atoi(argv[3]);
    if (argc > 4) iters = atoi(argv[4]);
    if (argc > 5) nblocks = atoi(argv[5]);
    if (argc > 6) {
        if (strcmp(argv[6], "set") == 0) {
            sig_op = NVSHMEM_SIGNAL_SET;
            sig_op_name = "set";
        } else if (strcmp(argv[6], "add") != 0) {
            fprintf(stderr, "unknown signal op '%s' (use add|set)\n", argv[6]);
            return 1;
        }
    }

    MPI_Init(&argc, &argv);
    nvshmem_init();
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int dev = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(dev);

    if (mype == 0) {
        const char *ro_env = getenv("NVSHMEM_IB_ENABLE_RELAXED_ORDERING");
        printf("put_signal_bench: N=%d, msg_size=%zuB (%.1fMB), "
               "warmup=%d, iters=%d, nblocks=%d, npes=%d\n",
               N, msg_size, (double)msg_size / (1024 * 1024),
               warmup, iters, nblocks, npes);
        printf("signal_op=%s, NVSHMEM_IB_ENABLE_RELAXED_ORDERING=%s\n",
               sig_op_name, ro_env ? ro_env : "(unset, default 1)");
        if (sig_op == NVSHMEM_SIGNAL_SET && (!ro_env || ro_env[0] != '0')) {
            printf("WARNING: SET signal with relaxed ordering enabled is not a "
                   "sound configuration on IBRC (see file header). Use add, or "
                   "set NVSHMEM_IB_ENABLE_RELAXED_ORDERING=0.\n");
        }
        printf("Total data per iteration: %.1f MB\n\n",
               (double)N * msg_size / (1024 * 1024));
    }

    // Allocate symmetric heap
    size_t heap_size = (size_t)N * 2 * msg_size;
    char *local_buf = (char *)nvshmem_malloc(heap_size);
    char *remote_buf = (char *)nvshmem_malloc((size_t)N * msg_size);
    uint64_t *signals = (uint64_t *)nvshmem_calloc(N, sizeof(uint64_t));

    if (!local_buf || !remote_buf || !signals) {
        fprintf(stderr, "PE %d: nvshmem_malloc failed\n", mype);
        nvshmem_finalize();
        MPI_Finalize();
        return 1;
    }

    cudaMemset(local_buf, mype + 1, heap_size);

    // Build dest PE list: round-robin to remote PEs
    int *h_dest = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        int peer = (mype + 1 + (i % (npes - 1))) % npes;
        h_dest[i] = peer;
    }
    int *d_dest;
    cudaMalloc(&d_dest, N * sizeof(int));
    cudaMemcpy(d_dest, h_dest, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    nvshmem_barrier_all();

    // ── Benchmark each mode ──────────────────────────────────────────────
    const char *names[] = {
        "put_nbi + quiet  (pipelined)",
        "putmem_signal    (coupled)",
        "put_nbi + fence + signal (decoupled)"
    };
    bench_kernel_t kernels[] = {
        kernel_put_nbi_quiet,
        kernel_putmem_signal,
        kernel_decoupled
    };

    if (mype == 0) {
        printf("%-45s  %12s  %10s\n", "Mode", "ms/iter", "GB/s");
        printf("─────────────────────────────────────────────────────────────────────\n");
    }

    for (int m = 0; m < 3; m++) {
        cudaMemsetAsync(signals, 0, N * sizeof(uint64_t), stream);
        cudaStreamSynchronize(stream);
        nvshmem_barrier_all();

        float per_iter_ms = run_bench(kernels[m], nblocks, threads,
                                       local_buf, remote_buf, signals,
                                       d_dest, msg_size, N, npes, mype,
                                       warmup, iters, sig_op, stream);

        double total_bytes = (double)N * msg_size;
        double bw = (total_bytes / (per_iter_ms / 1000.0)) / (1024.0 * 1024 * 1024);

        if (mype == 0) {
            printf("%-45s  %9.3f ms  %7.2f GB/s\n", names[m], per_iter_ms, bw);
        }
        nvshmem_barrier_all();
    }

    // ── Sweep N at fixed msg_size ────────────────────────────────────────
    if (mype == 0) {
        printf("\n=== Sweep: N transfers at msg_size=%zuB (%.1fMB) ===\n",
               msg_size, (double)msg_size / (1024 * 1024));
        printf("%-6s  %15s  %15s  %15s  %15s  %15s  %15s\n",
               "N", "pipelined(ms)", "pipe(GB/s)",
               "coupled(ms)", "coup(GB/s)",
               "decoupled(ms)", "decoup(GB/s)");
    }

    int N_sweep[] = {1, 2, 4, 8, 16, 32, 64, 96, 128};
    int n_sweep = sizeof(N_sweep) / sizeof(N_sweep[0]);

    for (int ni = 0; ni < n_sweep; ni++) {
        int Ns = N_sweep[ni];
        if (Ns > N) break;

        float times[3];
        for (int m = 0; m < 3; m++) {
            cudaMemsetAsync(signals, 0, N * sizeof(uint64_t), stream);
            cudaStreamSynchronize(stream);
            nvshmem_barrier_all();

            times[m] = run_bench(kernels[m], nblocks, threads,
                                  local_buf, remote_buf, signals,
                                  d_dest, msg_size, Ns, npes, mype,
                                  warmup, iters, sig_op, stream);
            nvshmem_barrier_all();
        }

        if (mype == 0) {
            double bytes = (double)Ns * msg_size;
            printf("%-6d  %12.3f ms  %12.2f GB/s  %12.3f ms  %12.2f GB/s  %12.3f ms  %12.2f GB/s\n",
                   Ns,
                   times[0], (bytes / (times[0] / 1000.0)) / (1024.0*1024*1024),
                   times[1], (bytes / (times[1] / 1000.0)) / (1024.0*1024*1024),
                   times[2], (bytes / (times[2] / 1000.0)) / (1024.0*1024*1024));
        }
    }

    // Cleanup
    cudaStreamDestroy(stream);
    free(h_dest);
    cudaFree(d_dest);
    nvshmem_free(local_buf);
    nvshmem_free(remote_buf);
    nvshmem_free(signals);
    nvshmem_finalize();
    MPI_Finalize();
    return 0;
}
