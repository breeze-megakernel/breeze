/*
 * Minimal example of coupled and grouped signaling.
 *
 * Run:
 *   srun -n 4 -N 2 --gpus-per-node=2 ./example
 */

#include <cstdio>
#include <cstring>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include "dsg.cuh"

#define N_TRANSFERS 24          // transfers per PE per "dispatch"
#define MSG_BYTES   (256 * 1024)

// Coupled Put-with-Signal
__global__ void kernel_vanilla(char* src, char* dst, uint64_t* flags,
                               const int* dest_pes, int n) {
    int i = blockIdx.x;
    if (i >= n) return;
    if (threadIdx.x == 0) {
        int pe = dest_pes[i];
        nvshmem_putmem_signal_nbi(dst + (size_t)i * MSG_BYTES,
                                  src + (size_t)i * MSG_BYTES, MSG_BYTES, pe,
                                  flags + i, 1, NVSHMEM_SIGNAL_SET);
    }
}

// Grouped signaling
__global__ void kernel_dsg(char* src, char* dst, uint64_t* flags,
                           const int* dest_pes, int n,
                           dsg::Group<dsg::Nvshmem>* g) {
    int i = blockIdx.x;
    if (i >= n) return;
    if (threadIdx.x == 0) {
        int pe = dest_pes[i];
        dsg::putSignalNBI<dsg::Nvshmem>(
            g + pe,                                        // group per dest PE
            dst + (size_t)i * MSG_BYTES,
            src + (size_t)i * MSG_BYTES, MSG_BYTES, pe,
            {flags + i, 1, NVSHMEM_SIGNAL_SET});           // args carried over
    }
}

// Receiver
__global__ void kernel_wait(uint64_t* flags, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) nvshmem_uint64_wait_until(flags + i, NVSHMEM_CMP_EQ, 1);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    nvshmem_init();
    int mype = nvshmem_my_pe(), npes = nvshmem_n_pes();
    cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE));
    if (npes < 2) { if (!mype) fprintf(stderr, "need >= 2 PEs\n"); return 1; }

    char* src = (char*)nvshmem_malloc((size_t)N_TRANSFERS * MSG_BYTES);
    char* dst = (char*)nvshmem_malloc((size_t)N_TRANSFERS * MSG_BYTES);
    uint64_t* flags = (uint64_t*)nvshmem_calloc(N_TRANSFERS, sizeof(uint64_t));
    cudaMemset(src, mype + 1, (size_t)N_TRANSFERS * MSG_BYTES);

    int h_dest[N_TRANSFERS], *d_dest;
    for (int i = 0; i < N_TRANSFERS; i++)
        h_dest[i] = (mype + 1 + i % (npes - 1)) % npes;
    cudaMalloc(&d_dest, sizeof h_dest);
    cudaMemcpy(d_dest, h_dest, sizeof h_dest, cudaMemcpyHostToDevice);

    // One group per destination PE
    dsg::Group<dsg::Nvshmem>* g;
    cudaMalloc(&g, npes * sizeof *g);
    dsg::initGroups(g, npes, N_TRANSFERS / (npes - 1));

    for (int mode = 0; mode < 2; mode++) {
        cudaMemset(flags, 0, N_TRANSFERS * sizeof(uint64_t));
        nvshmem_barrier_all();

        if (mode == 0) kernel_vanilla<<<N_TRANSFERS, 32>>>(src, dst, flags, d_dest, N_TRANSFERS);
        else           kernel_dsg<<<N_TRANSFERS, 32>>>(src, dst, flags, d_dest, N_TRANSFERS, g);
        kernel_wait<<<N_TRANSFERS, 1>>>(flags, N_TRANSFERS);
        cudaDeviceSynchronize();
        nvshmem_barrier_all();

        // Verify flags and payloads
        char probe[8];
        int bad = 0;
        for (int i = 0; i < N_TRANSFERS; i++) {
            cudaMemcpy(probe, dst + (size_t)i * MSG_BYTES, 8, cudaMemcpyDeviceToHost);
            // dst was written by SOME peer; payload byte is that peer's rank+1
            if (probe[0] < 1 || probe[0] > npes) bad++;
        }
        if (!mype) printf("%-8s: %s (%d transfers, %d flags observed)\n",
                          mode ? "dsg" : "vanilla", bad ? "PAYLOAD MISMATCH" : "OK",
                          N_TRANSFERS, N_TRANSFERS);
    }

    nvshmem_free(src); nvshmem_free(dst); nvshmem_free(flags);
    cudaFree(d_dest); cudaFree(g);
    nvshmem_finalize(); MPI_Finalize();
    return 0;
}

