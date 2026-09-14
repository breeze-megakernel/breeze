# Decoupled signaling

`dsg.cuh` provides a header-only device API for grouped signaling with CUDA and NVSHMEM.

Each transfer issues a nonblocking PUT and stores its signal descriptor. The last transfer in a group issues one ordering point and then the stored signals.

## Example

```cpp
dsg::Group<dsg::Nvshmem>* g = ...;
dsg::initGroups(g, world, groupSize);
dsg::putSignalNBI<dsg::Nvshmem>(g + pe, dst, src, bytes, pe,
                                {flag, val, NVSHMEM_SIGNAL_SET});
```

`example.cu` includes the baseline and grouped versions.

## Build and run

```bash
module load PrgEnv-gnu cudatoolkit
nvcc -o example example.cu -I. -I$NVSHMEM_HOME/include -I$CRAY_MPICH_DIR/include \
     -L$NVSHMEM_HOME/lib -lnvshmem_host -lnvshmem_device \
     -L$CRAY_MPICH_DIR/lib -lmpi_gnu -rdc=true \
     -gencode arch=compute_80,code=sm_80 \
     -Xlinker -rpath=$NVSHMEM_HOME/lib -Xlinker --allow-shlib-undefined
srun -n 4 -N 2 --gpus-per-node=2 ./example
```

Expected output contains `vanilla : OK` and `dsg : OK`.
