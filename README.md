
Test code for demonstrating the performance difference for row-wise versus column-wise for CPU and GPU.

# Building
Things are controlled by makefile parameters.
The following commands compile executables for CPU and GPU respectively on Tellico.
``` bash
make KOKKOS_DEVICES=OpenMP
make KOKKOS_DEVICES=Cuda
```
