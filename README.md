
# Building
Things are controlled by makefile parameters.
The following commands compile executables for CPU and GPU respectively on Tellico.
``` bash
make KOKKOS_DEVICES=OpenMP
make KOKKOS_DEVICES=Cuda
```
