
Test code for demonstrating the performance difference for row-wise versus column-wise for CPU and GPU.
`LayoutLeft` represents a struct of arrays, and `LayoutRight` represents an array of structs.

# Building
Things are controlled by makefile parameters.
The following commands compile executables for CPU and GPU respectively on Tellico.
``` bash
make KOKKOS_DEVICES=OpenMP
make KOKKOS_DEVICES=Cuda
```

# Running Tests
To ensure that memory bandwidth is correctly measured, each test dataset should be larger than L3 cache for CPU tests.
On Tellico, this requires at least 15,728,640 doubles.

The following commands will submit tests for CPU and GPU respectively to LSF.
``` bash
OMP_PROC_BIND=spread OMP_PLACES=threads bsub -I ./test.h
ost 10000000 10
bsub -I -gpu "num=1" ./test.cuda 10000000 10
```
