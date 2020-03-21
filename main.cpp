/*
 * Based on code from Kokkos, licenced under the 3-clause BSD licence
 */

#include<Kokkos_Core.hpp>
#include<cstdlib>

#include "copy.hpp"
#include "euler_particle.hpp"

void run_tests(size_t n, size_t tests) {
	copy<Kokkos::LayoutLeft> copy_left (n);
	copy<Kokkos::LayoutRight> copy_right (n);
	euler_particles<Kokkos::LayoutLeft> euler_left (n);
	euler_particles<Kokkos::LayoutRight> euler_right (n);

	for (size_t i = 0; i < tests; i++) {
		copy_left.test();
		copy_right.test();
		euler_left.test();
		euler_right.test();
	}

	// TODO compute mean, stdev, etc
}

int main(int argc, char* argv[]) {
	Kokkos::initialize();

	if (argc < 3) {
		printf("Arguments: N tests\n");
		printf("  N:   Number of vector entries\n");
		printf("  tests: number of times each kernels should be run\n");
	}

	const size_t n = atoi(argv[1]);
	const size_t tests = atoi(argv[2]);

	run_tests(n, tests);

	Kokkos::finalize();
}

