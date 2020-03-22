/*
 * Based on code from Kokkos, licenced under the 3-clause BSD licence
 */

#include<Kokkos_Core.hpp>
#include<cstdlib>
#include<iostream>

#include "copy.hpp"
#include "euler_particle.hpp"


// t values for 99% confidence iterval
constexpr double t_values_99[] = {
		63.657, 9.925, 5.841, 4.604, 4.032, 3.707, 3.49, 3.355, 3.250, 3.169,
		 3.106, 3.055, 3.012, 2.977, 2.947, 2.921, 2.898, 2.878, 2.861, 2.845,
		 2.831, 2.819, 2.807, 2.797, 2.787, 2.779, 2.771, 2.763, 2.756, 2.750,
		 2.744, 2.738, 2.733, 2.728, 2.724, 2.719, 2.715, 2.712, 2.708, 2.704
	};

// Computes mean, standard deviation, ect of the execution times for the given test
template<class Test>
void compute_stats(Test& t, size_t tests) {
	// total time in ns
	size_t total = 0;
	for (size_t time : t.times) {
		total += time;
	}
	// mean time in ns
	const size_t mean = total/tests;

	// ensure that overflow does not happen
	double total_varience = 0;
	for (size_t time : t.times) {
		int64_t diff = int64_t(time) - int64_t(mean);
		total_varience += double(diff)*double(diff);
	}
	// sample standard deviation
	const double std_dev = std::sqrt(total_varience/(tests-1));

	const double ci_99 = t_values_99[(tests <= 40?tests:40)-1]*std_dev/std::sqrt(tests);

	std::cout << "Mean: " << mean/1000.0/1000.0 << " (ms); "
			<< "Stdev: " << std_dev/1000.0/1000.0  << " (ms); "
			<< "99% CI: " << ci_99/1000.0/1000.0 << " (ms)" <<  std::endl;
}

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

	std::cout << "copy  left:  ";
	compute_stats(copy_left, tests);
	std::cout << "copy  right: ";
	compute_stats(copy_right, tests);
	std::cout << "euler left:  ";
	compute_stats(euler_left, tests);
	std::cout << "euler right: ";
	compute_stats(euler_right, tests);
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

