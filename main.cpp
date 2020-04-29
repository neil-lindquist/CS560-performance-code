/*
 * Based on code from Kokkos, licenced under the 3-clause BSD licence
 */

#include<Kokkos_Core.hpp>
#include<cstdlib>
#include<iostream>

#include "copy.hpp"
#include "copy_mixed.hpp"
#include "euler_particle.hpp"
#include "capacity.hpp"


// t values for 99% confidence iterval
constexpr double t_values_99[] = {
    63.657, 9.925, 5.841, 4.604, 4.032, 3.707, 3.49, 3.355, 3.250, 3.169,
     3.106, 3.055, 3.012, 2.977, 2.947, 2.921, 2.898, 2.878, 2.861, 2.845,
     2.831, 2.819, 2.807, 2.797, 2.787, 2.779, 2.771, 2.763, 2.756, 2.750,
     2.744, 2.738, 2.733, 2.728, 2.724, 2.719, 2.715, 2.712, 2.708, 2.704
  };

// Computes mean, standard deviation, ect of the execution times for the given test
template<class Test>
void compute_stats(const Test& t, const size_t trials) {
  // total time in ns
  size_t total = 0;
  for (size_t time : t.times) {
    total += time;
  }
  // mean time in ns
  const size_t mean = total/trials;

  // ensure that overflow does not happen
  double total_varience = 0;
  for (size_t time : t.times) {
    int64_t diff = int64_t(time) - int64_t(mean);
    total_varience += double(diff)*double(diff);
  }
  // sample standard deviation
  const double std_dev = std::sqrt(total_varience/(trials-1));

  const double ci_99 = t_values_99[(trials <= 40?trials:40)-1]*std_dev/std::sqrt(trials);

  std::cout << "Mean: " << mean/1000.0/1000.0 << " (ms); "
      << "Stdev: " << std_dev/1000.0/1000.0  << " (ms); "
      << "99% CI: " << ci_99/1000.0/1000.0 << " (ms)" <<  std::endl;
}

template<class Test>
void run_test(const char* name, const size_t n, const size_t trials) {
  Test test (n);

  for (size_t i = 0; i < trials; i++) {
    test.test();
  }

  std::cout << name << "  ";
  compute_stats(test, trials);
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    printf("Arguments: N tests\n");
    printf("  N:   Number of vector entries\n");
    printf("  tests: number of times each kernels should be run\n");
        return 1;
  }

  Kokkos::initialize();

  const size_t n = atoi(argv[1]);
  const size_t trials = atoi(argv[2]);

  std::cout << "Copy kernel with only doubles" << std::endl;
  run_test<copy<Kokkos::LayoutLeft>>("copy 2dview left ", n, trials);
  run_test<copy<Kokkos::LayoutRight>>("copy 2dview right", n, trials);
  run_test<copy_struct<Kokkos::LayoutLeft>>("copy        left ", n, trials);
  run_test<copy_struct<Kokkos::LayoutRight>>("copy        right", n, trials);
  run_test<copy_vos<Kokkos::LayoutLeft>>("copy VoS    left ", n, trials);
  run_test<copy_vos<Kokkos::LayoutRight>>("copy VoS    right", n, trials);

  std::cout << "Copy kernel with mixed types" << std::endl;
  run_test<copy_mixed<Kokkos::LayoutLeft>>("copy        left ", n, trials);
  run_test<copy_mixed<Kokkos::LayoutRight>>("copy        right", n, trials);
  run_test<copy_mixed_vos<Kokkos::LayoutLeft>>("copy VoS    left ", n, trials);
  run_test<copy_mixed_vos<Kokkos::LayoutRight>>("copy VoS    right", n, trials);

  std::cout << "Euler particle simulation" << std::endl;
  run_test<euler_particles<Kokkos::LayoutLeft>>("euler       left ", n, trials);
  run_test<euler_particles<Kokkos::LayoutRight>>("euler       right", n, trials);
  run_test<euler_particles_vos<Kokkos::LayoutLeft>>("euler sov   left ", n, trials);
  run_test<euler_particles_vos<Kokkos::LayoutRight>>("euler sov   right", n, trials);

  std::cout << "Memory usage" << std::endl;
  run_test<capacity<SoA>>("capacity SoA ", n, trials);
  run_test<capacity<AoS>>("capacity AoS ", n, trials);

  Kokkos::finalize();
}

