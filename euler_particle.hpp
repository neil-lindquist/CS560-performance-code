
// A simple, 3d particle system based on Euler integration
// Each particle has an position, velocity, and acceleration
// acceleration is constant, the other properties are updated
// note that each particle is unaffected in any way by other particles

#ifndef EULER_PARTICLE_HPP
#define EULER_PARTICLE_HPP

#include <chrono>
#include <cstdint>
#include <Kokkos_Core.hpp>
#include <vector>

template<class Layout>
struct euler_particles {
	// particles(i, 0) = x of ith particle
	// particles(i, 1) = y of ith particle
	// particles(i, 2) = z of ith particle
	// particles(i, 3) = x velocity of ith particle
	// particles(i, 4) = y velocity of ith particle
	// particles(i, 5) = z velocity of ith particle
	// particles(i, 6) = x acceleration of ith particle
	// particles(i, 7) = y acceleration of ith particle
	// particles(i, 8) = z acceleration of ith particle

	const size_t n;

	Kokkos::View<double*[9], Layout> particles;

	std::vector<uint64_t> times;

	euler_particles(size_t n) : n(n), particles("euler_particles::particles", n) {
		setup();
	}

	void setup() {
		Kokkos::parallel_for(n, KOKKOS_LAMBDA(const size_t& i) {
			particles(i, 0) = 5*i;
			particles(i, 1) = 2.4*i - 10000;
			particles(i, 2) = 0.87*(i*i);

			particles(i, 3) = 1.0/i;
			particles(i, 4) = -2.0/i;
			particles(i, 5) = 1.0/(n-i);

			particles(i, 6) = -1.5/(i*i);
			particles(i, 7) = 2.0/(n-i*i);
			particles(i, 8) = 1.0/(i*i);
		});
	}


	void test() {
		// time copy kernel
		auto t1 = std::chrono::high_resolution_clock::now();
		Kokkos::parallel_for(n, KOKKOS_LAMBDA(const size_t& i) {
			const double dt = 0.001;

			particles(i, 3) += dt*particles(i, 6);
			particles(i, 4) += dt*particles(i, 7);
			particles(i, 5) += dt*particles(i, 8);

			particles(i, 0) += dt*particles(i, 3);
			particles(i, 1) += dt*particles(i, 4);
			particles(i, 2) += dt*particles(i, 5);
		});
		Kokkos::fence();
		auto t2 = std::chrono::high_resolution_clock::now();

		times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
	}
};


#endif // EULER_PARTICLE_HPP
