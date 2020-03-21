
// copy test

#ifndef COPY_HPP
#define COPY_HPP

#include <chrono>
#include <cstdint>
#include <Kokkos_Core.hpp>
#include <vector>

template<class Layout>
struct copy {
	const size_t n;

	Kokkos::View<double*[6], Layout> src;
	Kokkos::View<double*[6], Layout> dst;

	std::vector<uint64_t> times;

	copy(size_t n) : n(n), src("copy::src", n), dst("copy::dst", n) {
		setup();
	}

	void setup() {
		Kokkos::deep_copy(src, double(n));
		Kokkos::deep_copy(dst, double(0));
		Kokkos::fence();
	}

	void test() {
		// time copy kernel
		auto t1 = std::chrono::high_resolution_clock::now();
		Kokkos::parallel_for("copy::test", n, KOKKOS_LAMBDA(const size_t& i) {
			for (int j = 0; j < 6; j++) {
				dst(i, j) = src(i, j);
			}
		});
		Kokkos::fence();
		auto t2 = std::chrono::high_resolution_clock::now();

		// reset for next iteration
		Kokkos::deep_copy(dst, double(0));
		Kokkos::fence();

		times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
	}
};


#endif // COPY_HPP
