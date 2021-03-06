
// copy test with mixed datatypes

#ifndef COPY_MIXED_HPP
#define COPY_MIXED_HPP

#include <chrono>
#include <cstdint>
#include <Kokkos_Core.hpp>
#include <vector>

template<class Layout>
struct copy_mixed {
};

template <>
struct copy_mixed<Kokkos::LayoutRight> {

  const size_t n;

  struct object {

    KOKKOS_INLINE_FUNCTION object()
      : object(0, 0, 0, 0, 0, 0, 0, 0) {
    }

    KOKKOS_INLINE_FUNCTION
    object(double f0, float f1, int32_t f2, uint32_t f3, uint16_t f4, int16_t f5, int64_t f6, uint64_t f7)
      : field0(f0), field1(f1), field2(f2), field3(f3), field4(f4), field5(f5), field6(f6), field7(f7) {
    }

    double   field0;
    float    field1;
    int32_t  field2;
    uint32_t field3;
    uint16_t field4;
    int16_t  field5;
    int64_t  field6;
    uint64_t field7;
  };

  Kokkos::View<object*, Kokkos::LayoutRight> src;
  Kokkos::View<object*, Kokkos::LayoutRight> dst;

  std::vector<uint64_t> times;

	copy_mixed(size_t n) : n(n), src("copy_struct::src", n), dst("copy_struct::dst", n) {
		setup();
	}

	void setup() {
		Kokkos::deep_copy(src, object(n, n, 1000, 2000, 100, 200, 3000, 4000));
		Kokkos::deep_copy(dst, object(0, 0, 0, 0, 0, 0, 0, 0));
		Kokkos::fence();
	}

	void test() {
		// time copy kernel
		auto t1 = std::chrono::high_resolution_clock::now();
		Kokkos::parallel_for("copy_struct::test", n, KOKKOS_LAMBDA(const size_t& i) {
		  dst(i).field0 = src(i).field0;
		  dst(i).field1 = src(i).field1;
		  dst(i).field2 = src(i).field2;
		  dst(i).field3 = src(i).field3;
		  dst(i).field4 = src(i).field4;
		  dst(i).field5 = src(i).field5;
                  dst(i).field6 = src(i).field6;
                  dst(i).field7 = src(i).field7;
		});
		Kokkos::fence();
		auto t2 = std::chrono::high_resolution_clock::now();

		// reset for next iteration
		Kokkos::deep_copy(dst, object(0, 0, 0, 0, 0, 0, 0, 0));
		Kokkos::fence();

		times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
	}
};

template <>
struct copy_mixed<Kokkos::LayoutLeft> {

  const size_t n;

  struct sov {
    
    Kokkos::View<double*> field0;
    Kokkos::View<float*> field1;
    Kokkos::View<int32_t*> field2;
    Kokkos::View<uint32_t*> field3;
    Kokkos::View<uint16_t*> field4;
    Kokkos::View<int16_t*> field5;
    Kokkos::View<int64_t*> field6;
    Kokkos::View<uint64_t*> field7;

    sov(size_t n)
      : field0("sov::field0", n),
        field1("sov::field1", n),
        field2("sov::field2", n),
        field3("sov::field3", n),
        field4("sov::field4", n),
        field5("sov::field5", n),
        field6("sov::field6", n),
        field7("sov::field7", n) {
    }
  };

  sov src;
  sov dst;

  std::vector<uint64_t> times;

	copy_mixed(size_t n) : n(n), src(n), dst(n) {
		setup();
	}

	void setup() {
		Kokkos::deep_copy(src.field0, double(n));
		Kokkos::deep_copy(src.field1, double(n));
		Kokkos::deep_copy(src.field2, double(1000));
		Kokkos::deep_copy(src.field3, double(2000));
		Kokkos::deep_copy(src.field4, double(100));
		Kokkos::deep_copy(src.field5, double(200));
		Kokkos::deep_copy(src.field6, double(3000));
		Kokkos::deep_copy(src.field7, double(4000));

		Kokkos::deep_copy(dst.field0, double(0));
		Kokkos::deep_copy(dst.field1, double(0));
		Kokkos::deep_copy(dst.field2, double(0));
		Kokkos::deep_copy(dst.field3, double(0));
		Kokkos::deep_copy(dst.field4, double(0));
		Kokkos::deep_copy(dst.field5, double(0));
		Kokkos::deep_copy(dst.field6, double(0));
		Kokkos::deep_copy(dst.field7, double(0));
		Kokkos::fence();
	}

	void test() {
		// time copy kernel
		auto t1 = std::chrono::high_resolution_clock::now();
		Kokkos::parallel_for("copy_struct::test", n, KOKKOS_LAMBDA(const size_t& i) {
		  dst.field0(i) = src.field0(i);
		  dst.field1(i) = src.field1(i);
		  dst.field2(i) = src.field2(i);
		  dst.field3(i) = src.field3(i);
		  dst.field4(i) = src.field4(i);
		  dst.field5(i) = src.field5(i);
		  dst.field6(i) = src.field6(i);
		  dst.field7(i) = src.field7(i);
		});
		Kokkos::fence();
		auto t2 = std::chrono::high_resolution_clock::now();

		// reset for next iteration
		Kokkos::deep_copy(dst.field0, double(0));
		Kokkos::deep_copy(dst.field1, double(0));
		Kokkos::deep_copy(dst.field2, double(0));
		Kokkos::deep_copy(dst.field3, double(0));
		Kokkos::deep_copy(dst.field4, double(0));
		Kokkos::deep_copy(dst.field5, double(0));
		Kokkos::deep_copy(dst.field6, double(0));
		Kokkos::deep_copy(dst.field7, double(0));
		Kokkos::fence();

		times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
	}
};

// Copy test implementing using the ViewOfStructs type
template<class Layout>
struct copy_mixed_vos {
  const size_t n;

  using struct_type = Kokkos::Struct<double, float, int32_t, uint32_t, int16_t, uint16_t, int64_t, uint64_t>;
  Kokkos::ViewOfStructs<struct_type*, Layout> src;
  Kokkos::ViewOfStructs<struct_type*, Layout> dst;

  std::vector<uint64_t> times;

  copy_mixed_vos(size_t n) : n(n), src("copy_vos::src", n), dst("copy_vos::dst", n) {
    setup();
  }

  void setup() {
    constexpr auto field_0 = Kokkos::Field<0>();
    constexpr auto field_1 = Kokkos::Field<1>();
    constexpr auto field_2 = Kokkos::Field<2>();
    constexpr auto field_3 = Kokkos::Field<3>();
    constexpr auto field_4 = Kokkos::Field<4>();
    constexpr auto field_5 = Kokkos::Field<5>();
    constexpr auto field_6 = Kokkos::Field<6>();
    constexpr auto field_7 = Kokkos::Field<7>();

    Kokkos::parallel_for("copy_vos::setup", n, KOKKOS_LAMBDA(const size_t& i) {
      src(i, field_0) = n;
      src(i, field_1) = n;
      src(i, field_2) = 1000;
      src(i, field_3) = 2000;
      src(i, field_4) = 100;
      src(i, field_5) = 200;
      src(i, field_6) = 3000;
      src(i, field_7) = 4000;

      dst(i, field_0) = 0;
      dst(i, field_1) = 0;
      dst(i, field_2) = 0;
      dst(i, field_3) = 0;
      dst(i, field_4) = 0;
      dst(i, field_5) = 0;
      dst(i, field_6) = 0;
      dst(i, field_7) = 0;
    });
    Kokkos::fence();
  }

  void test() {
    constexpr auto field_0 = Kokkos::Field<0>();
    constexpr auto field_1 = Kokkos::Field<1>();
    constexpr auto field_2 = Kokkos::Field<2>();
    constexpr auto field_3 = Kokkos::Field<3>();
    constexpr auto field_4 = Kokkos::Field<4>();
    constexpr auto field_5 = Kokkos::Field<5>();
    constexpr auto field_6 = Kokkos::Field<6>();
    constexpr auto field_7 = Kokkos::Field<7>();

		// time copy kernel
		auto t1 = std::chrono::high_resolution_clock::now();
		Kokkos::parallel_for("copy_vos::test", n, KOKKOS_LAMBDA(const size_t& i) {
      dst(i, field_0) = src(i, field_0);
      dst(i, field_1) = src(i, field_1);
      dst(i, field_2) = src(i, field_2);
      dst(i, field_3) = src(i, field_3);
      dst(i, field_4) = src(i, field_4);
      dst(i, field_5) = src(i, field_5);
      dst(i, field_6) = src(i, field_6);
      dst(i, field_7) = src(i, field_7);
		});
		Kokkos::fence();
		auto t2 = std::chrono::high_resolution_clock::now();

		// reset for next iteration
    Kokkos::parallel_for("copy_vos::test_reset", n, KOKKOS_LAMBDA(const size_t& i) {
      dst(i, field_0) = 0;
      dst(i, field_1) = 0;
      dst(i, field_2) = 0;
      dst(i, field_3) = 0;
      dst(i, field_4) = 0;
      dst(i, field_5) = 0;
      dst(i, field_6) = 0;
      dst(i, field_7) = 0;
    });
		Kokkos::fence();

		times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
	}

};


#endif // COPY_HPP
