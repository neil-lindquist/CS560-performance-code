
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

template<class Layout>
struct copy_struct {
};

template <>
struct copy_struct<Kokkos::LayoutRight> {

  const size_t n;

  struct object {

    KOKKOS_INLINE_FUNCTION object()
      : object(0.0) {
    }

    KOKKOS_INLINE_FUNCTION object(double i)
      : field0(i), field1(i), field2(i), field3(i), field4(i), field5(i) {
    }

    double field0;
    double field1;
    double field2;
    double field3;
    double field4;
    double field5;
  };

  Kokkos::View<object*, Kokkos::LayoutRight> src;
  Kokkos::View<object*, Kokkos::LayoutRight> dst;

  std::vector<uint64_t> times;

  copy_struct(size_t n) : n(n), src("copy_struct::src", n), dst("copy_struct::dst", n) {
    setup();
  }

  void setup() {
    Kokkos::deep_copy(src, object(n));
    Kokkos::deep_copy(dst, object(0));
    Kokkos::fence();
  }

  void test() {
    // time copy kernel
    auto t1 = std::chrono::high_resolution_clock::now();
    Kokkos::parallel_for("copy_struct::test", n, KOKKOS_LAMBDA(const size_t& i) {
      dst(i) = src(i);
//      dst(i).field0 = src(i).field0;
//      dst(i).field1 = src(i).field1;
//      dst(i).field2 = src(i).field2;
//      dst(i).field3 = src(i).field3;
//      dst(i).field4 = src(i).field4;
//      dst(i).field5 = src(i).field5;
    });
    Kokkos::fence();
    auto t2 = std::chrono::high_resolution_clock::now();

    // reset for next iteration
    Kokkos::deep_copy(dst, object(0));
    Kokkos::fence();

    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
  }
};

template <>
struct copy_struct<Kokkos::LayoutLeft> {

  const size_t n;

  struct sov {
    
    Kokkos::View<double*> field0;
    Kokkos::View<double*> field1;
    Kokkos::View<double*> field2;
    Kokkos::View<double*> field3;
    Kokkos::View<double*> field4;
    Kokkos::View<double*> field5;

    sov(size_t n)
      : field0("sov::field0", n),
        field1("sov::field0", n),
        field2("sov::field0", n),
        field3("sov::field0", n),
        field4("sov::field0", n),
        field5("sov::field0", n) {
    }
  };

  sov src;
  sov dst;

  std::vector<uint64_t> times;

  copy_struct(size_t n) : n(n), src(n), dst(n) {
    setup();
  }

  void setup() {
    Kokkos::deep_copy(src.field0, double(n));
    Kokkos::deep_copy(src.field1, double(n));
    Kokkos::deep_copy(src.field2, double(n));
    Kokkos::deep_copy(src.field3, double(n));
    Kokkos::deep_copy(src.field4, double(n));
    Kokkos::deep_copy(src.field5, double(n));
    Kokkos::deep_copy(dst.field0, double(0));
    Kokkos::deep_copy(dst.field1, double(0));
    Kokkos::deep_copy(dst.field2, double(0));
    Kokkos::deep_copy(dst.field3, double(0));
    Kokkos::deep_copy(dst.field4, double(0));
    Kokkos::deep_copy(dst.field5, double(0));
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
    Kokkos::fence();

    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
  }
};

// Copy test implementing using the ViewOfStructs type
template<class Layout>
struct copy_vos {
  const size_t n;

  using struct_type = Kokkos::Struct<double, double, double, double, double, double>;
  static constexpr auto field_0 = Kokkos::Field<0>();
  static constexpr auto field_1 = Kokkos::Field<1>();
  static constexpr auto field_2 = Kokkos::Field<2>();
  static constexpr auto field_3 = Kokkos::Field<3>();
  static constexpr auto field_4 = Kokkos::Field<4>();
  static constexpr auto field_5 = Kokkos::Field<5>();

  Kokkos::ViewOfStructs<struct_type*, Layout> src;
  Kokkos::ViewOfStructs<struct_type*, Layout> dst;

  std::vector<uint64_t> times;

  copy_vos(size_t n) : n(n), src("copy_vos::src", n), dst("copy_vos::dst", n) {
    setup();
  }

  void setup() {
    constexpr auto field_0 = Kokkos::Field<0>();
    constexpr auto field_1 = Kokkos::Field<1>();
    constexpr auto field_2 = Kokkos::Field<2>();
    constexpr auto field_3 = Kokkos::Field<3>();
    constexpr auto field_4 = Kokkos::Field<4>();
    constexpr auto field_5 = Kokkos::Field<5>();

    Kokkos::parallel_for("copy_vos::setup", n, KOKKOS_LAMBDA(const size_t& i) {
      src(i, field_0) = n;
      src(i, field_1) = n;
      src(i, field_2) = n;
      src(i, field_3) = n;
      src(i, field_4) = n;
      src(i, field_5) = n;
      dst(i, field_0) = 0;
      dst(i, field_1) = 0;
      dst(i, field_2) = 0;
      dst(i, field_3) = 0;
      dst(i, field_4) = 0;
      dst(i, field_5) = 0;
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

    // time copy kernel
    auto t1 = std::chrono::high_resolution_clock::now();
    Kokkos::parallel_for("copy_vos::test", n, KOKKOS_LAMBDA(const size_t& i) {
      dst(i, field_0) = src(i, field_0);
      dst(i, field_1) = src(i, field_1);
      dst(i, field_2) = src(i, field_2);
      dst(i, field_3) = src(i, field_3);
      dst(i, field_4) = src(i, field_4);
      dst(i, field_5) = src(i, field_5);
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
    });
    Kokkos::fence();

    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
  }

};


#endif // COPY_HPP
