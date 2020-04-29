
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
};

template<>
struct euler_particles<Kokkos::LayoutRight> {

  const size_t n;

  struct particle_t {
    double x_accel;
    double y_accel;
    double z_accel;
    double x_vel;
    double y_vel;
    double z_vel;
    double x;
    double y;
    double z;
    uint32_t lifetime;
    uint8_t x_resistance;
    uint8_t y_resistance;
    uint8_t z_resistance;
  };

  Kokkos::View<particle_t*> particles;

  std::vector<uint64_t> times;

  euler_particles(size_t n) : n(n), particles("euler_particles_sov::particles", n) {
    setup();
  }

  void setup() {
    Kokkos::parallel_for("euler_particles_sov::setup", n,
      KOKKOS_LAMBDA(const size_t& i) {
      particles(i).x_accel = 5*i;
      particles(i).y_accel = 2.4*i - 10000;
      particles(i).z_accel = 0.87*(i*i);

      particles(i).x_vel = 1.0/i;
      particles(i).y_vel = -2.0/i;
      particles(i).z_vel = 1.0/(n-i);

      particles(i).x = -1.5/(i*i);
      particles(i).y = 2.0/(n-i*i);
      particles(i).z = 1.0/(i*i);

      // Create a mix of particle lifetimes such that
      // * some dead particles
      // * The number of tests run doesn't affect the amount of work done
      particles(i).lifetime = ((i*31)%1024) * 100;

      particles(i).x_resistance = (i*71) < 10;
      particles(i).y_resistance = (i*91) < 10;
      particles(i).z_resistance = (i*81) < 10;
    });
    Kokkos::fence();
  }

  void test() {
    // time copy kernel
    auto t1 = std::chrono::high_resolution_clock::now();
    Kokkos::parallel_for("euler_particles_sov::test", n, KOKKOS_LAMBDA(const size_t& i) {
      const double dt = 0.001;
      const double drag = 0.01;

      if (particles(i).lifetime > 0) {

        double x_acceleration, y_acceleration, z_acceleration;

        if (particles(i).x_resistance) {
          x_acceleration = particles(i).x_accel - drag;
        } else {
          x_acceleration = particles(i).x_accel;
        }

        if (particles(i).y_resistance) {
          y_acceleration = particles(i).y_accel - drag;
        } else {
          y_acceleration = particles(i).y_accel;
        }

        if (particles(i).z_resistance) {
          z_acceleration = particles(i).z_accel - drag;
        } else {
          z_acceleration = particles(i).z_accel;
        }

        particles(i).x_vel += dt*x_acceleration;
        particles(i).y_vel += dt*y_acceleration;
        particles(i).z_vel += dt*z_acceleration;

        particles(i).x += dt*particles(i).x_vel;
        particles(i).y += dt*particles(i).y_vel;
        particles(i).z += dt*particles(i).z_vel;

        particles(i).lifetime -= 1;
      }
    });
    Kokkos::fence();
    auto t2 = std::chrono::high_resolution_clock::now();

    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
  }
};


template<>
struct euler_particles<Kokkos::LayoutLeft> {

  const size_t n;

  Kokkos::View<double*> x_accel;
  Kokkos::View<double*> y_accel;
  Kokkos::View<double*> z_accel;
  Kokkos::View<double*> x_vel;
  Kokkos::View<double*> y_vel;
  Kokkos::View<double*> z_vel;
  Kokkos::View<double*> x;
  Kokkos::View<double*> y;
  Kokkos::View<double*> z;
  Kokkos::View<uint32_t*> lifetime;
  Kokkos::View<uint8_t*> x_resistance;
  Kokkos::View<uint8_t*> y_resistance;
  Kokkos::View<uint8_t*> z_resistance;

  std::vector<uint64_t> times;

  euler_particles(size_t n)
    : n(n),
      x_accel("euler_particles_sov::x_accel", n),
      y_accel("euler_particles_sov::y_accel", n),
      z_accel("euler_particles_sov::z_accel", n),
      x_vel("euler_particles_sov::x_accel", n),
      y_vel("euler_particles_sov::y_accel", n),
      z_vel("euler_particles_sov::z_accel", n),
      x("euler_particles_sov::x_accel", n),
      y("euler_particles_sov::y_accel", n),
      z("euler_particles_sov::z_accel", n),
      lifetime("euler_particles_sov::lifetime", n),
      x_resistance("euler_particles_sov::x_resistance", n),
      y_resistance("euler_particles_sov::y_resistance", n),
      z_resistance("euler_particles_sov::z_resistance", n) {
    setup();
  }

  void setup() {
    Kokkos::parallel_for("euler_sov::setup", n,
      KOKKOS_LAMBDA(const size_t& i) {
      x_accel(i) = 5*i;
      y_accel(i) = 2.4*i - 10000;
      z_accel(i) = 0.87*(i*i);

      x_vel(i) = 1.0/i;
      y_vel(i) = -2.0/i;
      z_vel(i) = 1.0/(n-i);

      x(i) = -1.5/(i*i);
      y(i) = 2.0/(n-i*i);
      z(i) = 1.0/(i*i);

      // Create a mix of particle lifetimes such that
      // * some dead particles
      // * The number of tests run doesn't affect the amount of work done
      lifetime(i) = ((i*31)%1024) * 100;

      x_resistance(i) = (i*71) < 10;
      y_resistance(i) = (i*91) < 10;
      z_resistance(i) = (i*81) < 10;
    });
    Kokkos::fence();
  }

  void test() {
    // time copy kernel
    auto t1 = std::chrono::high_resolution_clock::now();
    Kokkos::parallel_for("euler_sov::test", n, KOKKOS_LAMBDA(const size_t& i) {
      const double dt = 0.001;
      const double drag = 0.01;

      if (lifetime(i) > 0) {

        double x_acceleration, y_acceleration, z_acceleration;

        if (x_resistance(i)) {
          x_acceleration = x_accel(i) - drag;
        } else {
          x_acceleration = x_accel(i);
        }

        if (y_resistance(i)) {
          y_acceleration = y_accel(i) - drag;
        } else {
          y_acceleration = y_accel(i);
        }

        if (z_resistance(i)) {
          z_acceleration = z_accel(i) - drag;
        } else {
          z_acceleration = z_accel(i);
        }

        x_vel(i) += dt*x_acceleration;
        y_vel(i) += dt*y_acceleration;
        z_vel(i) += dt*z_acceleration;

        x(i) += dt*x_vel(i);
        y(i) += dt*y_vel(i);
        z(i) += dt*z_vel(i);

        lifetime(i) -= 1;
      }
    });
    Kokkos::fence();
    auto t2 = std::chrono::high_resolution_clock::now();

    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
  }
};

template<class Layout>
struct euler_particles_vos {

  const size_t n;

  typedef Kokkos::Struct<double, double, double,
                         double, double, double,
                         double, double, double,
                         uint32_t, uint8_t, uint8_t, uint8_t>
          particle_t;

  typedef Kokkos::Field<0>  x_accel;
  typedef Kokkos::Field<1>  y_accel;
  typedef Kokkos::Field<2>  z_accel;
  typedef Kokkos::Field<3>  x_vel;
  typedef Kokkos::Field<4>  y_vel;
  typedef Kokkos::Field<5>  z_vel;
  typedef Kokkos::Field<6>  x;
  typedef Kokkos::Field<7>  y;
  typedef Kokkos::Field<8>  z;

  typedef Kokkos::Field<9>  lifetime;
  typedef Kokkos::Field<10> x_resistance;
  typedef Kokkos::Field<11> y_resistance;
  typedef Kokkos::Field<12> z_resistance;


  Kokkos::ViewOfStructs<particle_t*, Layout> particles;

  std::vector<uint64_t> times;

  euler_particles_vos(size_t n) : n(n), particles("euler_particles_sov::particles", n) {
    setup();
  }

  void setup() {
    Kokkos::parallel_for("euler_particles_sov::setup", n,
      KOKKOS_LAMBDA(const size_t& i) {
      particles(i, x_accel()) = 5*i;
      particles(i, y_accel()) = 2.4*i - 10000;
      particles(i, z_accel()) = 0.87*(i*i);

      particles(i, x_vel()) = 1.0/i;
      particles(i, y_vel()) = -2.0/i;
      particles(i, z_vel()) = 1.0/(n-i);

      particles(i, x()) = -1.5/(i*i);
      particles(i, y()) = 2.0/(n-i*i);
      particles(i, z()) = 1.0/(i*i);

      // Create a mix of particle lifetimes such that
      // * some dead particles
      // * The number of tests run doesn't affect the amount of work done
      particles(i, lifetime()) = ((i*31)%1024) * 100;

      particles(i, x_resistance()) = (i*71) < 10;
      particles(i, y_resistance()) = (i*91) < 10;
      particles(i, z_resistance()) = (i*81) < 10;
    });
    Kokkos::fence();
  }

  void test() {
    // time copy kernel
    auto t1 = std::chrono::high_resolution_clock::now();
    Kokkos::parallel_for("euler_particles_sov::test", n, KOKKOS_LAMBDA(const size_t& i) {
      const double dt = 0.001;
      const double drag = 0.01;

      if (particles(i, lifetime()) > 0) {

        double x_acceleration, y_acceleration, z_acceleration;

        if (particles(i, x_resistance())) {
          x_acceleration = particles(i, x_accel()) - drag;
        } else {
          x_acceleration = particles(i, x_accel());
        }

        if (particles(i, y_resistance())) {
          y_acceleration = particles(i, y_accel()) - drag;
        } else {
          y_acceleration = particles(i, y_accel());
        }

        if (particles(i, z_resistance())) {
          z_acceleration = particles(i, z_accel()) - drag;
        } else {
          z_acceleration = particles(i, z_accel());
        }

        particles(i, x_vel()) += dt*x_acceleration;
        particles(i, y_vel()) += dt*y_acceleration;
        particles(i, z_vel()) += dt*z_acceleration;

        particles(i, x()) += dt*particles(i, x_vel());
        particles(i, y()) += dt*particles(i, y_vel());
        particles(i, z()) += dt*particles(i, z_vel());

        particles(i, lifetime()) -= 1;
      }
    });
    Kokkos::fence();
    auto t2 = std::chrono::high_resolution_clock::now();

    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
  }
};


#endif // EULER_PARTICLE_HPP
