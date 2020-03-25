// Capacity test
// Compare memory allocation size differences between SoA and AoS

#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/sysinfo.h>
#include <iostream>
#include <ios>
#include <fstream>
#include <string>
#include <unistd.h>

int parseLine(char* line) {
  int i = strlen(line);
  const char* p = line;
  while(*p <'0' || *p > '9') p++;
  line[i-3] = '\0';
  i = atoi(p);
  return i;
}

int getPhysMemKB() {
  FILE* file = fopen("/proc/self/status", "r");
  int result = -1;
  char line[128];

  while(fgets(line, 128, file) != NULL) {
    if(strncmp(line, "VmRSS:", 6) == 0) {
      result = parseLine(line);
      break;
    }
  }
  fclose(file);
  return result;
}

void process_mem_usage(double& vm_usage, double& resident_set) {
  vm_usage = 0.0;
  resident_set = 0.0;

  std::ifstream stat_stream("/proc/self/stat", std::ios_base::in);

  std::string pid, comm, state, ppid, pgrp, session, tty_nr;
  std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  std::string utime, stime, cutime, cstime, priority, nice;
  std::string O, itrealvalue, starttime;

  unsigned long vsize;
  long rss;

  stat_stream >> pid >> comm >> state >> ppid >> session >> tty_nr
              >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
              >> utime >> stime >> cutime >> cstime >> priority >> nice
              >> O >> itrealvalue >> starttime >> vsize >> rss;
  stat_stream.close();
  
  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024;
  vm_usage = vsize/1024.0;
  resident_set = rss*page_size_kb; 
}

typedef struct {
  char a;
  float b;
  char c;
  double d;
  char e;
  int f;
} data;

class SoA {
public:
  Kokkos::View<char*> a;
  Kokkos::View<float*> b;
  Kokkos::View<char*> c;
  Kokkos::View<double*> d;
  Kokkos::View<char*> e;
  Kokkos::View<int*> f;

  SoA(size_t n) :
    a("SoA: a", n),
    b("SoA: b", n),
    c("SoA: c", n),
    d("SoA: d", n),
    e("SoA: e", n),
    f("SoA: f", n)
  {}

  void fill() {
    Kokkos::parallel_for("Fill with 0s", a.extent(0), KOKKOS_LAMBDA(const size_t i) {
      a(i) = 0;
      b(i) = 0;
      c(i) = 0;
      d(i) = 0;
      e(i) = 0;
      f(i) = 0;
    });
  }
};

class AoS {
public:
  Kokkos::View<data*> _data;
  AoS(size_t n) : _data("AoS data", n) {}
  void fill() {
    Kokkos::parallel_for("Fill with 0s", _data.extent(0), KOKKOS_LAMBDA(const size_t i) {
      auto entry = _data(i);
      entry.a = 0;
      entry.b = 0;
      entry.c = 0;
      entry.d = 0;
      entry.e = 0;
      entry.f = 0;
    });
  }
};

template <typename Layout>
class capacity {
public:
	const size_t _n;
  Layout _data;
	std::vector<uint64_t> times;

  capacity(size_t n): _n(n), _data(n) {
    setup();
  }

  void setup() {
    _data.fill();
  }

  void test() {
//    struct rusage usage;
//    getrusage(RUSAGE_SELF, &usage);
//    int maxrss = usage.ru_maxrss;
//
//    double vm_usage, resident_set;
//    process_mem_usage(vm_usage, resident_set);
//    std::cout << typeid(Layout).name() << ": VM Usage: " << vm_usage << "KB\tRSS: " << resident_set << "KB\n";
    int maxrss = getPhysMemKB();
    std::cout << typeid(Layout).name() << ": RSS Usage: " << maxrss << "KB\n";
//    times.push_back(maxrss);
  }
};
