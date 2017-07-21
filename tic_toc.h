#pragma once
#include <chrono>
#include <iostream>
#include <map>
#include <string>

// Aggregate timing results.
struct TimingCollector {
 public:
  static TimingCollector& GetInstance() {
    static const std::unique_ptr<TimingCollector> instance(
        new TimingCollector());
    return *instance.get();
  }

  void Update(const std::string& name, const unsigned long duration_us,
              const unsigned long reps) {
    Values& entry = recorded_values_[name];
    entry.total += duration_us;
    entry.count += reps;
  }

  ~TimingCollector() {
    // print all the values, w/ times per operation
    std::cout << "name, iterations, average time (us)\n";
    for (const std::pair<const std::string, Values>& entry : recorded_values_) {
      const double average_us =
          entry.second.total / static_cast<double>(entry.second.count);
      std::cout << entry.first << ", " << entry.second.count << ", "
                << average_us << "\n";
    }
  }

 private:
  struct Values {
    unsigned long total{0};
    unsigned long count{0};
  };

  std::map<std::string, Values> recorded_values_;
};

// Collect start and end time + number of repititions executed in that time.
struct TicToc {
  explicit TicToc(const std::string& name, const int reps = 1)
      : name_(name), reps_(reps), start_(std::chrono::steady_clock::now()) {}

  ~TicToc() {
    const auto now = std::chrono::steady_clock::now();
    const auto delta =
        std::chrono::duration_cast<std::chrono::microseconds>(now - start_);
    // submit to collector
    TimingCollector::GetInstance().Update(name_, delta.count(), reps_);
  }

 private:
  const std::string name_;
  const int reps_;
  const std::chrono::steady_clock::time_point start_;
};
