
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

#include <unistd.h>

#include "run_report.h"

namespace secred {

inline bool stdout_is_tty() { return ::isatty(1) != 0; }

// Single-line progress bar; non-interactive mode prints every 10 %.
class ProgressBar {
 public:
  ProgressBar(std::string label, std::size_t total, bool interactive = true,
              unsigned width = 20)
      : label_(std::move(label)), total_(total), interactive_(interactive),
        width_(width) {}

  std::string render(std::size_t done) const {
    const unsigned pct = percent(done);
    const std::size_t filled =
        total_ ? std::min<std::size_t>(done * width_ / total_, width_) : width_;
    std::string bar(filled, '#');
    bar.append(width_ - filled, '-');
    return label_ + " [" + bar + "] " + pad3(pct) + " %  " +
           std::to_string(done) + "/" + std::to_string(total_);
  }

  std::string update(std::size_t done) {
    const unsigned pct = percent(done);
    const unsigned step = interactive_ ? 1u : 10u;
    if (started_ && pct < last_pct_ + step)
      return std::string();
    started_ = true;
    last_pct_ = pct;
    return decorate(render(done));
  }

  std::string finish() {
    started_ = true;
    last_pct_ = 100;
    return (interactive_ ? std::string("\r") : std::string()) + render(total_) +
           "\n";
  }

 private:
  static unsigned percent(std::size_t done, std::size_t total) {
    if (total == 0)
      return 100;
    return static_cast<unsigned>(std::min<std::size_t>(done * 100 / total, 100));
  }
  unsigned percent(std::size_t done) const { return percent(done, total_); }

  static std::string pad3(unsigned pct) {
    const std::string s = std::to_string(pct);
    return std::string(s.size() < 3 ? 3 - s.size() : 0, ' ') + s;
  }

  std::string decorate(const std::string &line) const {
    return interactive_ ? "\r" + line : line + "\n";
  }

  std::string label_;
  std::size_t total_;
  bool interactive_;
  unsigned width_;
  unsigned last_pct_ = 0;
  bool started_ = false;
};

// Reconstruction progress: elapsed time, prime fields and probe rate.
class ReconStatus {
 public:
  ReconStatus(std::string label, bool interactive = true)
      : label_(std::move(label)), interactive_(interactive) {}

  std::string render(double elapsed, std::uint64_t primes, std::uint64_t probes,
                     double idle) const {
    std::ostringstream oss;
    oss << label_ << ": " << format_duration(elapsed) << " elapsed, " << primes
        << " prime field(s), " << probes << " probe(s)";
    if (elapsed >= 1.0)
      oss << " (" << std::fixed << std::setprecision(1)
          << static_cast<double>(probes) / elapsed << "/s)";
    if (idle >= kIdleSeconds)
      oss << ", no probe for " << format_duration(idle);
    return oss.str();
  }

  std::string update(double elapsed, std::uint64_t primes, std::uint64_t probes) {
    if (probes != last_probes_) {
      last_probes_ = probes;
      last_probe_at_ = elapsed;
    }
    const double step = interactive_ ? 1.0 : 10.0;
    if (elapsed < (started_ ? last_print_ + step : 1.0))
      return std::string();
    started_ = true;
    last_print_ = elapsed;
    return decorate(render(elapsed, primes, probes, elapsed - last_probe_at_));
  }

  std::string finish() const {
    return started_ && interactive_ ? std::string("\n") : std::string();
  }

 private:

  static constexpr double kIdleSeconds = 2.0;

  std::string decorate(const std::string &line) const {
    return interactive_ ? "\r" + line : line + "\n";
  }

  std::string label_;
  bool interactive_;
  double last_print_ = 0.0;
  double last_probe_at_ = 0.0;
  std::uint64_t last_probes_ = 0;
  bool started_ = false;
};

}
