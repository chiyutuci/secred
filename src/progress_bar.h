// One-line progress display for the long phases of the global reduction.
//
// The rendering is a pure function of (done, total), so it can be checked without a
// terminal, and the update rate is throttled: the sample-point loop of a symbolic
// reconstruction runs tens of thousands of times and must not spend its time writing
// to stdout.  Without a terminal every update is a complete line instead of a carriage
// return, so a redirected log stays readable.
//
// Header-only so tests drive the real renderer.

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

#include "run_report.h" // format_duration

namespace secred {

// True when stdout is a terminal, i.e. when redrawing one line with '\r' gives a
// progress bar rather than a file full of partial lines.
inline bool stdout_is_tty() { return ::isatty(1) != 0; }

class ProgressBar {
 public:
  ProgressBar(std::string label, std::size_t total, bool interactive = true,
              unsigned width = 20)
      : label_(std::move(label)), total_(total), interactive_(interactive),
        width_(width) {}

  // The display for `done` completed items, without a carriage return or newline:
  // "<label> [########------------]  42 %  20480/48815".
  std::string render(std::size_t done) const {
    const unsigned pct = percent(done);
    const std::size_t filled =
        total_ ? std::min<std::size_t>(done * width_ / total_, width_) : width_;
    std::string bar(filled, '#');
    bar.append(width_ - filled, '-');
    return label_ + " [" + bar + "] " + pad3(pct) + " %  " +
           std::to_string(done) + "/" + std::to_string(total_);
  }

  // Text to write for this update, or "" when it is throttled away: interactive mode
  // redraws once per percent (carriage return, no newline), a redirected log gets a
  // whole line every 10 %.
  std::string update(std::size_t done) {
    const unsigned pct = percent(done);
    const unsigned step = interactive_ ? 1u : 10u;
    if (started_ && pct < last_pct_ + step)
      return std::string();
    started_ = true;
    last_pct_ = pct;
    return decorate(render(done));
  }

  // The final 100 % line, newline-terminated; a fresh line under a terminal.
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

  // Right-aligned to three digits so the line width does not jump at 10 % / 100 %.
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

// The live line for a phase whose total is not known while it runs: the FireFly
// reconstruction, where the number of prime fields a rational function needs is decided by
// the reconstruction itself.  FireFly publishes no counter of its own at SILENT verbosity,
// and the ones it keeps are private and restart inside every prime field ("Done: 0 / 19950"
// through the first half of the F234 run), so there is no total to divide by and the line
// reports what the black box has actually finished -- prime fields re-reduced, probes
// answered -- and the wall clock.  Completion is the driver's "reconstruction done in ..."
// line, which carries the same counters once they are final.
//
// Rendering stays a pure function of those counters so it can be checked without a
// terminal; the update rate is throttled the way ProgressBar's is, once a second on a
// terminal and once every ten seconds in a log, with the first line held back until the
// run has a second behind it.
class ReconStatus {
 public:
  ReconStatus(std::string label, bool interactive = true)
      : label_(std::move(label)), interactive_(interactive) {}

  // "<label>: 1min12s elapsed, 7 prime field(s), 280 probe(s) (3.9/s)", with the rate
  // left out before the first second and, once the probe count has stood still for
  // `kIdleSeconds`, the stretch it has stood still for: between two prime fields the box
  // is asked for nothing while the tables are rebuilt and FireFly interpolates, and an
  // unexplained frozen counter reads as a stalled run.
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

  // Text to write for this update, or "" when it is throttled away.  `elapsed` is
  // measured from the start of the phase by the caller's clock and must not go
  // backwards; how long the counters have stood still is read off them here, so every
  // call has to be made even when its line is throttled away.
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

  // Closes the redrawn line, a no-op when every line was a complete one.
  std::string finish() const {
    return started_ && interactive_ ? std::string("\n") : std::string();
  }

 private:
  // Short enough that a live reconstruction never reaches it between two probes, long
  // enough that a prime field's own work is not reported as a stall.
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

} // namespace secred
