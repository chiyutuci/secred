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
#include <ostream>
#include <string>
#include <utility>

#include <unistd.h>

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

} // namespace secred
