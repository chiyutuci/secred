
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace secred {

inline constexpr const char *kFlowTag = "[fflow]";

inline constexpr const char *kFireFlyTag = "[firefly]";

// Run timers and the console report blocks.
void mark_run_start();
double run_elapsed_seconds();
double seconds_since(std::chrono::steady_clock::time_point t0);

namespace detail {
inline std::chrono::steady_clock::time_point &run_start_point() {
  static std::chrono::steady_clock::time_point t{};
  return t;
}
inline bool &run_started() {
  static bool started = false;
  return started;
}
}

inline void mark_run_start() {
  detail::run_start_point() = std::chrono::steady_clock::now();
  detail::run_started() = true;
}

inline double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

inline double run_elapsed_seconds() {
  if (!detail::run_started())
    return 0.0;
  return seconds_since(detail::run_start_point());
}

inline std::string format_duration(double seconds) {
  using ull = unsigned long long;
  ull h = static_cast<ull>(seconds) / 3600;
  ull m = (static_cast<ull>(seconds) % 3600) / 60;
  double s = seconds - static_cast<double>(h) * 3600.0 -
             static_cast<double>(m) * 60.0;
  std::ostringstream oss;
  if (h)
    oss << h << "h" << m << "min" << static_cast<ull>(s) << "s";
  else if (m)
    oss << m << "min" << static_cast<ull>(s) << "s";
  else
    oss << std::fixed << std::setprecision(1) << s << "s";
  return oss.str();
}

inline std::string format_memory(std::uint64_t bytes) {
  constexpr std::uint64_t kGiB = 1024ull * 1024 * 1024;
  constexpr std::uint64_t kMiB = 1024ull * 1024;
  std::ostringstream oss;
  if (bytes >= kGiB)
    oss << std::fixed << std::setprecision(2)
        << static_cast<double>(bytes) / static_cast<double>(kGiB) << " GB";
  else if (bytes >= kMiB)
    oss << std::fixed << std::setprecision(1)
        << static_cast<double>(bytes) / static_cast<double>(kMiB) << " MB";
  else
    oss << bytes << " B";
  return oss.str();
}

inline constexpr std::size_t kReportWidth = 57;

inline std::string block_header(const std::string &title) {
  const std::size_t pad = kReportWidth - title.size() - 2;
  const std::size_t left = pad / 2;
  return std::string(left, '-') + " " + title + " " +
         std::string(pad - left, '-') + "\n";
}

inline std::string block_rule() { return std::string(kReportWidth, '-') + "\n"; }

inline std::size_t block_label_width(const std::vector<std::string> &labels) {
  std::size_t width = 0;
  for (const auto &label : labels)
    width = std::max(width, label.size() + 1);
  return width + 1;
}

inline std::string block_row(const std::string &label, const std::string &value,
                             std::size_t label_width) {
  std::string field = label + ":";
  field.append(label_width > field.size() ? label_width - field.size() : 0, ' ');
  return "  " + field + value + "\n";
}

inline std::string block_rows(const std::vector<std::string> &rows) {
  std::string out;
  for (const auto &row : rows)
    out += "\n" + row;
  return out;
}

inline std::string format_percent(double ratio) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1) << ratio;
  return oss.str();
}

// Counters printed in the reduction statistics block.
struct ReductionStats {
  unsigned narrowing_success = 0;
  unsigned narrowing_fail = 0;
  std::uint64_t raw_equations = 0;
  std::uint64_t global_equations = 0;

  std::uint64_t final_equations = 0;
  unsigned masters = 0;
};

inline std::string format_reduction_statistics(const ReductionStats &s) {
  const unsigned total = s.narrowing_success + s.narrowing_fail;
  const double ratio =
      total ? 100.0 * static_cast<double>(s.narrowing_success) / total : 0.0;

  std::ostringstream oss;
  oss << block_header("Reduction Statistics");
  const std::size_t w = block_label_width({"Probe narrowing", "Raw equations",
                                           "Global equations", "Final equations",
                                           "Masters"});
  oss << block_rows(
             {block_row("Probe narrowing",
                        std::to_string(s.narrowing_success) + "/" +
                            std::to_string(s.narrowing_fail) + "/" +
                            std::to_string(total) + " (" + format_percent(ratio) +
                            " %)",
                        w),
              block_row("Raw equations", std::to_string(s.raw_equations), w),
              block_row("Global equations", std::to_string(s.global_equations), w),
              block_row("Final equations", std::to_string(s.final_equations), w),
              block_row("Masters", std::to_string(s.masters), w)})
      << "\n"
      << block_rule();
  return oss.str();
}

// One named phase of the run.
struct PhaseTiming {
  std::string label;
  double seconds = 0.0;
};

inline std::string format_timing_statistics(
    const std::vector<PhaseTiming> &phases, double total_seconds,
    std::uint64_t peak_bytes) {
  std::vector<std::string> labels;
  labels.reserve(phases.size() + 2);
  for (const auto &p : phases)
    labels.push_back(p.label);
  labels.push_back("Total");
  labels.push_back("Peak memory");

  std::ostringstream oss;
  oss << block_header("Timing Statistics");
  const std::size_t w = block_label_width(labels);
  std::vector<std::string> rows;
  rows.reserve(phases.size() + 2);
  for (const auto &p : phases)
    rows.push_back(block_row(p.label, format_duration(p.seconds), w));
  rows.push_back(block_row("Total", format_duration(total_seconds), w));
  rows.push_back(block_row("Peak memory", format_memory(peak_bytes), w));
  oss << block_rows(rows) << "\n" << block_rule();
  return oss.str();
}

// Totals printed when a global learn finishes.
struct LearnSummary {
  std::uint64_t independent_equations = 0;
  unsigned masters = 0;
  double seconds = 0.0;
};

inline std::string format_learn_summary(const LearnSummary &s,
                                        const char *tag = kFlowTag) {
  std::ostringstream oss;
  oss << "  " << tag << " learn done: " << s.independent_equations
      << " independent equations, " << s.masters << " masters ("
      << format_duration(s.seconds) << ")\n";
  return oss.str();
}

}
