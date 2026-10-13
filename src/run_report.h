// The run's reporting vocabulary: the wall clock, the formatting of durations and
// memory figures, the dashed statistics blocks, and the exact text of the blocks
// the in-process FiniteFlow paths print.  Header-only so the unit tests drive the
// real formatting rather than a copy of it.
//
// Every block shares one frame — 57 columns of dashes with the title centred, a
// blank line, the rows, a blank line, and a closing rule — and one row layout:
// two spaces, the "label:" padded to a common column, then the value.  The blank
// lines inside the frame are what separate a block from the log around it once a
// run has scrolled past.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace secred {

// The `[fflow]` tag every line of the in-process FiniteFlow global reduction
// carries, so `grep '\[fflow\]'` collects the whole phase.
inline constexpr const char *kFlowTag = "[fflow]";

// ---------------------------------------------------------------------------
// Wall clock
// ---------------------------------------------------------------------------

// Start of the run, marked once at the top of main(); the end-of-run Total is
// measured from here, so it covers everything the process did.
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
} // namespace detail

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

// ---------------------------------------------------------------------------
// Value formatting
// ---------------------------------------------------------------------------

// A duration in seconds as "XhYminZs" / "XminYs" / "Xs".
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

// A memory figure, in the same GB the max_memory setting is given in (1024^3),
// so the budget and the peak can be compared by eye.
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

// ---------------------------------------------------------------------------
// The dashed frame
// ---------------------------------------------------------------------------

inline constexpr std::size_t kReportWidth = 57;

inline std::string block_header(const std::string &title) {
  const std::size_t pad = kReportWidth - title.size() - 2;
  const std::size_t left = pad / 2;
  return std::string(left, '-') + " " + title + " " +
         std::string(pad - left, '-') + "\n";
}

inline std::string block_rule() { return std::string(kReportWidth, '-') + "\n"; }

// The column every row's value starts at: one space past the longest "label:".
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

// The rows of a block, one blank line apart — the rows come back as a unit, so
// the caller frames them with a single "\n" on each side.  A block of n rows
// reads as 2n+1 lines: every row is separated from its neighbours by a blank
// line, and from the title and the closing rule as well.
inline std::string block_rows(const std::vector<std::string> &rows) {
  std::string out;
  for (const auto &row : rows)
    out += "\n" + row;
  return out;
}

// A share of a whole, one decimal, as the statistics blocks print it.
inline std::string format_percent(double ratio) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1) << ratio;
  return oss.str();
}

// ---------------------------------------------------------------------------
// Reduction Statistics (the in-process FiniteFlow paths, printed after the learn
// pass and before the reconstruction)
// ---------------------------------------------------------------------------

struct ReductionStats {
  unsigned narrowing_success = 0;
  unsigned narrowing_fail = 0;
  std::uint64_t raw_equations = 0;
  std::uint64_t global_equations = 0;
  // Equations the mark-and-sweep keeps, i.e. the ones each prime's elimination
  // and the reconstruction actually run on.
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

// ---------------------------------------------------------------------------
// Timing Statistics (every path, printed at the end of the run)
// ---------------------------------------------------------------------------

struct PhaseTiming {
  std::string label;
  double seconds = 0.0;
};

// The peak memory the run reports is what the run needed, and the global stage of
// the external-solver path lives in a child process the kernel measures
// separately.  When that child's high-water mark is known (child_peak_bytes > 0)
// the block splits the figure in two: SecRed's own, and the run's including the
// solver — otherwise a run that spent its memory in kira would look frugal.  The
// second row is the larger of the two, since that is the peak the run reached.
inline std::string format_timing_statistics(
    const std::vector<PhaseTiming> &phases, double total_seconds,
    std::uint64_t peak_bytes, std::uint64_t child_peak_bytes = 0,
    const std::string &child_name = {}) {
  const bool with_child = child_peak_bytes > 0 && !child_name.empty();
  const std::uint64_t run_peak =
      std::max(peak_bytes, with_child ? child_peak_bytes : 0);
  const std::string own_label =
      with_child ? "Peak memory (SecRed)" : "Peak memory";
  const std::string child_label =
      "Peak memory (including " + child_name + ")";

  std::vector<std::string> labels;
  labels.reserve(phases.size() + 3);
  for (const auto &p : phases)
    labels.push_back(p.label);
  labels.push_back("Total");
  labels.push_back(own_label);
  if (with_child)
    labels.push_back(child_label);

  std::ostringstream oss;
  oss << block_header("Timing Statistics");
  const std::size_t w = block_label_width(labels);
  std::vector<std::string> rows;
  rows.reserve(phases.size() + 3);
  for (const auto &p : phases)
    rows.push_back(block_row(p.label, format_duration(p.seconds), w));
  rows.push_back(block_row("Total", format_duration(total_seconds), w));
  rows.push_back(block_row(own_label, format_memory(peak_bytes), w));
  if (with_child)
    rows.push_back(block_row(child_label, format_memory(run_peak), w));
  oss << block_rows(rows) << "\n" << block_rule();
  return oss.str();
}

// ---------------------------------------------------------------------------
// The learn summary of the in-process FiniteFlow paths
// ---------------------------------------------------------------------------

// What the learn pass of the in-process FiniteFlow paths reports as soon as it
// has fixed the basis.  The mark-and-sweep counters it used to follow with are
// not logged any more — the sweep's outcome is the Reduction Statistics block's
// "Final equations" row.
struct LearnSummary {
  std::uint64_t independent_equations = 0;
  unsigned masters = 0;
  double seconds = 0.0;
};

inline std::string format_learn_summary(const LearnSummary &s) {
  std::ostringstream oss;
  oss << "  " << kFlowTag << " learn done: " << s.independent_equations
      << " independent equations, " << s.masters << " masters ("
      << format_duration(s.seconds) << ")\n";
  return oss.str();
}

} // namespace secred
