#include "test_runner.h"

#include "run_report.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// The run's reporting vocabulary: how a duration and a memory figure are written,
// how the dashed statistics blocks are framed and aligned, and the exact text of
// the blocks printed by the in-process FiniteFlow paths and at the end of a run.
// These are pinned because the log is the program's user interface — a phase that
// silently stops being reported, or a block that loses its blank lines, is a
// regression that no other test would catch.

TEST(run_report_formats_a_duration_at_three_scales) {
  REQUIRE_EQ(secred::format_duration(0.0), std::string("0.0s"));
  REQUIRE_EQ(secred::format_duration(15.34), std::string("15.3s"));
  REQUIRE_EQ(secred::format_duration(90.0), std::string("1min30s"));
  REQUIRE_EQ(secred::format_duration(237.0), std::string("3min57s"));
  REQUIRE_EQ(secred::format_duration(3723.0), std::string("1h2min3s"));
}

TEST(run_report_formats_a_memory_figure_at_three_scales) {
  REQUIRE_EQ(secred::format_memory(0), std::string("0 B"));
  REQUIRE_EQ(secred::format_memory(512), std::string("512 B"));
  REQUIRE_EQ(secred::format_memory(5u * 1024 * 1024), std::string("5.0 MB"));
  REQUIRE_EQ(secred::format_memory(48000000000ull), std::string("44.70 GB"));
}

TEST(run_report_block_header_frames_the_title_in_57_columns) {
  const std::string header =
      secred::block_header("Reduction Statistics");
  REQUIRE_EQ(header, std::string("----------------- Reduction Statistics "
                                 "------------------\n"));
  REQUIRE_EQ(header.size(), std::size_t(58)); // 57 columns + the newline
  REQUIRE_EQ(secred::block_header("Timing Statistics"),
             std::string("------------------- Timing Statistics "
                         "-------------------\n"));
}

TEST(run_report_block_rule_matches_the_header_width) {
  REQUIRE_EQ(secred::block_rule(), std::string(57, '-') + "\n");
}

TEST(run_report_block_rows_align_on_a_common_column) {
  // One space past the longest "label:" — the column the statistics blocks use.
  REQUIRE_EQ(secred::block_label_width({"Probe narrowing", "Global equations"}),
             std::size_t(18));
  // The rule follows the longest label, colon included, plus the separating space.
  REQUIRE_EQ(secred::block_label_width({"Total"}), std::size_t(7));
  REQUIRE_EQ(secred::block_row("Probe narrowing", "9/4/13 (69.2 %)", 18),
             std::string("  Probe narrowing:  9/4/13 (69.2 %)\n"));
  REQUIRE_EQ(secred::block_row("Global equations", "108", 18),
             std::string("  Global equations: 108\n"));
}

TEST(run_report_block_rows_are_one_blank_line_apart) {
  REQUIRE_EQ(secred::block_rows({std::string("  a: 1\n"), std::string("  b: 2\n")}),
             std::string("\n  a: 1\n\n  b: 2\n"));
  REQUIRE_EQ(secred::block_rows({}), std::string(""));
}

TEST(run_report_reduction_statistics_block_has_blank_lines_inside_the_frame) {
  secred::ReductionStats s;
  s.narrowing_success = 9;
  s.narrowing_fail = 4;
  s.raw_equations = 536;
  s.global_equations = 108;
  s.final_equations = 91;
  s.masters = 9;

  REQUIRE_EQ(
      secred::format_reduction_statistics(s),
      std::string("----------------- Reduction Statistics ------------------\n"
                  "\n"
                  "  Probe narrowing:  9/4/13 (69.2 %)\n"
                  "\n"
                  "  Raw equations:    536\n"
                  "\n"
                  "  Global equations: 108\n"
                  "\n"
                  "  Final equations:  91\n"
                  "\n"
                  "  Masters:          9\n"
                  "\n"
                  "---------------------------------------------------------\n"));
}

TEST(run_report_reduction_statistics_survives_a_run_with_no_sectors) {
  const secred::ReductionStats s; // every count at zero
  const std::string block = secred::format_reduction_statistics(s);
  REQUIRE(block.find("  Probe narrowing:  0/0/0 (0.0 %)\n") != std::string::npos);
  REQUIRE(block.find("nan") == std::string::npos);
}

TEST(run_report_learn_summary_is_one_line_of_learn_done) {
  // The mark-and-sweep details used to follow on their own lines; they are no
  // longer logged, and nothing else may creep back in.
  secred::LearnSummary ls;
  ls.independent_equations = 898603;
  ls.masters = 374;
  ls.seconds = 15.34;

  const std::string summary = secred::format_learn_summary(ls);
  REQUIRE_EQ(summary,
             std::string("  [fflow] learn done: 898603 independent equations, "
                         "374 masters (15.3s)\n"));
  REQUIRE_EQ(std::count(summary.begin(), summary.end(), '\n'), std::ptrdiff_t(1));
}

TEST(run_report_timing_statistics_block_lists_every_phase) {
  const std::vector<secred::PhaseTiming> phases{
      {"Sector reduction", 90.0},
      {"Global reduction", 300.0},
      {"Global learn", 15.34},
      {"Reconstruction", 237.0}};

  REQUIRE_EQ(
      secred::format_timing_statistics(phases, 400.0, 48000000000ull),
      std::string("------------------- Timing Statistics -------------------\n"
                  "\n"
                  "  Sector reduction: 1min30s\n"
                  "\n"
                  "  Global reduction: 5min0s\n"
                  "\n"
                  "  Global learn:     15.3s\n"
                  "\n"
                  "  Reconstruction:   3min57s\n"
                  "\n"
                  "  Total:            6min40s\n"
                  "\n"
                  "  Peak memory:      44.70 GB\n"
                  "\n"
                  "---------------------------------------------------------\n"));
}


TEST(run_report_clock_measures_the_elapsed_time) {
  const auto t0 =
      std::chrono::steady_clock::now() - std::chrono::milliseconds(250);
  const double elapsed = secred::seconds_since(t0);
  REQUIRE(elapsed >= 0.25);
  REQUIRE(elapsed < 60.0);

  secred::mark_run_start();
  const double run = secred::run_elapsed_seconds();
  REQUIRE(run >= 0.0);
  REQUIRE(run < 60.0);
}

TEST(run_report_names_the_flow_tag_once) {
  REQUIRE_EQ(std::string(secred::kFlowTag), std::string("[fflow]"));
}
