#include "test_runner.h"

#include "progress_bar.h"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// The progress display of the symbolic reconstruction: its formatting is a pure
// function of (done, total) so it can be checked without a terminal, and its update
// rate is throttled because the sample-point loop must not spend its time writing to
// stdout.

TEST(progress_bar_render_shows_bar_percent_and_counts) {
  const secred::ProgressBar bar("ev", 10, /*interactive=*/true, /*width=*/10);
  REQUIRE_EQ(bar.render(0), std::string("ev [----------]   0 %  0/10"));
  REQUIRE_EQ(bar.render(4), std::string("ev [####------]  40 %  4/10"));
  REQUIRE_EQ(bar.render(10), std::string("ev [##########] 100 %  10/10"));
}

TEST(progress_bar_clamps_past_the_total) {
  const secred::ProgressBar bar("ev", 10, true, 10);
  // A total of 100 % and a full bar, never 150 % and a longer bar; the count is
  // reported as given, so a caller that overshoots can see it.
  REQUIRE_EQ(bar.render(15), std::string("ev [##########] 100 %  15/10"));
}

TEST(progress_bar_survives_a_zero_total) {
  const secred::ProgressBar bar("ev", 0, true, 4);
  REQUIRE_EQ(bar.render(0), std::string("ev [####] 100 %  0/0"));
}

TEST(progress_bar_update_throttles_to_percent_steps) {
  secred::ProgressBar bar("ev", 100, /*interactive=*/true, 10);
  // One redraw per percent step: a carriage return, no newline.
  REQUIRE_EQ(bar.update(0), std::string("\r") + bar.render(0));
  REQUIRE_EQ(bar.update(0), std::string(""));
  REQUIRE_EQ(bar.update(1), std::string("\r") + bar.render(1));
  REQUIRE_EQ(bar.update(1), std::string(""));
  REQUIRE_EQ(bar.update(50), std::string("\r") + bar.render(50));
}

TEST(progress_bar_noninteractive_updates_are_whole_lines) {
  secred::ProgressBar bar("ev", 100, /*interactive=*/false, 10);
  // Without a terminal each update is a complete line, and the step is coarser.
  REQUIRE_EQ(bar.update(0), bar.render(0) + "\n");
  REQUIRE_EQ(bar.update(5), std::string(""));
  REQUIRE_EQ(bar.update(10), bar.render(10) + "\n");
  REQUIRE_EQ(bar.update(19), std::string(""));
  REQUIRE_EQ(bar.update(20), bar.render(20) + "\n");
}

TEST(progress_bar_finish_closes_the_line) {
  secred::ProgressBar interactive("ev", 100, true, 10);
  REQUIRE_EQ(interactive.finish(), std::string("\r") + interactive.render(100) + "\n");

  secred::ProgressBar plain("ev", 100, false, 10);
  REQUIRE_EQ(plain.finish(), plain.render(100) + "\n");
}

// The FireFly reconstruction, whose total is not known while it runs: the number of prime
// fields a rational function needs is decided by the reconstruction itself, so the line
// reports the counters the black box owns and the wall clock instead of a percentage.

TEST(recon_status_render_reports_the_counters_and_the_clock) {
  const secred::ReconStatus s("  [firefly] reconstructing", /*interactive=*/true);
  // 280 probes in 72 s: the rate is the black box's throughput, and "1min12s" is the
  // house duration format.
  REQUIRE_EQ(s.render(72.0, 7, 280, /*idle=*/0.4),
             std::string("  [firefly] reconstructing: 1min12s elapsed, 7 prime field(s), "
                         "280 probe(s) (3.9/s)"));
}

TEST(recon_status_render_omits_the_rate_before_a_second) {
  const secred::ReconStatus s("r", true);
  // A rate over a fraction of a second is noise, so the first line carries none.
  REQUIRE_EQ(s.render(0.4, 0, 3, 0.0),
             std::string("r: 0.4s elapsed, 0 prime field(s), 3 probe(s)"));
}

TEST(recon_status_render_names_a_stalled_probe_count) {
  const secred::ReconStatus s("r", true);
  // Between two prime fields the box is asked for nothing, and the frozen probe count
  // is explained rather than left looking like a hung run.
  REQUIRE_EQ(s.render(75.0, 7, 280, /*idle=*/6.3),
             std::string("r: 1min15s elapsed, 7 prime field(s), 280 probe(s) (3.7/s), "
                         "no probe for 6.3s"));
  // Under the threshold the counters are simply live.
  REQUIRE_EQ(s.render(75.0, 7, 280, /*idle=*/1.9),
             std::string("r: 1min15s elapsed, 7 prime field(s), 280 probe(s) (3.7/s)"));
}

TEST(recon_status_update_throttles_to_one_line_a_second) {
  secred::ReconStatus s("r", /*interactive=*/true);
  REQUIRE_EQ(s.update(1.0, 0, 3), std::string("\r") + s.render(1.0, 0, 3, 0.0));
  REQUIRE_EQ(s.update(1.5, 0, 4), std::string(""));
  REQUIRE_EQ(s.update(2.0, 0, 6), std::string("\r") + s.render(2.0, 0, 6, 0.0));
}

TEST(recon_status_update_waits_for_the_first_second) {
  secred::ReconStatus s("r", true);
  // The reconstruction starts with nothing done, and a "0 probe(s)" line is not worth
  // printing: the display opens once the run has a second behind it.
  REQUIRE_EQ(s.update(0.1, 0, 0), std::string(""));
  REQUIRE_EQ(s.update(0.9, 0, 2), std::string(""));
  REQUIRE_EQ(s.update(1.0, 0, 2), std::string("\r") + s.render(1.0, 0, 2, 0.0));
}

TEST(recon_status_update_times_a_stall_from_the_last_probe) {
  secred::ReconStatus s("r", /*interactive=*/true);
  // The probe count moving restarts the stall clock, and a throttled-away update still
  // feeds it: the counter is watched on every call, not only on the ones that print.
  REQUIRE_EQ(s.update(1.0, 0, 3), std::string("\r") + s.render(1.0, 0, 3, 0.0));
  REQUIRE_EQ(s.update(4.0, 0, 3), std::string("\r") + s.render(4.0, 0, 3, 3.0));
  REQUIRE_EQ(s.update(4.5, 3, 3), std::string(""));            // throttled, but noticed
  REQUIRE_EQ(s.update(5.0, 3, 4), std::string("\r") + s.render(5.0, 3, 4, 0.5));
}

TEST(recon_status_noninteractive_updates_are_whole_lines_ten_seconds_apart) {
  secred::ReconStatus s("r", /*interactive=*/false);
  // A redirected log gets one complete line per ten seconds instead of a redrawn one.
  REQUIRE_EQ(s.update(1.0, 0, 3), s.render(1.0, 0, 3, 0.0) + "\n");
  REQUIRE_EQ(s.update(6.0, 0, 9), std::string(""));
  REQUIRE_EQ(s.update(11.0, 1, 12), s.render(11.0, 1, 12, 0.0) + "\n");
}

TEST(recon_status_finish_closes_only_a_line_it_started) {
  secred::ReconStatus untouched("r", true);
  REQUIRE_EQ(untouched.finish(), std::string());   // nothing was printed
  untouched.update(1.0, 0, 3);
  REQUIRE_EQ(untouched.finish(), std::string("\n"));

  secred::ReconStatus plain("r", false);
  plain.update(1.0, 0, 3);
  REQUIRE_EQ(plain.finish(), std::string());       // its lines are already closed
}
