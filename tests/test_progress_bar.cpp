#include "test_runner.h"

#include "progress_bar.h"

#include <cstddef>
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
