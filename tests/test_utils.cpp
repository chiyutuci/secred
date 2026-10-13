#include "test_runner.h"

#include "utils.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sys/wait.h>
#include <unistd.h>

// The memory figures the run reports: the current resident set (against which the
// max_memory budget is enforced) and the high-water mark of the whole run, which
// the end-of-run Timing Statistics block reports.  The peak must never come out
// below the current usage — a peak that moves with the current size is not a peak.

TEST(process_peak_memory_is_at_least_the_current_usage) {
  const std::uint64_t peak = process_peak_memory();
  const std::uint64_t used = process_used_memory();
  REQUIRE(peak > 0);
  REQUIRE(peak >= used);
}

TEST(process_peak_memory_does_not_shrink) {
  const std::uint64_t first = process_peak_memory();
  const std::uint64_t second = process_peak_memory();
  REQUIRE(second >= first);
}

TEST(system_memory_reports_a_total_above_the_available_memory) {
  const SystemMemory sm = system_memory();
  REQUIRE(sm.total > 0);
  REQUIRE(sm.available <= sm.total);
}

// The equations the global stage is measured by are the ones the external solver
// selected out of the system SecRed handed it, so the run reads that count back
// from the solver's own log rather than assuming it equals what it sent.

TEST(parse_selected_equations_reads_the_kira_selection_line) {
  REQUIRE_EQ(parse_selected_equations(
                 "***** Select equations recursively ************************\n"
                 "Number of selected equations to reduce: 90 equations\n"
                 "( 0.006167 s )\n"),
             std::optional<std::uint64_t>(90));
}

TEST(parse_selected_equations_takes_the_last_selection) {
  // Kira logs the selection once per reduction pass; the reconstruction the run
  // reports on is the last one.
  REQUIRE_EQ(parse_selected_equations(
                 "Number of selected equations to reduce: 108 equations\n"
                 "...\n"
                 "Number of selected equations to reduce: 90 equations\n"),
             std::optional<std::uint64_t>(90));
}

TEST(parse_selected_equations_accepts_zero) {
  // Kira does select nothing on some systems, and that is a result, not a
  // missing value: 0 must not read as "unknown".
  REQUIRE_EQ(parse_selected_equations(
                 "Number of selected equations to reduce: 0 equations\n"),
             std::optional<std::uint64_t>(0));
}

TEST(parse_selected_equations_reports_absence) {
  REQUIRE(!parse_selected_equations("").has_value());
  REQUIRE(!parse_selected_equations("Kira said nothing about equations\n")
               .has_value());
  REQUIRE(!parse_selected_equations("Number of selected equations to reduce:\n")
               .has_value());
}

TEST(parse_selected_equations_survives_ansi_colour_codes) {
  // Kira colours its output; the line must still be readable when the log keeps
  // the escape sequences — including when the colour wraps the number itself, so
  // the digits inside an escape sequence must not be mistaken for the count.
  REQUIRE_EQ(parse_selected_equations(
                 "\x1b[1;34mNumber of selected equations to reduce:\x1b[0m "
                 "\x1b[32m12\x1b[0m equations\n"),
             std::optional<std::uint64_t>(12));
}

TEST(selected_equations_from_log_reads_the_file_and_reports_a_missing_one) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "secred_test_kira_log.txt";
  {
    std::ofstream out(path);
    out << "some kira chatter\n"
        << "Number of selected equations to reduce: 90 equations\n"
        << "more chatter\n";
  }
  REQUIRE_EQ(selected_equations_from_log(path.string()),
             std::optional<std::uint64_t>(90));
  std::filesystem::remove(path);
  REQUIRE(!selected_equations_from_log(path.string()).has_value());
  REQUIRE(!selected_equations_from_log("/nonexistent/secred/kira.log").has_value());
}

// Peak memory of the whole run, not just of SecRed: the global stage lives in a
// child process (Kira), whose high-water mark the kernel attributes to the parent
// when the child is reaped.  A child that touches 64 MB must show up in it.
TEST(child_peak_memory_counts_a_child_allocation) {
  // A static buffer rather than malloc: the child of a fork must not allocate,
  // in case another thread held the allocator lock.
  constexpr std::size_t kBytes = 64ull * 1024 * 1024;
  static volatile char block[kBytes];

  const pid_t pid = fork();
  REQUIRE(pid >= 0);
  if (pid == 0) {
    for (std::size_t i = 0; i < kBytes; i += 4096)
      block[i] = 1;  // touch every page so the RSS is real, not just reserved
    _exit(0);
  }

  int status = 0;
  REQUIRE(waitpid(pid, &status, 0) == pid);
  REQUIRE(WIFEXITED(status));
  REQUIRE_EQ(WEXITSTATUS(status), 0);

  REQUIRE(child_peak_memory() >= 64ull * 1024 * 1024);
}
