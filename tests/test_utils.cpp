#include "test_runner.h"

#include "utils.h"

#include <cstdint>

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
