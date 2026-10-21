#include "test_runner.h"

#include "utils.h"

#include <atomic>
#include <cstdint>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

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

// Every sector worker stamps its log through this one function.  It used to call
// localtime()/asctime(), which hand back process-wide static buffers (and glibc's
// localtime() rewrites the shared timezone state), so concurrent calls raced.
// Hammer it from several threads: every line must be exactly the message plus one
// 24-character asctime stamp and the two newlines (asctime's own and std::endl's),
// and nothing may be lost or interleaved.  Under ThreadSanitizer the old form
// reports the race here; the _r form is clean.
TEST(time_stamp_is_safe_from_several_threads) {
  constexpr unsigned kThreads = 4;
  constexpr unsigned kIterations = 2000;
  std::atomic<unsigned> ready{0};
  std::atomic<unsigned> calls{0};
  std::atomic<unsigned> malformed{0};
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (unsigned t = 0; t < kThreads; ++t)
    threads.emplace_back([&, t]() {
      const std::string msg = "worker " + std::to_string(t) + ": ";
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
      }
      for (unsigned i = 0; i < kIterations; ++i) {
        std::ostringstream out;
        time_stamp(out, msg);
        const std::string text = out.str();
        const bool framed = text.size() == msg.size() + 26 &&
                            text.compare(0, msg.size(), msg) == 0 &&
                            text.compare(text.size() - 2, 2, "\n\n") == 0;
        if (!framed)
          malformed.fetch_add(1, std::memory_order_relaxed);
        calls.fetch_add(1, std::memory_order_relaxed);
      }
    });
  while (ready.load(std::memory_order_acquire) != kThreads) {
  }
  start.store(true, std::memory_order_release);
  for (auto &t : threads)
    t.join();
  REQUIRE_EQ(calls.load(), kThreads * kIterations);
  REQUIRE_EQ(malformed.load(), 0u);
}
