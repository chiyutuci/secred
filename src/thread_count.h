#pragma once

#include <algorithm>
#include <thread>

namespace secred {

// Above this count the extra threads stop paying for themselves on the
// measured workloads: F234D at 128 threads was no faster than at 64 and its
// peak RSS grew by 73% (every FireFly probe holds its own state), and F234N at
// 128 threads was 2% slower with 44% more RSS (the numeric reconstruction
// starts a worker store per thread).  The cap keeps the automatic default from
// paying that; the numeric reconstruction's memory budget still caps its
// workers below this number, and an explicit sthreads/gthreads can exceed it.
inline constexpr unsigned kMaxDefaultThreads = 64;

// A configured thread count may exceed the hardware and the automatic default,
// but this bound refuses an obvious typo (a missing unit, say) before the run
// tries to start thousands of threads.
inline constexpr unsigned kMaxConfiguredThreads = 4096;

// Default number of worker threads: the machine's hardware thread count, up to
// kMaxDefaultThreads.
inline unsigned default_thread_count() {
  return std::min(kMaxDefaultThreads,
                  std::max(1u, std::thread::hardware_concurrency()));
}

}

