#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace secred {

// A fixed set of worker threads that survives across loops.  The numeric
// reconstruction runs CRT/rat_rec after every prime, and building a thread
// vector for each of them spends the dispatch time in the kernel and wakes the
// same cores again a moment later.  A pool owns its threads for the whole
// reconstruction instead; the calling thread does one share of every loop, so
// a pool of n threads has n - 1 workers.
//
// The symbolic path keeps its one-shot parallel_for: its loops run once per
// prime with FireFly's own sample threads in between, and keeping these
// workers alive across a sample sweep measured 2-3% slower on small runs
// (box, sunrise) -- the idle threads cost more than the next dispatch saves.
//
// `blocks` splits [0, n) into contiguous ranges for the pool's threads (what
// the old parallel_for did); `dynamic` lets each free thread claim the next
// index (parallel_for_dynamic).  One loop runs at a time; a nested loop would
// overwrite the running job, so it throws std::logic_error.
//
// A job that throws stops its own share, the other shares finish, and the
// exception is rethrown on the calling thread once every worker has stopped.
// The pool is then idle and ready for the next loop, never left with a stale
// job in flight.
class ParallelPool {
public:
  explicit ParallelPool(unsigned nthreads) : size_(nthreads == 0 ? 1 : nthreads) {
    workers_.reserve(size_ - 1);
    try {
      for (unsigned w = 1; w < size_; ++w)
        workers_.emplace_back([this, w] { worker(w); });
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(m_);
        shutdown_ = true;
      }
      work_cv_.notify_all();
      for (auto &t : workers_)
        if (t.joinable())
          t.join();
      throw;
    }
  }

  ~ParallelPool() {
    {
      std::lock_guard<std::mutex> lock(m_);
      shutdown_ = true;
    }
    work_cv_.notify_all();
    for (auto &t : workers_)
      t.join();
  }

  ParallelPool(const ParallelPool &) = delete;
  ParallelPool &operator=(const ParallelPool &) = delete;

  template <typename F> void blocks(std::size_t n, F f) { run(n, false, f); }

  template <typename F> void dynamic(std::size_t n, F f) { run(n, true, f); }

private:
  template <typename F> void run(std::size_t n, bool dynamic, F &f) {
    if (n == 0)
      return;
    // With no workers there is no shared job state to protect, so a loop is
    // just a serial call and nesting is harmless.
    if (size_ == 1) {
      for (std::size_t i = 0; i < n; ++i)
        f(i);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(m_);
      if (busy_)
        throw std::logic_error(
            "ParallelPool: a loop is already running; nested loops are not supported");
      busy_ = true;
      ctx_ = &f;
      call_ = [](void *ctx, std::size_t i) { (*static_cast<F *>(ctx))(i); };
      n_ = n;
      dynamic_ = dynamic;
      next_.store(0, std::memory_order_relaxed);
      err_ = nullptr;
      active_ = size_; // the calling thread's share is 0
      ++generation_;
    }
    work_cv_.notify_all();
    try {
      share(0);
    } catch (...) {
      std::lock_guard<std::mutex> lock(m_);
      if (!err_)
        err_ = std::current_exception();
    }
    std::exception_ptr err;
    {
      std::unique_lock<std::mutex> lock(m_);
      // The calling thread's own share is done too; the loop ends when every
      // share has stopped.
      if (--active_ == 0)
        done_cv_.notify_all();
      done_cv_.wait(lock, [&] { return active_ == 0; });
      busy_ = false;
      err = err_;
      err_ = nullptr;
      ctx_ = nullptr;
    }
    if (err)
      std::rethrow_exception(err);
  }

  void worker(unsigned w) {
    unsigned seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(m_);
        work_cv_.wait(lock, [&] { return shutdown_ || generation_ != seen; });
        if (shutdown_)
          return;
        seen = generation_;
      }
      try {
        share(w);
      } catch (...) {
        std::lock_guard<std::mutex> lock(m_);
        if (!err_)
          err_ = std::current_exception();
      }
      {
        std::lock_guard<std::mutex> lock(m_);
        if (--active_ == 0)
          done_cv_.notify_all();
      }
    }
  }

  // The job fields below are written by run() under m_ before generation_ is
  // bumped, and read here after the worker's wait reacquires m_ and observes the
  // new generation.
  void share(unsigned w) {
    if (dynamic_) {
      for (;;) {
        const std::size_t i = next_.fetch_add(1, std::memory_order_relaxed);
        if (i >= n_)
          break;
        call_(ctx_, i);
      }
    } else {
      const std::size_t begin = n_ * w / size_;
      const std::size_t end = n_ * (w + 1) / size_;
      for (std::size_t i = begin; i < end; ++i)
        call_(ctx_, i);
    }
  }

  const unsigned size_;

  std::mutex m_;
  std::condition_variable work_cv_, done_cv_;

  // Published job, guarded by m_.
  void *ctx_ = nullptr;
  void (*call_)(void *, std::size_t) = nullptr;
  std::size_t n_ = 0;
  bool dynamic_ = false;
  unsigned generation_ = 0;
  unsigned active_ = 0;
  bool busy_ = false;
  bool shutdown_ = false;
  std::exception_ptr err_;

  // Claimed without the lock; reset under it before each loop.
  std::atomic<std::size_t> next_{0};

  std::vector<std::thread> workers_;
};

}

