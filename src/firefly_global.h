
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <utility>
#include <vector>

#include <firefly/BlackBoxBase.hpp>
#include <firefly/FFInt.hpp>
#include <firefly/FFIntVec.hpp>
#include <firefly/ReconstHelper.hpp>

#include "symbolic_ffred.h"

namespace secred {

// The reconstruction's prime cap: settings.max_primes, with no bound from FireFly's
// table.  lib/firefly_patch grows that table on demand (firefly::ensure_primes), so a
// reconstruction can use any number of prime fields.
inline unsigned firefly_prime_cap(unsigned max_primes) { return max_primes; }

// Lane access for scalar and vectorized FireFly values.
template <typename T> struct FireFlyLane;
template <> struct FireFlyLane<firefly::FFInt> {
  static constexpr unsigned lanes = 1;
  static std::uint64_t value(const firefly::FFInt &v, unsigned) { return v.n; }
  static firefly::FFInt make(const std::uint64_t *v, std::size_t) {
    return firefly::FFInt(v[0]);
  }
};
template <int N> struct FireFlyLane<firefly::FFIntVec<N>> {
  static constexpr unsigned lanes = N;
  static std::uint64_t value(const firefly::FFIntVec<N> &v, unsigned l) {
    return v.vec[l].n;
  }
  static firefly::FFIntVec<N> make(const std::uint64_t *v, std::size_t stride) {
    return assemble(v, stride, std::make_index_sequence<N>{});
  }

 private:

  template <std::size_t... I>
  static firefly::FFIntVec<N> assemble(const std::uint64_t *v, std::size_t stride,
                                       std::index_sequence<I...>) {
    return firefly::FFIntVec<N>(
        std::array<firefly::FFInt, N>{firefly::FFInt(v[I * stride])...});
  }
};

// FireFly black box backed by the symbolic evaluator.
class SecRedFireFlyBlackBox : public firefly::BlackBoxBase<SecRedFireFlyBlackBox> {
public:
  SecRedFireFlyBlackBox(SymbolicFFEvaluator &ev, unsigned nthreads)
      : ev_(ev), nthreads_(nthreads) {}

  void prime_changed() {
    const std::uint64_t p = firefly::FFInt::p;
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (reduced_prime_.load(std::memory_order_relaxed) != p) {
      ev_.reduce_for_prime(p, nthreads_);
      reduced_prime_.store(p, std::memory_order_release);
      primes_.fetch_add(1, std::memory_order_relaxed);
    }
    cur_prime_.store(p, std::memory_order_release);
  }

  template <typename FFTemp>
  std::vector<FFTemp> operator()(const std::vector<FFTemp> &x) {
    using Lane = FireFlyLane<FFTemp>;
    probes_.fetch_add(1, std::memory_order_relaxed);
    points_.fetch_add(Lane::lanes, std::memory_order_relaxed);
    std::vector<FFTemp> out(n_outputs());
    if (hard_failure_.load(std::memory_order_relaxed))
      return out;
    const unsigned n = n_outputs();
    if (n == 0)
      return out;

    for (;;) {
      const std::uint64_t p = cur_prime_.load(std::memory_order_acquire);
      ensure_reduced(p);
      std::shared_lock<std::shared_mutex> lock(mutex_);
      if (reduced_prime_.load(std::memory_order_acquire) != p)
        continue;
      const unsigned lanes = Lane::lanes;
      std::vector<std::uint64_t> values(static_cast<std::size_t>(lanes) * n);
      std::vector<std::uint64_t> point(ev_.n_params());
      for (unsigned l = 0; l < lanes; ++l) {
        for (unsigned i = 0; i < point.size(); ++i)
          point[i] = Lane::value(x[i], l);
        SymbolicFFEvaluator::Scratch &scratch = scratch_of();
        const SymbolicFFEvaluator::PointResult r =
            ev_.evaluate_point(p, point.data(), scratch,
                               &values[static_cast<std::size_t>(l) * n]);
        if (r != SymbolicFFEvaluator::PointResult::Ok) {

          hard_failure_.store(true, std::memory_order_relaxed);
          fail_reason_.store(static_cast<int>(r), std::memory_order_relaxed);
          fail_target_.store(scratch.fail_target, std::memory_order_relaxed);
          fail_integral_.store(scratch.fail_integral, std::memory_order_relaxed);
          return out;
        }

        std::uint64_t &reported = reported_zero_dens();
        bad_points_.fetch_add(scratch.zero_dens - reported, std::memory_order_relaxed);
        reported = scratch.zero_dens;

        PhaseSeconds &rep = reported_phase();
        eval_ns_.fetch_add(ns_since(rep.eval, scratch.eval_seconds),
                           std::memory_order_relaxed);
        fwd_ns_.fetch_add(ns_since(rep.fwd, scratch.fwd_seconds),
                          std::memory_order_relaxed);
        bwd_ns_.fetch_add(ns_since(rep.bwd, scratch.bwd_seconds),
                          std::memory_order_relaxed);

        FwdStats &seen = reported_fwd();
        fwd_rows_.fetch_add(scratch.fwd_stats.rows - seen.rows,
                            std::memory_order_relaxed);
        fwd_gathered_.fetch_add(scratch.fwd_stats.gathered - seen.gathered,
                                std::memory_order_relaxed);
        fwd_merges_.fetch_add(scratch.fwd_stats.merges - seen.merges,
                              std::memory_order_relaxed);
        fwd_moved_.fetch_add(scratch.fwd_stats.moved - seen.moved,
                             std::memory_order_relaxed);
        fwd_pivots_.fetch_add(scratch.fwd_stats.pivots - seen.pivots,
                              std::memory_order_relaxed);
        seen = scratch.fwd_stats;
      }
      for (unsigned i = 0; i < n; ++i)
        out[i] = Lane::make(&values[static_cast<std::size_t>(i)], n);
      return out;
    }
  }

  std::uint64_t probes() const { return probes_.load(std::memory_order_relaxed); }
  std::uint64_t points() const { return points_.load(std::memory_order_relaxed); }
  std::uint64_t bad_points() const { return bad_points_.load(std::memory_order_relaxed); }
  std::uint64_t primes() const { return primes_.load(std::memory_order_relaxed); }

  double eval_seconds() const { return eval_ns_.load(std::memory_order_relaxed) * 1e-9; }
  double fwd_seconds() const { return fwd_ns_.load(std::memory_order_relaxed) * 1e-9; }
  double bwd_seconds() const { return bwd_ns_.load(std::memory_order_relaxed) * 1e-9; }

  std::uint64_t fwd_rows() const { return fwd_rows_.load(std::memory_order_relaxed); }
  std::uint64_t fwd_gathered() const {
    return fwd_gathered_.load(std::memory_order_relaxed);
  }
  std::uint64_t fwd_merges() const {
    return fwd_merges_.load(std::memory_order_relaxed);
  }
  std::uint64_t fwd_moved() const { return fwd_moved_.load(std::memory_order_relaxed); }
  std::uint64_t fwd_pivots() const {
    return fwd_pivots_.load(std::memory_order_relaxed);
  }
  bool hard_failure() const { return hard_failure_.load(std::memory_order_relaxed); }

  SymbolicFFEvaluator::PointResult fail_reason() const {
    return static_cast<SymbolicFFEvaluator::PointResult>(
        fail_reason_.load(std::memory_order_relaxed));
  }
  unsigned fail_target() const { return fail_target_.load(std::memory_order_relaxed); }
  unsigned fail_integral() const { return fail_integral_.load(std::memory_order_relaxed); }

private:
  unsigned n_outputs() const { return ev_.n_outputs(); }

  void ensure_reduced(std::uint64_t p) {
    if (reduced_prime_.load(std::memory_order_acquire) == p)
      return;
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (reduced_prime_.load(std::memory_order_relaxed) != p) {
      ev_.reduce_for_prime(p, nthreads_);
      reduced_prime_.store(p, std::memory_order_release);
      primes_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  static SymbolicFFEvaluator::Scratch &scratch_of() {
    thread_local SymbolicFFEvaluator::Scratch scratch;
    return scratch;
  }

  static std::uint64_t &reported_zero_dens() {
    thread_local std::uint64_t reported = 0;
    return reported;
  }

  struct PhaseSeconds {
    double eval, fwd, bwd;
  };
  static PhaseSeconds &reported_phase() {
    thread_local PhaseSeconds reported{0.0, 0.0, 0.0};
    return reported;
  }

  static FwdStats &reported_fwd() {
    thread_local FwdStats reported;
    return reported;
  }

  static std::uint64_t ns_since(double &reported, double now) {
    const double d = now - reported;
    reported = now;
    return d > 0.0 ? static_cast<std::uint64_t>(d * 1e9) : 0;
  }

  SymbolicFFEvaluator &ev_;
  unsigned nthreads_;
  std::shared_mutex mutex_;
  std::atomic<std::uint64_t> cur_prime_{0};
  std::atomic<std::uint64_t> reduced_prime_{0};
  std::atomic<std::uint64_t> probes_{0};
  std::atomic<std::uint64_t> points_{0};
  std::atomic<std::uint64_t> bad_points_{0};
  std::atomic<std::uint64_t> eval_ns_{0};
  std::atomic<std::uint64_t> fwd_ns_{0};
  std::atomic<std::uint64_t> bwd_ns_{0};
  std::atomic<std::uint64_t> fwd_rows_{0};
  std::atomic<std::uint64_t> fwd_gathered_{0};
  std::atomic<std::uint64_t> fwd_merges_{0};
  std::atomic<std::uint64_t> fwd_moved_{0};
  std::atomic<std::uint64_t> fwd_pivots_{0};
  std::atomic<std::uint64_t> primes_{0};
  std::atomic<bool> hard_failure_{false};
  std::atomic<int> fail_reason_{0};
  std::atomic<unsigned> fail_target_{0};
  std::atomic<unsigned> fail_integral_{0};
};

}
