// SecRed's own reconstruction backend: FireFly's Reconstructor driven by a black box
// that wraps the shared SymbolicFFEvaluator.  FireFly takes no equations -- the whole
// system lives inside the box -- so this replaces the Kira global scheme (job file,
// FireFly subprocess, "dum" variable) with an in-process call.
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

#include "symbolic_ffred.h"

namespace secred {

// The lane helpers: FireFly instantiates its probe path for FFInt and for every bunch
// size FFIntVec<2,...,128>, so the box must be templated on the value type and read lane
// `l` of whichever it is handed.  We construct the four-argument Reconstructor, whose
// bunch_size stays 1, so only the plain path runs at runtime; the trait keeps the other
// instantiations correct rather than merely unreachable.
//
// The evaluator fills the outputs lane-major (`v[l * nout + i]`), so `make` takes the
// address of output `i`'s first lane and the distance between two of its lanes, and
// assembles whichever value type FireFly asked for.
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
  // FFIntVec<N> has no N-argument constructor; the array constructor is the one the
  // header offers.
  template <std::size_t... I>
  static firefly::FFIntVec<N> assemble(const std::uint64_t *v, std::size_t stride,
                                       std::index_sequence<I...>) {
    return firefly::FFIntVec<N>(
        std::array<firefly::FFInt, N>{firefly::FFInt(v[I * stride])...});
  }
};

class SecRedFireFlyBlackBox : public firefly::BlackBoxBase<SecRedFireFlyBlackBox> {
public:
  SecRedFireFlyBlackBox(SymbolicFFEvaluator &ev, unsigned nthreads)
      : ev_(ev), nthreads_(nthreads) {}

  // FireFly's contract: called after FFInt::p has been set to the new prime and before
  // any probe at it (its constructor calls this too, before the first probe).
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
    std::vector<FFTemp> out(n_outputs());
    if (hard_failure_.load(std::memory_order_relaxed))
      return out; // zeros
    const unsigned n = n_outputs();
    if (n == 0)
      return out; // every target is a master: nothing to reconstruct, nothing to reduce
    // One prime's tables per probe.  The prime can move between the read below and the
    // lock, so a probe that loses that race starts over rather than reporting a
    // failure: the shared lock then holds the tables still for the whole evaluation.
    for (;;) {
      const std::uint64_t p = cur_prime_.load(std::memory_order_acquire);
      ensure_reduced(p);
      std::shared_lock<std::shared_mutex> lock(mutex_);
      if (reduced_prime_.load(std::memory_order_acquire) != p)
        continue; // promoted under us; evaluate again at the prime it moved to
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
          // NotReduced cannot happen with the tables right; NonMasterRhs is the hard
          // failure the driver reports after reconstruct() returns.  Either way the
          // answer would be wrong, so no later probe is allowed to produce one.
          hard_failure_.store(true, std::memory_order_relaxed);
          fail_target_.store(scratch.fail_target, std::memory_order_relaxed);
          fail_integral_.store(scratch.fail_integral, std::memory_order_relaxed);
          return out;
        }
        // `Scratch::zero_dens` accumulates over the scratch's whole life, so count the
        // delta: the reported figure is a number of sample points, not a running sum.
        std::uint64_t &reported = reported_zero_dens();
        bad_points_.fetch_add(scratch.zero_dens - reported, std::memory_order_relaxed);
        reported = scratch.zero_dens;
      }
      for (unsigned i = 0; i < n; ++i)
        out[i] = Lane::make(&values[static_cast<std::size_t>(i)], n); // lane-major read
      return out;
    }
  }

  std::uint64_t probes() const { return probes_.load(std::memory_order_relaxed); }
  std::uint64_t bad_points() const { return bad_points_.load(std::memory_order_relaxed); }
  std::uint64_t primes() const { return primes_.load(std::memory_order_relaxed); }
  bool hard_failure() const { return hard_failure_.load(std::memory_order_relaxed); }
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

  SymbolicFFEvaluator &ev_;
  unsigned nthreads_;
  std::shared_mutex mutex_;
  std::atomic<std::uint64_t> cur_prime_{0};
  std::atomic<std::uint64_t> reduced_prime_{0};
  std::atomic<std::uint64_t> probes_{0};
  std::atomic<std::uint64_t> bad_points_{0};
  std::atomic<std::uint64_t> primes_{0};
  std::atomic<bool> hard_failure_{false};
  std::atomic<unsigned> fail_target_{0};
  std::atomic<unsigned> fail_integral_{0};
};

} // namespace secred
