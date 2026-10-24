
#include "firefly_global.h"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fflow/primes.hh>
#include <firefly/Reconstructor.hpp>

#include "firefly_patch.h"
#include "globalreduce.h"
#include "progress_bar.h"
#include "run_report.h"
#include "spill_reader.h"
#include "symbolic_ffred.h"

GlobalReductionResult secred::reduce_global_firefly_symbolic(
    const std::string &raw_equations_path,
    const std::vector<RawIntegral> &all_integrals,
    const std::unordered_map<RawIntegral, unsigned> &integral_id,
    const std::vector<RawIntegral> &targets,
    const std::vector<GiNaC::possymbol> &symbols,
    unsigned narrowing_success, unsigned narrowing_fail,
    unsigned raw_equations_total, std::size_t global_equations_count,
    const std::string &masters_global_path, unsigned nthreads,
    unsigned max_primes, const BasisVote &basis_vote) {
  GlobalReductionResult res;
  if (nthreads == 0)
    nthreads = 1;
  const unsigned prime_cap = secred::firefly_prime_cap(max_primes);
  // FireFly indexes its prime table directly up to the cap, and the per-function
  // interpolation may read one entry past the current prime; lib/firefly_patch grows
  // the table (FireFly's own 300 plus generated primes) so the cap has no table bound.
  const std::size_t table_primes = firefly::primes().size();
  firefly::ensure_primes(static_cast<std::size_t>(prime_cap) + 1);
  const unsigned nparams = symbols.size();
  const auto t0 = std::chrono::steady_clock::now();
  std::cout << "\n  " << secred::kFireFlyTag << " reading spill file, building symbolic "
               "system...\n";

  secred::SymCoeffPool pool;
  secred::SpillReadStats stats;
  std::string read_error;
  if (!secred::read_spill_interned(raw_equations_path, integral_id, symbols, pool, stats,
                                   read_error)) {
    std::cerr << "firefly symbolic global reduction: " << read_error << "\n";
    return res;
  }
  const unsigned neqs = static_cast<unsigned>(pool.n_equations());
  if (neqs == 0) {
    std::cerr << "firefly symbolic global reduction: empty system\n";
    return res;
  }
  const auto t1 = std::chrono::steady_clock::now();
  std::cout << "  " << secred::kFireFlyTag << " symbolic system built: " << neqs
            << " equations x " << all_integrals.size() << " variables, " << nparams
            << " parameters ("
            << secred::format_duration(std::chrono::duration<double>(t1 - t0).count())
            << ")\n";

  std::vector<unsigned> target_ids;
  for (const auto &t : targets) {
    auto it = integral_id.find(t);
    if (it != integral_id.end())
      target_ids.push_back(it->second);
  }
  if (target_ids.empty()) {
    std::cerr << "firefly symbolic global reduction: no targets\n";
    return res;
  }

  secred::SymbolicGlobalLearn learn =
      secred::symbolic_global_learn(pool, target_ids, all_integrals, symbols,
                                    fflow::BIG_UINT_PRIMES[0], masters_global_path,
                                    secred::kFireFlyTag, narrowing_success,
                                    narrowing_fail, raw_equations_total,
                                    global_equations_count);
  if (!learn.ok) {
    std::cerr << "firefly symbolic global reduction: " << learn.error << "\n";
    return res;
  }
  res.selected_equations = learn.selected_equations;
  res.raw_equations = neqs;
  const unsigned nnindeps = learn.nnindeps;

  for (unsigned m : learn.masters)
    res.masters.push_back(all_integrals[m]);
  res.learn_seconds = learn.learn_seconds;

  if (basis_vote && basis_vote(res.masters)) {
    res.reconstruction_skipped = true;
    res.ok = true;
    return res;
  }

  if (learn.dep_targets.empty() || nnindeps == 0) {
    for (const auto &t : targets)
      res.reductions.emplace_back(t, std::string("-1"));
    res.reconstruction_seconds = 0.0;
    res.ok = true;
    return res;
  }

  std::vector<GiNaC::ex> symbol_ex;
  symbol_ex.reserve(nparams);
  for (const auto &s : symbols)
    symbol_ex.push_back(GiNaC::ex(s));

  secred::SymbolicFFEvaluator ev(std::move(pool), std::move(learn.needed_eq_idx),
                                 std::move(learn.bwd_chain), learn.dep_targets,
                                 std::move(learn.master_index), nnindeps, learn.items,
                                 std::move(symbol_ex));
  const unsigned nparsout = ev.n_outputs();
  std::cout << "\n  " << secred::kFireFlyTag << " reconstruct (rational-function, up to "
            << prime_cap << " primes";
  if (prime_cap > table_primes)
    std::cout << "; " << table_primes << " from the FireFly table, the rest generated";
  std::cout << "): " << nparsout << " function(s) will be interpolated (the non-zero "
            << "entries of a " << learn.dep_targets.size() << " x " << nnindeps
            << " target x master block)\n";

  std::cout << "  " << secred::kFireFlyTag << " " << ev.n_coefficient_terms()
            << " function(s) to evaluate per sample point, of which "
            << ev.n_distinct_coeffs() << " are distinct\n";
  if (stats.fallbacks)
    std::cout << "  " << secred::kFireFlyTag << " " << stats.fallbacks
              << " coefficient(s) needed the GiNaC parser\n";

  secred::SecRedFireFlyBlackBox box(ev, nthreads);
  const auto t2 = std::chrono::steady_clock::now();
  {

    firefly::Reconstructor<secred::SecRedFireFlyBlackBox> rec(
        static_cast<unsigned>(nparams), nthreads, box,
        firefly::Reconstructor<secred::SecRedFireFlyBlackBox>::SILENT);

    secred::ReconStatus status(
        std::string("  ") + secred::kFireFlyTag + " reconstructing",
        secred::stdout_is_tty());
    std::jthread watcher([&](std::stop_token stop) {
      while (!stop.stop_requested()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (stop.stop_requested())
          break;
        const double elapsed = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - t2)
                                   .count();
        const std::string line = status.update(elapsed, box.primes(), box.probes());
        if (!line.empty())
          std::cout << line << std::flush;
      }
      std::cout << status.finish() << std::flush;
    });
    rec.reconstruct(prime_cap);

    {
      const std::uint64_t pts = box.points();
      const double per = pts ? static_cast<double>(pts) : 1.0;
      char line[256];
      std::snprintf(line, sizeof(line),
                    "  %s probe solution steps (average per probe): coefficient "
                    "evaluation %.3f s / forward elimination %.3f s / back "
                    "substitution %.3f s\n",
                    secred::kFireFlyTag, box.eval_seconds() / per,
                    box.fwd_seconds() / per, box.bwd_seconds() / per);
      std::cout << line;

      std::snprintf(line, sizeof(line),
                    "  %s probe elimination (average per probe): %llu row merges moving "
                    "%llu terms / %llu solutions from %llu terms gathered\n",
                    secred::kFireFlyTag,
                    static_cast<unsigned long long>(box.fwd_merges() / per),
                    static_cast<unsigned long long>(box.fwd_moved() / per),
                    static_cast<unsigned long long>(box.fwd_pivots() / per),
                    static_cast<unsigned long long>(box.fwd_gathered() / per));
      std::cout << line;
    }
    if (box.hard_failure()) {
      if (box.fail_reason() ==
          secred::SymbolicFFEvaluator::PointResult::MissingItem)
        std::cerr << "firefly symbolic global reduction: target " << box.fail_target()
                  << " reduces to integral " << box.fail_integral()
                  << ", a master the learn point did not carry (equation selection is "
                     "stale); aborting instead of emitting a truncated reduction\n";
      else if (box.fail_reason() ==
               secred::SymbolicFFEvaluator::PointResult::NonMasterRhs)
        std::cerr << "firefly symbolic global reduction: target " << box.fail_target()
                  << " reduces to integral " << box.fail_integral()
                  << ", which is not a master (equation selection is stale); aborting "
                     "instead of emitting a truncated reduction\n";
      else
        std::cerr << "firefly symbolic global reduction: a probe was refused (the run "
                     "is unreliable); aborting instead of emitting a truncated "
                     "reduction\n";
      return res;
    }
    const std::vector<firefly::RationalFunction> &out = rec.get_result();

    if (out.size() != nparsout) {
      std::cerr << "firefly symbolic global reduction: the reconstruction finished "
                << out.size() << " of " << nparsout
                << " coefficients; aborting instead of emitting a truncated reduction\n";
      return res;
    }

    std::vector<std::string> varnames(nparams);
    for (unsigned i = 0; i < nparams; ++i)
      varnames[i] = symbols[i].get_name();
    std::unordered_map<unsigned, unsigned> dep_index;
    for (unsigned i = 0; i < learn.dep_targets.size(); ++i)
      dep_index[learn.dep_targets[i]] = i;
    for (const auto &t : targets) {
      res.reductions.emplace_back(t, std::string("-1"));
      auto it = integral_id.find(t);
      if (it == integral_id.end())
        continue;
      auto dit = dep_index.find(it->second);
      if (dit == dep_index.end())
        continue;
      const unsigned i = dit->second;
      for (unsigned q = learn.items.begin[i]; q < learn.items.begin[i + 1]; ++q) {
        const firefly::RationalFunction &r = out[q];
        if (r.zero())
          continue;
        res.reductions.emplace_back(
            all_integrals[learn.masters[learn.items.col[q]]], r.to_string(varnames));
      }
    }
  }
  const auto t3 = std::chrono::steady_clock::now();
  std::cout << "  " << secred::kFireFlyTag << " reconstruction done in "
            << secred::format_duration(std::chrono::duration<double>(t3 - t2).count())
            << " (" << box.probes() << " probes, " << box.primes() << " primes, "
            << box.bad_points() << " distinct coefficient(s) with a vanishing "
               "denominator)\n";
  res.reconstruction_seconds = std::chrono::duration<double>(t3 - t2).count();
  res.ok = true;
  return res;
}
