// The `global_tool: firefly` backend's driver: the same spill read, the same shared
// learn and the same shared evaluator as the FiniteFlow symbolic path, with FireFly's
// Reconstructor (driven by the black box of firefly_global.h) as the rational-function
// reconstruction engine.  Nothing here talks to a subprocess.
#include "firefly_global.h"

#include <chrono>
#include <cstddef>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <fflow/primes.hh> // BIG_UINT_PRIMES: the shared learn prime
#include <firefly/Reconstructor.hpp>

#include "globalreduce.h"
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
    const std::string &masters_final_path, unsigned nthreads) {
  GlobalReductionResult res;
  if (nthreads == 0)
    nthreads = 1;
  const unsigned nparams = symbols.size();
  const auto t0 = std::chrono::steady_clock::now();
  std::cout << "\n  " << secred::kFireFlyTag << " reading spill file, building symbolic "
               "system...\n";

  // 1) Read the spill (the shared scanner) and 2) pick the equations (the shared learn).
  std::vector<SymEq> system;
  secred::SpillReadStats stats;
  std::string read_error;
  if (!secred::read_spill_symbolic(raw_equations_path, integral_id, symbols, system,
                                   stats, read_error)) {
    std::cerr << "firefly symbolic global reduction: " << read_error << "\n";
    return res;
  }
  const unsigned neqs = system.size();
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
      secred::symbolic_global_learn(system, target_ids, all_integrals, symbols,
                                    fflow::BIG_UINT_PRIMES[0], masters_final_path,
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
  res.phase_split = true;
  res.learn_seconds = learn.learn_seconds;

  // 3) Reconstruct.  Zero outputs is a legal outcome (every target is a master, so
  // there is nothing to reconstruct) and FireFly's Reconstructor is fatal on n == 0,
  // so this must return before one is constructed.
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

  // `dep_targets` is copied, not moved: the output assembly below needs it to map a
  // target id to its row of the reconstructed matrix.
  secred::SymbolicFFEvaluator ev(std::move(system), std::move(learn.needed_eq_idx),
                                 std::move(learn.bwd_chain), learn.dep_targets,
                                 std::move(learn.master_index), nnindeps,
                                 std::move(symbol_ex));
  const unsigned nparsout = ev.n_outputs();
  std::cout << "\n  " << secred::kFireFlyTag << " reconstruct (rational-function, "
            << nparsout << " outputs = " << learn.dep_targets.size() << " targets x "
            << nnindeps << " masters)...\n";
  if (stats.fallbacks)
    std::cout << "  " << secred::kFireFlyTag << " " << stats.fallbacks
              << " coefficient(s) needed the GiNaC parser\n";

  secred::SecRedFireFlyBlackBox box(ev, nthreads);
  const auto t2 = std::chrono::steady_clock::now();
  {
    // SILENT: FireFly's logo and INFO lines would break SecRed's report format, and the
    // driver reports the reconstruction's own counters itself below.
    firefly::Reconstructor<secred::SecRedFireFlyBlackBox> rec(
        static_cast<unsigned>(nparams), nthreads, box,
        firefly::Reconstructor<secred::SecRedFireFlyBlackBox>::SILENT);
    rec.reconstruct();
    if (box.hard_failure()) {
      if (box.fail_integral() != 0 || box.fail_target() != 0)
        std::cerr << "firefly symbolic global reduction: target " << box.fail_target()
                  << " reduces to integral " << box.fail_integral()
                  << ", which is not a master (equation selection is stale); aborting "
                     "instead of emitting a truncated reduction\n";
      else
        std::cerr << "firefly symbolic global reduction: a probe was refused (the run "
                     "is unreliable); aborting instead of emitting a truncated "
                     "reduction\n";
      return res; // res.ok stays false
    }
    const std::vector<firefly::RationalFunction> &out = rec.get_result();
    // Result assembly mirrors the FiniteFlow symbolic path: each target is preceded by
    // its (target, "-1") marker, and each master with a non-empty numerator appends
    // (master, coefficient text) -- FireFly's own spelling.
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
        continue; // the target is itself a master
      const unsigned i = dit->second;
      for (unsigned j = 0; j < nnindeps; ++j) {
        const firefly::RationalFunction &r =
            out[static_cast<std::size_t>(i) * nnindeps + j];
        if (r.numerator.coefs.empty())
          continue; // zero coefficient
        res.reductions.emplace_back(all_integrals[learn.masters[j]],
                                    r.to_string(varnames));
      }
    }
  }
  const auto t3 = std::chrono::steady_clock::now();
  std::cout << "  " << secred::kFireFlyTag << " reconstruction done in "
            << secred::format_duration(std::chrono::duration<double>(t3 - t2).count())
            << " (" << box.probes() << " probes, " << box.primes() << " primes, "
            << box.bad_points() << " points with a vanishing denominator)\n";
  res.reconstruction_seconds = std::chrono::duration<double>(t3 - t2).count();
  res.ok = true;
  return res;
}
