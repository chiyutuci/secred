// In-process global reduction: a pure-numeric and a symbolic FiniteFlow path.
#include "globalreduce.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <gmp.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fflow/alg_mp_reconstruction.hh>
#include <fflow/common.hh>
#include <fflow/gcd.hh>
#include <fflow/graph.hh>
#include <fflow/mp_common.hh>
#include <fflow/mp_gcd.hh>
#include <fflow/numeric_solver.hh>
#include <fflow/primes.hh>

#include <cln/cln.h>
#include <ginac/ginac.h>

#include <memory>
#include <random>

#include "global_select.h"
#include "ffred_solver.h"
#include "masters_file.h"
#include "progress_bar.h"
#include "spill_reader.h"
#include "sym_coeff.h"
#include "symbolic_ffred.h"

// ---------------------------------------------------------------------------
// The numeric reconstruction's hoisted plan (see globalreduce.h for why)
// ---------------------------------------------------------------------------

namespace {

// The distinct-coefficient map's key.  The reader writes coefficients in lowest terms
// with den > 0, so the pair is the identity of the rational: no normalisation before
// hashing, and two equations naming the same number share a slot.
struct Rat64Hash {
  std::size_t operator()(const secred::Rat64 &r) const {
    std::uint64_t h = static_cast<std::uint64_t>(r.num);
    h ^= static_cast<std::uint64_t>(r.den) + 0x9e3779b97f4a7c15ull + (h << 6) +
         (h >> 2);
    return static_cast<std::size_t>(h);
  }
};

struct Rat64Eq {
  bool operator()(const secred::Rat64 &a, const secred::Rat64 &b) const {
    return a.num == b.num && a.den == b.den;
  }
};

} // namespace

void NumericEvalPlan::build(const std::vector<secred::Eq> &system,
                            const std::vector<std::uint32_t> &eq_idx) {
  terms.clear();
  eq_begin.clear();
  coeffs.clear();
  sys_idx = eq_idx;
  std::size_t n_terms = 0;
  for (std::uint32_t i : eq_idx)
    n_terms += system[i].size();
  eq_begin.reserve(eq_idx.size() + 1);
  eq_begin.push_back(0);
  terms.reserve(n_terms);
  std::unordered_map<secred::Rat64, std::uint32_t, Rat64Hash, Rat64Eq> slot_of;
  std::uint32_t max_id = 0;
  for (std::uint32_t i : eq_idx) {
    for (const auto &t : system[i]) {
      // A coefficient that is zero as a rational is zero modulo every prime, so
      // reduce_mod drops it at every one of them.  Keeping such a term would leave a
      // slot that is always 0, and a zero in `values` is exactly what
      // forward_eliminate_planned reads as "these are not the plan's rows" -- the plan
      // check would fail at every prime and the rebuilt order would silently replace
      // the hoisted one, costing the very thing this table exists for.
      if (t.second.num == 0)
        continue;
      auto it = slot_of.find(t.second);
      std::uint32_t slot;
      if (it == slot_of.end()) {
        slot = static_cast<std::uint32_t>(coeffs.size());
        coeffs.push_back(t.second);
        slot_of.emplace(t.second, slot);
      } else {
        slot = it->second;
      }
      terms.emplace_back(t.first, slot);
      if (t.first > max_id)
        max_id = t.first;
    }
    eq_begin.push_back(static_cast<std::uint32_t>(terms.size()));
  }
  store_ids = static_cast<std::size_t>(max_id) + 1;
  // sys_idx (the raw equation index of each row) is the order's last tie-break, exactly
  // as the reference derivation passes it, so the plan is the reference's order by
  // construction rather than by luck.
  secred::build_fwd_plan(terms, eq_begin, sys_idx, plan);
}

secred::SymRowTable NumericEvalPlan::rows() const {
  secred::SymRowTable t;
  t.terms = terms.data();
  t.eq_begin = eq_begin.data();
  t.n_rows = sys_idx.size();
  return t;
}

void NumericEvalPlan::fill_values(std::uint64_t p,
                                  std::vector<std::uint64_t> &values) const {
  values.resize(coeffs.size());
  for (std::size_t s = 0; s < coeffs.size(); ++s)
    values[s] = secred::rat_mod_u64(coeffs[s].num, coeffs[s].den, p);
}

bool solve_prime_numeric(const NumericEvalPlan &plan, std::uint64_t p,
                         const std::vector<std::uint32_t> &bwd_chain,
                         const std::vector<unsigned> &dep_targets,
                         const std::vector<unsigned> &masters,
                         const std::unordered_map<unsigned, unsigned> &master_index,
                         secred::FwdStore &store, std::vector<std::uint64_t> &values,
                         std::vector<std::uint64_t> &xout,
                         std::uint32_t &fail_target, std::uint32_t &fail_integral) {
  const unsigned nnindeps = masters.size();
  // The rows the selection kept, eliminated at this prime: the plan was built from their
  // structure, and a coefficient that vanishes here is exactly the case
  // forward_eliminate_planned detects and reorders for, so this reproduces
  // reduce_mod + forward_eliminate row for row.
  plan.fill_values(p, values);
  store.reset(plan.store_ids);
  secred::forward_eliminate_planned(plan.rows(), values.data(), values.size(),
                                    plan.plan, p, store);
  secred::backward_eliminate_planned(store, p, bwd_chain);

  xout.assign(dep_targets.size() * static_cast<std::size_t>(nnindeps), 0);
  for (unsigned i = 0; i < dep_targets.size(); ++i) {
    const std::uint32_t target = dep_targets[i];
    if (!store.has_row(target))
      continue; // the target is itself a master: no reduction at this prime
    const std::uint32_t *red_ids = store.row_ids(target);
    const std::uint64_t *red_vals = store.row_vals(target);
    const std::uint32_t red_size = store.row_size(target);
    for (std::uint32_t k = 1; k < red_size; ++k) {
      auto mit = master_index.find(red_ids[k]);
      if (mit == master_index.end()) {
        // Hard check: a non-master on a right-hand side means the selection or the
        // bootstrap structure disagrees with the learn prime (cancelling columns
        // dropping an equation, pivot drift), so xout would miss a term and the
        // reconstructed coefficient would be wrong -- never emit a truncated reduction.
        fail_target = target;
        fail_integral = red_ids[k];
        return false;
      }
      xout[static_cast<std::size_t>(i) * nnindeps + mit->second] = red_vals[k];
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Pure-numeric FiniteFlow global reduction (no symbols, no Kira/FireFly, no "dum" variable)
// ---------------------------------------------------------------------------

namespace {
namespace ff = fflow;

// The shared evaluation half lives in symbolic_ffred.h, which owns `parallel_for`, the
// shared subset solve and the masters_final writer; this file feeds it the parsed
// system and drives the reconstruction.
using secred::parallel_for;
using secred::solve_res_eqs;
using secred::write_masters_final;

// Rational reconstruction with progress logging.  The elimination's input is the plan,
// so a prime costs one `values` fill and one planned elimination; the primes themselves
// are independent of each other, so a pool of workers runs them concurrently while the
// main thread consumes their results in order -- CRT and the convergence test are
// sequential in the prime index, and stay exactly as algorithm_reconstruct_numeric has
// them.  True on convergence, false on failure.
bool reconstruct_ffred(const NumericEvalPlan &plan,
                       const std::vector<uint32_t> &bwd_chain,
                       const std::vector<unsigned> &dep_targets,
                       const std::vector<unsigned> &masters,
                       const std::unordered_map<unsigned, unsigned> &master_index,
                       unsigned nthreads, unsigned max_primes,
                       ff::MPRational res[], double *elim_seconds) {
  const unsigned nndeps = dep_targets.size();
  const unsigned nnindeps = masters.size();
  const unsigned nparsout = nndeps * nnindeps;
  if (nthreads == 0)
    nthreads = 1;

  // One slot per prime the run could reach.  A worker fills its slot and wakes the
  // consumer; the consumer reads slot ip only after `ready` is set under the mutex,
  // which is what orders the worker's writes before the read.
  const std::size_t nprime_max = static_cast<std::size_t>(max_primes) + 1;
  struct PrimeSlot {
    std::vector<ff::UInt> xout;
    double seconds = 0.0;
    bool ready = false;
    bool ok = true;
    std::uint32_t fail_target = 0, fail_integral = 0;
  };
  std::vector<PrimeSlot> slots(nprime_max);

  std::atomic<std::size_t> next{0};
  std::atomic<bool> stop{false};
  std::mutex m;
  std::condition_variable cv;

  // Each worker owns one store and one slot array: a prime allocates nothing after the
  // first, and a worker reuses its memory for every prime it takes.  The plan's table is
  // shared and read-only.  More workers than concurrent primes is pure memory (one store
  // holds the whole selected solution), so the pool is capped by the prime count as well.
  const unsigned nworkers =
      static_cast<unsigned>(std::min<std::size_t>(nthreads, nprime_max));
  std::vector<std::thread> workers;
  workers.reserve(nworkers);
  for (unsigned w = 0; w < nworkers; ++w) {
    workers.emplace_back([&]() {
      secred::FwdStore store;
      std::vector<std::uint64_t> values;
      for (;;) {
        const std::size_t ip = next.fetch_add(1, std::memory_order_relaxed);
        if (ip >= nprime_max || stop.load(std::memory_order_relaxed))
          break;
        const std::uint64_t p = ff::BIG_UINT_PRIMES[ip % ff::BIG_UINT_PRIMES_SIZE];
        PrimeSlot &slot = slots[ip];
        const auto t0 = std::chrono::steady_clock::now();
        slot.ok = solve_prime_numeric(plan, p, bwd_chain, dep_targets, masters,
                                      master_index, store, values, slot.xout,
                                      slot.fail_target, slot.fail_integral);
        slot.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                    t0)
                           .count();
        {
          std::lock_guard<std::mutex> lock(m);
          slot.ready = true;
        }
        cv.notify_all();
        if (!slot.ok)
          break; // the consumer stops on a failure; do not start another prime
      }
    });
  }

  std::unique_ptr<ff::MPInt[]> zout(new ff::MPInt[nparsout]);
  ff::MPInt totmod, c1, c2;
  bool ok = false;

  for (std::size_t ip = 0; ip < nprime_max; ++ip) {
    {
      std::unique_lock<std::mutex> lock(m);
      cv.wait(lock, [&] { return slots[ip].ready; });
    }
    // The failure is reported in prime order, so a run that fails names the same prime
    // and the same target it would have named serially: convergence at an earlier prime
    // stops the loop before a later failure is ever looked at.
    if (!slots[ip].ok) {
      std::cerr << "finiteflow global reduction: prime "
                << ff::BIG_UINT_PRIMES[ip % ff::BIG_UINT_PRIMES_SIZE] << ": target "
                << slots[ip].fail_target << " reduces to integral "
                << slots[ip].fail_integral
                << ", which is not a master (equation selection is stale for this "
                   "prime); aborting instead of emitting a truncated reduction\n";
      break;
    }

    ff::Mod mod(ff::BIG_UINT_PRIMES[ip % ff::BIG_UINT_PRIMES_SIZE]);
    const ff::UInt *xout = slots[ip].xout.data();
    if (elim_seconds)
      *elim_seconds += slots[ip].seconds;

    // CRT (as in algorithm_reconstruct_numeric).
    const auto t_rest0 = std::chrono::steady_clock::now();
    bool converged = false;
    bool failed = false;
    if (ip == 0) {
      totmod = mod.n();
      parallel_for(nthreads, nparsout, [&](unsigned j) {
        if (xout[j]) {
          zout[j] = xout[j];
          res[j] = ff::rat_rec(xout[j], mod.n());
        } else {
          zout[j] = ff::UInt(0);
          res[j] = ff::Int(0);
        }
      });
    } else {
      ff::MPInt this_mod(mod.n());
      std::atomic<bool> failed_at{false};
      std::atomic<bool> consistent{true};
      parallel_for(nthreads, nparsout, [&](unsigned j) {
        if (!consistent.load(std::memory_order_relaxed))
          return;
        ff::MPInt xmp;
        if (!ff::rat_mod(res[j], this_mod, xmp)) {
          failed_at.store(true, std::memory_order_relaxed);
          consistent.store(false, std::memory_order_relaxed);
          return;
        }
        if (xout[j] != xmp.to_uint())
          consistent.store(false, std::memory_order_relaxed);
      });
      failed = failed_at.load();
      if (!failed && consistent.load())
        converged = true;
      else if (!failed && ip == max_primes)
        break;
      else if (!failed) {
        ff::chinese_remainder_coeffs(totmod, mod.n(), c1, c2, totmod);
        parallel_for(nthreads, nparsout, [&](unsigned j) {
          ff::chinese_remainder_from_coeffs(zout[j], xout[j], c1, c2, totmod,
                                            zout[j]);
          ff::rat_rec(zout[j], totmod, res[j]);
        });
      }
    }
    const auto t_rest1 = std::chrono::steady_clock::now();
    const double rest_s =
        std::chrono::duration<double>(t_rest1 - t_rest0).count();

    std::cout << "  " << secred::kFlowTag << " reconstruct: prime " << ip + 1 << "/"
              << max_primes << " evaluated (elimination "
              << secred::format_duration(slots[ip].seconds) << ", CRT/rat_rec "
              << secred::format_duration(rest_s) << ")\n"
              << std::flush;

    if (failed)
      break;
    if (converged) {
      std::cout << "  " << secred::kFlowTag << " reconstruct: converged after "
                << ip + 1 << " prime(s)\n";
      ok = true;
      break;
    }
    // The slot has been consumed; give its output back before the next prime's arrives.
    std::vector<ff::UInt>().swap(slots[ip].xout);
  }

  stop.store(true, std::memory_order_relaxed);
  cv.notify_all();
  for (auto &t : workers)
    t.join();
  return ok;
}

} // namespace

GlobalReductionResult reduce_global_finiteflow(
    const std::string &raw_equations_path,
    const std::vector<RawIntegral> &all_integrals,
    const std::unordered_map<RawIntegral, unsigned> &integral_id,
    const std::vector<RawIntegral> &targets,
    unsigned narrowing_success, unsigned narrowing_fail,
    unsigned raw_equations_total, std::size_t global_equations_count,
    const std::string &masters_final_path, unsigned nthreads) {
  GlobalReductionResult res;
  namespace ff = fflow;

  const unsigned nvars = all_integrals.size();
  const auto t0 = std::chrono::steady_clock::now();
  std::cout << "\n  " << secred::kFlowTag << " reading spill file, building system...\n";

  // 1) Read the spill file into vector<Eq> (descending (id, num/den)).
  std::vector<secred::Eq> system;
  secred::SpillReadStats stats;
  std::string read_error;
  if (!secred::read_spill_numeric(raw_equations_path, integral_id, system, stats,
                                  read_error)) {
    std::cerr << "finiteflow global reduction: " << read_error << "\n";
    res.ok = false;
    return res;
  }

  const unsigned neqs = system.size();
  if (neqs == 0) {
    std::cerr << "finiteflow global reduction: empty system\n";
    return res;
  }

  const auto t1 = std::chrono::steady_clock::now();
  std::cout << "  " << secred::kFlowTag << " system built: " << neqs << " equations x "
            << nvars << " variables ("
            << secred::format_duration(std::chrono::duration<double>(t1 - t0).count())
            << ")";
  if (stats.fallbacks)
    std::cout << " [" << stats.fallbacks << " coefficient(s) by the fallback parser]";
  if (stats.skipped_unknown)
    std::cout << " [" << stats.skipped_unknown << " term(s): unknown integral]";
  std::cout << "\n";

  // 2) Target ids.
  std::vector<unsigned> target_ids;
  target_ids.reserve(targets.size());
  for (const auto &t : targets) {
    auto it = integral_id.find(t);
    if (it != integral_id.end())
      target_ids.push_back(it->second);
  }
  if (target_ids.empty()) {
    std::cerr << "finiteflow global reduction: no targets\n";
    return res;
  }

  // 3) Learn once at the learn prime: forward elimination -> masters + structure +
  // dependencies (for mark-and-sweep).
  const uint64_t learn_prime = ff::BIG_UINT_PRIMES[0];
  std::unordered_map<uint32_t, secred::ResEq> learn_solmap;
  std::unordered_map<uint32_t, uint32_t> eq_index; // pivot -> raw equation index
  std::unordered_map<uint32_t, std::vector<uint32_t>> fwd_deps, bwd_deps;
  double learn_s = 0.0;
  {
    auto t_learn0 = std::chrono::steady_clock::now();
    std::vector<secred::ResEq> res_eqs;
    std::vector<uint32_t> sys_idx;
    res_eqs.reserve(neqs);
    sys_idx.reserve(neqs);
    for (std::size_t i = 0; i < system.size(); ++i) {
      secred::ResEq re;
      secred::reduce_mod(system[i], learn_prime, re);
      if (!re.empty()) {
        res_eqs.push_back(std::move(re));
        sys_idx.push_back(static_cast<uint32_t>(i));
      }
    }
    secred::forward_eliminate(res_eqs, sys_idx, learn_prime, learn_solmap,
                              &eq_index, &fwd_deps);
    secred::backward_eliminate(learn_solmap, learn_prime, &bwd_deps);
    auto t_learn1 = std::chrono::steady_clock::now();
    learn_s = std::chrono::duration<double>(t_learn1 - t_learn0).count();
  }

  // Masters = union of the integrals on the right-hand sides of the target reductions
  // (the reduction is fully recursive, so a right-hand side is exactly masters).
  // Not "ids in 0..nvars-1 never solved": that would count isolated variables as masters.
  std::set<unsigned> needed_masters;
  for (unsigned id : target_ids) {
    auto it = learn_solmap.find(id);
    if (it == learn_solmap.end())
      continue; // target is itself a master (no reduction)
    const secred::ResEq &red = it->second;
    for (size_t k = 1; k < red.size(); ++k)
      needed_masters.insert(red[k].first);
  }
  // Masters in descending id, matching the order of the original FiniteFlow path's
  // needed_indepvars() (ascending column = descending id).
  std::vector<unsigned> masters(needed_masters.rbegin(), needed_masters.rend());
  const unsigned nnindeps = masters.size();

  // Reducible targets (present in learn_solmap), in the caller's target order.
  std::vector<unsigned> dep_targets;
  for (unsigned id : target_ids)
    if (learn_solmap.find(id) != learn_solmap.end())
      dep_targets.push_back(id);
  const unsigned nndeps = dep_targets.size();

  std::unordered_map<unsigned, unsigned> master_index; // id -> index j
  for (unsigned j = 0; j < nnindeps; ++j)
    master_index[masters[j]] = j;

  // Mark-and-sweep from the targets along the forward / backward dependency closure,
  // yielding the raw equations reconstruction needs.  (ffred does the same with an
  // insertions database plus m_content; here the recorded dependencies are used directly.)
  auto sweep = [&](bool with_bwd) {
    std::unordered_set<uint32_t> s(dep_targets.begin(), dep_targets.end());
    std::vector<uint32_t> stack(dep_targets.begin(), dep_targets.end());
    while (!stack.empty()) {
      uint32_t w = stack.back();
      stack.pop_back();
      auto fit = fwd_deps.find(w);
      if (fit != fwd_deps.end())
        for (uint32_t u : fit->second)
          if (s.insert(u).second)
            stack.push_back(u);
      if (with_bwd) {
        auto bit = bwd_deps.find(w);
        if (bit != bwd_deps.end())
          for (uint32_t u : bit->second)
            if (s.insert(u).second)
              stack.push_back(u);
      }
    }
    return s;
  };
  std::unordered_set<uint32_t> needed = sweep(true); // fallback closure: forward and backward
  const std::unordered_set<uint32_t> needed_fwd = sweep(false);

  // Small-system diagnostic: print the whole reduction to compare against Kira.
  if (neqs <= 1000) {
    std::cerr << "\n===== RAW equations (" << neqs << ") =====\n";
    for (std::size_t i = 0; i < system.size(); ++i) {
      std::cerr << "eq[" << i << "] front=" << system[i].front().first
                << " :";
      for (const auto &t : system[i])
        std::cerr << " " << t.first << "(" << t.second.num << "/" << t.second.den
                  << ")";
      std::cerr << "\n";
    }
    std::cerr << "\n===== forward reductions =====\n";
    for (const auto &kv : eq_index) {
      uint32_t w = kv.first, i = kv.second;
      std::cerr << "pivot " << w << " (eq " << i << ", raw front "
                << system[i].front().first << "):";
      if (fwd_deps.count(w))
        for (uint32_t u : fwd_deps.at(w))
          std::cerr << " dep" << u;
      std::cerr << "\n";
    }
    std::cerr << "\n===== backward deps =====\n";
    for (const auto &kv : bwd_deps) {
      std::cerr << "pivot " << kv.first << " bwd:";
      for (uint32_t u : kv.second)
        std::cerr << " " << u;
      std::cerr << "\n";
    }
    std::cerr << "\n===== needed (fwd-only " << needed_fwd.size()
              << ", bwd-only " << (needed.size() - needed_fwd.size())
              << ") =====\n";
    std::vector<uint32_t> bwd_only;
    for (uint32_t w : needed)
      if (!needed_fwd.count(w))
        bwd_only.push_back(w);
    std::sort(bwd_only.begin(), bwd_only.end());
    for (uint32_t w : bwd_only) {
      std::cerr << "bwd-only pivot " << w << " :";
      auto it = learn_solmap.find(w);
      if (it != learn_solmap.end())
        for (std::size_t k = 1; k < it->second.size(); ++k)
          std::cerr << " " << it->second[k].first << ":" << it->second[k].second;
      std::cerr << "  [ref-by:";
      for (const auto &kv : bwd_deps)
        for (uint32_t u : kv.second)
          if (u == w)
            std::cerr << " " << kv.first;
      std::cerr << "]\n";
    }
  }
  // Equation selection: forward closure plus numeric sufficiency repair, falling back to
  // the forward-and-backward closure when the repair does not converge.
  std::vector<uint32_t> needed_eq_idx;
  std::vector<uint32_t> bwd_chain; // lazy chain: reconstruction rebuilds only this, not every reduction
  bool use_forward = false;
  {
    auto solve_at_learn = [&](const std::vector<uint32_t> &idx,
                              secred::SubSolve &sub) {
      secred::solve_subset(system, idx, learn_prime, sub, false, true);
    };
    GlobalSelection gsel =
        select_global_forward(eq_index, fwd_deps, learn_solmap, dep_targets,
                              master_index, solve_at_learn);
    if (gsel.fallback) {
      std::cerr << "finiteflow global reduction: forward selection did not "
                   "converge, falling back to the legacy closure\n";
    } else {
      use_forward = true;
      needed_eq_idx = std::move(gsel.needed_eq_idx);
      bwd_chain = std::move(gsel.bwd_chain);
    }
  }
  if (!use_forward) {
    needed_eq_idx.reserve(needed.size());
    for (uint32_t w : needed) {
      auto it = eq_index.find(w);
      if (it != eq_index.end())
        needed_eq_idx.push_back(it->second);
    }
    std::sort(needed_eq_idx.begin(), needed_eq_idx.end());
    // Targets plus the derived integrals in the backward dependency closure, matching
    // Kira's backsubs (~0.1 s).
    std::unordered_set<uint32_t> chain_set(dep_targets.begin(), dep_targets.end());
    std::vector<uint32_t> stack(dep_targets.begin(), dep_targets.end());
    while (!stack.empty()) {
      uint32_t w = stack.back();
      stack.pop_back();
      auto bit = bwd_deps.find(w);
      if (bit != bwd_deps.end())
        for (uint32_t u : bit->second)
          if (chain_set.insert(u).second)
            stack.push_back(u);
    }
    bwd_chain.assign(chain_set.begin(), chain_set.end());
    std::sort(bwd_chain.begin(), bwd_chain.end());
  }

  res.selected_equations = static_cast<std::uint64_t>(learn_solmap.size());
  res.raw_equations = neqs;

  {
    secred::LearnSummary ls;
    ls.independent_equations = res.selected_equations;
    ls.masters = nnindeps;
    ls.seconds = learn_s;
    std::cout << "\n" << secred::format_learn_summary(ls);
  }

  write_masters_final(masters_final_path, all_integrals, masters);

  // Consolidated Reduction Statistics.  "Final equations" is what the
  // mark-and-sweep keeps: the equations each prime's elimination runs on.
  {
    secred::ReductionStats rs;
    rs.narrowing_success = narrowing_success;
    rs.narrowing_fail = narrowing_fail;
    rs.raw_equations = raw_equations_total;
    rs.global_equations = global_equations_count;
    rs.final_equations = needed_eq_idx.size();
    rs.masters = nnindeps;
    std::cout << "\n" << secred::format_reduction_statistics(rs);
  }

  // 4) Reconstruct: per-prime forward elimination plus CRT.
  const unsigned max_primes = 200;
  std::unique_ptr<ff::MPRational[]> resmp(
      new ff::MPRational[(std::size_t)nndeps * nnindeps]);
  double elim_seconds = 0.0;
  std::cout << "\n  " << secred::kFlowTag << " reconstruct (rational reconstruction, up to "
            << max_primes << " primes, " << nthreads << " threads)...\n";
  const auto t2 = std::chrono::steady_clock::now();
  // The hoisted table: the rows the selection kept, their coefficients, and the order
  // the elimination will follow.  Built once; every prime reuses it.
  const auto tplan0 = std::chrono::steady_clock::now();
  NumericEvalPlan plan;
  plan.build(system, needed_eq_idx);
  const auto tplan1 = std::chrono::steady_clock::now();
  std::cout << "  " << secred::kFlowTag << " reconstruction plan: " << plan.n_rows()
            << " rows, " << plan.terms.size() << " terms, " << plan.n_slots()
            << " distinct coefficient(s) ("
            << secred::format_duration(
                   std::chrono::duration<double>(tplan1 - tplan0).count())
            << ")\n"
            << std::flush;
  bool ok = reconstruct_ffred(plan, bwd_chain, dep_targets, masters,
                              master_index, nthreads, max_primes, resmp.get(),
                              &elim_seconds);
  const auto t3 = std::chrono::steady_clock::now();
  if (!ok) {
    std::cerr << "finiteflow global reduction: reconstruct failed\n";
    return res;
  }
  std::cout << "  " << secred::kFlowTag << " reconstruct done ("
            << secred::format_duration(std::chrono::duration<double>(t3 - t2).count())
            << ")\n";

  // 5) Assemble the output (masters plus each target's reduction).
  for (unsigned j = 0; j < nnindeps; ++j)
    res.masters.push_back(all_integrals[masters[j]]);

  std::unordered_map<unsigned, unsigned> dep_index; // id -> dep index
  for (unsigned i = 0; i < nndeps; ++i)
    dep_index[dep_targets[i]] = i;

  for (const auto &t : targets) {
    res.reductions.emplace_back(t, std::string("-1"));
    auto it = integral_id.find(t);
    if (it == integral_id.end())
      continue;
    auto dit = dep_index.find(it->second);
    if (dit == dep_index.end())
      continue; // the target is a master -> nothing to reduce
    unsigned i = dit->second;
    for (unsigned j = 0; j < nnindeps; ++j) {
      ff::MPRational c = resmp[(std::size_t)i * nnindeps + j];
      if (c.sign() == 0)
        continue;
      std::ostringstream oss;
      oss << c;
      res.reductions.emplace_back(all_integrals[masters[j]], oss.str());
    }
  }

  // The phase split is reported by the end-of-run Timing Statistics block, which
  // family.cpp prints once the whole run — including the file writes — is done.
  res.phase_split = true;
  res.learn_seconds = learn_s;
  res.reconstruction_seconds =
      std::chrono::duration<double>(t3 - t2).count();

  res.ok = true;
  return res;
}

// ---------------------------------------------------------------------------
// Symbolic FiniteFlow global reduction (GiNaC polynomial coefficients, ffred elimination,
// FiniteFlow multivariate rational-function reconstruction).
// ---------------------------------------------------------------------------

namespace {

// The precompiled coefficient machinery lives in sym_coeff.h so the tests can drive
// it directly; this file only feeds it the parsed system and evaluates equations.
using secred::eval_sym_coeff_mod;
using secred::eval_sym_eqs_mod;
using secred::fill_powers;
using secred::make_powers;
using secred::make_sym_coeff;
using secred::sym_max_exps;
using secred::SymEq;

// The FiniteFlow adapter: the shared evaluator behind fflow::Algorithm.  FiniteFlow
// hands the `AlgorithmData*` its caller was given straight back to every evaluate, so
// the per-thread scratch stays an fflow::AlgorithmData around the evaluator's scratch.
class SymbolicFFRedAlgorithm : public fflow::Algorithm {
public:
  SymbolicFFRedAlgorithm(secred::SymCoeffPool pool,
                         std::vector<uint32_t> eq_idx,
                         std::vector<uint32_t> bwd_chain,
                         std::vector<unsigned> dep_targets,
                         std::unordered_map<unsigned, unsigned> master_index,
                         unsigned nnindeps, secred::ReductionItems items,
                         std::vector<GiNaC::ex> symbols)
      : ev_(std::move(pool), std::move(eq_idx), std::move(bwd_chain),
            std::move(dep_targets), std::move(master_index), nnindeps,
            std::move(items), std::move(symbols)) {
    nparsin.push_back(ev_.n_params());
    nparsout = ev_.n_outputs();
  }

  // Per-thread scratch for one sample point.  The buffers a sample point needs are
  // owned per thread instead of by the algorithm: that is what lets the sample points
  // be evaluated concurrently.  The coefficient tables are *not* here — they belong to
  // the system, which is reduced once per prime before any evaluation starts and only
  // read from then on.
  struct FFScratch : public fflow::AlgorithmData {
    secred::SymbolicFFEvaluator::Scratch scratch;
  };

  void reduce_for_prime(uint64_t p, unsigned nthreads) { ev_.reduce_for_prime(p, nthreads); }

  std::size_t n_equations() const { return ev_.n_equations(); }
  std::size_t n_coefficient_terms() const { return ev_.n_coefficient_terms(); }
  std::size_t n_distinct_coeffs() const { return ev_.n_distinct_coeffs(); }

  fflow::Ret evaluate(fflow::Context *, fflow::AlgInput xin[], fflow::Mod mod,
                      fflow::AlgorithmData *data,
                      fflow::UInt xout[]) const override {
    if (data)
      return evaluate_with(static_cast<FFScratch *>(data), xin, mod, xout);
    // The degree-estimation phases call FiniteFlow without an AlgorithmData of their
    // own; they run on the algorithm's scratch, serialised so that a caller mixing
    // both styles still sees one sample point at a time.
    std::lock_guard<std::mutex> lock(fallback_mutex_);
    return evaluate_with(&fallback_scratch_, xin, mod, xout);
  }

private:
  fflow::Ret evaluate_with(FFScratch *scratch, fflow::AlgInput xin[], fflow::Mod mod,
                           fflow::UInt xout[]) const {
    static_assert(sizeof(fflow::UInt) == sizeof(std::uint64_t));
    const uint64_t p = mod.n();
    const std::uint64_t *x = xin[0];
    const secred::SymbolicFFEvaluator::PointResult r =
        ev_.evaluate_point(p, x, scratch->scratch, xout);
    if (r == secred::SymbolicFFEvaluator::PointResult::Ok)
      return fflow::SUCCESS;
    if (r == secred::SymbolicFFEvaluator::PointResult::NotReduced) {
      std::cerr << "finiteflow symbolic global reduction: coefficients were not "
                   "reduced for prime "
                << p << "; aborting instead of evaluating the wrong modulus\n";
    } else if (r == secred::SymbolicFFEvaluator::PointResult::MissingItem) {
      // The right-hand side carries a master the learn point did not, so the selection
      // is stale and the item list has no slot for that term.
      std::cerr << "finiteflow symbolic global reduction: target "
                << scratch->scratch.fail_target << " reduces to integral "
                << scratch->scratch.fail_integral
                << ", a master the learn point did not carry (equation selection is "
                   "stale); aborting instead of emitting a truncated reduction\n";
    } else {
      // Hard check (as in the numeric path): a non-master on a right-hand side means
      // the selection is stale at this sample point and prime, so the output would
      // miss a term.  Return FAILED to abort the reconstruction rather than emit a
      // truncated reduction.
      std::cerr << "finiteflow symbolic global reduction: target "
                << scratch->scratch.fail_target << " reduces to integral "
                << scratch->scratch.fail_integral
                << ", which is not a master (equation selection is stale); "
                   "aborting instead of emitting a truncated reduction\n";
    }
    return fflow::FAILED;
  }

  secred::SymbolicFFEvaluator ev_;
  // Scratch for the FiniteFlow calls that pass no AlgorithmData.
  mutable FFScratch fallback_scratch_;
  mutable std::mutex fallback_mutex_;
};

} // namespace

GlobalReductionResult reduce_global_finiteflow_symbolic(
    const std::string &raw_equations_path,
    const std::vector<RawIntegral> &all_integrals,
    const std::unordered_map<RawIntegral, unsigned> &integral_id,
    const std::vector<RawIntegral> &targets,
    const std::vector<GiNaC::possymbol> &symbols,
    unsigned narrowing_success, unsigned narrowing_fail,
    unsigned raw_equations_total, std::size_t global_equations_count,
    const std::string &masters_final_path, unsigned nthreads) {
  GlobalReductionResult res;
  namespace ff = fflow;

  if (nthreads == 0)
    nthreads = 1;

  const unsigned nvars = all_integrals.size();
  const unsigned nparams = symbols.size();
  const auto t0 = std::chrono::steady_clock::now();
  std::cout << "\n  " << secred::kFlowTag << " reading spill file, building symbolic "
               "system...\n";

  // 1) Read the spill file into an interned pool: descending (integral id, coefficient
  // slot) rows over the distinct coefficients, never a system of per-term objects.
  secred::SymCoeffPool pool;
  secred::SpillReadStats stats;
  std::string read_error;
  if (!secred::read_spill_interned(raw_equations_path, integral_id, symbols, pool, stats,
                                   read_error)) {
    std::cerr << "finiteflow symbolic global reduction: " << read_error << "\n";
    res.ok = false;
    return res;
  }
  const unsigned neqs = static_cast<unsigned>(pool.n_equations());
  if (neqs == 0) {
    std::cerr << "finiteflow symbolic global reduction: empty system\n";
    return res;
  }
  const auto t1 = std::chrono::steady_clock::now();
  std::cout << "  " << secred::kFlowTag << " symbolic system built: " << neqs
            << " equations x " << nvars << " variables, " << nparams
            << " parameters ("
            << secred::format_duration(std::chrono::duration<double>(t1 - t0).count())
            << ")";
  if (stats.fallbacks)
    std::cout << " [" << stats.fallbacks << " coefficient(s) by the fallback parser]";
  if (stats.skipped_unknown)
    std::cout << " [" << stats.skipped_unknown << " term(s): unknown integral]";
  std::cout << "\n";

  // 2) Target ids.
  std::vector<unsigned> target_ids;
  target_ids.reserve(targets.size());
  for (const auto &t : targets) {
    auto it = integral_id.find(t);
    if (it != integral_id.end())
      target_ids.push_back(it->second);
  }
  if (target_ids.empty()) {
    std::cerr << "finiteflow symbolic global reduction: no targets\n";
    return res;
  }

  // 3) The learn both symbolic backends share: one fixed sample point, the elimination,
  // the masters, the mark-and-sweep selection and both report blocks (symbolic_ffred.h).
  const uint64_t learn_prime = ff::BIG_UINT_PRIMES[0];
  secred::SymbolicGlobalLearn learn = secred::symbolic_global_learn(
      pool, target_ids, all_integrals, symbols, learn_prime, masters_final_path,
      secred::kFlowTag, narrowing_success, narrowing_fail, raw_equations_total,
      global_equations_count);
  if (!learn.ok) {
    std::cerr << "finiteflow symbolic global reduction: " << learn.error << "\n";
    return res;
  }
  res.selected_equations = learn.selected_equations;
  res.raw_equations = neqs;
  const unsigned nnindeps = learn.nnindeps;
  const unsigned nndeps = learn.dep_targets.size();

  // 4) Build the algorithm and run FiniteFlow's multivariate rational reconstruction.
  std::vector<GiNaC::ex> symbol_ex;
  symbol_ex.reserve(nparams);
  for (const auto &s : symbols)
    symbol_ex.push_back(GiNaC::ex(s));

  // `dep_targets` is copied, not moved: the output assembly below still needs it to map
  // a target id to its row of the reconstructed matrix.
  SymbolicFFRedAlgorithm alg(std::move(pool), std::move(learn.needed_eq_idx),
                             std::move(learn.bwd_chain), learn.dep_targets,
                             std::move(learn.master_index), nnindeps, learn.items,
                             std::move(symbol_ex));

  const unsigned nparsout = alg.nparsout;
  std::cout << "\n  " << secred::kFlowTag << " reconstruct (rational-function): "
            << nparsout << " function(s) will be interpolated (the non-zero entries of a "
            << nndeps << " x " << nnindeps << " target x master block)\n";
  // Kira accounts for the same system as "Parsed N functions of which M are distinct";
  // ours are interned the same way, so the per-point cost is the distinct count.
  std::cout << "  " << secred::kFlowTag << " " << alg.n_coefficient_terms()
            << " function(s) to evaluate per sample point, of which "
            << alg.n_distinct_coeffs() << " are distinct\n";

  ff::ReconstructionOptions opt;
  opt.max_primes = 5; // CRT prime cap for the coefficient reconstruction

  // A default shift (as FiniteFlow's interface set_default_shift does) keeps a multivariate
  // rational function from hitting BAD_SHIFT at the unshifted default sample points.
  std::unique_ptr<ff::UInt[]> shift(new ff::UInt[nparams]);
  {
    ff::Mod smod(ff::BIG_UINT_PRIMES[opt.start_mod]);
    for (unsigned i = 0; i < nparams; ++i)
      shift[i] = ff::sample_uint(ff::OFFSET_4, i, smod);
  }

  std::unique_ptr<unsigned[]> numdeg(new unsigned[nparsout]);
  std::unique_ptr<unsigned[]> dendeg(new unsigned[nparsout]);
  std::unique_ptr<ff::RatFunVarDegrees[]> vdegs(
      new ff::RatFunVarDegrees[nparsout]);
  for (unsigned i = 0; i < nparsout; ++i)
    vdegs[i].resize(nparams);

  // The degree estimation evaluates the system at `opt.start_mod` alone, and the
  // evaluation reads the reduced coefficient tables, so reduce them for that prime
  // first.  This also covers the var-degree pass below, which uses the same prime.
  alg.reduce_for_prime(ff::BIG_UINT_PRIMES[opt.start_mod], nthreads);

  ff::Ret ret = ff::algorithm_get_degrees(alg, nullptr, nullptr, shift.get(),
                                          opt, numdeg.get(), dendeg.get());
  if (ret != ff::SUCCESS) {
    std::cerr << "finiteflow symbolic: degree estimation failed (ret " << ret
              << ")\n";
    return res;
  }
  for (unsigned var = 0; var < nparams; ++var) {
    ret = ff::algorithm_get_var_degrees(alg, nullptr, nullptr, var, shift.get(),
                                        opt, vdegs.get());
    if (ret != ff::SUCCESS) {
      std::cerr << "finiteflow symbolic: var-degree estimation failed (ret "
                << ret << ")\n";
      return res;
    }
  }

  ff::GenerateSamplePoints sfun(nparams);
  ff::algorithm_generate_sample_points(sfun, nparams, nparsout, shift.get(),
                                       opt, vdegs.get(), numdeg.get(),
                                       dendeg.get());
  ff::SamplePointsVector sample_points;
  sfun.append_to_vector(sample_points);
  ff::sort_by_mod(sample_points.data(),
                  sample_points.data() + sample_points.size(), nparams);
  std::cout << "  " << secred::kFlowTag << " " << sample_points.size()
            << " sample points to evaluate\n\n";

  ff::UIntCache cache;
  cache.init(nparams + 1);
  const auto t2 = std::chrono::steady_clock::now();
  {
    // One cache and one scratch per thread; FiniteFlow's `merge_caches` folds the
    // per-thread caches back into `cache` at the end.  The sample points are
    // modulus-sorted, so one prime is a contiguous run: the coefficient tables are
    // reduced for that run's prime (in parallel over the system) and the run is then
    // evaluated in parallel.  Reducing per run rather than for every prime up front
    // keeps one coefficient table in memory instead of one per prime.
    secred::ProgressBar bar(std::string("  ") + secred::kFlowTag +
                                " evaluating samples",
                            sample_points.size(), secred::stdout_is_tty());
    std::cout << bar.update(0) << std::flush;

    std::vector<std::unique_ptr<ff::UIntCache>> part_caches(nthreads);
    std::vector<std::unique_ptr<SymbolicFFRedAlgorithm::FFScratch>> part_scratch(
        nthreads);
    for (unsigned t = 0; t < nthreads; ++t) {
      part_caches[t].reset(new ff::UIntCache);
      part_caches[t]->init(nparams + 1);
      part_scratch[t].reset(new SymbolicFFRedAlgorithm::FFScratch);
    }

    std::atomic<std::size_t> done{0};
    std::mutex done_mutex;
    std::condition_variable done_cv;
    std::vector<std::thread> workers;
    const std::size_t npts = sample_points.size();
    std::size_t run_begin = 0;
    while (run_begin < npts) {
      const uint64_t p = sample_points[run_begin].get()[nparams];
      std::size_t run_end = run_begin + 1;
      while (run_end < npts && sample_points[run_end].get()[nparams] == p)
        ++run_end;

      alg.reduce_for_prime(p, nthreads);

      const std::size_t cnt = run_end - run_begin;
      const unsigned nt =
          static_cast<unsigned>(std::min<std::size_t>(nthreads, cnt));
      workers.clear();
      for (unsigned t = 0; t < nt; ++t) {
        const std::size_t begin = run_begin + cnt * t / nt;
        const std::size_t end = run_begin + cnt * (t + 1) / nt;
        if (begin >= end)
          continue;
        workers.emplace_back([&, t, begin, end]() {
          // A refused point is cached with `xout[0] = FAILED` by FiniteFlow itself, and
          // `SampledRatFun::evaluate` turns that into a failed reconstruction, so a
          // worker's failure needs no reporting here.
          ff::evaluate_and_cache_samples(alg, part_scratch[t].get(), nullptr,
                                         sample_points.data() + begin,
                                         sample_points.data() + end,
                                         *part_caches[t]);
          {
            std::lock_guard<std::mutex> lock(done_mutex);
            done.fetch_add(end - begin, std::memory_order_relaxed);
          }
          done_cv.notify_one();
        });
      }
      // The main thread reports progress while the workers evaluate.  The wait wakes on
      // every completed worker, so joining costs nothing beyond the last one, and the
      // bar throttles its own output.
      {
        std::unique_lock<std::mutex> lock(done_mutex);
        while (done.load(std::memory_order_relaxed) < run_end) {
          done_cv.wait_for(lock, std::chrono::milliseconds(100));
          std::cout << bar.update(done.load(std::memory_order_relaxed)) << std::flush;
        }
      }
      for (auto &w : workers)
        w.join();
      std::cout << bar.update(done.load(std::memory_order_relaxed)) << std::flush;
      run_begin = run_end;
    }

    std::vector<std::unique_ptr<ff::UIntCache>> filled;
    for (unsigned t = 0; t < nthreads; ++t)
      if (part_caches[t] && part_caches[t]->size())
        filled.push_back(std::move(part_caches[t]));
    if (!filled.empty())
      cache.merge_caches(filled.data(), filled.size());

    std::cout << bar.finish() << std::flush;
  }
  const auto t3 = std::chrono::steady_clock::now();
  std::cout << "  " << secred::kFlowTag << " evaluated " << cache.size()
            << " sample points ("
            << secred::format_duration(std::chrono::duration<double>(t3 - t2).count())
            << ")\n";

  std::unique_ptr<ff::MPReconstructedRatFun[]> recf(
      new ff::MPReconstructedRatFun[nparsout]);
  {
    // Each output is reconstructed on its own from the shared, read-only cache, so the
    // loop over the outputs is split across threads.  The first failure stops the
    // others: the run is abandoned either way.
    std::atomic<unsigned> failure{nparsout};
    std::atomic<unsigned> failure_ret{0};
    std::mutex err_mutex;
    parallel_for(nthreads, nparsout, [&](unsigned idx) {
      if (failure.load(std::memory_order_relaxed) != nparsout)
        return;
      const ff::Ret r =
          ff::algorithm_reconstruct(cache, nparams, nparsout, idx, shift.get(),
                                    opt, vdegs.get(), numdeg.get(), dendeg.get(),
                                    recf[idx]);
      if (r == ff::SUCCESS)
        return;
      std::lock_guard<std::mutex> lock(err_mutex);
      if (failure.load() == nparsout) {
        failure.store(idx);
        failure_ret.store(r);
      }
    });
    if (failure.load() != nparsout) {
      std::cerr << "finiteflow symbolic: reconstruction failed for output "
                << failure.load() << " (ret " << failure_ret.load() << ")\n";
      return res;
    }
  }

  // 5) Assemble the output.
  std::vector<std::string> varnames(nparams);
  for (unsigned i = 0; i < nparams; ++i)
    varnames[i] = symbols[i].get_name();

  for (unsigned j = 0; j < nnindeps; ++j)
    res.masters.push_back(all_integrals[learn.masters[j]]);

  std::unordered_map<unsigned, unsigned> dep_index;
  for (unsigned i = 0; i < nndeps; ++i)
    dep_index[learn.dep_targets[i]] = i;

  for (const auto &t : targets) {
    res.reductions.emplace_back(t, std::string("-1"));
    auto it = integral_id.find(t);
    if (it == integral_id.end())
      continue;
    auto dit = dep_index.find(it->second);
    if (dit == dep_index.end())
      continue; // the target is itself a master
    unsigned i = dit->second;
    for (unsigned q = learn.items.begin[i]; q < learn.items.begin[i + 1]; ++q) {
      const ff::MPReconstructedRatFun &r = recf[q];
      if (r.numerator().size() == 0)
        continue; // zero coefficient: not a term of the reduction
      res.reductions.emplace_back(
          all_integrals[learn.masters[learn.items.col[q]]], r.to_str(varnames.data()));
    }
  }

  // As in the numeric path, the split is reported by the end-of-run Timing
  // Statistics block.
  res.phase_split = true;
  res.learn_seconds = learn.learn_seconds;
  res.reconstruction_seconds =
      std::chrono::duration<double>(t3 - t2).count();

  res.ok = true;
  return res;
}
