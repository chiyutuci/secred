
#include "globalreduce.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
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

#include <unistd.h>

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
#include "reconstruct_budget.h"
#include "spill_reader.h"
#include "sym_coeff.h"
#include "symbolic_ffred.h"
#include "uni_linear_reconstruct.h"

namespace {

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

}

std::uint64_t reconstruction_prime(unsigned index) {

  static std::mutex mutex;
  static std::vector<std::uint64_t> table;
  std::lock_guard<std::mutex> lock(mutex);
  if (table.empty())
    table.assign(fflow::BIG_UINT_PRIMES,
                 fflow::BIG_UINT_PRIMES + fflow::BIG_UINT_PRIMES_SIZE);
  while (table.size() <= index) {
    mpz_t z;
    mpz_init_set_ui(z, static_cast<unsigned long>(table.back() - 2));
    while (mpz_probab_prime_p(z, 25) == 0)
      mpz_sub_ui(z, z, 2);
    table.push_back(static_cast<std::uint64_t>(mpz_get_ui(z)));
    mpz_clear(z);
  }
  return table[index];
}

unsigned finiteflow_symbolic_univariate_plan_degree(const unsigned need[],
                                                    const char unfinished[],
                                                    unsigned noutputs,
                                                    unsigned n_extra) {
  // Sampling at plan D provides 2D+1+n_extra points; n_extra covers the check
  // points the sampler appends beyond the interpolation itself.
  unsigned m = 0;
  for (unsigned i = 0; i < noutputs; ++i)
    if (unfinished[i])
      m = std::max(m, need[i]);
  return m > n_extra ? (m - n_extra) / 2 : 0u;
}

unsigned finiteflow_symbolic_univariate_batch_size(unsigned nthreads,
                                                   unsigned pts_per_prime,
                                                   unsigned primes_left) {
  // A prime costs one sample sweep per nthreads points it needs, so b primes
  // sharing a sweep cost ceil(b*pts/nthreads) sweeps: pick the b with the
  // fewest sweeps per prime.  A tie keeps the larger batch: the same number of
  // sweeps per prime, but more tasks per thread, which averages out the
  // per-point spread instead of making the slowest point the wall.
  constexpr unsigned kMaxBatch = 8;
  if (nthreads <= 1)
    return 1;
  const unsigned pts = std::max(1u, pts_per_prime);
  auto sweeps = [&](unsigned b) { return (b * pts + nthreads - 1) / nthreads; };
  unsigned best = 1;
  unsigned best_sweeps = sweeps(1);
  for (unsigned b = 2; b <= kMaxBatch && b <= primes_left; ++b) {
    const unsigned s = sweeps(b);
    if (s * best <= best_sweeps * b) {
      best = b;
      best_sweeps = s;
    }
  }
  return best;
}

// Precompute the term layout and the elimination plan.
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

      if (t.second.num == 0 && t.second.den != 0)
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

  secred::build_fwd_plan(terms, eq_begin, sys_idx, plan);
}

secred::SymRowTable NumericEvalPlan::rows() const {
  secred::SymRowTable t;
  t.terms = terms.data();
  t.eq_begin = eq_begin.data();
  t.n_rows = sys_idx.size();
  return t;
}

// Map the coefficients into the field for one prime.
void NumericEvalPlan::fill_values(std::uint64_t p,
                                  std::vector<std::uint64_t> &values) const {
  values.resize(coeffs.size());
  for (std::size_t s = 0; s < coeffs.size(); ++s)
    values[s] = secred::rat_mod_u64(coeffs[s].num, coeffs[s].den, p);
}

namespace secred {

void write_reductions_file(const std::string &path,
                           const GlobalReductionResult &result) {
  const auto &rows = result.reductions;
  const auto &starts = result.target_starts;
  if (starts.empty()) {
    if (!rows.empty())
      throw std::runtime_error("invalid reduction result: rows exist but no "
                               "target block boundaries were recorded");
  } else {
    if (starts.front() != 0)
      throw std::runtime_error("invalid reduction result: the first target "
                               "block boundary is not row 0");
    for (std::size_t i = 0; i < starts.size(); ++i) {
      if (starts[i] >= rows.size())
        throw std::runtime_error("invalid reduction result: a target block "
                                 "boundary is past the rows");
      if (i != 0 && starts[i] <= starts[i - 1])
        throw std::runtime_error("invalid reduction result: target block "
                                 "boundaries are not increasing");
    }
  }

  const std::filesystem::path destination(path);
  const std::filesystem::path temporary =
      destination.string() + ".tmp." + std::to_string(::getpid());
  std::error_code ec;

  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out)
      throw std::runtime_error("cannot open the reductions temporary file " +
                               temporary.string());
    for (std::size_t block = 0; block < starts.size(); ++block) {
      if (block != 0)
        out << '\n';
      const std::size_t end =
          block + 1 < starts.size() ? starts[block + 1] : rows.size();
      for (std::size_t i = starts[block]; i < end; ++i)
        out << rows[i].first << "*(" << rows[i].second << ")\n";
    }
    out.flush();
    if (!out) {
      out.close();
      std::filesystem::remove(temporary, ec);
      throw std::runtime_error("cannot write the reductions file " + path);
    }
    out.close();
    if (!out) {
      std::filesystem::remove(temporary, ec);
      throw std::runtime_error("cannot close the reductions file " + path);
    }
  }

  // The rename below publishes the complete file atomically, so a reader never
  // sees a partial block.  It deliberately does not fsync: durability across a
  // power loss is not part of the reduction contract, and an fsync of even a
  // small reductions file costs more than the rest of a short run's I/O.
  std::filesystem::rename(temporary, destination, ec);
  if (ec) {
    std::filesystem::remove(temporary, ec);
    throw std::runtime_error("cannot replace the reductions file " + path +
                             ": " + ec.message());
  }
}

void solve_subset_planned(const std::vector<Eq> &system,
                          const std::vector<std::uint32_t> &eq_idx, std::uint64_t p,
                          SubSolve &out, bool want_fwd, bool want_bwd,
                          const std::vector<std::uint32_t> *only_keys,
                          const std::vector<std::uint32_t> *solmap_keys,
                          FwdStats *fwd_stats) {
  out.solmap.clear();
  out.eq_index.clear();
  out.fwd_deps.clear();
  out.bwd_deps.clear();

  NumericEvalPlan plan;
  plan.build(system, eq_idx);
  std::vector<std::uint64_t> values;
  plan.fill_values(p, values);
  FwdStore store;
  store.reset(plan.store_ids);
  std::vector<std::uint32_t> row_src;
  forward_eliminate_planned(plan.rows(), values.data(), values.size(), plan.plan, p,
                            store, nullptr, want_fwd ? &out.fwd_deps : nullptr,
                            want_fwd ? &row_src : nullptr, fwd_stats);

  const std::size_t n_fwd_rows = store.rows();

  std::vector<std::uint32_t> keys;
  if (only_keys) {
    keys = *only_keys;
  } else {
    keys.reserve(n_fwd_rows);
    for (std::size_t r = 0; r < n_fwd_rows; ++r)
      keys.push_back(store.ids[store.begin[r]]);
    std::sort(keys.begin(), keys.end());
  }
  backward_eliminate_planned(store, p, keys, nullptr,
                             want_bwd ? &out.bwd_deps : nullptr);

  auto materialise = [&](std::uint32_t id) {
    const std::uint32_t *ri = store.row_ids(id);
    const std::uint64_t *rv = store.row_vals(id);
    const std::uint32_t n = store.row_size(id);
    ResEq re;
    re.reserve(n);
    for (std::uint32_t t = 0; t < n; ++t)
      re.emplace_back(ri[t], rv[t]);
    out.solmap[id] = std::move(re);
  };
  if (solmap_keys) {
    for (std::uint32_t id : *solmap_keys)
      if (store.has_row(id))
        materialise(id);
  } else {
    for (std::size_t r = 0; r < n_fwd_rows; ++r)
      materialise(store.ids[store.begin[r]]);
  }

  if (want_fwd)
    for (std::size_t r = 0; r < n_fwd_rows; ++r)
      out.eq_index[store.ids[store.begin[r]]] = plan.sys_idx[row_src[r]];
}

}

bool solve_prime_numeric(const NumericEvalPlan &plan, std::uint64_t p,
                         const std::vector<std::uint32_t> &bwd_chain,
                         const std::vector<unsigned> &dep_targets,
                         const std::vector<unsigned> &masters,
                         const std::unordered_map<unsigned, unsigned> &master_index,
                         secred::FwdStore &store, std::vector<std::uint64_t> &values,
                         std::vector<std::uint64_t> &xout,
                         std::uint32_t &fail_target, std::uint32_t &fail_integral,
                         const std::atomic<bool> *abort, secred::FwdStats *stats) {
  const unsigned nnindeps = masters.size();

  plan.fill_values(p, values);
  store.reset(plan.store_ids);

  if (!secred::forward_eliminate_planned(plan.rows(), values.data(), values.size(),
                                         plan.plan, p, store, abort, nullptr, nullptr,
                                         stats))
    return false;
  if (!secred::backward_eliminate_planned(store, p, bwd_chain, abort))
    return false;

  xout.assign(dep_targets.size() * static_cast<std::size_t>(nnindeps), 0);
  for (unsigned i = 0; i < dep_targets.size(); ++i) {
    const std::uint32_t target = dep_targets[i];
    if (!store.has_row(target))
      continue;
    const std::uint32_t *red_ids = store.row_ids(target);
    const std::uint64_t *red_vals = store.row_vals(target);
    const std::uint32_t red_size = store.row_size(target);
    for (std::uint32_t k = 1; k < red_size; ++k) {
      auto mit = master_index.find(red_ids[k]);
      if (mit == master_index.end()) {

        fail_target = target;
        fail_integral = red_ids[k];
        return false;
      }
      xout[static_cast<std::size_t>(i) * nnindeps + mit->second] = red_vals[k];
    }
  }
  return true;
}

namespace {
namespace ff = fflow;

using secred::parallel_for;
using secred::parallel_for_dynamic;
using secred::solve_res_eqs;
using secred::write_masters_global;

// Reconstruct every reduction coefficient from its prime samples.
bool reconstruct_ffred(const NumericEvalPlan &plan,
                       const std::vector<uint32_t> &bwd_chain,
                       const std::vector<unsigned> &dep_targets,
                       const std::vector<unsigned> &masters,
                       const std::unordered_map<unsigned, unsigned> &master_index,
                       unsigned nthreads, unsigned max_primes,
                       std::uint64_t max_memory, std::uint64_t memory_reserve,
                       ff::MPRational res[], double *elim_seconds) {
  const unsigned nndeps = dep_targets.size();
  const unsigned nnindeps = masters.size();
  const unsigned nparsout = nndeps * nnindeps;
  if (nthreads == 0)
    nthreads = 1;

  const std::size_t nprime_max = static_cast<std::size_t>(max_primes) + 1;
  struct PrimeSlot {
    std::vector<ff::UInt> xout;
    double seconds = 0.0;
    bool ready = false;
    bool ok = true;
    std::uint32_t fail_target = 0, fail_integral = 0;
    secred::FwdStats stats;
  };
  std::vector<PrimeSlot> slots(nprime_max);

  std::atomic<std::size_t> next{0};
  std::atomic<bool> stop{false};
  std::mutex m;
  std::condition_variable cv;

  const std::uint64_t ceiling = secred::reconstruction_ceiling(
      max_memory, memory_reserve, system_memory().total);
  const std::uint64_t rss_before = process_used_memory();
  std::uint64_t measured = 0;
  // The probe's fill sizes, handed to every worker store before its first point.
  std::size_t fill_terms = 0;
  std::size_t fill_rows = 0;
  {
    secred::FwdStore probe_store;
    std::vector<std::uint64_t> probe_values;
    PrimeSlot &probe_slot = slots[0];
    const auto t0 = std::chrono::steady_clock::now();
    probe_slot.ok = solve_prime_numeric(plan, reconstruction_prime(0), bwd_chain,
                                        dep_targets, masters, master_index, probe_store,
                                        probe_values, probe_slot.xout,
                                        probe_slot.fail_target, probe_slot.fail_integral,
                                        &stop, &probe_slot.stats);
    probe_slot.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const std::uint64_t rss_after = process_used_memory();
    if (probe_slot.ok && rss_after > rss_before)
      measured = rss_after - rss_before;
    // A probe that replayed the plan reached the sizes every worker's first point will
    // fill, so reserving them removes the growth reallocations without estimating.
    if (probe_slot.ok && probe_store.order_ptr == &plan.plan.order) {
      fill_terms = probe_store.ids.size();
      fill_rows = probe_store.begin.size();
    }
  }

  const std::uint64_t modelled =
      secred::reconstruction_worker_bytes(plan.terms.size(), plan.store_ids);
  const std::uint64_t per_worker = measured != 0 ? measured : modelled;

  const unsigned nworkers = secred::reconstruction_worker_cap(
      ceiling, rss_before, per_worker, nthreads,
      static_cast<unsigned>(nprime_max - 1));

  const unsigned trim_floor = secred::reconstruction_trim_floor(nworkers);
  std::atomic<unsigned> running{0};
  std::cout << "  " << secred::kFlowTag << " reconstruct pool: " << nworkers << " of "
            << nthreads << " thread(s), " << secred::format_memory(per_worker)
            << " per worker";
  if (measured != 0)
    std::cout << " measured (model " << secred::format_memory(modelled) << ")";
  else
    std::cout << " from the model (the pre-flight could not be measured)";
  std::cout << " within a " << secred::format_memory(ceiling) << " budget\n" << std::flush;

  next.store(1, std::memory_order_relaxed);
  slots[0].ready = true;
  std::vector<std::thread> workers;
  workers.reserve(nworkers);
  for (unsigned w = 0; w < nworkers; ++w) {
    workers.emplace_back([&]() {
      secred::FwdStore store;
      store.set_fill_reserve(fill_terms, fill_rows);
      std::vector<std::uint64_t> values;
      running.fetch_add(1, std::memory_order_relaxed);
      bool holds_store = true;
      for (;;) {

        if (secred::reconstruction_over_budget(process_used_memory(), ceiling)) {
          unsigned r = running.load(std::memory_order_relaxed);
          while (r > trim_floor &&
                 !running.compare_exchange_weak(r, r - 1, std::memory_order_relaxed)) {
          }
          if (r > trim_floor) {
            holds_store = false;
            break;
          }
        }
        const std::size_t ip = next.fetch_add(1, std::memory_order_relaxed);
        if (ip >= nprime_max || stop.load(std::memory_order_relaxed))
          break;
        const std::uint64_t p = reconstruction_prime(static_cast<unsigned>(ip));
        PrimeSlot &slot = slots[ip];
        const auto t0 = std::chrono::steady_clock::now();
        slot.ok = solve_prime_numeric(plan, p, bwd_chain, dep_targets, masters,
                                      master_index, store, values, slot.xout,
                                      slot.fail_target, slot.fail_integral, &stop,
                                      &slot.stats);
        slot.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                    t0)
                           .count();

        if (stop.load(std::memory_order_relaxed))
          break;
        {
          std::lock_guard<std::mutex> lock(m);
          slot.ready = true;
        }
        cv.notify_all();
        if (!slot.ok)
          break;
      }
      if (holds_store)
        running.fetch_sub(1, std::memory_order_relaxed);
    });
  }

  std::unique_ptr<ff::MPInt[]> zout(new ff::MPInt[nparsout]);
  ff::MPInt totmod, c1, c2;
  bool ok = false;
  secred::FwdStats fwd_total;
  std::size_t nprimes_consumed = 0;

  for (std::size_t ip = 0; ip < nprime_max; ++ip) {
    {
      std::unique_lock<std::mutex> lock(m);
      cv.wait(lock, [&] { return slots[ip].ready; });
    }

    if (!slots[ip].ok) {
      std::cerr << "finiteflow global reduction: prime "
                << reconstruction_prime(static_cast<unsigned>(ip)) << ": target "
                << slots[ip].fail_target << " reduces to integral "
                << slots[ip].fail_integral
                << ", which is not a master (equation selection is stale for this "
                   "prime); aborting instead of emitting a truncated reduction\n";
      break;
    }

    fwd_total.add(slots[ip].stats);
    ++nprimes_consumed;
    ff::Mod mod(reconstruction_prime(static_cast<unsigned>(ip)));
    const ff::UInt *xout = slots[ip].xout.data();
    if (elim_seconds)
      *elim_seconds += slots[ip].seconds;

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

    std::vector<ff::UInt>().swap(slots[ip].xout);
  }

  stop.store(true, std::memory_order_relaxed);
  cv.notify_all();
  for (auto &t : workers)
    t.join();

  if (nprimes_consumed != 0) {
    const double per = static_cast<double>(nprimes_consumed);
    char line[256];
    std::snprintf(line, sizeof(line),
                  "  %s prime elimination (average per prime): %llu row merges moving "
                  "%llu terms / %llu solutions from %llu terms gathered\n",
                  secred::kFlowTag,
                  static_cast<unsigned long long>(fwd_total.merges / per),
                  static_cast<unsigned long long>(fwd_total.moved / per),
                  static_cast<unsigned long long>(fwd_total.pivots / per),
                  static_cast<unsigned long long>(fwd_total.gathered / per));
    std::cout << line << std::flush;
  }
  return ok;
}

}

GlobalReductionResult reduce_global_finiteflow(
    const std::string &raw_equations_path,
    const std::vector<RawIntegral> &all_integrals,
    const std::unordered_map<RawIntegral, unsigned> &integral_id,
    const std::vector<RawIntegral> &targets,
    unsigned narrowing_success, unsigned narrowing_fail,
    unsigned raw_equations_total, std::size_t global_equations_count,
    const std::string &masters_global_path, unsigned nthreads,
    std::uint64_t max_memory, std::uint64_t memory_reserve, unsigned max_primes,
    const BasisVote &basis_vote) {
  GlobalReductionResult res;
  namespace ff = fflow;

  const unsigned nvars = all_integrals.size();
  const auto t0 = std::chrono::steady_clock::now();
  std::cout << "\n  " << secred::kFlowTag << " reading spill file, building system...\n";

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

  const uint64_t learn_prime = ff::BIG_UINT_PRIMES[0];

  secred::SubSolve learn;
  secred::FwdStats learn_stats;
  double learn_s = 0.0;
  {
    auto t_learn0 = std::chrono::steady_clock::now();
    std::vector<uint32_t> all_eq_idx;
    all_eq_idx.reserve(neqs);
    for (std::size_t i = 0; i < system.size(); ++i)
      all_eq_idx.push_back(static_cast<uint32_t>(i));
    secred::solve_subset_planned(system, all_eq_idx, learn_prime, learn, true, true,
                                 nullptr, nullptr, &learn_stats);
    auto t_learn1 = std::chrono::steady_clock::now();
    learn_s = std::chrono::duration<double>(t_learn1 - t_learn0).count();
  }
  std::unordered_map<uint32_t, secred::ResEq> &learn_solmap = learn.solmap;
  std::unordered_map<uint32_t, uint32_t> &eq_index = learn.eq_index;
  std::unordered_map<uint32_t, std::vector<uint32_t>> &fwd_deps = learn.fwd_deps;
  std::unordered_map<uint32_t, std::vector<uint32_t>> &bwd_deps = learn.bwd_deps;

  std::unordered_set<unsigned> needed_masters;
  for (unsigned id : target_ids) {
    auto it = learn_solmap.find(id);
    if (it == learn_solmap.end())
      continue;
    const secred::ResEq &red = it->second;
    for (std::size_t k = 1; k < red.size(); ++k)
      needed_masters.insert(red[k].first);
  }

  std::vector<unsigned> masters;
  masters.reserve(needed_masters.size());
  for (unsigned id : needed_masters)
    masters.push_back(id);
  std::sort(masters.begin(), masters.end(),
            [](unsigned a, unsigned b) { return a > b; });
  const unsigned nnindeps = masters.size();

  std::vector<unsigned> dep_targets;
  for (unsigned id : target_ids)
    if (learn_solmap.find(id) != learn_solmap.end())
      dep_targets.push_back(id);
  const unsigned nndeps = dep_targets.size();

  std::unordered_map<unsigned, unsigned> master_index;
  for (unsigned j = 0; j < nnindeps; ++j)
    master_index[masters[j]] = j;

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
  std::unordered_set<uint32_t> needed = sweep(true);
  const std::unordered_set<uint32_t> needed_fwd = sweep(false);

  if (neqs <= 1000) {
    std::cerr << "\n===== RAW equations (" << neqs << ") =====\n";
    for (std::size_t i = 0; i < system.size(); ++i) {
      std::cerr << "eq[" << i << "] front=" << system[i].front().first
                << " :";
      for (const auto &t : system[i])
        std::cerr << " " << t.first << "(" << secred::rat_text(t.second) << ")";
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

  std::vector<uint32_t> needed_eq_idx;
  std::vector<uint32_t> bwd_chain;
  bool use_forward = false;
  {
    auto solve_at_learn = [&](const std::vector<uint32_t> &idx,
                              secred::SubSolve &sub) {

      secred::solve_subset_planned(system, idx, learn_prime, sub, false, true, nullptr,
                                   &dep_targets);
    };
    GlobalSelection gsel =
        select_global_forward(eq_index, fwd_deps, bwd_deps, learn_solmap, dep_targets,
                              master_index, solve_at_learn);
    if (gsel.fallback) {
      std::cerr << "finiteflow global reduction: forward selection did not "
                   "converge, falling back to the legacy closure\n";
    } else {
      use_forward = true;

      log_selection("finiteflow global reduction: ", gsel);
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

  learn = secred::SubSolve{};

  {
    secred::LearnSummary ls;
    ls.independent_equations = res.selected_equations;
    ls.masters = nnindeps;
    ls.seconds = learn_s;
    std::cout << "\n" << secred::format_learn_summary(ls);
    char line[256];
    std::snprintf(line, sizeof(line),
                  "  %s learn elimination: %llu row merges moving %llu terms / %llu "
                  "solutions from %llu terms gathered\n",
                  secred::kFlowTag,
                  static_cast<unsigned long long>(learn_stats.merges),
                  static_cast<unsigned long long>(learn_stats.moved),
                  static_cast<unsigned long long>(learn_stats.pivots),
                  static_cast<unsigned long long>(learn_stats.gathered));
    std::cout << line << std::flush;
  }

  write_masters_global(masters_global_path, all_integrals, masters);

  for (unsigned j = 0; j < nnindeps; ++j)
    res.masters.push_back(all_integrals[masters[j]]);
  res.learn_seconds = learn_s;

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

  if (basis_vote && basis_vote(res.masters)) {
    res.reconstruction_skipped = true;
    res.ok = true;
    return res;
  }

  std::unique_ptr<ff::MPRational[]> resmp(
      new ff::MPRational[(std::size_t)nndeps * nnindeps]);
  double elim_seconds = 0.0;
  std::cout << "\n  " << secred::kFlowTag << " reconstruct (rational reconstruction, up to "
            << max_primes << " primes, " << nthreads << " threads)...\n";
  const auto t2 = std::chrono::steady_clock::now();

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

  std::vector<secred::Eq>().swap(system);
  bool ok = reconstruct_ffred(plan, bwd_chain, dep_targets, masters,
                              master_index, nthreads, max_primes, max_memory,
                              memory_reserve, resmp.get(), &elim_seconds);
  const auto t3 = std::chrono::steady_clock::now();
  if (!ok) {
    std::cerr << "finiteflow global reduction: reconstruct failed\n";
    return res;
  }
  std::cout << "  " << secred::kFlowTag << " reconstruct done ("
            << secred::format_duration(std::chrono::duration<double>(t3 - t2).count())
            << ")\n";

  std::unordered_map<unsigned, unsigned> dep_index;
  for (unsigned i = 0; i < nndeps; ++i)
    dep_index[dep_targets[i]] = i;

  for (const auto &t : targets) {
    res.add_target(t);
    auto it = integral_id.find(t);
    if (it == integral_id.end())
      continue;
    auto dit = dep_index.find(it->second);
    if (dit == dep_index.end())
      continue;
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

  res.reconstruction_seconds =
      std::chrono::duration<double>(t3 - t2).count();

  res.ok = true;
  return res;
}

namespace {

using secred::eval_sym_coeff_mod;
using secred::eval_sym_eqs_mod;
using secred::fill_powers;
using secred::make_powers;
using secred::make_sym_coeff;
using secred::sym_max_exps;
using secred::SymEq;

// One sampling plan for every output: per variable the widest span any output
// needs, and the largest total degrees (mirrors fflow's merged plan).
void merge_var_degrees(const fflow::RatFunVarDegrees degs[],
                       const unsigned numdeg[], const unsigned dendeg[],
                       unsigned nvars, unsigned noutputs,
                       fflow::RatFunVarDegrees &mdegs, unsigned &mnumdeg,
                       unsigned &mdendeg) {
  mdegs.resize(nvars);
  for (unsigned idx = 0; idx < noutputs; ++idx)
    for (unsigned v = 0; v < nvars; ++v) {
      mdegs.num_maxdegs(nvars)[v] =
          std::max(mdegs.num_maxdegs(nvars)[v],
                   degs[idx].num_maxdegs(nvars)[v] -
                       degs[idx].num_mindegs(nvars)[v]);
      mdegs.den_maxdegs(nvars)[v] =
          std::max(mdegs.den_maxdegs(nvars)[v],
                   degs[idx].den_maxdegs(nvars)[v] -
                       degs[idx].den_mindegs(nvars)[v]);
    }
  if (!noutputs) {
    mnumdeg = mdendeg = 0;
  } else {
    mnumdeg = *std::max_element(numdeg, numdeg + noutputs);
    mdendeg = *std::max_element(dendeg, dendeg + noutputs);
  }
}

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

  // Copy with its own coefficient pool, so several primes can be reduced and
  // evaluated side by side.
  SymbolicFFRedAlgorithm(const SymbolicFFRedAlgorithm &o)
      : fflow::Algorithm(o), ev_(o.ev_) {
    ev_.rebind_rows();
  }

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

      std::cerr << "finiteflow symbolic global reduction: target "
                << scratch->scratch.fail_target << " reduces to integral "
                << scratch->scratch.fail_integral
                << ", a master the learn point did not carry (equation selection is "
                   "stale); aborting instead of emitting a truncated reduction\n";
    } else {

      std::cerr << "finiteflow symbolic global reduction: target "
                << scratch->scratch.fail_target << " reduces to integral "
                << scratch->scratch.fail_integral
                << ", which is not a master (equation selection is stale); "
                   "aborting instead of emitting a truncated reduction\n";
    }
    return fflow::FAILED;
  }

  secred::SymbolicFFEvaluator ev_;

  mutable FFScratch fallback_scratch_;
  mutable std::mutex fallback_mutex_;
};

}

GlobalReductionResult reduce_global_finiteflow_symbolic(
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
  namespace ff = fflow;

  if (nthreads == 0)
    nthreads = 1;

  const unsigned nvars = all_integrals.size();
  const unsigned nparams = symbols.size();
  const auto t0 = std::chrono::steady_clock::now();
  std::cout << "\n  " << secred::kFlowTag << " reading spill file, building symbolic "
               "system...\n";

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

  const uint64_t learn_prime = ff::BIG_UINT_PRIMES[0];
  secred::SymbolicGlobalLearn learn = secred::symbolic_global_learn(
      pool, target_ids, all_integrals, symbols, learn_prime, masters_global_path,
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

  for (unsigned j = 0; j < nnindeps; ++j)
    res.masters.push_back(all_integrals[learn.masters[j]]);
  res.learn_seconds = learn.learn_seconds;

  if (basis_vote && basis_vote(res.masters)) {
    res.reconstruction_skipped = true;
    res.ok = true;
    return res;
  }

  std::vector<GiNaC::ex> symbol_ex;
  symbol_ex.reserve(nparams);
  for (const auto &s : symbols)
    symbol_ex.push_back(GiNaC::ex(s));

  SymbolicFFRedAlgorithm alg(std::move(pool), std::move(learn.needed_eq_idx),
                             std::move(learn.bwd_chain), learn.dep_targets,
                             std::move(learn.master_index), nnindeps, learn.items,
                             std::move(symbol_ex));

  const unsigned nparsout = alg.nparsout;
  std::cout << "\n  " << secred::kFlowTag << " reconstruct (rational-function): "
            << nparsout << " function(s) will be interpolated (the non-zero entries of a "
            << nndeps << " x " << nnindeps << " target x master block)\n";

  std::cout << "  " << secred::kFlowTag << " " << alg.n_coefficient_terms()
            << " function(s) to evaluate per sample point, of which "
            << alg.n_distinct_coeffs() << " are distinct\n";

  ff::ReconstructionOptions opt;

  if (max_primes == 0) {
    std::cerr << "finiteflow symbolic: reconstruction failed (prime cap 0)\n";
    return res;
  }
  std::cout << "  " << secred::kFlowTag << " prime cap " << max_primes;
  if (max_primes > fflow::BIG_UINT_PRIMES_SIZE)
    std::cout << " (" << fflow::BIG_UINT_PRIMES_SIZE
              << " from the FiniteFlow table, the rest generated)";
  std::cout << "\n";

  std::unique_ptr<ff::UInt[]> shift(new ff::UInt[nparams]);
  {
    ff::Mod smod(ff::BIG_UINT_PRIMES[opt.start_mod]);
    for (unsigned i = 0; i < nparams; ++i)
      shift[i] = ff::sample_uint(ff::OFFSET_4, i, smod);
  }

  std::unique_ptr<unsigned[]> numdeg(new unsigned[nparsout]());
  std::unique_ptr<unsigned[]> dendeg(new unsigned[nparsout]());
  std::unique_ptr<ff::RatFunVarDegrees[]> vdegs(
      new ff::RatFunVarDegrees[nparsout]);
  for (unsigned i = 0; i < nparsout; ++i)
    vdegs[i].resize(nparams);

  alg.reduce_for_prime(ff::BIG_UINT_PRIMES[opt.start_mod], nthreads);

  if (nparams > 1) {
    ff::Ret ret = ff::algorithm_get_degrees(alg, nullptr, nullptr, shift.get(),
                                            opt, numdeg.get(), dendeg.get());
    if (ret != ff::SUCCESS) {
      std::cerr << "finiteflow symbolic: degree estimation failed (ret " << ret
                << ")\n";
      return res;
    }
    for (unsigned var = 0; var < nparams; ++var) {
      ret = ff::algorithm_get_var_degrees(alg, nullptr, nullptr, var,
                                          shift.get(), opt, vdegs.get());
      if (ret != ff::SUCCESS) {
        std::cerr << "finiteflow symbolic: var-degree estimation failed (ret "
                  << ret << ")\n";
        return res;
      }
    }
  }

  ff::UIntCache cache;
  cache.init(nparams + 1);
  std::unique_ptr<ff::MPReconstructedRatFun[]> recf(
      new ff::MPReconstructedRatFun[nparsout]);
  std::vector<ff::MPInt> totmod(nparsout);
  std::vector<char> reconstructed(nparsout, 0);
  // Univariate path: the first prime reconstructs every output by Thiele at a
  // small trial degree, and its result fixes that output's exact degree pair
  // and normalization pivot for the linear reconstruction at the later primes.
  // The linear solve then recovers the remaining coefficients incrementally: a
  // coefficient that has been reconstructed rationally is fixed and drops out
  // of the next prime's system, so the number of sample points per prime
  // shrinks as the reconstruction progresses.  A prime at which the linear
  // solve is singular for an output falls back to Thiele for good.
  constexpr unsigned kFirstPrimeMaxDeg = 15;
  constexpr unsigned kMaxTrialDegree = 1023;
  std::vector<char> have_degrees(nparsout, nparams > 1 ? 1 : 0);
  std::vector<char> thiele_fallback(nparsout, 0);
  std::vector<secred::UniLinearState> lin(nparsout);
  std::unique_ptr<unsigned[]> pivot(new unsigned[nparsout]());
  if (nparams == 1 && nparsout)
    std::cout << "  " << secred::kFlowTag << " univariate plan: exact degrees "
              << "learned from a trial Thiele pass at the first prime, then "
              << "per-output degrees\n";
  std::atomic<unsigned> remaining{nparsout};
  double eval_seconds = 0.0;
  std::uint64_t evaluated_points = 0;

  // The degree pair of one reconstructed output, read off its sparse
  // polynomials (monomials are stored in descending degree).
  auto learn_degrees = [&](unsigned idx, const ff::SparseRationalFunction &f) {
    unsigned nd = 0, dd = 0;
    for (const auto &m : f.numerator())
      nd = std::max<unsigned>(nd, m.degree());
    for (const auto &m : f.denominator())
      dd = std::max<unsigned>(dd, m.degree());
    numdeg[idx] = std::max(numdeg[idx], nd);
    dendeg[idx] = std::max(dendeg[idx], dd);
    have_degrees[idx] = 1;
  };

  std::vector<char> not_done(nparsout, 0);
  std::vector<unsigned> need(nparsout, 0);
  auto plan_degree = [&]() {
    for (unsigned i = 0; i < nparsout; ++i) {
      not_done[i] = reconstructed[i] ? 0 : 1;
      if (!have_degrees[i])
        // a trial Thiele at the first-prime degree
        need[i] = 2 * kFirstPrimeMaxDeg + 1 + opt.n_uchecks;
      else if (thiele_fallback[i])
        // a continued fraction of numdeg+dendeg+1 terms plus its checks
        need[i] = numdeg[i] + dendeg[i] + 1 + opt.n_uchecks;
      else
        // one equation per unknown coefficient plus the check rows
        need[i] = secred::uni_unknown_count(lin[i], pivot[i]) + opt.n_checks;
    }
    return finiteflow_symbolic_univariate_plan_degree(
        need.data(), not_done.data(), nparsout, opt.n_uchecks);
  };

  // Verification points are only needed where the output is not settled by its
  // own linear system: the multi-symbol path and the Thiele fallbacks.  A
  // complete univariate linear system already requires consistency at its own
  // check rows, so on a single symbol output the extra points would only cost
  // a sample each.
  auto needs_verify_points = [&]() {
    if (nparams > 1)
      return true;
    for (unsigned i = 0; i < nparsout; ++i)
      if (thiele_fallback[i] && !reconstructed[i])
        return true;
    return false;
  };

  // Points for one prime: the shared reconstruction plan plus the verification
  // points, minus everything already in the cache.
  auto prime_points = [&](unsigned ip, ff::Mod mod, unsigned plan_deg,
                          bool verify_points) {
    ff::GenerateSamplePoints sfun(nparams);
    if (cache.size())
      sfun.set_complement(&cache);
    if (ip < max_primes) {
      ff::RatFunReconstruction rec(
          nparams, opt.polymethod, nparams == 1 ? plan_deg : opt.max_deg);
      ff::set_rec_options(opt, shift.get(), rec);
      if (nparams == 1) {
        // set_rec_options resets max_deg to the global default; the univariate
        // plan is the per-prime degree instead.
        rec.setMaxDegree(plan_deg);
        ff::RatFunVarDegrees unused;
        rec.sample(sfun, mod, 0, 0, std::move(unused));
      } else {
        ff::RatFunVarDegrees mdegs;
        unsigned mnumdeg = 0, mdendeg = 0;
        merge_var_degrees(vdegs.get(), numdeg.get(), dendeg.get(), nparams,
                          nparsout, mdegs, mnumdeg, mdendeg);
        rec.sample(sfun, mod, mnumdeg, mdendeg, std::move(mdegs));
      }
    }
    if (ip > 0 && verify_points) {
      std::unique_ptr<ff::UInt[]> x(new ff::UInt[nparams]);
      for (unsigned check_i = 0; check_i < opt.n_checks; ++check_i) {
        for (unsigned v = 0; v < nparams; ++v)
          x[v] = ff::sample_uint(ff::OFFSET_1, check_i + v, mod);
        sfun.evaluate(x.get(), mod);
      }
    }
    ff::SamplePointsVector points;
    sfun.append_to_vector(points);
    return points;
  };

  // Merge the per-thread caches into the shared one and drop the empties.
  auto merge_part_caches =
      [&](std::vector<std::unique_ptr<ff::UIntCache>> &part_caches) {
        std::vector<std::unique_ptr<ff::UIntCache>> filled;
        for (auto &c : part_caches)
          if (c && c->size())
            filled.push_back(std::move(c));
        if (!filled.empty())
          cache.merge_caches(filled.data(), filled.size());
      };

  // Evaluate one point set in parallel against one algorithm and merge it into
  // the cache.
  auto evaluate_points = [&](SymbolicFFRedAlgorithm &a,
                             const ff::SamplePointsVector &points) {
    if (points.empty())
      return;
    const bool show_bar = points.size() >= 256;
    secred::ProgressBar bar(std::string("  ") + secred::kFlowTag +
                                " evaluating samples",
                            points.size(), secred::stdout_is_tty());
    if (show_bar)
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
    const std::size_t npts = points.size();
    const unsigned nt =
        static_cast<unsigned>(std::min<std::size_t>(nthreads, npts));
    for (unsigned t = 0; t < nt; ++t) {
      const std::size_t begin = npts * t / nt;
      const std::size_t end = npts * (t + 1) / nt;
      if (begin >= end)
        continue;
      workers.emplace_back([&, t, begin, end]() {
        ff::evaluate_and_cache_samples(a, part_scratch[t].get(), nullptr,
                                       points.data() + begin,
                                       points.data() + end, *part_caches[t]);
        {
          std::lock_guard<std::mutex> lock(done_mutex);
          done.fetch_add(end - begin, std::memory_order_relaxed);
        }
        done_cv.notify_one();
      });
    }
    {
      std::unique_lock<std::mutex> lock(done_mutex);
      while (done.load(std::memory_order_relaxed) < npts) {
        done_cv.wait_for(lock, std::chrono::milliseconds(100));
        if (show_bar)
          std::cout << bar.update(done.load(std::memory_order_relaxed))
                    << std::flush;
      }
    }
    for (auto &w : workers)
      w.join();
    if (show_bar)
      std::cout << bar.finish() << std::flush;

    merge_part_caches(part_caches);
    evaluated_points += npts;
  };

  // Evaluate several primes' point sets in one sweep.  Every point is an
  // independent task, so a batch keeps the threads busy even though one prime
  // carries only a handful of points; each task runs against its own
  // prime-reduced copy of the system.
  auto evaluate_points_batched =
      [&](const std::vector<SymbolicFFRedAlgorithm *> &algos,
          const std::vector<ff::SamplePointsVector> &sets) {
        struct Task {
          SymbolicFFRedAlgorithm *algo;
          const std::unique_ptr<ff::UInt[]> *begin;
        };
        std::vector<Task> tasks;
        std::size_t total = 0;
        for (std::size_t b = 0; b < sets.size(); ++b) {
          total += sets[b].size();
          for (std::size_t i = 0; i < sets[b].size(); ++i)
            tasks.push_back(Task{algos[b], sets[b].data() + i});
        }
        if (tasks.empty()) {
          evaluated_points += total;
          return;
        }

        const bool show_bar = tasks.size() >= 256;
        secred::ProgressBar bar(std::string("  ") + secred::kFlowTag +
                                    " evaluating samples",
                                tasks.size(), secred::stdout_is_tty());
        if (show_bar)
          std::cout << bar.update(0) << std::flush;

        const unsigned nt = static_cast<unsigned>(
            std::min<std::size_t>(nthreads, tasks.size()));
        std::vector<std::unique_ptr<ff::UIntCache>> part_caches(nt);
        std::vector<std::unique_ptr<SymbolicFFRedAlgorithm::FFScratch>>
            part_scratch(nt);
        for (unsigned t = 0; t < nt; ++t) {
          part_caches[t].reset(new ff::UIntCache);
          part_caches[t]->init(nparams + 1);
          part_scratch[t].reset(new SymbolicFFRedAlgorithm::FFScratch);
        }

        std::atomic<std::size_t> next{0};
        std::atomic<std::size_t> done{0};
        std::mutex done_mutex;
        std::condition_variable done_cv;
        std::vector<std::thread> workers;
        workers.reserve(nt);
        for (unsigned t = 0; t < nt; ++t) {
          workers.emplace_back([&, t]() {
            for (;;) {
              const std::size_t k = next.fetch_add(1, std::memory_order_relaxed);
              if (k >= tasks.size())
                break;
              ff::evaluate_and_cache_samples(
                  *tasks[k].algo, part_scratch[t].get(), nullptr,
                  tasks[k].begin, tasks[k].begin + 1, *part_caches[t]);
              {
                std::lock_guard<std::mutex> lock(done_mutex);
                done.fetch_add(1, std::memory_order_relaxed);
              }
              done_cv.notify_one();
            }
          });
        }
        {
          std::unique_lock<std::mutex> lock(done_mutex);
          while (done.load(std::memory_order_relaxed) < tasks.size()) {
            done_cv.wait_for(lock, std::chrono::milliseconds(100));
            if (show_bar)
              std::cout << bar.update(done.load(std::memory_order_relaxed))
                        << std::flush;
          }
        }
        for (auto &w : workers)
          w.join();
        if (show_bar)
          std::cout << bar.finish() << std::flush;

        merge_part_caches(part_caches);
        evaluated_points += total;
      };

  // Verify one output at this prime; when the merged function is not confirmed,
  // reconstruct it at this prime and extend the CRT modulus.
  auto reconstruct_one = [&](unsigned idx, unsigned ip, ff::Mod mod,
                             unsigned plan_deg) -> ff::Ret {
    ff::SampledRatFun sampledfun;
    sampledfun.set_function_cache(&cache);
    sampledfun.idx = idx;
    std::unique_ptr<ff::UInt[]> x(new ff::UInt[nparams]);

    // A univariate output on the linear solver is confirmed by its own
    // consistency rows once its coefficient set is complete, so only the
    // multi-symbol path and the Thiele fallbacks are checked here.
    if (ip > 0 && (nparams > 1 || thiele_fallback[idx])) {
      bool ok = true;
      for (unsigned check_i = 0; ok && check_i < opt.n_checks; ++check_i) {
        for (unsigned v = 0; v < nparams; ++v)
          x[v] = ff::sample_uint(ff::OFFSET_1, check_i + v, mod);
        const ff::UInt y = sampledfun.evaluate(x.get(), mod);
        if (y == ff::FAILED || recf[idx].eval(x.get(), mod) != y)
          ok = false;
      }
      if (ok) {
        reconstructed[idx] = 1;
        remaining.fetch_sub(1, std::memory_order_relaxed);
        return ff::SUCCESS;
      }
    }
    if (ip == max_primes)
      return ff::MISSING_PRIMES;

    ff::SparseRationalFunction tmp;
    bool linear_complete = false;
    if (nparams > 1) {
      ff::RatFunReconstruction rec(nparams, opt.polymethod, opt.max_deg);
      ff::set_rec_options(opt, shift.get(), rec);
      if (rec.reconstructWithDegs(sampledfun, mod, numdeg[idx], dendeg[idx],
                                  vdegs[idx]) != ff::SUCCESS)
        return ff::FAILED;
      tmp = std::move(rec.getFunction());
    } else if (!have_degrees[idx]) {
      // Trial Thiele at the plan degree when this output has no degree pair
      // yet; a successful result fixes the pair and the normalization pivot,
      // and seeds the incremental linear reconstruction.
      ff::RatFunReconstruction rec(nparams, opt.polymethod, plan_deg);
      ff::set_rec_options(opt, shift.get(), rec);
      rec.setMaxDegree(plan_deg);
      if (rec.reconstruct(sampledfun, mod) != ff::SUCCESS)
        return ff::FAILED;
      learn_degrees(idx, rec.getFunction());
      const ff::SparseRationalFunction &trial = rec.getFunction();
      // lowest() is not const-qualified, so read it off a copy
      ff::SparsePoly trial_den = trial.denominator();
      pivot[idx] =
          trial_den.is_zero() ? 0 : trial_den.lowest().degree();
      secred::uni_seed(lin[idx], trial, numdeg[idx], dendeg[idx], pivot[idx],
                       mod.n());
      tmp = std::move(rec.getFunction());
    } else if (thiele_fallback[idx]) {
      // The exact degree pair bounds the continued fraction, so Thiele needs
      // numdeg+dendeg+1 terms, not 2*max(numdeg,dendeg)+1.
      const unsigned maxterms =
          std::max<unsigned>(numdeg[idx] + dendeg[idx] + 1, 2 * plan_deg + 1);
      ff::URatFunReconstruction th(maxterms);
      th.n_checks = opt.n_uchecks;
      th.n_singular = opt.n_singular;
      th.setX0(ff::OFFSET_0_HASH);
      ff::MtoURatFun ufun(sampledfun);
      if (th.reconstruct(ufun, mod) != ff::SUCCESS)
        return ff::FAILED;
      ff::URationalFunction urat;
      th.writeResult(mod, urat);
      tmp.fromUnivariate(urat);
      learn_degrees(idx, tmp);
    } else {
      // Partial linear solve: this prime only carries equations for the
      // coefficients not yet recovered, so a nearly complete output costs
      // little more than its check points.
      std::vector<ff::UInt> cnum, cden;
      if (secred::uni_linear_step(lin[idx], numdeg[idx], dendeg[idx],
                                  pivot[idx], mod, opt.n_checks,
                                  opt.n_singular, sampledfun, cnum, cden) !=
          ff::SUCCESS) {
        thiele_fallback[idx] = 1;
        lin[idx] = secred::UniLinearState();
        return ff::FAILED;
      }
      ff::URationalFunction urat;
      urat.setNumDegree(numdeg[idx]);
      urat.setDenDegree(dendeg[idx]);
      for (unsigned j = 0; j <= numdeg[idx]; ++j)
        urat.num(j) = cnum[j];
      for (unsigned j = 0; j <= dendeg[idx]; ++j)
        urat.den(j) = cden[j];
      tmp.fromUnivariate(urat);
      // With every coefficient known the step's rows were consistency checks
      // against this prime, so the output is final.
      linear_complete = secred::uni_unknown_count(lin[idx], pivot[idx]) == 0;
    }

    if (ip == 0) {
      totmod[idx] = mod.n();
      recf[idx].setMod(totmod[idx]);
      recf[idx].copy(std::move(tmp));
    } else {
      ff::MPInt c1, c2;
      ff::chinese_remainder_coeffs(totmod[idx], mod.n(), c1, c2, totmod[idx]);
      recf[idx].merge(std::move(tmp), c1, c2, totmod[idx]);
    }
    if (linear_complete) {
      reconstructed[idx] = 1;
      remaining.fetch_sub(1, std::memory_order_relaxed);
    }
    return ff::SUCCESS;
  };

  secred::ReconStatus status(
      std::string("  ") + secred::kFlowTag + " reconstructing",
      secred::stdout_is_tty());
  const auto t_recon0 = std::chrono::steady_clock::now();
  bool done_all = (nparsout == 0);

  // Prime batching: pass 0 learns the exact degrees and pass 1 settles the
  // Thiele fallbacks; from then on consecutive primes may share one parallel
  // sample sweep, and the batch size is re-chosen per sweep.
  for (unsigned ip = 0; !done_all && ip <= max_primes;) {
    const auto t_prime0 = std::chrono::steady_clock::now();

    unsigned plan_deg = (nparams == 1) ? plan_degree() : opt.max_deg;
    const bool verify_points = needs_verify_points();
    unsigned bsize = 1;
    if (nparams == 1 && ip >= 2) {
      const unsigned pts_per_prime = 2 * plan_deg + 1 + opt.n_uchecks +
                                     opt.n_singular +
                                     (verify_points ? opt.n_checks : 0u);
      bsize = finiteflow_symbolic_univariate_batch_size(
          nthreads, pts_per_prime, max_primes - ip + 1);
    }

    std::vector<std::uint64_t> primes(bsize);
    for (unsigned b = 0; b < bsize; ++b)
      primes[b] = reconstruction_prime(ip + b);

    // One algorithm per prime of the batch, each reduced to its own modulus.
    std::vector<SymbolicFFRedAlgorithm *> alg_for(bsize, &alg);
    std::vector<std::unique_ptr<SymbolicFFRedAlgorithm>> batch_algs;
    if (nparams == 1) {
      if (ip != 0) {
        if (bsize == 1) {
          alg.reduce_for_prime(primes[0], nthreads);
        } else {
          batch_algs.reserve(bsize);
          for (unsigned b = 0; b < bsize; ++b) {
            batch_algs.emplace_back(new SymbolicFFRedAlgorithm(alg));
            batch_algs.back()->reduce_for_prime(primes[b], nthreads);
            alg_for[b] = batch_algs.back().get();
          }
        }
      }
    } else if (ip != 0) {
      alg.reduce_for_prime(primes[0], nthreads);
    }

    std::vector<ff::SamplePointsVector> batch_points(bsize);
    for (unsigned b = 0; b < bsize; ++b)
      batch_points[b] =
          prime_points(ip + b, ff::Mod(primes[b]), plan_deg, verify_points);

    if (bsize == 1)
      evaluate_points(*alg_for[0], batch_points[0]);
    else
      evaluate_points_batched(alg_for, batch_points);

    for (unsigned b = 0; b < bsize && !done_all; ++b) {
      const unsigned ipb = ip + b;
      ff::Mod mod(primes[b]);
      std::vector<unsigned> failed;
      std::mutex failed_mutex;
      // Longest first: an output's per-prime cost tracks its coefficient count,
      // and the dynamic claims keep the large ones from stacking on one thread.
      std::vector<unsigned> pending;
      for (unsigned idx = 0; idx < nparsout; ++idx)
        if (!reconstructed[idx])
          pending.push_back(idx);
      std::stable_sort(pending.begin(), pending.end(),
                       [&](unsigned ia, unsigned ib) {
                         return recf[ia].numerator().size() +
                                    recf[ia].denominator().size() >
                                recf[ib].numerator().size() +
                                    recf[ib].denominator().size();
                       });
      auto pass = [&](const std::vector<unsigned> &ids, unsigned pd) {
        parallel_for_dynamic(nthreads, ids.size(), [&](std::size_t k) {
          const unsigned idx = ids[k];
          if (reconstructed[idx])
            return;
          if (reconstruct_one(idx, ipb, mod, pd) == ff::SUCCESS)
            return;
          std::lock_guard<std::mutex> lock(failed_mutex);
          failed.push_back(idx);
        });
      };
      pass(pending, plan_deg);

      if (!failed.empty() && ipb < max_primes) {
        // A singular linear solve switched its output to the Thiele fallback;
        // retry at the same plan, whose points are already cached.
        std::vector<unsigned> retry(std::move(failed));
        failed.clear();
        pass(retry, plan_deg);
      }

      // The trial degree was too low, or a fallback output needs more points
      // than the plan provides: grow the plan geometrically and rebuild only
      // the still-failing outputs.
      while (!failed.empty() && ipb < max_primes && nparams == 1 &&
             plan_deg < kMaxTrialDegree) {
        const unsigned old_deg = plan_deg;
        plan_deg = std::min(2 * plan_deg + 1, kMaxTrialDegree);
        ff::SamplePointsVector extra =
            prime_points(ipb, mod, plan_deg, verify_points);
        evaluate_points(*alg_for[b], extra);
        std::vector<unsigned> retry(std::move(failed));
        failed.clear();
        pass(retry, plan_deg);
        if (failed.empty())
          std::cout << "  " << secred::kFlowTag << " trial degree raised at "
                    << "prime " << ipb << ": " << 2 * old_deg + 1 << " -> "
                    << 2 * plan_deg + 1 << " interpolation points\n";
      }

      if (!failed.empty()) {
        std::cerr << "finiteflow symbolic: reconstruction failed for output "
                  << failed.front();
        if (ipb == max_primes)
          std::cerr << " (not verified after " << max_primes << " prime(s))";
        std::cerr << "\n";
        return res;
      }

      done_all = (remaining.load(std::memory_order_relaxed) == 0);
      const double elapsed = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - t_recon0)
                                 .count();
      const std::string line = status.update(elapsed, ipb + 1, evaluated_points);
      if (!line.empty())
        std::cout << line << std::flush;
    }

    eval_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      t_prime0)
            .count();
    ip += bsize;
  }
  std::cout << status.finish() << std::flush;

  if (nparams == 1 && nparsout) {
    std::size_t fallback = 0;
    for (unsigned idx = 0; idx < nparsout; ++idx)
      if (thiele_fallback[idx])
        ++fallback;
    std::cout << "  " << secred::kFlowTag << " univariate split: "
              << nparsout - fallback << " output(s) on the linear solver, "
              << fallback << " on Thiele\n";
  }

  std::cout << "  " << secred::kFlowTag << " reconstructed with "
            << evaluated_points << " sample point(s) in "
            << secred::format_duration(
                   std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                 t_recon0)
                       .count())
            << "\n";

  std::vector<std::string> varnames(nparams);
  for (unsigned i = 0; i < nparams; ++i)
    varnames[i] = symbols[i].get_name();

  std::unordered_map<unsigned, unsigned> dep_index;
  for (unsigned i = 0; i < nndeps; ++i)
    dep_index[learn.dep_targets[i]] = i;

  for (const auto &t : targets) {
    res.add_target(t);
    auto it = integral_id.find(t);
    if (it == integral_id.end())
      continue;
    auto dit = dep_index.find(it->second);
    if (dit == dep_index.end())
      continue;
    unsigned i = dit->second;
    for (unsigned q = learn.items.begin[i]; q < learn.items.begin[i + 1]; ++q) {
      const ff::MPReconstructedRatFun &r = recf[q];
      if (r.numerator().size() == 0)
        continue;
      res.reductions.emplace_back(
          all_integrals[learn.masters[learn.items.col[q]]], r.to_str(varnames.data()));
    }
  }

  res.reconstruction_seconds = eval_seconds;

  res.ok = true;
  return res;
}
