
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

#include <fflow/common.hh>
#include <fflow/gcd.hh>
#include <fflow/graph.hh>
#include <fflow/mp_common.hh>
#include <fflow/mp_gcd.hh>
#include <fflow/numeric_solver.hh>
#include <fflow/primes.hh>

#include <cln/cln.h>

#include "prime_checkpoint.h"
#include "resume.h"
#include <ginac/ginac.h>

#include <memory>
#include <random>

#include "global_select.h"
#include "ffred_solver.h"
#include "masters_file.h"
#include "parallel_pool.h"
#include "progress_bar.h"
#include "reconstruct_budget.h"
#include "spill_reader.h"
#include "sym_coeff.h"
#include "symbolic_ffred.h"

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

bool numeric_dag_fits(std::size_t n_terms, std::size_t n_rows) {
  return n_terms > 0 && n_rows > 0 && n_terms <= kNumericDagMaxTerms &&
         n_rows <= kNumericDagMaxRows;
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

void NumericEvalPlan::select_rows(const std::vector<std::uint32_t> &keep) {
  std::vector<std::uint32_t> new_eq_begin;
  new_eq_begin.reserve(keep.size() + 1);
  new_eq_begin.push_back(0);
  // Fresh slot numbering in first-use order reproduces the slot layout a
  // build() over the kept rows would produce, so the plan stays canonical.
  std::vector<secred::Rat64> new_coeffs;
  std::unordered_map<std::uint32_t, std::uint32_t> slot_map;
  std::size_t write = 0;
  std::uint32_t max_id = 0;
  for (std::uint32_t row : keep) {
    for (std::size_t t = eq_begin[row]; t < eq_begin[row + 1]; ++t) {
      const std::pair<std::uint32_t, std::uint32_t> term = terms[t];
      auto it = slot_map.find(term.second);
      std::uint32_t slot;
      if (it == slot_map.end()) {
        slot = static_cast<std::uint32_t>(new_coeffs.size());
        new_coeffs.push_back(coeffs[term.second]);
        slot_map.emplace(term.second, slot);
      } else {
        slot = it->second;
      }
      terms[write++] = {term.first, slot};
      if (term.first > max_id)
        max_id = term.first;
    }
    new_eq_begin.push_back(static_cast<std::uint32_t>(write));
  }
  terms.resize(write);
  coeffs = std::move(new_coeffs);
  eq_begin = std::move(new_eq_begin);
  // `keep` is ascending, so every source row lies at or past its destination
  // row and the forward copy never overwrites a row still to be read.
  for (std::size_t r = 0; r < keep.size(); ++r)
    sys_idx[r] = sys_idx[keep[r]];
  sys_idx.resize(keep.size());
  store_ids = static_cast<std::size_t>(max_id) + 1;

  secred::build_fwd_plan(terms, eq_begin, sys_idx, plan);
}

NumericEvalPlan NumericEvalPlan::subset(
    const std::vector<std::uint32_t> &eq_idx) const {
  NumericEvalPlan out;
  out.coeffs = coeffs;
  out.sys_idx = eq_idx;
  std::size_t n_terms = 0;
  for (std::uint32_t i : eq_idx)
    n_terms += eq_begin[i + 1] - eq_begin[i];
  out.terms.reserve(n_terms);
  out.eq_begin.reserve(eq_idx.size() + 1);
  out.eq_begin.push_back(0);
  std::uint32_t max_id = 0;
  for (std::uint32_t i : eq_idx) {
    for (std::size_t t = eq_begin[i]; t < eq_begin[i + 1]; ++t) {
      out.terms.push_back(terms[t]);
      if (terms[t].first > max_id)
        max_id = terms[t].first;
    }
    out.eq_begin.push_back(static_cast<std::uint32_t>(out.terms.size()));
  }
  out.store_ids = static_cast<std::size_t>(max_id) + 1;

  secred::build_fwd_plan(out.terms, out.eq_begin, out.sys_idx, out.plan);
  return out;
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
  NumericEvalPlan plan;
  plan.build(system, eq_idx);
  std::vector<std::uint64_t> values;
  plan.fill_values(p, values);
  solve_plan(plan, values, p, out, want_fwd, want_bwd, only_keys, solmap_keys,
             fwd_stats);
}

void solve_plan(const NumericEvalPlan &plan,
                const std::vector<std::uint64_t> &values, std::uint64_t p,
                SubSolve &out, bool want_fwd, bool want_bwd,
                const std::vector<std::uint32_t> *only_keys,
                const std::vector<std::uint32_t> *solmap_keys,
                FwdStats *fwd_stats) {
  out.solmap.clear();
  out.eq_index.clear();
  out.fwd_deps.clear();
  out.bwd_deps.clear();

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

bool build_numeric_schedule(const NumericEvalPlan &plan, std::uint64_t p,
                            secred::FwdDag &dag, secred::FwdStore &store,
                            std::vector<std::uint64_t> &values,
                            const std::atomic<bool> *abort) {
  plan.fill_values(p, values);
  if (std::find(values.begin(), values.end(), std::uint64_t(0)) != values.end())
    return false;
  store.reset(plan.store_ids);
  return secred::build_fwd_dag(plan.rows(), values.data(), values.size(), p,
                               plan.plan, dag, store, nullptr, abort);
}

bool solve_prime_numeric(const NumericEvalPlan &plan, std::uint64_t p,
                         const std::vector<std::uint32_t> &bwd_chain,
                         const std::vector<unsigned> &dep_targets,
                         const std::vector<unsigned> &masters,
                         const std::unordered_map<unsigned, unsigned> &master_index,
                         secred::FwdStore &store, std::vector<std::uint64_t> &values,
                         std::vector<std::uint64_t> &xout,
                         std::uint32_t &fail_target, std::uint32_t &fail_integral,
                         const std::atomic<bool> *abort, secred::FwdStats *stats,
                         const secred::FwdDag *replay_dag,
                         std::vector<std::uint64_t> *dag_acc) {
  const unsigned nnindeps = masters.size();
  plan.fill_values(p, values);
  store.reset(plan.store_ids);

  bool forward_ok = false;
  if (replay_dag != nullptr && !replay_dag->empty() && dag_acc != nullptr) {
    forward_ok = secred::forward_eliminate_dag(plan.rows(), values.data(), *replay_dag,
                                               p, store, *dag_acc, abort, stats);
    if (!forward_ok)
      store.reset(plan.store_ids);
  }
  if (!forward_ok &&
      !secred::forward_eliminate_planned(plan.rows(), values.data(), values.size(),
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
                       const secred::ResumeControl *resume,
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
    bool loaded = false;
    bool replayed = false;
    std::uint32_t fail_target = 0, fail_integral = 0;
    secred::FwdStats stats;
  };
  std::vector<PrimeSlot> slots(nprime_max);

  // One directory per spill content hash.  The contiguous run of files at its
  // start holds the primes a previous attempt finished; the consumer takes
  // those from disk while the workers cover everything from the first gap on.
  const std::filesystem::path prime_dir =
      resume != nullptr ? secred::prime_resume_dir(*resume, max_primes)
                        : std::filesystem::path();
  unsigned first_missing = 0;
  if (resume != nullptr) {
    std::error_code ec;
    while (first_missing < nprime_max &&
           std::filesystem::is_regular_file(
               secred::prime_checkpoint_path(prime_dir, first_missing), ec)) {
      ec.clear();
      ++first_missing;
    }
  }

  std::atomic<bool> save_warned{false};
  auto save_prime = [&](unsigned ip, const PrimeSlot &slot) {
    if (resume == nullptr || !slot.ok)
      return;
    secred::PrimeCheckpoint checkpoint;
    checkpoint.index = ip;
    checkpoint.prime = reconstruction_prime(ip);
    checkpoint.dep_targets = dep_targets;
    checkpoint.masters = masters;
    checkpoint.xout.assign(slot.xout.begin(), slot.xout.end());
    std::string error;
    if (!secred::write_prime_checkpoint(
            secred::prime_checkpoint_path(prime_dir, ip), checkpoint, &error)) {
      bool expected = false;
      if (save_warned.compare_exchange_strong(expected, true))
        std::cerr << "finiteflow global reduction: cannot save prime " << ip + 1
                  << ": " << error << "\n";
    }
  };
  // A prime file is only usable when it belongs to this prime and basis.
  auto load_prime = [&](unsigned ip, PrimeSlot &slot) -> bool {
    if (resume == nullptr || ip >= first_missing)
      return false;
    secred::PrimeCheckpoint checkpoint;
    std::string error;
    if (!secred::read_prime_checkpoint(
            secred::prime_checkpoint_path(prime_dir, ip), checkpoint, &error) ||
        !secred::prime_checkpoint_matches(checkpoint, ip,
                                          reconstruction_prime(ip), dep_targets,
                                          masters))
      return false;
    slot.xout.assign(checkpoint.xout.begin(), checkpoint.xout.end());
    slot.loaded = true;
    slot.ok = true;
    slot.ready = true;
    return true;
  };

  std::atomic<std::size_t> next{0};
  std::atomic<bool> stop{false};
  std::mutex m;
  std::condition_variable cv;

  const std::uint64_t ceiling = secred::reconstruction_ceiling(
      max_memory, memory_reserve, system_memory().total);
  const std::uint64_t rss_before = process_used_memory();
  std::uint64_t measured = 0;
  // A forward schedule is built only after the pool has run one wave of primes without
  // convergence, and on a thread of its own: a replay then has later primes to pay its
  // build back on, while the solvers keep their pipeline.  Building it at the probe
  // would charge every run several seconds of serial work, including runs whose primes
  // all fit in one wave and can never earn it back.
  const bool schedule_fits = numeric_dag_fits(plan.terms.size(), plan.n_rows());
  secred::FwdDag fwd_dag;
  std::atomic<int> dag_state{0}; // 0 none, 2 ready, 3 unavailable
  bool builder_started = false;
  std::thread builder;
  // The probe's fill sizes, handed to every worker store before its first point.
  std::size_t fill_terms = 0;
  std::size_t fill_rows = 0;
  if (first_missing > 0) {
    // The probe's samples are on disk; the consumer reads them in its turn.
    std::cout << "  " << secred::kFlowTag << " reconstruct: reusing " << first_missing
              << " prime checkpoint(s) for this spill\n"
              << std::flush;
  } else {
    secred::FwdStore probe_store;
    std::vector<std::uint64_t> probe_values;
    PrimeSlot &probe_slot = slots[0];
    const auto t0 = std::chrono::steady_clock::now();
    probe_slot.ok = solve_prime_numeric(
        plan, reconstruction_prime(0), bwd_chain, dep_targets, masters, master_index,
        probe_store, probe_values, probe_slot.xout, probe_slot.fail_target,
        probe_slot.fail_integral, &stop, &probe_slot.stats);
    probe_slot.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    save_prime(0, probe_slot);
    const std::uint64_t rss_after = process_used_memory();
    if (probe_slot.ok && rss_after > rss_before)
      measured = rss_after - rss_before;
    // The probe's store holds exactly what every worker's first point will fill, so
    // reserving those sizes removes the growth reallocations without estimating.
    if (probe_slot.ok) {
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

  // The workers start at the first prime without a checkpoint.  With no
  // checkpoints that is prime 1: prime 0 was just solved as the probe.
  next.store(first_missing > 1 ? first_missing : 1, std::memory_order_relaxed);
  if (first_missing == 0)
    slots[0].ready = true;
  std::vector<std::thread> workers;
  workers.reserve(nworkers);
  for (unsigned w = 0; w < nworkers; ++w) {
    workers.emplace_back([&]() {
      secred::FwdStore store;
      store.set_fill_reserve(fill_terms, fill_rows);
      std::vector<std::uint64_t> values;
      std::vector<std::uint64_t> dag_acc;
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
        const secred::FwdDag *replay_dag =
            dag_state.load(std::memory_order_acquire) == 2 ? &fwd_dag : nullptr;
        slot.replayed = replay_dag != nullptr;
        const auto t0 = std::chrono::steady_clock::now();
        slot.ok = solve_prime_numeric(plan, p, bwd_chain, dep_targets, masters,
                                      master_index, store, values, slot.xout,
                                      slot.fail_target, slot.fail_integral, &stop,
                                      &slot.stats, replay_dag,
                                      replay_dag != nullptr ? &dag_acc : nullptr);
        slot.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                    t0)
                           .count();

        if (stop.load(std::memory_order_relaxed))
          break;
        save_prime(static_cast<unsigned>(ip), slot);
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
  // CRT/rat_rec run after every prime; one pool created before the loop keeps
  // the same threads for all of them.  With one output there is nothing to
  // split, so the loop stays a plain serial call.
  std::unique_ptr<secred::ParallelPool> crt_pool;
  const auto for_each_output = [&](auto &&f) {
    if (nparsout == 0)
      return;
    if (nparsout == 1) {
      f(std::size_t(0));
      return;
    }
    if (!crt_pool)
      crt_pool.reset(new secred::ParallelPool(
          static_cast<unsigned>(std::min<std::size_t>(nthreads, nparsout))));
    crt_pool->blocks(nparsout, f);
  };
  ff::MPInt totmod, c1, c2;
  bool ok = false;
  secred::FwdStats fwd_total;
  std::size_t nprimes_consumed = 0;
  std::size_t nprimes_solved = 0;
  std::size_t nprimes_replayed = 0;

  for (std::size_t ip = 0; ip < nprime_max; ++ip) {
    PrimeSlot &slot = slots[ip];
    // A prime the previous attempt finished comes from its checkpoint.  A file
    // that stopped validating is solved here, because the workers started past
    // the checkpoint prefix and do not cover it.
    if (resume != nullptr && ip < first_missing && !slot.ready) {
      if (load_prime(static_cast<unsigned>(ip), slot)) {
        if (ip == 0)
          std::cout << "  " << secred::kFlowTag
                    << " reconstruct: prime 1 from its checkpoint\n"
                    << std::flush;
      } else {
        std::error_code ec;
        std::filesystem::remove(
            secred::prime_checkpoint_path(prime_dir, static_cast<unsigned>(ip)), ec);
        secred::FwdStore store;
        std::vector<std::uint64_t> values;
        std::vector<std::uint64_t> dag_acc;
        const secred::FwdDag *replay_dag =
            dag_state.load(std::memory_order_acquire) == 2 ? &fwd_dag : nullptr;
        slot.replayed = replay_dag != nullptr;
        const auto t0 = std::chrono::steady_clock::now();
        slot.ok = solve_prime_numeric(
            plan, reconstruction_prime(static_cast<unsigned>(ip)), bwd_chain,
            dep_targets, masters, master_index, store, values, slot.xout,
            slot.fail_target, slot.fail_integral, &stop, &slot.stats, replay_dag,
            replay_dag != nullptr ? &dag_acc : nullptr);
        slot.seconds = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t0)
                           .count();
        save_prime(static_cast<unsigned>(ip), slot);
        {
          std::lock_guard<std::mutex> lock(m);
          slot.ready = true;
        }
        cv.notify_all();
      }
    }
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
    if (!slots[ip].loaded)
      ++nprimes_solved;
    if (slots[ip].replayed)
      ++nprimes_replayed;
    ff::Mod mod(reconstruction_prime(static_cast<unsigned>(ip)));
    const ff::UInt *xout = slots[ip].xout.data();
    if (elim_seconds)
      *elim_seconds += slots[ip].seconds;

    const auto t_rest0 = std::chrono::steady_clock::now();
    bool converged = false;
    bool failed = false;
    if (ip == 0) {
      totmod = mod.n();
      for_each_output([&](unsigned j) {
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
      for_each_output([&](unsigned j) {
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
        for_each_output([&](unsigned j) {
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

    // One wave without convergence: more primes are coming, and a schedule built now
    // still has them to pay its build back on.  The builder takes no prime from the
    // pool, so the solvers keep their pipeline; a run whose primes fit in one wave
    // never starts it.
    if (schedule_fits && !builder_started && ip + 1 >= nworkers) {
      builder_started = true;
      builder = std::thread([&] {
        secred::FwdStore store;
        store.set_fill_reserve(fill_terms, fill_rows);
        std::vector<std::uint64_t> values;
        bool built = false;
        // The build needs a point with no vanishing coefficient; the first
        // reconstruction primes give one unless the plan's structure is special.
        for (unsigned k = 0; k < 8 && !stop.load(std::memory_order_relaxed); ++k) {
          built = build_numeric_schedule(plan, reconstruction_prime(k), fwd_dag, store,
                                         values, &stop);
          if (built)
            break;
        }
        dag_state.store(built ? 2 : 3,
                        built ? std::memory_order_release : std::memory_order_relaxed);
      });
    }

    std::vector<ff::UInt>().swap(slots[ip].xout);
  }

  stop.store(true, std::memory_order_relaxed);
  cv.notify_all();
  for (auto &t : workers)
    t.join();
  // The builder checks the flag between rows, so a schedule the run no longer needs
  // cannot hold this join for long.
  if (builder.joinable())
    builder.join();

  if (dag_state.load(std::memory_order_acquire) == 2) {
    std::cout << "  " << secred::kFlowTag << " reconstruct: a shared "
              << secred::format_memory(
                     static_cast<std::uint64_t>(fwd_dag.memory_bytes()))
              << " forward schedule (" << fwd_dag.rows.size() << " rows, "
              << fwd_dag.merges.size() << " merges) ";
    if (nprimes_replayed == 0)
      std::cout << "was built but no reconstructed prime replayed it\n";
    else
      std::cout << "replayed on " << nprimes_replayed << " of "
                << nprimes_consumed << " reconstructed prime(s)\n";
    std::cout << std::flush;
  }

  if (nprimes_solved != 0) {
    const double per = static_cast<double>(nprimes_solved);
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

GlobalReductionResult secred::reduce_global_finiteflow(
    const GlobalStageInput &input, std::uint64_t max_memory,
    std::uint64_t memory_reserve) {
  const std::string &raw_equations_path = input.raw_equations_path;
  const std::vector<RawIntegral> &all_integrals = input.all_integrals;
  const std::unordered_map<RawIntegral, unsigned> &integral_id = input.integral_id;
  const std::vector<RawIntegral> &targets = input.targets;
  const unsigned narrowing_success = input.narrowing_success;
  const unsigned narrowing_fail = input.narrowing_fail;
  const unsigned raw_equations_total = input.raw_equations_total;
  const std::size_t global_equations_count = input.global_equations_count;
  const std::string &masters_global_path = input.masters_global_path;
  const unsigned nthreads = input.nthreads;
  const unsigned max_primes = input.max_primes;
  const BasisVote &basis_vote = input.basis_vote;
  const secred::ResumeControl *resume = input.resume;

  GlobalReductionResult res;
  namespace ff = fflow;

  const unsigned nvars = all_integrals.size();
  const auto t0 = std::chrono::steady_clock::now();
  std::cout << "\n  " << secred::kFlowTag << " reading spill file, building system...\n";

  NumericEvalPlan plan;
  secred::SpillReadStats stats;
  std::string read_error;
  if (!secred::read_spill_numeric_plan(raw_equations_path, integral_id, plan, stats,
                                       read_error)) {
    std::cerr << "finiteflow global reduction: " << read_error << "\n";
    res.ok = false;
    return res;
  }

  const std::size_t neqs = plan.n_rows();
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

  const std::vector<unsigned> target_ids =
      global_target_ids(input, "finiteflow global reduction");
  if (target_ids.empty())
    return res;

  const uint64_t learn_prime = ff::BIG_UINT_PRIMES[0];

  secred::SubSolve learn;
  secred::FwdStats learn_stats;
  double learn_s = 0.0;
  std::vector<std::uint64_t> learn_values;
  {
    auto t_learn0 = std::chrono::steady_clock::now();
    plan.fill_values(learn_prime, learn_values);
    secred::solve_plan(plan, learn_values, learn_prime, learn, true, true, nullptr,
                       nullptr, &learn_stats);
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
    auto row_front = [&](std::size_t row) {
      return plan.eq_begin[row] == plan.eq_begin[row + 1]
                 ? 0u
                 : plan.terms[plan.eq_begin[row]].first;
    };
    std::cerr << "\n===== RAW equations (" << neqs << ") =====\n";
    for (std::size_t i = 0; i < plan.n_rows(); ++i) {
      std::cerr << "eq[" << i << "] front=" << row_front(i) << " :";
      for (std::size_t t = plan.eq_begin[i]; t < plan.eq_begin[i + 1]; ++t)
        std::cerr << " " << plan.terms[t].first << "("
                  << secred::rat_text(plan.coeffs[plan.terms[t].second]) << ")";
      std::cerr << "\n";
    }
    std::cerr << "\n===== forward reductions =====\n";
    for (const auto &kv : eq_index) {
      uint32_t w = kv.first, i = kv.second;
      std::cerr << "pivot " << w << " (eq " << i << ", raw front " << row_front(i)
                << "):";
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
      const NumericEvalPlan sub_plan = plan.subset(idx);
      secred::solve_plan(sub_plan, learn_values, learn_prime, sub, false, true,
                         nullptr, &dep_targets);
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
  plan.select_rows(needed_eq_idx);
  const auto tplan1 = std::chrono::steady_clock::now();
  std::cout << "  " << secred::kFlowTag << " reconstruction plan: " << plan.n_rows()
            << " rows, " << plan.terms.size() << " terms, " << plan.n_slots()
            << " distinct coefficient(s) ("
            << secred::format_duration(
                   std::chrono::duration<double>(tplan1 - tplan0).count())
            << ")\n"
            << std::flush;

  bool ok = reconstruct_ffred(plan, bwd_chain, dep_targets, masters,
                              master_index, nthreads, max_primes, max_memory,
                              memory_reserve, resume, resmp.get(), &elim_seconds);
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

std::vector<unsigned> secred::global_target_ids(const GlobalStageInput &input,
                                                const char *backend) {
  std::vector<unsigned> target_ids;
  target_ids.reserve(input.targets.size());
  for (const auto &t : input.targets) {
    auto it = input.integral_id.find(t);
    if (it != input.integral_id.end())
      target_ids.push_back(it->second);
  }
  if (target_ids.empty())
    std::cerr << backend << ": no targets\n";
  return target_ids;
}

GlobalReductionResult secred::run_global_stage(
    const GlobalStageInput &input, const std::vector<GiNaC::possymbol> &symbols,
    std::uint64_t max_memory, std::uint64_t memory_reserve) {
  if (global_engine_for(symbols.size()) == GlobalEngine::FireFlySymbolic)
    return reduce_global_firefly_symbolic(input, symbols);
  return reduce_global_finiteflow(input, max_memory, memory_reserve);
}
