#include "family_stages.h"

#include <atomic>
#include <bit>
#include <climits>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "BS_thread_pool.hpp"
#include "global_order.h"
#include "log_files.h"
#include "masters_file.h"
#include "resume.h"
#include "sector_resume.h"

namespace secred {

// Collect the targets and the per-run options for a reduction.
void SectorStage::prepare(const RunContext &ctx, const YAML::Node &config) {
  const FamilyModel &model = ctx.model;
  const RunSettings &settings = ctx.settings;
  Reduce &reduce = ctx.reduce;

  reduce._symbols = model._symbols;
  reduce._symIndices = model._symIndices;
  reduce._cut = settings._cut;
  reduce._ncut = settings._ncut;
  reduce._cache_dir = settings._cache_dir;
  reduce._dry_run = settings._dry_run;
  reduce._ctx.maxMemory = settings._max_memory;
  reduce._ctx.memoryReserve = settings._memory_reserve;

  std::cout << "\n #0.3   Collecting target integrals...\n";
  if (!config["targets"])
    throw std::runtime_error("reduce targets not found");
  reduce._rawTargets = config["targets"].as<std::vector<RawIntegral>>();

  reduce._nprops = model._nprops;
  reduce._seed_level_offset = settings._seed_level_offset;
  reduce._seed_level_offset_set = settings._seed_level_offset_set;
  reduce._probe_mode = settings._probe_mode;
  reduce._truncate = settings._truncate;
  reduce._mandatory_list = settings._mandatory_list;
  std::vector<bool> hasLine(model._nprops, false);
  for (const auto &integral : reduce._rawTargets) {
    if (integral.size() != model._nprops) {
      std::stringstream ss;
      ss << integral;
      throw std::runtime_error("target integral " + ss.str() + " is invalid");
    }
    for (unsigned i = 0; i < model._nprops; ++i) {
      if (integral[i] > 0 && !hasLine[i])
        hasLine[i] = true;
    }
    unsigned tarSector = integral.sector();
    reduce._targetSectors.insert(tarSector);
  }
  for (unsigned i = 0; i < model._nprops; ++i)
    if (hasLine[i])
      reduce._top |= 1 << i;

  if (config["preferred"] && !config["preferred"].IsNull()) {
    for (const auto &integral :
         config["preferred"].as<std::vector<RawIntegral>>()) {
      if (integral.size() != model._nprops) {
        std::stringstream ss;
        ss << integral;
        throw std::runtime_error("preferred integral " + ss.str() +
                                 " is invalid");
      }
      reduce._preferred.insert(integral);
    }
  }
  std::cout << "\n #0.3   Collecting target integrals "
               "finished.\n";

  std::cout << "\n #0.4   Searching trivial sectors...\n";
  search_trivial_sectors(ctx);
  std::cout << "\n #0.4   Searching trivial sectors "
               "finished.\n";

  if (settings._symmetry) {
    std::cout << "\n #0.4   Searching sector symmetries...\n";
    reduce._symmetries =
        SymmetryTable::find(model.symmetry_inputs(), reduce._sectors, reduce._top);
    if (settings._parallel_mode == "layer") {

      reduce._symmetries.dropCross();
      std::cout << " #0.4   parallel_mode layer: cross-sector symmetry relations "
                   "are dropped, self relations are kept.\n";
    }

    reduce._symmetries.try_to_numeric_all();
    std::cout << "\n #0.4   Searching sector symmetries finished.\n";
  }

  if (!model._symbols.empty()) {
    reduce._rowTable.build_all(ctx.generator._ibpM);
    reduce._ctx.rowTable = &reduce._rowTable;
  }

  reduce.prepare_sectors();
}

// One sector pass, with the schedule the settings select.
void SectorStage::run_pass(const RunContext &ctx,
                           const std::vector<bool> &replayed) {
  if (ctx.settings._parallel_mode == "fine")
    run_fine(ctx, replayed);
  else
    run_layer(ctx, replayed);
}

// One pass, sectors scheduled layer by layer by sector width.
void SectorStage::run_layer(const RunContext &ctx,
                            const std::vector<bool> &replayed) {
  const FamilyModel &model = ctx.model;
  const RunSettings &settings = ctx.settings;
  const FamilyGenerator &generator = ctx.generator;
  Reduce &reduce = ctx.reduce;

  BS::thread_pool pool(settings._sthreads);
  SectorConfig scfg{settings._cache_dir,     settings._subsector_filter_n,
                    settings._ordering_mode, settings._sector_debug,
                    &reduce._preferred,      &reduce._truncation,
                    reduce._probe_mode,      reduce._mandatory_list};

  unsigned current_layer = UINT_MAX;
  for (unsigned i = 0; i < reduce._reduceSectors.size(); ++i) {
    auto &sector = reduce._reduceSectors[i];
    unsigned layer = std::popcount(sector.id());
    if (layer != current_layer) {
      if (current_layer != UINT_MAX)
        pool.wait_for_tasks();
      current_layer = layer;
    }

    if (replayed[i])
      continue;

    if (model._symbols.size() == 0)
      pool.push_task([&sector, ibps = generator._ibpFF,
                      ibps_mod = generator._ibpFFmod, scfg,
                      &rctx = reduce._ctx]() mutable {
        sector.run_reduce_ff_n(ibps, ibps_mod, nullptr, scfg, rctx);
      });
    else
      pool.push_task([&sector, ibps = generator._ibpFF,
                      ibps_mod = generator._ibpFFmod,
                      ibps_sym = &generator._ibpM, scfg,
                      &rctx = reduce._ctx]() mutable {
        sector.run_reduce_ff_n(ibps, ibps_mod, ibps_sym, scfg, rctx);
      });
  }
  pool.wait_for_tasks();
}

// One pass, all sectors submitted independently.
void SectorStage::run_fine(const RunContext &ctx,
                           const std::vector<bool> &replayed) {
  const FamilyModel &model = ctx.model;
  const RunSettings &settings = ctx.settings;
  const FamilyGenerator &generator = ctx.generator;
  Reduce &reduce = ctx.reduce;

  unsigned nsec = reduce._reduceSectors.size();

  std::unordered_map<unsigned, unsigned> id_to_idx;
  id_to_idx.reserve(nsec);
  for (unsigned i = 0; i < nsec; ++i)
    id_to_idx[reduce._reduceSectors[i].id()] = i;

  std::vector<std::vector<unsigned>> children(nsec);
  for (unsigned j = 0; j < nsec; ++j) {
    unsigned jid = reduce._reduceSectors[j].id();
    for (unsigned k = 0; k < model._nprops; ++k) {
      if (jid & (1u << k)) continue;
      auto it = id_to_idx.find(jid | (1u << k));
      if (it != id_to_idx.end())
        children[it->second].push_back(j);
    }
  }

  for (unsigned j = 0; j < nsec; ++j) {
    unsigned sid = reduce._reduceSectors[j].id();
    unsigned rep = reduce._symmetries.representative(sid);
    if (rep == sid)
      continue;
    auto it = id_to_idx.find(rep);
    if (it == id_to_idx.end())
      continue;
    children[j].push_back(it->second);
  }

  // Only live parents gate a live child: a replayed parent's state is already
  // in the context.
  std::vector<std::atomic<unsigned>> pending(nsec);
  for (unsigned i = 0; i < nsec; ++i)
    pending[i].store(0, std::memory_order_relaxed);
  unsigned live = 0;
  for (unsigned i = 0; i < nsec; ++i) {
    if (replayed[i])
      continue;
    ++live;
    for (unsigned child : children[i])
      if (!replayed[child])
        pending[child].fetch_add(1, std::memory_order_relaxed);
  }

  SectorConfig scfg{settings._cache_dir,     settings._subsector_filter_n,
                    settings._ordering_mode, settings._sector_debug,
                    &reduce._preferred,      &reduce._truncation,
                    reduce._probe_mode,      reduce._mandatory_list};

  std::atomic<unsigned> done_count{0};
  std::mutex done_mutex;
  std::condition_variable done_cv;

  BS::thread_pool pool(settings._sthreads);

  std::function<void(unsigned)> run_sector = [&](unsigned i) {
    auto &sector = reduce._reduceSectors[i];

    if (model._symbols.empty())
      sector.run_reduce_ff_n(generator._ibpFF, generator._ibpFFmod, nullptr,
                             scfg, reduce._ctx);
    else
      sector.run_reduce_ff_n(generator._ibpFF, generator._ibpFFmod,
                             &generator._ibpM, scfg, reduce._ctx);

    for (unsigned child : children[i]) {
      if (replayed[child])
        continue;
      if (pending[child].fetch_sub(1, std::memory_order_acq_rel) == 1)
        pool.push_task([child, &run_sector] { run_sector(child); });
    }
    if (done_count.fetch_add(1, std::memory_order_acq_rel) + 1 == live)
      done_cv.notify_one();
  };

  for (unsigned i = 0; i < nsec; ++i)
    if (!replayed[i] && pending[i].load(std::memory_order_relaxed) == 0)
      pool.push_task([i, &run_sector] { run_sector(i); });

  if (live != 0) {
    std::unique_lock<std::mutex> lock(done_mutex);
    done_cv.wait(lock, [&] { return done_count.load() == live; });
  }

  pool.wait_for_tasks();
}

// Store the nontrivial sector list for the run.
void SectorStage::search_trivial_sectors(const RunContext &ctx) {
  ctx.reduce._sectors =
      ctx.model.nontrivial_sectors(ctx.reduce._top);
}

// Build and solve the global system, then vote on the settled basis.
GlobalReductionResult GlobalStage::run(const RunContext &ctx,
                                       bool allow_basis_vote) {
  const FamilyModel &model = ctx.model;
  const RunSettings &settings = ctx.settings;
  Reduce &reduce = ctx.reduce;
  auto &rctx = reduce._ctx;

  BasisVote basis_vote;
  if (allow_basis_vote)
    basis_vote = [&reduce](const std::vector<RawIntegral> &masters) {
      if (seed_dots_refinement(reduce._truncation, masters) > 0)
        return true;
      return !seed_request_refinement(masters, reduce._ctx.cutRequests).empty();
    };

  std::vector<RawIntegral> all_integrals;
  rctx.finishSpill(all_integrals);
  rctx.releaseSpillTables();

  // The completed files are what the global stage's checkpoints are keyed by.
  // On a resumed run the sector replay reproduces them from the records.
  if (rctx.resume != nullptr) {
    std::string error;
    if (!secred::hash_file(rctx.rawEquationsPath, rctx.resume->spill_hash, &error))
      throw std::runtime_error(
          "cannot hash the raw-equation spill for resume: " + error);
  }

  secred::order_global_integrals(all_integrals, settings._ordering_mode,
                                 &reduce._preferred);

  std::unordered_map<RawIntegral, unsigned> integral_id;
  integral_id.reserve(all_integrals.size());
  for (unsigned i = 0; i < all_integrals.size(); ++i)
    integral_id[all_integrals[i]] = i;

  const GlobalEngine engine = global_engine_for(model._symbols.size());
  const char *tool = global_engine_name(engine);
  const std::string mastersGlobalPath =
      settings._cache_dir + "/" + secred::kMastersGlobalFile;

  GlobalReductionResult result;
  {

    secred::reset_firefly_log();
    struct CollectFireflyLog {
      const std::string &cache_dir;
      ~CollectFireflyLog() { secred::collect_firefly_log(cache_dir); }
    } collect_log{settings._cache_dir};

    const secred::GlobalStageInput stage_input{
        .raw_equations_path = rctx.rawEquationsPath,
        .all_integrals = all_integrals,
        .integral_id = integral_id,
        .targets = reduce._rawTargets,
        .narrowing_success = rctx.narrowingSuccess.load(),
        .narrowing_fail = rctx.narrowingFail.load(),
        .raw_equations_total = rctx.rawEquationsTotal.load(),
        .global_equations_count = rctx.globalEquationsCount.load(),
        .masters_global_path = mastersGlobalPath,
        .nthreads = settings._gthreads,
        .max_primes = settings._max_primes,
        .basis_vote = basis_vote,
        .resume = rctx.resume,
    };
    result = secred::run_global_stage(stage_input, model._symbols,
                                      settings._max_memory,
                                      settings._memory_reserve);
    std::filesystem::remove(rctx.rawEquationsPath);

    secred::require_global_reduction(result, tool);

    if (!result.reconstruction_skipped)
      secred::write_reductions_file(settings._cache_dir + "/reductions", result);
  }
  return result;
}

} // namespace secred
