#include "sector_resume.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace secred {

namespace {

std::vector<RawIntegral> sorted_targets(
    const std::map<unsigned, std::set<RawIntegral>> &targets, unsigned id) {
  const auto it = targets.find(id);
  if (it == targets.end())
    return {};
  return {it->second.begin(), it->second.end()};
}

std::vector<RawIntegral> sorted_uptargets(
    const std::map<unsigned, std::set<RawIntegral>> &uptargets, unsigned id) {
  const auto it = uptargets.find(id);
  if (it == uptargets.end())
    return {};
  return {it->second.begin(), it->second.end()};
}

} // namespace

SectorReplayPlan plan_sector_replay(
    const std::filesystem::path &pass_dir,
    const std::vector<unsigned> &sector_ids,
    const std::map<unsigned, std::set<RawIntegral>> &targets) {
  SectorReplayPlan plan;
  plan.replay.assign(sector_ids.size(), false);
  plan.records.resize(sector_ids.size());
  plan.paths.resize(sector_ids.size());

  std::map<unsigned, std::set<RawIntegral>> simulated;
  for (std::size_t i = 0; i < sector_ids.size(); ++i) {
    const unsigned id = sector_ids[i];
    const std::filesystem::path path = sector_checkpoint_path(pass_dir, id);
    std::string error;
    SectorCheckpointFile file;
    if (!std::filesystem::is_regular_file(path) ||
        !read_sector_checkpoint_file(path, file, &error))
      continue;

    const std::vector<RawIntegral> expected_targets = sorted_targets(targets, id);
    const std::vector<RawIntegral> expected_uptargets =
        sorted_uptargets(simulated, id);
    // A sector with neither inputs never writes a record; one that exists is
    // not something this run produced, so rerun the sector instead of trusting
    // it.
    if (expected_targets.empty() && expected_uptargets.empty())
      continue;
    if (file.meta.id != id ||
        file.meta.index != static_cast<unsigned>(i) ||
        file.meta.input_targets != expected_targets ||
        file.meta.input_uptargets != expected_uptargets)
      continue;

    plan.replay[i] = true;
    plan.paths[i] = path;
    plan.records[i] = std::move(file);
    ++plan.replayed;
    for (const RawIntegral &uptarget : plan.records[i].meta.emitted_uptargets)
      simulated[uptarget.sector()].insert(uptarget);
  }
  return plan;
}

unsigned apply_sector_replay(const SectorReplayPlan &plan,
                             ReductionContext &ctx) {
  struct PendingBatch {
    std::uint64_t seq = 0;
    std::size_t sector = 0;
    const SectorCheckpointBatchRef *batch = nullptr;
  };
  std::vector<PendingBatch> batches;

  for (std::size_t i = 0; i < plan.replay.size(); ++i) {
    if (!plan.replay[i])
      continue;
    const SectorCheckpoint &meta = plan.records[i].meta;

    std::vector<RawIntegral> &masters = ctx.allMasters[meta.index];
    masters.insert(masters.end(), meta.masters.begin(), meta.masters.end());
    for (const RawIntegral &uptarget : meta.emitted_uptargets)
      ctx.uptargets[uptarget.sector()].insert(uptarget);
    {
      std::lock_guard<std::mutex> lock(ctx.cutRequestsMutex);
      std::unordered_set<RawIntegral> requests(meta.cut_requests.begin(),
                                               meta.cut_requests.end());
      ctx.cutRequests[meta.id] = std::move(requests);
    }

    ctx.narrowingSuccess.fetch_add(meta.narrowing_success,
                                   std::memory_order_relaxed);
    ctx.narrowingFail.fetch_add(meta.narrowing_fail,
                                std::memory_order_relaxed);
    ctx.truncatedSectors.fetch_add(meta.truncated, std::memory_order_relaxed);
    ctx.truncationLifted.fetch_add(meta.lifted, std::memory_order_relaxed);
    ctx.boxSpillRaised.fetch_add(meta.box_raised, std::memory_order_relaxed);
    ctx.boxSpillExtraDepth.fetch_add(meta.box_depth, std::memory_order_relaxed);
    ctx.boxSpillExtraRank.fetch_add(meta.box_rank, std::memory_order_relaxed);
    ctx.probeScopeSkipped.fetch_add(meta.scope_skipped,
                                    std::memory_order_relaxed);
    ctx.rawEquationsTotal.fetch_add(meta.raw_equations,
                                    std::memory_order_relaxed);
    ctx.recordTiming(meta.timing);

    for (const SectorCheckpointBatchRef &batch : plan.records[i].batches)
      batches.push_back(PendingBatch{batch.seq, i, &batch});
  }

  // The recorded commit order is the spill's order.  Ties can only come from
  // records written by different attempts at the same pass; keeping the
  // sector-list order for them makes the replay deterministic.
  std::stable_sort(batches.begin(), batches.end(),
                   [](const PendingBatch &a, const PendingBatch &b) {
                     return a.seq < b.seq;
                   });

  std::uint64_t max_seq = 0;
  std::vector<GlobalEquation> equations;
  for (const PendingBatch &pending : batches) {
    std::string error;
    if (!read_sector_checkpoint_batch(plan.paths[pending.sector], *pending.batch,
                                      equations, &error))
      throw std::runtime_error("cannot replay a sector checkpoint batch: " +
                               error);
    ctx.appendRawEquations(equations);
    max_seq = std::max(max_seq, pending.seq);
  }
  if (!batches.empty())
    ctx.spillSeq = max_seq + 1;
  return static_cast<unsigned>(batches.size());
}

} // namespace secred
