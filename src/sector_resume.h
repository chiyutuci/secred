#pragma once

// Replaying a pass's completed sectors from their checkpoint records.
//
// A record is reusable when the state the sector saw when it started still
// matches: its seed targets and the uptargets its finished parents emitted.
// Records that do not match -- because a parent will have to run again, or
// because the file is damaged -- are left for the normal scheduler; the
// sectors that do match are applied here, equations included, so the pass
// starts again from everything that had already been computed.

#include <filesystem>
#include <map>
#include <set>
#include <vector>

#include "sector.h"
#include "sector_checkpoint.h"

namespace secred {

struct SectorReplayPlan {
  // Indexed like the sector list handed to plan_sector_replay().
  std::vector<bool> replay;
  std::vector<SectorCheckpointFile> records;
  std::vector<std::filesystem::path> paths;

  // Sectors that will be replayed, not batches.
  unsigned replayed = 0;
};

// Decide which sectors can be replayed.  `sector_ids` is in scheduling order;
// `targets` is the pass's initial target map.  Sectors run in order, so a
// record is only accepted when the state simulated from the records accepted
// so far (their emitted uptargets) matches what it recorded seeing.
SectorReplayPlan plan_sector_replay(
    const std::filesystem::path &pass_dir,
    const std::vector<unsigned> &sector_ids,
    const std::map<unsigned, std::set<RawIntegral>> &targets);

// Apply the replayed records' state (masters, emitted uptargets, cut requests,
// statistics, timing) and write their equation batches to the spill in the
// recorded global commit order.  Returns the number of batches replayed.
unsigned apply_sector_replay(const SectorReplayPlan &plan, ReductionContext &ctx);

} // namespace secred
