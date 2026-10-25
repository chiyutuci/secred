#pragma once

#include <vector>

#include "yaml-cpp/yaml.h"

#include "family.h"
#include "family_generator.h"
#include "family_model.h"
#include "globalreduce.h"
#include "run_settings.h"

namespace secred {

// What both pipeline stages consume: the read-only model, settings and
// generated rows, plus the run's mutable state (targets, sectors, statistics
// and the global context) in Reduce.  Passing this reference bundle is the
// stages' whole interface -- no stage reaches through a friend declaration.
struct RunContext {
  const FamilyModel &model;
  const RunSettings &settings;
  const FamilyGenerator &generator;
  Reduce &reduce;
};

// The sector phase: turn the targets into a sector list, then run the sectors.
class SectorStage {
public:
  // Collect the targets, search the trivial sectors and the symmetries, build
  // the row table and prepare the sector list for the run.
  static void prepare(const RunContext &ctx, const YAML::Node &config);

  // One sector pass.  `replayed` marks the sectors a resume restored, whose
  // equations are already in the context.
  static void run_pass(const RunContext &ctx,
                       const std::vector<bool> &replayed);

private:
  static void search_trivial_sectors(const RunContext &ctx);
  static void run_layer(const RunContext &ctx,
                        const std::vector<bool> &replayed);
  static void run_fine(const RunContext &ctx,
                       const std::vector<bool> &replayed);
};

// The global phase: assemble the spilled system, reduce it to the settled
// basis and write the reductions.  `allow_basis_vote` lets the caller stop
// before reconstruction when the settled basis would need another pass.
class GlobalStage {
public:
  static GlobalReductionResult run(const RunContext &ctx, bool allow_basis_vote);
};

} // namespace secred
