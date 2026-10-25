#pragma once

#include <vector>

#include "family_model.h"
#include "sector.h"

// The IBP/LI rows for one run, in the three coefficient representations the
// sector and global stages consume.  Generation reads the model and writes
// only these vectors, so the run loop can free the sector-phase rows between
// passes without touching the family.  Member names keep their Family spelling.
struct FamilyGenerator {
  std::vector<IBPProto> _ibp;

  std::vector<IBPProtoFF> _ibpFF;

  std::vector<IBPProtoFFmod> _ibpFFmod;
  std::vector<IBPProtoM> _ibpM;

  // Generate every representation from the model.
  void generate(const FamilyModel &model);

  // Drop the symbolic rows before a pass that regenerates them.
  void clear_ibp();

  // Return the sector- and global-phase rows once the sector phase is done.
  void release_sector_rows();

  void _generate_ibp(const FamilyModel &model);

  // Generate the Lorentz-invariance rows.
  void _generate_li(const FamilyModel &model);

  // Generate the numeric-path IBP rows.
  void _generate_ibp_ff_n(const FamilyModel &model);

  // Generate the symbolic-path IBP rows.
  void _generate_ibp_ff_m(const FamilyModel &model);
};
