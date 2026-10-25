#include "family.h"

#include <algorithm>
#include <bit>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>

// Print the settled run configuration.
void Reduce::print() const {
  std::cout << "\n----------------- Reduce Info ------------------\n"
            << "\n  Targets: " << _rawTargets.size()
            << "   Top sector: " << _top;

  std::cout << "\n\n  Non-trivial sectors: "
            << std::count_if(_sectors.begin(), _sectors.end(),
                             [](bool value) { return value; });

  std::cout << "\n\n------------------------------------------------\n"
            << std::endl;
}

// Derive the truncation and build the sector list.
void Reduce::prepare_sectors() {

  if (!_truncate) {
    // Truncation off (the default): the whole seed box is admitted, no rule is
    // derived and seed_level_offset is void.
    _truncation = SeedTruncation{};
    _truncation.level_offset = 0;
  } else {
    const unsigned level_offset =
        _probe_mode == ProbeMode::All
            ? 0
            : (_seed_level_offset_set ? _seed_level_offset
                                      : automatic_seed_level_offset(_rawTargets));
    _truncation = derive_seed_truncation(_rawTargets, _top, level_offset);
  }

  _lines = std::vector<bool>(_nprops, false);
  for (unsigned i = 0; i < _nprops; ++i)
    if (_top & (1 << i))
      _lines[i] = true;

  if (_ncut != 0) {
    unsigned cut = 0;
    for (unsigned i = 0; i < _cut.size(); ++i) {
      if (_cut[i] == 1)
        cut += 1 << i;
    }
    for (unsigned i = 0; i < _sectors.size(); ++i) {
      if ((cut & i) != cut)
        _sectors[i] = false;
    }
  }

  unsigned topSect = 0;
  for (auto sector : _targetSectors)
    topSect = std::max(topSect, sector);
  for (unsigned i = 0; i < _sectors.size(); ++i) {
    if (i > topSect)
      _sectors[i] = false;
  }

  unsigned nsec = std::count_if(_sectors.begin(), _sectors.end(),
                                [](bool value) { return value; });
  if (nsec == 0)
    process_finish("no non-trivial sectors");

  std::vector<unsigned> sectors(nsec);
  std::set<unsigned> trivialsectors;
  for (unsigned i = 0, j = nsec - 1; i < _sectors.size(); ++i)
    if (_sectors[i])
      sectors[j--] = i;
    else
      trivialsectors.insert(i);

  std::sort(sectors.begin(), sectors.end(), [](unsigned a, unsigned b) {
    unsigned alines = std::popcount(a);
    unsigned blines = std::popcount(b);
    if (alines > blines)
      return true;
    else if (alines == blines)
      return a > b;
    else
      return false;
  });

  _ctx.targets.clear();
  _ctx.uptargets.clear();
  _ctx.cutRequests.clear();
  for (unsigned id : sectors) {
    _ctx.targets[id];
    _ctx.uptargets[id];
    _ctx.cutRequests[id];
  }
  for (const auto &integral : _rawTargets) {
    _ctx.targets[integral.sector()].insert(integral);
  }

  namespace fs = std::filesystem;
  std::ofstream sectorfile;
  if (!_dry_run) {
    fs::create_directories(_cache_dir + "/sectors");
    sectorfile.open(_cache_dir + "/sectors/sectors");
  }

  _ctx.allMasters = std::vector<std::vector<RawIntegral>>(nsec);
  _ctx.totalSectors = nsec;
  _ctx.completedSectors = 0;
  _reduceSectors.clear();
  _reduceSectors.reserve(nsec);
  for (unsigned i = 0; i < sectors.size(); ++i) {
    if (!_dry_run)
      sectorfile << sectors[i] << ": " << std::popcount(sectors[i]) << std::endl;
    std::vector<bool> slines(_nprops, false);
    for (unsigned j = 0; j < _nprops; ++j)
      if (sectors[i] & (1 << j))
        slines[j] = true;
    _reduceSectors.push_back(Sector::make(i, sectors[i], _nprops, slines, trivialsectors));
  }

  {
    std::set<unsigned> reduced;
    for (const Sector &s : _reduceSectors)
      reduced.insert(s.id());
    _symmetries.restrictTo(reduced);
    _ctx.symmetries = &_symmetries;
  }
}
