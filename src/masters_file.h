// The two master-basis files of a run, both under the cache directory:
//
//   masters_sector  the sector-local basis (the union of the per-sector masters,
//                   before the global reduction);
//   masters_global  the basis the global reduction settles on.
//
// They answer different questions — the sector-phase union is not the final basis —
// so they are two files under two names rather than one reused name.  Both hold one
// integral per line, so either can be diffed against the other.
//
// Header-only so tests pin the names and the on-disk format.

#pragma once

#include <fstream>
#include <string>
#include <vector>

namespace secred {

inline constexpr const char *kMastersSectorFile = "masters_sector";
inline constexpr const char *kMastersGlobalFile = "masters_global";

// One integral per line, newline-terminated; the file is truncated, so a shorter
// basis never leaves the tail of a longer one behind.  False when the file cannot be
// opened.
template <typename Integral>
bool write_masters_file(const std::string &path,
                        const std::vector<Integral> &masters) {
  std::ofstream out(path, std::ios::trunc);
  if (!out.is_open())
    return false;
  for (const Integral &m : masters)
    out << m << "\n";
  return out.good();
}

} // namespace secred
