
#pragma once

#include <fstream>
#include <string>
#include <vector>

namespace secred {

inline constexpr const char *kMastersSectorFile = "masters_sector";
inline constexpr const char *kMastersGlobalFile = "masters_global";

// Write one master per line.
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

}
