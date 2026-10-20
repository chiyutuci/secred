
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <ginac/ginac.h>

#include "ffred_solver.h"
#include "sector.h"
#include "sym_coeff.h"

namespace secred {

// Spill-file read counters.
struct SpillReadStats {
  std::uint64_t equations = 0;
  std::uint64_t terms = 0;
  std::uint64_t skipped_unknown = 0;
  std::uint64_t fallbacks = 0;
};

// Parse one spill-file coefficient.
bool parse_spill_coeff(const char *b, const char *e,
                       const std::unordered_map<std::string, unsigned> &sym_index,
                       SymCoeff &out);

// Read a numeric spill file.
bool read_spill_numeric(const std::string &path,
                        const std::unordered_map<RawIntegral, unsigned> &integral_id,
                        std::vector<Eq> &system, SpillReadStats &stats,
                        std::string &error);

// Read a symbolic spill file.
bool read_spill_symbolic(const std::string &path,
                         const std::unordered_map<RawIntegral, unsigned> &integral_id,
                         const std::vector<GiNaC::possymbol> &symbols,
                         std::vector<SymEq> &system, SpillReadStats &stats,
                         std::string &error);

// Read a symbolically interned spill file.
bool read_spill_interned(const std::string &path,
                         const std::unordered_map<RawIntegral, unsigned> &integral_id,
                         const std::vector<GiNaC::possymbol> &symbols,
                         SymCoeffPool &pool, SpillReadStats &stats, std::string &error);

}
