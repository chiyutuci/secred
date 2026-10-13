// Direct scanner over the global stage's raw-equation spill file.
//
// The spill is written by one function (`appendRawEquation`, sector.h:320-330) as
// "<RawIntegral>\t<coefficient>\n" per term, a blank line after each equation, and the
// coefficient text comes from one printer (`print_row_poly`, row_coeff.h) whose grammar
// is closed.  The global stage used to re-read that file -- 2.35 GB on F234 -- with
// std::getline plus a GiNaC parse per coefficient, and paid 231 s for it.
//
// This file maps the spill read-only and parses every line in place: no line copy, no
// per-line allocation, no GiNaC on the fast path.  Anything outside the strict grammar
// (an integral `RawIntegral::from_text` would accept but the strict parser refuses, a
// coefficient outside the printer's spelling) falls back to the original parser and is
// counted in SpillReadStats, so the fast path is never a silent reinterpretation.
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

// What one scan saw.  `terms` counts the (integral, coefficient) pairs kept;
// `skipped_unknown` the pairs whose integral is not in the id map; `fallbacks` the
// coefficients the strict grammar refused and the original GiNaC parser had to read.
struct SpillReadStats {
  std::uint64_t equations = 0;
  std::uint64_t terms = 0;
  std::uint64_t skipped_unknown = 0;
  std::uint64_t fallbacks = 0;
};

// Parse one coefficient span `[b, e)` of a spill line into `out`, normalised exactly as
// `make_sym_coeff` normalises GiNaC's numer()/denom(): integer coefficients, common
// denominator reduced, an empty denominator polynomial when it is 1, and
// `out.precompiled = true`.  False when the text is outside the grammar
// `print_row_poly` emits -- including a zero polynomial's spelling, duplicate
// monomials, and exponents outside 1..255 -- and the caller must use the GiNaC parser.
bool parse_spill_coeff(const char *b, const char *e,
                       const std::unordered_map<std::string, unsigned> &sym_index,
                       SymCoeff &out);

// Read `path` and append every equation whose integrals are all known to `system`.
// False and a message in `error` on a hard failure (unreadable file, a coefficient that
// does not fit in int64).  `stats.fallbacks` is the number of coefficients the strict
// grammar refused.
bool read_spill_numeric(const std::string &path,
                        const std::unordered_map<RawIntegral, unsigned> &integral_id,
                        std::vector<Eq> &system, SpillReadStats &stats,
                        std::string &error);

// Read `path` and append every equation whose integrals are all known to `system`, as
// (id, precompiled SymCoeff) in descending id, dropping zero coefficients exactly as the
// GiNaC reader did.  `symbols` are the rational function's parameters in index order:
// the reader derives the name -> index map, and builds a GiNaC parser from the same
// table for coefficients the strict grammar refuses (counted in `stats.fallbacks`).
bool read_spill_symbolic(const std::string &path,
                         const std::unordered_map<RawIntegral, unsigned> &integral_id,
                         const std::vector<GiNaC::possymbol> &symbols,
                         std::vector<SymEq> &system, SpillReadStats &stats,
                         std::string &error);

// Index of the equation in `path` whose largest known integral id is smallest, i.e. the
// equation with the simplest leading term (`select_dum_eq_from_file`'s rule).  Never
// looks at a coefficient; 0 when the file cannot be read, as before.
unsigned scan_spill_dum(const std::string &path,
                        const std::unordered_map<RawIntegral, unsigned> &integral_id);

}  // namespace secred
