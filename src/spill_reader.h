// Direct scanner over the global stage's raw-equation spill file.
//
// The spill is written by `ReductionContext` (sector.h: `openSpill`, `appendRawEquation`
// / `appendRawEquations`, `finishSpill`).  Two formats are read:
//
//   * the binary frame every current run writes -- "SRSP" + version, one frame per
//     equation, the interned integral and coefficient tables, a footer.  A term is
//     8 bytes and a coefficient is stored once, so F234's 2.35 GB text spill becomes
//     8 B per term plus its tables;
//   * the text format the earlier versions wrote ("<RawIntegral>\t<coefficient>\n" per
//     term, a blank line between equations), kept because the frozen bench spills under
//     bench/ are in it and because it is the reference the binary reader is
//     differentially tested against (tests/test_spill_reader.cpp).
//
// Either way the file is mapped read-only and parsed in place: no line copy, no
// per-line allocation, no GiNaC on the fast path.  The coefficient text comes from one
// printer (`print_row_poly`, row_coeff.h) whose grammar is closed; anything outside it
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
// does not fit in int64, a binary file that is truncated or from another version).
// `stats.fallbacks` is the number of coefficients the strict grammar refused -- per
// occurrence, the coefficient table's entries being what the binary format repeats
// least.
bool read_spill_numeric(const std::string &path,
                        const std::unordered_map<RawIntegral, unsigned> &integral_id,
                        std::vector<Eq> &system, SpillReadStats &stats,
                        std::string &error);

// Read `path` and append every equation whose integrals are all known to `system`, as
// (id, precompiled SymCoeff) in descending id, dropping zero coefficients exactly as the
// GiNaC reader did.  `symbols` are the rational function's parameters in index order:
// the reader derives the name -> index map, and builds a GiNaC parser from the same
// table for coefficients the strict grammar refuses (counted in `stats.fallbacks`).
//
// This is the materialising reader: every term of every kept equation owns a `SymCoeff`,
// which at F234's size is what makes the read cost gigabytes.  The production path reads
// through `read_spill_interned` instead; this one stays as the reference it is measured
// and verified against.
bool read_spill_symbolic(const std::string &path,
                         const std::unordered_map<RawIntegral, unsigned> &integral_id,
                         const std::vector<GiNaC::possymbol> &symbols,
                         std::vector<SymEq> &system, SpillReadStats &stats,
                         std::string &error);

// The production reader: the same scan, but each coefficient is interned as it is parsed
// instead of being stored per term, so the system exists only as the pool's term table
// (8 B per term) plus one object per *distinct* coefficient.  The equations, their term
// order and their slots are exactly what `build_sym_coeff_pool` would produce from the
// materialised system -- that equality is what `intern_sym_system` in the tests pins.
//
// On a binary spill the coefficient table is parsed once per entry (a `fallback` GiNaC
// parse included) and a row takes pointers into that cache, so `stats.fallbacks` counts
// the *entries* the grammar refused rather than their occurrences, as the text reader
// above it does.
bool read_spill_interned(const std::string &path,
                         const std::unordered_map<RawIntegral, unsigned> &integral_id,
                         const std::vector<GiNaC::possymbol> &symbols,
                         SymCoeffPool &pool, SpillReadStats &stats, std::string &error);

}  // namespace secred
