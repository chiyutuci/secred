#pragma once

#include <string>
#include <utility>
#include <vector>

#include "ginac/ginac.h"
#include "yaml-cpp/yaml.h"

#include "symmetry.h"

// The kinematic part of an integral family: its symbols, propagators and the
// scalar-product / Symanzik polynomials derived from them.  RunContext hands
// the model to the stages as a const reference, which is the read-only
// interface they consume.  The members keep the names they had inside Family,
// so the split stays a move rather than a rewrite.
struct FamilyModel {
  std::string _name;

  GiNaC::ex _dimension;

  std::vector<GiNaC::possymbol> _internals;

  std::vector<GiNaC::possymbol> _externals;

  unsigned _nints = 0;

  unsigned _nexts = 0;

  std::vector<std::pair<GiNaC::possymbol, unsigned>> _invariants;

  std::vector<GiNaC::possymbol> _symbols;

  std::vector<GiNaC::ex> _propagators;

  std::vector<GiNaC::ex> _propMomenta;

  GiNaC::lst _spsRules;

  GiNaC::lst _spsFromProps;

  GiNaC::lst _one;

  unsigned _nprops = 0;

  std::vector<GiNaC::symbol> _symIndices;

  std::vector<GiNaC::symbol> _symProps;

  GiNaC::ex _uPoly;

  GiNaC::ex _fPoly;

  // Read the family block, register its symbols in `st`, and derive the
  // scalar-product rules and the index symbols the generators use.
  void parse(const YAML::Node &family_config, GiNaC::symtab &st);

  // Solve for the scalar-product substitution rules.
  void _compute_sps();

  // Compute the Symanzik polynomials.
  void _compute_symanzik();

  // Sectors with a non-trivial line mask below top.
  std::vector<bool> nontrivial_sectors(unsigned top) const;

  SymmetryTable::Inputs symmetry_inputs() const;

private:
  // Per-candidate GiNaC classification, used when the support classifier's
  // assumptions do not hold (non-polynomial terms, exponents above two, or
  // more than 16 propagators).
  std::vector<bool> nontrivial_sectors_ginac(unsigned top) const;
};
