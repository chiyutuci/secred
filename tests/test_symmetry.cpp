#include "test_runner.h"

#include "family.h"
#include "symmetry.h"
#include "symmetry_golden.h"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// The archived goldens (bench/symmetry/native, spec 4.8), generated into
// tests/symmetry_golden.h by tests/make_symmetry_golden.py.
// ---------------------------------------------------------------------------

// The archived goldens must be present and non-empty for every family: the later
// tasks compare against them, and a truncated header must fail loudly here rather
// than silently skipping a comparison.
TEST(symmetry_golden_header_is_complete) {
  const char *fams[] = {"box", "sunrise", "dbox"};
  for (const char *fam : fams) {
    REQUIRE(std::string(golden_canonica(fam)).size() > 100);
    REQUIRE(std::string(golden_entries(fam, false)).size() > 100);
    REQUIRE(std::string(golden_entries(fam, true)).size() > 100);
  }
  // spot check: the box relations golden's first block is the verified 4 -> 1 entry
  REQUIRE(std::string(golden_entries("box", false)).rfind("0\n0 4\n0 1\n", 0) == 0);
}

// ---------------------------------------------------------------------------
// Families for the symmetry tests.  They reproduce, in SecRed's own YAML, the
// kinematic configuration of the archived Kira goldens
// (bench/symmetry/native/<fam>/config/*.yaml): same propagator order (so line
// numbers and sector ids match), symbolic dimension d, and the mass at
// dimension 2 in Kira (the invariant `m2`) against dimension 1 here (SecRed's
// `m`), i.e. m2 = m^2 -- the only renaming the goldens' coefficient strings need.
//
// The families are deliberately NOT test_family.cpp's box_config: that one
// fixes invariants with invar_one and massless scalar products, which is a
// different kinematic point from the goldens'.
// ---------------------------------------------------------------------------

static YAML::Node symmetry_box_config(const std::string &cache_dir) {
  std::string yaml = R"(
settings:
  cache_dir: )" + cache_dir + R"(
  threads: 1

family:
  name: box
  dimension: d
  internals: [l1]
  externals: [p1, p2, p3]
  invariants:
    - [s, 2]
    - [t, 2]
    - [m, 1]
  sps_rules:
    - [p1, p1, m^2]
    - [p2, p2, m^2]
    - [p3, p3, m^2]
    - [p1, p2, t-m^2]
    - [p1, p3, s-m^2]
    - [p2, p3, -t-s-m^2]
  propagators:
    - [l1, m]
    - [l1+p1, 0]
    - [l1+p1+p2, m]
    - [l1+p1+p2+p3, 0]
)";
  return YAML::Load(yaml);
}

static Family make_box(const std::string &tag) {
  Family::symtab.clear();
  Family f(symmetry_box_config("/tmp/secred_tests/" + tag));
  f.init();
  return f;
}

// sunrise and dbox: the propagator lists are transcribed verbatim from the
// archived goldens' own Kira configs (bench/symmetry/native/<fam>_sym/config/),
// which are in turn example/sunrise.yaml and example/dbox.yaml with m2 = m^2 and
// the loop momenta renamed.  A reordered or reformulated list changes the
// canonical partition, so these are not free to be "cleaned up".
static YAML::Node symmetry_sunrise_config(const std::string &cache_dir) {
  std::string yaml = R"(
settings:
  cache_dir: )" + cache_dir + R"(
  threads: 1

family:
  name: sunrise
  dimension: d
  internals: [l1, l2]
  externals: [p1]
  invariants:
    - [s, 2]
    - [m, 1]
  sps_rules:
    - [p1, p1, s]
  propagators:
    - [l1, m]
    - [l2, m]
    - [p1-l1-l2, m]
    - [p1-l2, m]
    - [p1-l1, m]

targets:
  - [1, 1, 1, 1, 1]
)";
  return YAML::Load(yaml);
}

static Family make_sunrise(const std::string &cache_dir) {
  Family::symtab.clear();
  Family f(symmetry_sunrise_config(cache_dir));
  f.init();
  return f;
}

static YAML::Node symmetry_dbox_config(const std::string &cache_dir) {
  std::string yaml = R"(
settings:
  cache_dir: )" + cache_dir + R"(
  threads: 1

family:
  name: dbox
  dimension: d
  internals: [l1, l2]
  externals: [p1, p2, p3]
  invariants:
    - [s, 2]
    - [t, 2]
  sps_rules:
    - [p1, p1, 0]
    - [p2, p2, 0]
    - [p3, p3, 0]
    - [p1, p2, t]
    - [p2, p3, s]
    - [p1, p3, -t-s]
  propagators:
    - [l1-p1, 0]
    - [l1-p1-p2, 0]
    - [l1, 0]
    - [l1+l2, 0]
    - [l2+p1+p2, 0]
    - [l2, 0]
    - [l2+p1+p2+p3, 0]
    - [l1+p3, 0]
    - [l2+p1, 0]

targets:
  - [1, 1, 1, 1, 1, 1, 1, 0, 0]
)";
  return YAML::Load(yaml);
}

static Family make_dbox(const std::string &cache_dir) {
  Family::symtab.clear();
  Family f(symmetry_dbox_config(cache_dir));
  f.init();
  return f;
}

// Kira's coefficient strings spell m^2 as m2; the goldens are compared as
// expressions, so the rewrite bench/symmetry/native/README.md documents is
// applied before parsing.
static GiNaC::ex kira_expr(const std::string &s) {
  GiNaC::parser p(Family::symtab);
  return p(kira_to_ginac(s));
}

static void require_row(const PropPrescription &p, unsigned k,
                        const std::vector<std::string> &expected) {
  for (unsigned j = 0; j + 1 < expected.size(); ++j)
    REQUIRE((p.f(k, j) - kira_expr(expected[j])).expand().is_zero());
  REQUIRE((p.c(k, 0) - kira_expr(expected.back())).expand().is_zero());
}

// ---------------------------------------------------------------------------
// verify() against Kira's own correspondences, read off the golden's canonica
// permutation lists (ing[p0[g]] = p1[g]; see spec 4.8).
// ---------------------------------------------------------------------------

TEST(symmetry_verify_box_4_to_1) {
  Family f = make_box("sym_box_41");
  const SymmetryTable::Inputs in = f.symmetry_inputs();
  // canonica: sector 4 -> "2" (line 2), sector 1 -> "0" (line 0)
  auto p = SymmetryTable::verify(4, 1, {-1, -1, 0, -1}, in);
  REQUIRE(p.has_value());
  REQUIRE(!p->dot_only);
  REQUIRE_EQ(p->from, 4u);
  REQUIRE_EQ(p->to, 1u);
  // golden relations, first block
  require_row(*p, 0, {"2", "0", "-1", "0", "4*t"});
  require_row(*p, 1, {"1", "1", "-1", "0", "2*t"});
  require_row(*p, 2, {"1", "0", "0", "0", "0"});
  require_row(*p, 3, {"1", "0", "-1", "1", "4*m2+2*t"});
}

TEST(symmetry_verify_box_5_to_5) {
  Family f = make_box("sym_box_55");
  const SymmetryTable::Inputs in = f.symmetry_inputs();
  // canonica: sector 5 -> "0 2" then "2 0", so ing[0] = 2, ing[2] = 0
  auto p = SymmetryTable::verify(5, 5, {2, -1, 0, -1}, in);
  REQUIRE(p.has_value());
  REQUIRE(!p->dot_only);
  require_row(*p, 0, {"0", "0", "1", "0", "0"});
  require_row(*p, 1, {"1", "-1", "1", "0", "4*m2-2*t"});
  require_row(*p, 2, {"1", "0", "0", "0", "0"});
  require_row(*p, 3, {"1", "0", "1", "-1", "-2*t"});
}

TEST(symmetry_verify_box_10_to_10) {
  Family f = make_box("sym_box_1010");
  const SymmetryTable::Inputs in = f.symmetry_inputs();
  // canonica: sector 10 -> "1 3" then "3 1"
  auto p = SymmetryTable::verify(10, 10, {-1, 3, -1, 1}, in);
  REQUIRE(p.has_value());
  REQUIRE(!p->dot_only);
  require_row(*p, 0, {"-1", "1", "0", "1", "-4*m2+2*t+2*s"});
  require_row(*p, 1, {"0", "0", "0", "1", "0"});
  require_row(*p, 2, {"0", "1", "-1", "1", "2*t+2*s"});
  require_row(*p, 3, {"0", "1", "0", "0", "0"});
}

TEST(symmetry_verify_rejects_box_dot_only_entries) {
  Family f = make_box("sym_box_dots");
  const SymmetryTable::Inputs in = f.symmetry_inputs();
  // 6 -> 3: ing from 6's "2 1" and 3's "0 1"
  REQUIRE(!SymmetryTable::verify(6, 3, {-1, 1, 0, -1}, in).has_value());
  // 12 -> 3: ing from 12's "2 3"
  REQUIRE(!SymmetryTable::verify(12, 3, {-1, -1, 0, 1}, in).has_value());
  // 7 -> 7: ing from 7's "0 2 1" and "2 0 1"
  REQUIRE(!SymmetryTable::verify(7, 7, {2, 1, 0, -1}, in).has_value());
  // 14 -> 14: ing from 14's "2 1 3" and "2 3 1"
  REQUIRE(!SymmetryTable::verify(14, 14, {-1, 3, 2, 1}, in).has_value());
}

TEST(symmetry_dot_only_rows_match_kira) {
  Family f = make_box("sym_box_faked");
  const SymmetryTable::Inputs in = f.symmetry_inputs();
  // Kira's faked rows (ReadYamlFiles.cpp:1896-1901): the unit vector at ing[k]
  // for a covered line, the unit vector at the lowest line outside `to` for an
  // uncovered one, every constant zero.
  PropPrescription p = SymmetryTable::dot_only_entry(6, 3, {-1, 1, 0, -1}, in);
  REQUIRE(p.dot_only);
  require_row(p, 0, {"0", "0", "1", "0", "0"});
  require_row(p, 1, {"0", "1", "0", "0", "0"});
  require_row(p, 2, {"1", "0", "0", "0", "0"});
  require_row(p, 3, {"0", "0", "1", "0", "0"});

  PropPrescription q = SymmetryTable::dot_only_entry(14, 14, {-1, 3, 2, 1}, in);
  require_row(q, 0, {"1", "0", "0", "0", "0"});  // first hole of sector 14: line 0
  require_row(q, 1, {"0", "0", "0", "1", "0"});
  require_row(q, 2, {"0", "0", "1", "0", "0"});
  require_row(q, 3, {"0", "1", "0", "0", "0"});
}

// rows_ok is the golden-comparison helper the later tasks use; it must accept a
// row that agrees with Kira's string only after the m2 rewrite and GiNaC
// canonicalisation (different factor order, m^2 folded).
TEST(symmetry_rows_ok_compares_as_expressions) {
  Family f = make_box("sym_box_rowsok");
  const SymmetryTable::Inputs in = f.symmetry_inputs();
  auto p = SymmetryTable::verify(4, 1, {-1, -1, 0, -1}, in);
  REQUIRE(p.has_value());
  REQUIRE(p->rows_ok({{"2", "0", "-1", "0", "4*t"},
                      {"1", "1", "-1", "0", "2*t"},
                      {"1", "0", "0", "0", "0"},
                      {"1", "0", "-1", "1", "4*m2+2*t"}},
                     4, Family::symtab));
  // a wrong number of rows, a wrong number of columns, and a wrong value all fail
  REQUIRE(!p->rows_ok({{"2", "0", "-1", "0", "4*t"}}, 4, Family::symtab));
  REQUIRE(!p->rows_ok({{"2", "0", "-1", "0"},
                       {"1", "1", "-1", "0"},
                       {"1", "0", "0", "0"},
                       {"1", "0", "-1", "1"}},
                      4, Family::symtab));
  REQUIRE(!p->rows_ok({{"2", "0", "-1", "0", "5*t"},
                       {"1", "1", "-1", "0", "2*t"},
                       {"1", "0", "0", "0", "0"},
                       {"1", "0", "-1", "1", "4*m2+2*t"}},
                      4, Family::symtab));
}

TEST(symmetry_numeric_conversion_round_trips) {
  // The numeric path (ibps_sym == nullptr) needs the same entries as exact
  // rationals; a numeric family is one whose _symbols is empty.
  Family f = make_box("sym_box_numeric");
  const SymmetryTable::Inputs in = f.symmetry_inputs();
  REQUIRE(!in.numeric);  // s, t, m and d are all symbols here
  auto p = SymmetryTable::verify(4, 1, {-1, -1, 0, -1}, in);
  REQUIRE(p.has_value());
  REQUIRE(p->f_num.empty());  // not a numeric family: nothing to convert
}

// ---------------------------------------------------------------------------
// Golden parsers.  Both formats are blank-line separated blocks; see spec 4.8.
// ---------------------------------------------------------------------------

struct CanonicaBlock {
  unsigned sector = 0;
  std::string hash;
  unsigned nrows = 0, ncols = 0;
  std::vector<std::vector<int>> perms;
};

static std::vector<std::string> golden_blocks(const std::string &text) {
  std::vector<std::string> out;
  std::stringstream ss(text);
  std::string line, cur;
  while (std::getline(ss, line)) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) {
      if (!cur.empty())
        out.push_back(cur);
      cur.clear();
    } else {
      cur += line + "\n";
    }
  }
  if (!cur.empty())
    out.push_back(cur);
  return out;
}

static std::vector<CanonicaBlock> parse_canonica(const std::string &text) {
  std::vector<CanonicaBlock> out;
  for (const std::string &b : golden_blocks(text)) {
    std::stringstream ss(b);
    CanonicaBlock cb;
    ss >> cb.sector >> cb.hash >> cb.nrows >> cb.ncols;
    for (unsigned r = 0; r < cb.nrows; ++r) {
      std::vector<int> perm(cb.ncols);
      for (unsigned c = 0; c < cb.ncols; ++c)
        ss >> perm[c];
      cb.perms.push_back(perm);
    }
    out.push_back(cb);
  }
  return out;
}

// A sector's canonical class is the set of sectors sharing its hash -- within one
// golden file that is exactly Kira's partition (the hash is a function of the key).
static std::map<unsigned, std::set<unsigned>> golden_classes(const std::string &text) {
  std::map<std::string, std::set<unsigned>> by_hash;
  for (const CanonicaBlock &b : parse_canonica(text))
    by_hash[b.hash].insert(b.sector);
  std::map<unsigned, std::set<unsigned>> cls;
  for (const auto &[h, members] : by_hash)
    for (unsigned s : members)
      cls[s] = members;
  return cls;
}

// The partition test's three families are driven with `top` = the golden's own
// top_level_sectors (box 15, sunrise 31, dbox 511), which is the full line mask
// because every target the goldens use is positive on every line.
TEST(symmetry_canonica_matches_golden_partition) {
  const char *fams[] = {"box", "sunrise", "dbox"};
  const char *dirs[] = {"tc_sym_canon_box", "tc_sym_canon_sunrise", "tc_sym_canon_dbox"};
  for (unsigned i = 0; i < 3; ++i) {
    Family f = i == 0 ? make_box(dirs[i])
                      : i == 1 ? make_sunrise(dirs[i]) : make_dbox(dirs[i]);
    const unsigned top = (1u << f.nprops()) - 1;
    const SymmetryTable t =
        SymmetryTable::find(f.symmetry_inputs(), f.nontrivial_sectors(top), top);
    const std::vector<CanonicaBlock> golden = parse_canonica(golden_canonica(fams[i]));
    std::map<unsigned, std::set<unsigned>> want = golden_classes(golden_canonica(fams[i]));
    std::map<unsigned, std::set<unsigned>> got;
    for (const CanonicaBlock &b : golden)
      got[b.sector] = {};
    for (const CanonicaBlock &b : golden) {
      got[t.representative(b.sector)].insert(b.sector);
    }
    // The same sectors are considered at all.
    for (const CanonicaBlock &b : golden)
      REQUIRE(t.nrows(b.sector) > 0);
    for (unsigned s = 0; s < 1024; ++s)
      if (t.nrows(s) > 0)
        REQUIRE(want.count(s) == 1);
    // nrows per sector, and the same partition into classes.
    for (const CanonicaBlock &b : golden)
      REQUIRE_EQ(t.nrows(b.sector), b.nrows);
    for (const auto &[s, members] : want) {
      std::set<unsigned> ours;
      for (unsigned m : members)
        ours.insert(t.representative(m));
      REQUIRE_EQ(ours.size(), 1u);
      REQUIRE(*ours.begin() == *members.begin());  // min bitmask, both sides
      REQUIRE_EQ(t.representative(s), *members.begin());
    }
    // Determinism: a second run of the same input gives the same table.
    const SymmetryTable t2 =
        SymmetryTable::find(f.symmetry_inputs(), f.nontrivial_sectors(top), top);
    REQUIRE(t.canonical() == t2.canonical());
    for (const CanonicaBlock &b : golden)
      REQUIRE(t.permutations(b.sector) == t2.permutations(b.sector));
  }
}

TEST(symmetry_canonica_permutations_are_reported_not_asserted) {
  // Diagnostic only (spec 4.9.1): our canonical permutations differ from Kira's on
  // some sectors because our row order is deterministic where Kira's is not.  The
  // task report lists the sectors; the test only pins that our permutations are
  // permutations of the sector's lines.
  const char *fams[] = {"box", "sunrise", "dbox"};
  const char *dirs[] = {"tc_sym_perms_box", "tc_sym_perms_sunrise", "tc_sym_perms_dbox"};
  for (unsigned i = 0; i < 3; ++i) {
    Family f = i == 0 ? make_box(dirs[i])
                      : i == 1 ? make_sunrise(dirs[i]) : make_dbox(dirs[i]);
    const unsigned top = (1u << f.nprops()) - 1;
    const SymmetryTable t =
        SymmetryTable::find(f.symmetry_inputs(), f.nontrivial_sectors(top), top);
    unsigned differing = 0, compared = 0;
    for (const CanonicaBlock &b : parse_canonica(golden_canonica(fams[i]))) {
      ++compared;
      if (!(t.permutations(b.sector) == b.perms))
        ++differing;
      std::vector<int> ours = t.permutations(b.sector).front();
      std::sort(ours.begin(), ours.end());
      std::vector<int> lines;
      for (unsigned k = 0; k < f.nprops(); ++k)
        if (b.sector & (1u << k))
          lines.push_back(int(k));
      REQUIRE(ours == lines);
    }
    std::cout << "  canonica permutations differing from the golden (" << fams[i]
              << "): " << differing << " of " << compared << "\n";
  }
}
