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

// A symbolic entry for a hand-built prescription: to_rational refuses anything
// that is not a rational numeric, and a plain symbol is the cheapest such entry
// to write.  Which symbol it is does not matter.
static GiNaC::ex symbolic_entry() { return GiNaC::symbol("d"); }

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

// ---------------------------------------------------------------------------
// Correspondences (spec 4.9.2): every one of Kira's 432 entries, driven with
// Kira's *own* correspondence -- recovered from the golden's canonica
// permutations as ing[p0[g]] = p1[g] (spec 3.1, trivial_sym.cpp:1331-1333) --
// and compared coefficient by coefficient as expressions.  Driving the
// comparison with the golden's pairing rather than ours is what keeps it
// independent of the row-order tie-break that makes our permutations differ
// from Kira's on some sectors (spec 3.1).
// ---------------------------------------------------------------------------

// One golden row: nprops coefficient cells then the constant.  An alias, not a
// struct, because that is exactly the shape rows_ok() compares against (the
// brief's struct does not convert to its own rows_ok parameter).
using EntryRow = std::vector<std::string>;

struct GoldenEntry {
  int dots = 0;
  unsigned from = 0, to = 0;
  std::vector<EntryRow> rows;
};

static std::vector<GoldenEntry> parse_entries(const std::string &text, unsigned nprops) {
  std::vector<GoldenEntry> out;
  for (const std::string &b : golden_blocks(text)) {
    std::stringstream ss(b);
    GoldenEntry e;
    unsigned itc = 0, topo = 0;
    ss >> e.dots >> itc >> e.from >> topo >> e.to;
    EntryRow r;
    std::string cell;
    while (ss >> cell) {
      r.push_back(cell);
      if (r.size() == nprops + 1) {
        e.rows.push_back(r);
        r.clear();
      }
    }
    if (!r.empty())
      e.rows.push_back(r);
    out.push_back(e);
  }
  return out;
}

// The comparison driver.  For a cross entry the source's first canonical
// permutation is paired with the target's g-th for every g; for a self entry
// with g >= 1 (Kira's symmetry_finder is called with permIt from 1 for it == itt,
// trivial_sym.cpp:1755-1764).  The first g that reproduces Kira's rows is
// accepted: the golden file's own entry order is a sorted one, so it does not
// carry permIt.  An entry no g reproduces is reported, not silently skipped.
static unsigned compare_family(const Family &f, const char *fam, unsigned nprops,
                               std::vector<std::string> &problems) {
  const SymmetryTable::Inputs in = f.symmetry_inputs();
  const std::vector<CanonicaBlock> canon = parse_canonica(golden_canonica(fam));
  std::map<unsigned, const CanonicaBlock *> by_sector;
  for (const CanonicaBlock &b : canon)
    by_sector[b.sector] = &b;

  const std::size_t before = problems.size();
  unsigned checked = 0, dots = 0, dots_we_verified = 0, dots_with_a_map = 0;
  unsigned first_g = 0;
  std::map<unsigned, unsigned> g_hist;
  std::vector<std::string> stronger;
  for (int selfs = 0; selfs <= 1; ++selfs) {
    for (const GoldenEntry &e : parse_entries(golden_entries(fam, selfs != 0), nprops)) {
      const CanonicaBlock *src = by_sector.at(e.from);
      const CanonicaBlock *dst = by_sector.at(e.to);
      // Every candidate pairing Kira itself considered: the target's g-th
      // canonical permutation against the source's first (g >= 1 for a self
      // entry, since permIt starts at 1 when it == itt).
      bool matched = false, map_where_matched = false, any_map = false;
      unsigned used = 0;
      for (unsigned g = selfs ? 1 : 0; g < dst->perms.size(); ++g) {
        std::vector<int> ing(nprops, -1);
        for (unsigned h = 0; h < src->perms.front().size(); ++h)
          ing[src->perms.front()[h]] = dst->perms[g][h];
        auto p = SymmetryTable::verify(e.from, e.to, ing, in);
        if (p)
          any_map = true;
        const PropPrescription cand =
            p ? *p : SymmetryTable::dot_only_entry(e.from, e.to, ing, in);
        if (cand.rows_ok(e.rows, nprops, Family::symtab) && !matched) {
          matched = true;
          map_where_matched = !cand.dot_only;
          used = g;
        }
      }
      if (!matched)
        problems.push_back(std::string(fam) + " " + std::to_string(e.from) + " -> " +
                           std::to_string(e.to));
      else {
        ++g_hist[used];
        if (used == (selfs ? 1u : 0u))
          ++first_g;
      }
      if (e.dots == 1) {
        ++dots;
        // Spec 4.9.2: Kira wrote the relation off as dots-only and our search is
        // strictly stronger -- either we find a map at all for a correspondence
        // Kira dotted, or (the stronger case) our map's rows are themselves what
        // Kira printed.
        if (any_map) {
          ++dots_with_a_map;
          if (stronger.size() < 10)
            stronger.push_back(std::to_string(e.from) + "->" + std::to_string(e.to));
        }
        if (map_where_matched)
          ++dots_we_verified;
      }
      ++checked;
    }
  }
  const unsigned unmatched = unsigned(problems.size() - before);
  std::cout << "  " << fam << ": " << checked << " entries compared, " << unmatched
            << " unmatched, " << first_g << " at the first candidate pairing, "
            << dots << " symDOTS=1 of which " << dots_with_a_map
            << " admit a map for us (first few:";
  for (const std::string &s : stronger)
    std::cout << ' ' << s;
  std::cout << ") and " << dots_we_verified << " whose Kira rows our own map reproduces"
            << "; pairings used:";
  for (const auto &[g, n] : g_hist)
    std::cout << " g" << g << "x" << n;
  std::cout << "\n";
  return checked;
}

TEST(symmetry_correspondences_match_every_golden_entry) {
  std::vector<std::string> problems;
  unsigned total = 0;
  {
    Family f = make_box("tc_sym_corr_box");
    total += compare_family(f, "box", 4, problems);
  }
  {
    Family f = make_sunrise("tc_sym_corr_sunrise");
    total += compare_family(f, "sunrise", 5, problems);
  }
  {
    Family f = make_dbox("tc_sym_corr_dbox");
    total += compare_family(f, "dbox", 9, problems);
  }
  std::cout << "  golden entries checked: " << total << "\n";
  for (const std::string &p : problems)
    std::cerr << "  unmatched golden entry: " << p << "\n";
  REQUIRE_EQ(total, 432u);  // 148 cross + 284 self (spec 4.1.2)
  REQUIRE(problems.empty());
}

// ---------------------------------------------------------------------------
// Emission (spec 4.5, 4.9.3): the shipped table's entry set, against Kira's own
// counts.  The counts are Kira's, read off the goldens (relations = cross,
// symmetries = self), and they follow from the emission rule alone -- a
// non-representative emits exactly one cross entry, a representative with
// nrows > 1 emits nrows - 1 self entries, nothing else emits.
// ---------------------------------------------------------------------------

TEST(symmetry_emission_rule_matches_the_golden) {
  struct Case {
    const char *fam;
    unsigned cross, selfs;
    std::vector<unsigned> cross_sources, self_sources;
  };
  const Case cases[] = {
      {"box", 3, 4, {4, 6, 12}, {5, 7, 10, 14}},
      {"sunrise", 17, 23, {}, {}},
      {"dbox", 128, 257, {}, {}},
  };
  Family box = make_box("tc_sym_emit_box");
  Family sunrise = make_sunrise("tc_sym_emit_sunrise");
  Family dbox = make_dbox("tc_sym_emit_dbox");
  for (const Case &c : cases) {
    const Family &f = std::string(c.fam) == "box"      ? box
                      : std::string(c.fam) == "sunrise" ? sunrise
                                                        : dbox;
    const unsigned top = (1u << f.nprops()) - 1;
    const SymmetryTable t =
        SymmetryTable::find(f.symmetry_inputs(), f.nontrivial_sectors(top), top);
    unsigned cross = 0, selfs = 0;
    for (unsigned s = 0; s < (1u << f.nprops()); ++s) {
      for (const PropPrescription &p : t.bySource(s)) {
        REQUIRE_EQ(p.from, s);
        if (p.from == p.to) {
          ++selfs;
          // A self source reports itself: no out-edge.
          REQUIRE_EQ(t.representative(s), s);
        } else {
          ++cross;
          // Exactly one cross entry per source, pointing at the representative.
          REQUIRE_EQ(t.representative(s), p.to);
          REQUIRE_EQ(t.bySource(s).size(), 1u);
        }
      }
      // A self-only sector reports itself; a sector with no entries reports itself.
      if (t.bySource(s).empty())
        REQUIRE_EQ(t.representative(s), s);
    }
    REQUIRE_EQ(cross, c.cross);
    REQUIRE_EQ(selfs, c.selfs);
    if (std::string(c.fam) == "box") {
      std::vector<unsigned> srcs, self_srcs;
      for (unsigned s = 0; s < 16; ++s)
        for (const PropPrescription &p : t.bySource(s))
          (p.from == p.to ? self_srcs : srcs).push_back(s);
      std::sort(srcs.begin(), srcs.end());
      std::sort(self_srcs.begin(), self_srcs.end());
      REQUIRE(srcs == c.cross_sources);
      REQUIRE(self_srcs == c.self_sources);
    }
  }
}

// The dot-only split of spec 8.1, through Kira's correspondences: 4 -> 1 is
// verified against 6 -> 3 and 12 -> 3 are not, 5 -> 5 and 10 -> 10 are verified
// against 7 -> 7 and 14 -> 14 are not.  All six of the box's emitted entries
// therefore appear, four of them as Kira's faked rows.
TEST(symmetry_box_dot_only_split_matches_the_golden) {
  Family f = make_box("tc_sym_split_box");
  const SymmetryTable::Inputs in = f.symmetry_inputs();
  REQUIRE(SymmetryTable::verify(4, 1, {-1, -1, 0, -1}, in).has_value());
  REQUIRE(!SymmetryTable::verify(6, 3, {-1, 1, 0, -1}, in).has_value());
  REQUIRE(!SymmetryTable::verify(12, 3, {-1, -1, 0, 1}, in).has_value());
  REQUIRE(SymmetryTable::verify(5, 5, {2, -1, 0, -1}, in).has_value());
  REQUIRE(SymmetryTable::verify(10, 10, {-1, 3, -1, 1}, in).has_value());
  REQUIRE(!SymmetryTable::verify(7, 7, {2, 1, 0, -1}, in).has_value());
  REQUIRE(!SymmetryTable::verify(14, 14, {-1, 3, 2, 1}, in).has_value());

  const unsigned top = (1u << f.nprops()) - 1;
  const SymmetryTable t =
      SymmetryTable::find(f.symmetry_inputs(), f.nontrivial_sectors(top), top);
  unsigned dots = 0, maps = 0;
  for (unsigned s = 0; s < 16; ++s)
    for (const PropPrescription &p : t.bySource(s))
      (p.dot_only ? dots : maps)++;
  REQUIRE_EQ(dots, 4u);
  REQUIRE_EQ(maps, 3u);
}

// Review Focus 1 and 2 (spec 4.7): a filter -- the cut mask or the top filter --
// can drop the representative of a class, and a prescription left pointing at a
// sector that is not reduced is a scheduler deadlock in stage B.  After
// restrictTo no surviving prescription may have an endpoint outside the reduced
// set, and a source left with none must report itself.
TEST(symmetry_restrict_to_prunes_and_self_reports) {
  Family f = make_box("tc_sym_restrict_box");
  const unsigned top = (1u << f.nprops()) - 1;
  SymmetryTable t =
      SymmetryTable::find(f.symmetry_inputs(), f.nontrivial_sectors(top), top);
  std::set<unsigned> reduced;
  for (unsigned s = 0; s < 16; ++s)
    if (t.nrows(s) > 0 && s != 1)  // sector 1 is the representative of {1, 4}
      reduced.insert(s);
  t.restrictTo(reduced);
  REQUIRE(t.bySource(4).empty());       // its only entry pointed at the dropped 1
  REQUIRE_EQ(t.representative(4), 4u);  // ... so it reports itself
  unsigned surviving = 0;
  for (unsigned s = 0; s < 16; ++s)
    for (const PropPrescription &p : t.bySource(s)) {
      ++surviving;
      REQUIRE(reduced.count(p.from) == 1);
      REQUIRE(reduced.count(p.to) == 1);
    }
  REQUIRE(surviving > 0);  // the invariant above is not vacuously true
  // 5 -> 5 is untouched: both endpoints are in the reduced set (and survive
  // because 5's own self entry has no partner to lose).
  REQUIRE_EQ(t.bySource(5).size(), 1u);
  REQUIRE_EQ(t.representative(5), 5u);

  // dropCross (spec 5.5, `layer`) keeps the self entries and nothing else.
  SymmetryTable u =
      SymmetryTable::find(f.symmetry_inputs(), f.nontrivial_sectors(top), top);
  u.dropCross();
  for (unsigned s = 0; s < 16; ++s)
    for (const PropPrescription &p : u.bySource(s)) {
      REQUIRE_EQ(p.from, p.to);
      REQUIRE_EQ(u.representative(s), s);
    }
  REQUIRE_EQ(u.bySource(6).size(), 0u);  // 6 was a cross source
  REQUIRE_EQ(u.bySource(5).size(), 1u);  // 5 was a self source
}

// ---------------------------------------------------------------------------
// The relation expansion (spec 5.1).  The hand prescriptions are written out so
// the algebra can be checked by hand.  The convention throughout is
// I(a) = prod_k D_k^{-a_k}: multiplying the integrand by D'_j therefore
// *subtracts* one from a'_j, which is what makes the numerator row of the
// spec's table read "subtract" and not "add".
//
//   hand_prescription():  D_0 = D'_0,   D_1 = D'_0 + D'_1 + 4
//
// Line 1's row is a combination, so line 1 can never be a denominator -- which
// is exactly the configuration Kira rejects (relations.cpp:514-521).
// ---------------------------------------------------------------------------

static PropPrescription hand_prescription(unsigned from, unsigned to, bool dot_only) {
  PropPrescription p;
  p.from = from;
  p.to = to;
  p.ing = {0, 1};
  p.dot_only = dot_only;
  p.f = GiNaC::matrix(2, 2);
  p.f(0, 0) = 1;
  p.f(1, 0) = 1;
  p.f(1, 1) = 1;
  p.c = GiNaC::matrix(2, 1);
  p.c(1, 0) = 4;
  return p;
}

// Rows given as plain numbers, for the shapes hand_prescription() does not have.
static PropPrescription rows_prescription(unsigned from, unsigned to, bool dot_only,
                                          const std::vector<std::vector<int>> &f,
                                          const std::vector<int> &c) {
  PropPrescription p;
  p.from = from;
  p.to = to;
  p.ing = {0, 1};
  p.dot_only = dot_only;
  p.f = GiNaC::matrix(2, 2);
  p.c = GiNaC::matrix(2, 1);
  for (unsigned i = 0; i < 2; ++i) {
    for (unsigned j = 0; j < 2; ++j)
      p.f(i, j) = f[i][j];
    p.c(i, 0) = c[i];
  }
  return p;
}

static std::vector<std::vector<GiNaC::ex>> ex_rows(const PropPrescription &p) {
  std::vector<std::vector<GiNaC::ex>> f(p.f.rows(), std::vector<GiNaC::ex>(p.f.cols()));
  for (unsigned i = 0; i < p.f.rows(); ++i)
    for (unsigned j = 0; j < p.f.cols(); ++j)
      f[i][j] = p.f(i, j);
  return f;
}

static std::vector<GiNaC::ex> ex_col(const PropPrescription &p) {
  std::vector<GiNaC::ex> c(p.c.rows());
  for (unsigned i = 0; i < p.c.rows(); ++i)
    c[i] = p.c(i, 0);
  return c;
}

// The brace form RawIntegral({1, 1}) is ambiguous (vector<int> and vector<bool>
// are both constructible from it), so the tests spell the type out.
static RawIntegral seed_of(std::initializer_list<int> il) {
  return RawIntegral(std::vector<int>(il));
}

static bool never_trivial(unsigned) { return false; }

static std::map<std::vector<int>, GiNaC::ex>
ex_terms(const std::vector<SymTerm<GiNaC::ex>> &out) {
  std::map<std::vector<int>, GiNaC::ex> m;
  for (const SymTerm<GiNaC::ex> &t : out)
    m[t.powers] = t.coeff;
  return m;
}

static std::map<std::vector<int>, rational> rat_terms(const std::vector<SymTerm<rational>> &out) {
  std::map<std::vector<int>, rational> m;
  for (const SymTerm<rational> &t : out)
    m[t.powers] = t.coeff;
  return m;
}

// D_0 = D'_1, D_1 = D'_0: a pure line swap.  I(2,1) = D_0^-2 D_1^-1
// = D'_1^-2 D'_0^-1, i.e. the image index vector is the seed's with the two
// lines exchanged, and the coefficient stays 1.
TEST(symmetry_expansion_denominators_are_a_permutation) {
  PropPrescription p = rows_prescription(3, 0, false, {{0, 1}, {1, 0}}, {0, 0});
  std::vector<SymTerm<GiNaC::ex>> out;
  REQUIRE(expand_prescription<GiNaC::ex>(p, ex_rows(p), ex_col(p), seed_of({2, 1}),
                                         never_trivial, out));
  REQUIRE_EQ(out.size(), 1u);
  REQUIRE(out[0].powers == std::vector<int>({1, 2}));
  REQUIRE(out[0].coeff.is_equal(1));
}

// D_0 = 2 D'_0: a single-term denominator row whose scalar is not 1, so the
// coefficient picks up 2^-e -- I(2,0) = D_0^-2 = 2^-2 D'_0^-2 = (1/4) I(2,0).
TEST(symmetry_expansion_denominator_scalar_multiplies_the_coefficient) {
  PropPrescription p = rows_prescription(3, 0, false, {{2, 0}, {0, 1}}, {0, 0});
  std::vector<SymTerm<GiNaC::ex>> out;
  REQUIRE(expand_prescription<GiNaC::ex>(p, ex_rows(p), ex_col(p), seed_of({2, 0}),
                                         never_trivial, out));
  REQUIRE_EQ(out.size(), 1u);
  REQUIRE(out[0].powers == std::vector<int>({2, 0}));
  REQUIRE(out[0].coeff.is_equal(GiNaC::ex(1) / 4));
}

// A denominator row that is a linear combination is a hard error, not a
// recovery path: the detector already cannot emit such a prescription
// (print_relations drops any mapping whose in-sector line has two non-zero
// entries, ReadYamlFiles.cpp:1955-1971), so the implementation asserts it.
// Seed (1,1) makes line 1 a denominator, and line 1's row is D'_0 + D'_1 + 4.
TEST(symmetry_expansion_denominator_combination_throws) {
  PropPrescription p = hand_prescription(3, 0, false);
  std::vector<SymTerm<GiNaC::ex>> out;
  REQUIRE_THROWS_AS(expand_prescription<GiNaC::ex>(p, ex_rows(p), ex_col(p), seed_of({1, 1}),
                                                   never_trivial, out),
                    std::runtime_error);
}

// I(1,-1) = D_0^-1 D_1 = D'_0^-1 (D'_0 + D'_1 + 4)
//         = 1 + D'_0^-1 D'_1 + 4 D'_0^-1
//         = I(0,0) + I(1,-1) + 4 I(1,0).
// The I(0,0) term is the pinch: the numerator's D'_0 cancels the denominator's,
// so line 0 leaves the target sector.
TEST(symmetry_expansion_numerator_cancels_a_denominator) {
  PropPrescription p = hand_prescription(3, 0, false);
  std::vector<SymTerm<GiNaC::ex>> out;
  REQUIRE(expand_prescription<GiNaC::ex>(p, ex_rows(p), ex_col(p), seed_of({1, -1}),
                                         never_trivial, out));
  std::map<std::vector<int>, GiNaC::ex> got = ex_terms(out);
  REQUIRE_EQ(out.size(), 3u);
  REQUIRE((got[{0, 0}].is_equal(1)));
  REQUIRE((got[{1, -1}].is_equal(1)));
  REQUIRE((got[{1, 0}].is_equal(4)));
}

// The same expansion with the pinched sector declared trivial: the I(0,0) term
// is a scaleless integral and drops, leaving the other two.  A trivial sector
// that is *not* the pinched one changes nothing -- the drop is on the term whose
// line leaves the target sector, never on the seed's own sector.
TEST(symmetry_expansion_pinch_drops_a_trivial_sector) {
  PropPrescription p = hand_prescription(3, 0, false);
  const auto sector_zero_trivial = [](unsigned sector) { return sector == 0; };
  std::vector<SymTerm<GiNaC::ex>> out;
  REQUIRE(expand_prescription<GiNaC::ex>(p, ex_rows(p), ex_col(p), seed_of({1, -1}),
                                         sector_zero_trivial, out));
  std::map<std::vector<int>, GiNaC::ex> got = ex_terms(out);
  REQUIRE_EQ(out.size(), 2u);
  REQUIRE((got.count({0, 0}) == 0));
  REQUIRE((got[{1, -1}].is_equal(1)));
  REQUIRE((got[{1, 0}].is_equal(4)));

  const auto sector_one_trivial = [](unsigned sector) { return sector == 1; };
  std::vector<SymTerm<GiNaC::ex>> out2;
  REQUIRE(expand_prescription<GiNaC::ex>(p, ex_rows(p), ex_col(p), seed_of({1, -1}),
                                         sector_one_trivial, out2));
  REQUIRE_EQ(out2.size(), 3u);
}

// dot_only applies to numerator-free seeds only (spec 5.1, pyred
// relations.cpp:481-483).  The rows are Kira's faked ones for a sector-1 source:
// line 0 maps to itself, and line 1 -- which the correspondence does not cover
// -- carries the unit vector at line 0, the lowest line outside the target.
TEST(symmetry_expansion_dot_only_skips_numerator_seeds) {
  PropPrescription p = rows_prescription(1, 0, true, {{1, 0}, {1, 0}}, {0, 0});
  std::vector<SymTerm<GiNaC::ex>> out;
  REQUIRE(expand_prescription<GiNaC::ex>(p, ex_rows(p), ex_col(p), seed_of({1, 0}),
                                         never_trivial, out));
  REQUIRE_EQ(out.size(), 1u);  // the numerator-free seed does expand
  REQUIRE(out[0].powers == std::vector<int>({1, 0}));
  REQUIRE(out[0].coeff.is_equal(1));

  std::vector<SymTerm<GiNaC::ex>> out2;
  REQUIRE(!expand_prescription<GiNaC::ex>(p, ex_rows(p), ex_col(p), seed_of({1, -1}),
                                           never_trivial, out2));
  REQUIRE(out2.empty());  // the refusal leaves nothing behind

  // The same seed under a verified prescription does expand.  With line 1's row
  // genuinely equal to line 0, the numerator on line 1 cancels line 0's
  // denominator and the only term left is I(0,0).
  PropPrescription v = rows_prescription(1, 0, false, {{1, 0}, {1, 0}}, {0, 0});
  std::vector<SymTerm<GiNaC::ex>> out3;
  REQUIRE(expand_prescription<GiNaC::ex>(v, ex_rows(v), ex_col(v), seed_of({1, -1}),
                                         never_trivial, out3));
  REQUIRE_EQ(out3.size(), 1u);
  REQUIRE(out3[0].powers == std::vector<int>({0, 0}));
  REQUIRE(out3[0].coeff.is_equal(1));
}

// The numeric path: the same expansion over FLINT rationals, no symbols.
//   I(1,-1) with D_1 = D'_0 + D'_1 + 1/2 gives I(0,0) + I(1,-1) + (1/2) I(1,0),
//   and the scalar denominator D_0 = 2 D'_0 sends I(2,0) to (1/4) I(2,0).
TEST(symmetry_expansion_numeric_path) {
  PropPrescription p;
  p.from = 3;
  p.to = 1;
  p.ing = {0, 1};
  p.f_num = {{rational(1), rational(0)}, {rational(1), rational(1)}};
  p.c_num = {rational(0), rational("1/2")};
  std::vector<SymTerm<rational>> out;
  REQUIRE(expand_prescription<rational>(p, p.f_num, p.c_num, seed_of({1, -1}),
                                        never_trivial, out));
  std::map<std::vector<int>, rational> got = rat_terms(out);
  REQUIRE_EQ(out.size(), 3u);
  REQUIRE((got[{0, 0}] == rational(1)));
  REQUIRE((got[{1, -1}] == rational(1)));
  REQUIRE((got[{1, 0}] == rational("1/2")));

  PropPrescription q;
  q.from = 3;
  q.to = 1;
  q.ing = {0, 1};
  q.f_num = {{rational(2), rational(0)}, {rational(0), rational(1)}};
  q.c_num = {rational(0), rational(0)};
  std::vector<SymTerm<rational>> out2;
  REQUIRE(expand_prescription<rational>(q, q.f_num, q.c_num, seed_of({2, 0}),
                                        never_trivial, out2));
  REQUIRE_EQ(out2.size(), 1u);
  REQUIRE(out2[0].powers == std::vector<int>({2, 0}));
  REQUIRE(out2[0].coeff == rational("1/4"));
}

// The expansion against Kira's own data: a dot_only entry's stored unit-vector
// rows must reproduce the correspondence `ing` says they encode.  The source
// sector's own seed -- one power on each of its lines, nothing outside -- is
// numerator-free, so every dot_only prescription applies, and each denominator
// line k must land its power on line ing[k] with coefficient 1.
TEST(symmetry_expansion_reproduces_the_dot_only_correspondence) {
  Family f = make_box("expansion_dots");
  const unsigned top = (1u << f.nprops()) - 1;
  SymmetryTable t = SymmetryTable::find(f.symmetry_inputs(), f.nontrivial_sectors(top), top);

  unsigned checked = 0;
  for (unsigned s = 0; s <= top; ++s)
    for (const PropPrescription &p : t.bySource(s)) {
      if (!p.dot_only)
        continue;
      ++checked;
      std::vector<int> seed(f.nprops(), 0);
      for (unsigned k = 0; k < f.nprops(); ++k)
        if (p.from & (1u << k))
          seed[k] = 1;
      std::vector<int> want(f.nprops(), 0);
      for (unsigned k = 0; k < f.nprops(); ++k)
        if (p.from & (1u << k)) {
          REQUIRE(p.ing[k] >= 0);
          want[unsigned(p.ing[k])] = 1;
        }
      std::vector<SymTerm<GiNaC::ex>> out;
      REQUIRE(expand_prescription<GiNaC::ex>(p, ex_rows(p), ex_col(p),
                                             RawIntegral(std::vector<int>(seed)), never_trivial,
                                             out));
      REQUIRE_EQ(out.size(), 1u);
      REQUIRE(out[0].powers == want);
      REQUIRE(out[0].coeff.is_equal(1));
    }
  REQUIRE_EQ(checked, 4u);  // box's four dot_only entries, so this is not vacuous
}

// ---------------------------------------------------------------------------
// Rational rows for every family (spec 5.1).  A prescription's rows are numbers
// whenever the momentum map and the masses allow it -- all of box's hand-built
// ones are -- so the rational expansion is available even when the family as a
// whole is symbolic.
// ---------------------------------------------------------------------------

TEST(symmetry_try_to_numeric_fills_the_rows_it_can) {
  PropPrescription p = hand_prescription(3, 1, false);
  REQUIRE(!p.has_numeric_rows);
  REQUIRE(p.try_to_numeric());
  REQUIRE(p.has_numeric_rows);
  REQUIRE_EQ(p.f_num.size(), 2u);
  REQUIRE(p.f_num[0][0] == rational(1));
  REQUIRE(p.f_num[1][1] == rational(1));
  REQUIRE(p.c_num[1] == rational(4));

  // A symbolic entry must report the miss instead of throwing, and must leave
  // the flag off.
  PropPrescription q = hand_prescription(3, 1, false);
  q.f(0, 0) = symbolic_entry(); // a symbolic entry: no rational rows exist here
  REQUIRE(!q.try_to_numeric());
  REQUIRE(!q.has_numeric_rows);

  // The throwing entry point keeps naming the coefficient it refused.
  PropPrescription r = hand_prescription(3, 1, false);
  r.f(0, 0) = symbolic_entry();
  REQUIRE_THROWS(r.to_numeric());
  REQUIRE(!r.has_numeric_rows);
}

// The table-level pass: every prescription whose entries are numbers gets its
// rational rows before any thread starts, and a symbolic one keeps the GiNaC
// path without a throw.  Box is the mixed case -- its dot-only entries are unit
// rows of 0s and 1s with a zero constant, i.e. numbers, while a relation that
// moves a massive line carries t -- and the mix is the whole reason the flag is
// per prescription rather than per family (spec 5.1).
//
// find() itself must not convert a symbolic family: no flag is set before the
// pass runs, because the detector's two to_numeric() calls sit behind
// `if (in.numeric)`.
TEST(symmetry_try_to_numeric_all_converts_every_prescription_it_can) {
  Family f = make_box("try_numeric_box");
  const unsigned top = (1u << f.nprops()) - 1;
  SymmetryTable t =
      SymmetryTable::find(f.symmetry_inputs(), f.nontrivial_sectors(top), top);
  unsigned total = 0;
  for (unsigned s = 0; s <= top; ++s)
    for (const PropPrescription &p : t.bySource(s)) {
      ++total;
      REQUIRE(!p.has_numeric_rows);
    }
  REQUIRE(total > 0);

  t.try_to_numeric_all();

  unsigned converted = 0;
  for (unsigned s = 0; s <= top; ++s)
    for (const PropPrescription &p : t.bySource(s)) {
      if (!p.has_numeric_rows)
        continue;
      ++converted;
      // The flag means the rows are there, and they are the right shape.
      REQUIRE_EQ(p.f_num.size(), p.f.rows());
      REQUIRE_EQ(p.c_num.size(), p.c.rows());
    }
  REQUIRE(converted > 0);     // the dot-only relations are unit rows
  REQUIRE(converted < total); // the t-dependent ones are not

  // A relation that carries t can never convert, and the flag says so.
  const SymmetryTable::Inputs in = f.symmetry_inputs();
  auto p41 = SymmetryTable::verify(4, 1, {-1, -1, 0, -1}, in);
  REQUIRE(p41.has_value());
  REQUIRE(!p41->try_to_numeric());
  REQUIRE(!p41->has_numeric_rows);
}

// ---------------------------------------------------------------------------
// The pre-change expansion, kept as the oracle for the fast path (spec 5.2) and
// for the container swap (spec 5.3).  A verbatim copy of the map-based body,
// with its two ring primitives: the point is that it shares no code with the
// implementation under test.
// ---------------------------------------------------------------------------

template <typename Coeff> static Coeff ref_one();

template <> GiNaC::ex ref_one<GiNaC::ex>() { return GiNaC::ex(1); }

template <> rational ref_one<rational>() { return rational(1); }

template <typename Coeff> static Coeff ref_pow(const Coeff &x, unsigned e) {
  Coeff r = ref_one<Coeff>();
  for (unsigned i = 0; i < e; ++i)
    r = r * x;
  return r;
}

template <> GiNaC::ex ref_pow<GiNaC::ex>(const GiNaC::ex &x, unsigned e) {
  return GiNaC::pow(x, e);
}

template <typename Coeff>
static bool reference_expand(const PropPrescription &p, const std::vector<std::vector<Coeff>> &f,
                             const std::vector<Coeff> &c, const RawIntegral &seed,
                             const std::function<bool(unsigned)> &trivial,
                             std::vector<SymTerm<Coeff>> &out) {
  out.clear();
  const unsigned n = f.size();
  if (p.dot_only) {
    for (unsigned k = 0; k < seed.size(); ++k)
      if (!(p.from & (1u << k)) && seed[k] != 0)
        return false;
  }
  std::vector<int> image(n, 0);
  Coeff coeff = ref_one<Coeff>();
  for (unsigned k = 0; k < seed.size(); ++k) {
    const int e = seed[k];
    if (e <= 0)
      continue;
    unsigned nonzero = 0;
    int j = -1;
    for (unsigned l = 0; l < n; ++l)
      if (!f[k][l].is_zero()) {
        ++nonzero;
        j = int(l);
      }
    if (nonzero != 1 || !c[k].is_zero())
      throw std::runtime_error("symmetry: denominator row is not a single propagator");
    image[unsigned(j)] += e;
    coeff = coeff / ref_pow(f[k][unsigned(j)], unsigned(e));
  }
  std::map<std::vector<int>, Coeff> acc;
  acc[image] = coeff;
  for (unsigned k = 0; k < seed.size(); ++k) {
    if (seed[k] >= 0)
      continue;
    for (int factor = 0; factor < -seed[k]; ++factor) {
      std::map<std::vector<int>, Coeff> next;
      for (const auto &[vec, cf] : acc) {
        for (unsigned j = 0; j < n; ++j) {
          if (f[k][j].is_zero())
            continue;
          std::vector<int> v = vec;
          if (v[j] == 1) {
            v[j] = 0;
            unsigned sector = 0;
            for (unsigned m = 0; m < n; ++m)
              if (v[m] > 0)
                sector |= 1u << m;
            if (trivial(sector))
              continue;
          } else {
            v[j] -= 1;
          }
          next[v] += cf * f[k][j];
        }
        if (!c[k].is_zero())
          next[vec] += cf * c[k];
      }
      acc.swap(next);
    }
  }
  out.reserve(acc.size());
  for (const auto &[vec, cf] : acc)
    if (!cf.is_zero())
      out.push_back(SymTerm<Coeff>{vec, cf});
  return true;
}

// A seed corpus for one prescription: the source's own support, one power
// doubled, one line negative (the numerator case) on each line in turn, two
// negatives at once, and the empty seed -- so the pinch test meets both a
// dropping and a surviving configuration.
static std::vector<std::vector<int>>
expansion_seeds(const PropPrescription &p, unsigned nprops) {
  std::vector<int> base(nprops, 0);
  for (unsigned k = 0; k < nprops; ++k)
    if (p.from & (1u << k))
      base[k] = 1;
  std::vector<std::vector<int>> seeds;
  seeds.push_back(base);
  seeds.push_back(std::vector<int>(nprops, 0));
  for (unsigned k = 0; k < nprops; ++k) {
    std::vector<int> sv = base;
    sv[k] = (p.from & (1u << k)) ? 2 : 1;
    seeds.push_back(sv);
    sv = base;
    sv[k] = -1;
    seeds.push_back(sv);
    sv = base;
    sv[k] = -2;
    seeds.push_back(sv);
    sv = base;
    sv[k] = -1;
    sv[(k + 1) % nprops] -= 1;
    seeds.push_back(sv);
  }
  return seeds;
}

// The predicates the pinch test is driven with; all three are used for both
// implementations, so the comparison covers the branch wherever it fires.
static std::vector<std::function<bool(unsigned)>> trivial_variants() {
  return {[](unsigned) { return false; },
          [](unsigned s) { return s == 0; },
          [](unsigned s) { return (s & 1u) == 0; }};
}

// Both expansions must agree term by term, in order, including which inputs they
// refuse: an expansion that throws must throw for both.
template <typename Coeff>
static void require_same_expansion(const PropPrescription &p,
                                   const std::vector<std::vector<Coeff>> &f,
                                   const std::vector<Coeff> &c,
                                   const std::vector<std::vector<int>> &seeds,
                                   const std::function<bool(unsigned)> &trivial) {
  for (const std::vector<int> &sv : seeds) {
    const RawIntegral seed(sv);
    std::vector<SymTerm<Coeff>> got, want;
    bool got_ok = true, want_ok = true;
    try {
      got_ok = expand_prescription<Coeff>(p, f, c, seed, trivial, got);
    } catch (const std::exception &) {
      got_ok = false;
    }
    try {
      want_ok = reference_expand<Coeff>(p, f, c, seed, trivial, want);
    } catch (const std::exception &) {
      want_ok = false;
    }
    REQUIRE_EQ(got_ok, want_ok);
    if (!want_ok)
      continue;
    REQUIRE_EQ(got.size(), want.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
      REQUIRE(got[i].powers == want[i].powers);
      REQUIRE((got[i].coeff - want[i].coeff).is_zero());
    }
  }
}

TEST(symmetry_expansion_matches_the_map_based_reference_on_numeric_rows) {
  const std::vector<PropPrescription> prescriptions = {
      hand_prescription(3, 1, false),
      rows_prescription(3, 1, false, {{1, 0}, {0, 1}}, {0, 0}),
      rows_prescription(3, 1, false, {{2, 3}, {5, 7}}, {4, -6}),
      rows_prescription(3, 1, false, {{1, 1}, {0, 1}}, {2, 0}),
      // A single-term denominator row that still carries a constant: the other
      // half of the throw's condition, and the only shape that reaches it -- a
      // mutation check on `nonzero != 1 || !c[k].is_zero()` showed the rest of
      // this corpus never does.
      rows_prescription(3, 1, false, {{1, 0}, {0, 1}}, {5, 0}),
      hand_prescription(3, 1, true), // dot_only: refused unless no numerator outside
  };
  for (const PropPrescription &p : prescriptions) {
    PropPrescription q = p;
    q.try_to_numeric(); // these rows are numbers, so this always succeeds
    REQUIRE(q.has_numeric_rows);
    for (const auto &trivial : trivial_variants())
      require_same_expansion<rational>(q, q.f_num, q.c_num, expansion_seeds(q, 2), trivial);
  }
}

TEST(symmetry_expansion_matches_the_map_based_reference_on_symbolic_rows) {
  Family f = make_box("oracle_box");
  const unsigned top = (1u << f.nprops()) - 1;
  SymmetryTable t =
      SymmetryTable::find(f.symmetry_inputs(), f.nontrivial_sectors(top), top);
  unsigned checked = 0;
  for (unsigned s = 0; s <= top; ++s)
    for (const PropPrescription &p : t.bySource(s)) {
      ++checked;
      for (const auto &trivial : trivial_variants())
        require_same_expansion<GiNaC::ex>(p, ex_rows(p), ex_col(p),
                                          expansion_seeds(p, f.nprops()), trivial);
    }
  REQUIRE(checked > 0);
}

// ---------------------------------------------------------------------------
// Flat accumulators (spec 5.3).  The map an expansion accumulates into is
// replaced by a vector merged in place; the emitted term set and its order must
// not move, so the merge sorts by key and drops what cancels -- exactly where
// the map path dropped them, at the end of the accumulation.
// ---------------------------------------------------------------------------

TEST(symmetry_flat_merge_sums_keys_in_sorted_order) {
  std::vector<std::pair<int, rational>> v = {{2, rational(1)}, {1, rational(2)},
                                             {2, rational(5)},  {3, rational(0)},
                                             {1, rational(-2)}, {0, rational(4)}};
  flat_merge(v);
  // Keys 1 and 3 sum to zero and are dropped; 0 and 2 survive in key order.
  REQUIRE_EQ(v.size(), 2u);
  REQUIRE_EQ(v[0].first, 0);
  REQUIRE(v[0].second == rational(4));
  REQUIRE_EQ(v[1].first, 2);
  REQUIRE(v[1].second == rational(6));
}

// The same for a key whose only order is `operator<`, which is what RawIntegral
// offers.  Its order is index-major, so {0,1} sorts before {1,0}.
TEST(symmetry_flat_merge_works_on_rawintegral_keys) {
  std::vector<std::pair<RawIntegral, rational>> v;
  v.emplace_back(RawIntegral(std::vector<int>({2, 0})), rational(1));
  v.emplace_back(RawIntegral(std::vector<int>({1, 0})), rational(1));
  v.emplace_back(RawIntegral(std::vector<int>({2, 0})), rational(-1));
  v.emplace_back(RawIntegral(std::vector<int>({1, 1})), rational(0));
  v.emplace_back(RawIntegral(std::vector<int>({0, 1})), rational(5));
  flat_merge(v);
  REQUIRE_EQ(v.size(), 2u);
  REQUIRE(v[0].first == RawIntegral(std::vector<int>({0, 1})));
  REQUIRE(v[0].second == rational(5));
  REQUIRE(v[1].first == RawIntegral(std::vector<int>({1, 0})));
  REQUIRE(v[1].second == rational(1));
}
