#include "spill_reader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <numeric>

namespace secred {
namespace {

// The mapped spill.  The scanner hands out spans into the mapping and never copies a
// line; MAP_PRIVATE + MADV_SEQUENTIAL keeps the pages evictable.
class MappedFile {
public:
  explicit MappedFile(const std::string &path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
      return;
    struct stat st {};
    if (::fstat(fd_, &st) != 0) {
      ::close(fd_);
      fd_ = -1;
      return;
    }
    opened_ = true;
    if (st.st_size <= 0)
      return; // an empty file scans as an empty system, as the getline readers did
    size_ = static_cast<std::size_t>(st.st_size);
    void *p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (p == MAP_FAILED) {
      ::close(fd_);
      fd_ = -1;
      size_ = 0;
      opened_ = false;
      return;
    }
    data_ = static_cast<const char *>(p);
    ::madvise(const_cast<char *>(data_), size_, MADV_SEQUENTIAL);
  }
  ~MappedFile() {
    if (data_)
      ::munmap(const_cast<char *>(data_), size_);
    if (fd_ >= 0)
      ::close(fd_);
  }
  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;

  bool ok() const { return opened_; }
  const char *begin() const { return data_; }
  const char *end() const { return data_ + size_; }

private:
  int fd_ = -1;
  const char *data_ = nullptr;
  std::size_t size_ = 0;
  bool opened_ = false;
};

// "[a, b, c]" in [b, e) -> out (reused across lines: no allocation).  Refuses anything
// `operator<<` cannot print, so `RawIntegral::from_text` (which accepts leading spaces,
// a '+', and trailing junk) can be the fallback and reproduce today's result exactly.
bool parse_integral_span(const char *b, const char *e, std::vector<int> &out) {
  out.clear();
  if (b == e || *b != '[' || *(e - 1) != ']')
    return false;
  const char *p = b + 1;
  const char *last = e - 1;
  if (p == last)
    return false;
  while (p < last) {
    bool neg = false;
    if (*p == '-') {
      neg = true;
      ++p;
    } else if (*p == '+') {
      return false;
    }
    if (p == last || *p < '0' || *p > '9')
      return false;
    long v = 0;
    while (p < last && *p >= '0' && *p <= '9') {
      v = v * 10 + (*p - '0');
      if (v > 1000000)
        return false; // indices are powers of propagators; refuse typos
      ++p;
    }
    out.push_back(static_cast<int>(neg ? -v : v));
    if (p == last)
      break;
    if (*p != ',' || p + 1 >= last || *(p + 1) != ' ')
      return false;
    p += 2;
  }
  if (out.empty() || out.size() > RawIntegral::MAX_PROPS)
    return false;
  return true;
}

// "num" or "num/den" over the whole span, den > 0, digits only.  The digit loop tests
// before it multiplies, so a coefficient too large for int64 is refused instead of
// wrapping into a value that looks small.
bool parse_rat64_span(const char *b, const char *e, int64_t &num, int64_t &den) {
  if (b == e)
    return false;
  bool neg = false;
  if (*b == '-') {
    neg = true;
    ++b;
  } else if (*b == '+') {
    return false;
  }
  if (b == e || *b < '0' || *b > '9')
    return false;
  const std::uint64_t limit = static_cast<std::uint64_t>(INT64_MAX);
  std::uint64_t n = 0;
  while (b < e && *b >= '0' && *b <= '9') {
    const std::uint64_t d = static_cast<std::uint64_t>(*b - '0');
    if (n > (limit - d) / 10)
      return false;
    n = n * 10 + d;
    ++b;
  }
  den = 1;
  if (b < e) {
    if (*b != '/')
      return false;
    ++b;
    if (b == e)
      return false;
    std::uint64_t d64 = 0;
    while (b < e && *b >= '0' && *b <= '9') {
      const std::uint64_t d = static_cast<std::uint64_t>(*b - '0');
      if (d64 > (limit - d) / 10)
        return false;
      d64 = d64 * 10 + d;
      ++b;
    }
    if (b != e || d64 == 0)
      return false;
    den = static_cast<int64_t>(d64);
  }
  if (b != e)
    return false;
  num = neg ? -static_cast<int64_t>(n) : static_cast<int64_t>(n);
  return true;
}

// Parse a coefficient string ("num" or "num/den") into int64 num/den (den > 0).  IBP
// coefficients are small integers or simple fractions and always fit; false on overflow.
// The scanner's fallback, moved here from globalreduce.cpp.
bool parse_rat64(const std::string &s, int64_t &num, int64_t &den) {
  std::size_t slash = s.find('/');
  try {
    if (slash == std::string::npos) {
      num = std::stoll(s);
      den = 1;
    } else {
      num = std::stoll(s.substr(0, slash));
      den = std::stoll(s.substr(slash + 1));
      if (den < 0) {
        num = -num;
        den = -den;
      }
      if (den == 0)
        return false;
    }
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

// Walk the spill and feed `sink`: `sink.term(id, coeff_begin, coeff_end)` per term line
// whose integral is known, `sink.end_equation()` on a blank line.  A `term` that returns
// false aborts the scan (the sink has already set `error`).  `flush_at_eof` reproduces
// each call site's treatment of a final equation with no blank line after it: the two
// readers flushed it, `select_dum_eq_from_file` did not.
template <typename Sink>
bool scan_spill(const std::string &path, Sink &sink,
                const std::unordered_map<RawIntegral, unsigned> &integral_id,
                bool flush_at_eof, SpillReadStats &stats, std::string &error) {
  MappedFile f(path);
  if (!f.ok()) {
    error = "cannot open spill file: " + path;
    return false;
  }
  const char *p = f.begin();
  const char *end = f.end();
  std::vector<int> idx;
  idx.reserve(16);
  while (p < end) {
    const char *nl = static_cast<const char *>(std::memchr(p, '\n', end - p));
    const char *line_end = nl ? nl : end;
    const char *next = nl ? nl + 1 : end;
    if (line_end == p) {
      sink.end_equation();
      p = next;
      continue;
    }
    const char *tab =
        static_cast<const char *>(std::memchr(p, '\t', line_end - p));
    if (tab == nullptr) {
      p = next; // as the getline readers: a line with no tab is ignored
      continue;
    }
    unsigned id = 0;
    bool known = false;
    if (parse_integral_span(p, tab, idx)) {
      auto it = integral_id.find(RawIntegral(idx));
      if (it != integral_id.end()) {
        id = it->second;
        known = true;
      }
    } else {
      auto it = integral_id.find(RawIntegral::from_text(std::string(p, tab)));
      if (it != integral_id.end()) {
        id = it->second;
        known = true;
      }
    }
    if (!known) {
      ++stats.skipped_unknown;
      p = next;
      continue;
    }
    if (!sink.term(id, tab + 1, line_end))
      return false;
    p = next;
  }
  if (flush_at_eof)
    sink.end_equation();
  return true;
}

// The numeric sink: keeps every (id, num/den) pair, sorts the equation's terms by
// descending id, and appends it.  A zero coefficient is kept: `reduce_mod` drops it.
struct NumericSink {
  std::vector<Eq> &system;
  SpillReadStats &stats;
  std::string &error;
  std::vector<std::pair<unsigned, Rat64>> eq;

  bool term(unsigned id, const char *b, const char *e) {
    int64_t num = 0, den = 1;
    if (!parse_rat64_span(b, e, num, den)) {
      const std::string s(b, e);
      if (!parse_rat64(s, num, den)) {
        error = "coefficient does not fit in int64: " + s;
        return false;
      }
      ++stats.fallbacks;
    }
    eq.emplace_back(id, Rat64{num, den});
    return true;
  }

  void end_equation() {
    if (eq.empty())
      return;
    std::sort(eq.begin(), eq.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });
    Eq out;
    out.reserve(eq.size());
    for (const auto &pr : eq)
      out.push_back(pr);
    eq.clear();
    stats.equations += 1;
    stats.terms += out.size();
    system.push_back(std::move(out));
  }
};

// The dum sink: the largest known integral id of the current equation, and the index of
// the equation whose value is smallest so far.
struct DumSink {
  unsigned best = 0;
  unsigned best_max = UINT_MAX;
  unsigned eq_idx = 0;
  unsigned cur_max = 0;
  bool have_term = false;

  bool term(unsigned id, const char *, const char *) {
    if (id > cur_max)
      cur_max = id;
    have_term = true;
    return true;
  }

  void end_equation() {
    if (have_term && cur_max < best_max) {
      best_max = cur_max;
      best = eq_idx;
    }
    ++eq_idx;
    cur_max = 0;
    have_term = false;
  }
};

// Read one coefficient span the way both sinks need it: the strict grammar first, the
// GiNaC parser (with the expand + make_sym_coeff pipeline the old reader ran) only for
// what it refuses.  `Zero` is the grammar's spelling of a zero polynomial -- the reader
// drops such a term, as the old one did -- and `Fail` means neither parse worked, with
// `error` set.
enum class CoeffText { Ok, Zero, Fail };

CoeffText parse_coeff_text(const char *b, const char *e,
                           const std::unordered_map<std::string, unsigned> &sym_index,
                           GiNaC::parser &parser, SymCoeff &out, SpillReadStats &stats,
                           std::string &error) {
  if (parse_spill_coeff(b, e, sym_index, out))
    return out.num.terms.empty() ? CoeffText::Zero : CoeffText::Ok;
  ++stats.fallbacks;
  try {
    GiNaC::ex ex = parser(std::string(b, e));
    ex = ex.expand();
    if (ex.is_zero())
      return CoeffText::Zero;
    out = make_sym_coeff(ex, sym_index);
  } catch (const std::exception &ex2) {
    error = "cannot parse coefficient '" + std::string(b, e) + "': " + ex2.what();
    return CoeffText::Fail;
  }
  return CoeffText::Ok;
}

// The symbolic sink: `started` keeps the old reader's equation bookkeeping.  It pushed
// an equation as soon as one of its terms had a known integral, even when every
// coefficient then turned out to be zero -- such an equation stays in the system, empty.
struct SymbolicSink {
  std::vector<SymEq> &system;
  const std::unordered_map<std::string, unsigned> &sym_index;
  GiNaC::parser &parser;
  SpillReadStats &stats;
  std::string &error;
  std::vector<std::pair<unsigned, SymCoeff>> eq;
  bool started = false;

  bool term(unsigned id, const char *b, const char *e) {
    started = true;
    SymCoeff c;
    const CoeffText r = parse_coeff_text(b, e, sym_index, parser, c, stats, error);
    if (r == CoeffText::Fail)
      return false;
    if (r == CoeffText::Zero)
      return true;
    eq.emplace_back(id, std::move(c));
    return true;
  }

  void end_equation() {
    if (!started)
      return;
    started = false;
    std::sort(eq.begin(), eq.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });
    SymEq out;
    out.reserve(eq.size());
    for (auto &pr : eq)
      out.push_back(std::move(pr));
    eq.clear();
    stats.equations += 1;
    stats.terms += out.size();
    system.push_back(std::move(out));
  }
};

// The interned sink: the same scan and the same equation bookkeeping, but a term keeps
// only its slot.  A coefficient is interned the moment it is parsed, so the system is
// the pool's term table plus one object per distinct function -- for F234, 8 B per term
// against a `SymCoeff` per term.  The row order, the row boundaries and the slots are
// the ones `build_sym_coeff_pool` assigns to the same system, so an interned read and a
// read followed by interning produce the same pool.
struct InternedSink {
  SymCoeffInterner &in;
  const std::unordered_map<std::string, unsigned> &sym_index;
  GiNaC::parser &parser;
  SpillReadStats &stats;
  std::string &error;
  std::vector<std::pair<unsigned, SymCoeff>> eq;
  bool started = false;

  bool term(unsigned id, const char *b, const char *e) {
    started = true;
    SymCoeff c;
    const CoeffText r = parse_coeff_text(b, e, sym_index, parser, c, stats, error);
    if (r == CoeffText::Fail)
      return false;
    if (r == CoeffText::Zero)
      return true;
    eq.emplace_back(id, std::move(c));
    return true;
  }

  void end_equation() {
    if (!started)
      return;
    started = false;
    std::sort(eq.begin(), eq.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });
    stats.equations += 1;
    stats.terms += eq.size();
    in.begin_row();
    for (auto &pr : eq)
      in.add_term(pr.first, pr.second);
    eq.clear(); // the coefficients were interned: only their slots survive
  }
};

// ---------------------------------------------------------------------------
// The coefficient grammar of `print_row_poly` (row_coeff.h:165-232).
//
//   factored:  t1[+-]t2...            each term  [n[/d]][*]name[^e][*name[^e]]*
//   paren:     (t1[+-]t2...)/([q*]mono)   the printer writes ")/(", so the denominator
//                                         is parenthesised too; the numerator is integral
//
// The factored form reduces each term's denominator by its own gcd(a, den)
// independently, so one polynomial's terms can carry different denominators
// (`1/3*d+1/2`); the coefficient's denominator is their lcm, and both forms are then
// reduced by the content gcd, which is what makes the result equal to the
// numer()/denom() normalisation the slow path builds from the same text.
// ---------------------------------------------------------------------------

struct CoeffCursor {
  const char *p;
  const char *e;
  bool at_end() const { return p == e; }
  bool peek(char c) const { return p < e && *p == c; }
  bool eat(char c) {
    if (p < e && *p == c) {
      ++p;
      return true;
    }
    return false;
  }
};

bool read_digits(CoeffCursor &c, std::uint64_t &v) {
  if (c.p == c.e || *c.p < '0' || *c.p > '9')
    return false;
  v = 0;
  const std::uint64_t limit = static_cast<std::uint64_t>(INT64_MAX);
  while (c.p < c.e && *c.p >= '0' && *c.p <= '9') {
    const std::uint64_t d = static_cast<std::uint64_t>(*c.p - '0');
    if (v > (limit - d) / 10) // check before multiplying: no wrap-around
      return false;
    v = v * 10 + d;
    ++c.p;
  }
  return true;
}

// `name[^e]` then `*name[^e]`...  The whole name must be in the symbol table: `d1` is
// looked up as `d1`, never as `d` followed by a stray `1`.  A symbol repeated inside
// one monomial is refused (`d*d` is not a spelling the printer emits -- it writes
// `d^2` -- and GiNaC would have combined the two factors).
bool parse_mono(CoeffCursor &c, const std::unordered_map<std::string, unsigned> &sym_index,
                std::vector<std::pair<std::uint8_t, std::uint8_t>> &vars) {
  while (true) {
    const char *b = c.p;
    if (c.p < c.e && (std::isalpha(static_cast<unsigned char>(*c.p)) || *c.p == '_')) {
      ++c.p;
      while (c.p < c.e &&
             (std::isalnum(static_cast<unsigned char>(*c.p)) || *c.p == '_'))
        ++c.p;
    }
    if (c.p == b)
      return false;
    // A std::string's small-string buffer holds the names the printer emits (d, msq,
    // x12), so this lookup does not allocate.
    auto it = sym_index.find(std::string(b, c.p));
    if (it == sym_index.end() || it->second > 255)
      return false;
    std::uint64_t exp = 1;
    if (c.eat('^')) {
      if (!read_digits(c, exp) || exp == 0 || exp > 255)
        return false;
    }
    const std::uint8_t idx = static_cast<std::uint8_t>(it->second);
    for (const auto &v : vars)
      if (v.first == idx)
        return false;
    vars.emplace_back(idx, static_cast<std::uint8_t>(exp));
    if (!c.eat('*'))
      return true;
  }
}

// One term body: an integer coefficient with an optional denominator, an optional '*'
// and an optional monomial.  `print_row_poly` omits a unit coefficient (`d`, `-d`), so
// both parts are optional but not both at once.
bool parse_term_body(CoeffCursor &c, const std::unordered_map<std::string, unsigned> &sym_index,
                     std::int64_t &num, std::int64_t &den,
                     std::vector<std::pair<std::uint8_t, std::uint8_t>> &vars) {
  num = 1;
  den = 1;
  vars.clear();
  if (c.p < c.e && *c.p >= '0' && *c.p <= '9') {
    std::uint64_t n = 0;
    if (!read_digits(c, n))
      return false;
    num = static_cast<std::int64_t>(n);
    if (c.eat('/')) {
      std::uint64_t d = 0;
      if (!read_digits(c, d) || d == 0)
        return false;
      den = static_cast<std::int64_t>(d);
    }
    if (!c.eat('*'))
      return true; // "<n>" or "<n>/<d>" alone: a constant term
  }
  return parse_mono(c, sym_index, vars);
}

struct ParsedTerm {
  std::int64_t num = 1;
  std::int64_t den = 1;
  bool neg = false;
  std::vector<std::pair<std::uint8_t, std::uint8_t>> vars;
};

// `in_paren` is the numerator of the paren form: the sum ends at the closing
// parenthesis, which the caller consumes.
bool parse_poly(CoeffCursor &c, const std::unordered_map<std::string, unsigned> &sym_index,
                bool in_paren, std::vector<ParsedTerm> &terms) {
  terms.clear();
  bool neg = c.eat('-');
  if (!neg)
    c.eat('+');
  while (true) {
    ParsedTerm t;
    t.neg = neg;
    if (!parse_term_body(c, sym_index, t.num, t.den, t.vars))
      return false;
    terms.push_back(std::move(t));
    if (c.at_end() || (in_paren && c.peek(')')))
      break;
    if (c.eat('+'))
      neg = false;
    else if (c.eat('-'))
      neg = true;
    else
      return false;
  }
  // Two terms with the same monomial would be combined by GiNaC, and in the factored
  // form they can carry different denominators; refuse instead of guessing (the caller
  // falls back to the GiNaC parser, which combines them correctly).
  for (std::size_t i = 0; i < terms.size(); ++i)
    for (std::size_t j = i + 1; j < terms.size(); ++j)
      if (terms[i].vars == terms[j].vars)
        return false;
  return !terms.empty();
}

} // namespace

bool parse_spill_coeff(const char *b, const char *e,
                       const std::unordered_map<std::string, unsigned> &sym_index,
                       SymCoeff &out) {
  out = SymCoeff{};
  CoeffCursor c{b, e};
  if (c.at_end())
    return false;
  std::vector<ParsedTerm> terms;
  std::uint64_t q = 1;
  std::vector<std::pair<std::uint8_t, std::uint8_t>> mono;
  const bool paren = c.peek('(');
  if (paren) {
    ++c.p;
    if (!parse_poly(c, sym_index, /*in_paren=*/true, terms))
      return false;
    if (!c.eat(')') || !c.eat('/') || !c.eat('(')) // the printer writes ")/("
      return false;
    if (c.p < c.e && *c.p >= '0' && *c.p <= '9') {
      if (!read_digits(c, q) || q == 0 || q > 0x7fffffffULL)
        return false;
      if (!c.eat('*'))
        return false;
    }
    if (!parse_mono(c, sym_index, mono))
      return false;
    if (!c.eat(')'))
      return false;
  } else {
    if (!parse_poly(c, sym_index, /*in_paren=*/false, terms))
      return false;
  }
  if (!c.at_end())
    return false;

  if (paren) {
    // The numerator's terms are integral by construction (q and the monomial are
    // printed around the parentheses).
    for (const ParsedTerm &t : terms) {
      if (t.den != 1)
        return false;
      out.num.terms.push_back(
          SymTerm{cln::cl_I(t.neg ? -t.num : t.num), t.vars});
    }
    // The denominator is the product q*mono -- one polynomial term, which is what
    // GiNaC's denom() makes of the same text (`(d-2)/(2*m^2)` -> `2*m^2`).
    SymTerm den;
    den.coef = cln::cl_I(static_cast<long>(q));
    den.vars = mono;
    out.den.terms.push_back(std::move(den));
  } else {
    // Common denominator: the lcm of the per-term denominators.
    std::uint64_t l = 1;
    for (const ParsedTerm &t : terms) {
      const std::uint64_t d = static_cast<std::uint64_t>(t.den);
      const std::uint64_t g = std::gcd(l, d);
      if (l / g > 0x7fffffffULL / (d / g))
        return false; // a denominator the int32 the printer uses cannot hold
      l = (l / g) * d;
    }
    for (const ParsedTerm &t : terms) {
      const std::int64_t scale = static_cast<std::int64_t>(l / static_cast<std::uint64_t>(t.den));
      if (t.num > INT64_MAX / scale)
        return false;
      const std::int64_t v = t.num * scale;
      out.num.terms.push_back(SymTerm{cln::cl_I(t.neg ? -v : v), t.vars});
    }
    if (l != 1)
      out.den.terms.push_back(SymTerm{cln::cl_I(static_cast<long>(l)), {}});
  }

  // Content gcd: divide numerator and denominator by what they share, so a paren form
  // the printer left unreduced (`(2*d+4)/(2*m^2)`) comes out as GiNaC's numer()/denom()
  // would give it.  Only what the denominator carries can cancel -- with no denominator
  // the numerator's content is untouched (`-3*d` stays `-3*d`, as GiNaC leaves it).
  cln::cl_I g = 0;
  if (!out.den.terms.empty()) {
    for (const SymTerm &t : out.num.terms)
      g = cln::gcd(g, t.coef);
    for (const SymTerm &t : out.den.terms)
      g = cln::gcd(g, t.coef);
  }
  if (g != 0 && g != 1) {
    for (SymTerm &t : out.num.terms)
      t.coef = cln::exquo(t.coef, g);
    for (SymTerm &t : out.den.terms)
      t.coef = cln::exquo(t.coef, g);
  }
  // GiNaC's `add` drops zero terms and the empty sum prints `0`: a zero coefficient
  // must not survive into the table, and a zero sum carries no denominator.
  out.num.terms.erase(std::remove_if(out.num.terms.begin(), out.num.terms.end(),
                                     [](const SymTerm &t) { return cln::zerop(t.coef); }),
                      out.num.terms.end());
  if (out.num.terms.empty()) {
    out.den.terms.clear();
  } else if (out.den.terms.size() == 1 && out.den.terms[0].vars.empty() &&
             out.den.terms[0].coef == 1) {
    out.den.terms.clear(); // make_sym_coeff empties a denominator of 1
  }
  out.precompiled = true;
  return true;
}

bool read_spill_numeric(const std::string &path,
                        const std::unordered_map<RawIntegral, unsigned> &integral_id,
                        std::vector<Eq> &system, SpillReadStats &stats,
                        std::string &error) {
  NumericSink sink{system, stats, error};
  return scan_spill(path, sink, integral_id, /*flush_at_eof=*/true, stats, error);
}

bool read_spill_symbolic(const std::string &path,
                         const std::unordered_map<RawIntegral, unsigned> &integral_id,
                         const std::vector<GiNaC::possymbol> &symbols,
                         std::vector<SymEq> &system, SpillReadStats &stats,
                         std::string &error) {
  GiNaC::symtab symtab;
  std::unordered_map<std::string, unsigned> sym_index;
  for (unsigned i = 0; i < symbols.size(); ++i) {
    symtab[symbols[i].get_name()] = symbols[i];
    sym_index[symbols[i].get_name()] = i;
  }
  GiNaC::parser parser(symtab);
  SymbolicSink sink{system, sym_index, parser, stats, error};
  return scan_spill(path, sink, integral_id, /*flush_at_eof=*/true, stats, error);
}

bool read_spill_interned(const std::string &path,
                         const std::unordered_map<RawIntegral, unsigned> &integral_id,
                         const std::vector<GiNaC::possymbol> &symbols, SymCoeffPool &pool,
                         SpillReadStats &stats, std::string &error) {
  GiNaC::symtab symtab;
  std::unordered_map<std::string, unsigned> sym_index;
  for (unsigned i = 0; i < symbols.size(); ++i) {
    symtab[symbols[i].get_name()] = symbols[i];
    sym_index[symbols[i].get_name()] = i;
  }
  GiNaC::parser parser(symtab);
  SymCoeffInterner in;
  InternedSink sink{in, sym_index, parser, stats, error};
  if (!scan_spill(path, sink, integral_id, /*flush_at_eof=*/true, stats, error))
    return false;
  in.finish();
  pool = std::move(in.pool);
  return true;
}

unsigned scan_spill_dum(const std::string &path,
                        const std::unordered_map<RawIntegral, unsigned> &integral_id) {
  DumSink sink;
  SpillReadStats stats;
  std::string error;
  if (!scan_spill(path, sink, integral_id, /*flush_at_eof=*/false, stats, error))
    return 0; // an unreadable file produced best == 0 before, too
  return sink.best;
}

}  // namespace secred
