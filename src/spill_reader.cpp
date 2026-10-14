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
  std::size_t size() const { return size_; } // `end() - begin()` is UB when empty

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

// Walk a text spill and feed `sink`: `sink.term(id, coeff_begin, coeff_end)` per term
// line whose integral is known, `sink.end_equation()` on a blank line.  A `term` that
// returns false aborts the scan (the sink has already set `error`).  A final equation
// with no blank line after it is flushed, as both getline readers did.
template <typename Sink>
bool scan_spill_text(const MappedFile &f, Sink &sink,
                     const std::unordered_map<RawIntegral, unsigned> &integral_id,
                     SpillReadStats &stats, std::string &error) {
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
  sink.end_equation();
  return true;
}

// ---------------------------------------------------------------------------
// The binary frame (sector.h, `openSpill`): "SRSP" + u32 version, one frame per
// equation (u32 n_terms, then (u32 integral index, u32 coefficient index) per term),
// the two interned tables, and a 36-byte footer with the table offsets and the two
// counters.  Everything little-endian, read byte by byte so the file's meaning does
// not depend on the writer's byte order.
//
// The tables are what the text format spelled out per occurrence, so the scan is the
// same work -- and the same sinks, since a table entry is handed to them as the span
// of text it holds.  What the frame adds is the ability to parse a coefficient once
// for all its occurrences, which is what `InternedSink::term_indexed` below uses.
// ---------------------------------------------------------------------------

std::uint32_t spill_load_u32(const char *p) {
  std::uint32_t v = 0;
  for (unsigned i = 0; i < 4; ++i)
    v |= static_cast<std::uint32_t>(static_cast<unsigned char>(p[i])) << (8 * i);
  return v;
}

std::uint64_t spill_load_u64(const char *p) {
  std::uint64_t v = 0;
  for (unsigned i = 0; i < 8; ++i)
    v |= static_cast<std::uint64_t>(static_cast<unsigned char>(p[i])) << (8 * i);
  return v;
}

// The two interned tables, as spans into the mapping, plus the term stream's extent.
struct SpillTables {
  std::vector<RawIntegral> integrals;   // integral table, by index
  std::vector<const char *> coeff_text; // coefficient table, by index
  std::vector<std::uint32_t> coeff_len;
  const char *terms_begin = nullptr; // the first frame
  const char *terms_end = nullptr;   // where the integral table begins
  std::uint64_t n_equations = 0;     // the writer's counters, checked against the walk
  std::uint64_t n_terms = 0;
};

// Decode the tables and the footer.  Every inconsistency a corrupt or truncated file
// can present -- a version this build does not know, a footer that is not there,
// offsets that do not line up, an entry running past its table -- is a hard failure:
// a spill that was cut short must not read back as a short system.
bool decode_spill(const MappedFile &f, SpillTables &t, std::string &error) {
  const char *begin = f.begin();
  const std::size_t size = f.size();
  const std::uint32_t version = spill_load_u32(begin + 4);
  if (version != kSpillVersion) {
    error = "unsupported spill version " + std::to_string(version) + " (this build " +
            "reads version " + std::to_string(kSpillVersion) + ")";
    return false;
  }
  constexpr std::size_t kFooterBytes = 4 * 8 + 4;
  if (size < kSpillHeaderBytes + kFooterBytes) {
    error = "truncated spill file: no footer";
    return false;
  }
  const char *footer = begin + size - kFooterBytes;
  if (std::memcmp(footer + 32, kSpillMagic, 4) != 0) {
    error = "truncated spill file: no footer at the end of the file";
    return false;
  }
  const std::uint64_t integral_off = spill_load_u64(footer);
  const std::uint64_t coeff_off = spill_load_u64(footer + 8);
  t.n_equations = spill_load_u64(footer + 16);
  t.n_terms = spill_load_u64(footer + 24);
  const std::uint64_t tables_end = static_cast<std::uint64_t>(footer - begin);
  if (integral_off < kSpillHeaderBytes || coeff_off < integral_off ||
      coeff_off > tables_end) {
    error = "truncated spill file: table offsets out of range";
    return false;
  }

  const char *p = begin + integral_off;
  const char *const int_end = begin + coeff_off;
  if (int_end - p < 8) {
    error = "truncated spill file: integral table";
    return false;
  }
  const std::uint64_t n_int = spill_load_u64(p);
  p += 8;
  if (n_int > static_cast<std::uint64_t>(int_end - p)) { // nprops alone is one byte
    error = "truncated spill file: integral table";
    return false;
  }
  t.integrals.reserve(static_cast<std::size_t>(n_int));
  std::vector<int> idx;
  idx.reserve(RawIntegral::MAX_PROPS);
  for (std::uint64_t i = 0; i < n_int; ++i) {
    if (p == int_end) {
      error = "truncated spill file: integral table";
      return false;
    }
    const unsigned nprops = static_cast<unsigned char>(*p++);
    if (nprops > RawIntegral::MAX_PROPS ||
        static_cast<std::uint64_t>(int_end - p) < nprops) {
      error = "truncated spill file: integral table";
      return false;
    }
    idx.clear();
    for (unsigned k = 0; k < nprops; ++k)
      idx.push_back(static_cast<int>(static_cast<std::int8_t>(p[k])));
    p += nprops;
    t.integrals.push_back(RawIntegral(idx));
  }
  if (p != int_end) {
    error = "malformed spill: the integral table does not fill its range";
    return false;
  }

  p = begin + coeff_off;
  const char *const coeff_end = begin + tables_end;
  if (coeff_end - p < 8) {
    error = "truncated spill file: coefficient table";
    return false;
  }
  const std::uint64_t n_coeff = spill_load_u64(p);
  p += 8;
  if (n_coeff > static_cast<std::uint64_t>(coeff_end - p) / 4) { // length alone
    error = "truncated spill file: coefficient table";
    return false;
  }
  t.coeff_text.reserve(static_cast<std::size_t>(n_coeff));
  t.coeff_len.reserve(static_cast<std::size_t>(n_coeff));
  for (std::uint64_t i = 0; i < n_coeff; ++i) {
    if (coeff_end - p < 4) {
      error = "truncated spill file: coefficient table";
      return false;
    }
    const std::uint32_t len = spill_load_u32(p);
    p += 4;
    if (static_cast<std::uint64_t>(coeff_end - p) < len) {
      error = "truncated spill file: coefficient table";
      return false;
    }
    t.coeff_text.push_back(p);
    t.coeff_len.push_back(len);
    p += len;
  }
  if (p != coeff_end) {
    error = "malformed spill: the coefficient table does not fill its range";
    return false;
  }

  t.terms_begin = begin + kSpillHeaderBytes;
  t.terms_end = begin + integral_off;
  return true;
}

// Walk the frames and feed the same `sink` the text scan feeds: one `end_equation()`
// per frame, so the equation bookkeeping (a frame that ends up empty, a frame whose
// terms are all unknown) cannot differ between the two formats.  A sink that can take
// the coefficient table index -- the interned one -- gets it, and parses each entry
// once instead of once per occurrence.
template <typename Sink>
bool scan_spill_binary(const SpillTables &t, Sink &sink,
                       const std::unordered_map<RawIntegral, unsigned> &integral_id,
                       SpillReadStats &stats, std::string &error) {
  const char *p = t.terms_begin;
  const char *const end = t.terms_end;
  std::uint64_t frames = 0;
  std::uint64_t terms = 0;
  while (p < end) {
    if (end - p < 4) {
      error = "truncated spill file: term stream";
      return false;
    }
    const std::uint32_t n = spill_load_u32(p);
    p += 4;
    if (static_cast<std::uint64_t>(end - p) < 8ull * n) {
      error = "truncated spill file: term stream";
      return false;
    }
    ++frames;
    for (std::uint32_t i = 0; i < n; ++i) {
      const std::uint32_t integ_i = spill_load_u32(p);
      const std::uint32_t coeff_i = spill_load_u32(p + 4);
      p += 8;
      ++terms;
      if (integ_i >= t.integrals.size() || coeff_i >= t.coeff_text.size()) {
        error = "malformed spill: a term index is outside its table";
        return false;
      }
      const auto it = integral_id.find(t.integrals[integ_i]);
      if (it == integral_id.end()) {
        ++stats.skipped_unknown;
        continue;
      }
      const char *cb = t.coeff_text[coeff_i];
      const char *ce = cb + t.coeff_len[coeff_i];
      const bool ok = [&] {
        if constexpr (requires { sink.term_indexed(0u, 0u, cb, ce); })
          return sink.term_indexed(it->second, coeff_i, cb, ce);
        else
          return sink.term(it->second, cb, ce);
      }();
      if (!ok)
        return false;
    }
    sink.end_equation();
  }
  if (frames != t.n_equations || terms != t.n_terms) {
    error = "malformed spill: the footer's counters do not match the term stream";
    return false;
  }
  return true;
}

// The spill's format, decided by its first eight bytes: "SRSP" + the version means the
// binary frame, anything else the text one -- which is what every run before this one
// wrote and what the frozen bench spills are.
bool spill_is_binary(const MappedFile &f) {
  return f.size() >= kSpillHeaderBytes &&
         std::memcmp(f.begin(), kSpillMagic, 4) == 0;
}

// Read either format into one sink.  The mapping must stay alive while the sink reads
// the coefficient spans it hands out, so it lives here, around the scan.
template <typename Sink>
bool scan_spill_file(const std::string &path, Sink &sink,
                     const std::unordered_map<RawIntegral, unsigned> &integral_id,
                     SpillReadStats &stats, std::string &error) {
  MappedFile f(path);
  if (!f.ok()) {
    error = "cannot open spill file: " + path;
    return false;
  }
  if (!spill_is_binary(f))
    return scan_spill_text(f, sink, integral_id, stats, error);
  SpillTables tables;
  if (!decode_spill(f, tables, error))
    return false;
  return scan_spill_binary(tables, sink, integral_id, stats, error);
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

// One coefficient table entry's parse, memoised across its occurrences: the binary
// frame stores each distinct coefficient once, so the GiNaC parse a fallback needs is
// paid once per entry rather than once per occurrence.  `state` is the outcome, so a
// zero -- which must still start its equation -- is remembered too.
struct CoeffEntry {
  enum State : std::uint8_t { Unparsed, Ok, Zero, Failed };
  SymCoeff value;
  std::uint8_t state = Unparsed;
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

// The same sink for a binary frame, where the coefficient table makes two things
// possible that the text scan cannot do: each entry is parsed once for all its
// occurrences (`cache`, sized to the table before the scan so an entry's address stays
// put), and a row holds *pointers* into that cache rather than copies -- a coefficient
// occurring a thousand times costs a thousand 8-byte frames and one parse, where the
// text sink pays a `SymCoeff` per occurrence.  The bookkeeping -- what starts an
// equation, when a row is pushed, what the slots are -- is `InternedSink`'s.
struct InternedBinarySink {
  SymCoeffInterner &in;
  const std::unordered_map<std::string, unsigned> &sym_index;
  GiNaC::parser &parser;
  SpillReadStats &stats;
  std::string &error;
  std::vector<CoeffEntry> cache; // one per coefficient table entry
  std::vector<std::pair<unsigned, const SymCoeff *>> eq;
  bool started = false;

  InternedBinarySink(SymCoeffInterner &in_,
                     const std::unordered_map<std::string, unsigned> &sym_index_,
                     GiNaC::parser &parser_, SpillReadStats &stats_,
                     std::string &error_, std::size_t n_coeff)
      : in(in_), sym_index(sym_index_), parser(parser_), stats(stats_),
        error(error_), cache(n_coeff) {}

  bool term_indexed(unsigned id, std::uint32_t ci, const char *b, const char *e) {
    CoeffEntry &entry = cache[ci];
    if (entry.state == CoeffEntry::Unparsed) {
      const CoeffText r =
          parse_coeff_text(b, e, sym_index, parser, entry.value, stats, error);
      if (r == CoeffText::Fail) {
        entry.state = CoeffEntry::Failed;
        return false;
      }
      entry.state = r == CoeffText::Zero ? CoeffEntry::Zero : CoeffEntry::Ok;
    }
    started = true; // a zero coefficient still starts its equation, as on the text path
    if (entry.state == CoeffEntry::Ok)
      eq.emplace_back(id, &entry.value);
    return true;
  }

  void end_equation() {
    if (!started)
      return;
    started = false;
    // Stable, so terms that share an integral id keep the frame's order.
    std::stable_sort(eq.begin(), eq.end(),
                     [](const auto &a, const auto &b) { return a.first > b.first; });
    stats.equations += 1;
    stats.terms += eq.size();
    in.begin_row();
    for (const auto &pr : eq)
      in.add_term(pr.first, *pr.second);
    eq.clear();
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
  return scan_spill_file(path, sink, integral_id, stats, error);
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
  return scan_spill_file(path, sink, integral_id, stats, error);
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
  // The mapping must outlive the scan: a binary frame's coefficient spans point into
  // it, and so does this dispatch.  The two formats need different sinks -- the binary
  // one parses per table entry and holds pointers into its cache -- so the dispatch
  // lives here rather than in `scan_spill_file`.
  MappedFile f(path);
  if (!f.ok()) {
    error = "cannot open spill file: " + path;
    return false;
  }
  if (!spill_is_binary(f)) {
    InternedSink sink{in, sym_index, parser, stats, error};
    if (!scan_spill_text(f, sink, integral_id, stats, error))
      return false;
  } else {
    SpillTables tables;
    if (!decode_spill(f, tables, error))
      return false;
    InternedBinarySink sink{in, sym_index, parser, stats, error,
                            tables.coeff_text.size()};
    if (!scan_spill_binary(tables, sink, integral_id, stats, error))
      return false;
  }
  in.finish();
  pool = std::move(in.pool);
  return true;
}


}  // namespace secred
