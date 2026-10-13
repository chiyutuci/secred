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

} // namespace

bool read_spill_numeric(const std::string &path,
                        const std::unordered_map<RawIntegral, unsigned> &integral_id,
                        std::vector<Eq> &system, SpillReadStats &stats,
                        std::string &error) {
  NumericSink sink{system, stats, error};
  return scan_spill(path, sink, integral_id, /*flush_at_eof=*/true, stats, error);
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
