// The FireFly black box: the templated probe path, the per-prime table reduction, the
// threading contract, and the degenerate systems the driver must not hand to FireFly.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <dlfcn.h>
#ifdef __ELF__
#include <link.h>
#endif
#include <sys/wait.h>
#include <unistd.h>

#include <gmp.h>

#include <ginac/ginac.h>

#include <fflow/primes.hh>

#include <firefly/FFInt.hpp>
#include <firefly/FFIntVec.hpp>
#include <firefly/Reconstructor.hpp>

#include "firefly_global.h"
#include "firefly_patch.h"
#include "globalreduce.h"
#include "spill_reader.h"
#include "symbolic_ffred.h"
#include "test_runner.h"
#include "tiny_symbolic_system.h"

namespace {

// The evaluator over the tiny system, built through the shared learn exactly as the
// driver builds it.  `masters_override` replaces the learned master map, which is how
// the failure test makes a target reduce to something that is not a master.
secred::SymbolicFFEvaluator
evaluator_of(std::unordered_map<unsigned, unsigned> *masters_override = nullptr) {
  const std::string spill = tiny::write_spill("/tmp/secred_test_firefly_tiny");
  secred::SymCoeffPool pool;
  secred::SpillReadStats stats;
  std::string error;
  const auto ids = tiny::integral_ids();
  const auto symbols = tiny::symbols();
  const auto integrals = tiny::integrals();
  if (!secred::read_spill_interned(spill, ids, symbols, pool, stats, error))
    throw std::runtime_error(error);
  secred::SymbolicGlobalLearn learn =
      secred::symbolic_global_learn(pool, {3u}, integrals, symbols,
                                    1125899906842597ULL,
                                    "/tmp/secred_test_firefly_masters", "[test]", 0, 0, 0, 0);
  if (!learn.ok)
    throw std::runtime_error(learn.error);
  std::unordered_map<unsigned, unsigned> master_index =
      masters_override ? *masters_override : learn.master_index;
  std::vector<GiNaC::ex> symbol_ex;
  for (const auto &s : symbols)
    symbol_ex.push_back(GiNaC::ex(s));
  const unsigned nnindeps = masters_override ? 1u : learn.nnindeps;
  // The item table comes from the learn exactly as the driver passes it; with overridden
  // masters the learned one would not match, so the (single) learned item is kept as is.
  secred::ReductionItems items = learn.items;
  return secred::SymbolicFFEvaluator(std::move(pool), std::move(learn.needed_eq_idx),
                                     std::move(learn.bwd_chain), std::move(learn.dep_targets),
                                     std::move(master_index), nnindeps,
                                     std::move(items), std::move(symbol_ex));
}

// `prime_changed()` reads `firefly::FFInt::p`, so the tests set the prime the way
// FireFly does: `set_new_prime` on the global, then the box's hook.
void promote_prime(std::uint64_t p) {
  firefly::FFInt::set_new_prime(p);
}

// A black box whose value depends on the prime field: at every prime it is the clean
// degree-1 function c(p)*(x + 7), so the per-prime interpolation is cheap, but the
// coefficient c(p) never stabilizes under the rational reconstruction across primes,
// so the reconstructor keeps promoting until it hits the cap.  It records the primes
// it was probed at, which is how the many-prime test observes how far it got.
class PrimeDependentBox : public firefly::BlackBoxBase<PrimeDependentBox> {
public:
  template <typename FFTemp>
  std::vector<FFTemp> operator()(const std::vector<FFTemp> &x) {
    using Lane = secred::FireFlyLane<FFTemp>;
    const std::uint64_t p = firefly::FFInt::p;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (std::find(primes_seen_.begin(), primes_seen_.end(), p) == primes_seen_.end())
        primes_seen_.push_back(p);
    }
    std::uint64_t values[Lane::lanes];
    for (unsigned l = 0; l < Lane::lanes; ++l) {
      const std::uint64_t c = (p >> 3) | 1;   // nonzero, different in every field
      const std::uint64_t xv = Lane::value(x[0], l);
      values[l] = static_cast<std::uint64_t>(
          (static_cast<unsigned __int128>(c) * (xv + 7)) % p);
    }
    return {Lane::make(values, 1)};
  }

  void prime_changed() {}

  std::vector<std::uint64_t> primes_seen() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return primes_seen_;
  }

private:
  mutable std::mutex mutex_;
  std::vector<std::uint64_t> primes_seen_;
};

}  // namespace

// The reconstruction prime cap is the caller's settings.max_primes, with no bound from
// FireFly's table: lib/firefly_patch grows the table on demand for any count.
TEST(firefly_prime_cap_is_max_primes) {
  REQUIRE_EQ(secred::firefly_prime_cap(1u), 1u);
  REQUIRE_EQ(secred::firefly_prime_cap(42u), 42u);
  REQUIRE_EQ(secred::firefly_prime_cap(300u), 300u);
  REQUIRE_EQ(secred::firefly_prime_cap(301u), 301u);
  REQUIRE_EQ(secred::firefly_prime_cap(500u), 500u);
  REQUIRE_EQ(secred::firefly_prime_cap(4096u), 4096u);
}

// FireFly's own 300 primes stay verbatim at the front of the table; ensure_primes()
// appends descending 63-bit primes below the last entry, does nothing when the table
// is already large enough, and never touches what is already there.
TEST(firefly_primes_grow_on_demand) {
  const std::vector<std::uint64_t> ship = firefly::primes();
  REQUIRE_EQ(ship.size(), std::size_t(300));

  firefly::ensure_primes(305);
  REQUIRE(firefly::primes().size() >= 305);
  REQUIRE_EQ(firefly::primes()[0], ship[0]);
  for (std::size_t i = 0; i < ship.size(); ++i)
    REQUIRE_EQ(firefly::primes()[i], ship[i]);

  for (std::size_t i = 1; i < firefly::primes().size(); ++i)
    REQUIRE(firefly::primes()[i] < firefly::primes()[i - 1]);   // descending, distinct
  for (std::size_t i = ship.size(); i < firefly::primes().size(); ++i) {
    REQUIRE(firefly::primes()[i] < (std::uint64_t(1) << 63));   // FFInt needs 63 bits
    mpz_t z;
    mpz_init_set_ui(z, static_cast<unsigned long>(firefly::primes()[i]));
    REQUIRE(mpz_probab_prime_p(z, 25) > 0);
    mpz_clear(z);
  }

  const std::size_t grown = firefly::primes().size();
  firefly::ensure_primes(302);   // below the current size: a no-op
  REQUIRE_EQ(firefly::primes().size(), grown);
}

#if defined(__ELF__) && defined(R_X86_64_COPY)

namespace {

struct CopyRelocScan {
  const char *name;   // mangled symbol the executable may import
  bool found;         // an R_X86_64_COPY relocation targets it
};

int scan_copy_relocations(dl_phdr_info *info, std::size_t, void *data) {
  auto *scan = static_cast<CopyRelocScan *>(data);
  if (info->dlpi_name != nullptr && info->dlpi_name[0] != '\0') return 0;   // main exe

  const ElfW(Dyn) *dynamic = nullptr;
  for (ElfW(Half) i = 0; i < info->dlpi_phnum; ++i)
    if (info->dlpi_phdr[i].p_type == PT_DYNAMIC)
      dynamic = reinterpret_cast<const ElfW(Dyn) *>(
          info->dlpi_addr + info->dlpi_phdr[i].p_vaddr);
  if (dynamic == nullptr) return 0;

  // glibc adjusts the dynamic section of the main executable in place, so these
  // pointers are already absolute there; a toolchain that leaves them link-time-relative
  // makes them smaller than the load base.  Normalize both conventions.
  const auto resolve = [info](ElfW(Addr) value) {
    return reinterpret_cast<const std::byte *>(
        value >= info->dlpi_addr ? value : value + info->dlpi_addr);
  };
  const ElfW(Sym) *symtab = nullptr;
  const char *strtab = nullptr;
  const ElfW(Rela) *rela = nullptr;
  std::size_t rela_bytes = 0;
  for (const ElfW(Dyn) *entry = dynamic; entry->d_tag != DT_NULL; ++entry) {
    switch (entry->d_tag) {
    case DT_SYMTAB:
      symtab = reinterpret_cast<const ElfW(Sym) *>(resolve(entry->d_un.d_ptr));
      break;
    case DT_STRTAB:
      strtab = reinterpret_cast<const char *>(resolve(entry->d_un.d_ptr));
      break;
    case DT_RELA:
      rela = reinterpret_cast<const ElfW(Rela) *>(resolve(entry->d_un.d_ptr));
      break;
    case DT_RELASZ:
      rela_bytes = entry->d_un.d_val;
      break;
    }
  }
  if (symtab == nullptr || strtab == nullptr || rela == nullptr) return 0;

  for (std::size_t i = 0; i < rela_bytes / sizeof(ElfW(Rela)); ++i) {
    if (ELF64_R_TYPE(rela[i].r_info) != R_X86_64_COPY) continue;
    const std::size_t index = ELF64_R_SYM(rela[i].r_info);
    if (std::string(strtab + symtab[index].st_name) == scan->name) {
      scan->found = true;
      break;
    }
  }
  return 0;
}

// True when the executable imports `name` through an ELF copy relocation: libfirefly.so
// defines the storage, and its static initializer constructs the single instance in the
// executable's copy.  A definition in the executable's own objects needs no such
// relocation, so its presence distinguishes an import from a duplicate definition.
bool executable_imports_symbol(const char *name) {
  CopyRelocScan scan{name, false};
  ::dl_iterate_phdr(&scan_copy_relocations, &scan);
  return scan.found;
}

}  // namespace

#endif

// RatReconst's static state must exist exactly once in the process, owned by
// libfirefly.so.  If the executable also defines it (as unpatched copies of
// RatReconst.cpp do), the executable's definitions win the dynamic symbol lookup:
// libfirefly.so's static initializer and the executable's construct the same objects,
// and both register the same destructors.  Each object is then destroyed twice at exit;
// with glibc's allocator that is "double free or corruption" and SIGABRT -- exit code
// -6 -- after an otherwise correct reconstruction.  The executable may reference the
// library's objects directly (the symbol resolves into libfirefly.so) or through an ELF
// copy relocation; what it must not do is define them itself.
TEST(firefly_ratreconst_statics_are_owned_by_libfirefly) {
  const char *const names[] = {
      "_ZN7firefly10RatReconst5shiftE",
      "_ZN7firefly10RatReconst10curr_shiftE",
      "_ZN7firefly10RatReconst13mutex_staticsE",
      "_ZN7firefly10RatReconst19singular_system_setE",
      "_ZN7firefly10RatReconst20global_anchor_pointsE",
  };
  for (const char *name : names) {
    void *addr = ::dlsym(RTLD_DEFAULT, name);
    REQUIRE_MSG(addr != nullptr, std::string("dlsym failed for ") + name);
    Dl_info info{};
    REQUIRE_MSG(::dladdr(addr, &info) != 0 && info.dli_fname != nullptr,
                std::string("dladdr failed for ") + name);
    const bool owned_by_library =
        std::string(info.dli_fname).find("libfirefly") != std::string::npos;
#if defined(__ELF__) && defined(R_X86_64_COPY)
    const bool imported = executable_imports_symbol(name);
#else
    const bool imported = false;
#endif
    REQUIRE_MSG(owned_by_library || imported,
                std::string(name) + " resolves to " + info.dli_fname +
                    " without a copy relocation, so the executable defines a duplicate "
                    "static that libfirefly.so will also construct and destroy");
  }
}

// The needed-feed hand-off is fixed in the executable's copy of RatReconst.cpp (the
// same lib/firefly_patch mechanism as the prime table).  A link-order regression would
// silently bring back libfirefly.so's unlocked move with no compile error, so pin the
// binding: the getter the reconstruction calls must not resolve into libfirefly.so.
// The cast is an address lookup, not a call; GCC warns about member-function pointers.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpmf-conversions"
#endif
TEST(firefly_patched_needed_feed_vec_is_linked) {
  using Getter = std::vector<std::pair<std::uint32_t, std::uint32_t>> (
      firefly::RatReconst::*)();
  const Getter getter = &firefly::RatReconst::get_needed_feed_vec;
  Dl_info info{};
  REQUIRE(::dladdr(reinterpret_cast<void *>(getter), &info) != 0);
  REQUIRE(info.dli_fname != nullptr);
  REQUIRE_MSG(std::string(info.dli_fname).find("libfirefly") == std::string::npos,
              std::string("get_needed_feed_vec resolves to ") + info.dli_fname +
                  "; the patched definition in lib/firefly_patch must win");
}

// get_zi_orders() is the second getter in that patch: it read coef_mat while the
// worker that owned the interpolation could be appending to it.  Pin its binding too.
TEST(firefly_patched_get_zi_orders_is_linked) {
  using Order = std::pair<std::vector<std::uint32_t>, std::uint32_t>;
  using Getter =
      std::pair<std::vector<Order>, std::uint32_t> (firefly::RatReconst::*)() const;
  const Getter getter = &firefly::RatReconst::get_zi_orders;
  Dl_info info{};
  REQUIRE(::dladdr(reinterpret_cast<void *>(getter), &info) != 0);
  REQUIRE(info.dli_fname != nullptr);
  REQUIRE_MSG(std::string(info.dli_fname).find("libfirefly") == std::string::npos,
              std::string("get_zi_orders resolves to ") + info.dli_fname +
                  "; the patched definition in lib/firefly_patch must win");
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

// The point of lib/firefly_patch: a reconstruction can run past FireFly's 300-entry
// table.  The walk runs in a child process -- FireFly's sampling stream and DenseSolver
// state are per-process and the suite otherwise runs exactly one real reconstruction --
// and reports the highest table index it was probed at; 300 or more means a prime that
// FireFly 2.0.3 does not have was used.
TEST(firefly_reconstruction_uses_primes_beyond_the_ship_table) {
  int fds[2];
  REQUIRE_EQ(::pipe(fds), 0);
  const pid_t pid = ::fork();
  REQUIRE(pid >= 0);

  if (pid == 0) {   // child: one reconstruction that can only end at the cap
    ::close(fds[0]);
    if (::chdir("/tmp") != 0)   // FireFly writes firefly.log here
      ::_exit(2);
    constexpr unsigned kCap = 305;
    firefly::ensure_primes(static_cast<std::size_t>(kCap) + 1);
    std::size_t max_index = 0;
    {
      PrimeDependentBox box;
      firefly::Reconstructor<PrimeDependentBox> rec(
          1, 1, box, firefly::Reconstructor<PrimeDependentBox>::SILENT);
      rec.reconstruct(kCap);
      const std::vector<std::uint64_t> &table = firefly::primes();
      for (const std::uint64_t p : box.primes_seen()) {
        const auto it =
            std::lower_bound(table.begin(), table.end(), p, std::greater<std::uint64_t>());
        max_index = std::max(max_index, static_cast<std::size_t>(it - table.begin()));
      }
    }
    const ssize_t written = ::write(fds[1], &max_index, sizeof max_index);
    ::close(fds[1]);
    ::_exit(written == static_cast<ssize_t>(sizeof max_index) ? 0 : 1);
  }

  ::close(fds[1]);
  std::size_t max_index = 0;
  ssize_t got = 0;
  while (got < static_cast<ssize_t>(sizeof max_index)) {
    const ssize_t n = ::read(fds[0], reinterpret_cast<char *>(&max_index) + got,
                             sizeof max_index - static_cast<std::size_t>(got));
    if (n <= 0)
      break;
    got += n;
  }
  ::close(fds[0]);
  int status = 0;
  ::waitpid(pid, &status, 0);

  REQUIRE(WIFEXITED(status));
  REQUIRE_EQ(WEXITSTATUS(status), 0);
  REQUIRE_EQ(got, static_cast<ssize_t>(sizeof max_index));
  REQUIRE(max_index >= 300u);   // FireFly's own table ends at index 299
}

TEST(firefly_box_returns_the_evaluators_values) {
  const std::uint64_t p = firefly::FFInt::p;   // what FireFly starts from
  secred::SymbolicFFEvaluator ev = evaluator_of();
  secred::SecRedFireFlyBlackBox box(ev, 1);
  box.prime_changed();
  std::vector<firefly::FFInt> x = {firefly::FFInt(5), firefly::FFInt(7)};
  const std::vector<firefly::FFInt> got = box(x);
  REQUIRE_EQ(got.size(), std::size_t(1));
  secred::SymbolicFFEvaluator::Scratch scratch;
  std::uint64_t ref[1] = {0};
  const std::uint64_t xv[2] = {5, 7};
  REQUIRE(ev.evaluate_point(p, xv, scratch, ref) ==
          secred::SymbolicFFEvaluator::PointResult::Ok);
  REQUIRE_EQ(got[0].n, ref[0]);
  REQUIRE_EQ(box.probes(), std::uint64_t(1));
  REQUIRE_EQ(box.points(), std::uint64_t(1));
  REQUIRE_EQ(box.primes(), std::uint64_t(1));
  REQUIRE(!box.hard_failure());
  REQUIRE(box.fail_reason() == secred::SymbolicFFEvaluator::PointResult::Ok);
}

// The box adds each probe's share of the three solve phases (the scratch is thread-local
// and outlives the probe, so what it reads is a delta), which the driver averages into
// the "probe solution steps" report line.
TEST(firefly_box_reports_the_probe_phase_seconds) {
  secred::SymbolicFFEvaluator ev = evaluator_of();
  secred::SecRedFireFlyBlackBox box(ev, 1);
  box.prime_changed();
  REQUIRE_EQ(box.eval_seconds(), 0.0);
  const std::vector<firefly::FFInt> x = {firefly::FFInt(5), firefly::FFInt(7)};
  box(x);
  REQUIRE(box.eval_seconds() > 0.0);   // fill + substitute + reduce
  REQUIRE(box.fwd_seconds() > 0.0);    // forward elimination
  REQUIRE(box.bwd_seconds() > 0.0);    // back substitution
}

// The lane trait: a bunched call must return, lane by lane, what N single calls return.
// FireFly only ever uses this path with bunch_size 1, but the template is instantiated
// for every bunch size, and a trait that silently mis-indexes would reconstruct garbage
// if a future driver ever asked for a bunch.
TEST(firefly_box_bunch_lanes_match_single_calls) {
  const std::uint64_t p = firefly::FFInt::p;
  secred::SymbolicFFEvaluator ev = evaluator_of();
  secred::SecRedFireFlyBlackBox box(ev, 1);
  box.prime_changed();
  // Four points for one output: lane l of the vector must equal the scalar probe.
  const std::uint64_t pts[4][2] = {{5, 7}, {11, 13}, {17, 19}, {23, 29}};
  const std::vector<firefly::FFIntVec<4>> bunched = box(
      std::vector<firefly::FFIntVec<4>>{
          firefly::FFIntVec<4>(std::array<firefly::FFInt, 4>{
              firefly::FFInt(pts[0][0]), firefly::FFInt(pts[1][0]),
              firefly::FFInt(pts[2][0]), firefly::FFInt(pts[3][0])}),
          firefly::FFIntVec<4>(std::array<firefly::FFInt, 4>{
              firefly::FFInt(pts[0][1]), firefly::FFInt(pts[1][1]),
              firefly::FFInt(pts[2][1]), firefly::FFInt(pts[3][1])})});
  REQUIRE_EQ(bunched.size(), std::size_t(1));
  for (unsigned l = 0; l < 4; ++l) {
    std::vector<firefly::FFInt> point = {firefly::FFInt(pts[l][0]),
                                         firefly::FFInt(pts[l][1])};
    // A scalar probe re-reads the same prime, so no re-reduction happens between them.
    REQUIRE_EQ(box(point)[0].n, bunched[0].vec[l].n);
  }
  REQUIRE_EQ(box.probes(), std::uint64_t(5));
}

// Spec 7: probes running while the prime moves must each see one prime's tables.
// Phase 1 hammers the box at one prime through 8 threads; phase 2 promotes the prime
// underneath them; every probe in both phases is compared against the reference value at
// one of the two primes -- a probe equal to neither is the failure this test exists for.
TEST(firefly_box_probes_never_mix_two_primes) {
  secred::SymbolicFFEvaluator ev = evaluator_of();
  secred::SecRedFireFlyBlackBox box(ev, 4);
  const std::uint64_t p1 = firefly::FFInt::p - 1000;   // force a promotion below
  promote_prime(p1);
  box.prime_changed();

  std::uint64_t ref1[1] = {0}, ref2[1] = {0};
  const std::uint64_t xv[2] = {5, 7};
  {
    secred::SymbolicFFEvaluator::Scratch s;
    REQUIRE(ev.evaluate_point(p1, xv, s, ref1) ==
            secred::SymbolicFFEvaluator::PointResult::Ok);
  }
  const std::uint64_t p2 = p1 - 1000;
  promote_prime(p2);
  {
    // A second evaluator so the reference for p2 is computed without disturbing `ev`'s
    // tables -- `reduce_for_prime` is exactly what a promotion does.
    secred::SymbolicFFEvaluator ev2 = evaluator_of();
    ev2.reduce_for_prime(p2, 1);
    secred::SymbolicFFEvaluator::Scratch s;
    REQUIRE(ev2.evaluate_point(p2, xv, s, ref2) ==
            secred::SymbolicFFEvaluator::PointResult::Ok);
  }
  promote_prime(p1);
  box.prime_changed();   // back to p1: the threads below start from a known state

  std::vector<firefly::FFInt> x = {firefly::FFInt(5), firefly::FFInt(7)};
  std::atomic<bool> stop{false};
  std::atomic<int> bad{0};
  std::vector<std::thread> ts;
  for (unsigned t = 0; t < 8; ++t)
    ts.emplace_back([&]() {
      while (!stop.load(std::memory_order_relaxed)) {
        const std::uint64_t v = box(x)[0].n;
        if (v != ref1[0] && v != ref2[0])
          ++bad;
      }
    });
  for (unsigned i = 0; i < 20; ++i) {   // promote back and forth under the probes
    promote_prime(p2);
    box.prime_changed();
    promote_prime(p1);
    box.prime_changed();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  stop.store(true);
  for (auto &t : ts)
    t.join();
  REQUIRE_EQ(bad.load(), 0);
  REQUIRE(!box.hard_failure());
}

TEST(firefly_box_reports_a_stale_selection_as_a_hard_failure) {
  // No masters at all: [0,0] sits on every right-hand side and matches none of them, so
  // the reduction would be truncated.  The box must latch the failure and return zeros.
  std::unordered_map<unsigned, unsigned> no_masters;
  secred::SymbolicFFEvaluator ev = evaluator_of(&no_masters);
  secred::SecRedFireFlyBlackBox box(ev, 1);
  box.prime_changed();
  std::vector<firefly::FFInt> x = {firefly::FFInt(5), firefly::FFInt(7)};
  const std::vector<firefly::FFInt> got = box(x);
  REQUIRE(box.hard_failure());
  REQUIRE_EQ(got[0].n, std::uint64_t(0));
  REQUIRE(box.fail_reason() == secred::SymbolicFFEvaluator::PointResult::NonMasterRhs);
  REQUIRE_EQ(box.fail_target(), 3u);
  REQUIRE_EQ(box.fail_integral(), 0u);
}

TEST(firefly_box_handles_a_zero_output_system) {
  // n_outputs() == 0 (every target is a master): the probe returns an empty vector and
  // the driver never constructs a Reconstructor with n == 0.
  const std::string spill = tiny::write_spill("/tmp/secred_test_firefly_zero");
  secred::SymCoeffPool pool;
  secred::SpillReadStats stats;
  std::string error;
  const auto ids = tiny::integral_ids();
  const auto symbols = tiny::symbols();
  const auto integrals = tiny::integrals();
  REQUIRE(secred::read_spill_interned(spill, ids, symbols, pool, stats, error));
  secred::SymbolicGlobalLearn learn =
      secred::symbolic_global_learn(pool, {0u}, integrals, symbols,
                                    1125899906842597ULL,
                                    "/tmp/secred_test_firefly_masters0", "[test]", 0, 0, 0, 0);
  REQUIRE(learn.ok);
  REQUIRE_EQ(learn.dep_targets.size(), std::size_t(0));
  std::vector<GiNaC::ex> symbol_ex;
  for (const auto &s : symbols)
    symbol_ex.push_back(GiNaC::ex(s));
  secred::SymbolicFFEvaluator ev(std::move(pool), std::move(learn.needed_eq_idx),
                                 std::move(learn.bwd_chain), std::move(learn.dep_targets),
                                 std::move(learn.master_index), 0u,
                                 std::move(learn.items), std::move(symbol_ex));
  REQUIRE_EQ(ev.n_outputs(), 0u);
  secred::SecRedFireFlyBlackBox box(ev, 1);
  box.prime_changed();
  const std::vector<firefly::FFInt> x = {firefly::FFInt(5), firefly::FFInt(7)};
  REQUIRE_EQ(box(x).size(), std::size_t(0));
  REQUIRE(!box.hard_failure());
}

// G1 at unit scale: the two symbolic backends must return the same reduction of the same
// system.  The selection is shared and deterministic (Task 6), so a difference here is a
// difference between reconstruction engines.
TEST(firefly_and_finiteflow_agree_on_the_tiny_system) {
  const std::string spill = tiny::write_spill("/tmp/secred_test_tiny_spill");
  const auto integrals = tiny::integrals();
  const auto ids = tiny::integral_ids();
  const auto symbols = tiny::symbols();
  const std::vector<RawIntegral> targets = {integrals[3]};
  const auto run = [&](auto &&fn) {
    return fn(spill, integrals, ids, targets, symbols, 0u, 0u, 0u, std::size_t(0),
              "/tmp/secred_test_tiny_masters", 2u);
  };
  const GlobalReductionResult ff = run(
      [](auto &&...args) { return reduce_global_finiteflow_symbolic(args...); });
  const GlobalReductionResult fly = run(
      [](auto &&...args) { return secred::reduce_global_firefly_symbolic(args...); });
  REQUIRE(ff.ok);
  REQUIRE(fly.ok);
  REQUIRE_EQ(ff.masters.size(), fly.masters.size());
  REQUIRE(ff.masters == fly.masters);
  REQUIRE_EQ(ff.reductions.size(), fly.reductions.size());
  // The two engines spell a rational function differently ((1/2*d^2+3/2)/(1/2*d+1) vs
  // (d^2+3)/(d+2)), so compare values: exact rational arithmetic at a few points.
  GiNaC::symtab tab;
  for (const auto &s : symbols)
    tab[s.get_name()] = s;
  GiNaC::parser parser(tab);
  for (std::size_t i = 0; i < ff.reductions.size(); ++i) {
    REQUIRE(ff.reductions[i].first == fly.reductions[i].first);
    if (ff.reductions[i].second == "-1")
      continue;
    const GiNaC::ex a = parser(ff.reductions[i].second);
    const GiNaC::ex b = parser(fly.reductions[i].second);
    for (const auto &pt : {std::pair<int, int>{3, 5}, {7, 11}, {-2, 13}}) {
      GiNaC::exmap sub;
      sub[symbols[0]] = GiNaC::numeric(pt.first);
      sub[symbols[1]] = GiNaC::numeric(pt.second);
      REQUIRE(GiNaC::ex_to<GiNaC::numeric>(a.subs(sub))
                  .is_equal(GiNaC::ex_to<GiNaC::numeric>(b.subs(sub))));
    }
  }
}

// Review Focus 4: every target is already a master -> zero outputs.  FireFly's
// Reconstructor is fatal with n == 0, so the driver must return before constructing it.
TEST(firefly_driver_handles_a_zero_output_system) {
  const std::string spill = tiny::write_spill("/tmp/secred_test_tiny_spill0");
  const auto integrals = tiny::integrals();
  const auto ids = tiny::integral_ids();
  const auto symbols = tiny::symbols();
  const std::vector<RawIntegral> targets = {integrals[0]};   // never a pivot
  const GlobalReductionResult r = secred::reduce_global_firefly_symbolic(
      spill, integrals, ids, targets, symbols, 0u, 0u, 0u, std::size_t(0),
      "/tmp/secred_test_tiny_masters0", 2u);
  REQUIRE(r.ok);
  REQUIRE_EQ(r.reductions.size(), std::size_t(1));   // just the (target, "-1") marker
  REQUIRE_EQ(r.reductions[0].second, std::string("-1"));
}

// A global phase that did not complete (`ok == false`) returns an empty or truncated
// table.  Written to cache/reductions and exited 0, that is a silently wrong result:
// the run failed, so it has to stop the run.  family.cpp calls this before it writes
// anything, and main.cpp turns the throw into a non-zero exit.
TEST(a_failed_global_reduction_is_fatal_not_an_empty_table) {
  const auto integrals = tiny::integrals();
  const auto ids = tiny::integral_ids();
  const auto symbols = tiny::symbols();
  const std::vector<RawIntegral> targets = {integrals[3]};

  // A spill that is not there: the FireFly backend reports the read failure the way a
  // refused probe is reported, as ok == false with nothing reconstructed.
  const GlobalReductionResult failed = secred::reduce_global_firefly_symbolic(
      "/tmp/secred_test_no_such_spill", integrals, ids, targets, symbols, 0u, 0u, 0u,
      std::size_t(0), "/tmp/secred_test_tiny_masters_req", 2u);
  REQUIRE(!failed.ok);
  REQUIRE(failed.reductions.empty());

  std::string message;
  try {
    secred::require_global_reduction(failed, "firefly");
  } catch (const std::runtime_error &e) {
    message = e.what();
  }
  REQUIRE(message.find("firefly") != std::string::npos);        // names the tool asked for
  REQUIRE(message.find("did not complete cleanly") != std::string::npos);

  // A phase that did complete passes the check unchanged.  (The FiniteFlow symbolic
  // path on the same tiny system: one FireFly reconstruction is all this process does,
  // since FireFly's sampling stream continues from the previous Reconstructor and the
  // tiny system has a degenerate sampling line -- see the ledger note.)
  const std::string spill = tiny::write_spill("/tmp/secred_test_tiny_spill_req");
  const GlobalReductionResult done = reduce_global_finiteflow_symbolic(
      spill, integrals, ids, targets, symbols, 0u, 0u, 0u, std::size_t(0),
      "/tmp/secred_test_tiny_masters_req", 2u);
  REQUIRE(done.ok);
  secred::require_global_reduction(done, "firefly");   // must not throw
  REQUIRE(!done.reductions.empty());   // what the failed phase does not have to offer
}

// The settled-basis vote: consulted once the learn has fixed the masters, before the
// reconstruction.  A pass the caller will discard (its basis raised the seed-truncation
// dots limit) must stop there -- the reconstruction is that discarded pass's most
// expensive phase, and FireFly's Reconstructor must never be constructed for it.
TEST(firefly_vote_fires_before_the_reconstruction) {
  const std::string spill = tiny::write_spill("/tmp/secred_test_tiny_spill_vote");
  const auto integrals = tiny::integrals();
  const auto ids = tiny::integral_ids();
  const auto symbols = tiny::symbols();
  const std::vector<RawIntegral> targets = {integrals[3]};

  unsigned calls = 0;
  std::vector<RawIntegral> voted;
  const BasisVote discard = [&](const std::vector<RawIntegral> &masters) {
    ++calls;
    voted = masters;
    return true;
  };
  const GlobalReductionResult r = secred::reduce_global_firefly_symbolic(
      spill, integrals, ids, targets, symbols, 0u, 0u, 0u, std::size_t(0),
      "/tmp/secred_test_tiny_masters_vote", 2u, 500u, discard);
  REQUIRE_EQ(calls, 1u);            // consulted exactly once, after the learn
  REQUIRE(r.ok);
  REQUIRE(r.reconstruction_skipped);
  REQUIRE(!voted.empty());          // the learn settled on a basis
  REQUIRE(r.masters == voted);      // and the caller receives exactly that basis
  REQUIRE(r.reductions.empty());    // the reconstruction never ran
  REQUIRE_EQ(r.reconstruction_seconds, 0.0);
}

// A vote that declines leaves the phase exactly as it was: the same settled basis goes
// through the reconstruction, and nothing is marked skipped.
TEST(finiteflow_symbolic_vote_that_declines_keeps_the_reduction) {
  const std::string spill = tiny::write_spill("/tmp/secred_test_tiny_spill_vote_declined");
  const auto integrals = tiny::integrals();
  const auto ids = tiny::integral_ids();
  const auto symbols = tiny::symbols();
  const std::vector<RawIntegral> targets = {integrals[3]};

  std::vector<RawIntegral> voted;
  const BasisVote keep = [&](const std::vector<RawIntegral> &masters) {
    voted = masters;
    return false;
  };
  const GlobalReductionResult r = reduce_global_finiteflow_symbolic(
      spill, integrals, ids, targets, symbols, 0u, 0u, 0u, std::size_t(0),
      "/tmp/secred_test_tiny_masters_vote_declined", 2u, 500u, keep);
  REQUIRE(r.ok);
  REQUIRE(!r.reconstruction_skipped);
  REQUIRE(!voted.empty());
  REQUIRE(r.masters == voted);
  REQUIRE(!r.reductions.empty());
}

// The reconstruction prime cap is the caller's (settings.max_primes), not a private
// constant: the tiny system reconstructs under the default cap, and a cap of zero
// leaves FiniteFlow's MPRatFunReconstruction without a prime to build on, so the
// phase fails instead of emitting a truncated table.  The Jpsi sample whose
// coefficients need more than five primes is the production case.
TEST(finiteflow_symbolic_reconstruction_honors_the_prime_cap) {
  const std::string spill = tiny::write_spill("/tmp/secred_test_tiny_spill_cap");
  const auto integrals = tiny::integrals();
  const auto ids = tiny::integral_ids();
  const auto symbols = tiny::symbols();
  const std::vector<RawIntegral> targets = {integrals[3]};

  const GlobalReductionResult ok = reduce_global_finiteflow_symbolic(
      spill, integrals, ids, targets, symbols, 0u, 0u, 0u, std::size_t(0),
      "/tmp/secred_test_tiny_masters_cap_ok", 2u, 500u);
  REQUIRE(ok.ok);
  REQUIRE(!ok.reductions.empty());

  const GlobalReductionResult capped = reduce_global_finiteflow_symbolic(
      spill, integrals, ids, targets, symbols, 0u, 0u, 0u, std::size_t(0),
      "/tmp/secred_test_tiny_masters_cap_zero", 2u, 0u);
  REQUIRE(!capped.ok);
  REQUIRE(capped.reductions.empty());
}

// The univariate per-prime plan follows the degree pairs already learned: every
// known output asks for ceil((numdeg+dendeg)/2), which provides at least
// numdeg+dendeg+1 sample points for the linear solver and for Thiele alike; an
// output whose pair is not known yet asks for the trial degree; finished
// outputs are ignored.
TEST(finiteflow_symbolic_univariate_plan_covers_the_sample_need) {
  // Sampling at plan D provides 2D+1+n_extra points; n_extra is the number of
  // check points the sampler appends (three on the univariate path).
  const unsigned need[] = {6, 34, 5, 10};
  const char all_unfinished[] = {1, 1, 1, 1};
  REQUIRE_EQ(finiteflow_symbolic_univariate_plan_degree(need, all_unfinished, 4,
                                                       3),
             15u);  // (34 - 3) / 2, i.e. 2*15+1+3 = 34 sampled points

  const char only_second[] = {0, 1, 0, 0};
  REQUIRE_EQ(finiteflow_symbolic_univariate_plan_degree(need, only_second, 4, 3),
             15u);

  // A 64-equation system asks for (64-3)/2 = 30, and 2*30+1+3 covers it.
  const unsigned wide[] = {64};
  const char one_unfinished[] = {1};
  REQUIRE_EQ(finiteflow_symbolic_univariate_plan_degree(wide, one_unfinished, 1,
                                                       3),
             30u);

  // A need below the extra count needs no interpolation degree at all.
  const unsigned narrow[] = {4};
  REQUIRE_EQ(finiteflow_symbolic_univariate_plan_degree(narrow, one_unfinished, 1,
                                                       3),
             0u);

  const char none_unfinished[] = {0, 0, 0, 0};
  REQUIRE_EQ(finiteflow_symbolic_univariate_plan_degree(need, none_unfinished, 4,
                                                       3),
             0u);

  REQUIRE_EQ(finiteflow_symbolic_univariate_plan_degree(need, all_unfinished, 0,
                                                       3),
             0u);
}

// The batch size minimizes sample sweeps per prime: a set that already fills
// the threads runs alone, a set that overshoots them by a few points packs into
// more primes per sweep, and the batch never runs past the remaining primes.
TEST(finiteflow_symbolic_univariate_batch_size_packs_the_sweeps) {
  // One prime already fills a sweep, but a tie goes to the larger batch: eight
  // primes still cost one sweep each, with more tasks per thread.
  REQUIRE_EQ(finiteflow_symbolic_univariate_batch_size(64, 64, 100), 8u);
  // 68 points alone cost two sweeps; eight primes cost nine, i.e. 1.125.
  REQUIRE_EQ(finiteflow_symbolic_univariate_batch_size(64, 68, 100), 8u);
  // 34 points alone cost one sweep; seven primes cost four, i.e. 0.57.
  REQUIRE_EQ(finiteflow_symbolic_univariate_batch_size(64, 34, 100), 7u);
  REQUIRE_EQ(finiteflow_symbolic_univariate_batch_size(64, 4, 100), 8u);
  REQUIRE_EQ(finiteflow_symbolic_univariate_batch_size(64, 9, 100), 7u);
  REQUIRE_EQ(finiteflow_symbolic_univariate_batch_size(64, 4, 3), 3u);
  // With one thread no batch can overlap anything, so it only costs copies.
  REQUIRE_EQ(finiteflow_symbolic_univariate_batch_size(1, 4, 100), 1u);
}

// The dynamic scheduler visits every index exactly once, for any thread count,
// even when the first item is much slower than the rest.
TEST(parallel_for_dynamic_visits_every_index_once) {
  constexpr std::size_t n = 129;
  for (const unsigned nthreads : {1u, 3u, 8u}) {
    std::unique_ptr<std::atomic<unsigned>[]> hits(new std::atomic<unsigned>[n]());
    secred::parallel_for_dynamic(nthreads, n, [&](std::size_t i) {
      hits[i].fetch_add(1, std::memory_order_relaxed);
      if (i == 0 && nthreads > 1)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    });
    for (std::size_t i = 0; i < n; ++i)
      REQUIRE_EQ(hits[i].load(std::memory_order_relaxed), 1u);
  }

  unsigned calls = 0;
  secred::parallel_for_dynamic(4, 0, [&](std::size_t) { ++calls; });
  REQUIRE_EQ(calls, 0u);
}

// The one-symbol path end to end: the same tiny system with s = d+2 goes through
// the FiniteFlow driver's degree-driven univariate loop and must reproduce the
// reduction the two-symbol system gives at s = d+2.
TEST(finiteflow_symbolic_reduces_the_univariate_tiny_system) {
  const std::string spill =
      tiny::write_univariate_spill("/tmp/secred_test_tiny_uni_spill");
  const auto integrals = tiny::integrals();
  const auto ids = tiny::integral_ids();
  const auto symbols = tiny::univariate_symbols();
  const std::vector<RawIntegral> targets = {integrals[3]};

  const GlobalReductionResult r = reduce_global_finiteflow_symbolic(
      spill, integrals, ids, targets, symbols, 0u, 0u, 0u, std::size_t(0),
      "/tmp/secred_test_tiny_uni_masters", 2u);
  REQUIRE(r.ok);
  REQUIRE_EQ(r.reductions.size(), std::size_t(2));   // the marker and one master
  REQUIRE_EQ(r.reductions[0].second, std::string("-1"));

  GiNaC::symtab tab;
  for (const auto &s : symbols)
    tab[s.get_name()] = s;
  GiNaC::parser parser(tab);
  const GiNaC::ex got = parser(r.reductions[1].second);
  const GiNaC::ex want = parser("2*(d-2)/(d+2) + (d-4)/(d*(d+2)) - 3");
  for (const int d : {3, 5, 7, 11}) {
    GiNaC::exmap sub;
    sub[symbols[0]] = GiNaC::numeric(d);
    REQUIRE(GiNaC::ex_to<GiNaC::numeric>(got.subs(sub))
                .is_equal(GiNaC::ex_to<GiNaC::numeric>(want.subs(sub))));
  }
}

// The one-symbol linear reconstruction: a reduction whose denominator has a
// non-zero constant term is rebuilt by the degree-known solver at every prime
// after the first, and the merged result must match the exact function.
TEST(finiteflow_symbolic_univariate_linear_path_handles_a_polynomial_denominator) {
  const std::string spill =
      tiny::write_univariate_polynomial_spill("/tmp/secred_test_tiny_poly_spill");
  const auto integrals = tiny::integrals();
  const auto ids = tiny::integral_ids();
  const auto symbols = tiny::univariate_symbols();
  const std::vector<RawIntegral> targets = {integrals[3]};

  const GlobalReductionResult r = reduce_global_finiteflow_symbolic(
      spill, integrals, ids, targets, symbols, 0u, 0u, 0u, std::size_t(0),
      "/tmp/secred_test_tiny_poly_masters", 2u);
  REQUIRE(r.ok);
  REQUIRE_EQ(r.reductions.size(), std::size_t(2));
  REQUIRE_EQ(r.reductions[0].second, std::string("-1"));

  GiNaC::symtab tab;
  for (const auto &s : symbols)
    tab[s.get_name()] = s;
  GiNaC::parser parser(tab);
  const GiNaC::ex got = parser(r.reductions[1].second);
  // The spill rows are homogeneous equations: [1,1] + (d-2)/d [1,0] + [0,1]
  // + 3 [0,0] = 0 with [1,0] = -2d/(d+2) [0,0], [0,1] = -(d-4)/(d+2) [0,0].
  const GiNaC::ex want = parser("-14/(d+2)");
  for (const int d : {3, 5, 7, 11}) {
    GiNaC::exmap sub;
    sub[symbols[0]] = GiNaC::numeric(d);
    REQUIRE(GiNaC::ex_to<GiNaC::numeric>(got.subs(sub))
                .is_equal(GiNaC::ex_to<GiNaC::numeric>(want.subs(sub))));
  }
}

// Prime batching: a coefficient beyond one 63-bit prime keeps the univariate
// loop going past prime 1, where the remaining primes are sampled in one
// batched sweep against per-prime copies of the system.  The merged result must
// still be the exact function.  Sixteen threads against an eight-point prime
// put two primes in the batch.
TEST(finiteflow_symbolic_univariate_reconstructs_across_batched_primes) {
  const std::string spill = tiny::write_univariate_big_coeff_spill(
      "/tmp/secred_test_tiny_big_spill");
  const auto integrals = tiny::integrals();
  const auto ids = tiny::integral_ids();
  const auto symbols = tiny::univariate_symbols();
  const std::vector<RawIntegral> targets = {integrals[3]};

  const GlobalReductionResult r = reduce_global_finiteflow_symbolic(
      spill, integrals, ids, targets, symbols, 0u, 0u, 0u, std::size_t(0),
      "/tmp/secred_test_tiny_big_masters", 16u);
  REQUIRE(r.ok);
  REQUIRE_EQ(r.reductions.size(), std::size_t(2));
  REQUIRE_EQ(r.reductions[0].second, std::string("-1"));

  GiNaC::symtab tab;
  for (const auto &s : symbols)
    tab[s.get_name()] = s;
  GiNaC::parser parser(tab);
  const GiNaC::ex got = parser(r.reductions[1].second);
  // [1,0] = -2*d*N/(d+2) with N = 2^70+3, so
  // [1,1] = 2N(d-2)/(d+2) + (d-4)/(d+2) - 3 = ((2N-2)d - 4N - 10)/(d+2).
  const GiNaC::ex N = GiNaC::pow(GiNaC::numeric(2), 70) + 3;
  const GiNaC::ex want =
      ((2 * N - 2) * GiNaC::ex(symbols[0]) - 4 * N - 10) /
      (GiNaC::ex(symbols[0]) + 2);
  for (const int d : {3, 5, 7, 11}) {
    GiNaC::exmap sub;
    sub[symbols[0]] = GiNaC::numeric(d);
    REQUIRE(GiNaC::ex_to<GiNaC::numeric>(got.subs(sub))
                .is_equal(GiNaC::ex_to<GiNaC::numeric>(want.subs(sub))));
  }
}

// The box hands the elimination counters to the driver the way it hands the phase
// timers: a thread's scratch outlives the probe, so the probe's share is the delta.
TEST(firefly_box_reports_the_probe_elimination_fill_in) {
  secred::SymbolicFFEvaluator ev = evaluator_of();
  secred::SecRedFireFlyBlackBox box(ev, 1);
  box.prime_changed();
  REQUIRE_EQ(box.fwd_merges(), std::uint64_t(0));
  const std::vector<firefly::FFInt> x = {firefly::FFInt(5), firefly::FFInt(7)};
  box(x);
  REQUIRE(box.fwd_merges() > 0);
  REQUIRE(box.fwd_moved() > 0);
  REQUIRE(box.fwd_gathered() > 0);
  REQUIRE(box.fwd_pivots() > 0); // the box eliminates the learn's selected subset
  // The first probe's delta can carry a previous test's unreported tail (the scratch is
  // thread_local and outlives the box), so compare the deltas of two later probes: in
  // steady state each probe adds exactly its own point's work.
  box(x);
  const std::uint64_t a = box.fwd_merges();
  box(x);
  const std::uint64_t b = box.fwd_merges();
  REQUIRE(b > a);
  box(x);
  REQUIRE_EQ(box.fwd_merges() - b, b - a);
}

// Two workers that return from RatReconst::interpolate() for the same function both
// ask for the next sampling plan; in FireFly 2.0.3 that getter moved and cleared
// needed_feed_vec without mutex_status, so the two moves raced.  Hammer the hand-off
// from several threads.  A normal build only checks that nothing is lost or corrupted;
// under ThreadSanitizer the pre-patch getter reports a data race here even with an
// empty plan, and the patched getter is clean.  This test ends the file because it
// constructs a bare RatReconst after this file's in-process reconstruction.
TEST(firefly_needed_feed_vec_handoff_is_serialized) {
  firefly::RatReconst rec(2);
  constexpr unsigned kThreads = 4;
  constexpr unsigned kIterations = 20000;
  std::atomic<unsigned> ready{0};
  std::atomic<unsigned> calls{0};
  std::atomic<unsigned> non_empty{0};
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (unsigned t = 0; t < kThreads; ++t)
    threads.emplace_back([&]() {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
      }
      for (unsigned i = 0; i < kIterations; ++i) {
        const std::vector<std::pair<std::uint32_t, std::uint32_t>> plan =
            rec.get_needed_feed_vec();
        if (!plan.empty())
          non_empty.fetch_add(1, std::memory_order_relaxed);
        calls.fetch_add(1, std::memory_order_relaxed);
      }
    });
  while (ready.load(std::memory_order_acquire) != kThreads) {
  }
  start.store(true, std::memory_order_release);
  for (auto &t : threads)
    t.join();
  REQUIRE_EQ(calls.load(), kThreads * kIterations);
  REQUIRE_EQ(non_empty.load(), 0u);
}

// The second getter in that patch: get_zi_orders() now also takes mutex_status and
// refuses to read coef_mat while another worker owns the interpolation.  With no
// interpolation in flight it must still answer normally from several threads.
TEST(firefly_get_zi_orders_is_serialized) {
  firefly::RatReconst rec(2);
  constexpr unsigned kThreads = 4;
  constexpr unsigned kIterations = 20000;
  std::atomic<unsigned> ready{0};
  std::atomic<unsigned> calls{0};
  std::atomic<unsigned> unexpected{0};
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (unsigned t = 0; t < kThreads; ++t)
    threads.emplace_back([&]() {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
      }
      for (unsigned i = 0; i < kIterations; ++i) {
        const auto orders = rec.get_zi_orders();
        // zi == 1: one order, nothing solved yet, system size 0.
        if (orders.first.size() != 1 || orders.first.front().second != 0 ||
            orders.second != 0)
          unexpected.fetch_add(1, std::memory_order_relaxed);
        calls.fetch_add(1, std::memory_order_relaxed);
      }
    });
  while (ready.load(std::memory_order_acquire) != kThreads) {
  }
  start.store(true, std::memory_order_release);
  for (auto &t : threads)
    t.join();
  REQUIRE_EQ(calls.load(), kThreads * kIterations);
  REQUIRE_EQ(unexpected.load(), 0u);
}
