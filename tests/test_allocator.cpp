#include "test_runner.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

// The allocator this build actually calls.
//
// CMakeLists.txt makes jemalloc a required dependency and stops the configure when the
// library is missing, but "the library is on the link line" is not the same claim as
// "the program's malloc is jemalloc's": a static archive the linker never pulls from, or
// a link line that reaches libc first, gives a binary that links jemalloc cleanly and
// still allocates from glibc.  That failure is the expensive kind -- it costs a factor
// of two on the sector stage (F234 at 64 threads: 1min1s -> 27.8s) and changes nothing
// else, so nothing else would notice it.
//
// mallctl is jemalloc's control interface and exists in no other allocator, so it is the
// probe.  It is declared here rather than included from jemalloc/jemalloc.h because the
// library and its headers are separate packages on most distributions and the header is
// not needed for anything else.
extern "C" int mallctl(const char *name, void *oldp, size_t *oldlenp, void *newp,
                       size_t newlen);

// malloc_conf is the second half of the same argument: jemalloc defines it, glibc does
// not.  It is null unless the run set a configuration string (jemalloc 5.3 on this box
// reports null here), which is why the check is on the symbol resolving at link time
// rather than on its value.  It is referenced as a data symbol rather than called, so a
// wrapper that merely forwards mallctl would not satisfy it.
extern "C" const char *malloc_conf;

TEST(the_linked_allocator_is_jemalloc) {
  const char *version = nullptr;
  size_t len = sizeof(version);
  REQUIRE_EQ(mallctl("version", &version, &len, nullptr, 0), 0);
  REQUIRE(version != nullptr);
  std::cout << "  [allocator] jemalloc " << version << " (malloc_conf=\""
            << (malloc_conf ? malloc_conf : "(null)") << "\")\n";
}

// The stronger claim, when the library was built with statistics: jemalloc counts the
// bytes it hands out per thread, so a malloc that moves that counter is a malloc that
// went through jemalloc.  Under glibc the counter stays put -- and under glibc this
// whole test does not link.  Distributions vary in whether stats are compiled in, so an
// unavailable counter is reported rather than failed; the linkage checks above are the
// ones that must hold everywhere.
TEST(malloc_goes_through_jemalloc) {
  std::uint64_t before = 0, after = 0;
  size_t sz = sizeof(before);
  if (mallctl("thread.allocated", &before, &sz, nullptr, 0) != 0) {
    std::cout << "  [allocator] thread.allocated unavailable (jemalloc built without "
                 "stats); ownership not verified here\n";
    return;
  }
  void *p = std::malloc(4096);
  REQUIRE(p != nullptr);
  REQUIRE_EQ(mallctl("thread.allocated", &after, &sz, nullptr, 0), 0);
  std::free(p);
  REQUIRE_MSG(after > before,
              "a 4096-byte malloc did not move jemalloc's per-thread counter, so the "
              "allocation did not come from jemalloc");
}
