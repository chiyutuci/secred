# lib/firefly_patch — FireFly without the 300-prime cap, with locked sampling-plan getters

FireFly 2.0.3 ships a fixed table of exactly 300 63-bit primes and exits when a
reconstruction needs one more:

- `source/ReconstHelper.cpp` defines `firefly::primes()` as a 300-entry table;
- `source/RatReconst.cpp` has three `if (prime_number == 300) { ERROR_MSG(...); std::exit(EXIT_FAILURE); }`
  guards and two other places that assume 300 (`set_safe_interpolation`,
  `set_prime_to_max`);
- `Reconstructor::reconstruct(uint32_t prime_counter = 300)` defaults to that count
  (SecRed already passes its own `settings.max_primes`, so only the first two matter).

This directory holds patched copies of the two `.cpp` files, taken verbatim from
FireFly 2.0.3 (`firefly-master`, the same version the installed
`libfirefly.so` and headers report in `firefly/version.hpp`), with the changes
below: the prime-table changes, and fixes for two data races in the multi-threaded
sampling-plan hand-off that FireFly 2.0.3's public API cannot reach.
`RatReconst.cpp` refuses to compile against a different FireFly version.

## Changes

`ReconstHelper.cpp`
- The 300 primes are kept verbatim, in the same order.  The vector is no longer
  `const`, so it can grow; `primes()` returns the same table as before.
- New `firefly::ensure_primes(std::size_t n)` (declared in `firefly_patch.h`):
  appends the primes below the current last entry until the table holds `n`
  entries, using GMP's `mpz_probab_prime_p` (25 rounds) on descending odd
  candidates.  Thread-safe; calling it with a count the table already meets is a
  no-op.  `SecRed` calls it with `settings.max_primes + 1` before starting a
  FireFly reconstruction — the reconstructor indexes `primes()[prime_it]` up to
  `prime_counter - 1`, and the per-function interpolation reads
  `primes()[prime_number + 1]`.

`RatReconst.cpp`
- The three `prime_number == 300` guards become
  `prime_number >= primes().size()`, with a message that names the table rather
  than the constant.
- `set_safe_interpolation()` sets `interpolations` to the table size (300 unless
  `ensure_primes` was asked for more); `set_prime_to_max()` starts at the last
  table entry.
- `get_needed_feed_vec()` takes `mutex_status` and returns an empty plan while
  `is_interpolating` is set.  FireFly 2.0.3 moved and cleared `needed_feed_vec`
  without a lock.  `interpolate()` releases `mutex_status` around its private
  `interpolate()`, which fills the vector in `check_if_done()`; the worker then
  returns from `interpolate()` — the lock is gone — and only afterwards asks
  `get_needed_feed_vec()` for the plan.  A second worker can enter `interpolate()`
  for the same `RatReconst` in that window, so both move the same vector.  Every
  writer of the vector runs while `is_interpolating` is set, and that flag is
  set and cleared under `mutex_status`, so the patched getter is serialized both
  against the writers and against another consumer; the worker that owns the
  interpolation takes the plan when it returns, so returning empty loses none.
  ThreadSanitizer reports six races at the unlocked move in the unpatched
  getter and none in the patched one; the unit tests
  `firefly_patched_needed_feed_vec_is_linked` and
  `firefly_needed_feed_vec_handoff_is_serialized` pin, respectively, that the
  executable's getter is the one linked and that the hand-off survives four
  concurrent callers.
- `get_zi_orders()` gets the same treatment.  It takes `mutex_status` and read
  `coef_mat.size()`, but the private `interpolate()` appends to `coef_mat`
  (in `build_uni_gauss()`, `build_factor_uni_gauss()` and
  `build_homogenized_multi_gauss()`) while the public `interpolate()` holds that
  mutex released, so the lock does not exclude the writer — ThreadSanitizer
  reports the `emplace_back` against `get_zi_orders()` on a plain `box.yaml`
  run.  The patched getter returns no orders and a zero system size while
  `is_interpolating` is set; the caller schedules nothing for that round, and
  the worker that owns the interpolation asks again once the flag is cleared,
  scheduling from the newer state.  `firefly_patched_get_zi_orders_is_linked`
  pins the binding.
- The file defines the patched **methods** and none of `RatReconst`'s static
  data members.  `shift`, `curr_shift`, `global_anchor_points`,
  `singular_system_set` and `mutex_statics` are defined by `libfirefly.so`; the
  executable's references reach the library's objects through ELF copy
  relocations.

## How it takes effect

`secred` links the shared `libfirefly.so` and compiles these two files into the
executable.  Both the header-inlined `Reconstructor<BlackBox>` code and the
`RatReconst` methods called from the executable bind to the definitions in the
executable; the copies inside the `.so` are not used on SecRed's path.  The
static archive is placed before `${FIREFLY_LIBRARY}` in both link lines
(`CMakeLists.txt`, `tests/CMakeLists.txt`) so the linker takes the local
definitions.

That rule applies to **functions**, and deliberately not to data.  A data
definition in the executable wins the dynamic symbol lookup just as a function
one does, but the library's static initializer also runs: it constructs the
executable's copy of each `RatReconst` static (through its own GOT, relocated
to the executable's definition) and so does the executable's initializer, and
both register the same destructor.  Each object is then destroyed twice at
exit.  Under glibc's allocator that is `double free or corruption` and SIGABRT
(exit code -6) as soon as the vectors hold heap buffers — a correct
reconstruction followed by a crash on the way out; jemalloc silently tolerates
the second free.  Leaving the definitions to `libfirefly.so` keeps exactly one
construction and one destruction per static.  The executable still reaches them
through copy relocations (`R_X86_64_COPY`), and
`tests/test_firefly_global.cpp`'s
`firefly_ratreconst_statics_are_owned_by_libfirefly` pins the invariant: each
symbol must resolve into `libfirefly.so` or carry a copy relocation in the
executable — a definition in the executable's own objects would have neither.

## Licensing

These files are part of FireFly, Copyright (C) 2020 Jonas Klappert and Fabian
Lange, and are licensed under the GNU General Public License v3 (headers
retained).  They are copied into this tree because FireFly's cap cannot be
lifted through its public API.  SecRed's own sources are GPLv3 too (see the
top-level `LICENSE`), as is a binary that links FireFly.
