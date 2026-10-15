#pragma once
// The reconstruction phase's memory budget.
//
// The reconstruction solves each prime in parallel, and each worker holds the reduced
// system for as long as it runs: `FwdStore` keeps one row per solved integral -- the ids
// and the values, plus `row_of` per integral id -- and a worker reuses that store for
// every prime it takes, so the phase's memory is (workers x store) and it is not given
// back when a prime ends.  Nothing bounded it: the pool was `gthreads` clipped by the
// prime count, while `max_memory` and `memory_reserve` gated the sector stage only.
//
// At F1 balance n=0 (64 workers, a plan of 319,589,746 terms) that pool put the run's
// peak at 391.31 GB, of which the plan and the global system already resident before the
// pool started are ~48 GiB and the stores the rest: 64 x 18 B x 319,589,746 = 343 GiB.
// n=1's plan has 387,712,396 terms, i.e. 425 GiB of stores alone at 64 workers -- past
// the 480 GiB whole-machine ceiling the driver watching that machine enforced, before
// the run's own earlier phases are even added in.  It was killed in the reconstruction
// (rc=143) with the machine at 486 GiB.
//
// 18 B per term is the calibrated figure: 12 B of id + value per term, times the 1.5
// average tail capacity of a vector grown by doubling, plus 4 B per integral id for
// `row_of`, which `reset` sizes eagerly.  It is a model of a store's cost, not a
// measurement of the run's -- which is why `reconstruct_ffred` also keeps a backstop
// that retires workers once the process's own RSS passes the ceiling the pool was sized
// against.

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace secred {

// The per-worker store model: ids and values of every term of the eliminated system,
// plus `row_of` for every integral id (`has` is a 64th of that, and folds in here).
inline constexpr std::uint64_t kStoreBytesPerTerm = 18;
inline constexpr std::uint64_t kStoreBytesPerId = 4;

inline std::uint64_t reconstruction_worker_bytes(std::size_t terms, std::size_t store_ids) {
  return static_cast<std::uint64_t>(terms) * kStoreBytesPerTerm +
         static_cast<std::uint64_t>(store_ids) * kStoreBytesPerId;
}

// The ceiling the stores may fill up to: the configured limit, never above what the
// machine actually has, less the reserve that stays free below it.  The sector stage
// reads the same two settings the same way (`avail = maxMemory - used`, admit while
// `avail > memoryReserve`); clamping to the machine is what makes a configured 512 GiB
// on a smaller box mean "as much as there is" instead of a promise the allocator cannot
// keep.
inline std::uint64_t reconstruction_ceiling(std::uint64_t max_memory,
                                            std::uint64_t memory_reserve,
                                            std::uint64_t machine_bytes) {
  std::uint64_t ceiling = max_memory;
  if (machine_bytes != 0 && machine_bytes < ceiling)
    ceiling = machine_bytes;
  return ceiling > memory_reserve ? ceiling - memory_reserve : 0;
}

// How many workers fit: what is left of the ceiling after the structures already
// resident (`used` -- the plan, the global system and the output arrays among them)
// buys one worker's store each.  Never zero: one worker is the smallest pool that makes
// progress, and a single store that does not fit is a task too big for the machine, on
// which the sector stage's own gate has already had its say.
inline unsigned reconstruction_worker_cap(std::uint64_t ceiling, std::uint64_t used,
                                          std::uint64_t per_worker, unsigned nthreads,
                                          unsigned nprime_max) {
  unsigned hard = std::min(nthreads, nprime_max);
  if (hard == 0)
    hard = 1;
  if (per_worker == 0)
    return hard; // no plan: nothing per worker to count
  const std::uint64_t free_bytes = ceiling > used ? ceiling - used : 0;
  std::uint64_t k = free_bytes / per_worker;
  if (k < 1)
    k = 1;
  if (k > hard)
    k = hard;
  return static_cast<unsigned>(k);
}

// Where the retire backstop stops retiring: half the pool the budget admitted.  Retiring
// is how a run that outgrew its own estimate saves itself, but a smaller pool is also
// the slower way to run the phase, so a ceiling the resident structures already exceed
// -- which no amount of retiring fixes -- must not ratchet the pool down to one worker.
inline unsigned reconstruction_trim_floor(unsigned cap) { return cap > 2 ? cap / 2 : 1; }

// True once the process has grown past the ceiling the pool was sized against.
inline bool reconstruction_over_budget(std::uint64_t used, std::uint64_t ceiling) {
  return used > ceiling;
}

} // namespace secred
