#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace secred {

inline constexpr std::uint64_t kStoreBytesPerTerm = 18;
inline constexpr std::uint64_t kStoreBytesPerId = 4;

// Estimated bytes held by one reconstruction worker.
inline std::uint64_t reconstruction_worker_bytes(std::size_t terms, std::size_t store_ids) {
  return static_cast<std::uint64_t>(terms) * kStoreBytesPerTerm +
         static_cast<std::uint64_t>(store_ids) * kStoreBytesPerId;
}

// Memory available to reconstruction after the reserve.
inline std::uint64_t reconstruction_ceiling(std::uint64_t max_memory,
                                            std::uint64_t memory_reserve,
                                            std::uint64_t machine_bytes) {
  std::uint64_t ceiling = max_memory;
  if (machine_bytes != 0 && machine_bytes < ceiling)
    ceiling = machine_bytes;
  return ceiling > memory_reserve ? ceiling - memory_reserve : 0;
}

// Worker count that fits in the free budget.
inline unsigned reconstruction_worker_cap(std::uint64_t ceiling, std::uint64_t used,
                                          std::uint64_t per_worker, unsigned nthreads,
                                          unsigned nprime_max) {
  unsigned hard = std::min(nthreads, nprime_max);
  if (hard == 0)
    hard = 1;
  if (per_worker == 0)
    return hard;
  const std::uint64_t free_bytes = ceiling > used ? ceiling - used : 0;
  std::uint64_t k = free_bytes / per_worker;
  if (k < 1)
    k = 1;
  if (k > hard)
    k = hard;
  return static_cast<unsigned>(k);
}

inline unsigned reconstruction_trim_floor(unsigned cap) { return cap > 2 ? cap / 2 : 1; }

inline bool reconstruction_over_budget(std::uint64_t used, std::uint64_t ceiling) {
  return used > ceiling;
}

}
