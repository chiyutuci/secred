
#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <set>
#include <vector>

#include "sector.h"

namespace secred {

// Global integral order: sector width first, then the mode's weight.
inline bool global_integral_less(const RawIntegral &a, const RawIntegral &b,
                                 OrderingMode mode) {
  unsigned pa = std::popcount(a.sector());
  unsigned pb = std::popcount(b.sector());
  if (pa != pb)
    return pa < pb;
  if (mode == OrderingMode::Balance) {
    if (balance_less(a, b))
      return true;
    if (balance_less(b, a))
      return false;
  } else {
    if (a.depth() != b.depth())
      return a.depth() < b.depth();
    if (a.rank() != b.rank())
      return a.rank() < b.rank();
  }
  if (a.sector() != b.sector())
    return a.sector() < b.sector();
  return a < b;
}

// Packed sort key for the global order.
struct GlobalOrderKey {
  std::uint64_t head;
  std::uint32_t sector;
  std::uint32_t idx;
};

inline GlobalOrderKey global_order_key(const RawIntegral &x, OrderingMode mode,
                                       std::uint32_t idx) {
  constexpr unsigned k_pop = 58, k_total = 43, k_depth = 29, k_rank = 15,
                     k_maxpos = 8, k_negmax = 0;
  GlobalOrderKey k;
  k.head = static_cast<std::uint64_t>(std::popcount(x.sector())) << k_pop;
  if (mode == OrderingMode::Balance) {
    k.head |= static_cast<std::uint64_t>(x.depth() + x.rank()) << k_total;
    k.head |= static_cast<std::uint64_t>(x.maxpos()) << k_maxpos;
    k.head |= static_cast<std::uint64_t>(-x.maxneg()) << k_negmax;
  }
  k.head |= static_cast<std::uint64_t>(x.depth()) << k_depth;
  k.head |= static_cast<std::uint64_t>(x.rank()) << k_rank;
  k.sector = x.sector();
  k.idx = idx;
  return k;
}

inline bool global_order_key_less(const GlobalOrderKey &a,
                                  const GlobalOrderKey &b,
                                  const std::vector<RawIntegral> &v) {
  if (a.head != b.head)
    return a.head < b.head;
  if (a.sector != b.sector)
    return a.sector < b.sector;
  return v[a.idx] < v[b.idx];
}

// Sort integrals into the global order, optionally preferring a set.
inline void order_global_integrals(std::vector<RawIntegral> &integrals,
                                   OrderingMode mode,
                                   const std::set<RawIntegral> *preferred = nullptr) {
  const std::size_t n = integrals.size();
  if (n < 2)
    return;
  std::vector<GlobalOrderKey> keys(n);
  for (std::size_t i = 0; i < n; ++i)
    keys[i] = global_order_key(integrals[i], mode,
                               static_cast<std::uint32_t>(i));
  std::sort(keys.begin(), keys.end(),
            [&integrals](const GlobalOrderKey &a, const GlobalOrderKey &b) {
              return global_order_key_less(a, b, integrals);
            });

  std::vector<RawIntegral> ordered;
  ordered.reserve(n);
  if (preferred) {
    for (std::size_t i = 0; i < n; ++i)
      if (preferred->contains(integrals[keys[i].idx]))
        ordered.emplace_back(integrals[keys[i].idx]);
    for (std::size_t i = 0; i < n; ++i)
      if (!preferred->contains(integrals[keys[i].idx]))
        ordered.emplace_back(integrals[keys[i].idx]);
  } else {
    for (std::size_t i = 0; i < n; ++i)
      ordered.emplace_back(integrals[keys[i].idx]);
  }
  integrals = std::move(ordered);
}

}
