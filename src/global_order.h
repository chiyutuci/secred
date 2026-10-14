// Integral numbering order of the global stage; family.cpp's _run_global_reduction
// orders the numbering table of the system it hands to the reduction.
// Both modes lead with popcount(sector) ASC and only differ inside a layer:
// Normal uses depth ASC then rank ASC, Balance the five balance keys (total
// power, depth, rank, max denominator power, max numerator power — see
// balance_less in sector.h).  Both fall back to sector mask ASC and then to the
// lexicographic order, so each is a total order and std::sort suffices.
//
// Header-only so the unit tests can drive the real sort: a change here moves the
// column order of the system file, the master basis and the global equation
// count.
#pragma once

#include <algorithm>
#include <bit>
#include <vector>

#include "sector.h"

namespace secred {

// Comparator of the global numbering.  Balance enables the five-key order within
// a layer; Normal keeps the depth/rank order.
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

// Sort the global integral table into numbering order (number = sorted index).
inline void order_global_integrals(std::vector<RawIntegral> &integrals,
                                   OrderingMode mode) {
  std::sort(integrals.begin(), integrals.end(),
            [mode](const RawIntegral &a, const RawIntegral &b) {
              return global_integral_less(a, b, mode);
            });
}

}  // namespace secred
