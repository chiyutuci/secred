// Integral numbering order of the global stage; family.cpp's _run_global_reduction
// orders the numbering table of the system it hands to the reduction.
// Both modes lead with popcount(sector) ASC and only differ inside a layer:
// Normal uses depth ASC then rank ASC, Balance the five balance keys (total
// power, depth, rank, max denominator power, max numerator power — see
// balance_less in sector.h).  Both fall back to sector mask ASC and then to the
// lexicographic order, so each is a total order and std::sort suffices.
//
// Ahead of all of this sits one optional key: the input file's `preferred`
// list.  It cannot be packed into the key below (all 64 bits are spoken for),
// so `order_global_integrals` applies it to the sorted table as a stable
// partition -- the preferred integrals are the simplest ones and take the
// lowest numbers, which is what the solver prefers as a master.
//
// Header-only so the unit tests can drive the real sort: a change here moves the
// column order of the system file, the master basis and the global equation
// count.
#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <set>
#include <vector>

#include "sector.h"

namespace secred {

// Comparator of the global numbering.  Balance enables the five-key order within
// a layer; Normal keeps the depth/rank order.  This is the order *within* a
// preferred group: `order_global_integrals` puts the input file's `preferred`
// integrals ahead of everything else, so a preferred integral with a larger key
// still takes a lower number than a non-preferred one with a smaller key.
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

// The decorate-sort-undecorate form of the comparator above.  Every key down to
// the sector mask is packed into one integer, so a comparison is two integer
// compares instead of ~16 passes over the index array (each of depth, rank,
// maxpos, maxneg and sector walks all propagators, and Balance walks them twice
// through balance_less).  On a five-million-row table that is the difference
// between ~4.5 s and well under a second.
//
// Field widths come from RawIntegral's own limits: the indices are int8_t and at
// most MAX_PROPS = 32 of them, so depth and rank are at most 32 * 127 = 4064 and
// 32 * 128 = 4096 (14 bits each, total 15), while maxpos is at most 127 (7 bits)
// and -maxneg at most 128 (8 bits).  63 bits in all.
//
// The keys sit in the head from the most significant field down, so comparing
// two heads as integers is the lexicographic comparison of the key sequences:
//
//   bits 63-58  popcount(sector)  6 bits (0..32)
//   bits 57-43  depth + rank     15 bits (Balance; 0 in Normal)
//   bits 42-29  depth            14 bits (0..4064)
//   bits 28-15  rank             14 bits (0..4096)
//   bits 14- 8  maxpos            7 bits (1..127)
//   bits  7- 0  -maxneg           8 bits (1..128)
//
// Normal leaves the three Balance-only fields at 0, so their constant values
// cannot reorder anything and the same layout yields its (depth, rank) order.
struct GlobalOrderKey {
  std::uint64_t head;    // the keys above, packed
  std::uint32_t sector;  // the sector mask, compared after the head
  std::uint32_t idx;     // position of the integral in the vector being sorted
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

// The key's order.  The last key, the lexicographic order of the integrals, is
// read back from the vector the keys point into: storing the index array in the
// key would triple its size for a key that decides only the tie inside one sector
// and one set of powers.
inline bool global_order_key_less(const GlobalOrderKey &a,
                                  const GlobalOrderKey &b,
                                  const std::vector<RawIntegral> &v) {
  if (a.head != b.head)
    return a.head < b.head;
  if (a.sector != b.sector)
    return a.sector < b.sector;
  return v[a.idx] < v[b.idx];
}

// Sort the global integral table into numbering order (number = sorted index).
// `preferred` is the input file's `preferred` list, or nullptr when the file has
// none.  A preferred integral leads the whole numbering, whatever its keys say.
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
  // Gather into numbering order.  `preferred` is a stable partition of the
  // sorted table, and two passes over the sorted keys are that partition for
  // free: the relative order inside each group is the key order either way, and
  // the empty case costs one `contains` per row.  One extra table, freed at the
  // end of the move: the caller has just released the distinct-integral set that
  // held the same elements, so the peak does not grow.
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

}  // namespace secred
