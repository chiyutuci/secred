#pragma once

// Restoring a saved unordered_map's iteration order.
//
// FireFly stores a reconstruction's coefficient maps (g_ni, g_di, combined_ni,
// combined_di) by walking them in iteration order and writes one entry per line.
// start_from_saved_file() reads them back with one emplace() per line.  For
// libstdc++'s unordered_map those insertions do not reproduce the saved order:
// an insertion into an empty bucket lands at the front of the whole list, and an
// insertion into a non-empty bucket at the front of that bucket's run, so the
// order depends on the insertion history, not on the key set.  FireFly prints a
// rational function's terms in this order (Polynomial::to_string() walks the
// vector built from the map) and has no sort in that path, so a resumed
// reconstruction printed the same terms in a different order than the
// uninterrupted run it continued -- mathematically equal, not byte-identical.
//
// A target order can be restored exactly with one observation about libstdc++'s
// layout: the iteration order is a concatenation of bucket runs, one contiguous
// run per non-empty bucket.  If the map is preallocated to the saved bucket
// count, inserting the runs in reverse order of appearance -- each run itself
// reversed -- rebuilds the target: every first insertion of a run finds its
// bucket empty and lands at the global head (later runs end up further back),
// and every later insertion of the same run lands at the front of that bucket.
//
// restore_saved_iteration_order() checks that the saved order has that shape
// before touching the map, verifies the result afterwards, and on any mismatch
// (a different standard library, a changed hash, an order that does not describe
// the map) rebuilds what the unpatched loader would have produced and returns
// false.  A caller can then keep the old behaviour rather than risk a layout
// that does not correspond to anything.

#include <cstddef>
#include <unordered_set>
#include <utility>
#include <vector>

namespace firefly_layout {

template <class Map, class Key>
bool restore_saved_iteration_order(Map &map, const std::vector<Key> &saved_order) {
  if (saved_order.size() != map.size())
    return false;
  if (saved_order.empty())
    return true;

  // The order must list exactly the map's keys, once each; only then can every
  // entry be extracted without losing one.
  {
    std::unordered_set<Key, typename Map::hasher> seen;
    seen.reserve(saved_order.size());
    for (const Key &key : saved_order) {
      if (map.find(key) == map.end() || !seen.insert(key).second)
        return false;
    }
  }

  const std::size_t buckets = map.bucket_count();
  if (buckets == 0)
    return false;
  const typename Map::hasher hash = map.hash_function();
  std::vector<std::size_t> bucket_of(saved_order.size());
  for (std::size_t i = 0; i != saved_order.size(); ++i)
    bucket_of[i] = hash(saved_order[i]) % buckets;

  // The target order must be a concatenation of bucket runs with each bucket
  // appearing in one run.  Anything else is not a libstdc++ layout this
  // construction can reproduce.
  std::vector<std::pair<std::size_t, std::size_t>> runs; // half-open [begin, end)
  for (std::size_t i = 0; i != saved_order.size();) {
    std::size_t j = i + 1;
    while (j != saved_order.size() && bucket_of[j] == bucket_of[i])
      ++j;
    runs.emplace_back(i, j);
    i = j;
  }
  {
    std::unordered_set<std::size_t> run_buckets;
    for (const auto &run : runs) {
      if (!run_buckets.insert(bucket_of[run.first]).second)
        return false;
    }
  }

  std::vector<typename Map::node_type> nodes;
  nodes.reserve(saved_order.size());
  for (const Key &key : saved_order)
    nodes.push_back(map.extract(key));

  map.clear();
  map.rehash(buckets);
  for (auto run = runs.rbegin(); run != runs.rend(); ++run) {
    for (std::size_t i = run->second; i != run->first; --i)
      map.insert(std::move(nodes[i - 1]));
  }

  std::size_t i = 0;
  bool restored = true;
  for (const auto &entry : map) {
    if (i == saved_order.size() || entry.first != saved_order[i]) {
      restored = false;
      break;
    }
    ++i;
  }
  if (restored && i == saved_order.size())
    return true;

  // Not restorable here: leave the map holding what a plain re-insertion of the
  // saved order produces -- the layout the unpatched loader had.
  nodes.clear();
  nodes.reserve(map.size());
  for (const Key &key : saved_order)
    nodes.push_back(map.extract(key));
  map.clear();
  map.rehash(0);
  for (auto &node : nodes)
    map.insert(std::move(node));
  return false;
}

} // namespace firefly_layout
