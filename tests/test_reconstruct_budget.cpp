#include "test_runner.h"

#include "reconstruct_budget.h"

#include <cstdint>

using secred::reconstruction_ceiling;
using secred::reconstruction_over_budget;
using secred::reconstruction_trim_floor;
using secred::reconstruction_worker_bytes;
using secred::reconstruction_worker_cap;

// The reconstruction's worker pool is sized by a memory budget rather than by the
// requested thread count alone, because one worker's `FwdStore` holds the whole reduced
// system for the run and the pool's memory is that store times the pool.  The numbers
// below are the two F1 runs that set the calibration: balance n=0 completed at 391.31 GB
// with 64 workers on a 319,589,746-term plan (the stores are 343 GiB of that, the plan
// and the global system the other ~48 GiB), and balance n=1's 387,712,396-term plan ran
// past the 480 GiB whole-machine ceiling the run was watched against and was killed in
// the reconstruction.

namespace {

constexpr std::uint64_t kGiB = 1024ull * 1024 * 1024;
constexpr std::uint64_t kMiB = 1024ull * 1024;

// F1 balance n=0 / n=1 as the runs reported them.  `kStructuresBytes` is what the n=0
// peak leaves for everything the pool does not own -- the plan, the global system and
// the output arrays -- i.e. 391.31 GB minus the stores the model below accounts for.
constexpr std::size_t kN0Terms = 319589746;
constexpr std::size_t kN1Terms = 387712396;
constexpr std::uint64_t kN0PeakBytes = 419832430592ull; // 391.31 GB, as reported
constexpr std::uint64_t kN0StoreBytes = 368420342272ull; // 343 GiB, as documented
constexpr std::uint64_t kStructuresBytes = kN0PeakBytes - kN0StoreBytes;

// This machine: MemTotal 528,180,040 kB, and the defaults the two settings carry.
constexpr std::uint64_t kMachineBytes = 528180040ull * 1024ull;
constexpr std::uint64_t kMaxMemory = 512ull * kGiB;
constexpr std::uint64_t kMemoryReserve = 60ull * kGiB;
constexpr std::uint64_t kWatchdogBytes = 480ull * kGiB; // the driver's whole-machine line
constexpr unsigned kNthreads = 64;
constexpr unsigned kNprimeMax = 201; // max_primes + 1

std::uint64_t n0_per_worker() { return reconstruction_worker_bytes(kN0Terms, 0); }
std::uint64_t n1_per_worker() { return reconstruction_worker_bytes(kN1Terms, 0); }
std::uint64_t machine_ceiling() {
  return reconstruction_ceiling(kMaxMemory, kMemoryReserve, kMachineBytes);
}

} // namespace

// The model's calibration: 64 workers on n=0's plan must come to the 343 GiB of stores
// the measured 391.31 GB peak leaves room for.  This is the number the 18 B per term
// stands on, and the one a different constant would move.
TEST(store_model_matches_the_measured_n0_stores) {
  const std::uint64_t stores = 64 * n0_per_worker();
  const std::uint64_t diff = stores > kN0StoreBytes ? stores - kN0StoreBytes
                                                    : kN0StoreBytes - stores;
  REQUIRE(diff < kN0StoreBytes / 100); // within 1%
}

// And the model may not claim more than the run used: the stores alone stay under the
// measured peak, leaving the structures already resident a positive share of it.
TEST(store_model_leaves_the_resident_structures_under_the_peak) {
  REQUIRE(64 * n0_per_worker() < kN0PeakBytes);
  REQUIRE(kStructuresBytes > 0);
}

TEST(worker_bytes_counts_terms_and_integral_ids) {
  // 12 B of id + value per term at the doubling-grown tail's 1.5 average, plus row_of.
  REQUIRE(reconstruction_worker_bytes(1000, 0) == 18000);
  REQUIRE(reconstruction_worker_bytes(0, 1000) == 4000);
  REQUIRE(reconstruction_worker_bytes(1000, 1000) == 22000);
}

// A budget the machine cannot back is not a budget: 512 GiB configured on a box with
// less than that must read as the box's own total.
TEST(ceiling_is_clamped_to_the_machine) {
  REQUIRE(reconstruction_ceiling(512 * kGiB, 60 * kGiB, 100 * kGiB) == 40 * kGiB);
}

TEST(ceiling_keeps_a_limit_below_the_machine) {
  REQUIRE(reconstruction_ceiling(200 * kGiB, 60 * kGiB, 512 * kGiB) == 140 * kGiB);
}

// The reserve is subtracted before the clamp can make it meaningless, so a limit below
// the reserve saturates at zero instead of wrapping around.
TEST(ceiling_saturates_when_the_reserve_exceeds_the_limit) {
  REQUIRE(reconstruction_ceiling(30 * kGiB, 60 * kGiB, 512 * kGiB) == 0);
  REQUIRE(reconstruction_ceiling(60 * kGiB, 60 * kGiB, 512 * kGiB) == 0);
}

TEST(ceiling_ignores_an_unknown_machine_total) {
  REQUIRE(reconstruction_ceiling(512 * kGiB, 60 * kGiB, 0) == 452 * kGiB);
}

// n=0's pool fits: the run that produced the reference numbers keeps all 64 threads, so
// sizing by budget does not change the configuration the earlier results were measured
// in.
TEST(n0_plan_keeps_the_requested_threads) {
  REQUIRE(reconstruction_worker_cap(machine_ceiling(), kStructuresBytes, n0_per_worker(),
                                    kNthreads, kNprimeMax) == kNthreads);
}

// n=1's pool does not: this is the run that was killed, and the cap must shorten it.
TEST(n1_plan_is_capped_below_the_requested_threads) {
  const unsigned cap = reconstruction_worker_cap(machine_ceiling(), kStructuresBytes,
                                                 n1_per_worker(), kNthreads, kNprimeMax);
  REQUIRE(cap < kNthreads);
  REQUIRE(cap >= 1);
}

// What the cap is for: whatever it admits, the structures already resident plus the
// stores the pool will allocate stay under the ceiling -- and with the machine's own
// ceiling under the driver's line, under that too.
TEST(admitted_pool_stays_under_the_ceiling) {
  const std::uint64_t ceiling = machine_ceiling();
  REQUIRE(ceiling < kWatchdogBytes);
  for (std::uint64_t per_worker : {n0_per_worker(), n1_per_worker()}) {
    const unsigned cap =
        reconstruction_worker_cap(ceiling, kStructuresBytes, per_worker, kNthreads, kNprimeMax);
    REQUIRE(kStructuresBytes + cap * per_worker <= ceiling);
  }
}

// A plan with no terms costs nothing per worker, and a single prime cannot use more than
// one worker however many threads were asked for.
TEST(cap_respects_the_prime_count_and_the_plan) {
  REQUIRE(reconstruction_worker_cap(kGiB, 0, 1000, 64, 3) == 3);
  REQUIRE(reconstruction_worker_cap(kGiB, 0, 0, 64, kNprimeMax) == kNthreads);
}

// One worker is the floor: a machine on which a single store does not fit is a task too
// big for it, and running one prime at a time is the only thing left to try.
TEST(cap_never_falls_below_one_worker) {
  REQUIRE(reconstruction_worker_cap(0, 0, n1_per_worker(), kNthreads, kNprimeMax) == 1);
  REQUIRE(reconstruction_worker_cap(kGiB, 100 * kGiB, n1_per_worker(), kNthreads, kNprimeMax) ==
          1);
  REQUIRE(reconstruction_worker_cap(kGiB, 0, kGiB, 0, kNprimeMax) == 1);
}

TEST(trim_floor_is_half_the_admitted_pool) {
  REQUIRE(reconstruction_trim_floor(60) == 30);
  REQUIRE(reconstruction_trim_floor(64) == 32);
  REQUIRE(reconstruction_trim_floor(3) == 1);
  REQUIRE(reconstruction_trim_floor(2) == 1);
  REQUIRE(reconstruction_trim_floor(1) == 1);
}

// Reaching the ceiling exactly is inside the budget; the backstop is for going past it.
// A ceiling of zero is a configuration with no headroom at all, so anything resident is
// already over it.
TEST(over_budget_is_reaching_past_the_ceiling) {
  REQUIRE(!reconstruction_over_budget(100 * kMiB, 100 * kMiB));
  REQUIRE(reconstruction_over_budget(100 * kMiB + 1, 100 * kMiB));
  REQUIRE(!reconstruction_over_budget(0, kMiB));
  REQUIRE(reconstruction_over_budget(kMiB, 0));
}
