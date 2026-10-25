#pragma once

// Per-prime samples of the numeric global reduction.
//
// Reconstructing every coefficient from its prime residues is the expensive
// half of a numeric run.  A checkpoint holds the complete residue vector of
// one prime, so a resumed run can feed it to the CRT loop instead of solving
// again.  A record is only usable for the same reduction: the caller checks
// the prime, the targets and the master basis, and the directory is named
// after the spill's content hash.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace secred {

struct PrimeCheckpoint {
  unsigned index = 0;
  std::uint64_t prime = 0;
  std::vector<unsigned> dep_targets;
  std::vector<unsigned> masters;
  // Row-major: dep_targets.size() * masters.size() residues.
  std::vector<std::uint64_t> xout;
};

std::filesystem::path prime_checkpoint_path(const std::filesystem::path &dir,
                                            unsigned index);

bool write_prime_checkpoint(const std::filesystem::path &path,
                            const PrimeCheckpoint &checkpoint,
                            std::string *error);
bool read_prime_checkpoint(const std::filesystem::path &path,
                           PrimeCheckpoint &checkpoint, std::string *error);

// True when the record belongs to this prime and this basis layout.
bool prime_checkpoint_matches(const PrimeCheckpoint &checkpoint, unsigned index,
                              std::uint64_t prime,
                              const std::vector<unsigned> &dep_targets,
                              const std::vector<unsigned> &masters);

} // namespace secred
