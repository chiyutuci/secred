#pragma once

// Resuming an interrupted reduction.
//
// Everything here guards a single question: may this run trust what a previous
// run left in the cache directory?  The pieces are deliberately small --
// canonical hashing, checksummed files written through a rename, the run
// manifest, and the FireFly save-directory inspection -- so that each rejection
// path is testable without running a reduction.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "sector.h"

namespace secred {

// Bump when the layout of any checkpoint payload changes, or when the meaning
// of a stored field changes.  Every reader compares it.
inline constexpr std::uint32_t kResumeFormatVersion = 1;
inline constexpr std::uint32_t kResumeCheckpointVersion = 1;

inline constexpr const char *kManifestFile = ".secred_manifest";
inline constexpr const char *kResumeDirName = "resume";

inline constexpr std::uint32_t kCheckpointTagSector = 0x53454352u; // "SECR"
inline constexpr std::uint32_t kCheckpointTagPrime = 0x5052494du;  // "PRIM"
inline constexpr std::uint32_t kCheckpointTagFireFly = 0x46464b59u; // "FFKY"
inline constexpr std::uint32_t kCheckpointTagManifest = 0x4d414e49u; // "MANI"

// "SRCK" magic, version, tag, payload length, CRC-32 of the payload.
inline constexpr std::size_t kCheckpointHeaderBytes = 24;

// A fingerprint of the code that wrote the checkpoint.  A rebuild invalidates
// saved state instead of risking a format that silently means something else.
std::uint64_t resume_build_stamp();

// FNV-1a 64.  Used for run identity hashes, where collisions only cost a
// discarded cache, and for the incremental spill hash that names prime and
// FireFly checkpoints.
class ResumeHasher {
public:
  void add(const void *data, std::size_t size);
  void add_u8(std::uint8_t value);
  void add_u32(std::uint32_t value);
  void add_u64(std::uint64_t value);
  void add_int(std::int64_t value);
  void add_string(std::string_view text);
  void add_raw_integral(const RawIntegral &integral);
  std::uint64_t value() const { return _state; }
  std::string hex() const;

private:
  std::uint64_t _state = 1469598103934665603ULL;
};

// Canonical hash of a run's config.  `settings.cache_dir` and
// `settings.resume` are skipped: moving a cache directory, or choosing to
// resume, does not change the reduction.  Map keys are sorted; sequence order
// is kept.
std::uint64_t resume_config_hash(const YAML::Node &config);

// Hash a file's bytes into ResumeHasher.  `value` is written only on success.
// Used for the completed spill, whose bytes name the global stage's
// checkpoints: a replayed pass reproduces them exactly.
bool hash_file(const std::filesystem::path &path, std::uint64_t &value,
               std::string *error);

// Content key for one sector pass.  The run identity alone is not enough:
// later passes run with a raised truncation limit or with request exceptions,
// and their sectors produce different equations.
std::string sector_pass_key(std::uint64_t input_hash, unsigned pass,
                            bool truncation_active, unsigned truncation_l,
                            unsigned level_offset, unsigned dots,
                            bool exempt_cut_requests);

// Little-endian payload writer/reader shared by the checkpoint kinds.
class CheckpointWriter {
public:
  void add_u8(std::uint8_t value);
  void add_u32(std::uint32_t value);
  void add_u64(std::uint64_t value);
  void add_int(std::int64_t value);
  void add_bool(bool value) { add_u8(value ? 1 : 0); }
  void add_string(std::string_view text);
  void add_raw_integral(const RawIntegral &integral);
  void add_raw_integrals(const std::vector<RawIntegral> &integrals);
  const std::string &data() const { return _bytes; }
  std::string &data() { return _bytes; }

private:
  std::string _bytes;
};

class CheckpointReader {
public:
  explicit CheckpointReader(std::string_view bytes) : _bytes(bytes) {}
  bool ok() const { return _ok; }
  std::size_t remaining() const { return _ok ? _bytes.size() - _pos : 0; }
  std::uint8_t u8();
  std::uint32_t u32();
  std::uint64_t u64();
  std::int64_t int_value();
  bool boolean() { return u8() != 0; }
  std::string string();
  RawIntegral raw_integral();
  std::vector<RawIntegral> raw_integrals();

private:
  bool need(std::size_t n);

  std::string_view _bytes;
  std::size_t _pos = 0;
  bool _ok = true;
};

// Write `payload` to `path` atomically: a temporary file in the same directory,
// then a rename.  The header carries the tag, the format version, the exact
// payload length and a CRC-32, so a torn or stale file is rejected on read.
bool write_checkpoint(const std::filesystem::path &path, std::uint32_t tag,
                      std::uint32_t version, std::string_view payload,
                      std::string *error);
bool read_checkpoint(const std::filesystem::path &path, std::uint32_t tag,
                     std::uint32_t version, std::string &payload,
                     std::string *error);

struct ResumeManifest {
  std::uint32_t format = kResumeFormatVersion;
  std::uint64_t build = 0;
  std::uint64_t input_hash = 0;

  bool operator==(const ResumeManifest &) const = default;
};

bool write_manifest(const std::filesystem::path &path,
                    const ResumeManifest &manifest, std::string *error);
bool read_manifest(const std::filesystem::path &path, ResumeManifest &manifest,
                   std::string *error);

// True when the file is a complete gzip stream.  A checkpoint written by a
// killed process can end mid-stream; FireFly would abort the whole run on one.
bool gzip_file_ok(const std::filesystem::path &path);

struct FireFlySaveStatus {
  bool usable = false;
  unsigned prime = 0;
  // Files the preparation repaired (truncated probes, promoted or discarded
  // state temporaries, stray files).  Only a diagnostic.
  unsigned repairs = 0;
  std::string reason;
};

// Prepare an `ff_save` directory for FireFly's resume_from_saved_state() and
// report where it continues.  A killed process leaves the directory in one of
// the states save_state()/the probe writer can be interrupted in: state
// temporaries not yet renamed into place, probes of a prime the state already
// finished, and a probe stream whose last appended gzip member was cut.  This
// picks each function's most advanced complete state, promotes or discards its
// temporary, and leaves exactly one state and one probe file per function, the
// probe holding only complete lines for the prime the state continues from.
// `items` is the number of interpolated functions and `nparams` the number of
// parameters; both are also checked against FireFly's own bookkeeping
// (validation.gz, anchor_points, shift) because FireFly's resume paths std::exit()
// on a mismatch.  The returned status is usable only when FireFly can read the
// whole directory without exiting; the caller starts over otherwise.
FireFlySaveStatus prepare_firefly_save(const std::filesystem::path &ff_save,
                                       unsigned items, unsigned nparams);

// The output layout saved FireFly states belong to.  The spill hash fixes the
// equations, but the same equations can still be turned into different black
// boxes -- a different master basis, target order or item mapping -- and
// FireFly answers such a mismatch by exiting the process.  This key pins the
// evaluator's inputs and outputs so its save directory is only trusted when
// they are the ones it was written for.
struct FireFlyPlanKey {
  unsigned nparams = 0;
  std::vector<unsigned> dep_targets;
  std::vector<unsigned> masters;
  std::vector<std::uint32_t> item_master;
  std::vector<std::uint32_t> item_col;
  std::vector<std::uint32_t> item_begin;

  bool operator==(const FireFlyPlanKey &) const = default;
};

bool write_firefly_plan_key(const std::filesystem::path &path,
                            const FireFlyPlanKey &key, std::string *error);
bool read_firefly_plan_key(const std::filesystem::path &path,
                           FireFlyPlanKey &key, std::string *error);

// Move the process into `dir` for a scope (FireFly's save paths are relative to
// the working directory), restoring the old one on every exit path.
class ScopedCurrentPath {
public:
  explicit ScopedCurrentPath(const std::filesystem::path &dir);
  ~ScopedCurrentPath();
  ScopedCurrentPath(const ScopedCurrentPath &) = delete;
  ScopedCurrentPath &operator=(const ScopedCurrentPath &) = delete;
  bool ok() const { return _ok; }
  explicit operator bool() const { return _ok; }

private:
  std::filesystem::path _old;
  bool _ok = false;
};

// What the global stage needs to find and validate its checkpoints.
struct ResumeControl {
  std::string cache_dir;
  std::uint64_t spill_hash = 0;
  std::uint64_t input_hash = 0;
};

// Directory names for the three checkpoint kinds.  They are keyed by content:
// a sector pass by its settings, a prime set and the FireFly save by the spill
// hash, so checkpoints from another configuration can never be read as this
// run's.
std::filesystem::path sector_resume_dir(const std::string &cache_dir,
                                        const std::string &pass_key);
std::filesystem::path prime_resume_dir(const ResumeControl &control,
                                       unsigned max_primes);
std::filesystem::path firefly_resume_dir(const ResumeControl &control,
                                         unsigned max_primes);

} // namespace secred
