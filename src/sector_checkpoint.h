#pragma once

// One completed sector's contribution to a pass: the equations it wrote to the
// spill (with the global commit order they went in), the state it contributed
// to the pass context, and the statistics it accumulated.
//
// The equation bytes dominate a record, so a running sector streams them to
// its checkpoint file as the spill accepts them instead of holding a second
// copy in memory.  The file is only renamed into place once the sector has
// finished and its final metadata is appended; an abandoned record leaves no
// readable checkpoint behind.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "resume.h"
#include "sector.h"

struct SectorCheckpoint {
  unsigned id = 0;
  unsigned index = 0;

  // The pass state when the sector started: its seed targets and the
  // uptargets its (already finished) parents had contributed.  A record may
  // only be replayed when both still match.
  std::vector<RawIntegral> input_targets;
  std::vector<RawIntegral> input_uptargets;

  // Emitted in run order, duplicates included: masters_sector is a plain
  // concatenation, so replay must reproduce the vector exactly.
  std::vector<RawIntegral> masters;

  // Inserted into ctx.uptargets[sector of the integral].
  std::vector<RawIntegral> emitted_uptargets;

  // The sector's final ctx.cutRequests[id].
  std::vector<RawIntegral> cut_requests;

  unsigned narrowing_success = 0;
  unsigned narrowing_fail = 0;
  unsigned truncated = 0;
  unsigned lifted = 0;
  unsigned box_raised = 0;
  unsigned box_depth = 0;
  unsigned box_rank = 0;
  unsigned scope_skipped = 0;
  unsigned raw_equations = 0;
  std::uint64_t probe_nanos = 0;
  std::uint64_t full_nanos = 0;

  SectorTiming timing;

  struct Batch {
    std::uint64_t seq = 0;
    std::vector<GlobalEquation> equations;
  };
  std::vector<Batch> batches;
};

// Payload layout, written after the checkpoint container's 24-byte header:
//
//   u32 version, u32 id, u32 index,
//   raw_integrals input_targets, raw_integrals input_uptargets,
//   { u32 kSectorBatchTag, u64 seq, u32 equations, equations... }*,
//   u32 kSectorTailTag, masters, emitted_uptargets, cut_requests, statistics
//
// A batch or tail record begins with its tag, so a reader can walk the payload
// without knowing the batch count in advance.
inline constexpr std::uint32_t kSectorPayloadVersion = 2;
inline constexpr std::uint32_t kSectorBatchTag = 0x42415443u; // "BATC"
inline constexpr std::uint32_t kSectorTailTag = 0x5441494cu;  // "TAIL"

std::string serialize_sector_checkpoint(const SectorCheckpoint &checkpoint);
bool parse_sector_checkpoint(std::string_view bytes,
                             SectorCheckpoint &checkpoint, std::string *error);

// The record for one sector inside a pass directory.
std::filesystem::path sector_checkpoint_path(const std::filesystem::path &pass_dir,
                                             unsigned id);

// Where a batch sits in a checkpoint file, found by one streaming pass.
struct SectorCheckpointBatchRef {
  std::uint64_t seq = 0;
  std::uint64_t offset = 0; // payload offset of the batch's tag
  std::uint32_t equations = 0;
};

struct SectorCheckpointFile {
  SectorCheckpoint meta;
  std::vector<SectorCheckpointBatchRef> batches;
};

// Validate the container (magic, version, tag, length, CRC-32) and collect the
// metadata and batch descriptors without holding the equations in memory.
bool read_sector_checkpoint_file(const std::filesystem::path &path,
                                 SectorCheckpointFile &file, std::string *error);

// Read one batch's equations back.  The file was CRC-checked by
// read_sector_checkpoint_file().
bool read_sector_checkpoint_batch(const std::filesystem::path &path,
                                  const SectorCheckpointBatchRef &batch,
                                  std::vector<GlobalEquation> &equations,
                                  std::string *error);

// The writer a running sector owns.  set_inputs() must precede open(), which
// writes the prologue.  Only the owning thread touches it.
class SectorCheckpointStream {
public:
  SectorCheckpointStream(unsigned id, unsigned index,
                         std::filesystem::path path);
  ~SectorCheckpointStream();
  SectorCheckpointStream(const SectorCheckpointStream &) = delete;
  SectorCheckpointStream &operator=(const SectorCheckpointStream &) = delete;

  void set_inputs(const std::set<RawIntegral> &targets,
                  const std::set<RawIntegral> &uptargets);
  bool open(std::string *error);
  bool failed() const { return _failed; }
  bool finished() const { return _finished; }

  void add_master(const RawIntegral &integral) {
    _meta.masters.push_back(integral);
  }
  void add_uptarget(const RawIntegral &integral) {
    _meta.emitted_uptargets.push_back(integral);
  }

  // Final metadata (statistics, timing, cut requests), written by finish().
  SectorCheckpoint &meta() { return _meta; }
  const SectorCheckpoint &meta() const { return _meta; }

  // Stream one spilled equation batch.  Returns false and marks the stream
  // failed on an I/O error; the sector itself carries on without a record.
  bool add_batch(std::uint64_t seq, const std::vector<GlobalEquation> &equations,
                 std::string *error = nullptr);

  // Append the tail, write the header, and rename the file into place.
  bool finish(std::string *error);

  // Writer interface shared with the in-memory serializer; add_batch() and
  // finish() build their records from these.
  bool add_u8(std::uint8_t value);
  bool add_u32(std::uint32_t value);
  bool add_u64(std::uint64_t value);
  bool add_bool(bool value) { return add_u8(value ? 1 : 0); }
  bool add_string(std::string_view text);
  bool add_raw_integral(const RawIntegral &integral);
  bool add_raw_integrals(const std::vector<RawIntegral> &integrals);

private:
  bool put(const void *data, std::size_t size);
  bool flush();
  bool put_equation(const GlobalEquation &equation);
  void discard();

  std::filesystem::path _path;
  std::filesystem::path _temporary;
  std::ofstream _out;
  std::string _buffer;
  SectorCheckpoint _meta;
  std::uint64_t _payload_bytes = 0;
  std::uint32_t _crc = 0;
  bool _opened = false;
  bool _finished = false;
  bool _failed = false;
};
