#include "prime_checkpoint.h"

#include "resume.h"

namespace secred {

namespace {

void add_u32s(CheckpointWriter &writer, const std::vector<unsigned> &values) {
  writer.add_u32(static_cast<std::uint32_t>(values.size()));
  for (unsigned value : values)
    writer.add_u32(value);
}

bool read_u32s(CheckpointReader &reader, std::vector<unsigned> &values) {
  const std::uint32_t count = reader.u32();
  values.clear();
  values.reserve(count);
  for (std::uint32_t i = 0; i < count && reader.ok(); ++i)
    values.push_back(reader.u32());
  return reader.ok();
}

} // namespace

std::filesystem::path prime_checkpoint_path(const std::filesystem::path &dir,
                                            unsigned index) {
  return dir / (std::to_string(index) + ".ckpt");
}

bool write_prime_checkpoint(const std::filesystem::path &path,
                            const PrimeCheckpoint &checkpoint,
                            std::string *error) {
  CheckpointWriter writer;
  writer.add_u32(checkpoint.index);
  writer.add_u64(checkpoint.prime);
  add_u32s(writer, checkpoint.dep_targets);
  add_u32s(writer, checkpoint.masters);
  writer.add_u64(static_cast<std::uint64_t>(checkpoint.xout.size()));
  for (std::uint64_t sample : checkpoint.xout)
    writer.add_u64(sample);
  return write_checkpoint(path, kCheckpointTagPrime, kResumeCheckpointVersion,
                          writer.data(), error);
}

bool read_prime_checkpoint(const std::filesystem::path &path,
                           PrimeCheckpoint &checkpoint, std::string *error) {
  std::string payload;
  if (!read_checkpoint(path, kCheckpointTagPrime, kResumeCheckpointVersion,
                       payload, error))
    return false;

  CheckpointReader reader(payload);
  checkpoint.index = reader.u32();
  checkpoint.prime = reader.u64();
  const bool lists_ok = read_u32s(reader, checkpoint.dep_targets) &&
                        read_u32s(reader, checkpoint.masters);
  const std::uint64_t count = reader.u64();
  if (!lists_ok || !reader.ok()) {
    if (error)
      *error = path.string() + ": prime checkpoint payload is truncated";
    return false;
  }
  checkpoint.xout.clear();
  // One sample is eight bytes, so a plausible count never exceeds the payload.
  if (count > payload.size() / 8) {
    if (error)
      *error = path.string() + ": prime checkpoint sample count is too large";
    return false;
  }
  checkpoint.xout.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t i = 0; i < count && reader.ok(); ++i)
    checkpoint.xout.push_back(reader.u64());
  if (!reader.ok() || reader.remaining() != 0) {
    if (error)
      *error = path.string() + ": prime checkpoint payload is malformed";
    return false;
  }
  if (error)
    error->clear();
  return true;
}

bool prime_checkpoint_matches(const PrimeCheckpoint &checkpoint, unsigned index,
                              std::uint64_t prime,
                              const std::vector<unsigned> &dep_targets,
                              const std::vector<unsigned> &masters) {
  if (checkpoint.index != index || checkpoint.prime != prime)
    return false;
  if (checkpoint.dep_targets != dep_targets || checkpoint.masters != masters)
    return false;
  return checkpoint.xout.size() ==
         static_cast<std::size_t>(dep_targets.size()) * masters.size();
}

} // namespace secred
