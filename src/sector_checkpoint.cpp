#include "sector_checkpoint.h"

#include <algorithm>
#include <cstring>
#include <system_error>

#include <zlib.h>

namespace {

constexpr char kCheckpointMagic[4] = {'S', 'R', 'C', 'K'};
constexpr std::size_t kHeaderBytes = 24;

void put_u32_le(std::string &out, std::uint32_t value) {
  for (unsigned i = 0; i < 4; ++i)
    out.push_back(static_cast<char>((value >> (8 * i)) & 0xff));
}

void put_u64_le(std::string &out, std::uint64_t value) {
  for (unsigned i = 0; i < 8; ++i)
    out.push_back(static_cast<char>((value >> (8 * i)) & 0xff));
}

std::uint32_t get_u32(const char *p) {
  std::uint32_t value = 0;
  for (unsigned i = 0; i < 4; ++i)
    value |= static_cast<std::uint32_t>(static_cast<unsigned char>(p[i]))
             << (8 * i);
  return value;
}

std::uint64_t get_u64(const char *p) {
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i)
    value |= static_cast<std::uint64_t>(static_cast<unsigned char>(p[i]))
             << (8 * i);
  return value;
}

// A sequential reader over the file's payload that keeps the running CRC of
// every byte it consumes, so skipped equation bodies are still checked.
class FileCursor {
public:
  FileCursor(std::ifstream &in, std::uint64_t bytes)
      : _in(in), _remaining(bytes) {}

  bool ok() const { return _ok; }
  std::uint64_t remaining() const { return _remaining; }
  std::uint64_t offset() const { return _offset; }
  std::uint32_t crc() const { return _crc; }

  bool read(void *data, std::size_t size) {
    if (!_ok)
      return false;
    if (size > _remaining) {
      _ok = false;
      return false;
    }
    _in.read(static_cast<char *>(data), static_cast<std::streamsize>(size));
    if (!_in) {
      _ok = false;
      return false;
    }
    _crc = static_cast<std::uint32_t>(crc32(
        _crc, static_cast<const Bytef *>(data), static_cast<uInt>(size)));
    _offset += size;
    _remaining -= size;
    return true;
  }

  bool skip(std::uint64_t size) {
    char buffer[4096];
    while (size != 0) {
      const std::size_t take = static_cast<std::size_t>(
          std::min<std::uint64_t>(size, sizeof buffer));
      if (!read(buffer, take))
        return false;
      size -= take;
    }
    return true;
  }

  std::uint8_t u8() {
    std::uint8_t value = 0;
    read(&value, 1);
    return value;
  }

  std::uint32_t u32() {
    char bytes[4] = {};
    if (!read(bytes, 4))
      return 0;
    return get_u32(bytes);
  }

  std::uint64_t u64() {
    char bytes[8] = {};
    if (!read(bytes, 8))
      return 0;
    return get_u64(bytes);
  }

  bool boolean() { return u8() != 0; }

  std::string string() {
    const std::uint32_t size = u32();
    if (!_ok || size > _remaining) {
      _ok = false;
      return {};
    }
    std::string text(size, '\0');
    if (size != 0 && !read(text.data(), size))
      return {};
    return text;
  }

  RawIntegral raw_integral() {
    const std::uint8_t size = u8();
    if (!_ok || size > RawIntegral::MAX_PROPS) {
      _ok = false;
      return RawIntegral();
    }
    std::int8_t bytes[RawIntegral::MAX_PROPS] = {};
    if (size != 0 && !read(bytes, size))
      return RawIntegral();
    std::vector<int> values(size);
    for (unsigned i = 0; i < size; ++i)
      values[i] = bytes[i];
    return RawIntegral(std::move(values));
  }

  std::vector<RawIntegral> raw_integrals() {
    const std::uint32_t count = u32();
    if (!_ok || count > _remaining) {
      _ok = false;
      return {};
    }
    std::vector<RawIntegral> integrals;
    integrals.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
      integrals.push_back(raw_integral());
      if (!_ok) {
        integrals.clear();
        return integrals;
      }
    }
    return integrals;
  }

  bool skip_raw_integral() {
    const std::uint8_t size = u8();
    if (!_ok || size > RawIntegral::MAX_PROPS)
      return false;
    return skip(size);
  }

  bool skip_string() {
    const std::uint32_t size = u32();
    return _ok && skip(size);
  }

private:
  std::ifstream &_in;
  std::uint64_t _remaining = 0;
  std::uint64_t _offset = 0;
  std::uint32_t _crc = 0;
  bool _ok = true;
};

// CheckpointWriter cannot fail, but the record writers are shared with the
// streaming writer and report success, so give the in-memory one the same
// shape.
class BytesWriter {
public:
  explicit BytesWriter(secred::CheckpointWriter &writer) : _writer(writer) {}
  bool add_u8(std::uint8_t value) {
    _writer.add_u8(value);
    return true;
  }
  bool add_u32(std::uint32_t value) {
    _writer.add_u32(value);
    return true;
  }
  bool add_u64(std::uint64_t value) {
    _writer.add_u64(value);
    return true;
  }
  bool add_bool(bool value) {
    _writer.add_bool(value);
    return true;
  }
  bool add_string(std::string_view text) {
    _writer.add_string(text);
    return true;
  }
  bool add_raw_integral(const RawIntegral &integral) {
    _writer.add_raw_integral(integral);
    return true;
  }
  bool add_raw_integrals(const std::vector<RawIntegral> &integrals) {
    _writer.add_raw_integrals(integrals);
    return true;
  }

private:
  secred::CheckpointWriter &_writer;
};

template <typename Writer>
bool write_prologue(Writer &writer, const SectorCheckpoint &meta) {
  return writer.add_u32(kSectorPayloadVersion) && writer.add_u32(meta.id) &&
         writer.add_u32(meta.index) &&
         writer.add_raw_integrals(meta.input_targets) &&
         writer.add_raw_integrals(meta.input_uptargets);
}

template <typename Reader>
bool read_prologue(Reader &reader, SectorCheckpoint &meta) {
  if (reader.u32() != kSectorPayloadVersion)
    return false;
  meta.id = reader.u32();
  meta.index = reader.u32();
  meta.input_targets = reader.raw_integrals();
  meta.input_uptargets = reader.raw_integrals();
  return reader.ok();
}

template <typename Writer>
bool write_tail(Writer &writer, const SectorCheckpoint &meta) {
  if (!writer.add_u32(kSectorTailTag) ||
      !writer.add_raw_integrals(meta.masters) ||
      !writer.add_raw_integrals(meta.emitted_uptargets) ||
      !writer.add_raw_integrals(meta.cut_requests))
    return false;
  return writer.add_u32(meta.narrowing_success) &&
         writer.add_u32(meta.narrowing_fail) &&
         writer.add_u32(meta.truncated) && writer.add_u32(meta.lifted) &&
         writer.add_u32(meta.box_raised) && writer.add_u32(meta.box_depth) &&
         writer.add_u32(meta.box_rank) && writer.add_u32(meta.scope_skipped) &&
         writer.add_u32(meta.raw_equations) &&
         writer.add_u64(meta.probe_nanos) && writer.add_u64(meta.full_nanos) &&
         writer.add_u32(meta.timing.id) &&
         writer.add_bool(meta.timing.expanded) &&
         writer.add_u64(meta.timing.probeNanos) &&
         writer.add_u64(meta.timing.fullNanos);
}

template <typename Reader>
bool read_tail(Reader &reader, SectorCheckpoint &meta) {
  meta.masters = reader.raw_integrals();
  meta.emitted_uptargets = reader.raw_integrals();
  meta.cut_requests = reader.raw_integrals();
  meta.narrowing_success = reader.u32();
  meta.narrowing_fail = reader.u32();
  meta.truncated = reader.u32();
  meta.lifted = reader.u32();
  meta.box_raised = reader.u32();
  meta.box_depth = reader.u32();
  meta.box_rank = reader.u32();
  meta.scope_skipped = reader.u32();
  meta.raw_equations = reader.u32();
  meta.probe_nanos = reader.u64();
  meta.full_nanos = reader.u64();
  meta.timing.id = reader.u32();
  meta.timing.expanded = reader.boolean();
  meta.timing.probeNanos = reader.u64();
  meta.timing.fullNanos = reader.u64();
  return reader.ok();
}

template <typename Writer>
bool write_equation(Writer &writer, const GlobalEquation &equation) {
  if (!writer.add_u32(static_cast<std::uint32_t>(equation.size())))
    return false;
  for (const auto &[integral, coefficient] : equation)
    if (!writer.add_raw_integral(integral) ||
        !writer.add_string(coefficient))
      return false;
  return true;
}

template <typename Reader>
bool read_equation(Reader &reader, GlobalEquation &equation) {
  const std::uint32_t terms = reader.u32();
  if (!reader.ok() || terms > reader.remaining())
    return false;
  equation.clear();
  equation.reserve(terms);
  for (std::uint32_t i = 0; i < terms; ++i) {
    RawIntegral integral = reader.raw_integral();
    std::string coefficient = reader.string();
    if (!reader.ok())
      return false;
    equation.emplace_back(std::move(integral), std::move(coefficient));
  }
  return reader.ok();
}

bool read_header(std::ifstream &in, std::uint64_t &payload_bytes,
                 std::uint32_t &crc, std::string *error) {
  char header[kHeaderBytes] = {};
  in.read(header, static_cast<std::streamsize>(kHeaderBytes));
  if (!in || std::memcmp(header, kCheckpointMagic, sizeof kCheckpointMagic) != 0) {
    if (error)
      *error = "not a checkpoint file";
    return false;
  }
  const std::uint32_t version = get_u32(header + 4);
  const std::uint32_t tag = get_u32(header + 8);
  payload_bytes = get_u64(header + 12);
  crc = get_u32(header + 20);
  if (version != secred::kResumeCheckpointVersion) {
    if (error)
      *error = "checkpoint format version mismatch";
    return false;
  }
  if (tag != secred::kCheckpointTagSector) {
    if (error)
      *error = "not a sector checkpoint";
    return false;
  }
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  if (size < 0 ||
      static_cast<std::uint64_t>(size) != kHeaderBytes + payload_bytes) {
    if (error)
      *error = "checkpoint length mismatch";
    return false;
  }
  in.seekg(static_cast<std::streamoff>(kHeaderBytes));
  return true;
}

} // namespace

std::filesystem::path sector_checkpoint_path(
    const std::filesystem::path &pass_dir, unsigned id) {
  return pass_dir / ("sector-" + std::to_string(id) + ".ckpt");
}

std::string serialize_sector_checkpoint(const SectorCheckpoint &checkpoint) {
  secred::CheckpointWriter bytes;
  BytesWriter writer(bytes);
  write_prologue(writer, checkpoint);
  for (const SectorCheckpoint::Batch &batch : checkpoint.batches) {
    writer.add_u32(kSectorBatchTag);
    writer.add_u64(batch.seq);
    writer.add_u32(static_cast<std::uint32_t>(batch.equations.size()));
    for (const GlobalEquation &equation : batch.equations)
      write_equation(writer, equation);
  }
  write_tail(writer, checkpoint);
  return bytes.data();
}

bool parse_sector_checkpoint(std::string_view bytes,
                             SectorCheckpoint &checkpoint, std::string *error) {
  secred::CheckpointReader reader(bytes);
  SectorCheckpoint parsed;
  if (!read_prologue(reader, parsed)) {
    if (error)
      *error = "sector checkpoint has a bad prologue";
    return false;
  }
  bool saw_tail = false;
  while (reader.ok() && reader.remaining() != 0) {
    const std::uint32_t tag = reader.u32();
    if (tag == kSectorBatchTag) {
      SectorCheckpoint::Batch batch;
      batch.seq = reader.u64();
      const std::uint32_t equations = reader.u32();
      if (!reader.ok() || equations > reader.remaining()) {
        if (error)
          *error = "sector checkpoint has a malformed equation batch";
        return false;
      }
      batch.equations.reserve(equations);
      for (std::uint32_t i = 0; i < equations; ++i) {
        GlobalEquation equation;
        if (!read_equation(reader, equation)) {
          if (error)
            *error = "sector checkpoint has a malformed equation";
          return false;
        }
        batch.equations.push_back(std::move(equation));
      }
      parsed.batches.push_back(std::move(batch));
    } else if (tag == kSectorTailTag) {
      if (!read_tail(reader, parsed)) {
        if (error)
          *error = "sector checkpoint has a malformed tail";
        return false;
      }
      saw_tail = true;
      break;
    } else {
      if (error)
        *error = "sector checkpoint has an unknown record";
      return false;
    }
  }
  if (!reader.ok() || reader.remaining() != 0 || !saw_tail) {
    if (error)
      *error = "sector checkpoint has trailing or truncated data";
    return false;
  }
  checkpoint = std::move(parsed);
  if (error)
    error->clear();
  return true;
}

bool read_sector_checkpoint_file(const std::filesystem::path &path,
                                 SectorCheckpointFile &file, std::string *error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    if (error)
      *error = "cannot read " + path.string();
    return false;
  }
  std::uint64_t payload_bytes = 0;
  std::uint32_t expected_crc = 0;
  if (!read_header(in, payload_bytes, expected_crc, error))
    return false;

  FileCursor cursor(in, payload_bytes);
  SectorCheckpoint meta;
  if (!read_prologue(cursor, meta)) {
    if (error)
      *error = "sector checkpoint has a bad prologue";
    return false;
  }
  std::vector<SectorCheckpointBatchRef> batches;
  bool saw_tail = false;
  while (cursor.ok() && cursor.remaining() != 0) {
    const std::uint64_t offset = cursor.offset();
    const std::uint32_t tag = cursor.u32();
    if (tag == kSectorBatchTag) {
      SectorCheckpointBatchRef batch;
      batch.offset = offset;
      batch.seq = cursor.u64();
      batch.equations = cursor.u32();
      if (!cursor.ok() || batch.equations > cursor.remaining()) {
        if (error)
          *error = "sector checkpoint has a malformed equation batch";
        return false;
      }
      for (std::uint32_t i = 0; i < batch.equations; ++i) {
        const std::uint32_t terms = cursor.u32();
        if (!cursor.ok() || terms > cursor.remaining()) {
          if (error)
            *error = "sector checkpoint has a malformed equation";
          return false;
        }
        for (std::uint32_t t = 0; t < terms; ++t)
          if (!cursor.skip_raw_integral() || !cursor.skip_string()) {
            if (error)
              *error = "sector checkpoint has a malformed equation";
            return false;
          }
      }
      batches.push_back(batch);
    } else if (tag == kSectorTailTag) {
      if (!read_tail(cursor, meta)) {
        if (error)
          *error = "sector checkpoint has a malformed tail";
        return false;
      }
      saw_tail = true;
      break;
    } else {
      if (error)
        *error = "sector checkpoint has an unknown record";
      return false;
    }
  }
  if (!cursor.ok() || cursor.remaining() != 0 || !saw_tail) {
    if (error)
      *error = "sector checkpoint has trailing or truncated data";
    return false;
  }
  if (cursor.crc() != expected_crc) {
    if (error)
      *error = "sector checkpoint checksum mismatch";
    return false;
  }
  file.meta = std::move(meta);
  file.batches = std::move(batches);
  if (error)
    error->clear();
  return true;
}

bool read_sector_checkpoint_batch(const std::filesystem::path &path,
                                  const SectorCheckpointBatchRef &batch,
                                  std::vector<GlobalEquation> &equations,
                                  std::string *error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    if (error)
      *error = "cannot read " + path.string();
    return false;
  }
  std::uint64_t payload_bytes = 0;
  std::uint32_t expected_crc = 0;
  if (!read_header(in, payload_bytes, expected_crc, error))
    return false;
  (void)expected_crc;
  if (batch.offset >= payload_bytes) {
    if (error)
      *error = "sector checkpoint batch offset is outside the payload";
    return false;
  }
  in.seekg(static_cast<std::streamoff>(kHeaderBytes + batch.offset));
  FileCursor cursor(in, payload_bytes - batch.offset);
  if (cursor.u32() != kSectorBatchTag || cursor.u64() != batch.seq ||
      cursor.u32() != batch.equations) {
    if (error)
      *error = "sector checkpoint batch does not match its descriptor";
    return false;
  }
  equations.clear();
  equations.reserve(batch.equations);
  for (std::uint32_t i = 0; i < batch.equations; ++i) {
    GlobalEquation equation;
    if (!read_equation(cursor, equation)) {
      if (error)
        *error = "sector checkpoint has a malformed equation";
      return false;
    }
    equations.push_back(std::move(equation));
  }
  if (error)
    error->clear();
  return true;
}

SectorCheckpointStream::SectorCheckpointStream(unsigned id, unsigned index,
                                               std::filesystem::path path)
    : _path(std::move(path)) {
  _meta.id = id;
  _meta.index = index;
  _crc = static_cast<std::uint32_t>(crc32(0, Z_NULL, 0));
}

SectorCheckpointStream::~SectorCheckpointStream() { discard(); }

void SectorCheckpointStream::set_inputs(const std::set<RawIntegral> &targets,
                                        const std::set<RawIntegral> &uptargets) {
  _meta.input_targets.assign(targets.begin(), targets.end());
  _meta.input_uptargets.assign(uptargets.begin(), uptargets.end());
}

bool SectorCheckpointStream::open(std::string *error) {
  if (_opened)
    return true;
  if (_failed)
    return false;
  std::error_code ec;
  const std::filesystem::path parent = _path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, ec);
    if (ec && !std::filesystem::is_directory(parent)) {
      if (error)
        *error = "cannot create " + parent.string() + ": " + ec.message();
      _failed = true;
      return false;
    }
  }
  _temporary = _path.string() + ".tmp";
  _out.open(_temporary, std::ios::binary | std::ios::trunc);
  if (!_out) {
    if (error)
      *error = "cannot write " + _temporary.string();
    _failed = true;
    return false;
  }
  // Reserve the container header; finish() seeks back to it.
  const char zeros[kHeaderBytes] = {};
  _out.write(zeros, static_cast<std::streamsize>(kHeaderBytes));
  if (!_out) {
    if (error)
      *error = "cannot write " + _temporary.string();
    discard();
    return false;
  }
  _opened = true;
  if (!write_prologue(*this, _meta)) {
    if (error)
      *error = "cannot write " + _temporary.string();
    discard();
    return false;
  }
  return true;
}

bool SectorCheckpointStream::put(const void *data, std::size_t size) {
  if (_failed)
    return false;
  _buffer.append(static_cast<const char *>(data), size);
  if (_buffer.size() >= (1u << 16)) {
    if (!flush())
      return false;
  }
  return true;
}

bool SectorCheckpointStream::flush() {
  if (_failed)
    return false;
  if (_buffer.empty())
    return true;
  _out.write(_buffer.data(), static_cast<std::streamsize>(_buffer.size()));
  if (!_out) {
    _failed = true;
    return false;
  }
  _crc = static_cast<std::uint32_t>(
      crc32(_crc, reinterpret_cast<const Bytef *>(_buffer.data()),
            static_cast<uInt>(_buffer.size())));
  _payload_bytes += _buffer.size();
  _buffer.clear();
  return true;
}

bool SectorCheckpointStream::add_u8(std::uint8_t value) {
  return put(&value, 1);
}

bool SectorCheckpointStream::add_u32(std::uint32_t value) {
  std::string bytes;
  put_u32_le(bytes, value);
  return put(bytes.data(), bytes.size());
}

bool SectorCheckpointStream::add_u64(std::uint64_t value) {
  std::string bytes;
  put_u64_le(bytes, value);
  return put(bytes.data(), bytes.size());
}

bool SectorCheckpointStream::add_string(std::string_view text) {
  return add_u32(static_cast<std::uint32_t>(text.size())) &&
         put(text.data(), text.size());
}

bool SectorCheckpointStream::add_raw_integral(const RawIntegral &integral) {
  if (!add_u8(static_cast<std::uint8_t>(integral.size())))
    return false;
  for (unsigned i = 0; i < integral.size(); ++i) {
    const auto value =
        static_cast<std::uint8_t>(static_cast<std::int8_t>(integral[i]));
    if (!put(&value, 1))
      return false;
  }
  return true;
}

bool SectorCheckpointStream::add_raw_integrals(
    const std::vector<RawIntegral> &integrals) {
  if (!add_u32(static_cast<std::uint32_t>(integrals.size())))
    return false;
  for (const RawIntegral &integral : integrals)
    if (!add_raw_integral(integral))
      return false;
  return true;
}

bool SectorCheckpointStream::put_equation(const GlobalEquation &equation) {
  return write_equation(*this, equation);
}

bool SectorCheckpointStream::add_batch(
    std::uint64_t seq, const std::vector<GlobalEquation> &equations,
    std::string *error) {
  if (!_opened || _finished || _failed) {
    if (error)
      *error = "sector checkpoint stream is not open";
    return false;
  }
  add_u32(kSectorBatchTag);
  add_u64(seq);
  add_u32(static_cast<std::uint32_t>(equations.size()));
  for (const GlobalEquation &equation : equations) {
    if (_failed)
      break;
    put_equation(equation);
  }
  if (_failed) {
    if (error)
      *error = "cannot write " + _temporary.string();
    return false;
  }
  return true;
}

bool SectorCheckpointStream::finish(std::string *error) {
  if (_finished)
    return true;
  if (!_opened || _failed) {
    if (error)
      *error = "sector checkpoint stream is not open";
    discard();
    return false;
  }
  if (!write_tail(*this, _meta) || !flush()) {
    if (error)
      *error = "cannot write " + _temporary.string();
    discard();
    return false;
  }

  std::string header(kCheckpointMagic, sizeof kCheckpointMagic);
  put_u32_le(header, secred::kResumeCheckpointVersion);
  put_u32_le(header, secred::kCheckpointTagSector);
  put_u64_le(header, _payload_bytes);
  put_u32_le(header, _crc);
  _out.seekp(0);
  _out.write(header.data(), static_cast<std::streamsize>(header.size()));
  _out.flush();
  const bool wrote = static_cast<bool>(_out);
  _out.close();
  if (!wrote) {
    if (error)
      *error = "cannot finish " + _temporary.string();
    discard();
    return false;
  }
  std::error_code ec;
  std::filesystem::rename(_temporary, _path, ec);
  if (ec) {
    if (error)
      *error = "cannot rename " + _temporary.string() + " to " + _path.string() +
               ": " + ec.message();
    discard();
    return false;
  }
  _finished = true;
  if (error)
    error->clear();
  return true;
}

void SectorCheckpointStream::discard() {
  if (_finished)
    return;
  if (_out.is_open())
    _out.close();
  if (!_temporary.empty()) {
    std::error_code ec;
    std::filesystem::remove(_temporary, ec);
    _temporary.clear();
  }
  _failed = true;
}
