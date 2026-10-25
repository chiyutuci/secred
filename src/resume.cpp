#include "resume.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>

#include <zlib.h>

namespace secred {
namespace {

constexpr char kCheckpointMagic[4] = {'S', 'R', 'C', 'K'};

void put_u32(std::string &out, std::uint32_t value) {
  for (unsigned i = 0; i < 4; ++i)
    out.push_back(static_cast<char>((value >> (8 * i)) & 0xff));
}

void put_u64(std::string &out, std::uint64_t value) {
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

std::uint32_t payload_crc(std::string_view payload) {
  uLong crc = crc32(0L, Z_NULL, 0);
  std::size_t offset = 0;
  while (offset < payload.size()) {
    const std::size_t chunk =
        std::min<std::size_t>(payload.size() - offset, 1u << 30);
    crc = crc32(crc, reinterpret_cast<const Bytef *>(payload.data() + offset),
                static_cast<uInt>(chunk));
    offset += chunk;
  }
  return static_cast<std::uint32_t>(crc);
}

std::string scalar_key(const YAML::Node &key) {
  if (key.IsScalar())
    return key.Tag() + std::string(1, '\0') + key.Scalar();
  ResumeHasher hasher;
  const std::string text = YAML::Dump(key);
  hasher.add_string(text);
  return "$" + hasher.hex();
}

void hash_yaml(const YAML::Node &node, ResumeHasher &hasher,
               const std::string &path) {
  switch (node.Type()) {
  case YAML::NodeType::Undefined:
    hasher.add_u8(4);
    return;
  case YAML::NodeType::Null:
    hasher.add_u8(0);
    return;
  case YAML::NodeType::Scalar:
    hasher.add_u8(1);
    hasher.add_string(node.Tag());
    hasher.add_string(node.Scalar());
    return;
  case YAML::NodeType::Sequence:
    hasher.add_u8(2);
    hasher.add_u64(node.size());
    for (const YAML::Node &item : node)
      hash_yaml(item, hasher, path);
    return;
  case YAML::NodeType::Map: {
    hasher.add_u8(3);
    struct Entry {
      std::string key;
      std::string plain;
      YAML::Node value;
    };
    std::vector<Entry> entries;
    entries.reserve(node.size());
    for (const auto &entry : node) {
      // Copy-construct, never assign: yaml-cpp's Node::operator= rewrites the
      // node's contents in place, so an assignment here would share (and later
      // scramble) the caller's document.
      entries.push_back(Entry{
          scalar_key(entry.first),
          entry.first.IsScalar() ? entry.first.Scalar() : scalar_key(entry.first),
          YAML::Node(entry.second)});
    }
    // Sort an index permutation, not the entries themselves, for the same
    // reason: moving a Node-bearing struct through std::sort would assign the
    // Nodes and corrupt the document being hashed.
    std::vector<std::size_t> order(entries.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&entries](std::size_t a, std::size_t b) {
                       return entries[a].key < entries[b].key;
                     });
    hasher.add_u64(entries.size());
    for (const std::size_t index : order) {
      const Entry &entry = entries[index];
      hasher.add_string(entry.key);
      const std::string child = path.empty() ? entry.plain
                                             : path + "." + entry.plain;
      if (child == "settings.cache_dir" || child == "settings.resume") {
        hasher.add_u8(5); // "skipped": stable even if the key disappears
        continue;
      }
      hash_yaml(entry.value, hasher, child);
    }
    return;
  }
  }
}

bool parse_gzip_name(std::string_view name, unsigned &item, unsigned &prime) {
  const std::size_t underscore = name.find('_');
  if (underscore == std::string_view::npos || underscore == 0)
    return false;
  // The shortest valid name is "0_1.gz" (6 characters).
  if (name.size() < 6 || name.substr(name.size() - 3) != ".gz")
    return false;
  const std::size_t dot = name.size() - 3;
  if (dot <= underscore + 1)
    return false;
  const auto parse = [](std::string_view text, unsigned &out) {
    unsigned value = 0;
    const auto result =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc() || result.ptr != text.data() + text.size())
      return false;
    out = value;
    return true;
  };
  return parse(name.substr(0, underscore), item) &&
         parse(name.substr(underscore + 1, dot - underscore - 1), prime);
}

bool list_regular_files(const std::filesystem::path &dir,
                        std::vector<std::string> &files) {
  std::error_code ec;
  if (!std::filesystem::is_directory(dir, ec) || ec)
    return false;
  std::filesystem::directory_iterator it(dir, ec);
  if (ec)
    return false;
  for (const std::filesystem::directory_entry &entry : it) {
    std::error_code type_ec;
    if (entry.is_regular_file(type_ec) && !type_ec)
      files.push_back(entry.path().filename().string());
  }
  return true;
}

} // namespace

std::uint64_t resume_build_stamp() {
  static const std::uint64_t stamp = [] {
    ResumeHasher hasher;
    hasher.add_string(__DATE__ " " __TIME__);
    hasher.add_u32(kResumeFormatVersion);
    hasher.add_u32(kResumeCheckpointVersion);
    return hasher.value();
  }();
  return stamp;
}

void ResumeHasher::add(const void *data, std::size_t size) {
  const auto *bytes = static_cast<const unsigned char *>(data);
  for (std::size_t i = 0; i < size; ++i) {
    _state ^= bytes[i];
    _state *= 1099511628211ULL;
  }
}

void ResumeHasher::add_u8(std::uint8_t value) { add(&value, 1); }

void ResumeHasher::add_u32(std::uint32_t value) {
  std::array<unsigned char, 4> bytes{};
  for (unsigned i = 0; i < 4; ++i)
    bytes[i] = static_cast<unsigned char>((value >> (8 * i)) & 0xff);
  add(bytes.data(), bytes.size());
}

void ResumeHasher::add_u64(std::uint64_t value) {
  std::array<unsigned char, 8> bytes{};
  for (unsigned i = 0; i < 8; ++i)
    bytes[i] = static_cast<unsigned char>((value >> (8 * i)) & 0xff);
  add(bytes.data(), bytes.size());
}

void ResumeHasher::add_int(std::int64_t value) {
  add_u64(static_cast<std::uint64_t>(value));
}

void ResumeHasher::add_string(std::string_view text) {
  add_u64(text.size());
  add(text.data(), text.size());
}

void ResumeHasher::add_raw_integral(const RawIntegral &integral) {
  add_u8(static_cast<std::uint8_t>(integral.size()));
  for (unsigned i = 0; i < integral.size(); ++i)
    add_u8(static_cast<std::uint8_t>(static_cast<std::int8_t>(integral[i])));
}

std::string ResumeHasher::hex() const {
  char text[17];
  std::snprintf(text, sizeof(text), "%016llx",
                static_cast<unsigned long long>(_state));
  return std::string(text, 16);
}

std::uint64_t resume_config_hash(const YAML::Node &config) {
  ResumeHasher hasher;
  hash_yaml(config, hasher, {});
  return hasher.value();
}

bool hash_file(const std::filesystem::path &path, std::uint64_t &value,
               std::string *error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    if (error)
      *error = "cannot open " + path.string();
    return false;
  }
  ResumeHasher hasher;
  std::vector<char> buffer(1u << 20);
  while (in) {
    in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize got = in.gcount();
    if (got > 0)
      hasher.add(buffer.data(), static_cast<std::size_t>(got));
  }
  if (in.bad()) {
    if (error)
      *error = "cannot read " + path.string();
    return false;
  }
  value = hasher.value();
  if (error)
    error->clear();
  return true;
}

std::string sector_pass_key(std::uint64_t input_hash, unsigned pass,
                            bool truncation_active, unsigned truncation_l,
                            unsigned level_offset, unsigned dots,
                            bool exempt_cut_requests) {
  ResumeHasher hasher;
  hasher.add_u64(input_hash);
  hasher.add_u32(pass);
  hasher.add_u8(truncation_active ? 1 : 0);
  hasher.add_u32(truncation_l);
  hasher.add_u32(level_offset);
  hasher.add_u32(dots);
  hasher.add_u8(exempt_cut_requests ? 1 : 0);
  return hasher.hex();
}

void CheckpointWriter::add_u8(std::uint8_t value) {
  _bytes.push_back(static_cast<char>(value));
}

void CheckpointWriter::add_u32(std::uint32_t value) { put_u32(_bytes, value); }

void CheckpointWriter::add_u64(std::uint64_t value) { put_u64(_bytes, value); }

void CheckpointWriter::add_int(std::int64_t value) {
  add_u64(static_cast<std::uint64_t>(value));
}

void CheckpointWriter::add_string(std::string_view text) {
  add_u32(static_cast<std::uint32_t>(text.size()));
  _bytes.append(text.data(), text.size());
}

void CheckpointWriter::add_raw_integral(const RawIntegral &integral) {
  add_u8(static_cast<std::uint8_t>(integral.size()));
  for (unsigned i = 0; i < integral.size(); ++i)
    _bytes.push_back(
        static_cast<char>(static_cast<std::uint8_t>(integral[i])));
}

void CheckpointWriter::add_raw_integrals(
    const std::vector<RawIntegral> &integrals) {
  add_u32(static_cast<std::uint32_t>(integrals.size()));
  for (const RawIntegral &integral : integrals)
    add_raw_integral(integral);
}

bool CheckpointReader::need(std::size_t n) {
  if (!_ok)
    return false;
  if (n > _bytes.size() - _pos) {
    _ok = false;
    return false;
  }
  return true;
}

std::uint8_t CheckpointReader::u8() {
  if (!need(1))
    return 0;
  return static_cast<std::uint8_t>(_bytes[_pos++]);
}

std::uint32_t CheckpointReader::u32() {
  if (!need(4))
    return 0;
  const std::uint32_t value = get_u32(_bytes.data() + _pos);
  _pos += 4;
  return value;
}

std::uint64_t CheckpointReader::u64() {
  if (!need(8))
    return 0;
  const std::uint64_t value = get_u64(_bytes.data() + _pos);
  _pos += 8;
  return value;
}

std::int64_t CheckpointReader::int_value() {
  return static_cast<std::int64_t>(u64());
}

std::string CheckpointReader::string() {
  const std::uint32_t length = u32();
  if (!need(length))
    return {};
  std::string value(_bytes.substr(_pos, length));
  _pos += length;
  return value;
}

RawIntegral CheckpointReader::raw_integral() {
  const unsigned nprops = u8();
  if (!_ok || nprops > RawIntegral::MAX_PROPS) {
    _ok = false;
    return RawIntegral();
  }
  if (!need(nprops)) {
    _ok = false;
    return RawIntegral();
  }
  std::vector<int> powers(nprops);
  for (unsigned i = 0; i < nprops; ++i)
    powers[i] = static_cast<std::int8_t>(_bytes[_pos++]);
  return RawIntegral(std::move(powers));
}

std::vector<RawIntegral> CheckpointReader::raw_integrals() {
  const std::uint32_t count = u32();
  // Each integral occupies at least its size byte; a count past the end is a
  // corrupt file and must not be used as a reserve() size.
  if (!_ok || count > remaining()) {
    _ok = false;
    return {};
  }
  std::vector<RawIntegral> integrals;
  integrals.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    integrals.push_back(raw_integral());
    if (!_ok)
      return {};
  }
  return integrals;
}

bool write_checkpoint(const std::filesystem::path &path, std::uint32_t tag,
                      std::uint32_t version, std::string_view payload,
                      std::string *error) {
  std::error_code ec;
  const std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, ec);
    if (ec && !std::filesystem::is_directory(parent)) {
      if (error)
        *error = "cannot create " + parent.string() + ": " + ec.message();
      return false;
    }
  }

  std::string bytes(kCheckpointMagic, sizeof kCheckpointMagic);
  put_u32(bytes, version);
  put_u32(bytes, tag);
  put_u64(bytes, static_cast<std::uint64_t>(payload.size()));
  put_u32(bytes, payload_crc(payload));
  bytes.append(payload.data(), payload.size());

  const std::filesystem::path temporary = path.string() + ".tmp";
  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) {
      if (error)
        *error = "cannot write " + temporary.string();
      return false;
    }
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.flush();
    if (!out) {
      if (error)
        *error = "cannot write " + temporary.string();
      out.close();
      std::filesystem::remove(temporary, ec);
      return false;
    }
  }

  std::filesystem::rename(temporary, path, ec);
  if (ec) {
    if (error)
      *error = "cannot rename " + temporary.string() + " to " + path.string() +
               ": " + ec.message();
    std::filesystem::remove(temporary, ec);
    return false;
  }
  if (error)
    error->clear();
  return true;
}

bool read_checkpoint(const std::filesystem::path &path, std::uint32_t tag,
                     std::uint32_t version, std::string &payload,
                     std::string *error) {
  payload.clear();
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    if (error)
      *error = "cannot read " + path.string();
    return false;
  }
  std::string bytes((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
  if (bytes.size() < kCheckpointHeaderBytes) {
    if (error)
      *error = path.string() + ": checkpoint header is short";
    return false;
  }
  if (std::memcmp(bytes.data(), kCheckpointMagic, sizeof kCheckpointMagic) !=
      0) {
    if (error)
      *error = path.string() + ": checkpoint magic is wrong";
    return false;
  }
  const std::uint32_t stored_version = get_u32(bytes.data() + 4);
  const std::uint32_t stored_tag = get_u32(bytes.data() + 8);
  const std::uint64_t length = get_u64(bytes.data() + 12);
  const std::uint32_t checksum = get_u32(bytes.data() + 20);
  if (stored_version != version) {
    if (error)
      *error = path.string() + ": checkpoint version " +
               std::to_string(stored_version) + " is not " +
               std::to_string(version);
    return false;
  }
  if (stored_tag != tag) {
    if (error)
      *error = path.string() + ": checkpoint tag is wrong";
    return false;
  }
  if (length != bytes.size() - kCheckpointHeaderBytes) {
    if (error)
      *error = path.string() + ": checkpoint length does not match its file";
    return false;
  }
  std::string_view body(bytes.data() + kCheckpointHeaderBytes,
                        static_cast<std::size_t>(length));
  if (payload_crc(body) != checksum) {
    if (error)
      *error = path.string() + ": checkpoint checksum does not match";
    return false;
  }
  payload.assign(body);
  if (error)
    error->clear();
  return true;
}

bool write_manifest(const std::filesystem::path &path,
                    const ResumeManifest &manifest, std::string *error) {
  CheckpointWriter writer;
  writer.add_u32(manifest.format);
  writer.add_u64(manifest.build);
  writer.add_u64(manifest.input_hash);
  return write_checkpoint(path, kCheckpointTagManifest,
                          kResumeCheckpointVersion, writer.data(), error);
}

bool read_manifest(const std::filesystem::path &path, ResumeManifest &manifest,
                   std::string *error) {
  std::string payload;
  if (!read_checkpoint(path, kCheckpointTagManifest, kResumeCheckpointVersion,
                       payload, error))
    return false;
  CheckpointReader reader(payload);
  manifest.format = reader.u32();
  manifest.build = reader.u64();
  manifest.input_hash = reader.u64();
  if (!reader.ok() || reader.remaining() != 0) {
    if (error)
      *error = path.string() + ": manifest is malformed";
    return false;
  }
  if (error)
    error->clear();
  return true;
}

bool gzip_file_ok(const std::filesystem::path &path) {
  // gzread() reads a file without a gzip header transparently as raw bytes, so
  // a plain file would pass every check below.  FireFly writes real gzip
  // streams, so require the magic before handing anything to zlib.
  {
    std::ifstream head(path, std::ios::binary);
    char magic[2] = {};
    head.read(magic, 2);
    if (!head || static_cast<unsigned char>(magic[0]) != 0x1f ||
        static_cast<unsigned char>(magic[1]) != 0x8b)
      return false;
  }
  gzFile file = gzopen(path.string().c_str(), "rb");
  if (!file)
    return false;
  bool ok = true;
  char buffer[1 << 16];
  for (;;) {
    const int got = gzread(file, buffer, sizeof buffer);
    if (got < 0) {
      ok = false;
      break;
    }
    if (got == 0)
      break;
  }
  int error_number = 0;
  (void)gzerror(file, &error_number);
  if (error_number != Z_OK)
    ok = false;
  if (gzclose(file) != Z_OK)
    ok = false;
  return ok;
}

namespace {

// FireFly's probe writer appends one gzip member per flush; a kill can cut the
// last one.  gzread() still yields everything the complete members hold, so the
// complete probe lines can be recovered.
std::string read_gzip_lossy(const std::filesystem::path &path) {
  gzFile file = gzopen(path.string().c_str(), "rb");
  if (file == nullptr)
    return {};
  std::string text;
  char buffer[1 << 16];
  int got = 0;
  while ((got = gzread(file, buffer, sizeof buffer)) > 0)
    text.append(buffer, static_cast<std::size_t>(got));
  gzclose(file);
  return text;
}

bool write_gzip_file(const std::filesystem::path &path, std::string_view text) {
  gzFile file = gzopen(path.string().c_str(), "wb");
  if (file == nullptr)
    return false;
  bool ok = true;
  std::size_t offset = 0;
  while (offset < text.size()) {
    const std::size_t chunk =
        std::min<std::size_t>(text.size() - offset, 1u << 30);
    if (gzwrite(file, text.data() + offset, static_cast<unsigned>(chunk)) !=
        static_cast<int>(chunk)) {
      ok = false;
      break;
    }
    offset += chunk;
  }
  if (gzclose(file) != Z_OK)
    ok = false;
  return ok;
}

// Replace `path` with a gzip stream holding `text`.  The write goes through
// ff_save/tmp and a rename, so a kill during the repair leaves the previous
// file in place; prepare_firefly_save() removes the temporary next time.
bool replace_with_gzip(const std::filesystem::path &ff_save,
                       const std::filesystem::path &path,
                       std::string_view text) {
  const std::filesystem::path temporary =
      ff_save / "tmp" / ("repair_" + path.filename().string());
  std::error_code ec;
  std::filesystem::create_directories(ff_save / "tmp", ec);
  if (ec || !write_gzip_file(temporary, text))
    return false;
  std::filesystem::rename(temporary, path, ec);
  if (ec) {
    std::filesystem::remove(temporary, ec);
    return false;
  }
  return true;
}

// The first line of a gzip state file, without its newline.
bool gzip_first_line(const std::filesystem::path &path, std::string &line) {
  line.clear();
  gzFile file = gzopen(path.string().c_str(), "rb");
  if (file == nullptr)
    return false;
  char buffer[256];
  const char *const got = gzgets(file, buffer, sizeof buffer);
  const bool ok = got != nullptr;
  if (ok)
    line.assign(got);
  gzclose(file);
  while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
    line.pop_back();
  return ok;
}

bool decimal_token(std::string_view token) {
  if (token.empty())
    return false;
  return std::all_of(token.begin(), token.end(),
                     [](char c) { return c >= '0' && c <= '9'; });
}

// FireFly's vector parsers only produce a token when a space follows it
// (`parse_vector_FFInt` searches for ' '), so count exactly those tokens.
bool count_space_tokens(std::string_view line, std::size_t &count) {
  count = 0;
  std::size_t start = 0;
  for (std::size_t i = 0; i < line.size(); ++i) {
    if (line[i] != ' ')
      continue;
    if (!decimal_token(line.substr(start, i - start)))
      return false;
    ++count;
    start = i + 1;
  }
  return true;
}

// A probe line is complete when it ends with the space before its newline and
// carries the nparams - 1 zi orders plus t and num.
bool probe_line_ok(std::string_view line, unsigned nparams) {
  if (line.empty() || line.back() != ' ')
    return false;
  std::size_t tokens = 0;
  if (!count_space_tokens(line, tokens))
    return false;
  return tokens == static_cast<std::size_t>(nparams) + 1;
}

enum class StateKind { tag, combined, zero };

unsigned state_rank(StateKind kind, unsigned prime) {
  // A tag state is the checkpoint before the first prime.  Every other state
  // at file prime q means prime q is complete, and FireFly continues at q + 1.
  return kind == StateKind::tag ? 0u : prime + 1;
}

bool parse_state_kind(const std::filesystem::path &path, StateKind &kind,
                      std::string &reason) {
  std::string line;
  if (!gzip_first_line(path, line)) {
    reason = "unreadable state " + path.filename().string();
    return false;
  }
  if (line == "tag_name") {
    kind = StateKind::tag;
  } else if (line == "combined_prime") {
    kind = StateKind::combined;
  } else if (line == "ZERO") {
    kind = StateKind::zero;
  } else {
    reason = "unrecognized state " + path.filename().string();
    return false;
  }
  return true;
}

} // namespace

FireFlySaveStatus prepare_firefly_save(const std::filesystem::path &ff_save,
                                       unsigned items, unsigned nparams) {
  FireFlySaveStatus status;
  if (items == 0) {
    status.reason = "the black box has no functions";
    return status;
  }
  const auto fail = [&status](std::string reason) {
    status.reason = std::move(reason);
    return status;
  };
  const auto remove_file = [&status](const std::filesystem::path &path) {
    std::error_code ec;
    const bool removed = std::filesystem::remove(path, ec);
    if (ec || !removed)
      return false;
    ++status.repairs;
    return true;
  };

  struct Candidate {
    unsigned prime = 0;
    StateKind kind = StateKind::combined;
    bool temporary = false;
    std::string name;
  };

  std::vector<std::string> state_files;
  if (!list_regular_files(ff_save / "states", state_files))
    return fail("no ff_save/states directory");

  std::vector<std::vector<Candidate>> states_for(items);
  for (const std::string &name : state_files) {
    unsigned item = 0;
    unsigned prime = 0;
    if (!parse_gzip_name(name, item, prime)) {
      // FireFly counts every regular file as a state; a stray one would make
      // its function count wrong.
      if (!remove_file(ff_save / "states" / name))
        return fail("cannot remove stray ff_save/states/" + name);
      continue;
    }
    if (item >= items)
      return fail("state file " + name +
                  " names a function outside the black box");
    if (prime == std::numeric_limits<unsigned>::max())
      return fail("state file " + name + " names an impossible prime");
    const std::filesystem::path path = ff_save / "states" / name;
    if (!gzip_file_ok(path))
      return fail("truncated ff_save/states/" + name);
    StateKind kind = StateKind::combined;
    std::string reason;
    if (!parse_state_kind(path, kind, reason))
      return fail(reason);
    // FireFly's tag parser always resumes in the first prime field, whatever
    // the file is called; a later name would desynchronize it from the probes.
    if (kind == StateKind::tag && prime != 0)
      return fail("ff_save/states/" + name +
                  " is a tag state past the first prime");
    states_for[item].push_back(Candidate{prime, kind, false, name});
  }

  // State temporaries: save_state() writes the new state to tmp/ and only then
  // renames the old state forward and the temporary into place.  A kill in
  // between leaves a complete temporary that is newer than any state file at
  // its prime, or a partial one that never replaced the previous state.
  std::vector<std::string> tmp_files;
  list_regular_files(ff_save / "tmp", tmp_files);
  std::vector<std::vector<Candidate>> tmps_for(items);
  for (const std::string &name : tmp_files) {
    const std::filesystem::path path = ff_save / "tmp" / name;
    if (name.rfind("repair_", 0) == 0 || name.rfind("probe_", 0) == 0) {
      // An interrupted repair, or a probe stream the zero path replaced.
      // FireFly ignores tmp/ on resume; keeping these only confuses the next
      // attempt.
      if (!remove_file(path))
        return fail("cannot remove ff_save/tmp/" + name);
      continue;
    }
    unsigned item = 0;
    unsigned prime = 0;
    if (!parse_gzip_name(name, item, prime))
      continue; // Some other temporary; FireFly never reads it back.
    // A temporary that a kill cut short was never renamed into place, so the
    // previous state is still the valid one.  Anything outside the black box
    // cannot belong to this reconstruction.
    StateKind kind = StateKind::combined;
    std::string reason;
    if (item >= items || prime == std::numeric_limits<unsigned>::max() ||
        !gzip_file_ok(path) || !parse_state_kind(path, kind, reason) ||
        (kind == StateKind::tag && prime != 0)) {
      if (!remove_file(path))
        return fail("cannot remove ff_save/tmp/" + name);
      continue;
    }
    tmps_for[item].push_back(Candidate{prime, kind, true, name});
  }

  std::vector<Candidate> chosen(items);
  for (unsigned item = 0; item < items; ++item) {
    bool have = false;
    Candidate best{};
    for (const Candidate &candidate : states_for[item]) {
      if (!have) {
        best = candidate;
        have = true;
        continue;
      }
      const unsigned rank = state_rank(candidate.kind, candidate.prime);
      const unsigned best_rank = state_rank(best.kind, best.prime);
      if (rank > best_rank ||
          (rank == best_rank && candidate.temporary && !best.temporary))
        best = candidate;
    }
    for (const Candidate &candidate : tmps_for[item]) {
      if (!have) {
        best = candidate;
        have = true;
        continue;
      }
      const unsigned rank = state_rank(candidate.kind, candidate.prime);
      const unsigned best_rank = state_rank(best.kind, best.prime);
      if (rank > best_rank ||
          (rank == best_rank && candidate.temporary && !best.temporary))
        best = candidate;
    }
    if (!have)
      return fail("no state file for function " + std::to_string(item));
    chosen[item] = best;
  }

  const std::filesystem::path states_dir = ff_save / "states";
  const std::filesystem::path tmp_dir = ff_save / "tmp";
  for (unsigned item = 0; item < items; ++item) {
    const Candidate &pick = chosen[item];
    const std::string target_name =
        std::to_string(item) + "_" + std::to_string(pick.prime) + ".gz";
    if (pick.temporary) {
      std::error_code ec;
      std::filesystem::rename(tmp_dir / pick.name, states_dir / target_name, ec);
      if (ec)
        return fail("cannot promote ff_save/tmp/" + pick.name + ": " +
                    ec.message());
      ++status.repairs;
    }
    // FireFly's resume lists the directory instead of taking a set of files:
    // exactly one state per function, and only the chosen one.
    for (const Candidate &candidate : states_for[item]) {
      if (candidate.name == target_name)
        continue;
      if (!remove_file(states_dir / candidate.name))
        return fail("cannot remove ff_save/states/" + candidate.name);
    }
    for (const Candidate &candidate : tmps_for[item]) {
      if (pick.temporary && candidate.name == pick.name)
        continue;
      if (!remove_file(tmp_dir / candidate.name))
        return fail("cannot remove ff_save/tmp/" + candidate.name);
    }
  }

  std::error_code ec;
  std::filesystem::create_directories(ff_save / "probes", ec);
  if (ec)
    return fail("cannot create ff_save/probes: " + ec.message());
  std::vector<std::string> probe_files;
  (void)list_regular_files(ff_save / "probes", probe_files);
  std::vector<std::vector<std::pair<unsigned, std::string>>> probes_for(items);
  for (const std::string &name : probe_files) {
    unsigned item = 0;
    unsigned prime = 0;
    if (!parse_gzip_name(name, item, prime)) {
      // The probe count FireFly checks counts every regular file.
      if (!remove_file(ff_save / "probes" / name))
        return fail("cannot remove stray ff_save/probes/" + name);
      continue;
    }
    if (item >= items)
      return fail("probe file " + name +
                  " names a function outside the black box");
    probes_for[item].emplace_back(prime, name);
  }

  for (unsigned item = 0; item < items; ++item) {
    const Candidate &pick = chosen[item];
    // FireFly's start_from_saved_file() continues a combined or zero state at
    // the file's prime + 1, and a tag state in the prime it names.
    const unsigned expected =
        pick.kind == StateKind::tag ? 0u : pick.prime + 1;
    const std::string expected_name =
        std::to_string(item) + "_" + std::to_string(expected) + ".gz";
    std::error_code probe_ec;
    bool have_expected = false;
    for (const auto &[prime, name] : probes_for[item]) {
      if (prime == expected) {
        have_expected = true;
        continue;
      }
      // Probes of another prime are never replayed; the values belong to a
      // prime the state has already accounted for (or has not reached).
      if (!remove_file(ff_save / "probes" / name))
        return fail("cannot remove ff_save/probes/" + name);
    }
    const std::filesystem::path probe_path =
        ff_save / "probes" / expected_name;
    if (!have_expected) {
      if (!replace_with_gzip(ff_save, probe_path, std::string_view{}))
        return fail("cannot create ff_save/probes/" + expected_name);
      ++status.repairs;
    } else if (!gzip_file_ok(probe_path)) {
      // The last appended gzip member was cut.  Everything before it is intact,
      // and only complete lines can be replayed; FireFly recomputes the rest.
      const std::string text = read_gzip_lossy(probe_path);
      std::string kept;
      std::size_t start = 0;
      while (start < text.size()) {
        const std::size_t end = text.find('\n', start);
        if (end == std::string::npos)
          break;
        const std::string_view line(text.data() + start, end - start);
        if (probe_line_ok(line, nparams)) {
          kept.append(line.data(), line.size());
          kept.push_back('\n');
        }
        start = end + 1;
      }
      if (!replace_with_gzip(ff_save, probe_path, kept))
        return fail("cannot repair ff_save/probes/" + expected_name);
      ++status.repairs;
    }
  }

  unsigned prime = std::numeric_limits<unsigned>::max();
  for (unsigned item = 0; item < items; ++item)
    prime = std::min(prime, chosen[item].prime);

  if (prime == 0) {
    // FireFly reads both before continuing an unfinished set at the first
    // prime, and exits the process when either is missing or short.
    const unsigned anchor_tokens = nparams > 0 ? nparams - 1 : 0;
    std::ifstream anchor(ff_save / "anchor_points");
    std::string line;
    std::size_t tokens = 0;
    if (!anchor || !std::getline(anchor, line))
      return fail("missing ff_save/anchor_points");
    if (!count_space_tokens(line, tokens) || tokens != anchor_tokens)
      return fail("ff_save/anchor_points does not hold " +
                  std::to_string(anchor_tokens) + " values");
    std::ifstream shift(ff_save / "shift");
    if (!shift || !std::getline(shift, line))
      return fail("missing ff_save/shift");
    if (!count_space_tokens(line, tokens) || tokens != nparams)
      return fail("ff_save/shift does not hold " + std::to_string(nparams) +
                  " values");
  }

  // FireFly validates the saved states against the black box before it uses
  // them and exits the process when the file is missing, cut, or holds a
  // different number of results.
  const std::filesystem::path validation = ff_save / "validation.gz";
  if (!gzip_file_ok(validation))
    return fail("missing or truncated ff_save/validation.gz");
  const std::string validation_text = read_gzip_lossy(validation);
  std::vector<std::string_view> lines;
  for (std::size_t start = 0; start < validation_text.size();) {
    const std::size_t end = validation_text.find('\n', start);
    if (end == std::string::npos) {
      lines.emplace_back(std::string_view(validation_text).substr(start));
      break;
    }
    lines.emplace_back(std::string_view(validation_text).substr(start,
                                                                end - start));
    start = end + 1;
  }
  if (lines.empty())
    return fail("ff_save/validation.gz holds no values");
  std::size_t tokens = 0;
  if (!count_space_tokens(lines[0], tokens) || tokens != nparams)
    return fail("ff_save/validation.gz does not hold " +
                std::to_string(nparams) + " parameters");
  if (lines.size() != static_cast<std::size_t>(items) + 1)
    return fail("ff_save/validation.gz holds " +
                std::to_string(lines.size() - 1) + " results for " +
                std::to_string(items) + " functions");
  for (std::size_t i = 1; i < lines.size(); ++i) {
    // FireFly reads every result with std::stoul.
    std::size_t digits = 0;
    while (digits < lines[i].size() && lines[i][digits] >= '0' &&
           lines[i][digits] <= '9')
      ++digits;
    if (digits == 0)
      return fail("ff_save/validation.gz has a malformed result line");
  }
  if (std::filesystem::exists(ff_save / "var_order.gz", ec) &&
      !gzip_file_ok(ff_save / "var_order.gz"))
    return fail("truncated ff_save/var_order.gz");

  status.usable = true;
  status.prime = prime;
  status.reason.clear();
  return status;
}

namespace {

void add_unsigned_vector(CheckpointWriter &writer,
                         const std::vector<unsigned> &values) {
  writer.add_u32(static_cast<std::uint32_t>(values.size()));
  for (unsigned value : values)
    writer.add_u32(value);
}

void add_u32_vector(CheckpointWriter &writer,
                    const std::vector<std::uint32_t> &values) {
  writer.add_u32(static_cast<std::uint32_t>(values.size()));
  for (std::uint32_t value : values)
    writer.add_u32(value);
}

bool read_unsigned_vector(CheckpointReader &reader,
                          std::vector<unsigned> &values) {
  const std::uint32_t count = reader.u32();
  if (!reader.ok() || count > reader.remaining() / 4)
    return false;
  values.resize(count);
  for (std::uint32_t i = 0; i < count; ++i)
    values[i] = reader.u32();
  return reader.ok();
}

bool read_u32_vector(CheckpointReader &reader,
                     std::vector<std::uint32_t> &values) {
  const std::uint32_t count = reader.u32();
  if (!reader.ok() || count > reader.remaining() / 4)
    return false;
  values.resize(count);
  for (std::uint32_t i = 0; i < count; ++i)
    values[i] = reader.u32();
  return reader.ok();
}

} // namespace

bool write_firefly_plan_key(const std::filesystem::path &path,
                            const FireFlyPlanKey &key, std::string *error) {
  CheckpointWriter writer;
  writer.add_u32(key.nparams);
  add_unsigned_vector(writer, key.dep_targets);
  add_unsigned_vector(writer, key.masters);
  add_u32_vector(writer, key.item_master);
  add_u32_vector(writer, key.item_col);
  add_u32_vector(writer, key.item_begin);
  return write_checkpoint(path, kCheckpointTagFireFly, kResumeCheckpointVersion,
                          writer.data(), error);
}

bool read_firefly_plan_key(const std::filesystem::path &path,
                           FireFlyPlanKey &key, std::string *error) {
  std::string payload;
  if (!read_checkpoint(path, kCheckpointTagFireFly, kResumeCheckpointVersion,
                       payload, error))
    return false;
  CheckpointReader reader(payload);
  FireFlyPlanKey parsed;
  parsed.nparams = reader.u32();
  const bool ok = read_unsigned_vector(reader, parsed.dep_targets) &&
                  read_unsigned_vector(reader, parsed.masters) &&
                  read_u32_vector(reader, parsed.item_master) &&
                  read_u32_vector(reader, parsed.item_col) &&
                  read_u32_vector(reader, parsed.item_begin) &&
                  reader.remaining() == 0;
  if (!ok) {
    if (error != nullptr)
      *error = "malformed FireFly plan checkpoint: " + path.string();
    return false;
  }
  key = std::move(parsed);
  return true;
}

ScopedCurrentPath::ScopedCurrentPath(const std::filesystem::path &dir) {
  std::error_code ec;
  _old = std::filesystem::current_path(ec);
  if (ec)
    return;
  std::filesystem::current_path(dir, ec);
  _ok = !ec;
}

ScopedCurrentPath::~ScopedCurrentPath() {
  if (!_ok)
    return;
  std::error_code ec;
  std::filesystem::current_path(_old, ec);
}

namespace {

std::string hex_u64(std::uint64_t value) {
  char text[17];
  std::snprintf(text, sizeof(text), "%016llx",
                static_cast<unsigned long long>(value));
  return std::string(text, 16);
}

std::filesystem::path run_key_dir(const std::string &cache_dir,
                                  const char *kind, std::uint64_t spill_hash,
                                  unsigned max_primes) {
  return std::filesystem::path(cache_dir) / kResumeDirName / kind /
         (hex_u64(spill_hash) + "-" + std::to_string(max_primes));
}

} // namespace

std::filesystem::path sector_resume_dir(const std::string &cache_dir,
                                        const std::string &pass_key) {
  return std::filesystem::path(cache_dir) / kResumeDirName / "sectors" /
         pass_key;
}

std::filesystem::path prime_resume_dir(const ResumeControl &control,
                                       unsigned max_primes) {
  return run_key_dir(control.cache_dir, "primes", control.spill_hash,
                     max_primes);
}

std::filesystem::path firefly_resume_dir(const ResumeControl &control,
                                         unsigned max_primes) {
  return run_key_dir(control.cache_dir, "firefly", control.spill_hash,
                     max_primes);
}

} // namespace secred
