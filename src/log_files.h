
#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <streambuf>
#include <string>
#include <system_error>

namespace secred {

inline constexpr const char *kRunLogFile = "secred.log";
inline constexpr const char *kFireFlyLogFile = "firefly.log";

// Remove a stale firefly.log from the CWD.
inline void reset_firefly_log() {
  std::error_code ec;
  std::filesystem::remove(kFireFlyLogFile, ec);
}

// Move firefly.log into the cache directory.
inline bool collect_firefly_log(const std::string &cache_dir) {
  std::error_code ec;
  if (!std::filesystem::exists(kFireFlyLogFile, ec))
    return false;
  const std::filesystem::path dst =
      std::filesystem::path(cache_dir) / kFireFlyLogFile;
  std::filesystem::rename(kFireFlyLogFile, dst, ec);
  if (ec) {

    ec.clear();
    std::filesystem::copy_file(kFireFlyLogFile, dst,
                               std::filesystem::copy_options::overwrite_existing, ec);
    if (ec)
      return false;
    std::filesystem::remove(kFireFlyLogFile, ec);
  }
  return true;
}

// Tee stdout and stderr into the run log.
class RunLog {
public:
  explicit RunLog(const std::string &cache_dir)
      : _path((std::filesystem::path(cache_dir) / kRunLogFile).string()) {
    _file.open(_path, std::ios::out | std::ios::trunc);
    if (!_file.is_open()) {
      std::cerr << "cannot write the run log " << _path << "\n";
      return;
    }
    _cout.emplace(std::cout.rdbuf(), _file.rdbuf(), &_mutex);
    _cerr.emplace(std::cerr.rdbuf(), _file.rdbuf(), &_mutex);
    std::cout.rdbuf(&*_cout);
    std::cerr.rdbuf(&*_cerr);
  }

  ~RunLog() {
    if (!_cout)
      return;
    std::cout.flush();
    std::cerr.flush();
    std::cout.rdbuf(_cout->sink());
    std::cerr.rdbuf(_cerr->sink());
  }

  RunLog(const RunLog &) = delete;
  RunLog &operator=(const RunLog &) = delete;

  bool ok() const { return _file.is_open(); }
  const std::string &path() const { return _path; }

private:

  class Tee : public std::streambuf {
  public:
    Tee(std::streambuf *sink, std::streambuf *file, std::mutex *mutex)
        : _sink(sink), _file(file), _mutex(mutex) {}

    std::streambuf *sink() const { return _sink; }

  protected:
    int overflow(int c) override {
      if (traits_type::eq_int_type(c, traits_type::eof()))
        return sync();
      std::lock_guard<std::mutex> lock(*_mutex);
      _file->sputc(traits_type::to_char_type(c));
      return _sink->sputc(traits_type::to_char_type(c));
    }

    std::streamsize xsputn(const char *s, std::streamsize n) override {
      std::lock_guard<std::mutex> lock(*_mutex);
      _file->sputn(s, n);
      return _sink->sputn(s, n);
    }

    int sync() override {
      std::lock_guard<std::mutex> lock(*_mutex);
      _file->pubsync();
      return _sink->pubsync();
    }

  private:
    std::streambuf *_sink;
    std::streambuf *_file;
    std::mutex *_mutex;
  };

  std::string _path;
  std::ofstream _file;
  std::mutex _mutex;

  std::optional<Tee> _cout;
  std::optional<Tee> _cerr;
};

}
