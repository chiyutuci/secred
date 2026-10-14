// The run's own log files.  Both live in the cache directory, so a finished run
// leaves everything it produced in one place and the working directory clean:
//
//   secred.log    a copy of the run's console output, written by RunLog;
//   firefly.log   FireFly's log, collected out of the working directory by
//                 collect_firefly_log.
//
// The copy of the console output is a tee on std::cout and std::cerr, not a
// redirection of the file descriptors: everything the program prints with the C++
// streams lands in the log, in the order it reaches the terminal, while a write that
// bypasses the streams (a library's printf(3), a subprocess) does not.  Nothing in a
// run goes through such a path today.
//
// Header-only so tests can drive both against their own directories.

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

// FireFly's Reconstructor opens its log at the literal path "firefly.log", relative
// to the process's working directory (Reconstructor.hpp, in every place it opens the
// logger); the path is not a parameter, so the log cannot be redirected while the
// reconstruction is in flight.  These two calls bracket the reconstruction instead:
// reset before it, so a leftover file from an earlier run is not prefixed onto this
// run's log (FireFly opens in append mode), and collect after it, so the log ends up
// beside the run's other artifacts.

// Drop a "firefly.log" left in the working directory.  A no-op when there is none.
inline void reset_firefly_log() {
  std::error_code ec;
  std::filesystem::remove(kFireFlyLogFile, ec);
}

// Move the "firefly.log" this run wrote into `cache_dir`, replacing any earlier one.
// False when there is nothing to collect (the numeric FiniteFlow path never writes
// one, and a failed reconstruction may have left none either), which is not an error.
inline bool collect_firefly_log(const std::string &cache_dir) {
  std::error_code ec;
  if (!std::filesystem::exists(kFireFlyLogFile, ec))
    return false;
  const std::filesystem::path dst =
      std::filesystem::path(cache_dir) / kFireFlyLogFile;
  std::filesystem::rename(kFireFlyLogFile, dst, ec);
  if (ec) {
    // rename(2) cannot cross a filesystem boundary: copy, then drop the original.
    ec.clear();
    std::filesystem::copy_file(kFireFlyLogFile, dst,
                               std::filesystem::copy_options::overwrite_existing, ec);
    if (ec)
      return false;
    std::filesystem::remove(kFireFlyLogFile, ec);
  }
  return true;
}

// Tees std::cout and std::cerr into <cache_dir>/secred.log for as long as it lives,
// handing the streams back untouched on destruction.  A log that cannot be opened is
// not fatal: the run continues, its output still reaches the terminal, and ok() says
// the log is missing.
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

  // False when the log could not be opened.
  bool ok() const { return _file.is_open(); }
  const std::string &path() const { return _path; }

private:
  // One tee per stream, both writing to the same file.  Every write goes to the
  // stream's own sink (the terminal) and to the file under one mutex: the sector
  // phase prints from many threads at once, and the streams' own locking only
  // covers a single insertion.
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
  // Set only once the file is open, so that a RunLog which could not open its log
  // never touches the streams.
  std::optional<Tee> _cout;
  std::optional<Tee> _cerr;
};

} // namespace secred
