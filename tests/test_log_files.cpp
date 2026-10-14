#include "test_runner.h"

#include "log_files.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

// The run's two log files, both under the cache directory: the copy of the run's
// console output (RunLog) and FireFly's own log, which FireFly writes to a
// hardcoded path in the working directory and which is collected from there.

namespace {

std::string slurp(const std::string &path) {
  std::ifstream in(path);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

void write_file(const std::string &path, const std::string &text) {
  std::ofstream out(path, std::ios::trunc);
  out << text;
}

// A directory of its own per test, removed on the way out.  The FireFly-log tests
// need one to collect into and one to work from, so they use two.
struct TempDir {
  explicit TempDir(const std::string &name) : path(name) {
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    // The tests below move the process into another directory, so anything they
    // hand to the code under test has to be spelled absolutely.
    abs = std::filesystem::absolute(path).string();
  }
  ~TempDir() { std::filesystem::remove_all(path); }
  std::string file(const std::string &name) const {
    return (std::filesystem::path(path) / name).string();
  }
  std::string path;
  std::string abs;
};

// FireFly's log path is relative to the working directory, so those tests have to
// move the process into their own directory and back -- and back even when an
// assertion throws, or every later test would run somewhere it did not expect.
struct CwdGuard {
  explicit CwdGuard(const std::string &to) : old(std::filesystem::current_path()) {
    std::filesystem::current_path(to);
  }
  ~CwdGuard() { std::filesystem::current_path(old); }
  std::filesystem::path old;
};

} // namespace

TEST(log_file_names_pin_the_two_names) {
  REQUIRE_EQ(std::string(secred::kRunLogFile), std::string("secred.log"));
  REQUIRE_EQ(std::string(secred::kFireFlyLogFile), std::string("firefly.log"));
}

TEST(run_log_saves_the_console_output_to_the_cache_directory) {
  const TempDir dir("test_run_log_tee");
  {
    secred::RunLog log(dir.path);
    REQUIRE(log.ok());
    std::cout << "run log: this line belongs in the log\n";
    std::cerr << "run log: and so does this one\n";
  }
  const std::string text = slurp(dir.file("secred.log"));
  REQUIRE(text.find("run log: this line belongs in the log") != std::string::npos);
  REQUIRE(text.find("run log: and so does this one") != std::string::npos);
}

TEST(run_log_stops_capturing_when_it_goes_out_of_scope) {
  const TempDir dir("test_run_log_scope");
  {
    secred::RunLog log(dir.path);
    std::cout << "captured\n";
  }
  std::cout << "not captured\n";
  const std::string text = slurp(dir.file("secred.log"));
  REQUIRE(text.find("captured") != std::string::npos);
  // The streams are handed back on destruction: what is written afterwards stays
  // on the terminal instead of landing in a log whose owner is gone.
  REQUIRE(text.find("not captured") == std::string::npos);
}

TEST(run_log_truncates_a_log_left_by_an_earlier_run) {
  const TempDir dir("test_run_log_truncate");
  {
    secred::RunLog log(dir.path);
    std::cout << "first run\n";
  }
  {
    secred::RunLog log(dir.path);
    std::cout << "second run\n";
  }
  const std::string text = slurp(dir.file("secred.log"));
  REQUIRE(text.find("second run") != std::string::npos);
  REQUIRE(text.find("first run") == std::string::npos);
}

TEST(run_log_reports_a_log_it_cannot_open_without_stopping_the_run) {
  // A cache directory that does not exist: the run keeps going, the terminal still
  // carries its output, and ok() is what says the log is missing.
  const secred::RunLog log("test_run_log_no_such_directory/");
  REQUIRE(!log.ok());
}

TEST(collect_firefly_log_moves_the_log_into_the_cache_directory) {
  const TempDir work("test_firefly_log_work");
  const TempDir dir("test_firefly_log_cache");
  {
    const CwdGuard cwd(work.path);
    write_file(secred::kFireFlyLogFile, "firefly log body\n");
    REQUIRE(secred::collect_firefly_log(dir.abs));
  }
  REQUIRE_EQ(slurp(dir.file("firefly.log")), std::string("firefly log body\n"));
  // Moved, not copied: the working directory is left without it.
  REQUIRE(!std::filesystem::exists(work.file("firefly.log")));
}

TEST(collect_firefly_log_is_a_no_op_without_a_log_to_collect) {
  const TempDir work("test_firefly_log_work_empty");
  const TempDir dir("test_firefly_log_cache_empty");
  bool collected = true;
  {
    const CwdGuard cwd(work.path);
    collected = secred::collect_firefly_log(dir.abs);
  }
  // The numeric path never writes one; collecting nothing is not a failure, and it
  // must not leave an empty file behind either.
  REQUIRE(!collected);
  REQUIRE(!std::filesystem::exists(dir.file("firefly.log")));
}

TEST(reset_firefly_log_drops_a_log_left_by_an_earlier_run) {
  // FireFly opens its log in append mode, so a stale file would be prefixed onto
  // this run's log instead of being replaced by it.
  const TempDir work("test_firefly_log_reset");
  bool gone = false;
  {
    const CwdGuard cwd(work.path);
    write_file(secred::kFireFlyLogFile, "an earlier run\n");
    secred::reset_firefly_log();
    gone = !std::filesystem::exists(secred::kFireFlyLogFile);
  }
  REQUIRE(gone);
}

TEST(reset_firefly_log_is_a_no_op_without_a_log_to_reset) {
  const TempDir work("test_firefly_log_reset_empty");
  {
    const CwdGuard cwd(work.path);
    secred::reset_firefly_log();
  }
  REQUIRE(!std::filesystem::exists(work.file("firefly.log")));
}
