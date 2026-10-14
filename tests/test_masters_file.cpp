#include "test_runner.h"

#include "masters_file.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

// The two master-basis files of a run: the sector-local basis the sector phase
// produces, and the basis the global reduction settles on.  They answer different
// questions (the sector phase's union is not the final basis), so they are two files
// with two names, and both hold one integral per line so either can be diffed against
// the other.

namespace {

std::string slurp(const std::string &path) {
  std::ifstream in(path);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

struct TempFile {
  explicit TempFile(const std::string &name) : path(name) {}
  ~TempFile() { std::remove(path.c_str()); }
  std::string path;
};

} // namespace

TEST(masters_file_names_pin_the_two_bases) {
  REQUIRE_EQ(std::string(secred::kMastersSectorFile), std::string("masters_sector"));
  REQUIRE_EQ(std::string(secred::kMastersGlobalFile), std::string("masters_global"));
}

TEST(write_masters_file_writes_one_integral_per_line) {
  const TempFile f("test_masters_file.txt");
  const std::vector<std::string> masters{"[1, 1, 1, 1]", "[0, 1, 1, 1]"};
  REQUIRE(secred::write_masters_file(f.path, masters));
  REQUIRE_EQ(slurp(f.path), std::string("[1, 1, 1, 1]\n[0, 1, 1, 1]\n"));
}

TEST(write_masters_file_truncates_an_existing_file) {
  const TempFile f("test_masters_file_trunc.txt");
  REQUIRE(secred::write_masters_file(f.path, std::vector<std::string>{"[1, 0, 0, 0]"}));
  // A shorter basis must not leave the tail of the previous one behind.
  REQUIRE(secred::write_masters_file(f.path, std::vector<std::string>{"[0, 0, 0, 1]"}));
  REQUIRE_EQ(slurp(f.path), std::string("[0, 0, 0, 1]\n"));
}

TEST(write_masters_file_handles_an_empty_basis) {
  const TempFile f("test_masters_file_empty.txt");
  REQUIRE(secred::write_masters_file(f.path, std::vector<std::string>{}));
  REQUIRE_EQ(slurp(f.path), std::string(""));
}

TEST(write_masters_file_reports_an_unwritable_path) {
  REQUIRE(!secred::write_masters_file("no_such_directory/masters_global",
                                      std::vector<std::string>{"[1, 1, 1, 1]"}));
}
