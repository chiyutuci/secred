#pragma once

#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal single-header test runner.
// Usage:
//   TEST(my_test_name) { REQUIRE(1 + 1 == 2); }
//   int main() { return run_all_tests(); }
// ---------------------------------------------------------------------------

struct TestCase {
  std::string name;
  std::function<void()> fn;
};

inline std::vector<TestCase> &test_registry() {
  static std::vector<TestCase> reg;
  return reg;
}

struct TestRegistrar {
  TestRegistrar(const char *name, std::function<void()> fn) {
    test_registry().push_back({name, std::move(fn)});
  }
};

#define TEST(name)                                                             \
  static void _test_body_##name();                                            \
  static TestRegistrar _reg_##name(#name, _test_body_##name);                 \
  static void _test_body_##name()

// Assertion: fails the test with the expression text and line number.
#define REQUIRE(expr)                                                          \
  do {                                                                        \
    if (!(expr)) {                                                             \
      std::ostringstream _oss;                                                 \
      _oss << "  REQUIRE failed: " << #expr << "  (line " << __LINE__ << ")"; \
      throw std::runtime_error(_oss.str());                                   \
    }                                                                         \
  } while (0)

// Assertion with a caller-supplied message, for the cases where the expression
// text alone cannot say what went wrong (a printed string that does not parse,
// for instance).
#define REQUIRE_MSG(expr, msg)                                                 \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::ostringstream _oss;                                                 \
      _oss << "  REQUIRE_MSG failed: " << (msg) << "  (line " << __LINE__      \
           << ")";                                                             \
      throw std::runtime_error(_oss.str());                                    \
    }                                                                          \
  } while (0)

// Check that the expression throws any exception.
#define REQUIRE_THROWS(expr)                                                   \
  do {                                                                        \
    bool _threw = false;                                                      \
    try {                                                                     \
      (void)(expr);                                                           \
    } catch (...) {                                                           \
      _threw = true;                                                          \
    }                                                                         \
    if (!_threw) {                                                            \
      std::ostringstream _oss;                                                 \
      _oss << "  REQUIRE_THROWS: expected exception from: " << #expr          \
           << "  (line " << __LINE__ << ")";                                  \
      throw std::runtime_error(_oss.str());                                   \
    }                                                                         \
  } while (0)

// Check that the expression throws a specific exception type.
#define REQUIRE_THROWS_AS(expr, ExcType)                                       \
  do {                                                                        \
    bool _threw = false;                                                      \
    try {                                                                     \
      (void)(expr);                                                           \
    } catch (const ExcType &) {                                               \
      _threw = true;                                                          \
    } catch (...) {                                                           \
    }                                                                         \
    if (!_threw) {                                                            \
      std::ostringstream _oss;                                                 \
      _oss << "  REQUIRE_THROWS_AS: expected " << #ExcType                    \
           << " from: " << #expr << "  (line " << __LINE__ << ")";           \
      throw std::runtime_error(_oss.str());                                   \
    }                                                                         \
  } while (0)

#define REQUIRE_EQ(a, b)                                                        \
  do {                                                                          \
    if (!((a) == (b))) {                                                        \
      std::ostringstream _oss;                                                   \
      _oss << "  REQUIRE_EQ failed: " << #a << " == " << #b                     \
           << "  (line " << __LINE__ << ")";                                    \
      throw std::runtime_error(_oss.str());                                     \
    }                                                                           \
  } while (0)

#define REQUIRE_NE(a, b)                                                        \
  do {                                                                          \
    if (!((a) != (b))) {                                                        \
      std::ostringstream _oss;                                                   \
      _oss << "  REQUIRE_NE failed: " << #a << " != " << #b                     \
           << "  (line " << __LINE__ << ")";                                    \
      throw std::runtime_error(_oss.str());                                     \
    }                                                                           \
  } while (0)

#define REQUIRE_NOTHROW(expr)                                                   \
  do {                                                                          \
    try {                                                                       \
      (void)(expr);                                                             \
    } catch (const std::exception &_e) {                                        \
      std::ostringstream _oss;                                                   \
      _oss << "  REQUIRE_NOTHROW: unexpected exception from: " << #expr         \
           << ": " << _e.what() << "  (line " << __LINE__ << ")";              \
      throw std::runtime_error(_oss.str());                                     \
    }                                                                           \
  } while (0)

inline int run_all_tests() {
  int passed = 0, failed = 0;
  for (auto &tc : test_registry()) {
    std::cout << "[ RUN  ] " << tc.name << "\n";
    try {
      tc.fn();
      std::cout << "[ PASS ] " << tc.name << "\n";
      ++passed;
    } catch (const std::exception &e) {
      std::cout << "[ FAIL ] " << tc.name << "\n" << e.what() << "\n";
      ++failed;
    } catch (...) {
      std::cout << "[ FAIL ] " << tc.name << "\n  (unknown exception)\n";
      ++failed;
    }
  }
  std::cout << "\n=== Results: " << passed << " passed, " << failed
            << " failed ===\n";
  return failed == 0 ? 0 : 1;
}
