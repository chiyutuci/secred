#include "test_runner.h"

// All test cases are registered via the TEST() macro in their respective
// translation units.  This file only provides main().

int main() { return run_all_tests(); }
