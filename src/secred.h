#pragma once

#include <iostream>
#include <optional>

#include <yaml-cpp/yaml.h>

#include "family.h"
#include "log_files.h"

// Top-level driver: one family, one reduction.
struct SecRedOptions {
  // Accept settings keys this build does not know, with a warning.
  bool allow_unknown_settings = false;
  // Validate and report only; write nothing and run no reduction.
  bool dry_run = false;
};

class SecRed {
public:
  explicit SecRed(const YAML::Node &, const SecRedOptions & = {});

  // Execute the run described by the input node.  A dry run has already done
  // its work in the constructor and returns without reducing.
  void run();

private:
  Family _family;
  Reduce _reduce;
  bool _dry_run = false;

  std::optional<secred::RunLog> _log;
};
