#pragma once

#include <iostream>
#include <optional>

#include <yaml-cpp/yaml.h>

#include "family.h"
#include "log_files.h"

// Top-level driver: one family, one reduction.
class SecRed {
public:
  explicit SecRed(const YAML::Node &);

  // Execute the run described by the input node.
  void run();

private:
  Family _family;
  Reduce _reduce;

  std::optional<secred::RunLog> _log;
};
