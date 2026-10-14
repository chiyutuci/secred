#pragma once

#include <iostream>
#include <optional>

#include <yaml-cpp/yaml.h>

#include "family.h"
#include "log_files.h"

class SecRed {
public:
  explicit SecRed(const YAML::Node &);

public:
  // run the reduction
  void run();

private:
  Family _family;
  Reduce _reduce;
  // The copy of this run's console output under the cache directory.  Declared
  // last so that it is destroyed first: the streams go back to the terminal
  // before the family and the reduction state are torn down.
  std::optional<secred::RunLog> _log;
};
