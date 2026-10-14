#include "secred.h"

SecRed::SecRed(const YAML::Node &node) : _family(node) {
  // init() wipes and recreates the cache directory, so the log copy can only be
  // opened after it.  Opened here rather than in run() so that the whole run is in
  // the log, this constructor's own printout included.
  _family.init();
  _log.emplace(_family.cache_dir());

  _family.init_reduce(node, _reduce);

  _family.print();
  _reduce.print();
}

void SecRed::run() { _family.run_reduce(_reduce); }
