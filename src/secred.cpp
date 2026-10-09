#include "secred.h"

SecRed::SecRed(const YAML::Node &node) : _family(node) {
  _family.init();
  _family.init_reduce(node, _reduce);

  _family.print();
  _reduce.print();
}

void SecRed::run() { _family.run_reduce(_reduce); }