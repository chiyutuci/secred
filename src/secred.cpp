#include "secred.h"

namespace {
FamilyOptions family_options(const SecRedOptions &options) {
  FamilyOptions family;
  family.allow_unknown_settings = options.allow_unknown_settings;
  family.dry_run = options.dry_run;
  return family;
}
} // namespace

SecRed::SecRed(const YAML::Node &node, const SecRedOptions &options)
    : _family(node, family_options(options)), _dry_run(options.dry_run) {

  _family.init();
  if (!_dry_run)
    _log.emplace(_family.cache_dir());

  _family.init_reduce(node, _reduce);

  _family.print();
  _reduce.print();
  if (_dry_run)
    _family.print_dry_run(_reduce);
}

void SecRed::run() {
  if (_dry_run)
    return;
  _family.run_reduce(_reduce);
}
