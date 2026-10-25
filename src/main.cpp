#include <chrono>
#include <ctime>
#include <iostream>

#include "CLI11.hpp"
#include "yaml-cpp/yaml.h"

#include "secred.h"

int main(int argc, char **argv) {

  secred::mark_run_start();

  CLI::App app;

  std::string configPath;
  app.add_option("CONFIG", configPath, "The config file to read")
      ->type_name("");

  // Resume an interrupted run from its cache directory.  This is the command
  // line's copy of settings.resume: auto; the config file is the durable way
  // to ask for it.  A flag is used so `--resume config.yaml` cannot swallow
  // the config path the way an optional-valued option would.
  bool resume = false;
  app.add_flag("--resume", resume,
               "Resume an interrupted run from the cache directory "
               "(equivalent to settings.resume: auto)");

  // Print the run's plan and estimates without writing anything or reducing.
  bool dry_run = false;
  app.add_flag("--dry-run", dry_run,
               "Validate the config and report the sector, seed, equation, "
               "memory and thread estimates; write nothing and run nothing");

  // Older input files may carry settings this build dropped from the schema.
  bool allow_unknown_settings = false;
  app.add_flag("--allow-unknown-settings", allow_unknown_settings,
               "Accept unknown settings keys with a warning instead of "
               "failing");

  CLI11_PARSE(app, argc, argv)

  std::cout << "\n--------------- SecRed: sectorwise IBP reduction ---------------"
            << std::endl;
  try {
    YAML::Node config = YAML::LoadFile(configPath);

    if (resume) {
      if (!config.IsMap())
        throw std::runtime_error("--resume needs a config file with settings");
      config["settings"]["resume"] = true;
    }

    SecRedOptions options;
    options.dry_run = dry_run;
    options.allow_unknown_settings = allow_unknown_settings;
    SecRed secred(config, options);
    secred.run();
  } catch (std::runtime_error &e) {
    std::cerr << "\n Error: " << e.what() << "\n" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << std::endl;
  return EXIT_SUCCESS;
}
