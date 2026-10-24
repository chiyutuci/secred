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

  CLI11_PARSE(app, argc, argv)

  std::cout << "\n--------------- SecRed: sectorwise IBP reduction ---------------"
            << std::endl;
  try {
    YAML::Node config = YAML::LoadFile(configPath);

    SecRed secred(config);
    secred.run();
  } catch (std::runtime_error &e) {
    std::cerr << "\n Error: " << e.what() << "\n" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << std::endl;
  return EXIT_SUCCESS;
}
