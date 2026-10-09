#include <chrono>
#include <ctime>
#include <iostream>

#include <malloc.h>

#include "CLI11.hpp"
#include "yaml-cpp/yaml.h"

#include "secred.h"

int main(int argc, char **argv) {
  // glibc tuning, before any allocation: (1) allocations above 64 KB go
  // straight to mmap, so they are munmap'd on free and the process RSS drops
  // immediately instead of lingering in a per-thread malloc arena; (2) cap the
  // arena count so the worker threads share fewer arenas and freed memory does
  // not fragment across dozens of them.
  mallopt(M_MMAP_THRESHOLD, 64 * 1024);
  mallopt(M_ARENA_MAX, 8);

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