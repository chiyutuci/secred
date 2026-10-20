cmake_minimum_required(VERSION 3.16)

# set c++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add compile options

# project name
project(secred VERSION 0.1.0)

# Optional build tuning, off by default so the released package stays portable.
#
# Both were measured against the default build on the F234 benchmark at 64 threads,
# back to back on an idle machine, and both are left OFF because they made the run
# *slower*: -march=native -mtune=native with LTO moved the per-point forward
# elimination from 2.670 s to 3.204 s and the total from 3min13s to 3min26s, while the
# default build reproduced itself to within 1 %.  See bench/README.md for the table.
# They stay as switches because the balance can differ on another machine or another
# compiler.
option(SECRED_NATIVE "Build with -march=native -mtune=native" OFF)
option(SECRED_LTO "Build with interprocedural optimisation (LTO)" OFF)

if (SECRED_NATIVE)
    add_compile_options(-march=native -mtune=native)
endif ()

if (SECRED_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT SECRED_LTO_OK OUTPUT SECRED_LTO_MSG)
    if (SECRED_LTO_OK)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    else ()
        message(WARNING "SECRED_LTO=ON but LTO is not supported: ${SECRED_LTO_MSG}")
    endif ()
endif ()

# jemalloc is a required dependency, not a tuning option.  The sector stage runs 64
# threads, each allocating GiNaC expressions, against glibc's M_ARENA_MAX=8 (main.cpp
# caps it for the memory gate of ARCHITECTURE.md 8.3), and that cap serialises them.
# Measured on F234 at 64 threads, back to back with the same binary: sector reduction
# 1min1s -> 27.8s, total 3min13s -> 2min30s, peak 7.26 GB -> 6.20 GB, with the
# reductions byte-identical.  Linking a library that defines malloc is enough --
# libjemalloc comes before libc on the link line -- so no LD_PRELOAD is needed.
#
# This was `option(SECRED_JEMALLOC ...)` with a warning when the library was missing,
# which produced builds that are correct and twice as slow with nothing saying so.  A
# dependency worth that much is either present or the configure stops here.
find_library(JEMALLOC_LIBRARY NAMES jemalloc)
if (NOT JEMALLOC_LIBRARY)
    message(FATAL_ERROR
        "libjemalloc not found, and SecRed requires it.  Without jemalloc the 64 "
        "sector threads serialise on glibc's arena lock: on F234 at 64 threads the "
        "sector stage takes 1min1s instead of 27.8s and the run 3min13s instead of "
        "2min30s, with the results identical -- i.e. nothing else would tell you.\n"
        "  Debian/Ubuntu:  sudo apt install libjemalloc-dev\n"
        "  Fedora/RHEL:    sudo dnf install jemalloc-devel\n"
        "  macOS:          brew install jemalloc\n"
        "Installed under a prefix CMake does not search?  Pass it in directly:\n"
        "  cmake -S . -B build -DJEMALLOC_LIBRARY=/path/to/libjemalloc.so")
endif ()
message(STATUS "jemalloc: ${JEMALLOC_LIBRARY}")

# set project directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(SECRED_SRC_DIR src)
set(SECRED_LIB_DIR lib)
set(SECRED_INCLUDE_DIR include)
set(SECRED_LIBRARIES arith)

# add source files
aux_source_directory(${SECRED_SRC_DIR} SECRED_SRC)
add_executable(secred ${SECRED_SRC})
target_include_directories(secred PUBLIC ${SECRED_SRC_DIR} ${SECRED_INCLUDE_DIR} ${SECRED_LIB_DIR})

# add libraries
foreach (LIBRARY ${SECRED_LIBRARIES})
    add_subdirectory(${SECRED_LIB_DIR}/${LIBRARY})
endforeach ()
target_link_libraries(secred ${SECRED_LIBRARIES})

# Patched FireFly sources: lift the 300-prime reconstruction cap (lib/firefly_patch).
# Linked before libfirefly so its definitions win inside the executable.
add_subdirectory(${SECRED_LIB_DIR}/firefly_patch)

# add third party libraries
# yaml-cpp library
find_library(YAML_LIBRARY NAMES yaml-cpp)
if (NOT YAML_LIBRARY)
    message(FATAL_ERROR "yaml-cpp library not found")
endif ()
target_link_libraries(secred ${YAML_LIBRARY})

#ginac library
find_library(GINAC_LIBRARY NAMES ginac)
find_library(CLN_LIBRARY NAMES cln)
if (NOT GINAC_LIBRARY)
    message(FATAL_ERROR "ginac library not found")
endif ()
target_link_libraries(secred ${GINAC_LIBRARY} ${CLN_LIBRARY})

#finiteflow library
find_library(FFLOW_LIBRARY NAMES fflow)
if (NOT FFLOW_LIBRARY)
    message(FATAL_ERROR "finiteflow library not found")
endif ()
target_link_libraries(secred ${FFLOW_LIBRARY})

#firefly library (the global_tool: firefly backend; FLINT/GMP/MPFR arrive transitively
# through libfirefly/libfflow)
find_library(FIREFLY_LIBRARY NAMES firefly PATHS /usr/local/lib)
if (NOT FIREFLY_LIBRARY)
    message(FATAL_ERROR "libfirefly not found (FireFly is required for global_tool: firefly)")
endif ()
target_link_libraries(secred firefly_patch ${FIREFLY_LIBRARY})

# jemalloc last on the line, but before libc, which the linker appends: it is the first
# definition of malloc/free the link sees, so it is the one the program calls (see
# above).  It is not optional, so it is not guarded.
target_link_libraries(secred ${JEMALLOC_LIBRARY})

# unit tests
add_subdirectory(tests)
