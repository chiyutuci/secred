cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Collect all test translation units (everything except test_main.cpp)
set(TEST_SOURCES
    test_main.cpp
    test_symmetry.cpp
    test_rawintegral.cpp
    test_arithmetic.cpp
    test_equation.cpp
    test_sector.cpp
    test_family.cpp
    test_ffred.cpp
    test_sector_engine.cpp
    test_forward_selector.cpp
    test_global_order.cpp
    test_sym_coeff.cpp
    test_progress_bar.cpp
    test_masters_file.cpp
    test_run_report.cpp
    test_utils.cpp
)

# Core secred source files needed by the tests (exclude main.cpp and secred.cpp).
set(SECRED_CORE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/equation.cpp
    ${CMAKE_SOURCE_DIR}/src/family.cpp
    ${CMAKE_SOURCE_DIR}/src/forward_selector.cpp
    ${CMAKE_SOURCE_DIR}/src/globalreduce.cpp
    ${CMAKE_SOURCE_DIR}/src/numeric_solver_mod.cpp
    ${CMAKE_SOURCE_DIR}/src/ffred_solver.cpp
    ${CMAKE_SOURCE_DIR}/src/sector.cpp
    ${CMAKE_SOURCE_DIR}/src/symmetry.cpp
    ${CMAKE_SOURCE_DIR}/src/utils.cpp
)

add_executable(secred_tests ${TEST_SOURCES} ${SECRED_CORE_SOURCES})

# The test runner header lives here
target_include_directories(secred_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/lib
    ${CMAKE_SOURCE_DIR}/include
)

# Reuse the same libraries as the main executable
target_link_libraries(secred_tests
    arith
    ${YAML_LIBRARY}
    ${GINAC_LIBRARY}
    ${CLN_LIBRARY}
    ${FFLOW_LIBRARY}
)

# Register with CTest so "cmake --build build --target test" works
enable_testing()
add_test(NAME secred_unit_tests COMMAND secred_tests)
