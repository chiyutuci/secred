cmake_minimum_required(VERSION 3.16)

# set c++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add compile options

# project name
project(secred VERSION 0.1.0)

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
target_link_libraries(secred ${FIREFLY_LIBRARY})

# unit tests
add_subdirectory(tests)
