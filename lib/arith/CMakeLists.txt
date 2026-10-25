# set library sources
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ARITH_SRC)
add_library(arith ${ARITH_SRC})
target_include_directories(arith PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# add third party libraries
# gmp library
find_library(GMP_LIBRARY NAMES gmp)
if (NOT GMP_LIBRARY)
    message(FATAL_ERROR "gmp library not found")
endif ()
target_link_libraries(arith ${GMP_LIBRARY})
# flint library
find_library(FLINT_LIBRARY NAMES flint)
if (NOT FLINT_LIBRARY)
    message(FATAL_ERROR "flint library not found")
endif ()
target_link_libraries(arith ${FLINT_LIBRARY})
# mpfr library
find_library(MPFR_LIBRARY NAMES mpfr)
if (NOT MPFR_LIBRARY)
    message(FATAL_ERROR "MPFR library not found")
endif()
target_link_libraries(arith ${MPFR_LIBRARY})