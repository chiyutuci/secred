# FireFly 2.0.3 sources patched to drop the 300-prime reconstruction cap.  They are
# compiled into the executable, so their definitions win over the ones in
# libfirefly.so; see README.md for what is patched and why.
add_library(firefly_patch STATIC ReconstHelper.cpp RatReconst.cpp)

target_include_directories(firefly_patch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_library(GMP_LIBRARY NAMES gmp)
if (NOT GMP_LIBRARY)
    message(FATAL_ERROR "libgmp not found (needed by lib/firefly_patch)")
endif ()
target_link_libraries(firefly_patch PUBLIC ${GMP_LIBRARY})
