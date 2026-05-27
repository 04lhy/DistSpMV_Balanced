cmake_minimum_required(VERSION 3.16)
project(DistSpMV_Balanced LANGUAGES CXX C
        VERSION 2.0.0
        DESCRIPTION "Distributed SpMV with balanced computation and communication")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Dependencies ──────────────────────────────────────────────────────
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# ── Build type defaults ───────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── Executable ────────────────────────────────────────────────────────
add_executable(dist_spmv
    src/main.cpp
    src/mmio.cpp
    src/utils.cpp
    src/reordering.cpp
    src/partition.cpp
    src/comm_setup.cpp
    src/spmv_solver.cpp
)

target_include_directories(dist_spmv PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# MS-MPI support (needed when using plain g++ instead of mpicxx)
# On Windows, cmake's FindMPI doesn't always populate include/library paths
# for non-MSVC compilers.  We add them explicitly.
if(WIN32)
    set(MSMPI_INC "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
    # Search for msmpi import library in common MinGW locations
    find_library(MSMPI_LIB msmpi
        PATHS
            /mingw64/x86_64-w64-mingw32/lib/msmpi
            /d/Tools/mingw64/x86_64-w64-mingw32/lib/msmpi
            "C:/msys64/mingw64/x86_64-w64-mingw32/lib/msmpi"
        PATH_SUFFIXES lib
        NO_DEFAULT_PATH
    )

    if(EXISTS "${MSMPI_INC}")
        target_include_directories(dist_spmv PRIVATE "${MSMPI_INC}")
        message(STATUS "Adding MS-MPI include: ${MSMPI_INC}")
    endif()
    if(MSMPI_LIB AND EXISTS "${MSMPI_LIB}")
        target_link_libraries(dist_spmv PRIVATE "${MSMPI_LIB}")
        message(STATUS "Adding MS-MPI lib: ${MSMPI_LIB}")
    else()
        message(WARNING "MS-MPI import library not found.  Set -DMSMPI_LIB=<path> to your msmpi/libmsmpi.a")
    endif()
else()
    target_link_libraries(dist_spmv PRIVATE MPI::MPI_CXX)
endif()

target_link_libraries(dist_spmv PRIVATE OpenMP::OpenMP_CXX)

target_compile_options(dist_spmv PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /openmp>
    $<$<CXX_COMPILER_ID:GNU>:-O3 -march=native>
    $<$<CXX_COMPILER_ID:Clang>:-O3 -march=native>
)

# ── Optional METIS ────────────────────────────────────────────────────
# Set -DMETIS_INCLUDE_DIR, -DMETIS_LIBRARY, -DGKLIB_LIBRARY to enable.
# On Linux/macOS with system METIS (apt/brew), these are found automatically.
find_path(METIS_INCLUDE_DIR metis.h
    PATHS ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES include)
find_library(METIS_LIBRARY metis
    PATHS ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib)
find_library(GKLIB_LIBRARY GKlib
    PATHS ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib)

if(METIS_INCLUDE_DIR AND METIS_LIBRARY AND GKLIB_LIBRARY)
    add_library(METIS::METIS STATIC IMPORTED)
    set_target_properties(METIS::METIS PROPERTIES
        IMPORTED_LOCATION "${METIS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${METIS_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${GKLIB_LIBRARY}")
    target_compile_definitions(dist_spmv PRIVATE
        DISTSPMV_HAS_METIS IDXTYPEWIDTH=32 REALTYPEWIDTH=32)
    target_link_libraries(dist_spmv PRIVATE METIS::METIS)
    message(STATUS "METIS found — enabling --reorder metis")
    message(STATUS "  METIS include: ${METIS_INCLUDE_DIR}")
    message(STATUS "  METIS lib:     ${METIS_LIBRARY}")
    message(STATUS "  GKlib lib:     ${GKLIB_LIBRARY}")
else()
    message(STATUS "METIS not found — --reorder metis disabled (use rcm or none)")
endif()

# ── Tests (optional) ──────────────────────────────────────────────────
option(BUILD_TESTS "Build test executable (requires tests/test_spmv.cpp)" OFF)
if(BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_spmv.cpp")
    enable_testing()
    add_executable(test_spmv tests/test_spmv.cpp
        src/mmio.cpp src/utils.cpp src/reordering.cpp
        src/partition.cpp src/comm_setup.cpp src/spmv_solver.cpp
    )
    target_include_directories(test_spmv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_spmv PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
    add_test(NAME test_spmv COMMAND test_spmv)
elseif(BUILD_TESTS)
    message(WARNING "BUILD_TESTS=ON but tests/test_spmv.cpp not found — skipping")
endif()

# ── Install ───────────────────────────────────────────────────────────
install(TARGETS dist_spmv RUNTIME DESTINATION bin)
