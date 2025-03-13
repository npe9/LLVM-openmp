# CMakeLists.txt for Lithe integration tests

cmake_minimum_required(VERSION 3.10)
project(lithe_tests C)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Find Lithe library
find_path(LITHE_INCLUDE_DIR lithe/lithe.h
  PATHS
  /Users/npe/src/scythe/lithe/include
  /usr/local/include
  /usr/include
  NO_DEFAULT_PATH
)

find_library(LITHE_LIBRARY
  NAMES lithe
  PATHS
  /Users/npe/src/scythe/lithe/lib
  /usr/local/lib
  /usr/lib
  NO_DEFAULT_PATH
)

# Find DL library for dynamic loading tests
find_library(DL_LIBRARY dl)

# Set include directories
include_directories(${LITHE_INCLUDE_DIR})

# Basic test executable
add_executable(lithe_test lithe_test.c)
target_link_libraries(lithe_test PRIVATE OpenMP::OpenMP_C ${LITHE_LIBRARY})

# Test suite executables
add_executable(test_parallel tests/test_parallel.c)
target_link_libraries(test_parallel PRIVATE OpenMP::OpenMP_C ${LITHE_LIBRARY})

add_executable(test_worksharing tests/test_worksharing.c)
target_link_libraries(test_worksharing PRIVATE OpenMP::OpenMP_C ${LITHE_LIBRARY})

add_executable(test_synchronization tests/test_synchronization.c)
target_link_libraries(test_synchronization PRIVATE OpenMP::OpenMP_C ${LITHE_LIBRARY})

add_executable(test_lithe_integration tests/test_lithe_integration.c)
target_link_libraries(test_lithe_integration PRIVATE OpenMP::OpenMP_C ${LITHE_LIBRARY} ${DL_LIBRARY})

# Make the test runner script executable
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/run_tests.sh
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh ${CMAKE_CURRENT_BINARY_DIR}/run_tests.sh
  COMMAND chmod +x ${CMAKE_CURRENT_BINARY_DIR}/run_tests.sh
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh
  COMMENT "Copying test runner script"
)

# Add custom target to run all tests
add_custom_target(run_all_tests
  COMMAND ./run_tests.sh
  DEPENDS lithe_test test_parallel test_worksharing test_synchronization test_lithe_integration ${CMAKE_CURRENT_BINARY_DIR}/run_tests.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running all Lithe integration tests"
)
