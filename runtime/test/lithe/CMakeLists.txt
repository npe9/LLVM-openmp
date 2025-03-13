# CMakeLists.txt for Lithe integration test

cmake_minimum_required(VERSION 3.10)
project(lithe_test C)

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

# Add the executable
add_executable(lithe_test lithe_test.c)

# Include directories
include_directories(${LITHE_INCLUDE_DIR})

# Link against OpenMP and Lithe
target_link_libraries(lithe_test PRIVATE OpenMP::OpenMP_C ${LITHE_LIBRARY})

# Add custom target to run the test
add_custom_target(run_lithe_test
  COMMAND ./lithe_test
  DEPENDS lithe_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running Lithe integration test"
)
