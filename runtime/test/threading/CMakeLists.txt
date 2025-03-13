# CMake build script for the threading portability layer tests

# Add the threading test executable
add_executable(threading_test threading_test.c)

# Set include directories
target_include_directories(threading_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../parlib/include
)

# Link with parlib
target_link_libraries(threading_test PRIVATE
  parlib
)

# Add pthread test variant
add_executable(threading_test_pthread threading_test.c)
target_include_directories(threading_test_pthread PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../parlib/include
)
target_compile_definitions(threading_test_pthread PRIVATE
  -DPARLIB_USE_PTHREADS
)
target_link_libraries(threading_test_pthread PRIVATE
  parlib
  pthread
)

# Add Lithe test variant
add_executable(threading_test_lithe threading_test.c)
target_include_directories(threading_test_lithe PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../parlib/include
)
target_compile_definitions(threading_test_lithe PRIVATE
  -DPARLIB_USE_LITHE
)
target_link_libraries(threading_test_lithe PRIVATE
  parlib
  lithe
)

# Add tests to CTest
add_test(NAME threading_test_pthread COMMAND threading_test_pthread)
add_test(NAME threading_test_lithe COMMAND threading_test_lithe)

# Install the test executables
install(TARGETS 
  threading_test
  threading_test_pthread
  threading_test_lithe
  RUNTIME DESTINATION test
)
