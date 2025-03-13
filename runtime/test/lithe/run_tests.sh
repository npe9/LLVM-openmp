#!/bin/bash
#
# run_tests.sh - Compile and run all Lithe-OpenMP integration tests
#

set -e

# Directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "=== Configuring tests with CMake ==="
cmake .. -DLIBOMP_USE_LITHE=ON

# Build tests
echo "=== Building tests ==="
make -j4

# Run tests
echo "=== Running tests ==="
echo ""

# Function to run a test and report results
run_test() {
    local test_name="$1"
    echo "Running test: $test_name"
    
    if [ -x "./$test_name" ]; then
        ./$test_name
        local result=$?
        if [ $result -eq 0 ]; then
            echo "Test $test_name PASSED"
        else
            echo "Test $test_name FAILED with exit code $result"
            FAILED_TESTS="$FAILED_TESTS $test_name"
        fi
    else
        echo "Error: Test executable $test_name not found or not executable"
        FAILED_TESTS="$FAILED_TESTS $test_name"
    fi
    echo ""
}

# List of tests to run
TESTS=(
    "test_parallel"
    "test_worksharing"
    "test_synchronization"
    "test_lithe_integration"
)

# Run all tests
FAILED_TESTS=""
for test in "${TESTS[@]}"; do
    run_test "$test"
done

# Report summary
echo "=== Test Summary ==="
if [ -z "$FAILED_TESTS" ]; then
    echo "All tests PASSED"
    exit 0
else
    echo "The following tests FAILED:$FAILED_TESTS"
    exit 1
fi
