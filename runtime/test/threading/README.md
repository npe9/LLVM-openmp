# Threading Portability Layer Integration with LLVM-OpenMP

This directory contains tests and examples for the threading portability layer integration with LLVM-OpenMP. The threading portability layer allows OpenMP to seamlessly switch between pthreads and Lithe as the underlying threading implementation.

## Overview

The threading portability layer provides a unified API for thread management and synchronization that can be implemented using either pthreads or Lithe. This allows applications to switch between threading backends without changing their code.

## Test Programs

### Basic Threading Test

The `threading_test.c` program demonstrates the basic functionality of the threading portability layer:

- Thread creation and management
- Synchronization using mutexes and barriers
- Incrementing a shared counter across multiple threads

### OpenMP Integration Example

The `openmp_threading_example.c` program demonstrates how to use the threading portability layer with OpenMP:

- Creating OpenMP parallel regions
- Using threading portability layer synchronization primitives within OpenMP
- Demonstrating thread-safe counter increments

## Building and Running

Each test program has three variants:

1. Default variant (uses the configured threading backend)
2. Pthread variant (explicitly uses pthreads backend)
3. Lithe variant (explicitly uses Lithe backend)

To build and run the tests:

```bash
# Configure with threading portability layer enabled
cmake -DLIBOMP_USE_THREADING_LAYER=TRUE -DLIBOMP_THREADING_BACKEND=pthreads ..

# Build
make

# Run tests
./threading_test_pthread
./openmp_threading_example_pthread

# Or with Lithe backend
cmake -DLIBOMP_USE_THREADING_LAYER=TRUE -DLIBOMP_THREADING_BACKEND=lithe ..
make
./threading_test_lithe
./openmp_threading_example_lithe
```

## Implementation Details

The threading portability layer is implemented in:

- `parlib/include/parlib/threading/thread.h`: Thread management API
- `parlib/include/parlib/threading/sync.h`: Synchronization primitives
- `parlib/src/threading/pthread/`: Pthreads backend implementation
- `parlib/src/threading/lithe/`: Lithe backend implementation

The integration with LLVM-OpenMP is implemented in:

- `runtime/src/threading_layer.h`: Unified interface for OpenMP
- `runtime/src/threading_layer.c`: Implementation of the interface

## Configuration

The threading portability layer can be configured through CMake options:

- `LIBOMP_USE_THREADING_LAYER`: Enable/disable the threading portability layer
- `LIBOMP_THREADING_BACKEND`: Select the threading backend (pthreads or lithe)

## Future Work

1. Add more comprehensive tests for all threading portability layer features
2. Optimize performance for specific OpenMP patterns
3. Add support for more threading backends
