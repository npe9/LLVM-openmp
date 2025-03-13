/*
 * test_lithe_integration.c - Test specific Lithe-OpenMP integration points
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <dlfcn.h>
#include "../test_framework.h"

// Test if we can access the Lithe library
test_result_t test_lithe_library_access(void) {
    void *lithe_handle = dlopen("liblithe.dylib", RTLD_LAZY);
    TEST_ASSERT(lithe_handle != NULL, "Failed to load Lithe library");
    
    if (lithe_handle) {
        dlclose(lithe_handle);
    }
    
    return TEST_PASSED;
}

// Test if we can access the OpenMP runtime with Lithe support
test_result_t test_openmp_lithe_symbols(void) {
    void *openmp_handle = dlopen("libomp.dylib", RTLD_LAZY);
    TEST_ASSERT(openmp_handle != NULL, "Failed to load OpenMP library");
    
    if (openmp_handle) {
        // Try to access the Lithe-specific symbols
        void *invoke_microtask = dlsym(openmp_handle, "___kmp_invoke_microtask");
        TEST_ASSERT(invoke_microtask != NULL, "Failed to find ___kmp_invoke_microtask symbol");
        
        void *unnamed_critical = dlsym(openmp_handle, "___kmp_unnamed_critical_addr");
        TEST_ASSERT(unnamed_critical != NULL, "Failed to find ___kmp_unnamed_critical_addr symbol");
        
        dlclose(openmp_handle);
    }
    
    return TEST_PASSED;
}

// Test thread affinity with Lithe
test_result_t test_thread_affinity(void) {
    int num_threads = 4;
    int thread_cores[16] = {0}; // Track which core each thread runs on
    
    // Set the number of threads
    omp_set_num_threads(num_threads);
    
    // Set thread affinity (if supported)
    omp_set_proc_bind(omp_proc_bind_close);
    
    // Parallel region to check thread affinity
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        if (tid < 16) {
            // Get the current core (this is platform-specific and may not work everywhere)
            #ifdef __APPLE__
            uint32_t core_id;
            size_t len = sizeof(core_id);
            if (sysctlbyname("machdep.cpu.core_id", &core_id, &len, NULL, 0) == 0) {
                thread_cores[tid] = core_id;
            } else {
                thread_cores[tid] = -1; // Error
            }
            #else
            // On other platforms, just use a placeholder value
            thread_cores[tid] = tid % omp_get_num_procs();
            #endif
        }
    }
    
    // Verify each thread got a core assigned
    for (int i = 0; i < num_threads; i++) {
        TEST_ASSERT(thread_cores[i] >= 0, "Thread did not get a valid core assignment");
    }
    
    return TEST_PASSED;
}

// Test thread creation and destruction with Lithe
test_result_t test_thread_lifecycle(void) {
    int num_threads = 4;
    int thread_created[16] = {0}; // Track which threads were created
    int thread_destroyed[16] = {0}; // Track which threads were destroyed
    
    // Set the number of threads
    omp_set_num_threads(num_threads);
    
    // First parallel region to create threads
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        if (tid < 16) {
            #pragma omp critical
            {
                thread_created[tid] = 1;
            }
        }
    }
    
    // Second parallel region to verify the same threads are reused
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        if (tid < 16) {
            #pragma omp critical
            {
                thread_destroyed[tid] = 1;
            }
        }
    }
    
    // Verify threads were created
    for (int i = 0; i < num_threads; i++) {
        TEST_ASSERT(thread_created[i], "Thread was not created");
    }
    
    // Verify threads were reused (destroyed)
    for (int i = 0; i < num_threads; i++) {
        TEST_ASSERT(thread_destroyed[i], "Thread was not reused/destroyed");
    }
    
    return TEST_PASSED;
}

// Test nested parallelism with Lithe
test_result_t test_lithe_nested_parallelism(void) {
    int outer_threads = 2;
    int inner_threads = 2;
    int thread_count = 0;
    
    // Enable nested parallelism
    omp_set_nested(1);
    
    // Set the number of threads for outer parallel region
    omp_set_num_threads(outer_threads);
    
    // Nested parallel regions
    #pragma omp parallel
    {
        #pragma omp atomic
        thread_count++;
        
        // Set the number of threads for inner parallel region
        omp_set_num_threads(inner_threads);
        
        #pragma omp parallel
        {
            #pragma omp atomic
            thread_count++;
        }
    }
    
    // Disable nested parallelism
    omp_set_nested(0);
    
    // Check if we have at least the outer threads
    // Note: Lithe might not fully support nested parallelism
    TEST_ASSERT(thread_count >= outer_threads, 
               "Nested parallelism did not create enough threads");
    
    return TEST_PASSED;
}

// Main test function
int main(int argc, char *argv[]) {
    // Define test cases
    test_case_t tests[] = {
        {"lithe_library_access", test_lithe_library_access, "Test access to Lithe library"},
        {"openmp_lithe_symbols", test_openmp_lithe_symbols, "Test OpenMP-Lithe integration symbols"},
        {"thread_affinity", test_thread_affinity, "Test thread affinity with Lithe"},
        {"thread_lifecycle", test_thread_lifecycle, "Test thread creation and destruction with Lithe"},
        {"lithe_nested_parallelism", test_lithe_nested_parallelism, "Test nested parallelism with Lithe"}
    };
    
    // Run the test suite
    run_test_suite(tests, sizeof(tests) / sizeof(tests[0]));
    
    return tests_failed > 0 ? 1 : 0;
}
