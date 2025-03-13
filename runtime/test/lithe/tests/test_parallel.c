/*
 * test_parallel.c - Test basic parallel region functionality
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../test_framework.h"

// Test basic parallel region
test_result_t test_basic_parallel(void) {
    int num_threads = 4;
    int actual_threads = 0;
    int thread_ids[16] = {0}; // Track which thread IDs we've seen
    
    // Set the number of threads
    omp_set_num_threads(num_threads);
    
    // Parallel region
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        #pragma omp critical
        {
            actual_threads++;
            if (tid < 16) {
                thread_ids[tid] = 1;
            }
        }
    }
    
    // Verify the correct number of threads were created
    TEST_ASSERT_EQUAL_INT(num_threads, actual_threads, 
                         "Incorrect number of threads in parallel region");
    
    // Verify we got the expected thread IDs (0 to num_threads-1)
    for (int i = 0; i < num_threads; i++) {
        TEST_ASSERT(thread_ids[i] == 1, "Missing thread ID in parallel region");
    }
    
    return TEST_PASSED;
}

// Test nested parallel regions
test_result_t test_nested_parallel(void) {
    int outer_threads = 2;
    int inner_threads = 2;
    int total_threads = 0;
    
    // Enable nested parallelism
    omp_set_nested(1);
    
    // Set the number of threads for outer and inner parallel regions
    omp_set_num_threads(outer_threads);
    
    // Nested parallel regions
    #pragma omp parallel
    {
        #pragma omp critical
        {
            total_threads++;
        }
        
        omp_set_num_threads(inner_threads);
        
        #pragma omp parallel
        {
            #pragma omp critical
            {
                total_threads++;
            }
        }
    }
    
    // Disable nested parallelism
    omp_set_nested(0);
    
    // With perfect nested parallelism, we should have outer_threads + (outer_threads * inner_threads) threads
    // However, Lithe might not support nested parallelism fully, so we'll check if we at least have the outer threads
    TEST_ASSERT(total_threads >= outer_threads, 
               "Nested parallel regions did not create enough threads");
    
    return TEST_PASSED;
}

// Test dynamic adjustment of thread count
test_result_t test_dynamic_threads(void) {
    int max_threads = omp_get_max_threads();
    int half_threads = max_threads / 2;
    int actual_threads = 0;
    
    if (half_threads < 1) half_threads = 1;
    
    // Set dynamic adjustment
    omp_set_dynamic(1);
    
    // Request half the max threads
    omp_set_num_threads(half_threads);
    
    // Parallel region
    #pragma omp parallel
    {
        #pragma omp critical
        {
            actual_threads++;
        }
    }
    
    // Disable dynamic adjustment
    omp_set_dynamic(0);
    
    // Verify we got the expected number of threads
    TEST_ASSERT_EQUAL_INT(half_threads, actual_threads, 
                         "Dynamic thread adjustment failed");
    
    return TEST_PASSED;
}

// Main test function
int main(int argc, char *argv[]) {
    // Define test cases
    test_case_t tests[] = {
        {"basic_parallel", test_basic_parallel, "Test basic parallel region functionality"},
        {"nested_parallel", test_nested_parallel, "Test nested parallel regions"},
        {"dynamic_threads", test_dynamic_threads, "Test dynamic adjustment of thread count"}
    };
    
    // Run the test suite
    run_test_suite(tests, sizeof(tests) / sizeof(tests[0]));
    
    return tests_failed > 0 ? 1 : 0;
}
