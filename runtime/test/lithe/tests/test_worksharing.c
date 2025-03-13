/*
 * test_worksharing.c - Test OpenMP worksharing constructs with Lithe
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../test_framework.h"

// Test parallel for loop
test_result_t test_parallel_for(void) {
    int num_threads = 4;
    int num_iterations = 100;
    int iteration_count[100] = {0}; // Track which iterations were executed
    
    // Set the number of threads
    omp_set_num_threads(num_threads);
    
    // Parallel for loop
    #pragma omp parallel for
    for (int i = 0; i < num_iterations; i++) {
        // Mark this iteration as executed
        iteration_count[i]++;
    }
    
    // Verify each iteration was executed exactly once
    for (int i = 0; i < num_iterations; i++) {
        TEST_ASSERT_EQUAL_INT(1, iteration_count[i], 
                             "Iteration was not executed exactly once");
    }
    
    return TEST_PASSED;
}

// Test parallel sections
test_result_t test_parallel_sections(void) {
    int num_threads = 4;
    int section_executed[4] = {0}; // Track which sections were executed
    
    // Set the number of threads
    omp_set_num_threads(num_threads);
    
    // Parallel sections
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            section_executed[0] = 1;
        }
        
        #pragma omp section
        {
            section_executed[1] = 1;
        }
        
        #pragma omp section
        {
            section_executed[2] = 1;
        }
        
        #pragma omp section
        {
            section_executed[3] = 1;
        }
    }
    
    // Verify each section was executed
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT(section_executed[i], "Section was not executed");
    }
    
    return TEST_PASSED;
}

// Test parallel for with reduction
test_result_t test_reduction(void) {
    int num_threads = 4;
    int num_iterations = 100;
    int sum = 0;
    int expected_sum = 0;
    
    // Calculate expected sum
    for (int i = 0; i < num_iterations; i++) {
        expected_sum += i;
    }
    
    // Set the number of threads
    omp_set_num_threads(num_threads);
    
    // Parallel for with reduction
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < num_iterations; i++) {
        sum += i;
    }
    
    // Verify the reduction worked correctly
    TEST_ASSERT_EQUAL_INT(expected_sum, sum, "Reduction produced incorrect result");
    
    return TEST_PASSED;
}

// Test static scheduling
test_result_t test_static_scheduling(void) {
    int num_threads = 4;
    int num_iterations = 100;
    int chunk_size = 10;
    int thread_iterations[16][100] = {0}; // Track which thread executed which iteration
    
    // Set the number of threads
    omp_set_num_threads(num_threads);
    
    // Parallel for with static scheduling
    #pragma omp parallel for schedule(static, chunk_size)
    for (int i = 0; i < num_iterations; i++) {
        int tid = omp_get_thread_num();
        if (tid < 16) {
            thread_iterations[tid][i] = 1;
        }
    }
    
    // Verify each iteration was executed by exactly one thread
    for (int i = 0; i < num_iterations; i++) {
        int count = 0;
        for (int t = 0; t < 16; t++) {
            count += thread_iterations[t][i];
        }
        TEST_ASSERT_EQUAL_INT(1, count, "Iteration was not executed exactly once");
    }
    
    // Verify static scheduling (each thread should have contiguous chunks)
    for (int t = 0; t < num_threads; t++) {
        int in_chunk = 0;
        int chunk_start = -1;
        
        for (int i = 0; i < num_iterations; i++) {
            if (thread_iterations[t][i] && !in_chunk) {
                in_chunk = 1;
                chunk_start = i;
            } else if (!thread_iterations[t][i] && in_chunk) {
                in_chunk = 0;
                int chunk_end = i - 1;
                TEST_ASSERT_EQUAL_INT(chunk_size, chunk_end - chunk_start + 1, 
                                     "Static scheduling chunk size is incorrect");
            }
        }
    }
    
    return TEST_PASSED;
}

// Main test function
int main(int argc, char *argv[]) {
    // Define test cases
    test_case_t tests[] = {
        {"parallel_for", test_parallel_for, "Test parallel for loop"},
        {"parallel_sections", test_parallel_sections, "Test parallel sections"},
        {"reduction", test_reduction, "Test parallel for with reduction"},
        {"static_scheduling", test_static_scheduling, "Test static scheduling"}
    };
    
    // Run the test suite
    run_test_suite(tests, sizeof(tests) / sizeof(tests[0]));
    
    return tests_failed > 0 ? 1 : 0;
}
