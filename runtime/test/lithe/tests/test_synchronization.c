/*
 * test_synchronization.c - Test OpenMP synchronization primitives with Lithe
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../test_framework.h"

// Test critical section
test_result_t test_critical_section(void) {
    int num_threads = 4;
    int counter = 0;
    
    // Set the number of threads
    omp_set_num_threads(num_threads);
    
    // Parallel region with critical section
    #pragma omp parallel
    {
        for (int i = 0; i < 1000; i++) {
            #pragma omp critical
            {
                counter++;
            }
        }
    }
    
    // Verify the counter has the expected value
    TEST_ASSERT_EQUAL_INT(num_threads * 1000, counter, 
                         "Critical section failed to protect shared variable");
    
    return TEST_PASSED;
}

// Test barrier synchronization
test_result_t test_barrier(void) {
    int num_threads = 4;
    int phase1_complete[16] = {0};
    int phase2_complete[16] = {0};
    int all_phase1_complete = 0;
    
    // Set the number of threads
    omp_set_num_threads(num_threads);
    
    // Parallel region with barrier
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        // Phase 1
        #pragma omp critical
        {
            phase1_complete[tid] = 1;
        }
        
        // Barrier to ensure all threads complete phase 1 before any start phase 2
        #pragma omp barrier
        
        // Check if all threads completed phase 1
        #pragma omp single
        {
            all_phase1_complete = 1;
            for (int i = 0; i < num_threads; i++) {
                if (!phase1_complete[i]) {
                    all_phase1_complete = 0;
                    break;
                }
            }
        }
        
        // Phase 2
        #pragma omp critical
        {
            phase2_complete[tid] = 1;
        }
    }
    
    // Verify all threads completed phase 1 before any started phase 2
    TEST_ASSERT(all_phase1_complete, "Barrier failed to synchronize threads");
    
    // Verify all threads completed phase 2
    for (int i = 0; i < num_threads; i++) {
        TEST_ASSERT(phase2_complete[i], "Thread did not complete phase 2");
    }
    
    return TEST_PASSED;
}

// Test atomic operations
test_result_t test_atomic(void) {
    int num_threads = 4;
    int counter = 0;
    
    // Set the number of threads
    omp_set_num_threads(num_threads);
    
    // Parallel region with atomic operations
    #pragma omp parallel
    {
        for (int i = 0; i < 1000; i++) {
            #pragma omp atomic
            counter++;
        }
    }
    
    // Verify the counter has the expected value
    TEST_ASSERT_EQUAL_INT(num_threads * 1000, counter, 
                         "Atomic operations failed to protect shared variable");
    
    return TEST_PASSED;
}

// Test locks
test_result_t test_locks(void) {
    int num_threads = 4;
    int counter = 0;
    omp_lock_t lock;
    
    // Initialize the lock
    omp_init_lock(&lock);
    
    // Set the number of threads
    omp_set_num_threads(num_threads);
    
    // Parallel region with locks
    #pragma omp parallel
    {
        for (int i = 0; i < 1000; i++) {
            omp_set_lock(&lock);
            counter++;
            omp_unset_lock(&lock);
        }
    }
    
    // Destroy the lock
    omp_destroy_lock(&lock);
    
    // Verify the counter has the expected value
    TEST_ASSERT_EQUAL_INT(num_threads * 1000, counter, 
                         "Locks failed to protect shared variable");
    
    return TEST_PASSED;
}

// Test unnamed critical section (uses ___kmp_unnamed_critical_addr)
test_result_t test_unnamed_critical(void) {
    int num_threads = 4;
    int counter = 0;
    
    // Set the number of threads
    omp_set_num_threads(num_threads);
    
    // Parallel region with unnamed critical section
    #pragma omp parallel
    {
        for (int i = 0; i < 1000; i++) {
            #pragma omp critical
            {
                counter++;
            }
        }
    }
    
    // Verify the counter has the expected value
    TEST_ASSERT_EQUAL_INT(num_threads * 1000, counter, 
                         "Unnamed critical section failed to protect shared variable");
    
    return TEST_PASSED;
}

// Main test function
int main(int argc, char *argv[]) {
    // Define test cases
    test_case_t tests[] = {
        {"critical_section", test_critical_section, "Test critical section"},
        {"barrier", test_barrier, "Test barrier synchronization"},
        {"atomic", test_atomic, "Test atomic operations"},
        {"locks", test_locks, "Test locks"},
        {"unnamed_critical", test_unnamed_critical, "Test unnamed critical section"}
    };
    
    // Run the test suite
    run_test_suite(tests, sizeof(tests) / sizeof(tests[0]));
    
    return tests_failed > 0 ? 1 : 0;
}
