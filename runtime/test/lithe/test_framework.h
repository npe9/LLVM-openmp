/*
 * test_framework.h - Simple test framework for Lithe-OpenMP integration tests
 */

#ifndef TEST_FRAMEWORK_H
#define TEST_FRAMEWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// Test result status
typedef enum {
    TEST_PASSED,
    TEST_FAILED,
    TEST_SKIPPED
} test_result_t;

// Test function type
typedef test_result_t (*test_func_t)(void);

// Test case structure
typedef struct {
    const char *name;
    test_func_t func;
    const char *description;
} test_case_t;

// Global test statistics
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;
static int tests_skipped = 0;

// Assertion macros
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            printf("ASSERTION FAILED: %s\n", message); \
            printf("  at %s:%d\n", __FILE__, __LINE__); \
            return TEST_FAILED; \
        } \
    } while (0)

#define TEST_ASSERT_EQUAL_INT(expected, actual, message) \
    do { \
        if ((expected) != (actual)) { \
            printf("ASSERTION FAILED: %s\n", message); \
            printf("  Expected: %d\n", expected); \
            printf("  Actual: %d\n", actual); \
            printf("  at %s:%d\n", __FILE__, __LINE__); \
            return TEST_FAILED; \
        } \
    } while (0)

// Run a single test
static void run_test(const test_case_t *test) {
    printf("Running test: %s\n", test->name);
    printf("  Description: %s\n", test->description);
    
    tests_run++;
    test_result_t result = test->func();
    
    switch (result) {
        case TEST_PASSED:
            printf("  Result: PASSED\n\n");
            tests_passed++;
            break;
        case TEST_FAILED:
            printf("  Result: FAILED\n\n");
            tests_failed++;
            break;
        case TEST_SKIPPED:
            printf("  Result: SKIPPED\n\n");
            tests_skipped++;
            break;
    }
}

// Run all tests in a test suite
static void run_test_suite(const test_case_t tests[], int num_tests) {
    printf("=== Starting Lithe-OpenMP Integration Test Suite ===\n\n");
    
    for (int i = 0; i < num_tests; i++) {
        run_test(&tests[i]);
    }
    
    printf("=== Test Suite Summary ===\n");
    printf("Total tests: %d\n", tests_run);
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    printf("Skipped: %d\n", tests_skipped);
    printf("==========================\n");
}

#endif /* TEST_FRAMEWORK_H */
