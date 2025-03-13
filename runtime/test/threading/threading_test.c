/*
 * threading_test.c - Test program for the threading portability layer
 */

#include <parlib/threading/thread.h>
#include <parlib/threading/sync.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NUM_THREADS 4
#define NUM_ITERATIONS 1000000

/* Shared counter and mutex for synchronization */
static int counter = 0;
static parlib_mutex_t counter_mutex;

/* Barrier for thread synchronization */
static parlib_barrier_t barrier;

/* Thread function that increments the counter */
void *increment_counter(void *arg) {
    int thread_id = *((int *)arg);
    int local_counter = 0;
    
    printf("Thread %d started\n", thread_id);
    
    /* Wait for all threads to start */
    parlib_barrier_wait(&barrier);
    
    /* Increment the counter NUM_ITERATIONS times */
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        parlib_mutex_lock(&counter_mutex);
        counter++;
        parlib_mutex_unlock(&counter_mutex);
        local_counter++;
    }
    
    printf("Thread %d finished, local_counter = %d\n", thread_id, local_counter);
    
    return NULL;
}

int main(int argc, char *argv[]) {
    parlib_thread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];
    int result;
    
    /* Initialize the threading subsystem */
    result = parlib_threading_init();
    if (result != PARLIB_SUCCESS) {
        fprintf(stderr, "Failed to initialize threading subsystem: %d\n", result);
        return 1;
    }
    
    printf("Threading subsystem initialized\n");
    printf("Using %s backend\n", 
#ifdef PARLIB_USE_PTHREADS
           "pthreads"
#else
           "lithe"
#endif
          );
    
    /* Initialize mutex */
    result = parlib_mutex_init(&counter_mutex, NULL);
    if (result != PARLIB_SUCCESS) {
        fprintf(stderr, "Failed to initialize mutex: %d\n", result);
        parlib_threading_fini();
        return 1;
    }
    
    /* Initialize barrier */
    result = parlib_barrier_init(&barrier, NUM_THREADS + 1);
    if (result != PARLIB_SUCCESS) {
        fprintf(stderr, "Failed to initialize barrier: %d\n", result);
        parlib_mutex_destroy(&counter_mutex);
        parlib_threading_fini();
        return 1;
    }
    
    /* Create threads */
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i;
        result = parlib_thread_create(&threads[i], NULL, increment_counter, &thread_ids[i]);
        if (result != PARLIB_SUCCESS) {
            fprintf(stderr, "Failed to create thread %d: %d\n", i, result);
            parlib_mutex_destroy(&counter_mutex);
            parlib_barrier_destroy(&barrier);
            parlib_threading_fini();
            return 1;
        }
    }
    
    /* Wait for all threads to start */
    parlib_barrier_wait(&barrier);
    
    /* Wait for threads to complete */
    for (int i = 0; i < NUM_THREADS; i++) {
        result = parlib_thread_join(threads[i], NULL);
        if (result != PARLIB_SUCCESS) {
            fprintf(stderr, "Failed to join thread %d: %d\n", i, result);
        }
    }
    
    /* Check the final counter value */
    printf("Final counter value: %d (expected: %d)\n", 
           counter, NUM_THREADS * NUM_ITERATIONS);
    
    /* Clean up */
    parlib_mutex_destroy(&counter_mutex);
    parlib_barrier_destroy(&barrier);
    parlib_threading_fini();
    
    return 0;
}
