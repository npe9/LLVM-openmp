/*
 * OpenMP Threading Portability Layer Example
 * 
 * This example demonstrates how to use the threading portability layer
 * with OpenMP. It shows a simple parallel region that uses the underlying
 * threading implementation (either pthreads or Lithe).
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <parlib/threading/thread.h>
#include <parlib/threading/sync.h>

/* Global counter and mutex for thread-safe updates */
int counter = 0;
parlib_mutex_t counter_mutex;

/* Function to increment counter in a thread-safe manner */
void increment_counter(int amount) {
    parlib_mutex_lock(&counter_mutex);
    counter += amount;
    parlib_mutex_unlock(&counter_mutex);
}

int main(int argc, char *argv[]) {
    int num_threads = 4;
    int i;
    
    /* Initialize the mutex */
    parlib_mutex_init(&counter_mutex, NULL);
    
    /* Print threading backend information */
    printf("Using threading backend: %s\n", 
#ifdef PARLIB_USE_PTHREADS
           "pthreads"
#elif defined(PARLIB_USE_LITHE)
           "lithe"
#else
           "unknown"
#endif
    );
    
    /* Set number of threads if provided as argument */
    if (argc > 1) {
        num_threads = atoi(argv[1]);
        if (num_threads <= 0) {
            num_threads = 4;
        }
    }
    
    /* Set the number of OpenMP threads */
    omp_set_num_threads(num_threads);
    
    printf("Starting OpenMP parallel region with %d threads\n", num_threads);
    
    /* OpenMP parallel region */
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int thread_count = omp_get_num_threads();
        
        printf("Thread %d/%d: Hello from OpenMP thread\n", thread_id, thread_count);
        
        /* Each thread increments the counter by its thread ID + 1 */
        increment_counter(thread_id + 1);
        
        /* Do some work */
        #pragma omp for
        for (i = 0; i < 100; i++) {
            /* Just a simple computation to keep the thread busy */
            double result = 0.0;
            int j;
            for (j = 0; j < 10000; j++) {
                result += j * (i + 1) * 0.001;
            }
            
            /* Occasionally print progress */
            if (i % 25 == 0) {
                printf("Thread %d processing iteration %d, result = %f\n", 
                       thread_id, i, result);
            }
        }
    }
    
    /* Print final counter value */
    printf("Final counter value: %d\n", counter);
    
    /* Calculate expected counter value: sum of (thread_id + 1) for all threads */
    int expected = 0;
    for (i = 0; i < num_threads; i++) {
        expected += (i + 1);
    }
    printf("Expected counter value: %d\n", expected);
    
    /* Clean up */
    parlib_mutex_destroy(&counter_mutex);
    
    return (counter == expected) ? 0 : 1;
}
