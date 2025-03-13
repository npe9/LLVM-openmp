/*
 * lithe_test.c - Test program for OpenMP with Lithe integration
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    int num_threads = 4;
    int i;
    
    printf("Testing OpenMP with Lithe integration\n");
    
    // Set the number of threads
    omp_set_num_threads(num_threads);
    
    // Parallel region
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        
        printf("Thread %d of %d: Hello, world!\n", tid, nthreads);
        
        // Barrier to ensure all threads print before continuing
        #pragma omp barrier
        
        if (tid == 0) {
            printf("Master thread: All threads have reported in\n");
        }
    }
    
    // Parallel for loop
    printf("\nParallel for loop test:\n");
    #pragma omp parallel for
    for (i = 0; i < num_threads; i++) {
        int tid = omp_get_thread_num();
        printf("Thread %d processing iteration %d\n", tid, i);
    }
    
    printf("\nTest completed successfully\n");
    
    return 0;
}
