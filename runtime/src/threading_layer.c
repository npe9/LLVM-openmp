/*
 * threading_layer.c - Implementation of the threading portability layer interface
 */

#include "threading_layer.h"
#include <stdio.h>
#include <stdlib.h>

/*
 * Initialize the threading subsystem based on the selected backend
 * (pthreads or Lithe)
 */
int __kmp_threading_layer_init(void) {
    int result = parlib_threading_init();
    
    if (result != 0) {
        fprintf(stderr, "Error initializing threading portability layer: %d\n", result);
        return result;
    }
    
#ifdef PARLIB_USE_PTHREADS
    printf("Initialized threading portability layer with pthreads backend\n");
#elif defined(PARLIB_USE_LITHE)
    printf("Initialized threading portability layer with Lithe backend\n");
#else
    printf("Initialized threading portability layer with unknown backend\n");
#endif
    
    return 0;
}

/*
 * Finalize the threading subsystem
 */
void __kmp_threading_layer_fini(void) {
    parlib_threading_fini();
}

/*
 * Get the name of the current threading backend
 */
const char* __kmp_threading_get_backend_name(void) {
#ifdef PARLIB_USE_PTHREADS
    return "pthreads";
#elif defined(PARLIB_USE_LITHE)
    return "lithe";
#else
    return "unknown";
#endif
}

/*
 * Check if the threading layer is using Lithe
 */
int __kmp_threading_using_lithe(void) {
#ifdef PARLIB_USE_LITHE
    return 1;
#else
    return 0;
#endif
}

/*
 * Check if the threading layer is using pthreads
 */
int __kmp_threading_using_pthreads(void) {
#ifdef PARLIB_USE_PTHREADS
    return 1;
#else
    return 0;
#endif
}
