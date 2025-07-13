/*
 * kmp_lithe_entry.cpp - Entry points for OpenMP runtime integration with Lithe
 */

#ifdef __cplusplus
#define typeof __typeof__
#endif
#include "kmp_lithe.h"
#include "kmp.h"  // For OpenMP types and declarations
#include "kmp_i18n.h"
#include "kmp_io.h"

// Global Lithe scheduler for OpenMP
static kmp_lithe_scheduler_t __kmp_lithe_scheduler;

// Flag to indicate if Lithe integration is initialized
static int __kmp_lithe_initialized = 0;

// Initialize the OpenMP runtime with Lithe integration
extern "C" void
__kmp_lithe_initialize(void) {
    if (__kmp_lithe_initialized) {
        return;
    }
    
    // At this point, parallel initialization is complete and __kmp_threads[0] is guaranteed to be valid
    KMP_ASSERT(__kmp_threads != NULL);
    KMP_ASSERT(__kmp_threads[0] != NULL);
    
    // Initialize Lithe runtime
    __kmp_lithe_runtime_initialize();
    
    // Initialize the Lithe scheduler for OpenMP
    __kmp_lithe_scheduler_init(&__kmp_lithe_scheduler, (void*)__kmp_threads[0]);
    
    // Enter the Lithe scheduler
    lithe_sched_enter(&__kmp_lithe_scheduler.sched);
    
    __kmp_lithe_initialized = 1;
    
    KMP_INFORM(LitheInitialized, "KMP_LITHE");
}

// Finalize the OpenMP runtime with Lithe integration
extern "C" void
__kmp_lithe_finalize(void) {
    if (!__kmp_lithe_initialized) {
        return;
    }
    
    // Exit the Lithe scheduler
    lithe_sched_exit();
    
    // Finalize the Lithe scheduler for OpenMP
    __kmp_lithe_scheduler_finalize(&__kmp_lithe_scheduler);
    
    // Finalize Lithe runtime
    __kmp_lithe_runtime_finalize();
    
    __kmp_lithe_initialized = 0;
    
    KMP_INFORM(LitheFinalized, "KMP_LITHE");
}

// Create worker threads using Lithe
extern "C" int
__kmp_lithe_fork_call(int argc, microtask_t microtask, int gtid, void *wrapper_argv[]) {
    // Ensure lithe is initialized (should already be done by now)
    KMP_ASSERT(__kmp_lithe_initialized);
    
    kmp_info_t *master_th = __kmp_threads[gtid];
    kmp_team_t *team = master_th->th.th_team;
    int nthreads = team->t.t_nproc;
    
    KMP_INFORM(LitheForkCall, "KMP_LITHE", nthreads);
    
    // Request harts from Lithe
    int granted_harts = __kmp_lithe_request_harts(&__kmp_lithe_scheduler, nthreads - 1);
    
    // Create worker threads using Lithe
    for (int i = 1; i < nthreads; i++) {
        if (!__kmp_lithe_create_worker(&__kmp_lithe_scheduler, team->t.t_threads[i]->th.th_info.ds.ds_gtid)) {
            KMP_WARNING(CantCreateWorkerThread);
        }
    }
    
    // Invoke the microtask for the master thread
    KMP_ASSERT(wrapper_argv != NULL);
    return __kmp_invoke_microtask(microtask, gtid, 0, argc, wrapper_argv, nullptr);
}

// Join worker threads using Lithe
extern "C" void
__kmp_lithe_join_call(int gtid) {
    KMP_INFORM(LitheJoinCall, "KMP_LITHE", gtid);
    
    // Wait for all worker threads to complete
    // This is handled by the OpenMP runtime's barrier mechanism
    
    // Yield harts back to Lithe
    if (gtid == 0) {
        __kmp_lithe_yield_hart(&__kmp_lithe_scheduler);
    }
}

// Lithe-specific implementation of unnamed critical section
extern "C" kmp_critical_name *
__kmp_lithe_get_unnamed_critical_addr(void) {
    // This is referenced by the assembly code
    static kmp_critical_name __kmp_unnamed_critical_addr;
    return &__kmp_unnamed_critical_addr;
}

