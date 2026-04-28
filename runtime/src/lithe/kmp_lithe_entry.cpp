/*
 * kmp_lithe_entry.cpp - Entry points for OpenMP runtime integration with Lithe
 * When using Lithe, z_Linux_asm.S is not built, so we provide __kmp_hardware_timestamp here.
 */

#include "kmp_lithe.h"
#include "kmp_i18n.h"
#include "kmp_io.h"
#include "kmp_os.h"
#include "kmp.h"  // for kmp_cpuid_t, __kmp_x86_cpuid declaration

// Global Lithe scheduler for OpenMP
static kmp_lithe_scheduler_t __kmp_lithe_scheduler;

// The unnamed critical section - replaces the assembly definition in z_Linux_asm.S
// This is the actual critical section lock
static kmp_critical_name __kmp_gomp_critical_user = {0};

// Global pointer to the unnamed critical section - referenced by kmp_gsupport.cpp
extern "C" kmp_critical_name *__kmp_unnamed_critical_addr = &__kmp_gomp_critical_user;

// Flag to indicate if Lithe integration is initialized
static int __kmp_lithe_initialized = 0;

// Entry point for microtask invocation via Lithe
// This is the function that will be called by the assembly stub
extern "C" int
__kmp_invoke_microtask(microtask_t pkfn, int gtid, int npr, int argc, void *p_argv[], void **exit_frame) {
    // Call the microtask function
    (*pkfn)(&gtid, &npr, p_argv[0], p_argv[1], p_argv[2], p_argv[3]);
    return 1;
}

// Initialize the OpenMP runtime with Lithe integration
extern "C" void
__kmp_lithe_initialize(void) {
    if (__kmp_lithe_initialized) {
        return;
    }
    
    // Initialize Lithe runtime
    __kmp_lithe_runtime_initialize();
    
    // Initialize the Lithe scheduler for OpenMP
    __kmp_lithe_scheduler_init(&__kmp_lithe_scheduler, __kmp_threads[0]);
    
    // Enter the Lithe scheduler
    lithe_sched_enter((lithe_sched_t *)&__kmp_lithe_scheduler);
    
    __kmp_lithe_initialized = 1;
    
    // // // // KMP_INFORM(LitheInitialized, "KMP_LITHE");
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
    
    // // // // KMP_INFORM(LitheFinalized, "KMP_LITHE");
}

// Create worker threads using Lithe
extern "C" int
__kmp_lithe_fork_call(int argc, microtask_t microtask, int gtid, void *wrapper_argv[]) {
    int nthreads = __kmp_threads[gtid]->th.th_team->t.t_nproc;
    
    // // // // KMP_INFORM(LitheForkCall, "KMP_LITHE", nthreads);
    
    // Request harts from Lithe
    int granted_harts = __kmp_lithe_request_harts(&__kmp_lithe_scheduler, nthreads - 1);
    
    // Create worker threads using Lithe
    for (int i = 1; i < nthreads; i++) {
        if (!__kmp_lithe_create_worker(&__kmp_lithe_scheduler, __kmp_threads[gtid]->th.th_team->t.t_threads[i]->th.th_info.ds.ds_gtid)) {
            // // // // KMP_WARNING(CantCreateWorkerThread);
        }
    }
    
    // Execute the microtask on the master thread
    (*microtask)(&gtid, &gtid, argc, wrapper_argv);
    
    return 1;
}

// Join worker threads using Lithe
extern "C" void
__kmp_lithe_join_call(int gtid) {
    // // // // KMP_INFORM(LitheJoinCall, "KMP_LITHE", gtid);
    
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

// Replaces symbols from z_Linux_asm.S when Lithe is used (asm file not built)
#if defined(__x86_64__) || defined(__i386__)
extern "C" kmp_uint64 __kmp_hardware_timestamp(void) {
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return (kmp_uint64)lo | ((kmp_uint64)hi << 32);
}

extern "C" void __kmp_x86_cpuid(int mode, int mode2, kmp_cpuid_t *p) {
    unsigned eax = (unsigned)mode, ebx, ecx = (unsigned)mode2, edx;
    __asm__ __volatile__ ("cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(eax), "c"(ecx));
    p->eax = eax;
    p->ebx = ebx;
    p->ecx = ecx;
    p->edx = edx;
}

extern "C" void __kmp_load_x87_fpu_control_word(kmp_int16 *p) {
    __asm__ __volatile__ ("fldcw %0" : : "m"(*p));
}
extern "C" void __kmp_store_x87_fpu_control_word(kmp_int16 *p) {
    __asm__ __volatile__ ("fstcw %0" : "=m"(*p));
}
extern "C" void __kmp_clear_x87_fpu_status_word(void) {
    __asm__ __volatile__ ("fnclex");
}
#endif
