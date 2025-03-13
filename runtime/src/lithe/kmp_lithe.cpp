/*
 * kmp_lithe.cpp - OpenMP runtime integration with Lithe threading library
 */

#include "kmp_lithe.h"
#include "kmp_i18n.h"
#include "kmp_io.h"

// Lithe scheduler function implementations
static void lithe_hart_request(lithe_sched_t *__this, lithe_sched_t *child, int h);
static void lithe_hart_enter(lithe_sched_t *__this);
static void lithe_hart_return(lithe_sched_t *__this, lithe_sched_t *child);
static void lithe_sched_enter(lithe_sched_t *__this);
static void lithe_sched_exit(lithe_sched_t *__this);
static void lithe_child_enter(lithe_sched_t *__this, lithe_sched_t *child);
static void lithe_child_exit(lithe_sched_t *__this, lithe_sched_t *child);
static void lithe_context_block(lithe_sched_t *__this, lithe_context_t *context);
static void lithe_context_unblock(lithe_sched_t *__this, lithe_context_t *context);
static void lithe_context_yield(lithe_sched_t *__this, lithe_context_t *context);
static void lithe_context_exit(lithe_sched_t *__this, lithe_context_t *context);

// Lithe scheduler function table
static const lithe_sched_funcs_t kmp_lithe_sched_funcs = {
    lithe_hart_request,
    lithe_hart_enter,
    lithe_hart_return,
    lithe_sched_enter,
    lithe_sched_exit,
    lithe_child_enter,
    lithe_child_exit,
    lithe_context_block,
    lithe_context_unblock,
    lithe_context_yield,
    lithe_context_exit
};

// Worker thread entry point
static void kmp_lithe_worker_entry(void *arg) {
    int gtid = (int)(size_t)arg;
    kmp_info_t *thread = __kmp_threads[gtid];
    
    KMP_INFORM(LitheWorkerStarted, "KMP_LITHE", gtid);
    
    // Enter the OpenMP worker loop
    __kmp_launch_worker(gtid);
    
    KMP_INFORM(LitheWorkerFinished, "KMP_LITHE", gtid);
}

// Initialize the Lithe scheduler for OpenMP
void __kmp_lithe_scheduler_init(kmp_lithe_scheduler_t *scheduler, kmp_info_t *root_thread) {
    // Initialize the scheduler structure
    scheduler->sched.funcs = &kmp_lithe_sched_funcs;
    scheduler->root_thread = root_thread;
    scheduler->team = NULL;
    scheduler->num_workers = 0;
    scheduler->requested_harts = 0;
    scheduler->granted_harts = 0;
    scheduler->worker_contexts = NULL;
    
    // Allocate and initialize the main context
    scheduler->sched.main_context = (lithe_context_t *)malloc(sizeof(lithe_context_t));
    if (!scheduler->sched.main_context) {
        KMP_FATAL(MemoryAllocFailed);
        return;
    }
    
    // Initialize the main context with the current thread's context
    lithe_context_init(scheduler->sched.main_context, NULL, NULL);
    
    KMP_INFORM(LitheSchedulerInitialized, "KMP_LITHE");
}

// Finalize the Lithe scheduler for OpenMP
void __kmp_lithe_scheduler_finalize(kmp_lithe_scheduler_t *scheduler) {
    // Clean up worker contexts
    if (scheduler->worker_contexts) {
        for (int i = 0; i < scheduler->num_workers; i++) {
            if (scheduler->worker_contexts[i]) {
                lithe_context_cleanup(scheduler->worker_contexts[i]);
                free(scheduler->worker_contexts[i]);
            }
        }
        free(scheduler->worker_contexts);
        scheduler->worker_contexts = NULL;
    }
    
    // Clean up main context
    if (scheduler->sched.main_context) {
        lithe_context_cleanup(scheduler->sched.main_context);
        free(scheduler->sched.main_context);
        scheduler->sched.main_context = NULL;
    }
    
    scheduler->num_workers = 0;
    scheduler->requested_harts = 0;
    scheduler->granted_harts = 0;
    
    KMP_INFORM(LitheSchedulerFinalized, "KMP_LITHE");
}

// Request harts from Lithe for OpenMP parallelism
int __kmp_lithe_request_harts(kmp_lithe_scheduler_t *scheduler, int num_harts) {
    scheduler->requested_harts = num_harts;
    
    // Request harts from Lithe
    lithe_hart_request(num_harts);
    
    // Return the number of harts granted (this will be updated by hart_enter)
    return scheduler->granted_harts;
}

// Create a worker thread using Lithe
int __kmp_lithe_create_worker(kmp_lithe_scheduler_t *scheduler, int gtid) {
    // Allocate context for the worker
    lithe_context_t *context = (lithe_context_t *)malloc(sizeof(lithe_context_t));
    if (!context) {
        KMP_FATAL(MemoryAllocFailed);
        return 0;
    }
    
    // Allocate stack for the context
    void *stack = malloc(KMP_DEFAULT_STACK_SIZE);
    if (!stack) {
        free(context);
        KMP_FATAL(MemoryAllocFailed);
        return 0;
    }
    
    // Initialize the context
    context->stack = stack;
    context->stack_size = KMP_DEFAULT_STACK_SIZE;
    lithe_context_init(context, kmp_lithe_worker_entry, (void*)(size_t)gtid);
    
    // Add the context to our list of workers
    if (!scheduler->worker_contexts) {
        scheduler->worker_contexts = (lithe_context_t **)malloc(sizeof(lithe_context_t *) * __kmp_xproc);
        if (!scheduler->worker_contexts) {
            lithe_context_cleanup(context);
            free(stack);
            free(context);
            KMP_FATAL(MemoryAllocFailed);
            return 0;
        }
        for (int i = 0; i < __kmp_xproc; i++) {
            scheduler->worker_contexts[i] = NULL;
        }
    }
    
    scheduler->worker_contexts[scheduler->num_workers] = context;
    scheduler->num_workers++;
    
    return 1;
}

// Yield the current hart back to Lithe
void __kmp_lithe_yield_hart(kmp_lithe_scheduler_t *scheduler) {
    lithe_hart_yield();
}

// Initialize the OpenMP runtime to use Lithe
void __kmp_lithe_runtime_initialize(void) {
    KMP_INFORM(LitheRuntimeInitializing, "KMP_LITHE");
    
    // Ensure Lithe is initialized
    lithe_lib_init();
    
    KMP_INFORM(LitheRuntimeInitialized, "KMP_LITHE");
}

// Finalize the OpenMP runtime's use of Lithe
void __kmp_lithe_runtime_finalize(void) {
    KMP_INFORM(LitheRuntimeFinalized, "KMP_LITHE");
}

// Lithe scheduler function implementations
static void lithe_hart_request(lithe_sched_t *__this, lithe_sched_t *child, int h) {
    kmp_lithe_scheduler_t *scheduler = (kmp_lithe_scheduler_t *)__this;
    
    KMP_INFORM(LitheHartRequest, "KMP_LITHE", h);
    
    // Handle hart requests from child schedulers
    // For now, we don't support nested parallelism with Lithe
    KMP_INFORM(LitheNestedParallelismNotSupported, "KMP_LITHE");
}

static void lithe_hart_enter(lithe_sched_t *__this) {
    kmp_lithe_scheduler_t *scheduler = (kmp_lithe_scheduler_t *)__this;
    
    // Increment the number of granted harts
    scheduler->granted_harts++;
    
    KMP_INFORM(LitheHartEnter, "KMP_LITHE", scheduler->granted_harts);
    
    // If we have worker contexts available, start one
    if (scheduler->num_workers > 0 && scheduler->worker_contexts) {
        int worker_idx = scheduler->granted_harts - 1;
        if (worker_idx < scheduler->num_workers && scheduler->worker_contexts[worker_idx]) {
            lithe_context_t *context = scheduler->worker_contexts[worker_idx];
            lithe_context_run(context);
        }
    }
}

static void lithe_hart_return(lithe_sched_t *__this, lithe_sched_t *child) {
    kmp_lithe_scheduler_t *scheduler = (kmp_lithe_scheduler_t *)__this;
    
    KMP_INFORM(LitheHartReturn, "KMP_LITHE");
    
    // Handle hart returns from child schedulers
    // For now, we don't support nested parallelism with Lithe
}

static void lithe_sched_enter(lithe_sched_t *__this) {
    kmp_lithe_scheduler_t *scheduler = (kmp_lithe_scheduler_t *)__this;
    
    KMP_INFORM(LitheSchedEnter, "KMP_LITHE");
    
    // This is called when our scheduler is entered
    // The root thread is already running, so we don't need to do anything here
}

static void lithe_sched_exit(lithe_sched_t *__this) {
    kmp_lithe_scheduler_t *scheduler = (kmp_lithe_scheduler_t *)__this;
    
    KMP_INFORM(LitheSchedExit, "KMP_LITHE");
    
    // This is called when our scheduler is exiting
    // We should clean up any remaining worker threads
}

static void lithe_child_enter(lithe_sched_t *__this, lithe_sched_t *child) {
    kmp_lithe_scheduler_t *scheduler = (kmp_lithe_scheduler_t *)__this;
    
    KMP_INFORM(LitheChildEnter, "KMP_LITHE");
    
    // This is called when a child scheduler is entered
    // For now, we don't support nested parallelism with Lithe
}

static void lithe_child_exit(lithe_sched_t *__this, lithe_sched_t *child) {
    kmp_lithe_scheduler_t *scheduler = (kmp_lithe_scheduler_t *)__this;
    
    KMP_INFORM(LitheChildExit, "KMP_LITHE");
    
    // This is called when a child scheduler exits
    // For now, we don't support nested parallelism with Lithe
}

static void lithe_context_block(lithe_sched_t *__this, lithe_context_t *context) {
    kmp_lithe_scheduler_t *scheduler = (kmp_lithe_scheduler_t *)__this;
    
    KMP_INFORM(LitheContextBlock, "KMP_LITHE");
    
    // This is called when a context is blocked
    // We should try to schedule another worker if available
}

static void lithe_context_unblock(lithe_sched_t *__this, lithe_context_t *context) {
    kmp_lithe_scheduler_t *scheduler = (kmp_lithe_scheduler_t *)__this;
    
    KMP_INFORM(LitheContextUnblock, "KMP_LITHE");
    
    // This is called when a context is unblocked
    // We should add the context back to our runnable queue
}

static void lithe_context_yield(lithe_sched_t *__this, lithe_context_t *context) {
    kmp_lithe_scheduler_t *scheduler = (kmp_lithe_scheduler_t *)__this;
    
    KMP_INFORM(LitheContextYield, "KMP_LITHE");
    
    // This is called when a context yields
    // We should try to schedule another worker if available
}

static void lithe_context_exit(lithe_sched_t *__this, lithe_context_t *context) {
    kmp_lithe_scheduler_t *scheduler = (kmp_lithe_scheduler_t *)__this;
    
    KMP_INFORM(LitheContextExit, "KMP_LITHE");
    
    // This is called when a context exits
    // We should clean up the context and potentially schedule another worker
}
