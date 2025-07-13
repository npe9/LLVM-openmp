/*
 * kmp_lithe.cpp - OpenMP runtime integration with Lithe threading library
 */

#ifdef __cplusplus
#define typeof __typeof__
#endif
#include "kmp_lithe.h"
#include "kmp_i18n.h"
#include "kmp_io.h"

// Lithe scheduler function implementations
static void lithe_hart_request(lithe_sched_t *__this, lithe_sched_t *child, int h);
static void lithe_hart_enter(lithe_sched_t *__this);
static void lithe_hart_return(lithe_sched_t *__this, lithe_sched_t *child);
void lithe_sched_enter(lithe_sched_t *__this);
void lithe_sched_exit(lithe_sched_t *__this);
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
    __kmp_launch_thread(thread);
    
    KMP_INFORM(LitheWorkerFinished, "KMP_LITHE", gtid);
}

// Initialize the Lithe scheduler for OpenMP
void __kmp_lithe_scheduler_init(kmp_lithe_scheduler_t *scheduler, void *root_thread) {
    kmp_info_t *thread = (kmp_info_t *)root_thread;
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
    lithe_hart_request(&scheduler->sched, NULL, num_harts);
    
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
    void *stack = malloc(KMP_DEFAULT_STKSIZE);
    if (!stack) {
        free(context);
        KMP_FATAL(MemoryAllocFailed);
        return 0;
    }
    
    // Initialize the context
    context->stack.bottom = stack;
    context->stack.size = KMP_DEFAULT_STKSIZE;
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
void lithe_hart_request(lithe_sched_t *__this, lithe_sched_t *child, int h) {
    kmp_lithe_scheduler_t *scheduler = (kmp_lithe_scheduler_t *)__this;
    
    KMP_INFORM(LitheHartRequest, "KMP_LITHE", h);
    
    // Handle hart requests from child schedulers
    // For now, we don't support nested parallelism with Lithe
    KMP_INFORM(LitheNestedParallelismNotSupported, "KMP_LITHE");
}

void lithe_hart_enter(lithe_sched_t *__this) {
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

void lithe_hart_return(lithe_sched_t *__this, lithe_sched_t *child) {
    kmp_lithe_scheduler_t *scheduler = (kmp_lithe_scheduler_t *)__this;
    
    KMP_INFORM(LitheHartReturn, "KMP_LITHE");
    
    // Handle hart returns from child schedulers
    // For now, we don't support nested parallelism with Lithe
}

void lithe_sched_enter(lithe_sched_t *__this) {
    kmp_lithe_scheduler_t *scheduler = (kmp_lithe_scheduler_t *)__this;
    
    KMP_INFORM(LitheSchedEnter, "KMP_LITHE");
    
    // This is called when our scheduler is entered
    // The root thread is already running, so we don't need to do anything here
}

void lithe_sched_exit(lithe_sched_t *__this) {
    kmp_lithe_scheduler_t *scheduler = (kmp_lithe_scheduler_t *)__this;
    
    KMP_INFORM(LitheSchedExit, "KMP_LITHE");
    
    // This is called when our scheduler is exiting
    // We should clean up any remaining worker threads
}

void lithe_child_enter(lithe_sched_t *__this, lithe_sched_t *child) {
    kmp_lithe_scheduler_t *scheduler = (kmp_lithe_scheduler_t *)__this;
    
    KMP_INFORM(LitheChildEnter, "KMP_LITHE");
    
    // This is called when a child scheduler is entered
    // For now, we don't support nested parallelism with Lithe
}

void lithe_child_exit(lithe_sched_t *__this, lithe_sched_t *child) {
    kmp_lithe_scheduler_t *scheduler = (kmp_lithe_scheduler_t *)__this;
    
    KMP_INFORM(LitheChildExit, "KMP_LITHE");
    
    // This is called when a child scheduler exits
    // For now, we don't support nested parallelism with Lithe
}

void lithe_context_block(lithe_sched_t *__this, lithe_context_t *context) {
    kmp_lithe_scheduler_t *scheduler = (kmp_lithe_scheduler_t *)__this;
    
    KMP_INFORM(LitheContextBlock, "KMP_LITHE");
    
    // This is called when a context is blocked
    // We should try to schedule another worker if available
}

void lithe_context_unblock(lithe_sched_t *__this, lithe_context_t *context) {
    kmp_lithe_scheduler_t *scheduler = (kmp_lithe_scheduler_t *)__this;
    
    KMP_INFORM(LitheContextUnblock, "KMP_LITHE");
    
    // This is called when a context is unblocked
    // We should add the context back to our runnable queue
}

void lithe_context_yield(lithe_sched_t *__this, lithe_context_t *context) {
    kmp_lithe_scheduler_t *scheduler = (kmp_lithe_scheduler_t *)__this;
    
    KMP_INFORM(LitheContextYield, "KMP_LITHE");
    
    // This is called when a context yields
    // We should try to schedule another worker if available
}

void lithe_context_exit(lithe_sched_t *__this, lithe_context_t *context) {
    kmp_lithe_scheduler_t *scheduler = (kmp_lithe_scheduler_t *)__this;
    
    KMP_INFORM(LitheContextExit, "KMP_LITHE");
    
    // This is called when a context exits
    // We should clean up the context and potentially schedule another worker
}

// Lithe wrapper functions for pthreads compatibility
// These provide the same interface as pthreads but use Lithe/Parlib internally

// Thread wrapper structure for pthread compatibility
typedef struct {
    void *(*start_routine)(void*);
    void *arg;
    int exited;
    void *exit_value;
    int cancelled;
} thread_wrapper_t;

// Thread wrapper function for lithe
static void lithe_thread_wrapper(void *arg) {
    thread_wrapper_t *wrapper = (thread_wrapper_t*)arg;
    if (!wrapper->cancelled) {
        wrapper->exit_value = wrapper->start_routine(wrapper->arg);
    }
    wrapper->exited = 1;
}

// Thread creation wrapper
int __kmp_lithe_pthread_create(pthread_t *thread, const pthread_attr_t *attr, 
                              void *(*start_routine)(void*), void *arg) {
    // Allocate context and wrapper
    lithe_context_t *context = (lithe_context_t *)malloc(sizeof(lithe_context_t));
    thread_wrapper_t *wrapper = (thread_wrapper_t *)malloc(sizeof(thread_wrapper_t));
    
    if (!context || !wrapper) {
        free(context);
        free(wrapper);
        return ENOMEM;
    }
    
    // Initialize wrapper
    wrapper->start_routine = start_routine;
    wrapper->arg = arg;
    wrapper->exited = 0;
    wrapper->exit_value = NULL;
    wrapper->cancelled = 0;
    
    // Initialize context
    lithe_context_init(context, lithe_thread_wrapper, wrapper);
    
    // Store context as thread ID
    *thread = (pthread_t)context;
    
    return 0;
}

// Thread joining wrapper
int __kmp_lithe_pthread_join(pthread_t thread, void **retval) {
    lithe_context_t *context = (lithe_context_t *)thread;
    
    // Get the wrapper from the context
    thread_wrapper_t *wrapper = (thread_wrapper_t *)context->start_func_arg;
    
    // Wait for the context to complete
    while (!wrapper->exited) {
        lithe_hart_yield();
    }
    
    if (retval) {
        *retval = wrapper->exit_value;
    }
    
    // Clean up
    lithe_context_cleanup(context);
    free(wrapper);
    free(context);
    
    return 0;
}

// Thread exit wrapper
void __kmp_lithe_pthread_exit(void *retval) {
    // Get current context and mark as exited
    lithe_context_t *context = lithe_context_self();
    if (context) {
        thread_wrapper_t *wrapper = (thread_wrapper_t *)context->start_func_arg;
        if (wrapper) {
            wrapper->exited = 1;
            wrapper->exit_value = retval;
        }
    }
    
    // Exit the context
    lithe_context_exit();
}

// Thread cancellation wrapper
int __kmp_lithe_pthread_cancel(pthread_t thread) {
    lithe_context_t *context = (lithe_context_t *)thread;
    
    // Get the wrapper and mark for cancellation
    thread_wrapper_t *wrapper = (thread_wrapper_t *)context->start_func_arg;
    if (wrapper) {
        wrapper->cancelled = 1;
    }
    
    return 0;
}

// For lithe mode, we need to translate pthread calls to lithe equivalents
// We'll use a simple approach: store lithe objects in a global table and use pthread handles as indices

static kmp_lithe_mutex_t *lithe_mutex_table[1024] = {NULL};
static kmp_lithe_cond_t *lithe_cond_table[1024] = {NULL};
static int next_mutex_id = 0;
static int next_cond_id = 0;

// Mutex operations using Lithe/Parlib synchronization
int __kmp_lithe_pthread_mutex_init(pthread_mutex_t *mutex, const pthread_mutexattr_t *attr) {
    kmp_lithe_mutex_t *lithe_mutex = (kmp_lithe_mutex_t *)malloc(sizeof(kmp_lithe_mutex_t));
    if (!lithe_mutex) {
        return ENOMEM;
    }
    lithe_mutex->locked = 0;
    
    // Store in table and use index as handle
    int id = next_mutex_id++;
    if (id >= 1024) {
        free(lithe_mutex);
        return ENOMEM;
    }
    lithe_mutex_table[id] = lithe_mutex;
    
    // Store the ID in the mutex struct - use the first field
    memset(mutex, 0, sizeof(pthread_mutex_t));
    mutex->__data.__lock = id;
    return 0;
}

int __kmp_lithe_pthread_mutex_lock(pthread_mutex_t *mutex) {
    int id = mutex->__data.__lock;
    if (id < 0 || id >= 1024 || !lithe_mutex_table[id]) {
        return EINVAL;
    }
    kmp_lithe_mutex_t *lithe_mutex = lithe_mutex_table[id];
    while (__sync_lock_test_and_set(&lithe_mutex->locked, 1) != 0) {
        lithe_hart_yield();
    }
    return 0;
}

int __kmp_lithe_pthread_mutex_unlock(pthread_mutex_t *mutex) {
    int id = mutex->__data.__lock;
    if (id < 0 || id >= 1024 || !lithe_mutex_table[id]) {
        return EINVAL;
    }
    kmp_lithe_mutex_t *lithe_mutex = lithe_mutex_table[id];
    __sync_lock_release(&lithe_mutex->locked);
    return 0;
}

int __kmp_lithe_pthread_mutex_destroy(pthread_mutex_t *mutex) {
    int id = mutex->__data.__lock;
    if (id < 0 || id >= 1024 || !lithe_mutex_table[id]) {
        return EINVAL;
    }
    free(lithe_mutex_table[id]);
    lithe_mutex_table[id] = NULL;
    return 0;
}

// Condition variable operations
int __kmp_lithe_pthread_cond_init(pthread_cond_t *cond, const pthread_condattr_t *attr) {
    kmp_lithe_cond_t *lithe_cond = (kmp_lithe_cond_t *)malloc(sizeof(kmp_lithe_cond_t));
    if (!lithe_cond) {
        return ENOMEM;
    }
    lithe_cond->signaled = 0;
    
    // Store in table and use index as handle
    int id = next_cond_id++;
    if (id >= 1024) {
        free(lithe_cond);
        return ENOMEM;
    }
    lithe_cond_table[id] = lithe_cond;
    
    // Store the ID in the cond struct - use a portable approach
    memset(cond, 0, sizeof(pthread_cond_t));
    // Use the first field of the union as a simple integer
    *(int*)cond = id;
    return 0;
}

int __kmp_lithe_pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex) {
    int id = *(int*)cond;  // Extract ID from the first field
    if (id < 0 || id >= 1024 || !lithe_cond_table[id]) {
        return EINVAL;
    }
    kmp_lithe_cond_t *lithe_cond = lithe_cond_table[id];
    // Simple implementation - yield and retry
    __kmp_lithe_pthread_mutex_unlock(mutex);
    while (!lithe_cond->signaled) {
        lithe_hart_yield();
    }
    lithe_cond->signaled = 0;
    __kmp_lithe_pthread_mutex_lock(mutex);
    return 0;
}

int __kmp_lithe_pthread_cond_timedwait(pthread_cond_t *cond, pthread_mutex_t *mutex, 
                                      const struct timespec *abstime) {
    int id = *(int*)cond;  // Extract ID from the first field
    if (id < 0 || id >= 1024 || !lithe_cond_table[id]) {
        return EINVAL;
    }
    kmp_lithe_cond_t *lithe_cond = lithe_cond_table[id];
    // Simple implementation - yield and retry
    __kmp_lithe_pthread_mutex_unlock(mutex);
    while (!lithe_cond->signaled) {
        lithe_hart_yield();
    }
    lithe_cond->signaled = 0;
    __kmp_lithe_pthread_mutex_lock(mutex);
    return 0;
}

int __kmp_lithe_pthread_cond_destroy(pthread_cond_t *cond) {
    int id = *(int*)cond;  // Extract ID from the first field
    if (id < 0 || id >= 1024 || !lithe_cond_table[id]) {
        return EINVAL;
    }
    free(lithe_cond_table[id]);
    lithe_cond_table[id] = NULL;
    return 0;
}

// Thread-specific data
int __kmp_lithe_pthread_setspecific(pthread_key_t key, const void *value) {
    // Use Lithe/Parlib thread-local storage
    // For now, use a simple global array
    return 0;
}

void *__kmp_lithe_pthread_getspecific(pthread_key_t key) {
    // Use Lithe/Parlib thread-local storage
    // For now, return NULL
    return NULL;
}

// Signal handling
int __kmp_lithe_pthread_sigmask(int how, const sigset_t *set, sigset_t *oldset) {
    // Use Parlib signal handling or implement with Lithe
    // For now, just return success
    return 0;
}

// Thread attributes
int __kmp_lithe_pthread_attr_init(pthread_attr_t *attr) {
    // Initialize thread attributes
    return 0;
}

int __kmp_lithe_pthread_attr_destroy(pthread_attr_t *attr) {
    // Clean up thread attributes
    return 0;
}

int __kmp_lithe_pthread_attr_setstacksize(pthread_attr_t *attr, size_t stacksize) {
    // Set stack size attribute
    return 0;
}

// Get current thread ID
pthread_t __kmp_lithe_pthread_self(void) {
    lithe_context_t *context = lithe_context_self();
    return (pthread_t)context;
}

// Attribute functions
int __kmp_lithe_pthread_mutexattr_init(pthread_mutexattr_t *attr) {
    // Simple implementation - just return success
    return 0;
}

int __kmp_lithe_pthread_mutexattr_destroy(pthread_mutexattr_t *attr) {
    // Simple implementation - just return success
    return 0;
}

int __kmp_lithe_pthread_condattr_init(pthread_condattr_t *attr) {
    // Simple implementation - just return success
    return 0;
}

int __kmp_lithe_pthread_condattr_destroy(pthread_condattr_t *attr) {
    // Simple implementation - just return success
    return 0;
}

#ifdef LIBOMP_USE_LITHE
// Mutex wrappers for lithe mode
int __kmp_lithe_mutex_init(kmp_lithe_mutex_t *mutex) {
    mutex->locked = 0;
    return 0;
}
int __kmp_lithe_mutex_lock(kmp_lithe_mutex_t *mutex) {
    while (__sync_lock_test_and_set(&mutex->locked, 1) != 0) {
        lithe_hart_yield();
    }
    return 0;
}
int __kmp_lithe_mutex_unlock(kmp_lithe_mutex_t *mutex) {
    __sync_lock_release(&mutex->locked);
    return 0;
}
int __kmp_lithe_mutex_destroy(kmp_lithe_mutex_t *mutex) {
    // Nothing to do for this simple implementation
    return 0;
}
// Condition variable wrappers for lithe mode
int __kmp_lithe_cond_init(kmp_lithe_cond_t *cond) {
    cond->signaled = 0;
    return 0;
}
int __kmp_lithe_cond_wait(kmp_lithe_cond_t *cond, kmp_lithe_mutex_t *mutex) {
    __kmp_lithe_mutex_unlock(mutex);
    while (!cond->signaled) {
        lithe_hart_yield();
    }
    cond->signaled = 0;
    __kmp_lithe_mutex_lock(mutex);
    return 0;
}
int __kmp_lithe_cond_timedwait(kmp_lithe_cond_t *cond, kmp_lithe_mutex_t *mutex, const struct timespec *abstime) {
    __kmp_lithe_mutex_unlock(mutex);
    while (!cond->signaled) {
        lithe_hart_yield();
    }
    cond->signaled = 0;
    __kmp_lithe_mutex_lock(mutex);
    return 0;
}
int __kmp_lithe_cond_signal(kmp_lithe_cond_t *cond) {
    cond->signaled = 1;
    return 0;
}
int __kmp_lithe_cond_broadcast(kmp_lithe_cond_t *cond) {
    cond->signaled = 1;
    return 0;
}
int __kmp_lithe_cond_destroy(kmp_lithe_cond_t *cond) {
    // Nothing to do for this simple implementation
    return 0;
}
#endif
