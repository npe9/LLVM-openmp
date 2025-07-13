/*
 * kmp_lithe.h - OpenMP runtime integration with Lithe threading library
 */

#ifndef KMP_LITHE_H
#define KMP_LITHE_H

#include "kmp_os.h"
#include "kmp_lock.h"
#include "kmp_i18n.h"
#include "kmp_io.h"

#ifdef __cplusplus
extern "C" {
#endif
#include <parlib/parlib.h>
#ifdef __cplusplus
}
#endif

#include <lithe/lithe.h>
#include <lithe/sched.h>

#ifdef LIBOMP_USE_LITHE
// Define guard macros to prevent kmp.h from defining these types
#define KMP_MUTEX_ALIGN_T_DEFINED
#define KMP_COND_ALIGN_T_DEFINED
// Underlying types
typedef struct { int locked; } kmp_lithe_mutex_t;
typedef struct { int signaled; } kmp_lithe_cond_t;
// OpenMP expects these wrappers
typedef struct { kmp_lithe_mutex_t m_mutex; } kmp_mutex_align_t;
typedef struct { kmp_lithe_cond_t c_cond; } kmp_cond_align_t;
#define KMP_MUTEX_T           kmp_mutex_align_t
#define KMP_COND_T            kmp_cond_align_t

#define KMP_MUTEX_INIT(m)     __kmp_lithe_mutex_init(&((m)->m_mutex))
#define KMP_MUTEX_LOCK(m)     __kmp_lithe_mutex_lock(&((m)->m_mutex))
#define KMP_MUTEX_UNLOCK(m)   __kmp_lithe_mutex_unlock(&((m)->m_mutex))
#define KMP_MUTEX_DESTROY(m)  __kmp_lithe_mutex_destroy(&((m)->m_mutex))
#define KMP_COND_INIT(c)      __kmp_lithe_cond_init(&((c)->c_cond))
#define KMP_COND_WAIT(c,m)    __kmp_lithe_cond_wait(&((c)->c_cond), &((m)->m_mutex))
#define KMP_COND_TIMEDWAIT(c,m,t) __kmp_lithe_cond_timedwait(&((c)->c_cond), &((m)->m_mutex), t)
#define KMP_COND_SIGNAL(c)    __kmp_lithe_cond_signal(&((c)->c_cond))
#define KMP_COND_BROADCAST(c) __kmp_lithe_cond_broadcast(&((c)->c_cond))
#define KMP_COND_DESTROY(c)   __kmp_lithe_cond_destroy(&((c)->c_cond))
#else
// pthreads backend
#include <pthread.h>
#define KMP_MUTEX_T           pthread_mutex_t
#define KMP_COND_T            pthread_cond_t
#define KMP_MUTEX_INIT(m)     pthread_mutex_init(m, NULL)
#define KMP_MUTEX_LOCK(m)     pthread_mutex_lock(m)
#define KMP_MUTEX_UNLOCK(m)   pthread_mutex_unlock(m)
#define KMP_MUTEX_DESTROY(m)  pthread_mutex_destroy(m)
#define KMP_COND_INIT(c)      pthread_cond_init(c, NULL)
#define KMP_COND_WAIT(c,m)    pthread_cond_wait(c,m)
#define KMP_COND_TIMEDWAIT(c,m,t) pthread_cond_timedwait(c,m,t)
#define KMP_COND_DESTROY(c)   pthread_cond_destroy(c)
#endif

#include "kmp.h"  // For OpenMP types like kmp_info_t, kmp_team_t

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Lithe scheduler implementation for OpenMP
 */
typedef struct kmp_lithe_scheduler {
    lithe_sched_t sched;
    void *root_thread;  // kmp_info_t* - avoid circular dependency
    void *team;         // kmp_team_t* - avoid circular dependency
    int num_workers;
    int requested_harts;
    int granted_harts;
    lithe_context_t **worker_contexts;
} kmp_lithe_scheduler_t;

/* Initialize the Lithe scheduler for OpenMP */
void __kmp_lithe_scheduler_init(kmp_lithe_scheduler_t *scheduler, void *root_thread);

/* Finalize the Lithe scheduler for OpenMP */
void __kmp_lithe_scheduler_finalize(kmp_lithe_scheduler_t *scheduler);

/* Request harts from Lithe for OpenMP parallelism */
int __kmp_lithe_request_harts(kmp_lithe_scheduler_t *scheduler, int num_harts);

/* Create a worker thread using Lithe */
int __kmp_lithe_create_worker(kmp_lithe_scheduler_t *scheduler, int gtid);

/* Yield the current hart back to Lithe */
void __kmp_lithe_yield_hart(kmp_lithe_scheduler_t *scheduler);

/* Initialize the OpenMP runtime to use Lithe */
void __kmp_lithe_runtime_initialize(void);

/* Finalize the OpenMP runtime's use of Lithe */
void __kmp_lithe_runtime_finalize(void);

/* Lithe pthreads compatibility wrappers */
/* These provide the same interface as pthreads but use Lithe/Parlib internally */

/* Thread operations */
int __kmp_lithe_pthread_create(pthread_t *thread, const pthread_attr_t *attr, 
                              void *(*start_routine)(void*), void *arg);
int __kmp_lithe_pthread_join(pthread_t thread, void **retval);
void __kmp_lithe_pthread_exit(void *retval);
int __kmp_lithe_pthread_cancel(pthread_t thread);

/* Lithe mutex/cond function declarations - always visible */
#ifdef LIBOMP_USE_LITHE
int __kmp_lithe_mutex_init(kmp_lithe_mutex_t *mutex);
int __kmp_lithe_mutex_lock(kmp_lithe_mutex_t *mutex);
int __kmp_lithe_mutex_unlock(kmp_lithe_mutex_t *mutex);
int __kmp_lithe_mutex_destroy(kmp_lithe_mutex_t *mutex);
int __kmp_lithe_cond_init(kmp_lithe_cond_t *cond);
int __kmp_lithe_cond_wait(kmp_lithe_cond_t *cond, kmp_lithe_mutex_t *mutex);
int __kmp_lithe_cond_timedwait(kmp_lithe_cond_t *cond, kmp_lithe_mutex_t *mutex, const struct timespec *abstime);
int __kmp_lithe_cond_signal(kmp_lithe_cond_t *cond);
int __kmp_lithe_cond_broadcast(kmp_lithe_cond_t *cond);
int __kmp_lithe_cond_destroy(kmp_lithe_cond_t *cond);
#endif

/* Condition variable operations */
int __kmp_lithe_pthread_cond_init(pthread_cond_t *cond, const pthread_condattr_t *attr);
int __kmp_lithe_pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex);
int __kmp_lithe_pthread_cond_timedwait(pthread_cond_t *cond, pthread_mutex_t *mutex, 
                                      const struct timespec *abstime);
int __kmp_lithe_pthread_cond_destroy(pthread_cond_t *cond);

/* Thread-specific data */
int __kmp_lithe_pthread_setspecific(pthread_key_t key, const void *value);
void *__kmp_lithe_pthread_getspecific(pthread_key_t key);

/* Signal handling */
int __kmp_lithe_pthread_sigmask(int how, const sigset_t *set, sigset_t *oldset);

/* Thread attributes */
int __kmp_lithe_pthread_attr_init(pthread_attr_t *attr);
int __kmp_lithe_pthread_attr_destroy(pthread_attr_t *attr);
int __kmp_lithe_pthread_attr_setstacksize(pthread_attr_t *attr, size_t stacksize);

/* Missing function declarations */
pthread_t __kmp_lithe_pthread_self(void);
int __kmp_lithe_pthread_mutexattr_init(pthread_mutexattr_t *attr);
int __kmp_lithe_pthread_mutexattr_destroy(pthread_mutexattr_t *attr);
int __kmp_lithe_pthread_condattr_init(pthread_condattr_t *attr);
int __kmp_lithe_pthread_condattr_destroy(pthread_condattr_t *attr);

#ifdef __cplusplus
}
#endif

#endif /* KMP_LITHE_H */
