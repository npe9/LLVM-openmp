/*
 * threading_layer.h - Interface for the threading portability layer in OpenMP
 */

#ifndef THREADING_LAYER_H
#define THREADING_LAYER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <parlib/threading/thread.h>
#include <parlib/threading/sync.h>

/*
 * This header provides a unified interface for the OpenMP runtime to use
 * the threading portability layer. It abstracts away the differences between
 * the pthreads and Lithe backends, allowing the OpenMP runtime to use either
 * implementation transparently.
 */

/* Thread management functions */
typedef parlib_thread_t kmp_thread_t;
typedef parlib_thread_attr_t kmp_thread_attr_t;
typedef parlib_thread_func_t kmp_thread_func_t;
typedef parlib_thread_id_t kmp_thread_id_t;

#define kmp_thread_create(thread, attr, func, arg) \
    parlib_thread_create(thread, attr, func, arg)

#define kmp_thread_join(thread, retval) \
    parlib_thread_join(thread, retval)

#define kmp_thread_detach(thread) \
    parlib_thread_detach(thread)

#define kmp_thread_self() \
    parlib_thread_self()

#define kmp_thread_equal(t1, t2) \
    parlib_thread_equal(t1, t2)

#define kmp_thread_yield() \
    parlib_thread_yield()

#define kmp_thread_exit(retval) \
    parlib_thread_exit(retval)

#define kmp_thread_get_num_processors() \
    parlib_thread_get_num_processors()

/* Mutex functions */
typedef parlib_mutex_t kmp_mutex_t;
typedef parlib_mutex_attr_t kmp_mutex_attr_t;

#define kmp_mutex_init(mutex, attr) \
    parlib_mutex_init(mutex, attr)

#define kmp_mutex_destroy(mutex) \
    parlib_mutex_destroy(mutex)

#define kmp_mutex_lock(mutex) \
    parlib_mutex_lock(mutex)

#define kmp_mutex_trylock(mutex) \
    parlib_mutex_trylock(mutex)

#define kmp_mutex_unlock(mutex) \
    parlib_mutex_unlock(mutex)

/* Read-write lock functions */
typedef parlib_rwlock_t kmp_rwlock_t;

#define kmp_rwlock_init(rwlock) \
    parlib_rwlock_init(rwlock)

#define kmp_rwlock_destroy(rwlock) \
    parlib_rwlock_destroy(rwlock)

#define kmp_rwlock_rdlock(rwlock) \
    parlib_rwlock_rdlock(rwlock)

#define kmp_rwlock_tryrdlock(rwlock) \
    parlib_rwlock_tryrdlock(rwlock)

#define kmp_rwlock_wrlock(rwlock) \
    parlib_rwlock_wrlock(rwlock)

#define kmp_rwlock_trywrlock(rwlock) \
    parlib_rwlock_trywrlock(rwlock)

#define kmp_rwlock_unlock(rwlock) \
    parlib_rwlock_unlock(rwlock)

/* Condition variable functions */
typedef parlib_cond_t kmp_cond_t;

#define kmp_cond_init(cond) \
    parlib_cond_init(cond)

#define kmp_cond_destroy(cond) \
    parlib_cond_destroy(cond)

#define kmp_cond_wait(cond, mutex) \
    parlib_cond_wait(cond, mutex)

#define kmp_cond_timedwait(cond, mutex, abstime) \
    parlib_cond_timedwait(cond, mutex, abstime)

#define kmp_cond_signal(cond) \
    parlib_cond_signal(cond)

#define kmp_cond_broadcast(cond) \
    parlib_cond_broadcast(cond)

/* Semaphore functions */
typedef parlib_sem_t kmp_sem_t;

#define kmp_sem_init(sem, value) \
    parlib_sem_init(sem, value)

#define kmp_sem_destroy(sem) \
    parlib_sem_destroy(sem)

#define kmp_sem_post(sem) \
    parlib_sem_post(sem)

#define kmp_sem_wait(sem) \
    parlib_sem_wait(sem)

#define kmp_sem_trywait(sem) \
    parlib_sem_trywait(sem)

#define kmp_sem_getvalue(sem, value) \
    parlib_sem_getvalue(sem, value)

/* Barrier functions */
typedef parlib_barrier_t kmp_barrier_t;

#define kmp_barrier_init(barrier, count) \
    parlib_barrier_init(barrier, count)

#define kmp_barrier_destroy(barrier) \
    parlib_barrier_destroy(barrier)

#define kmp_barrier_wait(barrier) \
    parlib_barrier_wait(barrier)

/* Once control functions */
typedef parlib_once_t kmp_once_t;

#define KMP_ONCE_INIT PARLIB_ONCE_INIT

#define kmp_once(once_control, init_routine) \
    parlib_once(once_control, init_routine)

/* Threading subsystem initialization/finalization */
#define kmp_threading_init() \
    parlib_threading_init()

#define kmp_threading_fini() \
    parlib_threading_fini()

#ifdef __cplusplus
}
#endif

#endif /* THREADING_LAYER_H */
