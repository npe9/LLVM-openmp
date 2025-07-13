/*
 * kmp_lithe_pthread_wrappers.h - Pthreads function wrappers for Lithe backend
 */

#ifndef KMP_LITHE_PTHREAD_WRAPPERS_H
#define KMP_LITHE_PTHREAD_WRAPPERS_H

#ifdef LIBOMP_USE_LITHE

// When using Lithe, redirect pthreads calls to Lithe wrappers
#define pthread_create __kmp_lithe_pthread_create
#define pthread_join __kmp_lithe_pthread_join
#define pthread_exit __kmp_lithe_pthread_exit
#define pthread_cancel __kmp_lithe_pthread_cancel

#define pthread_mutex_init __kmp_lithe_pthread_mutex_init
#define pthread_mutex_lock __kmp_lithe_pthread_mutex_lock
#define pthread_mutex_unlock __kmp_lithe_pthread_mutex_unlock
#define pthread_mutex_destroy __kmp_lithe_pthread_mutex_destroy

#define pthread_cond_init __kmp_lithe_pthread_cond_init
#define pthread_cond_wait __kmp_lithe_pthread_cond_wait
#define pthread_cond_timedwait __kmp_lithe_pthread_cond_timedwait
#define pthread_cond_destroy __kmp_lithe_pthread_cond_destroy

#define pthread_setspecific __kmp_lithe_pthread_setspecific
#define pthread_getspecific __kmp_lithe_pthread_getspecific

#define pthread_sigmask __kmp_lithe_pthread_sigmask

#define pthread_attr_init __kmp_lithe_pthread_attr_init
#define pthread_attr_destroy __kmp_lithe_pthread_attr_destroy
#define pthread_attr_setstacksize __kmp_lithe_pthread_attr_setstacksize

#endif /* LIBOMP_USE_LITHE */

#endif /* KMP_LITHE_PTHREAD_WRAPPERS_H */ 