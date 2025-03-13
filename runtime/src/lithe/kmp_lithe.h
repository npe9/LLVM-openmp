/*
 * kmp_lithe.h - OpenMP runtime integration with Lithe threading library
 */

#ifndef KMP_LITHE_H
#define KMP_LITHE_H

#include "kmp.h"
#include "lithe/lithe.h"
#include "lithe/sched.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Lithe scheduler implementation for OpenMP
 */
typedef struct kmp_lithe_scheduler {
    lithe_sched_t sched;
    kmp_info_t *root_thread;
    kmp_team_t *team;
    int num_workers;
    int requested_harts;
    int granted_harts;
    lithe_context_t **worker_contexts;
} kmp_lithe_scheduler_t;

/* Initialize the Lithe scheduler for OpenMP */
void __kmp_lithe_scheduler_init(kmp_lithe_scheduler_t *scheduler, kmp_info_t *root_thread);

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

#ifdef __cplusplus
}
#endif

#endif /* KMP_LITHE_H */
