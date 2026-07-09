/*
 * z_Linux_util.cpp -- platform specific routines.
 */

//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//

#include "kmp.h"
#include "kmp_affinity.h"
#include "kmp_i18n.h"
#include "kmp_io.h"
#include "kmp_itt.h"
#include "kmp_lock.h"
#include "kmp_stats.h"
#include "kmp_str.h"
#include "kmp_wait_release.h"
#include "kmp_wrapper_getpid.h"

#ifdef LIBOMP_USE_LITHE
#ifdef __cplusplus
#define typeof __typeof__
#endif
#include <lithe/lithe.h>
#include <lithe/fork_join_sched.h>
#include <lithe/mutex.h>
#include <lithe/condvar.h>
extern "C" lithe_fork_join_sched_t *pmix_lithe_get_fork_join_sched(void)
    __attribute__((weak));

/* ------------------------------------------------------------------------
 * Team-local inner-barrier spin policy (see kmp_wait_release.h and
 * kmp.h th_lithe_barrier_inner).
 *
 * The OpenMP barrier b_go/b_arrived sense bits are already team-local atomics.
 * Under Lithe, the cost that made hybrid miniFE anti-scale was not the flag
 * itself but what a waiter did once it spun briefly: it cooperatively BLOCKED
 * on a lithe condvar, which calls lithe_context_block -> lithe_hart_request(-1)
 * -> parlib vcore park (futex_wait_allocated), and the releaser had to
 * lithe_condvar_signal -> lithe_context_unblock -> hart_request(+1) ->
 * futex_wakeup. Across CG's thousands of inner barriers this PER-BARRIER global
 * scheduler interaction (hart accounting + FJS queue locks + vcore park/wake)
 * dominated and grew with thread count.
 *
 * Fix: at an INNER (intra-region) barrier, hold the hart and hot-spin on the
 * team-local flag -- wake becomes a plain atomic store the spinner observes,
 * exactly like vanilla. Only FORK/JOIN (region-boundary) barriers cooperatively
 * release the hart, so MPI/PMIx/UCX progress and other schedulers still get
 * vcores at coarse granularity (between parallel regions / MPI phases).
 *
 * Rule compliance: this uses only C/C++ atomics + KMP_CPU_PAUSE (cpu_relax) at
 * the OpenMP layer -- no pthread, no sem, no raw futex, no sched_yield above
 * parlib (lithified-runtimes-no-linux-primitives.mdc). It does NOT hoard harts
 * program-wide (fork/join still releases), so it is not the banned
 * KMP_BLOCKTIME=infinite behavior.
 * ------------------------------------------------------------------------ */
/* INNER (intra-region plain/reduction) barriers: hot-spin holding the hart.
 *   LITHE_OMP_INNER_SPIN     (default 1)  : enable.
 *   LITHE_OMP_INNER_SPIN_MAX (default 0)  : 0 = unbounded (spin until released);
 *                                           >0 = cap pause iters then fall to
 *                                           cooperative block (safety valve).
 * FORK/JOIN (region-boundary) barriers: bounded hot-spin holding the hart to
 * bridge rapid re-fork between per-operation parallel regions (miniFE re-forks
 * per dot/waxpby/matvec, so EVERY CG barrier is a fork/join barrier kept "hot"
 * by hot-teams), then fall through to the cooperative lithe-condvar block so a
 * genuinely long idle gap (MPI exchange / serial setup) RELEASES the hart for
 * sibling schedulers (MPI/PMIx/UCX progress). This is the coarse-grained
 * composition point.
 *   LITHE_OMP_FORK_SPIN_MAX  (default 0):
 *       0  = disabled (always cooperative-block immediately; old behavior, and
 *            the correct default: fork/join is the coarse composition point
 *            where harts MUST be released to MPI/PMIx/UCX progress -- spinning
 *            here regressed miniFE ~3x and starves MPI at NP>1).
 *      >0  = bounded pause iters then cooperative block (experimental).
 *      <0  = unbounded spin (NP=1 only; would starve MPI progress at NP>1).
 *      >0  = pause-iteration budget before cooperative release.
 * The default is bounded (not infinite) so it is NOT the banned
 * KMP_BLOCKTIME=infinite hart-hoarding: harts are still returned on long gaps.
 */
int __kmp_lithe_inner_spin = 1;            /* LITHE_OMP_INNER_SPIN      */
kmp_uint64 __kmp_lithe_inner_spin_max = 0; /* LITHE_OMP_INNER_SPIN_MAX  */
/* Inner-barrier hart-donation (liveness). DEFAULT ON: while hot-spinning on the
 * team-local inner-barrier flag, a worker donates its hart to any RUNNABLE-but-
 * hartless peer so the barrier cannot livelock when OMP width exceeds the harts/
 * vcores (oversubscription or under-provisioned VCORE_LIMIT). Set
 * LITHE_OMP_INNER_YIELD=0 to reproduce the old (livelocking) unbounded-hoard
 * behavior for A/B experiments only. */
int __kmp_lithe_inner_yield = 1;           /* LITHE_OMP_INNER_YIELD     */
kmp_int64 __kmp_lithe_fork_spin_max = 0; /* LITHE_OMP_FORK_SPIN_MAX (manual override) */
/* Hot-team (persistent team kept across regions) fork/join hot-spin. This is
 * the DEFAULT path for miniFE-style codes that re-fork a parallel region per
 * operation: instead of cooperatively blocking (parlib vcore park) at every
 * region boundary, the worker holds its hart and hot-spins on the team-local
 * go-flag, so the next region's fork wakes it with a plain memory write (like
 * vanilla) -- no per-region parlib park/wake, no global hart re-accounting,
 * and (critically) no per-region lithe_sched_enter/exit drain on the master
 * (see the sticky top fork-join sched below). Gated so composition is
 * preserved: only spins when a hart is reserved beyond the team for MPI/PMIx/
 * UCX progress (max_vcores() >= nproc + reserve), and only for a BOUNDED
 * adaptive budget so a genuinely long idle gap falls through to the cooperative
 * block that returns the hart. */
/* DEFAULT OFF for hybrid (measured): with the sticky persistent team (below)
 * the dominant per-region cost -- the master's lithe_sched_enter/exit hart-drain
 * -- is gone, and at that point cooperatively BLOCKING at the fork/join barrier
 * beats hot-spinning on miniFE (OMP=8: sticky+cooperative 23.5s vs sticky+hot-
 * spin 29.4s). L1-B: when LITHE_OMP_HOT_TEAM is unset and the binary does not
 * link pmix_lithe_get_fork_join_sched (pure OpenMP), __kmp_lithe_fork_hot_ok
 * auto-enables hot-team with reserve 0 — BabelStream OMP=96 triad ~0.99× van.
 * Explicit LITHE_OMP_HOT_TEAM=0/1 always wins; hybrid binaries stay OFF. */
int __kmp_lithe_hot_team = 0;            /* LITHE_OMP_HOT_TEAM (default off for hybrid) */
int __kmp_lithe_hot_team_env_set = 0;    /* 1 if LITHE_OMP_HOT_TEAM was set in env */
int __kmp_lithe_hot_reserve = 1;         /* LITHE_OMP_HOT_RESERVE harts for progress */
int __kmp_lithe_hot_reserve_env_set = 0;
kmp_uint64 __kmp_lithe_hot_budget_init = 200000;  /* LITHE_OMP_HOT_BUDGET initial pause-iters */
kmp_uint64 __kmp_lithe_hot_budget_min = 4096;
kmp_uint64 __kmp_lithe_hot_budget_max = 4000000;
/* WARM VCORES (default OFF -- EXPERIMENTAL, measured NOT beneficial).
 * Design: when a worker cooperatively blocks at a region boundary it parks
 * "warm" -- it returns its hart (composition preserved) but records itself in
 * its vcore's warm slot so the next fork resumes it IN PLACE from the FJS
 * hart_enter, replacing the per-worker condvar + runqueue enqueue/dequeue with a
 * go-flag write + one batched hart_request. See lithe_fork_join_park_warm /
 * _resume_warm.
 *
 * MEASUREMENT (miniFE 128^3, NP=1, node sweep, see HPDC_SUITE.md "WARM VCORES"):
 * removing the runqueue round-trip did NOT reduce CG time -- it is neutral at
 * OMP=1/4/8/32, REGRESSES at OMP=2 (8.3s vs 5.6s sticky), and STALLS at OMP=16
 * (>300s timeout vs 44s sticky). The real per-region cost is the vcore
 * park/wake (parlib futex) + hart-grant tree-walk, which warm does NOT remove
 * (it still does hart_request(-1)/+1 and a full vcore re-entry per worker per
 * region). So warm is left OFF by default. The code is kept (env-gated) for
 * documented future investigation; LITHE_OMP_WARM_VCORES=1 enables it but is
 * known to regress/stall at higher thread counts -- do not enable for
 * production runs. Default path is the proven per-worker lithe condvar. */
int __kmp_lithe_warm_vcores = 0;         /* LITHE_OMP_WARM_VCORES (default OFF) */
extern "C" void lithe_fork_join_park_warm(void *mx);
extern "C" void lithe_fork_join_resume_warm(lithe_context_t *ctx);
static int __kmp_lithe_inner_spin_inited = 0;

extern "C" void __kmp_lithe_inner_spin_config(void) {
  if (__atomic_load_n(&__kmp_lithe_inner_spin_inited, __ATOMIC_ACQUIRE))
    return;
  const char *e = getenv("LITHE_OMP_INNER_SPIN");
  if (e && *e)
    __kmp_lithe_inner_spin = atoi(e);
  e = getenv("LITHE_OMP_INNER_SPIN_MAX");
  if (e && *e)
    __kmp_lithe_inner_spin_max = (kmp_uint64)strtoull(e, NULL, 10);
  e = getenv("LITHE_OMP_FORK_SPIN_MAX");
  if (e && *e)
    __kmp_lithe_fork_spin_max = (kmp_int64)strtoll(e, NULL, 10);
  e = getenv("LITHE_OMP_HOT_TEAM");
  if (e && *e) {
    __kmp_lithe_hot_team = atoi(e);
    __kmp_lithe_hot_team_env_set = 1;
  }
  e = getenv("LITHE_OMP_HOT_RESERVE");
  if (e && *e) {
    __kmp_lithe_hot_reserve = atoi(e);
    __kmp_lithe_hot_reserve_env_set = 1;
  }
  e = getenv("LITHE_OMP_HOT_BUDGET");
  if (e && *e)
    __kmp_lithe_hot_budget_init = (kmp_uint64)strtoull(e, NULL, 10);
  e = getenv("LITHE_OMP_WARM_VCORES");
  if (e && *e)
    __kmp_lithe_warm_vcores = atoi(e);
  e = getenv("LITHE_OMP_INNER_YIELD");
  if (e && *e)
    __kmp_lithe_inner_yield = atoi(e);
  if (__kmp_lithe_hot_budget_init < __kmp_lithe_hot_budget_min)
    __kmp_lithe_hot_budget_min = __kmp_lithe_hot_budget_init;
  if (__kmp_lithe_hot_budget_init > __kmp_lithe_hot_budget_max)
    __kmp_lithe_hot_budget_max = __kmp_lithe_hot_budget_init;
  __atomic_store_n(&__kmp_lithe_inner_spin_inited, 1, __ATOMIC_RELEASE);
}

/* Pure-OMP (no MPI/OPAL host linked): weak pmix_lithe_get_fork_join_sched is
 * NULL. If the symbol exists (hybrid binary), keep hybrid defaults even before
 * MPI_Init so we never auto-enable hot-team for miniFE. Used by L1-B. */
static int __kmp_lithe_is_pure_omp(void) {
  if (pmix_lithe_get_fork_join_sched)
    return 0;
  return 1;
}

/* Gate: may a fork/join-barrier waiter hot-spin (hold its hart) instead of
 * cooperatively blocking? Only when hot-team is enabled and there is at least
 * __kmp_lithe_hot_reserve hart(s) beyond the team's worker count -- that spare
 * hart is what MPI/PMIx/UCX progress (and the master's serial/MPI work) run on
 * while the OMP workers spin, so spinning does NOT starve progress at NP>1.
 * With the launcher's cpus-per-task = VCORE_LIMIT = OMP+2 this is satisfied
 * (2 spare harts). If the allocation is tight (max_vcores() <= nproc), fall
 * back to cooperative blocking.
 *
 * L1-B auto path: when LITHE_OMP_HOT_TEAM is unset and there is no MPI/OPAL
 * host (pure OpenMP), treat hot-team as ON with reserve 0 (no progress hart
 * needed). Explicit LITHE_OMP_HOT_TEAM=0/1 always wins. Hybrid stays default
 * OFF so miniFE is unchanged. */
extern "C" int __kmp_lithe_fork_hot_ok(int nproc) {
  int hot = __kmp_lithe_hot_team;
  int reserve = __kmp_lithe_hot_reserve;
  if (!__kmp_lithe_hot_team_env_set && __kmp_lithe_is_pure_omp()) {
    hot = 1;
    if (!__kmp_lithe_hot_reserve_env_set)
      reserve = 0;
  }
  if (!hot)
    return 0;
  if (nproc < 1)
    nproc = 1;
  return max_vcores() >= nproc + reserve;
}

/* Inner-barrier liveness: a worker that hot-spins on the team-local
 * b_go/b_arrived flag (INNER, i.e. plain/reduction barrier) HOLDS its hart. If
 * the number of OpenMP worker contexts exceeds the available harts/vcores --
 * real oversubscription (e.g. OMP=192 on a 96-core node) or an under-provisioned
 * VCORE_LIMIT (< OMP width) -- the workers that got harts spin forever while the
 * workers WITHOUT harts sit RUNNABLE on the fork-join sched's runqueue and can
 * never reach the barrier. The barrier count never completes: a userspace
 * hart-starvation LIVELOCK (no syscalls, "fewer kernel switches", the process
 * hangs and must be reaped -- exactly the full-node OMP=96/192 symptom).
 *
 * Cure: while spinning, periodically check whether any peer is RUNNABLE but
 * hart-less on the current FJ sched (lithe_fork_join_current_runnable_count() >
 * 0). If so, cooperatively DONATE the hart with lithe_context_yield(): the
 * spinner is re-enqueued and hart_enter dispatches a waiting peer, so every
 * worker eventually gets a hart, reaches the barrier, and the count completes.
 * This keeps the barrier a spin (no condvar/futex, no vcore park/wake) but makes
 * it live under any width. lithe_context_yield stays WITHIN this FJ sched (it
 * re-enters hart_enter, not the parent), and hart_enter's reactor poll is gated
 * on pending fds, so there is no epoll storm for a pure-OMP team.
 *
 * At proper width every worker owns a hart, so runnable_count is 0 during the
 * barrier and this never yields -- the fast atomic-store wake path is unchanged.
 * Master and worker contexts both may yield when runnable_count > 0.
 *
 * Returns 1 if it yielded the hart (caller has been resumed), else 0. */
extern "C" int __kmp_lithe_inner_barrier_yield(kmp_info_t *th) {
  (void)th;
  if (!__kmp_lithe_inner_yield)
    return 0; /* A/B: LITHE_OMP_INNER_YIELD=0 restores the old hoard behavior */
  if (lithe_fork_join_current_runnable_count() <= 0)
    return 0; /* no hart-less peer waiting: keep the hart, hot-spin (fast path) */
  /* Master and workers alike may hold a hart while hot-spinning; when any peer
   * is RUNNABLE-but-hartless (threads > harts), the spinner must yield so the
   * peer can run (hierarchical reduction/release, BabelStream dot() at OMP>cores).
   * At proper width runnable_count==0 so this is a no-op aside from one load. */
  lithe_context_yield(); /* donate hart to a waiting peer; resume here later */
  return 1;
}

/* Per-thread cooperative suspend primitive. A thread that reaches its barrier
 * blocktime parks on this condvar via lithe_condvar_wait (which calls
 * lithe_context_block -> lithe_hart_request(-1), giving the hart back to the
 * parent scheduler so MPI progress or other OpenMP contexts can run). The
 * releasing thread wakes it with lithe_condvar_signal (lithe_context_unblock ->
 * hart_request(1) + reactor wake). This replaces the old busy spin-yield loop,
 * which re-entered the fork-join scheduler (and its epoll netpoll) on every
 * iteration and prevented progress/barriers from overlapping. */
struct kmp_lithe_suspend_ctl {
  lithe_mutex_t mx;
  lithe_condvar_t cv;
};

static struct kmp_lithe_suspend_ctl *
__kmp_lithe_suspend_ctl(kmp_info_t *th) {
  void **slot = &th->th.th_suspend_lithe_ctl;
  struct kmp_lithe_suspend_ctl *p =
      (struct kmp_lithe_suspend_ctl *)__atomic_load_n(slot, __ATOMIC_ACQUIRE);
  if (p)
    return p;
  p = (struct kmp_lithe_suspend_ctl *)malloc(sizeof(*p));
  if (!p) {
    KMP_FATAL(MemoryAllocFailed);
    return NULL;
  }
  lithe_mutex_init(&p->mx, NULL);
  lithe_condvar_init(&p->cv);
  void *expected = NULL;
  if (!__atomic_compare_exchange_n(slot, &expected, (void *)p, false,
                                   __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
    /* Lost the init race; another thread published first. Use theirs. */
    free(p);
    p = (struct kmp_lithe_suspend_ctl *)expected;
  }
  return p;
}

/* Worker reap join: a lithe fork-join worker self-destroys its context when it
 * returns from __kmp_launch_worker (lithe_context_exit ->
 * lithe_fork_join_sched_context_exit -> destroy + join_one). The reaper must
 * therefore NOT destroy the context (double free) -- it must only WAIT until the
 * worker is done touching its kmp_info_t before __kmp_free()ing it, exactly the
 * synchronization pthread_join gives the non-lithe path. The worker raises
 * `exited` under `mx` and signals `cv`; the reaper blocks on `cv`. lithe's
 * mutex/condvar make this lifetime-safe: the reaper re-acquires `mx` (held until
 * the worker's final unlock) before returning, after which the worker no longer
 * touches the ctl, so the reaper can free it. */
struct kmp_lithe_join_ctl {
  lithe_mutex_t mx;
  lithe_condvar_t cv;
  volatile int exited;
};

/* OpenMP parallel regions use a dedicated fork-join scheduler nested under
 * whatever is current (normally OPAL's opal_sched), like libgomp's
 * libgomp_lithe_sched_alloc + lithe_sched_enter in team.c. PMIx progress
 * contexts stay on the host-registered opal_sched from pmix_lithe_get_fork_join_sched. */
enum { __kmp_lithe_fj_stack_max = 32 };
static __thread lithe_fork_join_sched_t *__kmp_lithe_fj_stack[__kmp_lithe_fj_stack_max];
static __thread kmp_team_t *__kmp_lithe_fj_team[__kmp_lithe_fj_stack_max];
static __thread int __kmp_lithe_fj_sp;
/* One fork-join sched for all outermost OpenMP regions on this uthread: workers
 * stay on it across regions so we never destroy its vc_mgmt while contexts
 * still exist (per-region create/destroy + migrate raced RUNNING workers). */
static __thread lithe_fork_join_sched_t *__kmp_lithe_omp_top_fj;
/* Sticky/persistent top-level OMP fork-join sched. The single biggest cost of
 * the old per-region scheme was that the master did lithe_sched_enter(fj) at
 * every parallel-region entry and lithe_sched_exit() at every exit. sched_exit
 * has a hart-DRAIN busy-wait (`while (child->harts) lithe_context_yield()` in
 * lithe.c) and triggers the parent's child_exit (another drain +
 * child_sched_list churn): paid PER REGION. Over miniFE CG's thousands of
 * region boundaries this (plus the idle vcores thrashing in hart_enter waiting
 * to grant/return harts to the constantly-(un)registered child) dominated the
 * profile (~60% of cycles in lithe_fork_join_sched_hart_enter / steal_threads /
 * child_exit; matvec only ~13%). Fix: enter the top fj ONCE and KEEP it entered
 * across top-level regions (the team is "hot"); only nested regions still
 * create/enter/exit per region. __kmp_lithe_region_depth counts ACTIVE parallel
 * regions (begin/leave paired); __kmp_lithe_top_sticky records that index 0 of
 * the stack is the persistent top sched (entered, never exited per region). */
static __thread int __kmp_lithe_region_depth;
static __thread int __kmp_lithe_top_sticky;

/* The fork-join scheduler libomp creates its worker contexts under and nests
 * its per-region OMP schedulers below. When MPI/OPAL is linked it provides this
 * via PMIx (pmix_lithe_get_fork_join_sched) -- OPAL adopts the SAME lithe-owned
 * root, so there is exactly one root regardless of which runtime initialized
 * first. When libomp runs standalone (pure OpenMP, no MPI), there is no PMIx
 * host, so we fall back to the general lithe-owned root fork-join scheduler.
 * lithe_ensure_root_fork_join_sched() bootstraps main->vcore0 + the parlib
 * vcore pool the first time it is called and is idempotent thereafter, so this
 * works for OMP-only, MPI-only, and OMP+MPI without any per-app hack. See the
 * lithe-bootstrap-any-runtime-combination rule. */
static lithe_fork_join_sched_t *__kmp_lithe_opal_sched(void) {
  if (pmix_lithe_get_fork_join_sched) {
    lithe_fork_join_sched_t *s = pmix_lithe_get_fork_join_sched();
    if (s)
      return s;
  }
  /* No MPI/OPAL host present: bootstrap (or reuse) the lithe-owned root. */
  return lithe_ensure_root_fork_join_sched();
}

/* Trigger the general lithe bootstrap before running any OpenMP region (for ALL
 * thread counts, including OMP=1). If a host runtime (MPI/OPAL via PMIx) already
 * registered the root FJ sched, compose onto it -- do not create a second root.
 * Otherwise drive the lithe-owned root bootstrap. Idempotent; cheap after the
 * first call. Called from __kmp_fork_call. */
extern "C" void __kmp_lithe_ensure_bootstrap(void) {
  if (pmix_lithe_get_fork_join_sched && pmix_lithe_get_fork_join_sched())
    return;
  (void)lithe_ensure_root_fork_join_sched();
}

static lithe_fork_join_sched_t *__kmp_fj_from_worker_ctx(lithe_fork_join_context_t *ctx) {
  if (!ctx || !ctx->context.sched)
    return NULL;
  return (lithe_fork_join_sched_t *)ctx->context.sched;
}

static void __kmp_lithe_rebind_ctx(lithe_fork_join_context_t *ctx,
                                   lithe_fork_join_sched_t *to) {
  lithe_fork_join_sched_t *from;
  if (!ctx || !to)
    return;
  from = __kmp_fj_from_worker_ctx(ctx);
  if (!from || from == to)
    return;
  if ((lithe_sched_t *)from != ctx->context.sched)
    return;
  /* Queues + num_contexts: reassociate alone is insufficient (stale TAILQ on
     the old sched; destroying the OMP fj would corrupt runnable workers). */
  lithe_fork_join_context_migrate(from, to, ctx);
}

extern "C" lithe_fork_join_sched_t *__kmp_lithe_worker_sched_for_create(void) {
  if (__kmp_lithe_fj_sp > 0)
    return __kmp_lithe_fj_stack[__kmp_lithe_fj_sp - 1];
  return __kmp_lithe_opal_sched();
}

/* Migrate the team's worker contexts onto `fj` (the sched the master is in /
 * entering), so a hart granted to `fj` can actually run a worker. Workers are
 * CREATED on __kmp_lithe_worker_sched_for_create() (the current top of stack);
 * any that are not already on `fj` (e.g. brand-new workers, or workers left on
 * a parent) are migrated. Returns the number of RUNNABLE contexts migrated so
 * the caller can request that many harts. */
static int __kmp_lithe_migrate_workers(kmp_team_t *team,
                                       lithe_fork_join_sched_t *fj,
                                       lithe_fork_join_sched_t *only_from) {
  int runnable_migrated = 0;
  for (int i = 1; i < team->t.t_nproc; i++) {
    kmp_info_t *th = team->t.t_threads[i];
    if (!th || !th->th.th_lithe_ctx)
      continue;
    lithe_fork_join_context_t *wctx =
        (lithe_fork_join_context_t *)th->th.th_lithe_ctx;
    if (wctx->context.sched == (lithe_sched_t *)fj)
      continue;
    if (only_from && wctx->context.sched != (lithe_sched_t *)only_from)
      continue;
    if (wctx->state == FJS_CTX_RUNNABLE)
      runnable_migrated++;
    __kmp_lithe_rebind_ctx(wctx, fj);
  }
  return runnable_migrated;
}

extern "C" void __kmp_lithe_parallel_sched_begin(kmp_team_t *team) {
  lithe_fork_join_sched_t *fj;
  lithe_fork_join_sched_t *opal;

  if (!team || team->t.t_nproc <= 1)
    return;
  if (!lithe_sched_current())
    return;
  opal = __kmp_lithe_opal_sched();
  if (!opal)
    return;

  int depth = __kmp_lithe_region_depth; /* active regions BEFORE this one */

  if (depth == 0) {
    /* ---- top-level parallel region ---- */
    if (!__kmp_lithe_top_sticky) {
      /* First top-level region of this uthread: create + ENTER the persistent
       * top fork-join sched ONCE and keep it entered (sticky) across all later
       * top-level regions. This is the structural fix: no per-region
       * sched_enter/exit, so no per-region hart-drain busy-wait on the master
       * and no child_sched_list (un)registration churn on the parent. */
      if (__kmp_lithe_fj_sp >= __kmp_lithe_fj_stack_max)
        return;
      if (!__kmp_lithe_omp_top_fj) {
        __kmp_lithe_omp_top_fj = lithe_fork_join_sched_create();
        if (!__kmp_lithe_omp_top_fj)
          return;
      }
      fj = __kmp_lithe_omp_top_fj;
      int runnable_migrated = __kmp_lithe_migrate_workers(team, fj, NULL);
      lithe_sched_enter((lithe_sched_t *)fj);
      __kmp_lithe_fj_team[0] = team;
      __kmp_lithe_fj_stack[0] = fj;
      __kmp_lithe_fj_sp = 1;
      __kmp_lithe_top_sticky = 1;
      __kmp_lithe_region_depth = 1;
      if (runnable_migrated > 0)
        lithe_hart_request(runnable_migrated);
      return;
    }
    /* Reuse the already-entered sticky top sched. Workers from the previous
     * region stay on `fj` (parallel_leave does not rebind at the top), so this
     * is normally a no-op; only brand-new workers (team grew) need migrating.
     * No sched_enter, no hart-drain -- this is what makes the team "hot". */
    fj = __kmp_lithe_fj_stack[0];
    int runnable_migrated = __kmp_lithe_migrate_workers(team, fj, NULL);
    __kmp_lithe_fj_team[0] = team;
    __kmp_lithe_region_depth = 1;
    if (runnable_migrated > 0)
      lithe_hart_request(runnable_migrated);
    return;
  }

  /* ---- nested parallel region (depth >= 1): per-region sched as before ---- */
  if (__kmp_lithe_fj_sp >= __kmp_lithe_fj_stack_max)
    return;
  lithe_fork_join_sched_t *parent_fj = __kmp_lithe_fj_stack[__kmp_lithe_fj_sp - 1];
  fj = lithe_fork_join_sched_create();
  if (!fj)
    return;
  int runnable_migrated = __kmp_lithe_migrate_workers(team, fj, parent_fj);
  lithe_sched_enter((lithe_sched_t *)fj);
  __kmp_lithe_fj_team[__kmp_lithe_fj_sp] = team;
  __kmp_lithe_fj_stack[__kmp_lithe_fj_sp++] = fj;
  __kmp_lithe_region_depth = depth + 1;
  if (runnable_migrated > 0)
    lithe_hart_request(runnable_migrated);
}

extern "C" void __kmp_lithe_parallel_leave(kmp_team_t *team) {
  int i;
  lithe_fork_join_sched_t *fj;
  lithe_fork_join_sched_t *parent_sched;

  if (!team || team->t.t_nproc <= 1)
    return;
  if (__kmp_lithe_region_depth <= 0)
    return;

  int depth = __kmp_lithe_region_depth; /* region we are leaving */

  if (depth == 1) {
    /* Leaving a TOP-LEVEL region: keep the sticky top sched entered (hot). The
     * workers stay associated with it and (by default) hot-spin on the team
     * go-flag waiting for the next region's fork, so there is no per-region
     * park/wake or scheduler re-entry. Only drop the active-region counter. */
    __kmp_lithe_region_depth = 0;
    return;
  }

  /* Leaving a NESTED region: exit + destroy its per-region sched, rebinding its
   * workers back to the parent sched. */
  if (__kmp_lithe_fj_sp <= 1) {
    __kmp_lithe_region_depth = depth - 1;
    return;
  }
  fj = __kmp_lithe_fj_stack[__kmp_lithe_fj_sp - 1];
  parent_sched = __kmp_lithe_fj_stack[__kmp_lithe_fj_sp - 2];
  if (parent_sched) {
    for (i = 1; i < team->t.t_nproc; i++) {
      kmp_info_t *th = team->t.t_threads[i];
      if (!th || !th->th.th_lithe_ctx)
        continue;
      __kmp_lithe_rebind_ctx((lithe_fork_join_context_t *)th->th.th_lithe_ctx,
                             parent_sched);
    }
  }
  lithe_sched_exit();
  lithe_fork_join_sched_destroy(fj);
  __kmp_lithe_fj_sp--;
  __kmp_lithe_region_depth = depth - 1;
}
#endif /* LIBOMP_USE_LITHE */

#if !KMP_OS_FREEBSD && !KMP_OS_NETBSD
#include <alloca.h>
#endif
#include <math.h> // HUGE_VAL.
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/times.h>
#include <unistd.h>

#if KMP_OS_LINUX && !KMP_OS_CNK
#include <sys/sysinfo.h>
#if KMP_USE_FUTEX
// We should really include <futex.h>, but that causes compatibility problems on
// different Linux* OS distributions that either require that you include (or
// break when you try to include) <pci/types.h>. Since all we need is the two
// macros below (which are part of the kernel ABI, so can't change) we just
// define the constants here and don't include <futex.h>
#ifndef FUTEX_WAIT
#define FUTEX_WAIT 0
#endif
#ifndef FUTEX_WAKE
#define FUTEX_WAKE 1
#endif
#endif
#elif KMP_OS_DARWIN
#include <mach/mach.h>
#include <sys/sysctl.h>
#elif KMP_OS_FREEBSD
#include <pthread_np.h>
#endif

#include <ctype.h>
#include <dirent.h>
#include <fcntl.h>

#include "tsan_annotations.h"

struct kmp_sys_timer {
  struct timespec start;
};

// Convert timespec to nanoseconds.
#define TS2NS(timespec) (((timespec).tv_sec * 1e9) + (timespec).tv_nsec)

static struct kmp_sys_timer __kmp_sys_timer_data;

#if KMP_HANDLE_SIGNALS
typedef void (*sig_func_t)(int);
STATIC_EFI2_WORKAROUND struct sigaction __kmp_sighldrs[NSIG];
static sigset_t __kmp_sigset;
#endif

static int __kmp_init_runtime = FALSE;

static int __kmp_fork_count = 0;

static pthread_condattr_t __kmp_suspend_cond_attr;
static pthread_mutexattr_t __kmp_suspend_mutex_attr;

static kmp_cond_align_t __kmp_wait_cv;
static kmp_mutex_align_t __kmp_wait_mx;

kmp_uint64 __kmp_ticks_per_msec = 1000000;

#ifdef DEBUG_SUSPEND
static void __kmp_print_cond(char *buffer, kmp_cond_align_t *cond) {
  KMP_SNPRINTF(buffer, 128, "(cond (lock (%ld, %d)), (descr (%p)))",
               cond->c_cond.__c_lock.__status, cond->c_cond.__c_lock.__spinlock,
               cond->c_cond.__c_waiting);
}
#endif

#if (KMP_OS_LINUX && KMP_AFFINITY_SUPPORTED)

/* Affinity support */

void __kmp_affinity_bind_thread(int which) {
  KMP_ASSERT2(KMP_AFFINITY_CAPABLE(),
              "Illegal set affinity operation when not capable");

  kmp_affin_mask_t *mask;
  KMP_CPU_ALLOC_ON_STACK(mask);
  KMP_CPU_ZERO(mask);
  KMP_CPU_SET(which, mask);
  __kmp_set_system_affinity(mask, TRUE);
  KMP_CPU_FREE_FROM_STACK(mask);
}

/* Determine if we can access affinity functionality on this version of
 * Linux* OS by checking __NR_sched_{get,set}affinity system calls, and set
 * __kmp_affin_mask_size to the appropriate value (0 means not capable). */
void __kmp_affinity_determine_capable(const char *env_var) {
// Check and see if the OS supports thread affinity.

#define KMP_CPU_SET_SIZE_LIMIT (1024 * 1024)

  int gCode;
  int sCode;
  unsigned char *buf;
  buf = (unsigned char *)KMP_INTERNAL_MALLOC(KMP_CPU_SET_SIZE_LIMIT);

  // If Linux* OS:
  // If the syscall fails or returns a suggestion for the size,
  // then we don't have to search for an appropriate size.
  gCode = syscall(__NR_sched_getaffinity, 0, KMP_CPU_SET_SIZE_LIMIT, buf);
  KA_TRACE(30, ("__kmp_affinity_determine_capable: "
                "initial getaffinity call returned %d errno = %d\n",
                gCode, errno));

  // if ((gCode < 0) && (errno == ENOSYS))
  if (gCode < 0) {
    // System call not supported
    if (__kmp_affinity_verbose ||
        (__kmp_affinity_warnings && (__kmp_affinity_type != affinity_none) &&
         (__kmp_affinity_type != affinity_default) &&
         (__kmp_affinity_type != affinity_disabled))) {
      int error = errno;
      kmp_msg_t err_code = KMP_ERR(error);
      __kmp_msg(kmp_ms_warning, KMP_MSG(GetAffSysCallNotSupported, env_var),
                err_code, __kmp_msg_null);
      if (__kmp_generate_warnings == kmp_warnings_off) {
        __kmp_str_free(&err_code.str);
      }
    }
    KMP_AFFINITY_DISABLE();
    KMP_INTERNAL_FREE(buf);
    return;
  }
  if (gCode > 0) { // Linux* OS only
    // The optimal situation: the OS returns the size of the buffer it expects.
    //
    // A verification of correct behavior is that Isetaffinity on a NULL
    // buffer with the same size fails with errno set to EFAULT.
    sCode = syscall(__NR_sched_setaffinity, 0, gCode, NULL);
    KA_TRACE(30, ("__kmp_affinity_determine_capable: "
                  "setaffinity for mask size %d returned %d errno = %d\n",
                  gCode, sCode, errno));
    if (sCode < 0) {
      if (errno == ENOSYS) {
        if (__kmp_affinity_verbose ||
            (__kmp_affinity_warnings &&
             (__kmp_affinity_type != affinity_none) &&
             (__kmp_affinity_type != affinity_default) &&
             (__kmp_affinity_type != affinity_disabled))) {
          int error = errno;
          kmp_msg_t err_code = KMP_ERR(error);
          __kmp_msg(kmp_ms_warning, KMP_MSG(SetAffSysCallNotSupported, env_var),
                    err_code, __kmp_msg_null);
          if (__kmp_generate_warnings == kmp_warnings_off) {
            __kmp_str_free(&err_code.str);
          }
        }
        KMP_AFFINITY_DISABLE();
        KMP_INTERNAL_FREE(buf);
      }
      if (errno == EFAULT) {
        KMP_AFFINITY_ENABLE(gCode);
        KA_TRACE(10, ("__kmp_affinity_determine_capable: "
                      "affinity supported (mask size %d)\n",
                      (int)__kmp_affin_mask_size));
        KMP_INTERNAL_FREE(buf);
        return;
      }
    }
  }

  // Call the getaffinity system call repeatedly with increasing set sizes
  // until we succeed, or reach an upper bound on the search.
  KA_TRACE(30, ("__kmp_affinity_determine_capable: "
                "searching for proper set size\n"));
  int size;
  for (size = 1; size <= KMP_CPU_SET_SIZE_LIMIT; size *= 2) {
    gCode = syscall(__NR_sched_getaffinity, 0, size, buf);
    KA_TRACE(30, ("__kmp_affinity_determine_capable: "
                  "getaffinity for mask size %d returned %d errno = %d\n",
                  size, gCode, errno));

    if (gCode < 0) {
      if (errno == ENOSYS) {
        // We shouldn't get here
        KA_TRACE(30, ("__kmp_affinity_determine_capable: "
                      "inconsistent OS call behavior: errno == ENOSYS for mask "
                      "size %d\n",
                      size));
        if (__kmp_affinity_verbose ||
            (__kmp_affinity_warnings &&
             (__kmp_affinity_type != affinity_none) &&
             (__kmp_affinity_type != affinity_default) &&
             (__kmp_affinity_type != affinity_disabled))) {
          int error = errno;
          kmp_msg_t err_code = KMP_ERR(error);
          __kmp_msg(kmp_ms_warning, KMP_MSG(GetAffSysCallNotSupported, env_var),
                    err_code, __kmp_msg_null);
          if (__kmp_generate_warnings == kmp_warnings_off) {
            __kmp_str_free(&err_code.str);
          }
        }
        KMP_AFFINITY_DISABLE();
        KMP_INTERNAL_FREE(buf);
        return;
      }
      continue;
    }

    sCode = syscall(__NR_sched_setaffinity, 0, gCode, NULL);
    KA_TRACE(30, ("__kmp_affinity_determine_capable: "
                  "setaffinity for mask size %d returned %d errno = %d\n",
                  gCode, sCode, errno));
    if (sCode < 0) {
      if (errno == ENOSYS) { // Linux* OS only
        // We shouldn't get here
        KA_TRACE(30, ("__kmp_affinity_determine_capable: "
                      "inconsistent OS call behavior: errno == ENOSYS for mask "
                      "size %d\n",
                      size));
        if (__kmp_affinity_verbose ||
            (__kmp_affinity_warnings &&
             (__kmp_affinity_type != affinity_none) &&
             (__kmp_affinity_type != affinity_default) &&
             (__kmp_affinity_type != affinity_disabled))) {
          int error = errno;
          kmp_msg_t err_code = KMP_ERR(error);
          __kmp_msg(kmp_ms_warning, KMP_MSG(SetAffSysCallNotSupported, env_var),
                    err_code, __kmp_msg_null);
          if (__kmp_generate_warnings == kmp_warnings_off) {
            __kmp_str_free(&err_code.str);
          }
        }
        KMP_AFFINITY_DISABLE();
        KMP_INTERNAL_FREE(buf);
        return;
      }
      if (errno == EFAULT) {
        KMP_AFFINITY_ENABLE(gCode);
        KA_TRACE(10, ("__kmp_affinity_determine_capable: "
                      "affinity supported (mask size %d)\n",
                      (int)__kmp_affin_mask_size));
        KMP_INTERNAL_FREE(buf);
        return;
      }
    }
  }
  // save uncaught error code
  // int error = errno;
  KMP_INTERNAL_FREE(buf);
  // restore uncaught error code, will be printed at the next KMP_WARNING below
  // errno = error;

  // Affinity is not supported
  KMP_AFFINITY_DISABLE();
  KA_TRACE(10, ("__kmp_affinity_determine_capable: "
                "cannot determine mask size - affinity not supported\n"));
  if (__kmp_affinity_verbose ||
      (__kmp_affinity_warnings && (__kmp_affinity_type != affinity_none) &&
       (__kmp_affinity_type != affinity_default) &&
       (__kmp_affinity_type != affinity_disabled))) {
    KMP_WARNING(AffCantGetMaskSize, env_var);
  }
}

#endif // KMP_OS_LINUX && KMP_AFFINITY_SUPPORTED

#if KMP_USE_FUTEX

int __kmp_futex_determine_capable() {
  int loc = 0;
  int rc = syscall(__NR_futex, &loc, FUTEX_WAKE, 1, NULL, NULL, 0);
  int retval = (rc == 0) || (errno != ENOSYS);

  KA_TRACE(10,
           ("__kmp_futex_determine_capable: rc = %d errno = %d\n", rc, errno));
  KA_TRACE(10, ("__kmp_futex_determine_capable: futex syscall%s supported\n",
                retval ? "" : " not"));

  return retval;
}

#endif // KMP_USE_FUTEX

#if (KMP_ARCH_X86 || KMP_ARCH_X86_64) && (!KMP_ASM_INTRINS)
/* Only 32-bit "add-exchange" instruction on IA-32 architecture causes us to
   use compare_and_store for these routines */

kmp_int8 __kmp_test_then_or8(volatile kmp_int8 *p, kmp_int8 d) {
  kmp_int8 old_value, new_value;

  old_value = TCR_1(*p);
  new_value = old_value | d;

  while (!KMP_COMPARE_AND_STORE_REL8(p, old_value, new_value)) {
    KMP_CPU_PAUSE();
    old_value = TCR_1(*p);
    new_value = old_value | d;
  }
  return old_value;
}

kmp_int8 __kmp_test_then_and8(volatile kmp_int8 *p, kmp_int8 d) {
  kmp_int8 old_value, new_value;

  old_value = TCR_1(*p);
  new_value = old_value & d;

  while (!KMP_COMPARE_AND_STORE_REL8(p, old_value, new_value)) {
    KMP_CPU_PAUSE();
    old_value = TCR_1(*p);
    new_value = old_value & d;
  }
  return old_value;
}

kmp_uint32 __kmp_test_then_or32(volatile kmp_uint32 *p, kmp_uint32 d) {
  kmp_uint32 old_value, new_value;

  old_value = TCR_4(*p);
  new_value = old_value | d;

  while (!KMP_COMPARE_AND_STORE_REL32(p, old_value, new_value)) {
    KMP_CPU_PAUSE();
    old_value = TCR_4(*p);
    new_value = old_value | d;
  }
  return old_value;
}

kmp_uint32 __kmp_test_then_and32(volatile kmp_uint32 *p, kmp_uint32 d) {
  kmp_uint32 old_value, new_value;

  old_value = TCR_4(*p);
  new_value = old_value & d;

  while (!KMP_COMPARE_AND_STORE_REL32(p, old_value, new_value)) {
    KMP_CPU_PAUSE();
    old_value = TCR_4(*p);
    new_value = old_value & d;
  }
  return old_value;
}

#if KMP_ARCH_X86
kmp_int8 __kmp_test_then_add8(volatile kmp_int8 *p, kmp_int8 d) {
  kmp_int8 old_value, new_value;

  old_value = TCR_1(*p);
  new_value = old_value + d;

  while (!KMP_COMPARE_AND_STORE_REL8(p, old_value, new_value)) {
    KMP_CPU_PAUSE();
    old_value = TCR_1(*p);
    new_value = old_value + d;
  }
  return old_value;
}

kmp_int64 __kmp_test_then_add64(volatile kmp_int64 *p, kmp_int64 d) {
  kmp_int64 old_value, new_value;

  old_value = TCR_8(*p);
  new_value = old_value + d;

  while (!KMP_COMPARE_AND_STORE_REL64(p, old_value, new_value)) {
    KMP_CPU_PAUSE();
    old_value = TCR_8(*p);
    new_value = old_value + d;
  }
  return old_value;
}
#endif /* KMP_ARCH_X86 */

kmp_uint64 __kmp_test_then_or64(volatile kmp_uint64 *p, kmp_uint64 d) {
  kmp_uint64 old_value, new_value;

  old_value = TCR_8(*p);
  new_value = old_value | d;
  while (!KMP_COMPARE_AND_STORE_REL64(p, old_value, new_value)) {
    KMP_CPU_PAUSE();
    old_value = TCR_8(*p);
    new_value = old_value | d;
  }
  return old_value;
}

kmp_uint64 __kmp_test_then_and64(volatile kmp_uint64 *p, kmp_uint64 d) {
  kmp_uint64 old_value, new_value;

  old_value = TCR_8(*p);
  new_value = old_value & d;
  while (!KMP_COMPARE_AND_STORE_REL64(p, old_value, new_value)) {
    KMP_CPU_PAUSE();
    old_value = TCR_8(*p);
    new_value = old_value & d;
  }
  return old_value;
}

#endif /* (KMP_ARCH_X86 || KMP_ARCH_X86_64) && (! KMP_ASM_INTRINS) */

void __kmp_terminate_thread(int gtid) {
  int status;
  kmp_info_t *th = __kmp_threads[gtid];

  if (!th)
    return;

#ifdef KMP_CANCEL_THREADS
  KA_TRACE(10, ("__kmp_terminate_thread: kill (%d)\n", gtid));
  status = pthread_cancel(th->th.th_info.ds.ds_thread);
  if (status != 0 && status != ESRCH) {
    __kmp_fatal(KMP_MSG(CantTerminateWorkerThread), KMP_ERR(status),
                __kmp_msg_null);
  }
#endif
  __kmp_yield(TRUE);
} //

/* Set thread stack info according to values returned by pthread_getattr_np().
   If values are unreasonable, assume call failed and use incremental stack
   refinement method instead. Returns TRUE if the stack parameters could be
   determined exactly, FALSE if incremental refinement is necessary. */
static kmp_int32 __kmp_set_stack_info(int gtid, kmp_info_t *th) {
  int stack_data;
#if KMP_OS_LINUX || KMP_OS_FREEBSD || KMP_OS_NETBSD
  /* Linux* OS only -- no pthread_getattr_np support on OS X* */
  pthread_attr_t attr;
  int status;
  size_t size = 0;
  void *addr = 0;

  /* Always do incremental stack refinement for ubermaster threads since the
     initial thread stack range can be reduced by sibling thread creation so
     pthread_attr_getstack may cause thread gtid aliasing */
  if (!KMP_UBER_GTID(gtid)) {

    /* Fetch the real thread attributes */
    status = pthread_attr_init(&attr);
    KMP_CHECK_SYSFAIL("pthread_attr_init", status);
#if KMP_OS_FREEBSD || KMP_OS_NETBSD
    status = pthread_attr_get_np(pthread_self(), &attr);
    KMP_CHECK_SYSFAIL("pthread_attr_get_np", status);
#else
    status = pthread_getattr_np(pthread_self(), &attr);
    KMP_CHECK_SYSFAIL("pthread_getattr_np", status);
#endif
    status = pthread_attr_getstack(&attr, &addr, &size);
    KMP_CHECK_SYSFAIL("pthread_attr_getstack", status);
    KA_TRACE(60,
             ("__kmp_set_stack_info: T#%d pthread_attr_getstack returned size:"
              " %lu, low addr: %p\n",
              gtid, size, addr));
    status = pthread_attr_destroy(&attr);
    KMP_CHECK_SYSFAIL("pthread_attr_destroy", status);
  }

  if (size != 0 && addr != 0) { // was stack parameter determination successful?
    /* Store the correct base and size */
    TCW_PTR(th->th.th_info.ds.ds_stackbase, (((char *)addr) + size));
    TCW_PTR(th->th.th_info.ds.ds_stacksize, size);
    TCW_4(th->th.th_info.ds.ds_stackgrow, FALSE);
    return TRUE;
  }
#endif /* KMP_OS_LINUX || KMP_OS_FREEBSD || KMP_OS_NETBSD */
  /* Use incremental refinement starting from initial conservative estimate */
  TCW_PTR(th->th.th_info.ds.ds_stacksize, 0);
  TCW_PTR(th->th.th_info.ds.ds_stackbase, &stack_data);
  TCW_4(th->th.th_info.ds.ds_stackgrow, TRUE);
  return FALSE;
}

static void *__kmp_launch_worker(void *thr) {
  int status, old_type, old_state;
#ifdef KMP_BLOCK_SIGNALS
  sigset_t new_set, old_set;
#endif /* KMP_BLOCK_SIGNALS */
  void *exit_val;
#if KMP_OS_LINUX || KMP_OS_FREEBSD || KMP_OS_NETBSD
  void *volatile padding = 0;
#endif
  int gtid;

  gtid = ((kmp_info_t *)thr)->th.th_info.ds.ds_gtid;
  __kmp_gtid_set_specific(gtid);
#ifdef KMP_TDATA_GTID
  __kmp_gtid = gtid;
#endif
#if KMP_STATS_ENABLED
  // set thread local index to point to thread-specific stats
  __kmp_stats_thread_ptr = ((kmp_info_t *)thr)->th.th_stats;
  KMP_START_EXPLICIT_TIMER(OMP_worker_thread_life);
  KMP_SET_THREAD_STATE(IDLE);
  KMP_INIT_PARTITIONED_TIMERS(OMP_idle);
#endif

#if USE_ITT_BUILD
  __kmp_itt_thread_name(gtid);
#endif /* USE_ITT_BUILD */

#if KMP_AFFINITY_SUPPORTED
  __kmp_affinity_set_init_mask(gtid, FALSE);
#endif

#ifndef LIBOMP_USE_LITHE
#ifdef KMP_CANCEL_THREADS
  status = pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, &old_type);
  KMP_CHECK_SYSFAIL("pthread_setcanceltype", status);
  // josh todo: isn't PTHREAD_CANCEL_ENABLE default for newly-created threads?
  status = pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &old_state);
  KMP_CHECK_SYSFAIL("pthread_setcancelstate", status);
#endif
#endif /* !LIBOMP_USE_LITHE */

#if KMP_ARCH_X86 || KMP_ARCH_X86_64
  // Set FP control regs to be a copy of the parallel initialization thread's.
  __kmp_clear_x87_fpu_status_word();
  __kmp_load_x87_fpu_control_word(&__kmp_init_x87_fpu_control_word);
  __kmp_load_mxcsr(&__kmp_init_mxcsr);
#endif /* KMP_ARCH_X86 || KMP_ARCH_X86_64 */

#ifndef LIBOMP_USE_LITHE
#ifdef KMP_BLOCK_SIGNALS
  status = sigfillset(&new_set);
  KMP_CHECK_SYSFAIL_ERRNO("sigfillset", status);
  status = pthread_sigmask(SIG_BLOCK, &new_set, &old_set);
  KMP_CHECK_SYSFAIL("pthread_sigmask", status);
#endif /* KMP_BLOCK_SIGNALS */
#endif /* !LIBOMP_USE_LITHE */

#if KMP_OS_LINUX || KMP_OS_FREEBSD || KMP_OS_NETBSD
  if (__kmp_stkoffset > 0 && gtid > 0) {
    padding = KMP_ALLOCA(gtid * __kmp_stkoffset);
  }
#endif

  KMP_MB();
#ifndef LIBOMP_USE_LITHE
  __kmp_set_stack_info(gtid, (kmp_info_t *)thr);
  __kmp_check_stack_overlap((kmp_info_t *)thr);
#endif

  exit_val = __kmp_launch_thread((kmp_info_t *)thr);

#ifndef LIBOMP_USE_LITHE
#ifdef KMP_BLOCK_SIGNALS
  status = pthread_sigmask(SIG_SETMASK, &old_set, NULL);
  KMP_CHECK_SYSFAIL("pthread_sigmask", status);
#endif /* KMP_BLOCK_SIGNALS */
#endif /* !LIBOMP_USE_LITHE */

  return exit_val;
}

#ifdef LIBOMP_USE_LITHE
static void __kmp_lithe_launch_worker(void *thr) {
  kmp_info_t *th = (kmp_info_t *)thr;
  __kmp_launch_worker(thr);
  /* Tell the reaper we are done using kmp_info_t and about to self-destroy our
   * lithe context (the lithe analogue of a pthread returning, which lets
   * pthread_join complete in the non-lithe path). After the unlock below we
   * touch neither `th` nor the join ctl, so the reaper may free both. */
  struct kmp_lithe_join_ctl *jc =
      (struct kmp_lithe_join_ctl *)th->th.th_lithe_join_ctl;
  if (jc) {
    lithe_mutex_lock(&jc->mx);
    jc->exited = 1;
    lithe_condvar_signal(&jc->cv);
    lithe_mutex_unlock(&jc->mx);
  }
  lithe_context_exit();
}
#endif

#if KMP_USE_MONITOR
/* The monitor thread controls all of the threads in the complex */

static void *__kmp_launch_monitor(void *thr) {
  int status, old_type, old_state;
#ifdef KMP_BLOCK_SIGNALS
  sigset_t new_set;
#endif /* KMP_BLOCK_SIGNALS */
  struct timespec interval;
  int yield_count;
  int yield_cycles = 0;

  KMP_MB(); /* Flush all pending memory write invalidates.  */

  KA_TRACE(10, ("__kmp_launch_monitor: #1 launched\n"));

  /* register us as the monitor thread */
  __kmp_gtid_set_specific(KMP_GTID_MONITOR);
#ifdef KMP_TDATA_GTID
  __kmp_gtid = KMP_GTID_MONITOR;
#endif

  KMP_MB();

#if USE_ITT_BUILD
  // Instruct Intel(R) Threading Tools to ignore monitor thread.
  __kmp_itt_thread_ignore();
#endif /* USE_ITT_BUILD */

  __kmp_set_stack_info(((kmp_info_t *)thr)->th.th_info.ds.ds_gtid,
                       (kmp_info_t *)thr);

  __kmp_check_stack_overlap((kmp_info_t *)thr);

#ifdef KMP_CANCEL_THREADS
  status = pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, &old_type);
  KMP_CHECK_SYSFAIL("pthread_setcanceltype", status);
  // josh todo: isn't PTHREAD_CANCEL_ENABLE default for newly-created threads?
  status = pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &old_state);
  KMP_CHECK_SYSFAIL("pthread_setcancelstate", status);
#endif

#if KMP_REAL_TIME_FIX
  // This is a potential fix which allows application with real-time scheduling
  // policy work. However, decision about the fix is not made yet, so it is
  // disabled by default.
  { // Are program started with real-time scheduling policy?
    int sched = sched_getscheduler(0);
    if (sched == SCHED_FIFO || sched == SCHED_RR) {
      // Yes, we are a part of real-time application. Try to increase the
      // priority of the monitor.
      struct sched_param param;
      int max_priority = sched_get_priority_max(sched);
      int rc;
      KMP_WARNING(RealTimeSchedNotSupported);
      sched_getparam(0, &param);
      if (param.sched_priority < max_priority) {
        param.sched_priority += 1;
        rc = sched_setscheduler(0, sched, &param);
        if (rc != 0) {
          int error = errno;
          kmp_msg_t err_code = KMP_ERR(error);
          __kmp_msg(kmp_ms_warning, KMP_MSG(CantChangeMonitorPriority),
                    err_code, KMP_MSG(MonitorWillStarve), __kmp_msg_null);
          if (__kmp_generate_warnings == kmp_warnings_off) {
            __kmp_str_free(&err_code.str);
          }
        }
      } else {
        // We cannot abort here, because number of CPUs may be enough for all
        // the threads, including the monitor thread, so application could
        // potentially work...
        __kmp_msg(kmp_ms_warning, KMP_MSG(RunningAtMaxPriority),
                  KMP_MSG(MonitorWillStarve), KMP_HNT(RunningAtMaxPriority),
                  __kmp_msg_null);
      }
    }
    // AC: free thread that waits for monitor started
    TCW_4(__kmp_global.g.g_time.dt.t_value, 0);
  }
#endif // KMP_REAL_TIME_FIX

  KMP_MB(); /* Flush all pending memory write invalidates.  */

  if (__kmp_monitor_wakeups == 1) {
    interval.tv_sec = 1;
    interval.tv_nsec = 0;
  } else {
    interval.tv_sec = 0;
    interval.tv_nsec = (KMP_NSEC_PER_SEC / __kmp_monitor_wakeups);
  }

  KA_TRACE(10, ("__kmp_launch_monitor: #2 monitor\n"));

  if (__kmp_yield_cycle) {
    __kmp_yielding_on = 0; /* Start out with yielding shut off */
    yield_count = __kmp_yield_off_count;
  } else {
    __kmp_yielding_on = 1; /* Yielding is on permanently */
  }

  while (!TCR_4(__kmp_global.g.g_done)) {
    struct timespec now;
    struct timeval tval;

    /*  This thread monitors the state of the system */

    KA_TRACE(15, ("__kmp_launch_monitor: update\n"));

    status = gettimeofday(&tval, NULL);
    KMP_CHECK_SYSFAIL_ERRNO("gettimeofday", status);
    TIMEVAL_TO_TIMESPEC(&tval, &now);

    now.tv_sec += interval.tv_sec;
    now.tv_nsec += interval.tv_nsec;

    if (now.tv_nsec >= KMP_NSEC_PER_SEC) {
      now.tv_sec += 1;
      now.tv_nsec -= KMP_NSEC_PER_SEC;
    }

    status = pthread_mutex_lock(&__kmp_wait_mx.m_mutex);
    KMP_CHECK_SYSFAIL("pthread_mutex_lock", status);
    // AC: the monitor should not fall asleep if g_done has been set
    if (!TCR_4(__kmp_global.g.g_done)) { // check once more under mutex
      status = pthread_cond_timedwait(&__kmp_wait_cv.c_cond,
                                      &__kmp_wait_mx.m_mutex, &now);
      if (status != 0) {
        if (status != ETIMEDOUT && status != EINTR) {
          KMP_SYSFAIL("pthread_cond_timedwait", status);
        }
      }
    }
    status = pthread_mutex_unlock(&__kmp_wait_mx.m_mutex);
    KMP_CHECK_SYSFAIL("pthread_mutex_unlock", status);

    if (__kmp_yield_cycle) {
      yield_cycles++;
      if ((yield_cycles % yield_count) == 0) {
        if (__kmp_yielding_on) {
          __kmp_yielding_on = 0; /* Turn it off now */
          yield_count = __kmp_yield_off_count;
        } else {
          __kmp_yielding_on = 1; /* Turn it on now */
          yield_count = __kmp_yield_on_count;
        }
        yield_cycles = 0;
      }
    } else {
      __kmp_yielding_on = 1;
    }

    TCW_4(__kmp_global.g.g_time.dt.t_value,
          TCR_4(__kmp_global.g.g_time.dt.t_value) + 1);

    KMP_MB(); /* Flush all pending memory write invalidates.  */
  }

  KA_TRACE(10, ("__kmp_launch_monitor: #3 cleanup\n"));

#ifdef KMP_BLOCK_SIGNALS
  status = sigfillset(&new_set);
  KMP_CHECK_SYSFAIL_ERRNO("sigfillset", status);
  status = pthread_sigmask(SIG_UNBLOCK, &new_set, NULL);
  KMP_CHECK_SYSFAIL("pthread_sigmask", status);
#endif /* KMP_BLOCK_SIGNALS */

  KA_TRACE(10, ("__kmp_launch_monitor: #4 finished\n"));

  if (__kmp_global.g.g_abort != 0) {
    /* now we need to terminate the worker threads  */
    /* the value of t_abort is the signal we caught */

    int gtid;

    KA_TRACE(10, ("__kmp_launch_monitor: #5 terminate sig=%d\n",
                  __kmp_global.g.g_abort));

    /* terminate the OpenMP worker threads */
    /* TODO this is not valid for sibling threads!!
     * the uber master might not be 0 anymore.. */
    for (gtid = 1; gtid < __kmp_threads_capacity; ++gtid)
      __kmp_terminate_thread(gtid);

    __kmp_cleanup();

    KA_TRACE(10, ("__kmp_launch_monitor: #6 raise sig=%d\n",
                  __kmp_global.g.g_abort));

    if (__kmp_global.g.g_abort > 0)
      raise(__kmp_global.g.g_abort);
  }

  KA_TRACE(10, ("__kmp_launch_monitor: #7 exit\n"));

  return thr;
}
#endif // KMP_USE_MONITOR

void __kmp_create_worker(int gtid, kmp_info_t *th, size_t stack_size) {
  int status;

  th->th.th_info.ds.ds_gtid = gtid;

#if KMP_STATS_ENABLED
  __kmp_acquire_tas_lock(&__kmp_stats_lock, gtid);
  if (!KMP_UBER_GTID(gtid)) {
    th->th.th_stats = __kmp_stats_list->push_back(gtid);
  } else {
    th->th.th_stats = __kmp_stats_thread_ptr;
  }
  __kmp_release_tas_lock(&__kmp_stats_lock, gtid);
#endif // KMP_STATS_ENABLED

  if (KMP_UBER_GTID(gtid)) {
    KA_TRACE(10, ("__kmp_create_worker: uber thread (%d)\n", gtid));
#ifdef LIBOMP_USE_LITHE
    th->th.th_info.ds.ds_thread = pthread_self();
#else
    th->th.th_info.ds.ds_thread = pthread_self();
    __kmp_set_stack_info(gtid, th);
    __kmp_check_stack_overlap(th);
#endif
    return;
  }

  KA_TRACE(10, ("__kmp_create_worker: try to create thread (%d)\n", gtid));
  KMP_MB();

#ifdef LIBOMP_USE_LITHE
  {
    lithe_fork_join_sched_t *sched = __kmp_lithe_worker_sched_for_create();
    if (!sched) {
      __kmp_fatal(KMP_MSG(NoResourcesForWorkerThread), KMP_ERR(ENOMEM),
                  __kmp_msg_null);
    }

    if (stack_size < (size_t)(64 * 1024))
      stack_size = 4 * 1024 * 1024;

    lithe_fork_join_context_t *ctx =
        lithe_fork_join_context_create(sched, stack_size,
                                       __kmp_lithe_launch_worker, (void *)th);
    if (!ctx) {
      __kmp_fatal(KMP_MSG(NoResourcesForWorkerThread), KMP_ERR(ENOMEM),
                  __kmp_msg_null);
    }

    th->th.th_lithe_ctx = ctx;
    th->th.th_info.ds.ds_thread = pthread_self();

    /* Allocate the reap-join handshake now (before the worker can run to exit)
     * so __kmp_reap_worker can wait for this worker instead of destroying its
     * self-destroying context twice. */
    {
      struct kmp_lithe_join_ctl *jc =
          (struct kmp_lithe_join_ctl *)malloc(sizeof(*jc));
      if (!jc) {
        __kmp_fatal(KMP_MSG(NoResourcesForWorkerThread), KMP_ERR(ENOMEM),
                    __kmp_msg_null);
      }
      lithe_mutex_init(&jc->mx, NULL);
      lithe_condvar_init(&jc->cv);
      jc->exited = 0;
      th->th.th_lithe_join_ctl = jc;
    }

    KA_TRACE(10, ("__kmp_create_worker: created lithe context %p for T#%d\n",
                  (void *)ctx, gtid));
  }
#else /* !LIBOMP_USE_LITHE */
  {
    pthread_t handle;
    pthread_attr_t thread_attr;

#ifdef KMP_THREAD_ATTR
    status = pthread_attr_init(&thread_attr);
    if (status != 0) {
      __kmp_fatal(KMP_MSG(CantInitThreadAttrs), KMP_ERR(status), __kmp_msg_null);
    }
    status = pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);
    if (status != 0) {
      __kmp_fatal(KMP_MSG(CantSetWorkerState), KMP_ERR(status), __kmp_msg_null);
    }

    stack_size += gtid * __kmp_stkoffset * 2;

    KA_TRACE(10, ("__kmp_create_worker: T#%d, default stacksize = %lu bytes, "
                  "__kmp_stksize = %lu bytes, final stacksize = %lu bytes\n",
                  gtid, KMP_DEFAULT_STKSIZE, __kmp_stksize, stack_size));

#ifdef _POSIX_THREAD_ATTR_STACKSIZE
    status = pthread_attr_setstacksize(&thread_attr, stack_size);
#ifdef KMP_BACKUP_STKSIZE
    if (status != 0) {
      if (!__kmp_env_stksize) {
        stack_size = KMP_BACKUP_STKSIZE + gtid * __kmp_stkoffset;
        __kmp_stksize = KMP_BACKUP_STKSIZE;
        KA_TRACE(10, ("__kmp_create_worker: T#%d, default stacksize = %lu bytes, "
                      "__kmp_stksize = %lu bytes, (backup) final stacksize = %lu "
                      "bytes\n",
                      gtid, KMP_DEFAULT_STKSIZE, __kmp_stksize, stack_size));
        status = pthread_attr_setstacksize(&thread_attr, stack_size);
      }
    }
#endif /* KMP_BACKUP_STKSIZE */
    if (status != 0) {
      __kmp_fatal(KMP_MSG(CantSetWorkerStackSize, stack_size), KMP_ERR(status),
                  KMP_HNT(ChangeWorkerStackSize), __kmp_msg_null);
    }
#endif /* _POSIX_THREAD_ATTR_STACKSIZE */

#endif /* KMP_THREAD_ATTR */

    status =
        pthread_create(&handle, &thread_attr, __kmp_launch_worker, (void *)th);
    if (status != 0 || !handle) {
#ifdef _POSIX_THREAD_ATTR_STACKSIZE
      if (status == EINVAL) {
        __kmp_fatal(KMP_MSG(CantSetWorkerStackSize, stack_size), KMP_ERR(status),
                    KMP_HNT(IncreaseWorkerStackSize), __kmp_msg_null);
      }
      if (status == ENOMEM) {
        __kmp_fatal(KMP_MSG(CantSetWorkerStackSize, stack_size), KMP_ERR(status),
                    KMP_HNT(DecreaseWorkerStackSize), __kmp_msg_null);
      }
#endif /* _POSIX_THREAD_ATTR_STACKSIZE */
      if (status == EAGAIN) {
        __kmp_fatal(KMP_MSG(NoResourcesForWorkerThread), KMP_ERR(status),
                    KMP_HNT(Decrease_NUM_THREADS), __kmp_msg_null);
      }
      KMP_SYSFAIL("pthread_create", status);
    }

    th->th.th_info.ds.ds_thread = handle;

#ifdef KMP_THREAD_ATTR
    status = pthread_attr_destroy(&thread_attr);
    if (status) {
      kmp_msg_t err_code = KMP_ERR(status);
      __kmp_msg(kmp_ms_warning, KMP_MSG(CantDestroyThreadAttrs), err_code,
                __kmp_msg_null);
      if (__kmp_generate_warnings == kmp_warnings_off) {
        __kmp_str_free(&err_code.str);
      }
    }
#endif /* KMP_THREAD_ATTR */
  }
#endif /* LIBOMP_USE_LITHE */

  KMP_MB();
  KA_TRACE(10, ("__kmp_create_worker: done creating thread (%d)\n", gtid));

} // __kmp_create_worker

#if KMP_USE_MONITOR
void __kmp_create_monitor(kmp_info_t *th) {
  pthread_t handle;
  pthread_attr_t thread_attr;
  size_t size;
  int status;
  int auto_adj_size = FALSE;

  if (__kmp_dflt_blocktime == KMP_MAX_BLOCKTIME) {
    // We don't need monitor thread in case of MAX_BLOCKTIME
    KA_TRACE(10, ("__kmp_create_monitor: skipping monitor thread because of "
                  "MAX blocktime\n"));
    th->th.th_info.ds.ds_tid = 0; // this makes reap_monitor no-op
    th->th.th_info.ds.ds_gtid = 0;
    return;
  }
  KA_TRACE(10, ("__kmp_create_monitor: try to create monitor\n"));

  KMP_MB(); /* Flush all pending memory write invalidates.  */

  th->th.th_info.ds.ds_tid = KMP_GTID_MONITOR;
  th->th.th_info.ds.ds_gtid = KMP_GTID_MONITOR;
#if KMP_REAL_TIME_FIX
  TCW_4(__kmp_global.g.g_time.dt.t_value,
        -1); // Will use it for synchronization a bit later.
#else
  TCW_4(__kmp_global.g.g_time.dt.t_value, 0);
#endif // KMP_REAL_TIME_FIX

#ifdef LIBOMP_USE_LITHE
extern void __kmp_lithe_parlib_monitor_start(kmp_info_t *th);
#endif

#ifdef LIBOMP_USE_LITHE
  th->th.th_info.ds.ds_thread = (pthread_t)0;
  __kmp_lithe_parlib_monitor_start(th);

  KMP_MB(); /* Flush all pending memory write invalidates.  */

  KA_TRACE(10,
           ("__kmp_create_monitor: lithe parlib alarm monitor started\n"));

  return;
#else /* !LIBOMP_USE_LITHE */

#ifdef KMP_THREAD_ATTR
  if (__kmp_monitor_stksize == 0) {
    __kmp_monitor_stksize = KMP_DEFAULT_MONITOR_STKSIZE;
    auto_adj_size = TRUE;
  }
  status = pthread_attr_init(&thread_attr);
  if (status != 0) {
    __kmp_fatal(KMP_MSG(CantInitThreadAttrs), KMP_ERR(status), __kmp_msg_null);
  }
  status = pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);
  if (status != 0) {
    __kmp_fatal(KMP_MSG(CantSetMonitorState), KMP_ERR(status), __kmp_msg_null);
  }

#ifdef _POSIX_THREAD_ATTR_STACKSIZE
  status = pthread_attr_getstacksize(&thread_attr, &size);
  KMP_CHECK_SYSFAIL("pthread_attr_getstacksize", status);
#else
  size = __kmp_sys_min_stksize;
#endif /* _POSIX_THREAD_ATTR_STACKSIZE */
#endif /* KMP_THREAD_ATTR */

  if (__kmp_monitor_stksize == 0) {
    __kmp_monitor_stksize = KMP_DEFAULT_MONITOR_STKSIZE;
  }
  if (__kmp_monitor_stksize < __kmp_sys_min_stksize) {
    __kmp_monitor_stksize = __kmp_sys_min_stksize;
  }

  KA_TRACE(10, ("__kmp_create_monitor: default stacksize = %lu bytes,"
                "requested stacksize = %lu bytes\n",
                size, __kmp_monitor_stksize));

retry:

/* Set stack size for this thread now. */
#ifdef _POSIX_THREAD_ATTR_STACKSIZE
  KA_TRACE(10, ("__kmp_create_monitor: setting stacksize = %lu bytes,",
                __kmp_monitor_stksize));
  status = pthread_attr_setstacksize(&thread_attr, __kmp_monitor_stksize);
  if (status != 0) {
    if (auto_adj_size) {
      __kmp_monitor_stksize *= 2;
      goto retry;
    }
    kmp_msg_t err_code = KMP_ERR(status);
    __kmp_msg(kmp_ms_warning, // should this be fatal?  BB
              KMP_MSG(CantSetMonitorStackSize, (long int)__kmp_monitor_stksize),
              err_code, KMP_HNT(ChangeMonitorStackSize), __kmp_msg_null);
    if (__kmp_generate_warnings == kmp_warnings_off) {
      __kmp_str_free(&err_code.str);
    }
  }
#endif /* _POSIX_THREAD_ATTR_STACKSIZE */

  status =
      pthread_create(&handle, &thread_attr, __kmp_launch_monitor, (void *)th);

  if (status != 0) {
#ifdef _POSIX_THREAD_ATTR_STACKSIZE
    if (status == EINVAL) {
      if (auto_adj_size && (__kmp_monitor_stksize < (size_t)0x40000000)) {
        __kmp_monitor_stksize *= 2;
        goto retry;
      }
      __kmp_fatal(KMP_MSG(CantSetMonitorStackSize, __kmp_monitor_stksize),
                  KMP_ERR(status), KMP_HNT(IncreaseMonitorStackSize),
                  __kmp_msg_null);
    }
    if (status == ENOMEM) {
      __kmp_fatal(KMP_MSG(CantSetMonitorStackSize, __kmp_monitor_stksize),
                  KMP_ERR(status), KMP_HNT(DecreaseMonitorStackSize),
                  __kmp_msg_null);
    }
#endif /* _POSIX_THREAD_ATTR_STACKSIZE */
    if (status == EAGAIN) {
      __kmp_fatal(KMP_MSG(NoResourcesForMonitorThread), KMP_ERR(status),
                  KMP_HNT(DecreaseNumberOfThreadsInUse), __kmp_msg_null);
    }
    KMP_SYSFAIL("pthread_create", status);
  }

  th->th.th_info.ds.ds_thread = handle;

#if KMP_REAL_TIME_FIX
  // Wait for the monitor thread is really started and set its *priority*.
  KMP_DEBUG_ASSERT(sizeof(kmp_uint32) ==
                   sizeof(__kmp_global.g.g_time.dt.t_value));
  __kmp_wait_yield_4((kmp_uint32 volatile *)&__kmp_global.g.g_time.dt.t_value,
                     -1, &__kmp_neq_4, NULL);
#endif // KMP_REAL_TIME_FIX

#ifdef KMP_THREAD_ATTR
  status = pthread_attr_destroy(&thread_attr);
  if (status != 0) {
    kmp_msg_t err_code = KMP_ERR(status);
    __kmp_msg(kmp_ms_warning, KMP_MSG(CantDestroyThreadAttrs), err_code,
              __kmp_msg_null);
    if (__kmp_generate_warnings == kmp_warnings_off) {
      __kmp_str_free(&err_code.str);
    }
  }
#endif

  KMP_MB(); /* Flush all pending memory write invalidates.  */

  KA_TRACE(10, ("__kmp_create_monitor: monitor created %#.8lx\n",
                th->th.th_info.ds.ds_thread));

#endif /* !LIBOMP_USE_LITHE */

} // __kmp_create_monitor
#endif // KMP_USE_MONITOR

void __kmp_exit_thread(int exit_status) {
  pthread_exit((void *)(intptr_t)exit_status);
} // __kmp_exit_thread

#if KMP_USE_MONITOR
#ifdef LIBOMP_USE_LITHE
extern void __kmp_lithe_parlib_monitor_stop(void);
#endif
void __kmp_resume_monitor();

void __kmp_reap_monitor(kmp_info_t *th) {
  int status;
  void *exit_val;

  KA_TRACE(10, ("__kmp_reap_monitor: try to reap monitor thread with handle"
                " %#.8lx\n",
                th->th.th_info.ds.ds_thread));

  // If monitor has been created, its tid and gtid should be KMP_GTID_MONITOR.
  // If both tid and gtid are 0, it means the monitor did not ever start.
  // If both tid and gtid are KMP_GTID_DNE, the monitor has been shut down.
  KMP_DEBUG_ASSERT(th->th.th_info.ds.ds_tid == th->th.th_info.ds.ds_gtid);
  if (th->th.th_info.ds.ds_gtid != KMP_GTID_MONITOR) {
    KA_TRACE(10, ("__kmp_reap_monitor: monitor did not start, returning\n"));
    return;
  }

  KMP_MB(); /* Flush all pending memory write invalidates.  */

#ifdef LIBOMP_USE_LITHE
  __kmp_lithe_parlib_monitor_stop();
  th->th.th_info.ds.ds_tid = KMP_GTID_DNE;
  th->th.th_info.ds.ds_gtid = KMP_GTID_DNE;
  KMP_MB();
  KA_TRACE(10, ("__kmp_reap_monitor: done reaping parlib alarm monitor\n"));
  return;
#endif /* LIBOMP_USE_LITHE */

  /* First, check to see whether the monitor thread exists to wake it up. This
     is to avoid performance problem when the monitor sleeps during
     blocktime-size interval */

  status = pthread_kill(th->th.th_info.ds.ds_thread, 0);
  if (status != ESRCH) {
    __kmp_resume_monitor(); // Wake up the monitor thread
  }
  KA_TRACE(10, ("__kmp_reap_monitor: try to join with monitor\n"));
  status = pthread_join(th->th.th_info.ds.ds_thread, &exit_val);
  if (exit_val != th) {
    __kmp_fatal(KMP_MSG(ReapMonitorError), KMP_ERR(status), __kmp_msg_null);
  }

  th->th.th_info.ds.ds_tid = KMP_GTID_DNE;
  th->th.th_info.ds.ds_gtid = KMP_GTID_DNE;

  KA_TRACE(10, ("__kmp_reap_monitor: done reaping monitor thread with handle"
                " %#.8lx\n",
                th->th.th_info.ds.ds_thread));

  KMP_MB(); /* Flush all pending memory write invalidates.  */
}
#endif // KMP_USE_MONITOR

void __kmp_reap_worker(kmp_info_t *th) {
  int status;
  void *exit_val;

  KMP_MB();

  KA_TRACE(
      10, ("__kmp_reap_worker: try to reap T#%d\n", th->th.th_info.ds.ds_gtid));

#ifdef LIBOMP_USE_LITHE
  if (th->th.th_lithe_ctx) {
    /* The worker self-destroys its lithe context (and calls join_one) from
     * lithe_fork_join_sched_context_exit when it returns from
     * __kmp_launch_worker. Destroying it here as well was a double free that
     * corrupted the fjs zombie list and later ran a context whose TLS had been
     * freed (tls_desc==NULL -> SIGSEGV in __set_tls_desc), seen at OMP>=3.
     * Instead, just JOIN: wait until the worker signals it is done with this
     * kmp_info_t (lithe analogue of pthread_join), then free the handshake. */
    struct kmp_lithe_join_ctl *jc =
        (struct kmp_lithe_join_ctl *)th->th.th_lithe_join_ctl;
    if (jc) {
      lithe_mutex_lock(&jc->mx);
      while (!jc->exited)
        lithe_condvar_wait(&jc->cv, &jc->mx);
      lithe_mutex_unlock(&jc->mx);
      free(jc);
      th->th.th_lithe_join_ctl = NULL;
    }
    th->th.th_lithe_ctx = NULL;
    KA_TRACE(10, ("__kmp_reap_worker: joined lithe context T#%d\n",
                  th->th.th_info.ds.ds_gtid));
  }
#else
  status = pthread_join(th->th.th_info.ds.ds_thread, &exit_val);
#ifdef KMP_DEBUG
  if (status != 0) {
    __kmp_fatal(KMP_MSG(ReapWorkerError), KMP_ERR(status), __kmp_msg_null);
  }
  if (exit_val != th) {
    KA_TRACE(10, ("__kmp_reap_worker: worker T#%d did not reap properly, "
                  "exit_val = %p\n",
                  th->th.th_info.ds.ds_gtid, exit_val));
  }
#endif /* KMP_DEBUG */

  KA_TRACE(10, ("__kmp_reap_worker: done reaping T#%d\n",
                th->th.th_info.ds.ds_gtid));
#endif /* LIBOMP_USE_LITHE */

  KMP_MB();
}

#if KMP_HANDLE_SIGNALS

static void __kmp_null_handler(int signo) {
  //  Do nothing, for doing SIG_IGN-type actions.
} // __kmp_null_handler

static void __kmp_team_handler(int signo) {
  if (__kmp_global.g.g_abort == 0) {
/* Stage 1 signal handler, let's shut down all of the threads */
#ifdef KMP_DEBUG
    __kmp_debug_printf("__kmp_team_handler: caught signal = %d\n", signo);
#endif
    switch (signo) {
    case SIGHUP:
    case SIGINT:
    case SIGQUIT:
    case SIGILL:
    case SIGABRT:
    case SIGFPE:
    case SIGBUS:
    case SIGSEGV:
#ifdef SIGSYS
    case SIGSYS:
#endif
    case SIGTERM:
      if (__kmp_debug_buf) {
        __kmp_dump_debug_buffer();
      }
      KMP_MB(); // Flush all pending memory write invalidates.
      TCW_4(__kmp_global.g.g_abort, signo);
      KMP_MB(); // Flush all pending memory write invalidates.
      TCW_4(__kmp_global.g.g_done, TRUE);
      KMP_MB(); // Flush all pending memory write invalidates.
      break;
    default:
#ifdef KMP_DEBUG
      __kmp_debug_printf("__kmp_team_handler: unknown signal type");
#endif
      break;
    }
  }
} // __kmp_team_handler

static void __kmp_sigaction(int signum, const struct sigaction *act,
                            struct sigaction *oldact) {
  int rc = sigaction(signum, act, oldact);
  KMP_CHECK_SYSFAIL_ERRNO("sigaction", rc);
}

static void __kmp_install_one_handler(int sig, sig_func_t handler_func,
                                      int parallel_init) {
  KMP_MB(); // Flush all pending memory write invalidates.
  KB_TRACE(60,
           ("__kmp_install_one_handler( %d, ..., %d )\n", sig, parallel_init));
  if (parallel_init) {
    struct sigaction new_action;
    struct sigaction old_action;
    new_action.sa_handler = handler_func;
    new_action.sa_flags = 0;
    sigfillset(&new_action.sa_mask);
    __kmp_sigaction(sig, &new_action, &old_action);
    if (old_action.sa_handler == __kmp_sighldrs[sig].sa_handler) {
      sigaddset(&__kmp_sigset, sig);
    } else {
      // Restore/keep user's handler if one previously installed.
      __kmp_sigaction(sig, &old_action, NULL);
    }
  } else {
    // Save initial/system signal handlers to see if user handlers installed.
    __kmp_sigaction(sig, NULL, &__kmp_sighldrs[sig]);
  }
  KMP_MB(); // Flush all pending memory write invalidates.
} // __kmp_install_one_handler

static void __kmp_remove_one_handler(int sig) {
  KB_TRACE(60, ("__kmp_remove_one_handler( %d )\n", sig));
  if (sigismember(&__kmp_sigset, sig)) {
    struct sigaction old;
    KMP_MB(); // Flush all pending memory write invalidates.
    __kmp_sigaction(sig, &__kmp_sighldrs[sig], &old);
    if ((old.sa_handler != __kmp_team_handler) &&
        (old.sa_handler != __kmp_null_handler)) {
      // Restore the users signal handler.
      KB_TRACE(10, ("__kmp_remove_one_handler: oops, not our handler, "
                    "restoring: sig=%d\n",
                    sig));
      __kmp_sigaction(sig, &old, NULL);
    }
    sigdelset(&__kmp_sigset, sig);
    KMP_MB(); // Flush all pending memory write invalidates.
  }
} // __kmp_remove_one_handler

void __kmp_install_signals(int parallel_init) {
  KB_TRACE(10, ("__kmp_install_signals( %d )\n", parallel_init));
  if (__kmp_handle_signals || !parallel_init) {
    // If ! parallel_init, we do not install handlers, just save original
    // handlers. Let us do it even __handle_signals is 0.
    sigemptyset(&__kmp_sigset);
    __kmp_install_one_handler(SIGHUP, __kmp_team_handler, parallel_init);
    __kmp_install_one_handler(SIGINT, __kmp_team_handler, parallel_init);
    __kmp_install_one_handler(SIGQUIT, __kmp_team_handler, parallel_init);
    __kmp_install_one_handler(SIGILL, __kmp_team_handler, parallel_init);
    __kmp_install_one_handler(SIGABRT, __kmp_team_handler, parallel_init);
    __kmp_install_one_handler(SIGFPE, __kmp_team_handler, parallel_init);
    __kmp_install_one_handler(SIGBUS, __kmp_team_handler, parallel_init);
    __kmp_install_one_handler(SIGSEGV, __kmp_team_handler, parallel_init);
#ifdef SIGSYS
    __kmp_install_one_handler(SIGSYS, __kmp_team_handler, parallel_init);
#endif // SIGSYS
    __kmp_install_one_handler(SIGTERM, __kmp_team_handler, parallel_init);
#ifdef SIGPIPE
    __kmp_install_one_handler(SIGPIPE, __kmp_team_handler, parallel_init);
#endif // SIGPIPE
  }
} // __kmp_install_signals

void __kmp_remove_signals(void) {
  int sig;
  KB_TRACE(10, ("__kmp_remove_signals()\n"));
  for (sig = 1; sig < NSIG; ++sig) {
    __kmp_remove_one_handler(sig);
  }
} // __kmp_remove_signals

#endif // KMP_HANDLE_SIGNALS

void __kmp_enable(int new_state) {
#ifdef KMP_CANCEL_THREADS
  int status, old_state;
  status = pthread_setcancelstate(new_state, &old_state);
  KMP_CHECK_SYSFAIL("pthread_setcancelstate", status);
  KMP_DEBUG_ASSERT(old_state == PTHREAD_CANCEL_DISABLE);
#endif
}

void __kmp_disable(int *old_state) {
#ifdef KMP_CANCEL_THREADS
  int status;
  status = pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, old_state);
  KMP_CHECK_SYSFAIL("pthread_setcancelstate", status);
#endif
}

static void __kmp_atfork_prepare(void) {
  __kmp_acquire_bootstrap_lock(&__kmp_initz_lock);
  __kmp_acquire_bootstrap_lock(&__kmp_forkjoin_lock);
}

static void __kmp_atfork_parent(void) {
  __kmp_release_bootstrap_lock(&__kmp_initz_lock);
  __kmp_release_bootstrap_lock(&__kmp_forkjoin_lock);
}

/* Reset the library so execution in the child starts "all over again" with
   clean data structures in initial states.  Don't worry about freeing memory
   allocated by parent, just abandon it to be safe. */
static void __kmp_atfork_child(void) {
  __kmp_release_bootstrap_lock(&__kmp_forkjoin_lock);
  /* TODO make sure this is done right for nested/sibling */
  // ATT:  Memory leaks are here? TODO: Check it and fix.
  /* KMP_ASSERT( 0 ); */

  ++__kmp_fork_count;

#if KMP_AFFINITY_SUPPORTED
#if KMP_OS_LINUX
  // reset the affinity in the child to the initial thread
  // affinity in the parent
  kmp_set_thread_affinity_mask_initial();
#endif
  // Set default not to bind threads tightly in the child (we’re expecting
  // over-subscription after the fork and this can improve things for
  // scripting languages that use OpenMP inside process-parallel code).
  __kmp_affinity_type = affinity_none;
#if OMP_40_ENABLED
  if (__kmp_nested_proc_bind.bind_types != NULL) {
    __kmp_nested_proc_bind.bind_types[0] = proc_bind_false;
  }
#endif // OMP_40_ENABLED
#endif // KMP_AFFINITY_SUPPORTED

  __kmp_init_runtime = FALSE;
#if KMP_USE_MONITOR
  __kmp_init_monitor = 0;
#endif
  __kmp_init_parallel = FALSE;
  __kmp_init_middle = FALSE;
  __kmp_init_serial = FALSE;
  TCW_4(__kmp_init_gtid, FALSE);
  __kmp_init_common = FALSE;

  TCW_4(__kmp_init_user_locks, FALSE);
#if !KMP_USE_DYNAMIC_LOCK
  __kmp_user_lock_table.used = 1;
  __kmp_user_lock_table.allocated = 0;
  __kmp_user_lock_table.table = NULL;
  __kmp_lock_blocks = NULL;
#endif

  __kmp_all_nth = 0;
  TCW_4(__kmp_nth, 0);

  __kmp_thread_pool = NULL;
  __kmp_thread_pool_insert_pt = NULL;
  __kmp_team_pool = NULL;

  /* Must actually zero all the *cache arguments passed to __kmpc_threadprivate
     here so threadprivate doesn't use stale data */
  KA_TRACE(10, ("__kmp_atfork_child: checking cache address list %p\n",
                __kmp_threadpriv_cache_list));

  while (__kmp_threadpriv_cache_list != NULL) {

    if (*__kmp_threadpriv_cache_list->addr != NULL) {
      KC_TRACE(50, ("__kmp_atfork_child: zeroing cache at address %p\n",
                    &(*__kmp_threadpriv_cache_list->addr)));

      *__kmp_threadpriv_cache_list->addr = NULL;
    }
    __kmp_threadpriv_cache_list = __kmp_threadpriv_cache_list->next;
  }

  __kmp_init_runtime = FALSE;

  /* reset statically initialized locks */
  __kmp_init_bootstrap_lock(&__kmp_initz_lock);
  __kmp_init_bootstrap_lock(&__kmp_stdio_lock);
  __kmp_init_bootstrap_lock(&__kmp_console_lock);
  __kmp_init_bootstrap_lock(&__kmp_task_team_lock);

#if USE_ITT_BUILD
  __kmp_itt_reset(); // reset ITT's global state
#endif /* USE_ITT_BUILD */

  /* This is necessary to make sure no stale data is left around */
  /* AC: customers complain that we use unsafe routines in the atfork
     handler. Mathworks: dlsym() is unsafe. We call dlsym and dlopen
     in dynamic_link when check the presence of shared tbbmalloc library.
     Suggestion is to make the library initialization lazier, similar
     to what done for __kmpc_begin(). */
  // TODO: synchronize all static initializations with regular library
  //       startup; look at kmp_global.cpp and etc.
  //__kmp_internal_begin ();
}

void __kmp_register_atfork(void) {
  if (__kmp_need_register_atfork) {
    int status = pthread_atfork(__kmp_atfork_prepare, __kmp_atfork_parent,
                                __kmp_atfork_child);
    KMP_CHECK_SYSFAIL("pthread_atfork", status);
    __kmp_need_register_atfork = FALSE;
  }
}

void __kmp_suspend_initialize(void) {
  int status;
  status = pthread_mutexattr_init(&__kmp_suspend_mutex_attr);
  KMP_CHECK_SYSFAIL("pthread_mutexattr_init", status);
  status = pthread_condattr_init(&__kmp_suspend_cond_attr);
  KMP_CHECK_SYSFAIL("pthread_condattr_init", status);
}

static void __kmp_suspend_initialize_thread(kmp_info_t *th) {
  ANNOTATE_HAPPENS_AFTER(&th->th.th_suspend_init_count);
  if (th->th.th_suspend_init_count <= __kmp_fork_count) {
    /* this means we haven't initialized the suspension pthread objects for this
       thread in this instance of the process */
    int status;
    status = pthread_cond_init(&th->th.th_suspend_cv.c_cond,
                               &__kmp_suspend_cond_attr);
    KMP_CHECK_SYSFAIL("pthread_cond_init", status);
    status = pthread_mutex_init(&th->th.th_suspend_mx.m_mutex,
                                &__kmp_suspend_mutex_attr);
    KMP_CHECK_SYSFAIL("pthread_mutex_init", status);
    *(volatile int *)&th->th.th_suspend_init_count = __kmp_fork_count + 1;
    ANNOTATE_HAPPENS_BEFORE(&th->th.th_suspend_init_count);
  }
}

void __kmp_suspend_uninitialize_thread(kmp_info_t *th) {
  if (th->th.th_suspend_init_count > __kmp_fork_count) {
    /* this means we have initialize the suspension pthread objects for this
       thread in this instance of the process */
    int status;

    status = pthread_cond_destroy(&th->th.th_suspend_cv.c_cond);
    if (status != 0 && status != EBUSY) {
      KMP_SYSFAIL("pthread_cond_destroy", status);
    }
    status = pthread_mutex_destroy(&th->th.th_suspend_mx.m_mutex);
    if (status != 0 && status != EBUSY) {
      KMP_SYSFAIL("pthread_mutex_destroy", status);
    }
    --th->th.th_suspend_init_count;
    KMP_DEBUG_ASSERT(th->th.th_suspend_init_count == __kmp_fork_count);
  }
}

/* This routine puts the calling thread to sleep after setting the
   sleep bit for the indicated flag variable to true. */
template <class C>
static inline void __kmp_suspend_template(int th_gtid, C *flag) {
  KMP_TIME_DEVELOPER_PARTITIONED_BLOCK(USER_suspend);
  kmp_info_t *th = __kmp_threads[th_gtid];
  int status;
  typename C::flag_t old_spin;

  KF_TRACE(30, ("__kmp_suspend_template: T#%d enter for flag = %p\n", th_gtid,
                flag->get()));

#ifdef LIBOMP_USE_LITHE
  /* Cooperative block: serialize the sleep-bit decision and the wait/signal
   * with a per-thread lithe mutex (mirrors the pthread mutex/condvar protocol
   * below) so there is no lost wakeup, then truly BLOCK on a lithe condvar.
   * Blocking returns the hart to the parent scheduler, letting MPI progress
   * and other OpenMP contexts run while this thread waits -- the composition
   * Lithe exists to provide. The old implementation spin-yielded, which kept
   * the hart and forced an epoll netpoll on every yield. */
  {
    __kmp_lithe_inner_spin_config();
    struct kmp_lithe_suspend_ctl *ctl = __kmp_lithe_suspend_ctl(th);
    lithe_mutex_lock(&ctl->mx);
    old_spin = flag->set_sleeping();
    if (flag->done_check_val(old_spin)) {
      flag->unset_sleeping();
      lithe_mutex_unlock(&ctl->mx);
      return;
    }
    TCW_PTR(th->th.th_sleep_loc, (void *)flag);
    /* WARM VCORES: park in this vcore's warm slot (atomic unlock of ctl->mx,
     * relock on resume -- same contract as lithe_condvar_wait) so the next fork
     * resumes this worker IN PLACE with a single go-flag write + one batched
     * hart_request, no per-worker condvar/enqueue/steal. The mutex + sleep-bit
     * protocol is unchanged, so there is no lost wakeup vs __kmp_resume_template.
     * LITHE_OMP_WARM_VCORES=0 falls back to the per-worker lithe condvar. */
    if (__kmp_lithe_warm_vcores && th->th.th_lithe_ctx != NULL) {
      while (flag->is_sleeping()) {
        lithe_fork_join_park_warm(&ctl->mx);
      }
    } else {
      while (flag->is_sleeping()) {
        lithe_condvar_wait(&ctl->cv, &ctl->mx);
      }
    }
    TCW_PTR(th->th.th_sleep_loc, NULL);
    lithe_mutex_unlock(&ctl->mx);
  }
  KF_TRACE(30, ("__kmp_suspend_template: T#%d exit (lithe block)\n", th_gtid));
  return;
#endif /* LIBOMP_USE_LITHE */

  __kmp_suspend_initialize_thread(th);

  status = pthread_mutex_lock(&th->th.th_suspend_mx.m_mutex);
  KMP_CHECK_SYSFAIL("pthread_mutex_lock", status);

  KF_TRACE(10, ("__kmp_suspend_template: T#%d setting sleep bit for spin(%p)\n",
                th_gtid, flag->get()));

  /* TODO: shouldn't this use release semantics to ensure that
     __kmp_suspend_initialize_thread gets called first? */
  old_spin = flag->set_sleeping();

  KF_TRACE(5, ("__kmp_suspend_template: T#%d set sleep bit for spin(%p)==%x,"
               " was %x\n",
               th_gtid, flag->get(), flag->load(), old_spin));

  if (flag->done_check_val(old_spin)) {
    old_spin = flag->unset_sleeping();
    KF_TRACE(5, ("__kmp_suspend_template: T#%d false alarm, reset sleep bit "
                 "for spin(%p)\n",
                 th_gtid, flag->get()));
  } else {
    /* Encapsulate in a loop as the documentation states that this may
       "with low probability" return when the condition variable has
       not been signaled or broadcast */
    int deactivated = FALSE;
    TCW_PTR(th->th.th_sleep_loc, (void *)flag);

    while (flag->is_sleeping()) {
#ifdef DEBUG_SUSPEND
      char buffer[128];
      __kmp_suspend_count++;
      __kmp_print_cond(buffer, &th->th.th_suspend_cv);
      __kmp_printf("__kmp_suspend_template: suspending T#%d: %s\n", th_gtid,
                   buffer);
#endif
      // Mark the thread as no longer active (only in the first iteration of the
      // loop).
      if (!deactivated) {
        th->th.th_active = FALSE;
        if (th->th.th_active_in_pool) {
          th->th.th_active_in_pool = FALSE;
          KMP_ATOMIC_DEC(&__kmp_thread_pool_active_nth);
          KMP_DEBUG_ASSERT(TCR_4(__kmp_thread_pool_active_nth) >= 0);
        }
        deactivated = TRUE;
      }

#if USE_SUSPEND_TIMEOUT
      struct timespec now;
      struct timeval tval;
      int msecs;

      status = gettimeofday(&tval, NULL);
      KMP_CHECK_SYSFAIL_ERRNO("gettimeofday", status);
      TIMEVAL_TO_TIMESPEC(&tval, &now);

      msecs = (4 * __kmp_dflt_blocktime) + 200;
      now.tv_sec += msecs / 1000;
      now.tv_nsec += (msecs % 1000) * 1000;

      KF_TRACE(15, ("__kmp_suspend_template: T#%d about to perform "
                    "pthread_cond_timedwait\n",
                    th_gtid));
      status = pthread_cond_timedwait(&th->th.th_suspend_cv.c_cond,
                                      &th->th.th_suspend_mx.m_mutex, &now);
#else
      KF_TRACE(15, ("__kmp_suspend_template: T#%d about to perform"
                    " pthread_cond_wait\n",
                    th_gtid));
      status = pthread_cond_wait(&th->th.th_suspend_cv.c_cond,
                                 &th->th.th_suspend_mx.m_mutex);
#endif

      if ((status != 0) && (status != EINTR) && (status != ETIMEDOUT)) {
        KMP_SYSFAIL("pthread_cond_wait", status);
      }
#ifdef KMP_DEBUG
      if (status == ETIMEDOUT) {
        if (flag->is_sleeping()) {
          KF_TRACE(100,
                   ("__kmp_suspend_template: T#%d timeout wakeup\n", th_gtid));
        } else {
          KF_TRACE(2, ("__kmp_suspend_template: T#%d timeout wakeup, sleep bit "
                       "not set!\n",
                       th_gtid));
        }
      } else if (flag->is_sleeping()) {
        KF_TRACE(100,
                 ("__kmp_suspend_template: T#%d spurious wakeup\n", th_gtid));
      }
#endif
    } // while

    // Mark the thread as active again (if it was previous marked as inactive)
    if (deactivated) {
      th->th.th_active = TRUE;
      if (TCR_4(th->th.th_in_pool)) {
        KMP_ATOMIC_INC(&__kmp_thread_pool_active_nth);
        th->th.th_active_in_pool = TRUE;
      }
    }
  }
#ifdef DEBUG_SUSPEND
  {
    char buffer[128];
    __kmp_print_cond(buffer, &th->th.th_suspend_cv);
    __kmp_printf("__kmp_suspend_template: T#%d has awakened: %s\n", th_gtid,
                 buffer);
  }
#endif

  status = pthread_mutex_unlock(&th->th.th_suspend_mx.m_mutex);
  KMP_CHECK_SYSFAIL("pthread_mutex_unlock", status);
  KF_TRACE(30, ("__kmp_suspend_template: T#%d exit\n", th_gtid));
}

void __kmp_suspend_32(int th_gtid, kmp_flag_32 *flag) {
  __kmp_suspend_template(th_gtid, flag);
}
void __kmp_suspend_64(int th_gtid, kmp_flag_64 *flag) {
  __kmp_suspend_template(th_gtid, flag);
}
void __kmp_suspend_oncore(int th_gtid, kmp_flag_oncore *flag) {
  __kmp_suspend_template(th_gtid, flag);
}

/* This routine signals the thread specified by target_gtid to wake up
   after setting the sleep bit indicated by the flag argument to FALSE.
   The target thread must already have called __kmp_suspend_template() */
template <class C>
static inline void __kmp_resume_template(int target_gtid, C *flag) {
  KMP_TIME_DEVELOPER_PARTITIONED_BLOCK(USER_resume);
  kmp_info_t *th = __kmp_threads[target_gtid];
  int status;

#ifdef KMP_DEBUG
  int gtid = TCR_4(__kmp_init_gtid) ? __kmp_get_gtid() : -1;
#endif

  KF_TRACE(30, ("__kmp_resume_template: T#%d wants to wakeup T#%d enter\n",
                gtid, target_gtid));
  KMP_DEBUG_ASSERT(gtid != target_gtid);

#ifdef LIBOMP_USE_LITHE
  /* Wake a cooperatively-blocked waiter. The per-thread lithe mutex serializes
   * this with __kmp_suspend_template's sleep-bit check + condvar_wait, so there
   * is no lost wakeup: if the target is mid-suspend it still holds the mutex
   * (released only once it is parked on the condvar queue), so we block here
   * until it is safely waiting, then signal. Signaling an empty queue is a
   * harmless no-op for a target that never actually blocked. */
  {
    __kmp_lithe_inner_spin_config();
    struct kmp_lithe_suspend_ctl *ctl = __kmp_lithe_suspend_ctl(th);
    lithe_mutex_lock(&ctl->mx);
    if (!flag) {
      flag = (C *)CCAST(void *, th->th.th_sleep_loc);
    }
    if (flag) {
      flag->unset_sleeping();
    }
    TCW_PTR(th->th.th_sleep_loc, NULL);
    /* WARM VCORES: resume the warm-parked worker in place (go-flag + one
     * batched hart_request). Must mirror the suspend-side path exactly so a
     * warm-parked worker is never left on the condvar (and vice versa). Called
     * under ctl->mx with the master in the worker's sticky top FJ sched, so
     * lithe_fork_join_resume_warm's hart_request targets the right scheduler. */
    if (__kmp_lithe_warm_vcores && th->th.th_lithe_ctx != NULL) {
      lithe_fork_join_resume_warm((lithe_context_t *)th->th.th_lithe_ctx);
    } else {
      lithe_condvar_signal(&ctl->cv);
    }
    lithe_mutex_unlock(&ctl->mx);
  }
  KF_TRACE(30, ("__kmp_resume_template: T#%d wakeup T#%d (lithe)\n",
                gtid, target_gtid));
  return;
#endif /* LIBOMP_USE_LITHE */

  __kmp_suspend_initialize_thread(th);

  status = pthread_mutex_lock(&th->th.th_suspend_mx.m_mutex);
  KMP_CHECK_SYSFAIL("pthread_mutex_lock", status);

  if (!flag) { // coming from __kmp_null_resume_wrapper
    flag = (C *)CCAST(void *, th->th.th_sleep_loc);
  }

  // First, check if the flag is null or its type has changed. If so, someone
  // else woke it up.
  if (!flag || flag->get_type() != flag->get_ptr_type()) { // get_ptr_type
    // simply shows what
    // flag was cast to
    KF_TRACE(5, ("__kmp_resume_template: T#%d exiting, thread T#%d already "
                 "awake: flag(%p)\n",
                 gtid, target_gtid, NULL));
    status = pthread_mutex_unlock(&th->th.th_suspend_mx.m_mutex);
    KMP_CHECK_SYSFAIL("pthread_mutex_unlock", status);
    return;
  } else { // if multiple threads are sleeping, flag should be internally
    // referring to a specific thread here
    typename C::flag_t old_spin = flag->unset_sleeping();
    if (!flag->is_sleeping_val(old_spin)) {
      KF_TRACE(5, ("__kmp_resume_template: T#%d exiting, thread T#%d already "
                   "awake: flag(%p): "
                   "%u => %u\n",
                   gtid, target_gtid, flag->get(), old_spin, flag->load()));
      status = pthread_mutex_unlock(&th->th.th_suspend_mx.m_mutex);
      KMP_CHECK_SYSFAIL("pthread_mutex_unlock", status);
      return;
    }
    KF_TRACE(5, ("__kmp_resume_template: T#%d about to wakeup T#%d, reset "
                 "sleep bit for flag's loc(%p): "
                 "%u => %u\n",
                 gtid, target_gtid, flag->get(), old_spin, flag->load()));
  }
  TCW_PTR(th->th.th_sleep_loc, NULL);

#ifdef DEBUG_SUSPEND
  {
    char buffer[128];
    __kmp_print_cond(buffer, &th->th.th_suspend_cv);
    __kmp_printf("__kmp_resume_template: T#%d resuming T#%d: %s\n", gtid,
                 target_gtid, buffer);
  }
#endif
  status = pthread_cond_signal(&th->th.th_suspend_cv.c_cond);
  KMP_CHECK_SYSFAIL("pthread_cond_signal", status);
  status = pthread_mutex_unlock(&th->th.th_suspend_mx.m_mutex);
  KMP_CHECK_SYSFAIL("pthread_mutex_unlock", status);
  KF_TRACE(30, ("__kmp_resume_template: T#%d exiting after signaling wake up"
                " for T#%d\n",
                gtid, target_gtid));
}

void __kmp_resume_32(int target_gtid, kmp_flag_32 *flag) {
  __kmp_resume_template(target_gtid, flag);
}
void __kmp_resume_64(int target_gtid, kmp_flag_64 *flag) {
  __kmp_resume_template(target_gtid, flag);
}
void __kmp_resume_oncore(int target_gtid, kmp_flag_oncore *flag) {
  __kmp_resume_template(target_gtid, flag);
}

#if KMP_USE_MONITOR
void __kmp_resume_monitor() {
  KMP_TIME_DEVELOPER_PARTITIONED_BLOCK(USER_resume);
  int status;
#ifdef KMP_DEBUG
  int gtid = TCR_4(__kmp_init_gtid) ? __kmp_get_gtid() : -1;
  KF_TRACE(30, ("__kmp_resume_monitor: T#%d wants to wakeup T#%d enter\n", gtid,
                KMP_GTID_MONITOR));
  KMP_DEBUG_ASSERT(gtid != KMP_GTID_MONITOR);
#endif
  status = pthread_mutex_lock(&__kmp_wait_mx.m_mutex);
  KMP_CHECK_SYSFAIL("pthread_mutex_lock", status);
#ifdef DEBUG_SUSPEND
  {
    char buffer[128];
    __kmp_print_cond(buffer, &__kmp_wait_cv.c_cond);
    __kmp_printf("__kmp_resume_monitor: T#%d resuming T#%d: %s\n", gtid,
                 KMP_GTID_MONITOR, buffer);
  }
#endif
  status = pthread_cond_signal(&__kmp_wait_cv.c_cond);
  KMP_CHECK_SYSFAIL("pthread_cond_signal", status);
  status = pthread_mutex_unlock(&__kmp_wait_mx.m_mutex);
  KMP_CHECK_SYSFAIL("pthread_mutex_unlock", status);
  KF_TRACE(30, ("__kmp_resume_monitor: T#%d exiting after signaling wake up"
                " for T#%d\n",
                gtid, KMP_GTID_MONITOR));
}
#endif // KMP_USE_MONITOR

void __kmp_yield(int cond) {
#ifdef LIBOMP_USE_LITHE
  (void)cond;
  lithe_context_yield();
#else
  if (!cond)
    return;
#if KMP_USE_MONITOR
  if (!__kmp_yielding_on)
    return;
#else
  if (__kmp_yield_cycle && !KMP_YIELD_NOW())
    return;
#endif
  sched_yield();
#endif /* LIBOMP_USE_LITHE */
}

void __kmp_gtid_set_specific(int gtid) {
  if (__kmp_init_gtid) {
    int status;
    status = pthread_setspecific(__kmp_gtid_threadprivate_key,
                                 (void *)(intptr_t)(gtid + 1));
    KMP_CHECK_SYSFAIL("pthread_setspecific", status);
  } else {
    KA_TRACE(50, ("__kmp_gtid_set_specific: runtime shutdown, returning\n"));
  }
}

int __kmp_gtid_get_specific() {
  int gtid;
  if (!__kmp_init_gtid) {
    KA_TRACE(50, ("__kmp_gtid_get_specific: runtime shutdown, returning "
                  "KMP_GTID_SHUTDOWN\n"));
    return KMP_GTID_SHUTDOWN;
  }
  gtid = (int)(size_t)pthread_getspecific(__kmp_gtid_threadprivate_key);
  if (gtid == 0) {
    gtid = KMP_GTID_DNE;
  } else {
    gtid--;
  }
  KA_TRACE(50, ("__kmp_gtid_get_specific: key:%d gtid:%d\n",
                __kmp_gtid_threadprivate_key, gtid));
  return gtid;
}

double __kmp_read_cpu_time(void) {
  /*clock_t   t;*/
  struct tms buffer;

  /*t =*/times(&buffer);

  return (buffer.tms_utime + buffer.tms_cutime) / (double)CLOCKS_PER_SEC;
}

int __kmp_read_system_info(struct kmp_sys_info *info) {
  int status;
  struct rusage r_usage;

  memset(info, 0, sizeof(*info));

  status = getrusage(RUSAGE_SELF, &r_usage);
  KMP_CHECK_SYSFAIL_ERRNO("getrusage", status);

  // The maximum resident set size utilized (in kilobytes)
  info->maxrss = r_usage.ru_maxrss;
  // The number of page faults serviced without any I/O
  info->minflt = r_usage.ru_minflt;
  // The number of page faults serviced that required I/O
  info->majflt = r_usage.ru_majflt;
  // The number of times a process was "swapped" out of memory
  info->nswap = r_usage.ru_nswap;
  // The number of times the file system had to perform input
  info->inblock = r_usage.ru_inblock;
  // The number of times the file system had to perform output
  info->oublock = r_usage.ru_oublock;
  // The number of times a context switch was voluntarily
  info->nvcsw = r_usage.ru_nvcsw;
  // The number of times a context switch was forced
  info->nivcsw = r_usage.ru_nivcsw;

  return (status != 0);
}

void __kmp_read_system_time(double *delta) {
  double t_ns;
  struct timeval tval;
  struct timespec stop;
  int status;

  status = gettimeofday(&tval, NULL);
  KMP_CHECK_SYSFAIL_ERRNO("gettimeofday", status);
  TIMEVAL_TO_TIMESPEC(&tval, &stop);
  t_ns = TS2NS(stop) - TS2NS(__kmp_sys_timer_data.start);
  *delta = (t_ns * 1e-9);
}

void __kmp_clear_system_time(void) {
  struct timeval tval;
  int status;
  status = gettimeofday(&tval, NULL);
  KMP_CHECK_SYSFAIL_ERRNO("gettimeofday", status);
  TIMEVAL_TO_TIMESPEC(&tval, &__kmp_sys_timer_data.start);
}

static int __kmp_get_xproc(void) {

  int r = 0;

#if KMP_OS_LINUX || KMP_OS_FREEBSD || KMP_OS_NETBSD

  r = sysconf(_SC_NPROCESSORS_ONLN);

#elif KMP_OS_DARWIN

  // Bug C77011 High "OpenMP Threads and number of active cores".

  // Find the number of available CPUs.
  kern_return_t rc;
  host_basic_info_data_t info;
  mach_msg_type_number_t num = HOST_BASIC_INFO_COUNT;
  rc = host_info(mach_host_self(), HOST_BASIC_INFO, (host_info_t)&info, &num);
  if (rc == 0 && num == HOST_BASIC_INFO_COUNT) {
    // Cannot use KA_TRACE() here because this code works before trace support
    // is initialized.
    r = info.avail_cpus;
  } else {
    KMP_WARNING(CantGetNumAvailCPU);
    KMP_INFORM(AssumedNumCPU);
  }

#else

#error "Unknown or unsupported OS."

#endif

  return r > 0 ? r : 2; /* guess value of 2 if OS told us 0 */

} // __kmp_get_xproc

int __kmp_read_from_file(char const *path, char const *format, ...) {
  int result;
  va_list args;

  va_start(args, format);
  FILE *f = fopen(path, "rb");
  if (f == NULL)
    return 0;
  result = vfscanf(f, format, args);
  fclose(f);

  return result;
}

void __kmp_runtime_initialize(void) {
  int status;
  pthread_mutexattr_t mutex_attr;
  pthread_condattr_t cond_attr;

  if (__kmp_init_runtime) {
    return;
  }

#if (KMP_ARCH_X86 || KMP_ARCH_X86_64)
  if (!__kmp_cpuinfo.initialized) {
    __kmp_query_cpuid(&__kmp_cpuinfo);
  }
#endif /* KMP_ARCH_X86 || KMP_ARCH_X86_64 */

  __kmp_xproc = __kmp_get_xproc();

  if (sysconf(_SC_THREADS)) {

    /* Query the maximum number of threads */
    __kmp_sys_max_nth = sysconf(_SC_THREAD_THREADS_MAX);
    if (__kmp_sys_max_nth == -1) {
      /* Unlimited threads for NPTL */
      __kmp_sys_max_nth = INT_MAX;
    } else if (__kmp_sys_max_nth <= 1) {
      /* Can't tell, just use PTHREAD_THREADS_MAX */
      __kmp_sys_max_nth = KMP_MAX_NTH;
    }

    /* Query the minimum stack size */
    __kmp_sys_min_stksize = sysconf(_SC_THREAD_STACK_MIN);
    if (__kmp_sys_min_stksize <= 1) {
      __kmp_sys_min_stksize = KMP_MIN_STKSIZE;
    }
  }

  /* Set up minimum number of threads to switch to TLS gtid */
  __kmp_tls_gtid_min = KMP_TLS_GTID_MIN;

  status = pthread_key_create(&__kmp_gtid_threadprivate_key,
                              __kmp_internal_end_dest);
  KMP_CHECK_SYSFAIL("pthread_key_create", status);
  status = pthread_mutexattr_init(&mutex_attr);
  KMP_CHECK_SYSFAIL("pthread_mutexattr_init", status);
  status = pthread_mutex_init(&__kmp_wait_mx.m_mutex, &mutex_attr);
  KMP_CHECK_SYSFAIL("pthread_mutex_init", status);
  status = pthread_condattr_init(&cond_attr);
  KMP_CHECK_SYSFAIL("pthread_condattr_init", status);
  status = pthread_cond_init(&__kmp_wait_cv.c_cond, &cond_attr);
  KMP_CHECK_SYSFAIL("pthread_cond_init", status);
#if USE_ITT_BUILD
  __kmp_itt_initialize();
#endif /* USE_ITT_BUILD */

  __kmp_init_runtime = TRUE;
}

void __kmp_runtime_destroy(void) {
  int status;

  if (!__kmp_init_runtime) {
    return; // Nothing to do.
  }

#if USE_ITT_BUILD
  __kmp_itt_destroy();
#endif /* USE_ITT_BUILD */

  status = pthread_key_delete(__kmp_gtid_threadprivate_key);
  KMP_CHECK_SYSFAIL("pthread_key_delete", status);

  status = pthread_mutex_destroy(&__kmp_wait_mx.m_mutex);
  if (status != 0 && status != EBUSY) {
    KMP_SYSFAIL("pthread_mutex_destroy", status);
  }
  status = pthread_cond_destroy(&__kmp_wait_cv.c_cond);
  if (status != 0 && status != EBUSY) {
    KMP_SYSFAIL("pthread_cond_destroy", status);
  }
#if KMP_AFFINITY_SUPPORTED
  __kmp_affinity_uninitialize();
#endif

  __kmp_init_runtime = FALSE;
}

/* Put the thread to sleep for a time period */
/* NOTE: not currently used anywhere */
void __kmp_thread_sleep(int millis) { sleep((millis + 500) / 1000); }

/* Calculate the elapsed wall clock time for the user */
void __kmp_elapsed(double *t) {
  int status;
#ifdef FIX_SGI_CLOCK
  struct timespec ts;

  status = clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  KMP_CHECK_SYSFAIL_ERRNO("clock_gettime", status);
  *t =
      (double)ts.tv_nsec * (1.0 / (double)KMP_NSEC_PER_SEC) + (double)ts.tv_sec;
#else
  struct timeval tv;

  status = gettimeofday(&tv, NULL);
  KMP_CHECK_SYSFAIL_ERRNO("gettimeofday", status);
  *t =
      (double)tv.tv_usec * (1.0 / (double)KMP_USEC_PER_SEC) + (double)tv.tv_sec;
#endif
}

/* Calculate the elapsed wall clock tick for the user */
void __kmp_elapsed_tick(double *t) { *t = 1 / (double)CLOCKS_PER_SEC; }

/* Return the current time stamp in nsec */
kmp_uint64 __kmp_now_nsec() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return KMP_NSEC_PER_SEC * t.tv_sec + 1000 * t.tv_usec;
}

#if KMP_ARCH_X86 || KMP_ARCH_X86_64
/* Measure clock ticks per millisecond */
void __kmp_initialize_system_tick() {
  kmp_uint64 delay = 100000; // 50~100 usec on most machines.
  kmp_uint64 nsec = __kmp_now_nsec();
  kmp_uint64 goal = __kmp_hardware_timestamp() + delay;
  kmp_uint64 now;
  while ((now = __kmp_hardware_timestamp()) < goal)
    ;
  __kmp_ticks_per_msec =
      (kmp_uint64)(1e6 * (delay + (now - goal)) / (__kmp_now_nsec() - nsec));
}
#endif

/* Determine whether the given address is mapped into the current address
   space. */

int __kmp_is_address_mapped(void *addr) {

  int found = 0;
  int rc;

#if KMP_OS_LINUX || KMP_OS_FREEBSD

  /* On Linux* OS, read the /proc/<pid>/maps pseudo-file to get all the address
     ranges mapped into the address space. */

  char *name = __kmp_str_format("/proc/%d/maps", getpid());
  FILE *file = NULL;

  file = fopen(name, "r");
  KMP_ASSERT(file != NULL);

  for (;;) {

    void *beginning = NULL;
    void *ending = NULL;
    char perms[5];

    rc = fscanf(file, "%p-%p %4s %*[^\n]\n", &beginning, &ending, perms);
    if (rc == EOF) {
      break;
    }
    KMP_ASSERT(rc == 3 &&
               KMP_STRLEN(perms) == 4); // Make sure all fields are read.

    // Ending address is not included in the region, but beginning is.
    if ((addr >= beginning) && (addr < ending)) {
      perms[2] = 0; // 3th and 4th character does not matter.
      if (strcmp(perms, "rw") == 0) {
        // Memory we are looking for should be readable and writable.
        found = 1;
      }
      break;
    }
  }

  // Free resources.
  fclose(file);
  KMP_INTERNAL_FREE(name);

#elif KMP_OS_DARWIN

  /* On OS X*, /proc pseudo filesystem is not available. Try to read memory
     using vm interface. */

  int buffer;
  vm_size_t count;
  rc = vm_read_overwrite(
      mach_task_self(), // Task to read memory of.
      (vm_address_t)(addr), // Address to read from.
      1, // Number of bytes to be read.
      (vm_address_t)(&buffer), // Address of buffer to save read bytes in.
      &count // Address of var to save number of read bytes in.
      );
  if (rc == 0) {
    // Memory successfully read.
    found = 1;
  }

#elif KMP_OS_FREEBSD || KMP_OS_NETBSD

  // FIXME(FreeBSD, NetBSD): Implement this
  found = 1;

#else

#error "Unknown or unsupported OS"

#endif

  return found;

} // __kmp_is_address_mapped

#ifdef USE_LOAD_BALANCE

#if KMP_OS_DARWIN

// The function returns the rounded value of the system load average
// during given time interval which depends on the value of
// __kmp_load_balance_interval variable (default is 60 sec, other values
// may be 300 sec or 900 sec).
// It returns -1 in case of error.
int __kmp_get_load_balance(int max) {
  double averages[3];
  int ret_avg = 0;

  int res = getloadavg(averages, 3);

  // Check __kmp_load_balance_interval to determine which of averages to use.
  // getloadavg() may return the number of samples less than requested that is
  // less than 3.
  if (__kmp_load_balance_interval < 180 && (res >= 1)) {
    ret_avg = averages[0]; // 1 min
  } else if ((__kmp_load_balance_interval >= 180 &&
              __kmp_load_balance_interval < 600) &&
             (res >= 2)) {
    ret_avg = averages[1]; // 5 min
  } else if ((__kmp_load_balance_interval >= 600) && (res == 3)) {
    ret_avg = averages[2]; // 15 min
  } else { // Error occurred
    return -1;
  }

  return ret_avg;
}

#else // Linux* OS

// The fuction returns number of running (not sleeping) threads, or -1 in case
// of error. Error could be reported if Linux* OS kernel too old (without
// "/proc" support). Counting running threads stops if max running threads
// encountered.
int __kmp_get_load_balance(int max) {
  static int permanent_error = 0;
  static int glb_running_threads = 0; // Saved count of the running threads for
  // the thread balance algortihm
  static double glb_call_time = 0; /* Thread balance algorithm call time */

  int running_threads = 0; // Number of running threads in the system.

  DIR *proc_dir = NULL; // Handle of "/proc/" directory.
  struct dirent *proc_entry = NULL;

  kmp_str_buf_t task_path; // "/proc/<pid>/task/<tid>/" path.
  DIR *task_dir = NULL; // Handle of "/proc/<pid>/task/<tid>/" directory.
  struct dirent *task_entry = NULL;
  int task_path_fixed_len;

  kmp_str_buf_t stat_path; // "/proc/<pid>/task/<tid>/stat" path.
  int stat_file = -1;
  int stat_path_fixed_len;

  int total_processes = 0; // Total number of processes in system.
  int total_threads = 0; // Total number of threads in system.

  double call_time = 0.0;

  __kmp_str_buf_init(&task_path);
  __kmp_str_buf_init(&stat_path);

  __kmp_elapsed(&call_time);

  if (glb_call_time &&
      (call_time - glb_call_time < __kmp_load_balance_interval)) {
    running_threads = glb_running_threads;
    goto finish;
  }

  glb_call_time = call_time;

  // Do not spend time on scanning "/proc/" if we have a permanent error.
  if (permanent_error) {
    running_threads = -1;
    goto finish;
  }

  if (max <= 0) {
    max = INT_MAX;
  }

  // Open "/proc/" directory.
  proc_dir = opendir("/proc");
  if (proc_dir == NULL) {
    // Cannot open "/prroc/". Probably the kernel does not support it. Return an
    // error now and in subsequent calls.
    running_threads = -1;
    permanent_error = 1;
    goto finish;
  }

  // Initialize fixed part of task_path. This part will not change.
  __kmp_str_buf_cat(&task_path, "/proc/", 6);
  task_path_fixed_len = task_path.used; // Remember number of used characters.

  proc_entry = readdir(proc_dir);
  while (proc_entry != NULL) {
    // Proc entry is a directory and name starts with a digit. Assume it is a
    // process' directory.
    if (proc_entry->d_type == DT_DIR && isdigit(proc_entry->d_name[0])) {

      ++total_processes;
      // Make sure init process is the very first in "/proc", so we can replace
      // strcmp( proc_entry->d_name, "1" ) == 0 with simpler total_processes ==
      // 1. We are going to check that total_processes == 1 => d_name == "1" is
      // true (where "=>" is implication). Since C++ does not have => operator,
      // let us replace it with its equivalent: a => b == ! a || b.
      KMP_DEBUG_ASSERT(total_processes != 1 ||
                       strcmp(proc_entry->d_name, "1") == 0);

      // Construct task_path.
      task_path.used = task_path_fixed_len; // Reset task_path to "/proc/".
      __kmp_str_buf_cat(&task_path, proc_entry->d_name,
                        KMP_STRLEN(proc_entry->d_name));
      __kmp_str_buf_cat(&task_path, "/task", 5);

      task_dir = opendir(task_path.str);
      if (task_dir == NULL) {
        // Process can finish between reading "/proc/" directory entry and
        // opening process' "task/" directory. So, in general case we should not
        // complain, but have to skip this process and read the next one. But on
        // systems with no "task/" support we will spend lot of time to scan
        // "/proc/" tree again and again without any benefit. "init" process
        // (its pid is 1) should exist always, so, if we cannot open
        // "/proc/1/task/" directory, it means "task/" is not supported by
        // kernel. Report an error now and in the future.
        if (strcmp(proc_entry->d_name, "1") == 0) {
          running_threads = -1;
          permanent_error = 1;
          goto finish;
        }
      } else {
        // Construct fixed part of stat file path.
        __kmp_str_buf_clear(&stat_path);
        __kmp_str_buf_cat(&stat_path, task_path.str, task_path.used);
        __kmp_str_buf_cat(&stat_path, "/", 1);
        stat_path_fixed_len = stat_path.used;

        task_entry = readdir(task_dir);
        while (task_entry != NULL) {
          // It is a directory and name starts with a digit.
          if (proc_entry->d_type == DT_DIR && isdigit(task_entry->d_name[0])) {
            ++total_threads;

            // Consruct complete stat file path. Easiest way would be:
            //  __kmp_str_buf_print( & stat_path, "%s/%s/stat", task_path.str,
            //  task_entry->d_name );
            // but seriae of __kmp_str_buf_cat works a bit faster.
            stat_path.used =
                stat_path_fixed_len; // Reset stat path to its fixed part.
            __kmp_str_buf_cat(&stat_path, task_entry->d_name,
                              KMP_STRLEN(task_entry->d_name));
            __kmp_str_buf_cat(&stat_path, "/stat", 5);

            // Note: Low-level API (open/read/close) is used. High-level API
            // (fopen/fclose)  works ~ 30 % slower.
            stat_file = open(stat_path.str, O_RDONLY);
            if (stat_file == -1) {
              // We cannot report an error because task (thread) can terminate
              // just before reading this file.
            } else {
              /* Content of "stat" file looks like:
                 24285 (program) S ...

                 It is a single line (if program name does not include funny
                 symbols). First number is a thread id, then name of executable
                 file name in paretheses, then state of the thread. We need just
                 thread state.

                 Good news: Length of program name is 15 characters max. Longer
                 names are truncated.

                 Thus, we need rather short buffer: 15 chars for program name +
                 2 parenthesis, + 3 spaces + ~7 digits of pid = 37.

                 Bad news: Program name may contain special symbols like space,
                 closing parenthesis, or even new line. This makes parsing
                 "stat" file not 100 % reliable. In case of fanny program names
                 parsing may fail (report incorrect thread state).

                 Parsing "status" file looks more promissing (due to different
                 file structure and escaping special symbols) but reading and
                 parsing of "status" file works slower.
                  -- ln
              */
              char buffer[65];
              int len;
              len = read(stat_file, buffer, sizeof(buffer) - 1);
              if (len >= 0) {
                buffer[len] = 0;
                // Using scanf:
                //     sscanf( buffer, "%*d (%*s) %c ", & state );
                // looks very nice, but searching for a closing parenthesis
                // works a bit faster.
                char *close_parent = strstr(buffer, ") ");
                if (close_parent != NULL) {
                  char state = *(close_parent + 2);
                  if (state == 'R') {
                    ++running_threads;
                    if (running_threads >= max) {
                      goto finish;
                    }
                  }
                }
              }
              close(stat_file);
              stat_file = -1;
            }
          }
          task_entry = readdir(task_dir);
        }
        closedir(task_dir);
        task_dir = NULL;
      }
    }
    proc_entry = readdir(proc_dir);
  }

  // There _might_ be a timing hole where the thread executing this
  // code get skipped in the load balance, and running_threads is 0.
  // Assert in the debug builds only!!!
  KMP_DEBUG_ASSERT(running_threads > 0);
  if (running_threads <= 0) {
    running_threads = 1;
  }

finish: // Clean up and exit.
  if (proc_dir != NULL) {
    closedir(proc_dir);
  }
  __kmp_str_buf_free(&task_path);
  if (task_dir != NULL) {
    closedir(task_dir);
  }
  __kmp_str_buf_free(&stat_path);
  if (stat_file != -1) {
    close(stat_file);
  }

  glb_running_threads = running_threads;

  return running_threads;

} // __kmp_get_load_balance

#endif // KMP_OS_DARWIN

#endif // USE_LOAD_BALANCE

#if !(KMP_ARCH_X86 || KMP_ARCH_X86_64 || KMP_MIC ||                            \
      ((KMP_OS_LINUX || KMP_OS_DARWIN) && KMP_ARCH_AARCH64) || KMP_ARCH_PPC64)

// we really only need the case with 1 argument, because CLANG always build
// a struct of pointers to shared variables referenced in the outlined function
int __kmp_invoke_microtask(microtask_t pkfn, int gtid, int tid, int argc,
                           void *p_argv[]
#if OMPT_SUPPORT
                           ,
                           void **exit_frame_ptr
#endif
                           ) {
#if OMPT_SUPPORT
  *exit_frame_ptr = OMPT_GET_FRAME_ADDRESS(0);
#endif

  switch (argc) {
  default:
    fprintf(stderr, "Too many args to microtask: %d!\n", argc);
    fflush(stderr);
    exit(-1);
  case 0:
    (*pkfn)(&gtid, &tid);
    break;
  case 1:
    (*pkfn)(&gtid, &tid, p_argv[0]);
    break;
  case 2:
    (*pkfn)(&gtid, &tid, p_argv[0], p_argv[1]);
    break;
  case 3:
    (*pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2]);
    break;
  case 4:
    (*pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2], p_argv[3]);
    break;
  case 5:
    (*pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2], p_argv[3], p_argv[4]);
    break;
  case 6:
    (*pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2], p_argv[3], p_argv[4],
            p_argv[5]);
    break;
  case 7:
    (*pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2], p_argv[3], p_argv[4],
            p_argv[5], p_argv[6]);
    break;
  case 8:
    (*pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2], p_argv[3], p_argv[4],
            p_argv[5], p_argv[6], p_argv[7]);
    break;
  case 9:
    (*pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2], p_argv[3], p_argv[4],
            p_argv[5], p_argv[6], p_argv[7], p_argv[8]);
    break;
  case 10:
    (*pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2], p_argv[3], p_argv[4],
            p_argv[5], p_argv[6], p_argv[7], p_argv[8], p_argv[9]);
    break;
  case 11:
    (*pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2], p_argv[3], p_argv[4],
            p_argv[5], p_argv[6], p_argv[7], p_argv[8], p_argv[9], p_argv[10]);
    break;
  case 12:
    (*pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2], p_argv[3], p_argv[4],
            p_argv[5], p_argv[6], p_argv[7], p_argv[8], p_argv[9], p_argv[10],
            p_argv[11]);
    break;
  case 13:
    (*pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2], p_argv[3], p_argv[4],
            p_argv[5], p_argv[6], p_argv[7], p_argv[8], p_argv[9], p_argv[10],
            p_argv[11], p_argv[12]);
    break;
  case 14:
    (*pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2], p_argv[3], p_argv[4],
            p_argv[5], p_argv[6], p_argv[7], p_argv[8], p_argv[9], p_argv[10],
            p_argv[11], p_argv[12], p_argv[13]);
    break;
  case 15:
    (*pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2], p_argv[3], p_argv[4],
            p_argv[5], p_argv[6], p_argv[7], p_argv[8], p_argv[9], p_argv[10],
            p_argv[11], p_argv[12], p_argv[13], p_argv[14]);
    break;
  }

#if OMPT_SUPPORT
  *exit_frame_ptr = 0;
#endif

  return 1;
}

#endif

// end of file //
