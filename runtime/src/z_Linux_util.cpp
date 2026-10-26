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
#include <lithe/sched_trace.h>
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
/* Fork/join-barrier hart donation (2026-10-15, LITHE_OMP_FORK_YIELD, default
 * 0 for A/B). The inner barrier above donates its hart to a RUNNABLE-but-
 * hartless peer; the fork/join (region-boundary) hot-spin did not. In the
 * hosted model the peer is usually this worker's own MASTER (a logical MPI
 * rank) queued on the same OMP fork-join sched: hosted miniFE RPH=4 x OMP=2 on
 * VCORE_LIMIT=3 stalled in make_local_matrix with each OMP child sched showing
 * harts=1 runnable=1 and its hart running a worker in __kmp_wait_template at
 * bs_forkjoin_barrier. The master cannot reach the next fork (or progress MPI)
 * until the worker exhausts its hot budget AND KMP_BLOCKTIME (200 ms) -- per
 * region gap. KMP_BLOCKTIME=0 made the run complete, which is the confirming
 * A/B; this is the targeted version that keeps the hot spin when nobody waits.
 *
 * NEGATIVE RESULT, measured the same day (login node, RPH=4 x OMP=2, NX16,
 * LITHE_OMP_FLAT=1): runs complete but CG takes 17-21 s vs 0.3-0.5 s with
 * KMP_BLOCKTIME=0 and this OFF -- donating on every spin iteration is a yield
 * storm. Leave it off; use KMP_BLOCKTIME=0 with LITHE_OMP_FLAT=1 instead. */
int __kmp_lithe_fork_yield = 0;            /* LITHE_OMP_FORK_YIELD      */
/* DEMAND BLOCK (2026-10-16, LITHE_OMP_DEMAND_BLOCK; unset = on iff
 * LITHE_OMP_FLAT=1, explicit 0/1 wins). The two region-boundary policies above
 * are both wrong under hart scarcity, in opposite directions:
 *   - the default (no hot-team, hybrid binary) spins KMP_BLOCKTIME (200 ms)
 *     holding the hart before it parks, while its own master sits RUNNABLE
 *     and hartless on the same runqueue -> flat mode "slow, not hung"
 *     (RPH=4 x OMP=2 on 3 harts: CG 0.4 s with KMP_BLOCKTIME=0, far worse
 *     without);
 *   - LITHE_OMP_FORK_YIELD donates the hart but stays RUNNABLE, so the worker
 *     is re-dispatched, finds the flag still clear, yields again: a FIFO
 *     storm with every worker in it (CG 17-21 s).
 * Demand block is the third option: spin hot while NOBODY on this scheduler
 * is runnable-but-hartless (one atomic load per iteration, same probe as the
 * inner barrier), and the moment somebody is, go straight to the cooperative
 * block -- skip the blocktime poll -- so the hart changes hands exactly once
 * and the worker is NOT runnable until the next fork releases it. On an
 * exclusive node with harts >= contexts the runqueue is empty between regions
 * and this degenerates to the hot spin (memory-write wake, no park); on a
 * scarce node it degenerates to KMP_BLOCKTIME=0 without having to set it. */
int __kmp_lithe_demand_block = -1;         /* LITHE_OMP_DEMAND_BLOCK    */
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
/* FAST PARK (LITHE_OMP_FAST_PARK, 2026-10-22, GAPS C21). Oversubscribed teams
 * (OMP threads > harts) block at every fork/join barrier (demand block), so
 * each region pays one suspend + one resume per worker. Through the per-thread
 * lithe mutex + condvar that was ~0.4 us per wake on the master's serial
 * release path (mutex lock, condvar MCS lock, unblock, mutex unlock, plus the
 * woken worker re-taking the mutex): 8 threads on 4 harts ran a tiny region in
 * 5-7 us vs BOLT 2.2 us. Fast park replaces both with one state word: the
 * worker publishes PARKING, re-checks the sleep bit, and blocks; the block
 * callback CASes PARKING->PARKED (or unblocks itself if a wake already landed);
 * the waker CASes PARKED->RUNNING and unblocks, or PARKING/RUNNING->WAKE (a
 * token the waiter consumes instead of blocking). The waiter clears its own
 * sleep bit on the way out, so a NULL-flag resume (__kmp_null_resume_wrapper:
 * task wakeups, teardown) never dereferences th_sleep_loc, which may point at
 * a returned waiter's stack wrapper -- the mutex used to cover that. A stale
 * WAKE costs one early return from suspend; the caller re-checks its flag.
 * No lock on either side. */
int __kmp_lithe_fast_park = 1;           /* LITHE_OMP_FAST_PARK (default ON, =0 restores the mutex+condvar) */
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
  e = getenv("LITHE_OMP_FAST_PARK");
  if (e && *e)
    __kmp_lithe_fast_park = atoi(e);
  e = getenv("LITHE_OMP_INNER_YIELD");
  if (e && *e)
    __kmp_lithe_inner_yield = atoi(e);
  e = getenv("LITHE_OMP_FORK_YIELD");
  if (e && *e)
    __kmp_lithe_fork_yield = atoi(e);
  e = getenv("LITHE_OMP_DEMAND_BLOCK");
  if (e && *e)
    __kmp_lithe_demand_block = atoi(e);
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
/* Policy hart budget. With PARLIB_SPARE_VCORES, max_vcores() is the vcore CAPACITY
 * (VCORE_LIMIT + compensation spares, docs/GAPS.md B42); "how many harts may this
 * team have" is parlib_vcore_limit(). Weak so an older libparlib still links. */
extern "C" int parlib_vcore_limit(void) __attribute__((weak));
static int __kmp_lithe_hart_limit(void) {
  return &parlib_vcore_limit ? parlib_vcore_limit() : (int)max_vcores();
}

/* LITHE_OMP_REAP_YIELD (default 0): __kmp_free_team's reap wait yields the hart
 * via KMP_YIELD_SPIN instead of pure-pausing (kmp_runtime.cpp, GAPS B56). */
extern "C" int __kmp_lithe_reap_yield_on(void) {
  static int on = -1;
  if (__builtin_expect(on < 0, 0)) {
    const char *e = getenv("LITHE_OMP_REAP_YIELD");
    on = (e && *e) ? atoi(e) != 0 : 0;
  }
  return on;
}

/* ------------------------------------------------------------------------
 * DE-SPIN (2026-10-24, .build_logs/spin_census/DESPIN_RULES.md). A waiter on a
 * Lithe context should hold its hart only while doing useful work. Measured on
 * cores 32-35, 4 harts, interleaved, .build_logs/despin/openmp/ (ab_libomp*.csv,
 * ab_misc.csv; medians, n=5 unless noted):
 *   LITHE_OMP_QLOCK_BLOCK     (default 1) queuing-lock waiters (omp critical,
 *                             omp_set_lock: the default lock kind) block on
 *                             their th_spin_here word; the releaser wakes them.
 *                             With OVERSUB_HARTS, n=10: critical 8 thr 0.249 ->
 *                             0.139 s, omp_lock 8 thr 0.185 -> 0.154; 4 thr
 *                             0.055/0.056; rest parity or better (below).
 *   LITHE_OMP_OVERSUB_HARTS   (default 1) "threads > procs" tests
 *                             (KMP_OVERSUB_PROCS: TAS/ticket/queuing locks,
 *                             wait_yield, dispatch, tasking) use the hart limit,
 *                             not the OS cpuset, so waiters yield at once when
 *                             threads outnumber harts. n=10 with QLOCK: tasks 8 thr
 *                             0.083 -> 0.068 s, 16 thr 0.106 -> 0.088; ordered 8
 *                             thr 16.0 -> 13.5 ms, 16 thr 16.4 -> 13.7; nested
 *                             8x4 306 -> 294 us; region cost 4 thr 1.77/1.80 us,
 *                             8 thr 5.90/5.89; doacross 8/16 thr 18.0/15.8,
 *                             25.6/21.7 ms; B46 tasks 16 thr 0/20 hangs.
 *   LITHE_OMP_NOSPIN          (default 0) bitmask, __kmp_wait_template:
 *                             1 = inner barrier parks at once, 2 = fork/join
 *                             barrier parks at once (no hot-team budget, no
 *                             KMP_BLOCKTIME poll). NEGATIVE for bit 2: region
 *                             cost 4 thr 1.77 -> 3.52 us (the hot spin bridges
 *                             the re-fork; the demand block already parks as
 *                             soon as a peer is hartless), 8 thr 5.83 -> 5.60;
 *                             bit 1 neutral (inner barrier already donates);
 *                             nested/lock/tasks unchanged. Kept as an A/B arm.
 *   LITHE_OMP_DOACROSS_BLOCK  (default 1 since B61) __kmpc_doacross_wait blocks
 *                             on a futex slot keyed by (flags word, bit). Was
 *                             NEGATIVE with address-only slots (every post woke
 *                             all waiters on the word): 15.1/15.3/20.6 ->
 *                             16.5/17.8/22.2 ms. Keyed, n=12, 4/8/16 thr: yield
 *                             15.05/15.60/21.30 -> block 14.50/14.80/15.65 ms.
 *   LITHE_OMP_ORDERED_BLOCK   (default 1 since B61) `omp ordered` waits (dispatch
 *                             and __kmp_parallel_deo) block on a slot keyed by
 *                             the awaited iteration / tid; posts wake (old,new].
 *                             Address-only slots were NEGATIVE (13.2/13.2/14.1 ->
 *                             14.6/18.4/21.7 ms). Keyed, n=12, static,1 4/8/16 thr:
 *                             yield 13.90/14.00/14.60 -> block 13.05/13.35/13.90;
 *                             dynamic,7: 11.70/12.40/13.10 -> 12.00/12.25/12.65.
 *                             .build_logs/despin/openmp/ab_keyed2.csv.
 * ------------------------------------------------------------------------ */
static int __kmp_lithe_env_flag(const char *name, int dflt) {
  const char *e = getenv(name);
  return (e && *e) ? atoi(e) : dflt;
}
/* GAPS B60 (2026-10-25): under LITHE_OMP_FLAT=0 each team runs in its own child
 * scheduler and demand block is off (it reads the child's runnable count; the
 * waiting masters are runnable in the PARENT). Workers at the fork barrier then
 * hot-spin the whole blocktime on every hart while the other masters starve: two
 * tenants of 4 threads on 4 harts "hung" (200 ms per region; KMP_BLOCKTIME=1 ->
 * 2-3 ms rounds). Parking at the fork/join barrier at once (bit 2) is what
 * libgomp's GOMP_LITHE_OWN_SCHED does by default, so nested mode now defaults to
 * 2. With it, nested mode is the low-tail mode: cotenant 2x4 on 4 harts, n=6,
 * p99 143 us vs flat 206, noise 0.0065 vs 0.035, mean at ideal (135.5);
 * NOSPIN=2 under flat changes nothing (206). .build_logs/b60/ablate.txt. */
extern "C" int __kmp_lithe_flat_on(void);
extern "C" int __kmp_lithe_nospin_mask(void) {
  static int m = -1;
  if (__builtin_expect(m < 0, 0))
    m = __kmp_lithe_env_flag("LITHE_OMP_NOSPIN", __kmp_lithe_flat_on() ? 0 : 2);
  return m;
}
extern "C" int __kmp_lithe_avail_procs(void) {
  static int on = -1;
  if (__builtin_expect(on < 0, 0))
    on = __kmp_lithe_env_flag("LITHE_OMP_OVERSUB_HARTS", 1) != 0;
  if (on) {
    int h = __kmp_lithe_hart_limit();
    if (h > 0)
      return h;
  }
  return __kmp_avail_proc ? __kmp_avail_proc : __kmp_xproc;
}
extern "C" int __kmp_lithe_qlock_block_on(void) {
  static int on = -1;
  if (__builtin_expect(on < 0, 0))
    on = __kmp_lithe_env_flag("LITHE_OMP_QLOCK_BLOCK", 1) != 0;
  return on;
}
extern "C" int __kmp_lithe_doacross_block_on(void) {
  static int on = -1;
  if (__builtin_expect(on < 0, 0))
    on = __kmp_lithe_env_flag("LITHE_OMP_DOACROSS_BLOCK", 1) != 0;
  return on;
}
/* liblithe's futex emulation (src/lithe/src/futex.c): per-address wait queue,
 * value re-checked under its lock inside lithe_context_block. */
extern "C" int __kmp_lithe_futex_wait_(int *, int) __asm__("futex_wait")
    __attribute__((weak));
extern "C" int __kmp_lithe_futex_wake_some_(int *, int)
    __asm__("futex_wake_some") __attribute__((weak));
extern "C" int __kmp_lithe_may_block(void) {
  return &__kmp_lithe_futex_wait_ != NULL && lithe_can_block();
}
/* Block until *w != busy. *parked is published before the re-check so the
 * waker (store, full fence, load *parked) cannot miss us. */
extern "C" void __kmp_lithe_word_block(volatile kmp_uint32 *w, kmp_uint32 busy,
                                       volatile kmp_int32 *parked) {
  __atomic_store_n(parked, 1, __ATOMIC_SEQ_CST);
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  while (__atomic_load_n(w, __ATOMIC_ACQUIRE) == busy)
    __kmp_lithe_futex_wait_((int *)w, (int)busy);
  __atomic_store_n(parked, 0, __ATOMIC_RELAXED);
}
extern "C" void __kmp_lithe_word_wake(volatile kmp_uint32 *w, int n) {
  if (&__kmp_lithe_futex_wake_some_ != NULL)
    __kmp_lithe_futex_wake_some_((int *)w, n);
}
/* GAPS B61 (2026-10-25): ticket locks (bootstrap, forkjoin, atomic, user ticket locks) block
 * instead of spinning. Sleepers are counted per hashed lock address so an uncontended release
 * costs one load. Waiter: count, re-check now_serving, futex_wait on it (liblithe re-checks the
 * value under its queue lock). Releaser: store now_serving, full fence, load the count, wake all
 * (ticket order: each waiter checks its own ticket). Where the context cannot block
 * (constructors, TSD destructors, vcore context) the caller keeps the old spin. */
static volatile kmp_int32 __kmp_lithe_tk_sleepers[64];
static inline volatile kmp_int32 *__kmp_lithe_tk_slot(void *p) {
  return &__kmp_lithe_tk_sleepers[((uintptr_t)p >> 6) & 63];
}
extern "C" int __kmp_lithe_lock_block_on(void) {
  static int on = -1;
  if (__builtin_expect(on < 0, 0))
    on = __kmp_lithe_env_flag("LITHE_OMP_LOCK_BLOCK", 1) != 0;
  return on && __kmp_lithe_may_block();
}
extern "C" void __kmp_lithe_ticket_wait(std::atomic_uint *ns, kmp_uint32 ticket) {
  volatile kmp_int32 *slot = __kmp_lithe_tk_slot((void *)ns);
  for (;;) {
    kmp_uint32 cur = ns->load(std::memory_order_acquire);
    if (cur == ticket)
      return;
    __atomic_add_fetch(slot, 1, __ATOMIC_SEQ_CST);
    cur = ns->load(std::memory_order_seq_cst);
    if (cur != ticket)
      __kmp_lithe_futex_wait_((int *)ns, (int)cur);
    __atomic_sub_fetch(slot, 1, __ATOMIC_SEQ_CST);
  }
}
extern "C" void __kmp_lithe_ticket_wake(std::atomic_uint *ns) {
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  if (__atomic_load_n(__kmp_lithe_tk_slot((void *)ns), __ATOMIC_SEQ_CST) > 0 &&
      &__kmp_lithe_futex_wake_some_ != NULL)
    __kmp_lithe_futex_wake_some_((int *)ns, 0x7fffffff);
}
/* Doacross: the futex words are a fixed table of 64 {seq, waiters} slots
 * indexed by the awaited flags word's address, so the set of futex addresses
 * is bounded (liblithe's futex keeps every address it has seen on one linear
 * list that never shrinks, src/lithe/src/futex.c get_futex_list; a word per
 * flags word made the Lithe libgomp equivalent quadratic). A post wakes its
 * slot only when the slot has waiters (one load otherwise); a collision is a
 * spurious wake. */
struct kmp_lithe_dx_slot {
  volatile kmp_int32 seq;
  volatile kmp_int32 waiters;
  char pad[56];
};
static struct kmp_lithe_dx_slot __kmp_lithe_dx_slots[64]
    __attribute__((aligned(64)));
static inline struct kmp_lithe_dx_slot *
__kmp_lithe_dx_slot(volatile kmp_uint32 *w) {
  uintptr_t h = (uintptr_t)w >> 2;
  return &__kmp_lithe_dx_slots[(h ^ (h >> 6) ^ (h >> 12)) & 63];
}
/* Generic slot wait/wake for any word whose setter calls __kmp_lithe_slot_wake
 * after its store: arm (waiters++, fence, read seq), re-check, sleep. */
extern "C" kmp_int32 __kmp_lithe_slot_arm(volatile void *w) {
  struct kmp_lithe_dx_slot *sl = __kmp_lithe_dx_slot((volatile kmp_uint32 *)w);
  __atomic_add_fetch(&sl->waiters, 1, __ATOMIC_SEQ_CST);
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  return __atomic_load_n(&sl->seq, __ATOMIC_ACQUIRE);
}
extern "C" void __kmp_lithe_slot_disarm(volatile void *w) {
  struct kmp_lithe_dx_slot *sl = __kmp_lithe_dx_slot((volatile kmp_uint32 *)w);
  __atomic_sub_fetch(&sl->waiters, 1, __ATOMIC_RELAXED);
}
extern "C" void __kmp_lithe_slot_sleep(volatile void *w, kmp_int32 sq) {
  struct kmp_lithe_dx_slot *sl = __kmp_lithe_dx_slot((volatile kmp_uint32 *)w);
  __kmp_lithe_futex_wait_((int *)&sl->seq, (int)sq);
  __atomic_sub_fetch(&sl->waiters, 1, __ATOMIC_RELAXED);
}
extern "C" void __kmp_lithe_slot_wake(volatile void *w) {
  struct kmp_lithe_dx_slot *sl = __kmp_lithe_dx_slot((volatile kmp_uint32 *)w);
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  if (__atomic_load_n(&sl->waiters, __ATOMIC_RELAXED) != 0) {
    __atomic_add_fetch(&sl->seq, 1, __ATOMIC_RELEASE);
    __kmp_lithe_word_wake((volatile kmp_uint32 *)&sl->seq, INT_MAX);
  }
}
/* GAPS B61 (2026-10-25) KEYED slots. The address-only slots above wake EVERY
 * waiter on the word: `omp ordered` with 8/16 threads on 4 harts woke all 15
 * waiters per iteration and 14 went back to sleep, which is why the blocking
 * ordered/doacross waits measured slower than the yield loop (DE-SPIN notes).
 * A keyed slot is indexed by (word, the value or bit the waiter needs), so a
 * post wakes only the waiter whose turn it is; a hash collision is a spurious
 * wake (the caller re-checks). Waiters arm/re-check/sleep exactly as above. */
#define KMP_LITHE_KS_N 256
static struct kmp_lithe_dx_slot __kmp_lithe_ks_slots[KMP_LITHE_KS_N]
    __attribute__((aligned(64)));
static inline struct kmp_lithe_dx_slot *__kmp_lithe_ks(volatile void *w,
                                                       kmp_uint32 key) {
  uintptr_t h = ((uintptr_t)w >> 2) ^ ((uintptr_t)key * 0x9E3779B1u);
  h ^= h >> 13;
  h ^= h >> 7;
  return &__kmp_lithe_ks_slots[h & (KMP_LITHE_KS_N - 1)];
}
extern "C" kmp_int32 __kmp_lithe_kslot_arm(volatile void *w, kmp_uint32 key) {
  struct kmp_lithe_dx_slot *sl = __kmp_lithe_ks(w, key);
  __atomic_add_fetch(&sl->waiters, 1, __ATOMIC_SEQ_CST);
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  return __atomic_load_n(&sl->seq, __ATOMIC_ACQUIRE);
}
extern "C" void __kmp_lithe_kslot_disarm(volatile void *w, kmp_uint32 key) {
  __atomic_sub_fetch(&__kmp_lithe_ks(w, key)->waiters, 1, __ATOMIC_RELAXED);
}
extern "C" void __kmp_lithe_kslot_sleep(volatile void *w, kmp_uint32 key,
                                        kmp_int32 sq) {
  struct kmp_lithe_dx_slot *sl = __kmp_lithe_ks(w, key);
  __kmp_lithe_futex_wait_((int *)&sl->seq, (int)sq);
  __atomic_sub_fetch(&sl->waiters, 1, __ATOMIC_RELAXED);
}
static inline void __kmp_lithe_ks_wake_slot(struct kmp_lithe_dx_slot *sl) {
  if (__atomic_load_n(&sl->waiters, __ATOMIC_RELAXED) != 0) {
    __atomic_add_fetch(&sl->seq, 1, __ATOMIC_RELEASE);
    __kmp_lithe_word_wake((volatile kmp_uint32 *)&sl->seq, INT_MAX);
  }
}
extern "C" void __kmp_lithe_kslot_wake(volatile void *w, kmp_uint32 key) {
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  __kmp_lithe_ks_wake_slot(__kmp_lithe_ks(w, key));
}
/* A counter moved from `from` to `to`: wake every key in (from, to]. Ordered
 * waiters wait for >= their lower bound, which a chunked post can step over;
 * a range wider than the table wakes every armed slot. */
extern "C" void __kmp_lithe_kslot_wake_range(volatile void *w, kmp_uint64 from,
                                             kmp_uint64 to) {
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  if (to - from >= KMP_LITHE_KS_N) {
    for (int i = 0; i < KMP_LITHE_KS_N; i++)
      __kmp_lithe_ks_wake_slot(&__kmp_lithe_ks_slots[i]);
    return;
  }
  for (kmp_uint64 k = from + 1; k <= to && k != from; k++)
    __kmp_lithe_ks_wake_slot(__kmp_lithe_ks(w, (kmp_uint32)k));
}
extern "C" void __kmp_lithe_dx_wait(volatile kmp_uint32 *w, kmp_uint32 bit) {
  kmp_int32 sq = __kmp_lithe_kslot_arm(w, bit);
  if (__atomic_load_n(w, __ATOMIC_ACQUIRE) & bit)
    __kmp_lithe_kslot_disarm(w, bit);
  else
    __kmp_lithe_kslot_sleep(w, bit, sq);
}
extern "C" void __kmp_lithe_dx_posted(volatile kmp_uint32 *w, kmp_uint32 bit) {
  __kmp_lithe_kslot_wake(w, bit);
}
/* LITHE_OMP_ORDERED_BLOCK (default 1): `#pragma omp ordered` waiters block on
 * the slot of sh->u.s.ordered_iteration (kmp_dispatch.h __kmp_lithe_ordered_wait);
 * the three ordered_iteration increments wake it. */
extern "C" int __kmp_lithe_ordered_block_on(void) {
  static int on = -1;
  if (__builtin_expect(on < 0, 0))
    on = __kmp_lithe_env_flag("LITHE_OMP_ORDERED_BLOCK", 1) != 0;
  return on;
}

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
  return __kmp_lithe_hart_limit() >= nproc + reserve;
}

/* OVERSUBSCRIBED LINEAR GATHER (LITHE_OMP_OVERSUB_LINEAR_GATHER, 2026-10-22,
 * GAPS C21). With more team threads than harts, every join-barrier waiter
 * demand-blocks. The default hyper gather makes intermediate parents (tid 4
 * for 5..7) wait on their children too, so each tree level adds a
 * block -> unblock -> dispatch round on the critical path (trace: fork->last
 * worker start 5 us at 8 threads on 4 harts vs 0.5 us at 4). A linear gather
 * leaves only the master waiting. With LITHE_OMP_FAST_PARK: LULESH 16 threads
 * on 4 harts 2150-2270 -> 2540-2570 zones/s, 8 threads within noise, not
 * applied at 1:1. Default ON; =0 restores the configured pattern. Same answer on every thread of a team (team size and the
 * fixed max_vcores()), so the pattern never differs within one barrier. An
 * explicit KMP_FORKJOIN_BARRIER_PATTERN wins. */
extern "C" int __kmp_lithe_fj_linear_gather(int nproc) {
  static int mode = -1;
  if (__builtin_expect(mode < 0, 0)) {
    const char *e = getenv("LITHE_OMP_OVERSUB_LINEAR_GATHER"); /* default ON */
    int m = (e && *e) ? atoi(e) : 1;
    if (getenv("KMP_FORKJOIN_BARRIER_PATTERN"))
      m = 0;
    mode = m;
  }
  return mode && __kmp_lithe_flat_on() && nproc > __kmp_lithe_hart_limit();
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
/* IDLE-AWARE PEER CHECK (LITHE_OMP_IDLE_AWARE, 2026-10-16, GAPS C11 part 2).
 * The three "is a peer runnable-but-hartless" tests in this file and in
 * kmp_wait_release.h used lithe_fork_join_current_runnable_count() > 0, which
 * is also true when a RUNNABLE peer is about to be picked up by an IDLE hart
 * of our own sched. GROMACS PME at NP=2 (MPI_Alltoall inside a parallel
 * region: the primary yields in opal_progress_block while the workers sit at
 * an omp barrier) then made every worker yield on every spin (63k/s), and
 * with a spare hart the four contexts rotated over five harts: 191 vs 296
 * ns/day at 4 harts. lithe_fork_join_current_hartless_runnable() subtracts
 * the idle owned harts. Weak: hosted binaries embed an older liblithe.a
 * until relinked; then fall back to the raw count. */
extern "C" long lithe_fork_join_current_hartless_runnable(void)
    __attribute__((weak));
extern "C" long __kmp_lithe_hartless_peers(void) {
  static int idle_aware = -1;
  if (__builtin_expect(idle_aware < 0, 0)) {
    const char *e = getenv("LITHE_OMP_IDLE_AWARE");
    idle_aware = e ? atoi(e) : 0;
    if (idle_aware && &lithe_fork_join_current_hartless_runnable == NULL)
      idle_aware = 0; /* old liblithe without the symbol */
  }
  if (idle_aware)
    return lithe_fork_join_current_hartless_runnable();
  return lithe_fork_join_current_runnable_count();
}

/* GAPS B61 (LITHE_OMP_ELSEWHERE, default 1; =0 disables): the fork/join demand
 * block above parks a barrier waiter only for a hartless peer on its OWN sched.
 * A flat team's worker otherwise hot-spins up to KMP_BLOCKTIME on a hart that a
 * scheduler elsewhere on the chain (a co-tenant runtime, a sibling) is asking
 * for. liblithe's chain check takes ancestor list locks, so poll it every 64th
 * spin per thread. Weak: hosted binaries embed their liblithe.a. A/B n=10
 * (.build_logs/b60/b61_else.txt): neutral on cotenant flat/own/nested, hosted
 * ranks K=8, miniFE ser/omp/nested and region cost (4 thr 1.825/1.825 us). */
extern "C" int lithe_fork_join_hart_wanted_elsewhere(void) __attribute__((weak));
extern "C" int __kmp_lithe_elsewhere_poll(void) {
  static int on = -1;
  static __thread unsigned n;
  if (__builtin_expect(on < 0, 0))
    on = __kmp_lithe_env_flag("LITHE_OMP_ELSEWHERE", 1) != 0 &&
         &lithe_fork_join_hart_wanted_elsewhere != NULL;
  if (!on || (++n & 63) != 0)
    return 0;
  return lithe_fork_join_hart_wanted_elsewhere();
}
extern "C" int __kmp_lithe_inner_barrier_yield(kmp_info_t *th) {
  (void)th;
  if (!__kmp_lithe_inner_yield)
    return 0; /* A/B: LITHE_OMP_INNER_YIELD=0 restores the old hoard behavior */
  if (__kmp_lithe_hartless_peers() <= 0)
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
  volatile int park;           /* fast park state, KMP_LITHE_PARK_* */
  lithe_context_t *park_ctx;   /* the parked context, valid in PARKED */
};
enum { KMP_LITHE_PARK_RUNNING = 0, KMP_LITHE_PARK_PARKING = 1,
       KMP_LITHE_PARK_PARKED = 2, KMP_LITHE_PARK_WAKE = 3 };

/* Block callback (hart context, the waiter is already off its stack). */
static void __kmp_lithe_fast_park_cb(lithe_context_t *ctx, void *arg) {
  struct kmp_lithe_suspend_ctl *ctl = (struct kmp_lithe_suspend_ctl *)arg;
  ctl->park_ctx = ctx;
  int exp = KMP_LITHE_PARK_PARKING;
  if (!__atomic_compare_exchange_n(&ctl->park, &exp, KMP_LITHE_PARK_PARKED,
                                   false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
    /* a wake landed between the sleep-bit re-check and the block */
    __atomic_store_n(&ctl->park, KMP_LITHE_PARK_RUNNING, __ATOMIC_RELEASE);
    lithe_context_unblock(ctx);
  }
}

/* Waker side (after clearing the sleep bit when it has the flag). */
static void __kmp_lithe_fast_unpark(struct kmp_lithe_suspend_ctl *ctl) {
  int s = __atomic_load_n(&ctl->park, __ATOMIC_SEQ_CST);
  for (;;) {
    if (s == KMP_LITHE_PARK_PARKED) {
      if (__atomic_compare_exchange_n(&ctl->park, &s, KMP_LITHE_PARK_RUNNING,
                                      false, __ATOMIC_ACQ_REL,
                                      __ATOMIC_ACQUIRE)) {
        lithe_context_unblock(ctl->park_ctx);
        return;
      }
    } else if (s == KMP_LITHE_PARK_WAKE) {
      return; /* already woken */
    } else { /* PARKING or RUNNING: leave a token */
      if (__atomic_compare_exchange_n(&ctl->park, &s, KMP_LITHE_PARK_WAKE,
                                      false, __ATOMIC_ACQ_REL,
                                      __ATOMIC_ACQUIRE))
        return;
    }
  }
}

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
  p->park = KMP_LITHE_PARK_RUNNING;
  p->park_ctx = NULL;
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
  /* Set by the worker AFTER its final unlock of `mx`: its last access to the ctl. A reaper that
   * cannot block (a foreign thread -- e.g. a Python/joblib thread whose root is being torn down --
   * or process teardown) spins, and must wait for THIS, not `exited`: `exited` is stored while the
   * worker still holds `mx`, so freeing on `exited` let the worker's lithe_mutex_unlock write
   * freed memory (SIGSEGV at mutex.c:200 under __kmp_lithe_launch_worker, ~1-2 % of
   * .build_logs/ys_crash/omp_foreign_waves runs; docs/GAPS.md B41). */
  volatile int released;
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

/* FLAT MODE (2026-10-15, LITHE_OMP_FLAT=1, default 0 for A/B).
 *
 * The default nests one sticky OMP fork-join scheduler under every context that
 * opens a region. In the hosted model that is one child per logical MPI rank,
 * so harts must move SIDEWAYS between sibling children through the rank
 * scheduler, and every such path has its own starvation mode (ticket-lock
 * convoy, empty-child idle park, worker hot-spin over a runnable master).
 * Login-node sweep: RPH=4 x OMP=2 worked only at VCORE_LIMIT >= ~6, RPH=2 x
 * OMP=2 everywhere; the serial build (all ranks on ONE runqueue) works at
 * VCORE_LIMIT=3.
 *
 * Flat mode runs a region's workers on the scheduler the master is ALREADY in
 * (it must be a fork-join scheduler), so ranks and OMP workers share one FIFO
 * runqueue and a blocked rank's hart serves whatever is runnable next. No
 * sched_enter/exit, no child registration, nothing to release but the workers
 * themselves (omp_lithe_context_release moves this context's team off the
 * rank scheduler so its join_all can complete). */
static int __kmp_lithe_flat = -1;
extern "C" int __kmp_lithe_flat_on(void) {
  if (__kmp_lithe_flat < 0) {
    /* Default ON since 2026-10-19 (GAPS B24): every validated configuration (HPDC
     * suite, integration_check, placement pairs, hosted _omp arm) already set
     * LITHE_OMP_FLAT=1, and the per-region-scheduler mode it replaces livelocks on
     * nested parallelism. LITHE_OMP_FLAT=0 restores the old mode. */
    const char *e = getenv("LITHE_OMP_FLAT");
    __kmp_lithe_flat = (e && e[0] == '0') ? 0 : 1;
  }
  return __kmp_lithe_flat;
}
/* LITHE_OMP_SERIAL_DISP_CACHE (2026-10-21, default 0 for A/B): keep the outermost
 * serialized region's dispatch buffer instead of free + malloc per one-thread
 * region; see __kmpc_end_serialized_parallel (kmp_csupport.cpp). */
static int __kmp_lithe_serial_disp_cache = -1;
extern "C" int __kmp_lithe_serial_disp_cache_on(void) {
  if (__kmp_lithe_serial_disp_cache < 0) {
    const char *e = getenv("LITHE_OMP_SERIAL_DISP_CACHE");
    __kmp_lithe_serial_disp_cache = (e && e[0] == '1') ? 1 : 0;
  }
  return __kmp_lithe_serial_disp_cache;
}
/* Demand block (see __kmp_lithe_demand_block above): explicit env wins, else
 * on exactly when flat mode is. Resolved once; the result is cached. */
extern "C" int __kmp_lithe_demand_block_on(void) {
  if (__kmp_lithe_demand_block < 0) {
    __kmp_lithe_inner_spin_config();
    if (__kmp_lithe_demand_block < 0)
      __kmp_lithe_demand_block = __kmp_lithe_flat_on();
  }
  return __kmp_lithe_demand_block;
}
static lithe_fork_join_sched_t *__kmp_lithe_flat_sched(void) {
  lithe_sched_t *cur = lithe_sched_current();
  if (cur && cur->funcs == &lithe_fork_join_sched_funcs)
    return (lithe_fork_join_sched_t *)cur;
  return NULL;
}
static __thread kmp_team_t *__kmp_lithe_flat_team;

extern "C" lithe_fork_join_sched_t *__kmp_lithe_worker_sched_for_create(void) {
  if (__kmp_lithe_flat_on()) {
    lithe_fork_join_sched_t *f = __kmp_lithe_flat_sched();
    if (f)
      return f;
  }
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

/* Nested parallel regions (team level > 1) run in the CALLER's fork-join
 * scheduler, as every region does in flat mode (2026-10-19, GAPS B24). Before,
 * each outer-team worker took the depth-0 path for its first inner region and
 * created + entered its own sticky child scheduler: 4 outer threads made 4 child
 * schedulers competing for 4 harts, and a context runnable in one that held no
 * harts was invisible to the idle harts of another. bench/alt/omp_nested.c 4x4
 * livelocked (inner masters spinning on __kmp_forkjoin_lock behind a holder
 * parked in a hart-less child; 3 harts idle in hart_enter). 1x4 crashed at
 * teardown. LITHE_OMP_NESTED_SCHED=1 restores the per-region schedulers. */
static int __kmp_lithe_nested_sched = -1;
static int __kmp_lithe_nested_sched_on(void) {
  if (__kmp_lithe_nested_sched < 0) {
    const char *e = getenv("LITHE_OMP_NESTED_SCHED");
    __kmp_lithe_nested_sched = (e && e[0] == '1') ? 1 : 0;
  }
  return __kmp_lithe_nested_sched;
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

  if (team->t.t_level > 1 && !__kmp_lithe_nested_sched_on()) {
    lithe_fork_join_sched_t *f = __kmp_lithe_flat_sched();
    if (f) {
      int runnable_migrated = __kmp_lithe_migrate_workers(team, f, NULL);
      if (runnable_migrated > 0)
        lithe_hart_request(runnable_migrated);
    }
    return;
  }

  if (__kmp_lithe_flat_on()) {
    lithe_fork_join_sched_t *f = __kmp_lithe_flat_sched();
    if (f) {
      int runnable_migrated = __kmp_lithe_migrate_workers(team, f, NULL);
      __kmp_lithe_flat_team = team;
      if (runnable_migrated > 0)
        lithe_hart_request(runnable_migrated);
      return;
    }
  }

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
  if (team->t.t_level > 1 && !__kmp_lithe_nested_sched_on())
    return; /* nested region ran in the caller's scheduler: nothing to unwind */
  if (__kmp_lithe_flat_on() && __kmp_lithe_region_depth <= 0)
    return; /* flat: no per-region scheduler to unwind */
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

/* Process teardown: sticky top FJ is entered once and never exited per region.
 * After workers have been reaped (lithe_context_exit), master must sched_exit
 * to drain idle FJS harts, then destroy the sticky FJ. Otherwise
 * __kmp_reap_team / __kmp_cleanup frees OMP heap while hart_enter still runs
 * on that sched (BabelStream SIGSEGV / exit 139 at OMP>=4). */
extern "C" void __kmp_lithe_shutdown_sticky_fj(void) {
  lithe_fork_join_sched_t *top;
  lithe_sched_t *cur;

  if (!__kmp_lithe_omp_top_fj && !__kmp_lithe_top_sticky &&
      __kmp_lithe_fj_sp <= 0 && __kmp_lithe_region_depth <= 0)
    return;

  /* Nested per-region FJ first (if any remain), then sticky top. */
  while (__kmp_lithe_fj_sp > 1) {
    lithe_fork_join_sched_t *nested =
        __kmp_lithe_fj_stack[__kmp_lithe_fj_sp - 1];
    cur = lithe_sched_current();
    if (nested && cur == (lithe_sched_t *)nested) {
      lithe_sched_exit();
    }
    /* Skip destroy at process teardown (same race as sticky top). */
    __kmp_lithe_fj_stack[__kmp_lithe_fj_sp - 1] = NULL;
    __kmp_lithe_fj_team[__kmp_lithe_fj_sp - 1] = NULL;
    __kmp_lithe_fj_sp--;
  }

  top = __kmp_lithe_omp_top_fj;
  if (top) {
    cur = lithe_sched_current();
    /* Master should still be inside sticky FJ at process end. If not, re-enter
     * so lithe_sched_exit can drain child harts (while(harts) yield). */
    if (cur != (lithe_sched_t *)top) {
      if (cur != NULL)
        lithe_sched_enter((lithe_sched_t *)top);
    }
    cur = lithe_sched_current();
    if (cur == (lithe_sched_t *)top) {
      lithe_sched_exit();
    }
    /* Process teardown: do NOT destroy the sticky FJ. Destroy races idle
     * vcores still finishing hart_return / MCS unlock after sched_exit's
     * drain (SIGSEGV in mcs_pdr_lock). Leaking one FJ at exit is fine. */
  }

  __kmp_lithe_omp_top_fj = NULL;
  __kmp_lithe_top_sticky = 0;
  if (__kmp_lithe_fj_sp > 0) {
    __kmp_lithe_fj_stack[0] = NULL;
    __kmp_lithe_fj_team[0] = NULL;
  }
  __kmp_lithe_fj_sp = 0;
  __kmp_lithe_region_depth = 0;
}


/* PER-CONTEXT RELEASE of the sticky top-level OMP fork-join scheduler.
 * ---------------------------------------------------------------------------
 * The sticky design above is right for a classic MPI+OpenMP process: the master
 * thread lives until exit, so entering the top OMP FJ once and unwinding it in
 * __kmp_internal_end costs nothing and saves a per-region hart drain.
 *
 * It is WRONG for the hosted model, where a logical rank is a transient Lithe
 * CONTEXT. Such a context opens a parallel region, libomp enters the sticky
 * OMP FJ on it and never exits, and then the context RETURNS -- still inside a
 * child scheduler. The child stays registered on its parent with granted=1 and
 * harts=0, so base_child_exited() spins forever waiting for a hart nobody
 * owns and the application's own lithe_sched_exit() never returns. libomp's
 * unwind runs from __kmp_internal_end, i.e. at PROCESS teardown -- strictly
 * after the application has already tried to exit its scheduler. The lifetimes
 * are inverted.
 *
 * Measured 2026-10-14 with a minimal reproducer (.build_logs/ompctx/repro.c):
 * ONE region of ONE thread in ONE context computes the right answer and then
 * hangs in teardown; the no-OpenMP control arm exits cleanly.
 *
 * Since the sticky state is per-uthread TLS (and parlib gives every Lithe
 * context its own TLS), the unwind is per-context too. A context that used
 * OpenMP calls this before it returns, and leaves as it arrived.
 *
 * Deliberately NOT destroyed: the sticky FJ still owns this context's parked
 * OMP worker contexts, and libomp's thread pool (__kmp_thread_pool) is
 * process-wide -- a later region on another context migrates those same
 * workers onto its own sched via __kmp_lithe_migrate_workers. Destroying here
 * would free scheduler state that kmp_info_t still points at. Exiting is what
 * releases the harts and deregisters the child, which is what the parent needs;
 * the FJ struct itself is reclaimed at process teardown. */
extern "C" void omp_lithe_context_release(void) {
  lithe_fork_join_sched_t *top;
  lithe_sched_t *cur;

  if (__kmp_lithe_flat_on() && __kmp_lithe_flat_team) {
    /* Flat: this context's workers live on the scheduler it is about to leave
     * (the rank FJS). Park them on the OPAL root so the rank FJS's join_all
     * can reach num_contexts == 0; the next region anywhere migrates them back
     * through __kmp_lithe_migrate_workers. */
    lithe_fork_join_sched_t *f = __kmp_lithe_flat_sched();
    lithe_fork_join_sched_t *opal = __kmp_lithe_opal_sched();
    if (f && opal && f != opal)
      (void)__kmp_lithe_migrate_workers(__kmp_lithe_flat_team, opal, f);
    __kmp_lithe_flat_team = NULL;
  }

  if (!__kmp_lithe_omp_top_fj && !__kmp_lithe_top_sticky &&
      __kmp_lithe_fj_sp <= 0 && __kmp_lithe_region_depth <= 0)
    return;

  /* Nested per-region scheds are normally unwound by __kmp_lithe_parallel_leave;
   * unwind defensively (innermost first) in case a region was abandoned. */
  while (__kmp_lithe_fj_sp > 1) {
    lithe_fork_join_sched_t *nested =
        __kmp_lithe_fj_stack[__kmp_lithe_fj_sp - 1];
    cur = lithe_sched_current();
    if (nested && cur == (lithe_sched_t *)nested)
      lithe_sched_exit();
    __kmp_lithe_fj_stack[__kmp_lithe_fj_sp - 1] = NULL;
    __kmp_lithe_fj_team[__kmp_lithe_fj_sp - 1] = NULL;
    __kmp_lithe_fj_sp--;
  }

  top = __kmp_lithe_omp_top_fj;
  if (top) {
    cur = lithe_sched_current();
    if (cur == (lithe_sched_t *)top) {
      /* Move this context's parked OMP workers OFF the scheduler we are about
       * to exit, onto the parent, before exiting it. Skipping this leaves them
       * on a scheduler that has just been deregistered from its parent; the
       * next hart to pick one up enters __uthread_yield with no current
       * uthread and dies in __sigstack_free (observed: SIGSEGV at
       * parlib vcore.c:149 with __sigstack=0xa8, i.e. NULL + offsetof).
       * Same move __kmp_lithe_prepare_teardown_workers makes at process end,
       * restricted to the team on this context. */
      lithe_fork_join_sched_t *parent_fj = __kmp_lithe_opal_sched();
      kmp_team_t *team = __kmp_lithe_fj_team[0];
      if (parent_fj && team && parent_fj != top)
        (void)__kmp_lithe_migrate_workers(team, parent_fj, top);
      lithe_sched_exit();   /* drains this sched's harts, deregisters the child */
    }
  }

  __kmp_lithe_omp_top_fj = NULL;
  __kmp_lithe_top_sticky = 0;
  if (__kmp_lithe_fj_sp > 0) {
    __kmp_lithe_fj_stack[0] = NULL;
    __kmp_lithe_fj_team[0] = NULL;
  }
  __kmp_lithe_fj_sp = 0;
  __kmp_lithe_region_depth = 0;
}

/* Called from __kmp_internal_end before joining workers: move parked worker
 * contexts off the sticky OMP FJ onto the root parent, then sched_exit sticky
 * so idle FJS harts drain. Workers then exit on the parent (no MCS race). */
extern "C" void __kmp_lithe_prepare_teardown_workers(kmp_info_t *pool_head) {
  lithe_fork_join_sched_t *parent_fj = NULL;

  if (!__kmp_lithe_omp_top_fj && !__kmp_lithe_top_sticky &&
      __kmp_lithe_fj_sp <= 0)
    return;

  parent_fj = __kmp_lithe_opal_sched();
  if (parent_fj) {
    for (kmp_info_t *thread = pool_head; thread != NULL;
         thread = CCAST(kmp_info_t *, thread->th.th_next_pool)) {
      if (!thread->th.th_lithe_ctx)
        continue;
      lithe_fork_join_context_t *wctx =
          (lithe_fork_join_context_t *)thread->th.th_lithe_ctx;
      lithe_fork_join_sched_t *from =
          (lithe_fork_join_sched_t *)wctx->context.sched;
      if (from && from != parent_fj)
        lithe_fork_join_context_migrate(from, parent_fj, wctx);
    }
  }

  /* Drain idle sticky harts before workers lithe_context_exit. Skip destroy. */
  __kmp_lithe_shutdown_sticky_fj();
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
    __atomic_store_n(&jc->released, 1, __ATOMIC_RELEASE);
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
      jc->released = 0;
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
      if (lithe_can_block()) {
        lithe_mutex_lock(&jc->mx);
        while (!jc->exited)
          lithe_condvar_wait(&jc->cv, &jc->mx);
        lithe_mutex_unlock(&jc->mx);
        free(jc);
      } else {
        /* PROCESS TEARDOWN (2026-10-14). libomp's library destructor runs from
         * _dl_fini, by which point the application has exited its own
         * scheduler and main's context is back on the BASE scheduler, whose
         * context_block is a deliberate fatal(). Blocking here printed
         * "Contexts created by the base scheduler should never need to block!"
         * and exited 1 on 20/20 runs.
         *
         * There is no scheduler left to park on, so spin on the worker's own
         * exit flag with sched_yield() instead -- the worker is on a hart and
         * will set it. Bounded, because nothing may hang a destructor.
         *
         * If the flag never arrives we LEAK jc on purpose: the worker may
         * still write through it, and freeing memory a live worker can touch
         * at exit is strictly worse than leaking it microseconds before the
         * address space goes away.
         *
         * BOUND (2026-10-16): the bound used to be 1,000,000 sched_yield()s
         * per worker. When the worker's scheduler is already destroyed (the
         * ompctx reproducer: the app FJS owned the parked workers and
         * fjs_destroy'd before _dl_fini) the flag can NEVER arrive, and the
         * spin was the whole "2-3 s per trivial OMP cell" -- strace: 1.33 M
         * sched_yield calls after "stage 10: clean return". A worker that is
         * going to exit does so in microseconds; wait a few ms of wall time,
         * then leak. LITHE_OMP_REAP_WAIT_MS overrides (default 20). */
        static long reap_wait_ms = -1;
        if (reap_wait_ms < 0) {
          const char *e = getenv("LITHE_OMP_REAP_WAIT_MS");
          reap_wait_ms = (e && *e) ? atol(e) : 20;
        }
        struct timespec t0, tn;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        int spins = 0;
        while (!__atomic_load_n(&jc->released, __ATOMIC_ACQUIRE)) {
          sched_yield();
          if ((++spins & 255) == 0) {
            clock_gettime(CLOCK_MONOTONIC, &tn);
            long ms = (tn.tv_sec - t0.tv_sec) * 1000L +
                      (tn.tv_nsec - t0.tv_nsec) / 1000000L;
            if (ms >= reap_wait_ms)
              break;
          }
        }
        if (__atomic_load_n(&jc->released, __ATOMIC_ACQUIRE))
          free(jc);
      }
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
    /* B45 (2026-10-22): a foreign team master (a plain pthread, e.g. a joblib
     * thread that opened the region) is not a Lithe context and cannot block one:
     * lithe_context_block -> uthread_yield with current_uthread == NULL segfaulted
     * (omp_foreign_waves, VCORE_LIMIT=1 + PARLIB_SPARE_VCORES=1, 5/100, when the
     * join gather made that master wait). It holds no hart, so return and let the
     * wait loop spin. */
    if (!lithe_can_block())
      return;
    struct kmp_lithe_suspend_ctl *ctl = __kmp_lithe_suspend_ctl(th);
    if (__kmp_lithe_fast_park) {
      old_spin = flag->set_sleeping();
      if (flag->done_check_val(old_spin)) {
        flag->unset_sleeping();
        return;
      }
      TCW_PTR(th->th.th_sleep_loc, (void *)flag);
      int exp = KMP_LITHE_PARK_RUNNING;
      if (__atomic_compare_exchange_n(&ctl->park, &exp, KMP_LITHE_PARK_PARKING,
                                      false, __ATOMIC_SEQ_CST,
                                      __ATOMIC_SEQ_CST) &&
          flag->is_sleeping())
        lithe_context_block(__kmp_lithe_fast_park_cb, ctl);
      /* woken, token consumed, or released before parking */
      __atomic_store_n(&ctl->park, KMP_LITHE_PARK_RUNNING, __ATOMIC_SEQ_CST);
      flag->unset_sleeping();
      TCW_PTR(th->th.th_sleep_loc, NULL);
      KF_TRACE(30, ("__kmp_suspend_template: T#%d exit (lithe fast park)\n",
                    th_gtid));
      return;
    }
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
    if (__kmp_lithe_fast_park) {
      if (flag)
        flag->unset_sleeping();
      __kmp_lithe_fast_unpark(ctl);
      return;
    }
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
  /* PRE-SCHEDULER GUARD (2026-10-14). This is the single choke point for every
   * KMP_YIELD in libomp -- locks, barriers, teardown -- and it was yielding the
   * hart unconditionally. Not every thread that reaches it is a uthread:
   *
   *   parlib owns each Lithe context's TLS with a "backing pthread" that parks
   *   in futex_wait and, per its own comment, "never enters the lithe
   *   component". But that pthread inherits libomp's pthread_key destructor.
   *   When a context exits and free_tls() releases its TLS, the backing pthread
   *   exits, glibc runs __nptl_deallocate_tsd -> __kmp_internal_end_dest ->
   *   __kmp_internal_end_thread -> __kmp_reset_root -> __kmp_free_team ->
   *   __kmp_release_ticket_lock -> KMP_YIELD -> here, on a thread that has no
   *   uthread at all. lithe_context_yield() then enters unsafe_uthread_yield
   *   with current_uthread == NULL, which with parlib's asserts compiled out
   *   (see src/parlib/src/internal/parlib.h) dereferences NULL and dies in
   *   __sigstack_free with __sigstack = 0xa8. Measured: this is what made ONE
   *   OpenMP region in ONE Lithe context unable to tear down.
   *
   * Note a `current_uthread != NULL` test is NOT enough here: the backing
   * pthread runs destructors on the uthread's own TLS block, so
   * current_uthread is non-NULL and merely STALE (state UT_NOT_RUNNING).
   * parlib_can_cooperatively_yield() tests the real precondition. UCX's
   * ucs_coop_yield() and libfabric's ofi_lithe_yield() carry the weaker NULL
   * test, which covers their case (constructors) but should move to this
   * predicate too. Falling back to sched_yield() is correct: a thread that is
   * not a running uthread has no hart to donate, so the kernel scheduler is
   * the only one that can help.
   */
  /* LITHE_OMP_YIELD_COND (2026-10-16, GAPS C11). Vanilla's __kmp_yield(cond)
   * returns without a syscall when cond is false, and the hottest caller --
   * __kmp_release_ticket_lock, run by the primary once in __kmp_fork_call and
   * once in __kmp_join_call for EVERY parallel region -- passes
   * `distance > avail_proc`, i.e. false unless the lock is massively
   * contended. The (void)cond above turned that into an unconditional
   * lithe_context_yield(): the primary gave up its hart twice per region,
   * each time enqueue + eventfd write (fjs_maybe_reactor_wake, because the
   * PMIx progress context keeps parlib_reactor_pending()==1) + hart re-enter +
   * epoll_wait(0) + eventfd read + two context switches, and often came back
   * on a different hart. strace on GROMACS trp-cage NP=1 OMP=4 (4 harts,
   * demand-block off): 43k epoll_wait + 43k write + 48k read per 1000 steps,
   * all from this path; vanilla: none.
   *   0 = old behaviour, yield on every KMP_YIELD;
   *   1 = honour cond exactly like vanilla (KMP_YIELD(0) is a pause only);
   *   2 = honour cond, but still donate the hart when a runnable-but-hartless
   *       peer exists on this sched (liveness when threads > harts). DEFAULT.
   * A/B 2026-10-16 (login node, GROMACS trp-cage NP=1 OMP=4, 2000 steps, 3 reps
   * each, vanilla -pin on 178-181 ns/day): mode 0 = 147.6-148.0 (0.82x);
   * mode 1 = 180.8-181.1; mode 2 = 180.7-181.4 (1.00x). With mode 1/2 the
   * C11 penalties also vanish: 4 harts + demand-block 180.8-181.1, 5 harts +
   * demand-block 180.0-181.2 -- the "runnable-but-hartless peer" that
   * demand-block kept handing harts to was the primary after this yield. */
  static int yield_cond = -1;
  if (__builtin_expect(yield_cond < 0, 0)) {
    const char *e = getenv("LITHE_OMP_YIELD_COND");
    yield_cond = e ? atoi(e) : 2;
  }
  if (!cond && yield_cond != 0) {
    if (yield_cond == 1)
      return;
    if (!parlib_can_cooperatively_yield() ||
        __kmp_lithe_hartless_peers() <= 0)
      return;
  }
  if (parlib_can_cooperatively_yield())
    lithe_context_yield();
  else
    sched_yield();   /* backing pthread in a TSD destructor, or vcore context */
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

#ifdef LIBOMP_USE_LITHE
/* B43 (2026-10-22). __kmp_release_ticket_lock ends in KMP_YIELD(distance >
 * avail_proc), false unless the lock is massively contended, and
 * LITHE_OMP_YIELD_COND=2 turns a false cond into a hart donation whenever a
 * runnable peer is hartless -- which in an oversubscribed team is always. Task
 * deques are ticket-lock protected, so the producer gave its hart away after
 * every task push and the executors after every pop: OMP tasking with 8
 * threads on 4 harts (1000 regions x 64 tasks) ran 0.9 s to >90 s, bimodal,
 * vs 0.1 s for BOLT, stock libomp and Lithe at 1:1. A releaser waits for
 * nobody, so mode 2's liveness argument (spin loops waiting on a hartless
 * peer) does not apply here; honour cond like vanilla. LITHE_OMP_RELEASE_DONATE
 * =1 restores the donation. */
void __kmp_yield_release(int cond) {
  static int donate = -1;
  if (__builtin_expect(donate < 0, 0)) {
    const char *e = getenv("LITHE_OMP_RELEASE_DONATE");
    donate = (e && *e) ? atoi(e) : 0;
  }
  if (cond || donate)
    __kmp_yield(cond);
}
#endif

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
#ifdef LIBOMP_USE_LITHE
  /* Perfetto trace point (primary thread's share of a region comes through
   * here; workers through lithe/kmp_lithe_entry.cpp). */
  struct RunScope {
    int gtid; bool on;
    RunScope(int g, int t) : gtid(g), on(lithe_sched_trace_enabled() != 0) {
      if (on) lithe_sched_trace_emit(LITHE_ST_OMP_RUN_BEGIN, NULL, (uint32_t)gtid, 0, (int32_t)t);
    }
    ~RunScope() { if (on) lithe_sched_trace_emit(LITHE_ST_OMP_RUN_END, NULL, (uint32_t)gtid, 0, 0); }
  } _lt_run(gtid, tid);
#endif

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
