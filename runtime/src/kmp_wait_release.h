/*
 * kmp_wait_release.h -- Wait/Release implementation
 */

//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//

#ifndef KMP_WAIT_RELEASE_H
#define KMP_WAIT_RELEASE_H

#include "kmp.h"
#include "kmp_itt.h"
#include "kmp_stats.h"
#if OMPT_SUPPORT
#include "ompt-specific.h"
#endif

#ifdef LIBOMP_USE_LITHE
// Team-local inner-barrier spin policy. Defined in z_Linux_util.cpp.
//   __kmp_lithe_inner_spin (LITHE_OMP_INNER_SPIN, default 1): at an intra-region
//     (plain/reduction) barrier, HOLD the hart and hot-spin on the team-local
//     b_go/b_arrived flag instead of cooperatively blocking on a lithe condvar
//     (which parks the backing vcore via a parlib futex on EVERY barrier). 0
//     restores the old always-cooperative-block behavior.
//   __kmp_lithe_inner_spin_max (LITHE_OMP_INNER_SPIN_MAX, default 0): if >0,
//     cap the inner hot-spin at N pause iterations, then fall through to the
//     cooperative block (safety valve). 0 = unbounded (structural design point).
extern "C" {
extern int __kmp_lithe_inner_spin;
extern int __kmp_lithe_fork_yield;
extern kmp_uint64 __kmp_lithe_inner_spin_max;
extern kmp_int64 __kmp_lithe_fork_spin_max;
// Hot-team fork/join hot-spin (default path). See z_Linux_util.cpp.
extern int __kmp_lithe_hot_team;
extern int __kmp_lithe_hot_reserve;
extern kmp_uint64 __kmp_lithe_hot_budget_init;
extern kmp_uint64 __kmp_lithe_hot_budget_min;
extern kmp_uint64 __kmp_lithe_hot_budget_max;
int __kmp_lithe_fork_hot_ok(int nproc);
void __kmp_lithe_inner_spin_config(void);
// Inner-barrier liveness: donate the hart if a hart-less peer is waiting on the
// current FJ sched (prevents hart-starvation livelock when OMP width > harts).
// Returns 1 if it yielded (caller resumed), 0 if it kept spinning. See
// z_Linux_util.cpp __kmp_lithe_inner_barrier_yield.
int __kmp_lithe_inner_barrier_yield(kmp_info_t *th);
// Demand block at the fork/join barrier (LITHE_OMP_DEMAND_BLOCK, default on
// under LITHE_OMP_FLAT=1): spin hot while nobody on this sched is runnable-but-
// hartless, otherwise block NOW instead of polling out KMP_BLOCKTIME. See
// z_Linux_util.cpp __kmp_lithe_demand_block.
int __kmp_lithe_demand_block_on(void);
long lithe_fork_join_current_runnable_count(void);
long __kmp_lithe_hartless_peers(void); /* z_Linux_util.cpp, LITHE_OMP_IDLE_AWARE */
int __kmp_lithe_elsewhere_poll(void); /* z_Linux_util.cpp, LITHE_OMP_ELSEWHERE */
int __kmp_lithe_flat_on(void); /* z_Linux_util.cpp, LITHE_OMP_FLAT */
int __kmp_lithe_nospin_mask(void); /* z_Linux_util.cpp, LITHE_OMP_NOSPIN */
int lithe_can_block(void);
}
#endif

/*!
@defgroup WAIT_RELEASE Wait/Release operations

The definitions and functions here implement the lowest level thread
synchronizations of suspending a thread and awaking it. They are used to build
higher level operations such as barriers and fork/join.
*/

/*!
@ingroup WAIT_RELEASE
@{
*/

/*!
 * The flag_type describes the storage used for the flag.
 */
enum flag_type {
  flag32, /**< 32 bit flags */
  flag64, /**< 64 bit flags */
  flag_oncore /**< special 64-bit flag for on-core barrier (hierarchical) */
};

/*!
 * Base class for wait/release volatile flag
 */
template <typename P> class kmp_flag_native {
  volatile P *loc;
  flag_type t;

public:
  typedef P flag_t;
  kmp_flag_native(volatile P *p, flag_type ft) : loc(p), t(ft) {}
  volatile P *get() { return loc; }
  void set(volatile P *new_loc) { loc = new_loc; }
  flag_type get_type() { return t; }
  P load() { return *loc; }
  void store(P val) { *loc = val; }
};

/*!
 * Base class for wait/release atomic flag
 */
template <typename P> class kmp_flag {
  std::atomic<P>
      *loc; /**< Pointer to the flag storage that is modified by another thread
             */
  flag_type t; /**< "Type" of the flag in loc */
public:
  typedef P flag_t;
  kmp_flag(std::atomic<P> *p, flag_type ft) : loc(p), t(ft) {}
  /*!
   * @result the pointer to the actual flag
   */
  std::atomic<P> *get() { return loc; }
  /*!
   * @param new_loc in   set loc to point at new_loc
   */
  void set(std::atomic<P> *new_loc) { loc = new_loc; }
  /*!
   * @result the flag_type
   */
  flag_type get_type() { return t; }
  /*!
   * @result flag value
   */
  P load() { return loc->load(std::memory_order_acquire); }
  /*!
   * @param val the new flag value to be stored
   */
  void store(P val) { loc->store(val, std::memory_order_release); }
  // Derived classes must provide the following:
  /*
  kmp_info_t * get_waiter(kmp_uint32 i);
  kmp_uint32 get_num_waiters();
  bool done_check();
  bool done_check_val(P old_loc);
  bool notdone_check();
  P internal_release();
  void suspend(int th_gtid);
  void resume(int th_gtid);
  P set_sleeping();
  P unset_sleeping();
  bool is_sleeping();
  bool is_any_sleeping();
  bool is_sleeping_val(P old_loc);
  int execute_tasks(kmp_info_t *this_thr, kmp_int32 gtid, int final_spin,
                    int *thread_finished
                    USE_ITT_BUILD_ARG(void * itt_sync_obj), kmp_int32
                    is_constrained);
  */
};

#if OMPT_SUPPORT
static inline void __ompt_implicit_task_end(kmp_info_t *this_thr,
                                            omp_state_t omp_state,
                                            ompt_data_t *tId,
                                            ompt_data_t *pId) {
  int ds_tid = this_thr->th.th_info.ds.ds_tid;
  if (omp_state == omp_state_wait_barrier_implicit) {
    this_thr->th.ompt_thread_info.state = omp_state_overhead;
#if OMPT_OPTIONAL
    void *codeptr = NULL;
    if (ompt_enabled.ompt_callback_sync_region_wait) {
      ompt_callbacks.ompt_callback(ompt_callback_sync_region_wait)(
          ompt_sync_region_barrier, ompt_scope_end, NULL, tId, codeptr);
    }
    if (ompt_enabled.ompt_callback_sync_region) {
      ompt_callbacks.ompt_callback(ompt_callback_sync_region)(
          ompt_sync_region_barrier, ompt_scope_end, NULL, tId, codeptr);
    }
#endif
    if (!KMP_MASTER_TID(ds_tid)) {
      if (ompt_enabled.ompt_callback_implicit_task) {
        ompt_callbacks.ompt_callback(ompt_callback_implicit_task)(
            ompt_scope_end, NULL, tId, 0, ds_tid);
      }
#if OMPT_OPTIONAL
      if (ompt_enabled.ompt_callback_idle) {
        ompt_callbacks.ompt_callback(ompt_callback_idle)(ompt_scope_begin);
      }
#endif
      // return to idle state
      this_thr->th.ompt_thread_info.state = omp_state_idle;
    } else {
      this_thr->th.ompt_thread_info.state = omp_state_overhead;
    }
  }
}
#endif

/* Spin wait loop that first does pause, then yield, then sleep. A thread that
   calls __kmp_wait_*  must make certain that another thread calls __kmp_release
   to wake it back up to prevent deadlocks!  */
template <class C>
static inline void
__kmp_wait_template(kmp_info_t *this_thr, C *flag,
                    int final_spin USE_ITT_BUILD_ARG(void *itt_sync_obj)) {
  // NOTE: We may not belong to a team at this point.
  volatile void *spin = flag->get();
  kmp_uint32 spins;
  kmp_uint32 hibernate;
  int th_gtid;
  int tasks_completed = FALSE;
  int oversubscribed;
#if !KMP_USE_MONITOR
  kmp_uint64 poll_count;
  kmp_uint64 hibernate_goal;
#endif
#ifdef LIBOMP_USE_LITHE
  kmp_uint64 lithe_inner_spins = 0;
  __kmp_lithe_inner_spin_config();
  // Hot-team fork/join hot-spin gate + per-thread adaptive budget. Computed
  // once here (cheap) and consulted in the spin loop's fork/join branch.
  int lithe_fork_hot = 0;
  int lithe_did_block = 0;
  int lithe_force_block = 0; // demand block: skip the blocktime poll this pass
  const int lithe_demand = __kmp_lithe_demand_block_on();
  // LITHE_OMP_NOSPIN (default 0, NEGATIVE for bit 2 -- region cost 2x at 1:1;
  // z_Linux_util.cpp DE-SPIN): bit 1 inner barrier, bit 2
  // fork/join barrier: park at once (flag->suspend) instead of any hot spin.
  // Not under an infinite blocktime (releasers then skip the resume) and only
  // where suspend really blocks (lithe_can_block; otherwise it returns at once
  // and this would be a spin anyway).
  const int lithe_nospin = __kmp_lithe_nospin_mask();
  // FLAT MODE + KMP_BLOCKTIME=infinite is a hang (2026-10-16, docs/GAPS.md B15):
  // workers hot-spin on every hart forever while their masters sit RUNNABLE,
  // and demand block cannot help because the "infinite" early-continue below
  // runs before the suspend, while forcing the suspend anyway loses wakeups
  // (releasers skip resume under an infinite blocktime). On a 3-hart login node
  // RPH4xOMP2 hung 3/3 at "Creating OpenMP Thread Pool"; 200 ms is 8/8 clean.
  // So in flat mode an infinite blocktime is clamped to the libomp default,
  // once, before the first wait, and said once on stderr. LITHE_OMP_FLAT=0
  // (nested schedulers) keeps the user's value.
  if (__kmp_lithe_flat_on() && __kmp_dflt_blocktime == KMP_MAX_BLOCKTIME) {
    static int said = 0;
    __kmp_dflt_blocktime = KMP_DEFAULT_BLOCKTIME;
    if (!__atomic_exchange_n(&said, 1, __ATOMIC_ACQ_REL))
      fprintf(stderr, "[libomp lithe] LITHE_OMP_FLAT=1 with KMP_BLOCKTIME=infinite "
                      "would hang under hart scarcity; clamped to %d ms\n",
              KMP_DEFAULT_BLOCKTIME);
  }
  kmp_uint64 lithe_fork_budget = 0;
  if (!this_thr->th.th_lithe_barrier_inner) {
    kmp_team_t *lt = this_thr->th.th_team;
    int lnproc = lt ? lt->t.t_nproc : 1;
    lithe_fork_hot = __kmp_lithe_fork_hot_ok(lnproc);
    if (lithe_fork_hot) {
      lithe_fork_budget = this_thr->th.th_lithe_fork_spin_budget;
      if (lithe_fork_budget == 0)
        lithe_fork_budget = __kmp_lithe_hot_budget_init;
    }
  }
#endif

  KMP_FSYNC_SPIN_INIT(spin, NULL);
  if (flag->done_check()) {
    KMP_FSYNC_SPIN_ACQUIRED(CCAST(void *, spin));
    return;
  }
  th_gtid = this_thr->th.th_info.ds.ds_gtid;
  KA_TRACE(20,
           ("__kmp_wait_sleep: T#%d waiting for flag(%p)\n", th_gtid, flag));
#if KMP_STATS_ENABLED
  stats_state_e thread_state = KMP_GET_THREAD_STATE();
#endif

/* OMPT Behavior:
THIS function is called from
  __kmp_barrier (2 times)  (implicit or explicit barrier in parallel regions)
            these have join / fork behavior

       In these cases, we don't change the state or trigger events in THIS
function.
       Events are triggered in the calling code (__kmp_barrier):

                state := omp_state_overhead
            barrier-begin
            barrier-wait-begin
                state := omp_state_wait_barrier
          call join-barrier-implementation (finally arrive here)
          {}
          call fork-barrier-implementation (finally arrive here)
          {}
                state := omp_state_overhead
            barrier-wait-end
            barrier-end
                state := omp_state_work_parallel


  __kmp_fork_barrier  (after thread creation, before executing implicit task)
          call fork-barrier-implementation (finally arrive here)
          {} // worker arrive here with state = omp_state_idle


  __kmp_join_barrier  (implicit barrier at end of parallel region)
                state := omp_state_barrier_implicit
            barrier-begin
            barrier-wait-begin
          call join-barrier-implementation (finally arrive here
final_spin=FALSE)
          {
          }
  __kmp_fork_barrier  (implicit barrier at end of parallel region)
          call fork-barrier-implementation (finally arrive here final_spin=TRUE)

       Worker after task-team is finished:
            barrier-wait-end
            barrier-end
            implicit-task-end
            idle-begin
                state := omp_state_idle

       Before leaving, if state = omp_state_idle
            idle-end
                state := omp_state_overhead
*/
#if OMPT_SUPPORT
  omp_state_t ompt_entry_state;
  ompt_data_t *pId = NULL;
  ompt_data_t *tId;
  if (ompt_enabled.enabled) {
    ompt_entry_state = this_thr->th.ompt_thread_info.state;
    if (!final_spin || ompt_entry_state != omp_state_wait_barrier_implicit ||
        KMP_MASTER_TID(this_thr->th.th_info.ds.ds_tid)) {
      ompt_lw_taskteam_t *team =
          this_thr->th.th_team->t.ompt_serialized_team_info;
      if (team) {
        pId = &(team->ompt_team_info.parallel_data);
        tId = &(team->ompt_task_info.task_data);
      } else {
        pId = OMPT_CUR_TEAM_DATA(this_thr);
        tId = OMPT_CUR_TASK_DATA(this_thr);
      }
    } else {
      pId = NULL;
      tId = &(this_thr->th.ompt_thread_info.task_data);
    }
#if OMPT_OPTIONAL
    if (ompt_entry_state == omp_state_idle) {
      if (ompt_enabled.ompt_callback_idle) {
        ompt_callbacks.ompt_callback(ompt_callback_idle)(ompt_scope_begin);
      }
    } else
#endif
        if (final_spin && (__kmp_tasking_mode == tskm_immediate_exec ||
                           this_thr->th.th_task_team == NULL)) {
      // implicit task is done. Either no taskqueue, or task-team finished
      __ompt_implicit_task_end(this_thr, ompt_entry_state, tId, pId);
    }
  }
#endif

  // Setup for waiting
  KMP_INIT_YIELD(spins);

  if (__kmp_dflt_blocktime != KMP_MAX_BLOCKTIME) {
#if KMP_USE_MONITOR
// The worker threads cannot rely on the team struct existing at this point.
// Use the bt values cached in the thread struct instead.
#ifdef KMP_ADJUST_BLOCKTIME
    if (__kmp_zero_bt && !this_thr->th.th_team_bt_set)
      // Force immediate suspend if not set by user and more threads than
      // available procs
      hibernate = 0;
    else
      hibernate = this_thr->th.th_team_bt_intervals;
#else
    hibernate = this_thr->th.th_team_bt_intervals;
#endif /* KMP_ADJUST_BLOCKTIME */

    /* If the blocktime is nonzero, we want to make sure that we spin wait for
       the entirety of the specified #intervals, plus up to one interval more.
       This increment make certain that this thread doesn't go to sleep too
       soon.  */
    if (hibernate != 0)
      hibernate++;

    // Add in the current time value.
    hibernate += TCR_4(__kmp_global.g.g_time.dt.t_value);
    KF_TRACE(20, ("__kmp_wait_sleep: T#%d now=%d, hibernate=%d, intervals=%d\n",
                  th_gtid, __kmp_global.g.g_time.dt.t_value, hibernate,
                  hibernate - __kmp_global.g.g_time.dt.t_value));
#else
    hibernate_goal = KMP_NOW() + this_thr->th.th_team_bt_intervals;
    poll_count = 0;
#endif // KMP_USE_MONITOR
  }

  oversubscribed = (TCR_4(__kmp_nth) > __kmp_avail_proc);
  KMP_MB();

  // Main wait spin loop
  while (flag->notdone_check()) {
    int in_pool;
    kmp_task_team_t *task_team = NULL;
    if (__kmp_tasking_mode != tskm_immediate_exec) {
      task_team = this_thr->th.th_task_team;
      /* If the thread's task team pointer is NULL, it means one of 3 things:
         1) A newly-created thread is first being released by
         __kmp_fork_barrier(), and its task team has not been set up yet.
         2) All tasks have been executed to completion.
         3) Tasking is off for this region.  This could be because we are in a
         serialized region (perhaps the outer one), or else tasking was manually
         disabled (KMP_TASKING=0).  */
      if (task_team != NULL) {
        if (TCR_SYNC_4(task_team->tt.tt_active)) {
          if (KMP_TASKING_ENABLED(task_team))
            flag->execute_tasks(
                this_thr, th_gtid, final_spin,
                &tasks_completed USE_ITT_BUILD_ARG(itt_sync_obj), 0);
          else
            this_thr->th.th_reap_state = KMP_SAFE_TO_REAP;
        } else {
          KMP_DEBUG_ASSERT(!KMP_MASTER_TID(this_thr->th.th_info.ds.ds_tid));
#if OMPT_SUPPORT
          // task-team is done now, other cases should be catched above
          if (final_spin && ompt_enabled.enabled)
            __ompt_implicit_task_end(this_thr, ompt_entry_state, tId, pId);
#endif
          this_thr->th.th_task_team = NULL;
          this_thr->th.th_reap_state = KMP_SAFE_TO_REAP;
        }
      } else {
        this_thr->th.th_reap_state = KMP_SAFE_TO_REAP;
      } // if
    } // if

    KMP_FSYNC_SPIN_PREPARE(CCAST(void *, spin));
    if (TCR_4(__kmp_global.g.g_done)) {
      if (__kmp_global.g.g_abort)
        __kmp_abort_thread();
      break;
    }

    // If we are oversubscribed, or have waited a bit (and
    // KMP_LIBRARY=throughput), then yield
    // TODO: Should it be number of cores instead of thread contexts? Like:
    // KMP_YIELD(TCR_4(__kmp_nth) > __kmp_ncores);
    // Need performance improvement data to make the change...
#ifdef LIBOMP_USE_LITHE
    // Under Lithe: the spin here must be a *cheap* cpu_relax that HOLDS the
    // hart -- NOT lithe_context_yield(). KMP_YIELD(1) yields the hart back to
    // the fork-join scheduler on every iteration, which re-enters hart_enter ->
    // fjs_reactor_poll -> epoll_wait(0): an epoll storm and anti-scaling.
    KMP_CPU_PAUSE();
    // NEGATIVE RESULT (2026-10-22, lithified PyTorch dataloader_train 2 vs 20
    // images/s): neither a sched_yield every N spins (to share the CPU with
    // DataLoader worker PROCESSES; 8-11) nor polling parlib's reactor from this
    // spin (so a pipe-waiting context wakes without a free hart; no change)
    // fixed it. The cause was hart budget, not this loop: fork()ed workers
    // inherited the trainer's 4 harts and their idle harts spun on a pending
    // reactor waiter (PARLIB_FORK_CHILD_VCORES, LITHE_FJS_IDLE_PENDING_NEEDS_SLOT;
    // docs/GAPS.md B47). Do not add either here again without a new reason.
    // INNER (intra-region) barrier: keep spinning on the team-local flag while
    // HOLDING the hart. The releaser's internal_release() is a plain atomic
    // store this spinner observes (wake ~ns, like vanilla); because we never
    // set the sleep bit and never call flag->suspend(), the release side's
    // is_any_sleeping() is false -> no lithe_condvar_signal, no vcore unpark,
    // no parlib futex. This removes the PER-BARRIER global-scheduler interaction
    // (lithe_hart_request +/-1, FJS queue locks, vcore park/wake) that made
    // lithified OpenMP anti-scale over CG's thousands of inner barriers.
    //
    // FORK/JOIN (region-boundary) barriers: bounded hot-spin holding the hart
    // (LITHE_OMP_FORK_SPIN_MAX). miniFE-style codes re-fork a parallel region
    // per operation, so the hot-team workers wait HERE between regions; parking
    // (cooperative block) on every region is the futex park/wake storm that
    // made lithified OpenMP anti-scale. We instead spin on the team-local flag
    // (wake == atomic store) to bridge the rapid inter-region gap, then -- only
    // if the gap is long (genuine MPI/serial idle) -- fall through to the
    // cooperative block below, which returns the hart (lithe_hart_request(-1))
    // to sibling schedulers (MPI/PMIx/UCX progress). Composition is preserved
    // at this COARSE granularity, and the bound means this is NOT infinite
    // hart hoarding.
    if (lithe_nospin &&
        (lithe_nospin & (this_thr->th.th_lithe_barrier_inner ? 1 : 2)) &&
        __kmp_dflt_blocktime != KMP_MAX_BLOCKTIME && lithe_can_block()) {
      // Tasks may still be spawned into this team: the wait below cannot park
      // (no wake on task push in this libomp), so keep the hart-donation
      // liveness of the spin paths instead of a bare pause.
      if (task_team != NULL && TCR_4(task_team->tt.tt_found_tasks))
        (void)__kmp_lithe_inner_barrier_yield(this_thr);
      lithe_force_block = 1;
      lithe_did_block = 1;
    } else if (this_thr->th.th_lithe_barrier_inner) {
      if (__kmp_lithe_inner_spin &&
          (__kmp_lithe_inner_spin_max == 0 ||
           ++lithe_inner_spins < __kmp_lithe_inner_spin_max)) {
        // LIVENESS: the hot-spin above HOLDS this worker's hart. If OMP width
        // exceeds the available harts (oversubscription / under-provisioned
        // VCORE_LIMIT), hart-less peers sit RUNNABLE and can never reach this
        // barrier -> hart-starvation livelock (full-node OMP=96/192 hang).
        // Donate the hart when a hart-less peer is RUNNABLE (one atomic load when
        // none are waiting). Required for oversubscribed teams (OMP>cores) at
        // hierarchical reduction/release (BabelStream dot()).
        (void)__kmp_lithe_inner_barrier_yield(this_thr);
        continue;
      }
    } else if (lithe_demand && (__kmp_lithe_hartless_peers() > 0 ||
                                __kmp_lithe_elsewhere_poll())) {
      // (GAPS B61, LITHE_OMP_ELSEWHERE: also when a scheduler on the chain
      // above this team's has unmet hart demand -- a co-tenant runtime or a
      // sibling rank's team -- which the per-sched peer count cannot see.)
      // FORK/JOIN barrier, DEMAND BLOCK (LITHE_OMP_DEMAND_BLOCK): a peer on
      // this sched is RUNNABLE but has no hart -- in the hosted flat model
      // usually our own master, i.e. the logical MPI rank that has to reach
      // the next fork. Hand the hart over by BLOCKING now (one cooperative
      // park; we are not runnable again until the next release), not by
      // yielding (LITHE_OMP_FORK_YIELD: re-dispatched, flag still clear,
      // yield again -- the measured 40x storm) and not after KMP_BLOCKTIME of
      // spinning over the master's head (the measured "slow, not hung").
      // Falls through to flag->suspend() below with the time poll bypassed.
      lithe_force_block = 1;
      lithe_did_block = 1; // hot-team AIMD: this gap counts as "long"
    } else {
      // FORK/JOIN (region-boundary) barrier.
      lithe_force_block = 0;
      // 0) Liveness: donate the hart to a RUNNABLE-but-hartless peer on this
      //    sched (typically our own master). LITHE_OMP_FORK_YIELD, see
      //    z_Linux_util.cpp. One atomic load when nobody is waiting.
      if (__kmp_lithe_fork_yield && __kmp_lithe_inner_barrier_yield(this_thr))
        continue;
      // 1) Manual override (LITHE_OMP_FORK_SPIN_MAX != 0) takes precedence for
      //    A/B experiments.
      if (__kmp_lithe_fork_spin_max != 0) {
        if (__kmp_lithe_fork_spin_max < 0 ||
            ++lithe_inner_spins < (kmp_uint64)__kmp_lithe_fork_spin_max)
          continue;
      } else if (lithe_fork_hot) {
        // 2) Default hot-team path: hold the hart and hot-spin on the team-local
        //    go-flag so the next region's fork wakes us with a plain memory
        //    write (like vanilla), NOT a parlib vcore park/wake + global hart
        //    re-accounting. Bounded by the per-thread adaptive budget so a
        //    genuinely long idle gap (e.g. an MPI exchange / serial phase
        //    longer than the budget) falls through to the cooperative block,
        //    returning the hart to sibling schedulers (MPI/PMIx/UCX progress).
        if (++lithe_inner_spins < lithe_fork_budget)
          continue;
        // Budget exhausted: remember we are about to block so we shrink the
        // budget on the way out (this gap was longer than worth spinning).
        lithe_did_block = 1;
      }
    }
#else
    if (oversubscribed) {
      KMP_YIELD(1);
    } else {
      KMP_YIELD_SPIN(spins);
    }
#endif
    // Check if this thread was transferred from a team
    // to the thread pool (or vice-versa) while spinning.
    in_pool = !!TCR_4(this_thr->th.th_in_pool);
    if (in_pool != !!this_thr->th.th_active_in_pool) {
      if (in_pool) { // Recently transferred from team to pool
        KMP_ATOMIC_INC(&__kmp_thread_pool_active_nth);
        this_thr->th.th_active_in_pool = TRUE;
        /* Here, we cannot assert that:
           KMP_DEBUG_ASSERT(TCR_4(__kmp_thread_pool_active_nth) <=
           __kmp_thread_pool_nth);
           __kmp_thread_pool_nth is inc/dec'd by the master thread while the
           fork/join lock is held, whereas __kmp_thread_pool_active_nth is
           inc/dec'd asynchronously by the workers. The two can get out of sync
           for brief periods of time.  */
      } else { // Recently transferred from pool to team
        KMP_ATOMIC_DEC(&__kmp_thread_pool_active_nth);
        KMP_DEBUG_ASSERT(TCR_4(__kmp_thread_pool_active_nth) >= 0);
        this_thr->th.th_active_in_pool = FALSE;
      }
    }

#if KMP_STATS_ENABLED
    // Check if thread has been signalled to idle state
    // This indicates that the logical "join-barrier" has finished
    if (this_thr->th.th_stats->isIdle() &&
        KMP_GET_THREAD_STATE() == FORK_JOIN_BARRIER) {
      KMP_SET_THREAD_STATE(IDLE);
      KMP_PUSH_PARTITIONED_TIMER(OMP_idle);
    }
#endif

    // Don't suspend if KMP_BLOCKTIME is set to "infinite"
    // (Under Lithe flat mode this can no longer be MAX: see the clamp at the top
    // of this function. Forcing a suspend here under an infinite blocktime was
    // tried on 2026-10-16 and produced a lost wakeup -- releasers skip the
    // resume path when blocktime is infinite because nobody is expected to
    // sleep -- so the setting itself is what has to go.)
    if (__kmp_dflt_blocktime == KMP_MAX_BLOCKTIME)
      continue;

    // Don't suspend if there is a likelihood of new tasks being spawned.
    if ((task_team != NULL) && TCR_4(task_team->tt.tt_found_tasks))
      continue;

#if KMP_USE_MONITOR
    // If we have waited a bit more, fall asleep
    if (TCR_4(__kmp_global.g.g_time.dt.t_value) < hibernate)
      continue;
#else
#ifdef LIBOMP_USE_LITHE
    if (!lithe_force_block && KMP_BLOCKING(hibernate_goal, poll_count++))
      continue;
#else
    if (KMP_BLOCKING(hibernate_goal, poll_count++))
      continue;
#endif
#endif

    KF_TRACE(50, ("__kmp_wait_sleep: T#%d suspend time reached\n", th_gtid));
    flag->suspend(th_gtid);

    if (TCR_4(__kmp_global.g.g_done)) {
      if (__kmp_global.g.g_abort)
        __kmp_abort_thread();
      break;
    } else if (__kmp_tasking_mode != tskm_immediate_exec &&
               this_thr->th.th_reap_state == KMP_SAFE_TO_REAP) {
      this_thr->th.th_reap_state = KMP_NOT_SAFE_TO_REAP;
    }
    // TODO: If thread is done with work and times out, disband/free
  }

#ifdef LIBOMP_USE_LITHE
  // AIMD-adapt the per-thread fork/join hot-spin budget. If the release arrived
  // within budget (we never had to cooperatively block), the inter-region gap
  // was short -> spinning paid off; grow the budget (double, capped) so we stay
  // hot through the same gap next time. If we exhausted the budget and blocked,
  // the gap was long -> shrink (halve, floored) so we release the hart sooner
  // next time. This keeps miniFE-style short-gap teams resident on their vcores
  // (user-level memory-write wake) while long idle still returns harts for
  // composition -- and the bound means it is never infinite hart hoarding.
  if (lithe_fork_hot) {
    kmp_uint64 b = this_thr->th.th_lithe_fork_spin_budget;
    if (b == 0)
      b = __kmp_lithe_hot_budget_init;
    if (lithe_did_block) {
      b = b / 2;
      if (b < __kmp_lithe_hot_budget_min)
        b = __kmp_lithe_hot_budget_min;
    } else {
      b = b ? b * 2 : __kmp_lithe_hot_budget_min;
      if (b > __kmp_lithe_hot_budget_max)
        b = __kmp_lithe_hot_budget_max;
    }
    this_thr->th.th_lithe_fork_spin_budget = b;
  }
#endif

#if OMPT_SUPPORT
  omp_state_t ompt_exit_state = this_thr->th.ompt_thread_info.state;
  if (ompt_enabled.enabled && ompt_exit_state != omp_state_undefined) {
#if OMPT_OPTIONAL
    if (final_spin) {
      __ompt_implicit_task_end(this_thr, ompt_exit_state, tId, pId);
      ompt_exit_state = this_thr->th.ompt_thread_info.state;
    }
#endif
    if (ompt_exit_state == omp_state_idle) {
#if OMPT_OPTIONAL
      if (ompt_enabled.ompt_callback_idle) {
        ompt_callbacks.ompt_callback(ompt_callback_idle)(ompt_scope_end);
      }
#endif
      this_thr->th.ompt_thread_info.state = omp_state_overhead;
    }
  }
#endif
#if KMP_STATS_ENABLED
  // If we were put into idle state, pop that off the state stack
  if (KMP_GET_THREAD_STATE() == IDLE) {
    KMP_POP_PARTITIONED_TIMER();
    KMP_SET_THREAD_STATE(thread_state);
    this_thr->th.th_stats->resetIdleFlag();
  }
#endif

  KMP_FSYNC_SPIN_ACQUIRED(CCAST(void *, spin));
}

/* Release any threads specified as waiting on the flag by releasing the flag
   and resume the waiting thread if indicated by the sleep bit(s). A thread that
   calls __kmp_wait_template must call this function to wake up the potentially
   sleeping thread and prevent deadlocks!  */
template <class C> static inline void __kmp_release_template(C *flag) {
#ifdef KMP_DEBUG
  int gtid = TCR_4(__kmp_init_gtid) ? __kmp_get_gtid() : -1;
#endif
  KF_TRACE(20, ("__kmp_release: T#%d releasing flag(%x)\n", gtid, flag->get()));
  KMP_DEBUG_ASSERT(flag->get());
  KMP_FSYNC_RELEASING(flag->get());

  flag->internal_release();

  KF_TRACE(100, ("__kmp_release: T#%d set new spin=%d\n", gtid, flag->get(),
                 flag->load()));

  if (__kmp_dflt_blocktime != KMP_MAX_BLOCKTIME) {
    // Only need to check sleep stuff if infinite block time not set.
    // Are *any* threads waiting on flag sleeping?
    if (flag->is_any_sleeping()) {
      for (unsigned int i = 0; i < flag->get_num_waiters(); ++i) {
        // if sleeping waiter exists at i, sets current_waiter to i inside flag
        kmp_info_t *waiter = flag->get_waiter(i);
        if (waiter) {
          int wait_gtid = waiter->th.th_info.ds.ds_gtid;
          // Wake up thread if needed
          KF_TRACE(50, ("__kmp_release: T#%d waking up thread T#%d since sleep "
                        "flag(%p) set\n",
                        gtid, wait_gtid, flag->get()));
          flag->resume(wait_gtid); // unsets flag's current_waiter when done
        }
      }
    }
  }
}

template <typename FlagType> struct flag_traits {};

template <> struct flag_traits<kmp_uint32> {
  typedef kmp_uint32 flag_t;
  static const flag_type t = flag32;
  static inline flag_t tcr(flag_t f) { return TCR_4(f); }
  static inline flag_t test_then_add4(volatile flag_t *f) {
    return KMP_TEST_THEN_ADD4_32(RCAST(volatile kmp_int32 *, f));
  }
  static inline flag_t test_then_or(volatile flag_t *f, flag_t v) {
    return KMP_TEST_THEN_OR32(f, v);
  }
  static inline flag_t test_then_and(volatile flag_t *f, flag_t v) {
    return KMP_TEST_THEN_AND32(f, v);
  }
};

template <> struct flag_traits<kmp_uint64> {
  typedef kmp_uint64 flag_t;
  static const flag_type t = flag64;
  static inline flag_t tcr(flag_t f) { return TCR_8(f); }
  static inline flag_t test_then_add4(volatile flag_t *f) {
    return KMP_TEST_THEN_ADD4_64(RCAST(volatile kmp_int64 *, f));
  }
  static inline flag_t test_then_or(volatile flag_t *f, flag_t v) {
    return KMP_TEST_THEN_OR64(f, v);
  }
  static inline flag_t test_then_and(volatile flag_t *f, flag_t v) {
    return KMP_TEST_THEN_AND64(f, v);
  }
};

// Basic flag that does not use C11 Atomics
template <typename FlagType>
class kmp_basic_flag_native : public kmp_flag_native<FlagType> {
  typedef flag_traits<FlagType> traits_type;
  FlagType checker; /**< Value to compare flag to to check if flag has been
                       released. */
  kmp_info_t
      *waiting_threads[1]; /**< Array of threads sleeping on this thread. */
  kmp_uint32
      num_waiting_threads; /**< Number of threads sleeping on this thread. */
public:
  kmp_basic_flag_native(volatile FlagType *p)
      : kmp_flag_native<FlagType>(p, traits_type::t), num_waiting_threads(0) {}
  kmp_basic_flag_native(volatile FlagType *p, kmp_info_t *thr)
      : kmp_flag_native<FlagType>(p, traits_type::t), num_waiting_threads(1) {
    waiting_threads[0] = thr;
  }
  kmp_basic_flag_native(volatile FlagType *p, FlagType c)
      : kmp_flag_native<FlagType>(p, traits_type::t), checker(c),
        num_waiting_threads(0) {}
  /*!
   * param i in   index into waiting_threads
   * @result the thread that is waiting at index i
   */
  kmp_info_t *get_waiter(kmp_uint32 i) {
    KMP_DEBUG_ASSERT(i < num_waiting_threads);
    return waiting_threads[i];
  }
  /*!
   * @result num_waiting_threads
   */
  kmp_uint32 get_num_waiters() { return num_waiting_threads; }
  /*!
   * @param thr in   the thread which is now waiting
   *
   * Insert a waiting thread at index 0.
   */
  void set_waiter(kmp_info_t *thr) {
    waiting_threads[0] = thr;
    num_waiting_threads = 1;
  }
  /*!
   * @result true if the flag object has been released.
   */
  bool done_check() { return traits_type::tcr(*(this->get())) == checker; }
  /*!
   * @param old_loc in   old value of flag
   * @result true if the flag's old value indicates it was released.
   */
  bool done_check_val(FlagType old_loc) { return old_loc == checker; }
  /*!
   * @result true if the flag object is not yet released.
   * Used in __kmp_wait_template like:
   * @code
   * while (flag.notdone_check()) { pause(); }
   * @endcode
   */
  bool notdone_check() { return traits_type::tcr(*(this->get())) != checker; }
  /*!
   * @result Actual flag value before release was applied.
   * Trigger all waiting threads to run by modifying flag to release state.
   */
  void internal_release() {
    (void)traits_type::test_then_add4((volatile FlagType *)this->get());
  }
  /*!
   * @result Actual flag value before sleep bit(s) set.
   * Notes that there is at least one thread sleeping on the flag by setting
   * sleep bit(s).
   */
  FlagType set_sleeping() {
    return traits_type::test_then_or((volatile FlagType *)this->get(),
                                     KMP_BARRIER_SLEEP_STATE);
  }
  /*!
   * @result Actual flag value before sleep bit(s) cleared.
   * Notes that there are no longer threads sleeping on the flag by clearing
   * sleep bit(s).
   */
  FlagType unset_sleeping() {
    return traits_type::test_then_and((volatile FlagType *)this->get(),
                                      ~KMP_BARRIER_SLEEP_STATE);
  }
  /*!
   * @param old_loc in   old value of flag
   * Test whether there are threads sleeping on the flag's old value in old_loc.
   */
  bool is_sleeping_val(FlagType old_loc) {
    return old_loc & KMP_BARRIER_SLEEP_STATE;
  }
  /*!
   * Test whether there are threads sleeping on the flag.
   */
  bool is_sleeping() { return is_sleeping_val(*(this->get())); }
  bool is_any_sleeping() { return is_sleeping_val(*(this->get())); }
  kmp_uint8 *get_stolen() { return NULL; }
  enum barrier_type get_bt() { return bs_last_barrier; }
};

template <typename FlagType> class kmp_basic_flag : public kmp_flag<FlagType> {
  typedef flag_traits<FlagType> traits_type;
  FlagType checker; /**< Value to compare flag to to check if flag has been
                       released. */
  kmp_info_t
      *waiting_threads[1]; /**< Array of threads sleeping on this thread. */
  kmp_uint32
      num_waiting_threads; /**< Number of threads sleeping on this thread. */
public:
  kmp_basic_flag(std::atomic<FlagType> *p)
      : kmp_flag<FlagType>(p, traits_type::t), num_waiting_threads(0) {}
  kmp_basic_flag(std::atomic<FlagType> *p, kmp_info_t *thr)
      : kmp_flag<FlagType>(p, traits_type::t), num_waiting_threads(1) {
    waiting_threads[0] = thr;
  }
  kmp_basic_flag(std::atomic<FlagType> *p, FlagType c)
      : kmp_flag<FlagType>(p, traits_type::t), checker(c),
        num_waiting_threads(0) {}
  /*!
   * param i in   index into waiting_threads
   * @result the thread that is waiting at index i
   */
  kmp_info_t *get_waiter(kmp_uint32 i) {
    KMP_DEBUG_ASSERT(i < num_waiting_threads);
    return waiting_threads[i];
  }
  /*!
   * @result num_waiting_threads
   */
  kmp_uint32 get_num_waiters() { return num_waiting_threads; }
  /*!
   * @param thr in   the thread which is now waiting
   *
   * Insert a waiting thread at index 0.
   */
  void set_waiter(kmp_info_t *thr) {
    waiting_threads[0] = thr;
    num_waiting_threads = 1;
  }
  /*!
   * @result true if the flag object has been released.
   */
  bool done_check() { return this->load() == checker; }
  /*!
   * @param old_loc in   old value of flag
   * @result true if the flag's old value indicates it was released.
   */
  bool done_check_val(FlagType old_loc) { return old_loc == checker; }
  /*!
   * @result true if the flag object is not yet released.
   * Used in __kmp_wait_template like:
   * @code
   * while (flag.notdone_check()) { pause(); }
   * @endcode
   */
  bool notdone_check() { return this->load() != checker; }
  /*!
   * @result Actual flag value before release was applied.
   * Trigger all waiting threads to run by modifying flag to release state.
   */
  void internal_release() { KMP_ATOMIC_ADD(this->get(), 4); }
  /*!
   * @result Actual flag value before sleep bit(s) set.
   * Notes that there is at least one thread sleeping on the flag by setting
   * sleep bit(s).
   */
  FlagType set_sleeping() {
    return KMP_ATOMIC_OR(this->get(), KMP_BARRIER_SLEEP_STATE);
  }
  /*!
   * @result Actual flag value before sleep bit(s) cleared.
   * Notes that there are no longer threads sleeping on the flag by clearing
   * sleep bit(s).
   */
  FlagType unset_sleeping() {
    return KMP_ATOMIC_AND(this->get(), ~KMP_BARRIER_SLEEP_STATE);
  }
  /*!
   * @param old_loc in   old value of flag
   * Test whether there are threads sleeping on the flag's old value in old_loc.
   */
  bool is_sleeping_val(FlagType old_loc) {
    return old_loc & KMP_BARRIER_SLEEP_STATE;
  }
  /*!
   * Test whether there are threads sleeping on the flag.
   */
  bool is_sleeping() { return is_sleeping_val(this->load()); }
  bool is_any_sleeping() { return is_sleeping_val(this->load()); }
  kmp_uint8 *get_stolen() { return NULL; }
  enum barrier_type get_bt() { return bs_last_barrier; }
};

class kmp_flag_32 : public kmp_basic_flag<kmp_uint32> {
public:
  kmp_flag_32(std::atomic<kmp_uint32> *p) : kmp_basic_flag<kmp_uint32>(p) {}
  kmp_flag_32(std::atomic<kmp_uint32> *p, kmp_info_t *thr)
      : kmp_basic_flag<kmp_uint32>(p, thr) {}
  kmp_flag_32(std::atomic<kmp_uint32> *p, kmp_uint32 c)
      : kmp_basic_flag<kmp_uint32>(p, c) {}
  void suspend(int th_gtid) { __kmp_suspend_32(th_gtid, this); }
  void resume(int th_gtid) { __kmp_resume_32(th_gtid, this); }
  int execute_tasks(kmp_info_t *this_thr, kmp_int32 gtid, int final_spin,
                    int *thread_finished USE_ITT_BUILD_ARG(void *itt_sync_obj),
                    kmp_int32 is_constrained) {
    return __kmp_execute_tasks_32(
        this_thr, gtid, this, final_spin,
        thread_finished USE_ITT_BUILD_ARG(itt_sync_obj), is_constrained);
  }
  void wait(kmp_info_t *this_thr,
            int final_spin USE_ITT_BUILD_ARG(void *itt_sync_obj)) {
    __kmp_wait_template(this_thr, this,
                        final_spin USE_ITT_BUILD_ARG(itt_sync_obj));
  }
  void release() { __kmp_release_template(this); }
  flag_type get_ptr_type() { return flag32; }
};

class kmp_flag_64 : public kmp_basic_flag_native<kmp_uint64> {
public:
  kmp_flag_64(volatile kmp_uint64 *p) : kmp_basic_flag_native<kmp_uint64>(p) {}
  kmp_flag_64(volatile kmp_uint64 *p, kmp_info_t *thr)
      : kmp_basic_flag_native<kmp_uint64>(p, thr) {}
  kmp_flag_64(volatile kmp_uint64 *p, kmp_uint64 c)
      : kmp_basic_flag_native<kmp_uint64>(p, c) {}
  void suspend(int th_gtid) { __kmp_suspend_64(th_gtid, this); }
  void resume(int th_gtid) { __kmp_resume_64(th_gtid, this); }
  int execute_tasks(kmp_info_t *this_thr, kmp_int32 gtid, int final_spin,
                    int *thread_finished USE_ITT_BUILD_ARG(void *itt_sync_obj),
                    kmp_int32 is_constrained) {
    return __kmp_execute_tasks_64(
        this_thr, gtid, this, final_spin,
        thread_finished USE_ITT_BUILD_ARG(itt_sync_obj), is_constrained);
  }
  void wait(kmp_info_t *this_thr,
            int final_spin USE_ITT_BUILD_ARG(void *itt_sync_obj)) {
    __kmp_wait_template(this_thr, this,
                        final_spin USE_ITT_BUILD_ARG(itt_sync_obj));
  }
  void release() { __kmp_release_template(this); }
  flag_type get_ptr_type() { return flag64; }
};

// Hierarchical 64-bit on-core barrier instantiation
class kmp_flag_oncore : public kmp_flag_native<kmp_uint64> {
  kmp_uint64 checker;
  kmp_info_t *waiting_threads[1];
  kmp_uint32 num_waiting_threads;
  kmp_uint32
      offset; /**< Portion of flag that is of interest for an operation. */
  bool flag_switch; /**< Indicates a switch in flag location. */
  enum barrier_type bt; /**< Barrier type. */
  kmp_info_t *this_thr; /**< Thread that may be redirected to different flag
                           location. */
#if USE_ITT_BUILD
  void *
      itt_sync_obj; /**< ITT object that must be passed to new flag location. */
#endif
  unsigned char &byteref(volatile kmp_uint64 *loc, size_t offset) {
    return (RCAST(unsigned char *, CCAST(kmp_uint64 *, loc)))[offset];
  }

public:
  kmp_flag_oncore(volatile kmp_uint64 *p)
      : kmp_flag_native<kmp_uint64>(p, flag_oncore), num_waiting_threads(0),
        flag_switch(false) {}
  kmp_flag_oncore(volatile kmp_uint64 *p, kmp_uint32 idx)
      : kmp_flag_native<kmp_uint64>(p, flag_oncore), num_waiting_threads(0),
        offset(idx), flag_switch(false) {}
  kmp_flag_oncore(volatile kmp_uint64 *p, kmp_uint64 c, kmp_uint32 idx,
                  enum barrier_type bar_t,
                  kmp_info_t *thr USE_ITT_BUILD_ARG(void *itt))
      : kmp_flag_native<kmp_uint64>(p, flag_oncore), checker(c),
        num_waiting_threads(0), offset(idx), flag_switch(false), bt(bar_t),
        this_thr(thr) USE_ITT_BUILD_ARG(itt_sync_obj(itt)) {}
  kmp_info_t *get_waiter(kmp_uint32 i) {
    KMP_DEBUG_ASSERT(i < num_waiting_threads);
    return waiting_threads[i];
  }
  kmp_uint32 get_num_waiters() { return num_waiting_threads; }
  void set_waiter(kmp_info_t *thr) {
    waiting_threads[0] = thr;
    num_waiting_threads = 1;
  }
  bool done_check_val(kmp_uint64 old_loc) {
    return byteref(&old_loc, offset) == checker;
  }
  bool done_check() { return done_check_val(*get()); }
  bool notdone_check() {
    // Calculate flag_switch
    if (this_thr->th.th_bar[bt].bb.wait_flag == KMP_BARRIER_SWITCH_TO_OWN_FLAG)
      flag_switch = true;
    if (byteref(get(), offset) != 1 && !flag_switch)
      return true;
    else if (flag_switch) {
      this_thr->th.th_bar[bt].bb.wait_flag = KMP_BARRIER_SWITCHING;
      kmp_flag_64 flag(&this_thr->th.th_bar[bt].bb.b_go,
                       (kmp_uint64)KMP_BARRIER_STATE_BUMP);
      __kmp_wait_64(this_thr, &flag, TRUE USE_ITT_BUILD_ARG(itt_sync_obj));
    }
    return false;
  }
  void internal_release() {
    // Other threads can write their own bytes simultaneously.
    if (__kmp_dflt_blocktime == KMP_MAX_BLOCKTIME) {
      byteref(get(), offset) = 1;
    } else {
      kmp_uint64 mask = 0;
      byteref(&mask, offset) = 1;
      KMP_TEST_THEN_OR64(get(), mask);
    }
  }
  kmp_uint64 set_sleeping() {
    return KMP_TEST_THEN_OR64(get(), KMP_BARRIER_SLEEP_STATE);
  }
  kmp_uint64 unset_sleeping() {
    return KMP_TEST_THEN_AND64(get(), ~KMP_BARRIER_SLEEP_STATE);
  }
  bool is_sleeping_val(kmp_uint64 old_loc) {
    return old_loc & KMP_BARRIER_SLEEP_STATE;
  }
  bool is_sleeping() { return is_sleeping_val(*get()); }
  bool is_any_sleeping() { return is_sleeping_val(*get()); }
  void wait(kmp_info_t *this_thr, int final_spin) {
    __kmp_wait_template<kmp_flag_oncore>(
        this_thr, this, final_spin USE_ITT_BUILD_ARG(itt_sync_obj));
  }
  void release() { __kmp_release_template(this); }
  void suspend(int th_gtid) { __kmp_suspend_oncore(th_gtid, this); }
  void resume(int th_gtid) { __kmp_resume_oncore(th_gtid, this); }
  int execute_tasks(kmp_info_t *this_thr, kmp_int32 gtid, int final_spin,
                    int *thread_finished USE_ITT_BUILD_ARG(void *itt_sync_obj),
                    kmp_int32 is_constrained) {
    return __kmp_execute_tasks_oncore(
        this_thr, gtid, this, final_spin,
        thread_finished USE_ITT_BUILD_ARG(itt_sync_obj), is_constrained);
  }
  kmp_uint8 *get_stolen() { return NULL; }
  enum barrier_type get_bt() { return bt; }
  flag_type get_ptr_type() { return flag_oncore; }
};

// Used to wake up threads, volatile void* flag is usually the th_sleep_loc
// associated with int gtid.
static inline void __kmp_null_resume_wrapper(int gtid, volatile void *flag) {
  if (!flag)
    return;

  switch (RCAST(kmp_flag_64 *, CCAST(void *, flag))->get_type()) {
  case flag32:
    __kmp_resume_32(gtid, NULL);
    break;
  case flag64:
    __kmp_resume_64(gtid, NULL);
    break;
  case flag_oncore:
    __kmp_resume_oncore(gtid, NULL);
    break;
  }
}

/*!
@}
*/

#endif // KMP_WAIT_RELEASE_H
