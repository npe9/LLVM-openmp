/* Parlib alarm-based OpenMP monitor for LIBOMP_USE_LITHE.
 * Replaces the POSIX monitor thread (pthread_create/__kmp_launch_monitor) with
 * periodic EV_ALARM delivery on a vcore; callback mirrors the body of the
 * monitor loop in z_Linux_util.cpp (yield policy + g_time tick). */

#include <parlib/alarm.h>

#include "kmp.h"
#include "kmp_itt.h"

#if defined(LIBOMP_USE_LITHE) && KMP_USE_MONITOR

extern "C" {

static struct alarm_waiter kmp_lithe_mon_alarm;
static int kmp_lithe_mon_alarm_inited;
static int kmp_lithe_mon_yield_cycles;
static int kmp_lithe_mon_yield_count;
static int kmp_lithe_mon_itt_ignore_done;

static uint64_t kmp_lithe_mon_interval_usec(void) {
  if (__kmp_monitor_wakeups == 1) {
    return 1000000ULL;
  }
  if (__kmp_monitor_wakeups < 1) {
    return 1000000ULL;
  }
  /* __kmp_launch_monitor uses interval nsec = KMP_NSEC_PER_SEC / wakeups */
  return (uint64_t)(KMP_NSEC_PER_SEC / __kmp_monitor_wakeups / 1000ULL);
}

static void kmp_lithe_mon_alarm_cb(struct alarm_waiter *waiter);

static void kmp_lithe_mon_alarm_setup(void) {
  if (kmp_lithe_mon_alarm_inited) {
    return;
  }
  init_awaiter(&kmp_lithe_mon_alarm, kmp_lithe_mon_alarm_cb);
  /* Route alarm handling to vcore 0's event queue (see parlib/event.c). */
  kmp_lithe_mon_alarm.vcoreid = 0;
  kmp_lithe_mon_alarm_inited = 1;
}

static void kmp_lithe_mon_alarm_cb(struct alarm_waiter *waiter) {
  (void)waiter;

#ifdef USE_ITT_BUILD
  if (!kmp_lithe_mon_itt_ignore_done) {
    __kmp_itt_thread_ignore();
    kmp_lithe_mon_itt_ignore_done = 1;
  }
#endif

  __kmp_gtid_set_specific(KMP_GTID_MONITOR);
#ifdef KMP_TDATA_GTID
  __kmp_gtid = KMP_GTID_MONITOR;
#endif

  if (!TCR_4(__kmp_global.g.g_done)) {

    /* Same yield-cycle policy as __kmp_launch_monitor (z_Linux_util.cpp). */
    if (__kmp_yield_cycle) {
      kmp_lithe_mon_yield_cycles++;
      if ((kmp_lithe_mon_yield_cycles % kmp_lithe_mon_yield_count) == 0) {
        if (__kmp_yielding_on) {
          __kmp_yielding_on = 0;
          kmp_lithe_mon_yield_count = __kmp_yield_off_count;
        } else {
          __kmp_yielding_on = 1;
          kmp_lithe_mon_yield_count = __kmp_yield_on_count;
        }
        kmp_lithe_mon_yield_cycles = 0;
      }
    } else {
      __kmp_yielding_on = 1;
    }

    TCW_4(__kmp_global.g.g_time.dt.t_value,
          TCR_4(__kmp_global.g.g_time.dt.t_value) + 1);
    KMP_MB();

    if (!TCR_4(__kmp_global.g.g_done)) {
      uint64_t us = kmp_lithe_mon_interval_usec();
      set_awaiter_rel(&kmp_lithe_mon_alarm, us);
      set_alarm(&kmp_lithe_mon_alarm);
    }
  }
}

void __kmp_lithe_parlib_monitor_start(kmp_info_t *th) {
  (void)th;

  kmp_lithe_mon_alarm_setup();

  if (__kmp_yield_cycle) {
    __kmp_yielding_on = 0;
    kmp_lithe_mon_yield_count = __kmp_yield_off_count;
  } else {
    __kmp_yielding_on = 1;
  }
  kmp_lithe_mon_yield_cycles = 0;

  uint64_t us = kmp_lithe_mon_interval_usec();
  set_awaiter_rel(&kmp_lithe_mon_alarm, us);
  set_alarm(&kmp_lithe_mon_alarm);
}

void __kmp_lithe_parlib_monitor_stop(void) {
  (void)unset_alarm(&kmp_lithe_mon_alarm);
}

} // extern "C"

#else /* !LIBOMP_USE_LITHE || !KMP_USE_MONITOR */

extern "C" {

void __kmp_lithe_parlib_monitor_start(kmp_info_t *th) {
  (void)th;
}
void __kmp_lithe_parlib_monitor_stop(void) {}

}

#endif /* LIBOMP_USE_LITHE && KMP_USE_MONITOR */
