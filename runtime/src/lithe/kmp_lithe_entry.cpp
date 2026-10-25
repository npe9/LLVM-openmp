/*
 * kmp_lithe_entry.cpp - Entry points for OpenMP runtime integration with Lithe
 * When using Lithe, z_Linux_asm.S is not built, so we provide __kmp_hardware_timestamp here.
 */

#include <cstdio>
#include <lithe/sched_trace.h>
#include <cstdlib>
#include "kmp_lithe.h"
#include "kmp_i18n.h"
#include "kmp_io.h"
#include "kmp_os.h"
#include "kmp.h"  // for kmp_cpuid_t, __kmp_x86_cpuid declaration

// Global Lithe scheduler for OpenMP
static kmp_lithe_scheduler_t __kmp_lithe_scheduler;

// The unnamed critical section - replaces the assembly definition in z_Linux_asm.S
// This is the actual critical section lock
static kmp_critical_name __kmp_gomp_critical_user = {0};

// Global pointer to the unnamed critical section - referenced by kmp_gsupport.cpp
extern "C" kmp_critical_name *__kmp_unnamed_critical_addr = &__kmp_gomp_critical_user;

// Flag to indicate if Lithe integration is initialized
static int __kmp_lithe_initialized = 0;

// Entry point for microtask invocation via Lithe.
//
// In a normal libomp build this is implemented in z_Linux_asm.S and forwards an
// ARBITRARY number of outlined-region arguments (argc of them, from p_argv[])
// to the variadic microtask. The Lithe build does not assemble z_Linux_asm.S,
// so this C replacement must reproduce that contract. The previous version
// hardcoded exactly 4 arguments (p_argv[0..3]) regardless of argc: any OpenMP
// region capturing more than 4 shared variables (e.g. a persistent-region CG
// kernel with ~14 shared arrays/scalars) passed garbage pointers for args 5+,
// causing a segfault the first time a worker touched one. Forward all argc
// arguments via a switch (matches the generic non-asm libomp invoker).
extern "C" int
__kmp_invoke_microtask(microtask_t pkfn, int gtid, int npr, int argc, void *p_argv[],
                       void **exit_frame) {
  /* Perfetto trace point: one "run" slice per thread per region, primary
   * included (the primary's microtask is invoked from here as well). */
  struct RunScope {
    int gtid; bool on;
    RunScope(int g, int t) : gtid(g), on(lithe_sched_trace_enabled() != 0) {
      if (on) lithe_sched_trace_emit(LITHE_ST_OMP_RUN_BEGIN, NULL, (uint32_t)gtid, 0, (int32_t)t);
    }
    ~RunScope() { if (on) lithe_sched_trace_emit(LITHE_ST_OMP_RUN_END, NULL, (uint32_t)gtid, 0, 0); }
  } _lt_run(gtid, npr);

    if (exit_frame)
        *exit_frame = NULL;
    void **a = p_argv;
    switch (argc) {
    case 0: (*pkfn)(&gtid, &npr); break;
    case 1: (*pkfn)(&gtid, &npr, a[0]); break;
    case 2: (*pkfn)(&gtid, &npr, a[0], a[1]); break;
    case 3: (*pkfn)(&gtid, &npr, a[0], a[1], a[2]); break;
    case 4: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3]); break;
    case 5: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4]); break;
    case 6: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5]); break;
    case 7: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6]); break;
    case 8: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]); break;
    case 9: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]); break;
    case 10: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]); break;
    case 11: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10]); break;
    case 12: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11]); break;
    case 13: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12]); break;
    case 14: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13]); break;
    case 15: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14]); break;
    case 16: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]); break;
    case 17: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16]); break;
    case 18: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17]); break;
    case 19: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18]); break;
    case 20: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19]); break;
    case 21: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20]); break;
    case 22: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21]); break;
    case 23: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22]); break;
    case 24: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23]); break;
    case 25: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24]); break;
    case 26: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25]); break;
    case 27: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26]); break;
    case 28: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27]); break;
    case 29: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28]); break;
    case 30: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29]); break;
    case 31: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30]); break;
    case 32: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31]); break;
    case 33: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32]); break;
    case 34: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33]); break;
    case 35: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34]); break;
    case 36: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35]); break;
    case 37: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36]); break;
    case 38: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37]); break;
    case 39: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38]); break;
    case 40: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39]); break;
    case 41: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40]); break;
    case 42: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41]); break;
    case 43: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42]); break;
    case 44: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43]); break;
    case 45: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44]); break;
    case 46: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45]); break;
    case 47: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46]); break;
    case 48: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47]); break;
    case 49: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48]); break;
    case 50: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49]); break;
    case 51: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50]); break;
    case 52: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51]); break;
    case 53: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52]); break;
    case 54: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53]); break;
    case 55: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54]); break;
    case 56: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55]); break;
    case 57: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55], a[56]); break;
    case 58: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55], a[56], a[57]); break;
    case 59: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55], a[56], a[57], a[58]); break;
    case 60: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55], a[56], a[57], a[58], a[59]); break;
    case 61: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55], a[56], a[57], a[58], a[59], a[60]); break;
    case 62: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55], a[56], a[57], a[58], a[59], a[60], a[61]); break;
    case 63: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55], a[56], a[57], a[58], a[59], a[60], a[61], a[62]); break;
    case 64: (*pkfn)(&gtid, &npr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55], a[56], a[57], a[58], a[59], a[60], a[61], a[62], a[63]); break;
    default:
        // The non-asm Lithe invoker enumerates a fixed maximum of region
        // arguments (cases generated, 0..64). It was 20 until 2026-10-24: MKL's
        // threaded LAPACK (libmkl_intel_thread under scipy.linalg.lu_factor in
        // lpy) passes 32, and aborted here. Fail loudly above 64 rather than
        // forward garbage (the old 4-arg bug); do NOT fall back to an ABI
        // assumption.
        fprintf(stderr,
                "[LITHE-OMP] FATAL: __kmp_invoke_microtask argc=%d exceeds the "
                "supported maximum (64). Extend the switch in kmp_lithe_entry.cpp.\n",
                argc);
        abort();
    }
    return 1;
}

// Initialize the OpenMP runtime with Lithe integration
extern "C" void
__kmp_lithe_initialize(void) {
    if (__kmp_lithe_initialized) {
        return;
    }
    
    // Initialize Lithe runtime
    __kmp_lithe_runtime_initialize();
    
    // Initialize the Lithe scheduler for OpenMP
    __kmp_lithe_scheduler_init(&__kmp_lithe_scheduler, __kmp_threads[0]);
    
    // Enter the Lithe scheduler
    lithe_sched_enter((lithe_sched_t *)&__kmp_lithe_scheduler);
    
    __kmp_lithe_initialized = 1;
    
    // // // // KMP_INFORM(LitheInitialized, "KMP_LITHE");
}

// Finalize the OpenMP runtime with Lithe integration
extern "C" void
__kmp_lithe_finalize(void) {
    if (!__kmp_lithe_initialized) {
        return;
    }
    
    // Exit the Lithe scheduler
    lithe_sched_exit();
    
    // Finalize the Lithe scheduler for OpenMP
    __kmp_lithe_scheduler_finalize(&__kmp_lithe_scheduler);
    
    // Finalize Lithe runtime
    __kmp_lithe_runtime_finalize();
    
    __kmp_lithe_initialized = 0;
    
    // // // // KMP_INFORM(LitheFinalized, "KMP_LITHE");
}

// Create worker threads using Lithe
extern "C" int
__kmp_lithe_fork_call(int argc, microtask_t microtask, int gtid, void *wrapper_argv[]) {
    int nthreads = __kmp_threads[gtid]->th.th_team->t.t_nproc;
    
    // // // // KMP_INFORM(LitheForkCall, "KMP_LITHE", nthreads);
    
    // Request harts from Lithe
    int granted_harts = __kmp_lithe_request_harts(&__kmp_lithe_scheduler, nthreads - 1);
    
    // Create worker threads using Lithe
    for (int i = 1; i < nthreads; i++) {
        if (!__kmp_lithe_create_worker(&__kmp_lithe_scheduler, __kmp_threads[gtid]->th.th_team->t.t_threads[i]->th.th_info.ds.ds_gtid)) {
            // // // // KMP_WARNING(CantCreateWorkerThread);
        }
    }
    
    // Execute the microtask on the master thread
    (*microtask)(&gtid, &gtid, argc, wrapper_argv);
    
    return 1;
}

// Join worker threads using Lithe
extern "C" void
__kmp_lithe_join_call(int gtid) {
    // // // // KMP_INFORM(LitheJoinCall, "KMP_LITHE", gtid);
    
    // Wait for all worker threads to complete
    // This is handled by the OpenMP runtime's barrier mechanism
    
    // Yield harts back to Lithe
    if (gtid == 0) {
        __kmp_lithe_yield_hart(&__kmp_lithe_scheduler);
    }
}

// Lithe-specific implementation of unnamed critical section
extern "C" kmp_critical_name *
__kmp_lithe_get_unnamed_critical_addr(void) {
    // This is referenced by the assembly code
    static kmp_critical_name __kmp_unnamed_critical_addr;
    return &__kmp_unnamed_critical_addr;
}

// Replaces symbols from z_Linux_asm.S when Lithe is used (asm file not built)
#if defined(__x86_64__) || defined(__i386__)
extern "C" kmp_uint64 __kmp_hardware_timestamp(void) {
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return (kmp_uint64)lo | ((kmp_uint64)hi << 32);
}

extern "C" void __kmp_x86_cpuid(int mode, int mode2, kmp_cpuid_t *p) {
    unsigned eax = (unsigned)mode, ebx, ecx = (unsigned)mode2, edx;
    __asm__ __volatile__ ("cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(eax), "c"(ecx));
    p->eax = eax;
    p->ebx = ebx;
    p->ecx = ecx;
    p->edx = edx;
}

extern "C" void __kmp_load_x87_fpu_control_word(kmp_int16 *p) {
    __asm__ __volatile__ ("fldcw %0" : : "m"(*p));
}
extern "C" void __kmp_store_x87_fpu_control_word(kmp_int16 *p) {
    __asm__ __volatile__ ("fstcw %0" : "=m"(*p));
}
extern "C" void __kmp_clear_x87_fpu_status_word(void) {
    __asm__ __volatile__ ("fnclex");
}
#endif
