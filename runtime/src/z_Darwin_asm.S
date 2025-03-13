//===-- z_Darwin_asm.S - macOS ARM64 assembly routines ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//

// This file contains macOS-specific assembly routines for ARM64 architecture

#include "kmp_config.h"

#if KMP_OS_DARWIN && KMP_ARCH_AARCH64

// Define the global symbols needed by the runtime
.globl ___kmp_invoke_microtask
.globl ___kmp_unnamed_critical_addr

.data
// Define the unnamed critical section address
___kmp_unnamed_critical_addr:
    .space 8

.text
// Implementation of __kmp_invoke_microtask for ARM64
// This is a simplified implementation
___kmp_invoke_microtask:
    // Save frame pointer and link register
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    
    // Call the microtask function
    blr     x2
    
    // Restore frame pointer and link register
    ldp     x29, x30, [sp], #16
    ret

#endif // KMP_OS_DARWIN && KMP_ARCH_AARCH64
