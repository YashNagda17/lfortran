// LFortran-specific MLIR lowering and LLVM IR translation hooks.
// Linked into lfortran_mlir_c_api_core.

#ifndef MLIR_LFORTRAN_HOOKS_H
#define MLIR_LFORTRAN_HOOKS_H

#include <stdbool.h>
#include <stddef.h>

#include "mlir_api.h"

typedef struct MLIR_LFortranLowerState {
    MLIR_Context *ctx;
    MLIR_OpHandle module;
    MLIR_BlockHandle module_body;
    bool printf_decl;
    bool vp_fmt_i64_nl;
    bool vp_fmt_g_nl;
} MLIR_LFortranLowerState;

void MLIR_LFortranLowerStateInit(MLIR_LFortranLowerState *st,
        MLIR_Context *ctx, MLIR_OpHandle module);

// Per-op hook: insert lowered LLVM-dialect ops at `pos` in `parent`.
// Return true when the op was handled (caller erases the original).
bool MLIR_LFortranTryLowerOp(MLIR_LFortranLowerState *st, MLIR_OpHandle op,
        MLIR_BlockHandle parent, size_t pos);

// Pre-pass for upstream lowering: rewrite vector.print to libc printf.
bool mlir_lower_vector_print_native(MLIR_LFortranLowerState *st);

typedef struct {
    char *data;
    size_t len, cap;
} MLIR_LFortranBuf;

typedef void (*MLIR_LFortranPrintTypeFn)(MLIR_LFortranBuf *out,
                                         MLIR_Context *ctx,
                                         MLIR_TypeHandle ty);

bool MLIR_LFortranPrintGepIndexType(MLIR_LFortranBuf *out, MLIR_Context *ctx,
                                    MLIR_TypeHandle ty,
                                    MLIR_LFortranPrintTypeFn default_print);

bool MLIR_LFortranShouldEmitBlockLabel(size_t block_index);

#endif
