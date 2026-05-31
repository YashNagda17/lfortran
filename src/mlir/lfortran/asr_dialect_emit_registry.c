#include "asr_dialect_emit_registry.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    MLIR_OpHandle do_op;
    MLIR_OpHandle *stmts;
    size_t n_stmts;
} DoLoopBodyEntry;

static DoLoopBodyEntry *entries = NULL;
static size_t n_entries = 0;
static size_t cap_entries = 0;

void ASR_DialectEmitRegistryClear(void) {
    for (size_t i = 0; i < n_entries; ++i) {
        free(entries[i].stmts);
    }
    free(entries);
    entries = NULL;
    n_entries = 0;
    cap_entries = 0;
}

void ASR_DialectEmitRegistryAddDoLoopBody(
    MLIR_OpHandle do_op, MLIR_OpHandle *stmts, size_t n_stmts) {
    if (n_entries == cap_entries) {
        size_t new_cap = cap_entries ? cap_entries * 2 : 16;
        DoLoopBodyEntry *new_entries = (DoLoopBodyEntry *)realloc(
            entries, new_cap * sizeof(DoLoopBodyEntry));
        if (!new_entries) {
            return;
        }
        entries = new_entries;
        cap_entries = new_cap;
    }
    MLIR_OpHandle *copy = NULL;
    if (n_stmts > 0) {
        copy = (MLIR_OpHandle *)malloc(n_stmts * sizeof(MLIR_OpHandle));
        if (!copy) {
            return;
        }
        memcpy(copy, stmts, n_stmts * sizeof(MLIR_OpHandle));
    }
    entries[n_entries].do_op = do_op;
    entries[n_entries].stmts = copy;
    entries[n_entries].n_stmts = n_stmts;
    n_entries++;
}

size_t ASR_DialectEmitRegistryDoLoopBodyCount(MLIR_OpHandle do_op) {
    for (size_t i = 0; i < n_entries; ++i) {
        if (entries[i].do_op == do_op) {
            return entries[i].n_stmts;
        }
    }
    return 0;
}

MLIR_OpHandle ASR_DialectEmitRegistryDoLoopBodyOp(
    MLIR_OpHandle do_op, size_t index) {
    for (size_t i = 0; i < n_entries; ++i) {
        if (entries[i].do_op == do_op) {
            if (index < entries[i].n_stmts) {
                return entries[i].stmts[index];
            }
            return MLIR_INVALID_HANDLE;
        }
    }
    return MLIR_INVALID_HANDLE;
}
