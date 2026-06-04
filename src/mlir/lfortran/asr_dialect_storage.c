// Internal ASR dialect side storage (V1) — see asr_dialect_storage.h.
#include "asr_dialect_storage.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    MLIR_OpHandle parent;
    char field[64];
    MLIR_OpHandle child;
} FieldOpEntry;

typedef struct {
    MLIR_OpHandle parent;
    char field[64];
    MLIR_OpHandle *ops;
    size_t n_ops;
} FieldOpSeqEntry;

typedef struct {
    MLIR_TypeHandle type;
    int64_t asr_kind;
    bool is_array;
    int64_t array_len;
} TypeInfoEntry;

typedef struct {
    MLIR_Context *ctx;
    FieldOpEntry *field_ops;
    size_t n_field_ops;
    size_t cap_field_ops;
    FieldOpSeqEntry *field_op_seqs;
    size_t n_field_op_seqs;
    size_t cap_field_op_seqs;
    TypeInfoEntry *types;
    size_t n_types;
    size_t cap_types;
} ASR_ModuleStorage;

/* Active ctx for APIs that only receive op handles (emitter / inline readers). */
static MLIR_Context *active_asr_ctx = NULL;

static ASR_ModuleStorage *storage_for_ctx(MLIR_Context *ctx) {
    if (!ctx) {
        ctx = active_asr_ctx;
    }
    if (!ctx) {
        return NULL;
    }
    return (ASR_ModuleStorage *)ctx->asr_module_storage;
}

static void storage_grow(void **arr, size_t elem_size,
        size_t *n, size_t *cap, size_t min_cap) {
    if (*n < *cap) {
        return;
    }
    size_t new_cap = *cap ? *cap * 2 : min_cap;
    void *p = realloc(*arr, new_cap * elem_size);
    if (!p) {
        return;
    }
    *arr = p;
    *cap = new_cap;
}

void ASR_ModuleStorageInit(MLIR_Context *ctx) {
    if (!ctx) {
        return;
    }
    ASR_ModuleStorageClear(ctx);
    ASR_ModuleStorage *st = (ASR_ModuleStorage *)calloc(1, sizeof(ASR_ModuleStorage));
    if (!st) {
        return;
    }
    st->ctx = ctx;
    ctx->asr_module_storage = st;
    active_asr_ctx = ctx;
}

void ASR_ModuleStorageClear(MLIR_Context *ctx) {
    ASR_ModuleStorage *st = storage_for_ctx(ctx);
    if (!st) {
        return;
    }
    if (st->ctx) {
        st->ctx->asr_module_storage = NULL;
    }
    if (active_asr_ctx == st->ctx) {
        active_asr_ctx = NULL;
    }
    for (size_t i = 0; i < st->n_field_op_seqs; ++i) {
        free(st->field_op_seqs[i].ops);
    }
    free(st->field_ops);
    free(st->field_op_seqs);
    free(st->types);
    free(st);
}

void ASR_ModuleStorageSetFieldOp(
    MLIR_OpHandle parent, const char *field, MLIR_OpHandle child) {
    ASR_ModuleStorage *st = storage_for_ctx(NULL);
    if (!st || !field) {
        return;
    }
    for (size_t i = 0; i < st->n_field_ops; ++i) {
        if (st->field_ops[i].parent == parent &&
                strcmp(st->field_ops[i].field, field) == 0) {
            st->field_ops[i].child = child;
            return;
        }
    }
    storage_grow((void **)&st->field_ops, sizeof(FieldOpEntry),
        &st->n_field_ops, &st->cap_field_ops, 64);
    if (st->n_field_ops >= st->cap_field_ops) {
        return;
    }
    FieldOpEntry *e = &st->field_ops[st->n_field_ops++];
    e->parent = parent;
    snprintf(e->field, sizeof(e->field), "%s", field);
    e->child = child;
}

MLIR_OpHandle ASR_ModuleStorageGetFieldOp(
    MLIR_OpHandle parent, const char *field) {
    ASR_ModuleStorage *st = storage_for_ctx(NULL);
    if (!st || !field) {
        return MLIR_INVALID_HANDLE;
    }
    for (size_t i = 0; i < st->n_field_ops; ++i) {
        if (st->field_ops[i].parent == parent &&
                strcmp(st->field_ops[i].field, field) == 0) {
            return st->field_ops[i].child;
        }
    }
    return MLIR_INVALID_HANDLE;
}

void ASR_ModuleStorageSetFieldOpSeq(
    MLIR_OpHandle parent, const char *field,
    MLIR_OpHandle *ops, size_t n) {
    ASR_ModuleStorage *st = storage_for_ctx(NULL);
    if (!st || !field) {
        return;
    }
    for (size_t i = 0; i < st->n_field_op_seqs; ++i) {
        if (st->field_op_seqs[i].parent == parent &&
                strcmp(st->field_op_seqs[i].field, field) == 0) {
            free(st->field_op_seqs[i].ops);
            st->field_op_seqs[i].ops = NULL;
            st->field_op_seqs[i].n_ops = 0;
            if (n > 0 && ops) {
                st->field_op_seqs[i].ops =
                    (MLIR_OpHandle *)malloc(n * sizeof(MLIR_OpHandle));
                if (st->field_op_seqs[i].ops) {
                    memcpy(st->field_op_seqs[i].ops, ops,
                        n * sizeof(MLIR_OpHandle));
                    st->field_op_seqs[i].n_ops = n;
                }
            }
            return;
        }
    }
    storage_grow((void **)&st->field_op_seqs,
        sizeof(FieldOpSeqEntry), &st->n_field_op_seqs,
        &st->cap_field_op_seqs, 32);
    if (st->n_field_op_seqs >= st->cap_field_op_seqs) {
        return;
    }
    FieldOpSeqEntry *e = &st->field_op_seqs[st->n_field_op_seqs++];
    e->parent = parent;
    snprintf(e->field, sizeof(e->field), "%s", field);
    e->ops = NULL;
    e->n_ops = 0;
    if (n > 0 && ops) {
        e->ops = (MLIR_OpHandle *)malloc(n * sizeof(MLIR_OpHandle));
        if (e->ops) {
            memcpy(e->ops, ops, n * sizeof(MLIR_OpHandle));
            e->n_ops = n;
        }
    }
}

MLIR_OpHandle *ASR_ModuleStorageGetFieldOpSeq(
    MLIR_OpHandle parent, const char *field, size_t *n_out) {
    if (n_out) {
        *n_out = 0;
    }
    ASR_ModuleStorage *st = storage_for_ctx(NULL);
    if (!st || !field) {
        return NULL;
    }
    for (size_t i = 0; i < st->n_field_op_seqs; ++i) {
        if (st->field_op_seqs[i].parent == parent &&
                strcmp(st->field_op_seqs[i].field, field) == 0) {
            if (n_out) {
                *n_out = st->field_op_seqs[i].n_ops;
            }
            return st->field_op_seqs[i].ops;
        }
    }
    return NULL;
}

void ASR_ModuleStorageSetTypeInfo(
    MLIR_TypeHandle ty, int64_t asr_kind, bool is_array, int64_t array_len) {
    ASR_ModuleStorage *st = storage_for_ctx(NULL);
    if (!st || ty == MLIR_INVALID_HANDLE) {
        return;
    }
    for (size_t i = 0; i < st->n_types; ++i) {
        if (st->types[i].type == ty) {
            st->types[i].asr_kind = asr_kind;
            st->types[i].is_array = is_array;
            st->types[i].array_len = array_len;
            return;
        }
    }
    storage_grow((void **)&st->types, sizeof(TypeInfoEntry),
        &st->n_types, &st->cap_types, 32);
    if (st->n_types >= st->cap_types) {
        return;
    }
    TypeInfoEntry *e = &st->types[st->n_types++];
    e->type = ty;
    e->asr_kind = asr_kind;
    e->is_array = is_array;
    e->array_len = array_len;
}

bool ASR_ModuleStorageGetTypeInfo(
    MLIR_TypeHandle ty, int64_t *asr_kind_out,
    bool *is_array_out, int64_t *array_len_out) {
    ASR_ModuleStorage *st = storage_for_ctx(NULL);
    if (!st || ty == MLIR_INVALID_HANDLE) {
        return false;
    }
    for (size_t i = 0; i < st->n_types; ++i) {
        if (st->types[i].type == ty) {
            if (asr_kind_out) {
                *asr_kind_out = st->types[i].asr_kind;
            }
            if (is_array_out) {
                *is_array_out = st->types[i].is_array;
            }
            if (array_len_out) {
                *array_len_out = st->types[i].array_len;
            }
            return true;
        }
    }
    return false;
}
