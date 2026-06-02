// Module/context-owned ASR dialect side storage (V1).
#include "asr_dialect_module_storage.h"

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
    MLIR_OpHandle parent;
    MLIR_OpHandle *stmts;
    size_t n_stmts;
} BodyEntry;

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
    BodyEntry *bodies;
    size_t n_bodies;
    size_t cap_bodies;
    TypeInfoEntry *types;
    size_t n_types;
    size_t cap_types;
} ASR_ModuleStorage;

static ASR_ModuleStorage *active_storage = NULL;

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
    ASR_ModuleStorageClear(ctx);
    active_storage = (ASR_ModuleStorage *)calloc(1, sizeof(ASR_ModuleStorage));
    if (active_storage) {
        active_storage->ctx = ctx;
    }
}

void ASR_ModuleStorageClear(MLIR_Context *ctx) {
    (void)ctx;
    if (!active_storage) {
        return;
    }
    for (size_t i = 0; i < active_storage->n_field_op_seqs; ++i) {
        free(active_storage->field_op_seqs[i].ops);
    }
    for (size_t i = 0; i < active_storage->n_bodies; ++i) {
        free(active_storage->bodies[i].stmts);
    }
    free(active_storage->field_ops);
    free(active_storage->field_op_seqs);
    free(active_storage->bodies);
    free(active_storage->types);
    free(active_storage);
    active_storage = NULL;
}

void ASR_ModuleStorageSetFieldOp(
    MLIR_OpHandle parent, const char *field, MLIR_OpHandle child) {
    if (!active_storage || !field) {
        return;
    }
    for (size_t i = 0; i < active_storage->n_field_ops; ++i) {
        if (active_storage->field_ops[i].parent == parent &&
                strcmp(active_storage->field_ops[i].field, field) == 0) {
            active_storage->field_ops[i].child = child;
            return;
        }
    }
    storage_grow((void **)&active_storage->field_ops, sizeof(FieldOpEntry),
        &active_storage->n_field_ops, &active_storage->cap_field_ops, 64);
    if (active_storage->n_field_ops >= active_storage->cap_field_ops) {
        return;
    }
    FieldOpEntry *e = &active_storage->field_ops[active_storage->n_field_ops++];
    e->parent = parent;
    snprintf(e->field, sizeof(e->field), "%s", field);
    e->child = child;
}

MLIR_OpHandle ASR_ModuleStorageGetFieldOp(
    MLIR_OpHandle parent, const char *field) {
    if (!active_storage || !field) {
        return MLIR_INVALID_HANDLE;
    }
    for (size_t i = 0; i < active_storage->n_field_ops; ++i) {
        if (active_storage->field_ops[i].parent == parent &&
                strcmp(active_storage->field_ops[i].field, field) == 0) {
            return active_storage->field_ops[i].child;
        }
    }
    return MLIR_INVALID_HANDLE;
}

void ASR_ModuleStorageSetFieldOpSeq(
    MLIR_OpHandle parent, const char *field,
    MLIR_OpHandle *ops, size_t n) {
    if (!active_storage || !field) {
        return;
    }
    for (size_t i = 0; i < active_storage->n_field_op_seqs; ++i) {
        if (active_storage->field_op_seqs[i].parent == parent &&
                strcmp(active_storage->field_op_seqs[i].field, field) == 0) {
            free(active_storage->field_op_seqs[i].ops);
            active_storage->field_op_seqs[i].ops = NULL;
            active_storage->field_op_seqs[i].n_ops = 0;
            if (n > 0 && ops) {
                active_storage->field_op_seqs[i].ops =
                    (MLIR_OpHandle *)malloc(n * sizeof(MLIR_OpHandle));
                if (active_storage->field_op_seqs[i].ops) {
                    memcpy(active_storage->field_op_seqs[i].ops, ops,
                        n * sizeof(MLIR_OpHandle));
                    active_storage->field_op_seqs[i].n_ops = n;
                }
            }
            return;
        }
    }
    storage_grow((void **)&active_storage->field_op_seqs,
        sizeof(FieldOpSeqEntry), &active_storage->n_field_op_seqs,
        &active_storage->cap_field_op_seqs, 32);
    if (active_storage->n_field_op_seqs >= active_storage->cap_field_op_seqs) {
        return;
    }
    FieldOpSeqEntry *e =
        &active_storage->field_op_seqs[active_storage->n_field_op_seqs++];
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
    if (!active_storage || !field) {
        return NULL;
    }
    for (size_t i = 0; i < active_storage->n_field_op_seqs; ++i) {
        if (active_storage->field_op_seqs[i].parent == parent &&
                strcmp(active_storage->field_op_seqs[i].field, field) == 0) {
            if (n_out) {
                *n_out = active_storage->field_op_seqs[i].n_ops;
            }
            return active_storage->field_op_seqs[i].ops;
        }
    }
    return NULL;
}

void ASR_ModuleStorageSetBody(
    MLIR_OpHandle parent, MLIR_OpHandle *stmts, size_t n) {
    if (!active_storage) {
        return;
    }
    for (size_t i = 0; i < active_storage->n_bodies; ++i) {
        if (active_storage->bodies[i].parent == parent) {
            free(active_storage->bodies[i].stmts);
            active_storage->bodies[i].stmts = NULL;
            active_storage->bodies[i].n_stmts = 0;
            if (n > 0 && stmts) {
                active_storage->bodies[i].stmts =
                    (MLIR_OpHandle *)malloc(n * sizeof(MLIR_OpHandle));
                if (active_storage->bodies[i].stmts) {
                    memcpy(active_storage->bodies[i].stmts, stmts,
                        n * sizeof(MLIR_OpHandle));
                    active_storage->bodies[i].n_stmts = n;
                }
            }
            return;
        }
    }
    storage_grow((void **)&active_storage->bodies, sizeof(BodyEntry),
        &active_storage->n_bodies, &active_storage->cap_bodies, 16);
    if (active_storage->n_bodies >= active_storage->cap_bodies) {
        return;
    }
    BodyEntry *e = &active_storage->bodies[active_storage->n_bodies++];
    e->parent = parent;
    e->stmts = NULL;
    e->n_stmts = 0;
    if (n > 0 && stmts) {
        e->stmts = (MLIR_OpHandle *)malloc(n * sizeof(MLIR_OpHandle));
        if (e->stmts) {
            memcpy(e->stmts, stmts, n * sizeof(MLIR_OpHandle));
            e->n_stmts = n;
        }
    }
}

size_t ASR_ModuleStorageBodyCount(MLIR_OpHandle parent) {
    if (!active_storage) {
        return 0;
    }
    for (size_t i = 0; i < active_storage->n_bodies; ++i) {
        if (active_storage->bodies[i].parent == parent) {
            return active_storage->bodies[i].n_stmts;
        }
    }
    return 0;
}

MLIR_OpHandle ASR_ModuleStorageBodyOp(MLIR_OpHandle parent, size_t index) {
    if (!active_storage) {
        return MLIR_INVALID_HANDLE;
    }
    for (size_t i = 0; i < active_storage->n_bodies; ++i) {
        if (active_storage->bodies[i].parent == parent) {
            if (index < active_storage->bodies[i].n_stmts) {
                return active_storage->bodies[i].stmts[index];
            }
            return MLIR_INVALID_HANDLE;
        }
    }
    return MLIR_INVALID_HANDLE;
}

void ASR_ModuleStorageSetTypeInfo(
    MLIR_TypeHandle ty, int64_t asr_kind, bool is_array, int64_t array_len) {
    if (!active_storage || ty == MLIR_INVALID_HANDLE) {
        return;
    }
    for (size_t i = 0; i < active_storage->n_types; ++i) {
        if (active_storage->types[i].type == ty) {
            active_storage->types[i].asr_kind = asr_kind;
            active_storage->types[i].is_array = is_array;
            active_storage->types[i].array_len = array_len;
            return;
        }
    }
    storage_grow((void **)&active_storage->types, sizeof(TypeInfoEntry),
        &active_storage->n_types, &active_storage->cap_types, 32);
    if (active_storage->n_types >= active_storage->cap_types) {
        return;
    }
    TypeInfoEntry *e = &active_storage->types[active_storage->n_types++];
    e->type = ty;
    e->asr_kind = asr_kind;
    e->is_array = is_array;
    e->array_len = array_len;
}

bool ASR_ModuleStorageGetTypeInfo(
    MLIR_TypeHandle ty, int64_t *asr_kind_out,
    bool *is_array_out, int64_t *array_len_out) {
    if (!active_storage || ty == MLIR_INVALID_HANDLE) {
        return false;
    }
    for (size_t i = 0; i < active_storage->n_types; ++i) {
        if (active_storage->types[i].type == ty) {
            if (asr_kind_out) {
                *asr_kind_out = active_storage->types[i].asr_kind;
            }
            if (is_array_out) {
                *is_array_out = active_storage->types[i].is_array;
            }
            if (array_len_out) {
                *array_len_out = active_storage->types[i].array_len;
            }
            return true;
        }
    }
    return false;
}
