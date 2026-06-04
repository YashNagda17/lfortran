// Internal ASR dialect storage and field access (V1).
//
// NOT part of the public compiler contract — use asr_dialect_api.h externally.
//
// Prototype contract:
// - Side tables live on MLIR_Context::asr_module_storage (see ASR_ModuleStorageInit).
// - ASR dialect text dumps are not reparsable without that storage for op refs/sequences.
// - Lifetime: one Init/Clear pair per ASR dialect build/lowering on a context.
// - Do not interleave two dialect builds on the same context without Clear between them.
// - Scope regions (asr.symtab/metadata/body) are unregistered container ops; child lists
//   are stored under field "ops" in side storage until true MLIR regions exist.
//
// Combines context-owned side tables with inline readers over storage + asr.* attrs.
#pragma once

#include "asr_dialect_api.h"

#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Module/context-owned side storage (implementation in asr_dialect_storage.c) ---

void ASR_ModuleStorageInit(MLIR_Context *ctx);
void ASR_ModuleStorageClear(MLIR_Context *ctx);

void ASR_ModuleStorageSetFieldOp(
    MLIR_OpHandle parent, const char *field, MLIR_OpHandle child);
MLIR_OpHandle ASR_ModuleStorageGetFieldOp(
    MLIR_OpHandle parent, const char *field);

void ASR_ModuleStorageSetFieldOpSeq(
    MLIR_OpHandle parent, const char *field,
    MLIR_OpHandle *ops, size_t n);
MLIR_OpHandle *ASR_ModuleStorageGetFieldOpSeq(
    MLIR_OpHandle parent, const char *field, size_t *n_out);

void ASR_ModuleStorageSetTypeInfo(
    MLIR_TypeHandle ty, int64_t asr_kind, bool is_array, int64_t array_len);
bool ASR_ModuleStorageGetTypeInfo(
    MLIR_TypeHandle ty, int64_t *asr_kind_out,
    bool *is_array_out, int64_t *array_len_out);

#ifdef __cplusplus
}
#endif

#define ASR_DIALECT_META_PREFIX "asr."

static inline MLIR_AttributeHandle asr_get_meta_attr(
        MLIR_OpHandle op, const char *field) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s%s", ASR_DIALECT_META_PREFIX, field);
    return MLIR_GetOpAttributeByName(op, buf);
}

static inline MLIR_AttributeHandle asr_get_field_attr(
        MLIR_OpHandle op, const char *field) {
    return asr_get_meta_attr(op, field);
}

static inline MLIR_OpHandle asr_get_field_op(MLIR_OpHandle op, const char *field) {
    MLIR_OpHandle stored = ASR_ModuleStorageGetFieldOp(op, field);
    if (stored != MLIR_INVALID_HANDLE) {
        return stored;
    }
    // Legacy fallback for generic/debug dumps that still use old attrs.
    char buf[128];
    snprintf(buf, sizeof(buf), "asr.f.%s", field);
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, buf);
    if (a == MLIR_INVALID_HANDLE) {
        return MLIR_INVALID_HANDLE;
    }
    return (MLIR_OpHandle)MLIR_GetAttributeInteger(a);
}

static inline int64_t asr_get_field_i64(
        MLIR_OpHandle op, const char *field, int64_t def) {
    MLIR_AttributeHandle a = asr_get_meta_attr(op, field);
    if (a == MLIR_INVALID_HANDLE) {
        char buf[128];
        snprintf(buf, sizeof(buf), "asr.f.%s", field);
        a = MLIR_GetOpAttributeByName(op, buf);
    }
    if (a == MLIR_INVALID_HANDLE) {
        return def;
    }
    return MLIR_GetAttributeInteger(a);
}

static inline bool asr_get_field_bool(
        MLIR_OpHandle op, const char *field, bool def) {
    MLIR_AttributeHandle a = asr_get_meta_attr(op, field);
    if (a == MLIR_INVALID_HANDLE) {
        return def;
    }
    return MLIR_GetAttributeBool(a);
}

static inline string asr_get_field_str(MLIR_OpHandle op, const char *field) {
    MLIR_AttributeHandle a = asr_get_meta_attr(op, field);
    if (a == MLIR_INVALID_HANDLE) {
        char buf[128];
        snprintf(buf, sizeof(buf), "asr.f.%s", field);
        a = MLIR_GetOpAttributeByName(op, buf);
    }
    if (a == MLIR_INVALID_HANDLE) {
        return str_lit("");
    }
    return MLIR_GetAttributeString(a);
}

static inline MLIR_TypeHandle asr_get_field_type(MLIR_OpHandle op, const char *field) {
    MLIR_AttributeHandle a = asr_get_meta_attr(op, field);
    if (a == MLIR_INVALID_HANDLE) {
        char buf[128];
        snprintf(buf, sizeof(buf), "asr.f.%s", field);
        a = MLIR_GetOpAttributeByName(op, buf);
    }
    if (a == MLIR_INVALID_HANDLE) {
        return MLIR_INVALID_HANDLE;
    }
    return MLIR_GetAttributeTypeValue(a);
}

static inline bool asr_field_attr_present(MLIR_OpHandle op, const char *field) {
    return asr_get_meta_attr(op, field) != MLIR_INVALID_HANDLE;
}

static inline MLIR_OpHandle *asr_get_field_op_seq(
        MLIR_OpHandle op, const char *field) {
    size_t n = 0;
    MLIR_OpHandle *stored = ASR_ModuleStorageGetFieldOpSeq(op, field, &n);
    if (stored) {
        return stored;
    }
    MLIR_AttributeHandle a = asr_get_meta_attr(op, field);
    if (a == MLIR_INVALID_HANDLE) {
        char buf[128];
        snprintf(buf, sizeof(buf), "asr.f.%s", field);
        a = MLIR_GetOpAttributeByName(op, buf);
    }
    if (a == MLIR_INVALID_HANDLE) {
        return NULL;
    }
    return (MLIR_OpHandle *)(uintptr_t)MLIR_GetAttributeInteger(a);
}

static inline size_t asr_get_field_op_seq_count(MLIR_OpHandle op, const char *field) {
    size_t n = 0;
    if (ASR_ModuleStorageGetFieldOpSeq(op, field, &n)) {
        return n;
    }
    return (size_t)asr_get_field_i64(op, "n_args", 0);
}

static inline int64_t asr_get_type_kind_attr(MLIR_OpHandle var_op) {
    if (asr_field_attr_present(var_op, "type_kind")) {
        return asr_get_field_i64(var_op, "type_kind", 0);
    }
    MLIR_TypeHandle ty = asr_get_field_type(var_op, "type");
    int64_t kind = 0;
    bool is_array = false;
    int64_t len = 0;
    if (ASR_ModuleStorageGetTypeInfo(ty, &kind, &is_array, &len)) {
        return kind;
    }
    return 0;
}

static inline int64_t asr_get_array_len_attr(MLIR_OpHandle var_op) {
    if (asr_field_attr_present(var_op, "array_len")) {
        return asr_get_field_i64(var_op, "array_len", 0);
    }
    MLIR_TypeHandle ty = asr_get_field_type(var_op, "type");
    int64_t kind = 0;
    bool is_array = false;
    int64_t len = 0;
    if (ASR_ModuleStorageGetTypeInfo(ty, &kind, &is_array, &len) && is_array) {
        return len;
    }
    return 0;
}

static inline size_t asr_get_seq_n_args_attr(MLIR_OpHandle op) {
    size_t n = asr_get_field_op_seq_count(op, "args");
    if (n > 0) {
        return n;
    }
    return (size_t)asr_get_field_i64(op, "n_args", 0);
}

static inline size_t asr_get_body_count(MLIR_OpHandle op) {
    return asr_get_field_op_seq_count(op, "body");
}

static inline MLIR_OpHandle asr_get_body_op(MLIR_OpHandle op, size_t index) {
    size_t n = 0;
    MLIR_OpHandle *seq = ASR_ModuleStorageGetFieldOpSeq(op, "body", &n);
    if (!seq || index >= n) {
        return MLIR_INVALID_HANDLE;
    }
    return seq[index];
}

static inline bool asr_op_name_is(MLIR_OpHandle op, const char *name) {
    string nm = MLIR_GetOpName(op);
    size_t n = 0;
    while (name[n]) {
        n++;
    }
    return nm.size == n && nm.str && memcmp(nm.str, name, n) == 0;
}

static inline MLIR_OpHandle asr_get_scope_region(MLIR_OpHandle scope_op,
        const char *field) {
    return asr_get_field_op(scope_op, field);
}

static inline MLIR_OpHandle *asr_get_scope_region_ops(MLIR_OpHandle region_op,
        size_t *n_out) {
    if (region_op == MLIR_INVALID_HANDLE) {
        if (n_out) {
            *n_out = 0;
        }
        return NULL;
    }
    return ASR_ModuleStorageGetFieldOpSeq(region_op, "ops", n_out);
}
