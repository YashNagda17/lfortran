// Shared ASR dialect field readers (V1: module storage for op refs, attrs for metadata).
#pragma once

#include "asr_dialect_api.h"
#include "asr_dialect_module_storage.h"

#include <stdio.h>
#include <string.h>

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
    return asr_get_field_i64(op, "n_args", 0);
}

static inline int64_t asr_get_array_len_attr(MLIR_OpHandle var_op) {
    MLIR_TypeHandle ty = asr_get_field_type(var_op, "type");
    int64_t kind = 0;
    bool is_array = false;
    int64_t len = 0;
    if (ASR_ModuleStorageGetTypeInfo(ty, &kind, &is_array, &len) && is_array) {
        return len;
    }
    return asr_get_field_i64(var_op, "array_len", 0);
}

static inline size_t asr_get_seq_n_args_attr(MLIR_OpHandle op) {
    size_t n = asr_get_field_op_seq_count(op, "args");
    if (n > 0) {
        return n;
    }
    return (size_t)asr_get_field_i64(op, "n_args", 0);
}

static inline size_t asr_get_body_count(MLIR_OpHandle op) {
    return ASR_ModuleStorageBodyCount(op);
}

static inline MLIR_OpHandle asr_get_body_op(MLIR_OpHandle op, size_t index) {
    return ASR_ModuleStorageBodyOp(op, index);
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
