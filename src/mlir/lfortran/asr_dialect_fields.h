// Shared ASR dialect field readers (storage uses asr.f.* attributes; hybrid dump is separate).
#pragma once

#include "asr_dialect_api.h"

#include <stdio.h>
#include <string.h>

#define ASR_DIALECT_FIELD_PREFIX "asr.f."

static inline MLIR_AttributeHandle asr_get_field_attr(
        MLIR_OpHandle op, const char *field) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s%s", ASR_DIALECT_FIELD_PREFIX, field);
    return MLIR_GetOpAttributeByName(op, buf);
}

static inline MLIR_OpHandle asr_get_field_op(MLIR_OpHandle op, const char *field) {
    MLIR_AttributeHandle a = asr_get_field_attr(op, field);
    if (a == MLIR_INVALID_HANDLE) {
        return MLIR_INVALID_HANDLE;
    }
    return (MLIR_OpHandle)MLIR_GetAttributeInteger(a);
}

static inline int64_t asr_get_field_i64(
        MLIR_OpHandle op, const char *field, int64_t def) {
    MLIR_AttributeHandle a = asr_get_field_attr(op, field);
    if (a == MLIR_INVALID_HANDLE) {
        return def;
    }
    return MLIR_GetAttributeInteger(a);
}

static inline bool asr_get_field_bool(
        MLIR_OpHandle op, const char *field, bool def) {
    MLIR_AttributeHandle a = asr_get_field_attr(op, field);
    if (a == MLIR_INVALID_HANDLE) {
        return def;
    }
    return MLIR_GetAttributeBool(a);
}

static inline string asr_get_field_str(MLIR_OpHandle op, const char *field) {
    MLIR_AttributeHandle a = asr_get_field_attr(op, field);
    if (a == MLIR_INVALID_HANDLE) {
        return str_lit("");
    }
    return MLIR_GetAttributeString(a);
}

static inline MLIR_TypeHandle asr_get_field_type(MLIR_OpHandle op, const char *field) {
    MLIR_AttributeHandle a = asr_get_field_attr(op, field);
    if (a == MLIR_INVALID_HANDLE) {
        return MLIR_INVALID_HANDLE;
    }
    return MLIR_GetAttributeTypeValue(a);
}

static inline bool asr_field_attr_present(MLIR_OpHandle op, const char *field) {
    return asr_get_field_attr(op, field) != MLIR_INVALID_HANDLE;
}
