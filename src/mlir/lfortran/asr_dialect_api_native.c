// Native ASR dialect op storage, print, verify, and lowering driver (V1).
//
// Op identity comes from the MLIR operation name (asr.*), not an asr.op_kind
// attribute. Child expression/statement references and statement bodies live in
// module-owned side storage, not pointer-shaped integer attributes.
#include "asr_dialect_api.h"
#include "asr_dialect_storage.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <base/string.h>

#define ASR_META_PREFIX "asr."

static string asr_meta_attr_name(Arena *arena, const char *field_name) {
    size_t n = strlen(ASR_META_PREFIX) + strlen(field_name);
    char *buf = (char *)arena_alloc(arena, n + 1);
    memcpy(buf, ASR_META_PREFIX, strlen(ASR_META_PREFIX));
    memcpy(buf + strlen(ASR_META_PREFIX), field_name, strlen(field_name));
    buf[n] = '\0';
    return str_from_cstr_view(buf);
}

static bool field_is_op_ref(ASR_DialectFieldKind fk) {
    switch (fk) {
        case ASR_FIELD_EXPR:
        case ASR_FIELD_STMT:
        case ASR_FIELD_OP:
        case ASR_FIELD_EXPR_OPT:
        case ASR_FIELD_STMT_OPT:
        case ASR_FIELD_SYMBOL_OPT:
        case ASR_FIELD_NODE_OPT:
            return true;
        default:
            return false;
    }
}

static bool field_is_op_seq(ASR_DialectFieldKind fk) {
    switch (fk) {
        case ASR_FIELD_EXPR_SEQ:
        case ASR_FIELD_STMT_SEQ:
        case ASR_FIELD_SYMBOL_SEQ:
        case ASR_FIELD_NODE_SEQ:
        case ASR_FIELD_OP_SEQ:
            return true;
        default:
            return false;
    }
}

static MLIR_ValueHandle op_result_value(MLIR_OpHandle op) {
    if (op == MLIR_INVALID_HANDLE) {
        return MLIR_INVALID_HANDLE;
    }
    if (MLIR_GetOpNumResults(op) > 0) {
        return MLIR_GetOpResult(op, 0);
    }
    return MLIR_INVALID_HANDLE;
}

static void attach_expr_operands(MLIR_Context *ctx, MLIR_OpHandle op,
        const ASR_DialectField *fields, size_t n_fields) {
    MLIR_ValueHandle operands[8];
    size_t n_operands = 0;
    for (size_t i = 0; i < n_fields && n_operands < 8; ++i) {
        if (!field_is_op_ref(fields[i].kind)) {
            continue;
        }
        MLIR_ValueHandle v = op_result_value(fields[i].value.op);
        if (v != MLIR_INVALID_HANDLE) {
            operands[n_operands++] = v;
        }
    }
    if (n_operands > 0) {
        MLIR_SetOpOperands(ctx, op, operands, n_operands);
    }
}

static MLIR_AttributeHandle field_to_meta_attr(
    MLIR_Context *ctx, Arena *arena, const ASR_DialectField *field) {
    string aname = asr_meta_attr_name(arena, field->name);
    switch (field->kind) {
        case ASR_FIELD_I64:
        case ASR_FIELD_I64_OPT:
        case ASR_FIELD_VOID:
            return MLIR_CreateAttributeInteger(ctx, aname, field->value.i64,
                MLIR_CreateTypeInteger(ctx, 64, false));
        case ASR_FIELD_F64:
            return MLIR_CreateAttributeFloat(ctx, aname, field->value.f64,
                MLIR_CreateTypeFloat(ctx, 64, false));
        case ASR_FIELD_BOOL:
        case ASR_FIELD_BOOL_OPT:
            return MLIR_CreateAttributeBool(ctx, aname, field->value.b);
        case ASR_FIELD_STRING:
        case ASR_FIELD_IDENTIFIER:
        case ASR_FIELD_SYMBOL_REF:
        case ASR_FIELD_SYMBOL_REF_OPT:
        case ASR_FIELD_STRING_OPT:
        case ASR_FIELD_IDENTIFIER_OPT:
        case ASR_FIELD_SYMBOL_REF_SEQ:
        case ASR_FIELD_IDENTIFIER_SEQ:
        case ASR_FIELD_STRING_SEQ:
            return MLIR_CreateAttributeString(ctx, aname, field->value.str);
        case ASR_FIELD_TTYPE:
        case ASR_FIELD_TTYPE_OPT:
            return MLIR_CreateAttributeType(ctx, aname, field->value.type);
        default:
            return MLIR_INVALID_HANDLE;
    }
}

MLIR_OpHandle ASR_DialectCreateOpNative(
    MLIR_Context *ctx, ASR_DialectOpKind kind, MLIR_LocationHandle loc,
    const ASR_DialectField *fields, size_t n_fields) {
    const ASR_DialectOpSchema *schema = ASR_DialectLookupSchema(kind);
    if (!schema) {
        return MLIR_INVALID_HANDLE;
    }

    Arena *arena = ctx->arena;
    size_t n_attrs = 0;
    MLIR_AttributeHandle *attrs = NULL;
    if (n_fields > 0) {
        attrs = (MLIR_AttributeHandle *)arena_alloc(
            arena, n_fields * sizeof(MLIR_AttributeHandle));
        for (size_t i = 0; i < n_fields; ++i) {
            if (field_is_op_ref(fields[i].kind) || field_is_op_seq(fields[i].kind)) {
                continue;
            }
            MLIR_AttributeHandle a = field_to_meta_attr(ctx, arena, &fields[i]);
            if (a != MLIR_INVALID_HANDLE) {
                attrs[n_attrs++] = a;
            }
        }
    }

    MLIR_TypeHandle result_ty = MLIR_INVALID_HANDLE;
    if (schema->category == ASR_DIALECT_CATEGORY_EXPR ||
        schema->category == ASR_DIALECT_CATEGORY_TTYPE) {
        for (size_t i = 0; i < n_fields; ++i) {
            if (fields[i].kind == ASR_FIELD_TTYPE ||
                fields[i].kind == ASR_FIELD_TTYPE_OPT) {
                result_ty = fields[i].value.type;
                break;
            }
        }
    }

    MLIR_ValueHandle result = MLIR_INVALID_HANDLE;
    MLIR_TypeHandle rts[1] = {result_ty};
    MLIR_ValueHandle rs[1] = {MLIR_INVALID_HANDLE};
    if (result_ty != MLIR_INVALID_HANDLE) {
        char name_buf[32];
        snprintf(name_buf, sizeof(name_buf), "asr_v%zu", (size_t)kind);
        result = MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, result_ty,
            str_from_cstr_view(name_buf), loc);
        rs[0] = result;
    }

    string opname = str_from_cstr_len_view_const(
        schema->mlir_name, (uint64_t)strlen(schema->mlir_name));
    MLIR_OpHandle op = MLIR_CreateOp(ctx, OP_TYPE_UNREGISTERED, opname,
        attrs, n_attrs,
        result_ty != MLIR_INVALID_HANDLE ? rts : NULL,
        result_ty != MLIR_INVALID_HANDLE ? 1 : 0,
        result_ty != MLIR_INVALID_HANDLE ? rs : NULL,
        result_ty != MLIR_INVALID_HANDLE ? 1 : 0,
        NULL, 0, NULL, 0, loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    if (op == MLIR_INVALID_HANDLE) {
        return MLIR_INVALID_HANDLE;
    }

    for (size_t i = 0; i < n_fields; ++i) {
        if (field_is_op_seq(fields[i].kind)) {
            ASR_ModuleStorageSetFieldOpSeq(op, fields[i].name,
                fields[i].value.op_seq.items, fields[i].value.op_seq.n_items);
            continue;
        }
        if (field_is_op_ref(fields[i].kind) &&
                fields[i].value.op != MLIR_INVALID_HANDLE) {
            ASR_ModuleStorageSetFieldOp(op, fields[i].name, fields[i].value.op);
        }
    }

    attach_expr_operands(ctx, op, fields, n_fields);
    return op;
}

ASR_DialectOpKind ASR_DialectGetOpKindNative(MLIR_OpHandle op) {
    if (op == MLIR_INVALID_HANDLE) {
        return ASR_DIALECT_OP_INVALID;
    }
    string name = MLIR_GetOpName(op);
    if (!name.str || name.size == 0) {
        return ASR_DIALECT_OP_INVALID;
    }
    char buf[128];
    size_t n = name.size < sizeof(buf) - 1 ? name.size : sizeof(buf) - 1;
    memcpy(buf, name.str, n);
    buf[n] = '\0';
    return ASR_DialectLookupSchemaByName(buf);
}

static bool is_scope_region_op(MLIR_OpHandle op) {
    string name = MLIR_GetOpName(op);
    if (!name.str || name.size == 0) {
        return false;
    }
    static const char *regions[] = {"asr.symtab", "asr.metadata", "asr.body", NULL};
    for (size_t i = 0; regions[i] != NULL; ++i) {
        size_t n = strlen(regions[i]);
        if (name.size == n && memcmp(name.str, regions[i], n) == 0) {
            return true;
        }
    }
    return false;
}

static bool verify_missing_op_ref_allowed(
        ASR_DialectOpKind kind, const char *field) {
    if (kind == ASR_DIALECT_OP_SYMBOL_PROGRAM ||
            kind == ASR_DIALECT_OP_SYMBOL_FUNCTION) {
        return strcmp(field, "start_name") == 0 || strcmp(field, "end_name") == 0;
    }
    if (kind == ASR_DIALECT_OP_SYMBOL_VARIABLE) {
        return strcmp(field, "parent_symtab") == 0;
    }
    if (kind == ASR_DIALECT_OP_UNIT_TRANSLATIONUNIT) {
        return strcmp(field, "symtab") == 0;
    }
    return false;
}

static bool verify_scope_region_tree(
        MLIR_Context *ctx, MLIR_OpHandle op, int depth);

static bool verify_op_tree(MLIR_Context *ctx, MLIR_OpHandle op, int depth) {
    if (op == MLIR_INVALID_HANDLE || depth > 64) {
        return true;
    }
    ASR_DialectOpKind kind = ASR_DialectGetOpKindNative(op);
    const ASR_DialectOpSchema *schema = ASR_DialectLookupSchema(kind);
    if (!schema) {
        if (is_scope_region_op(op)) {
            return verify_scope_region_tree(ctx, op, depth);
        }
        return false;
    }
    for (size_t i = 0; i < schema->n_fields; ++i) {
        const ASR_DialectFieldDesc *fd = &schema->fields[i];
        if (fd->presence != ASR_FIELD_REQUIRED) {
            continue;
        }
        if (field_is_op_ref(fd->kind)) {
            if (ASR_ModuleStorageGetFieldOp(op, fd->name) == MLIR_INVALID_HANDLE) {
                if (verify_missing_op_ref_allowed(kind, fd->name)) {
                    continue;
                }
                (void)ctx;
                return false;
            }
        } else if (!field_is_op_seq(fd->kind)) {
            char buf[128];
            snprintf(buf, sizeof(buf), "%s%s", ASR_META_PREFIX, fd->name);
            if (MLIR_GetOpAttributeByName(op, buf) == MLIR_INVALID_HANDLE) {
                (void)ctx;
                return false;
            }
        }
    }
    for (size_t i = 0; i < schema->n_fields; ++i) {
        const ASR_DialectFieldDesc *fd = &schema->fields[i];
        if (!field_is_op_ref(fd->kind)) {
            continue;
        }
        MLIR_OpHandle child = ASR_ModuleStorageGetFieldOp(op, fd->name);
        if (child == MLIR_INVALID_HANDLE) {
            continue;
        }
        if (kind == ASR_DIALECT_OP_SYMBOL_VARIABLE &&
                strcmp(fd->name, "parent_symtab") == 0) {
            continue;
        }
        if (!verify_op_tree(ctx, child, depth + 1)) {
            return false;
        }
    }
    return true;
}

static bool verify_scope_region_tree(
        MLIR_Context *ctx, MLIR_OpHandle op, int depth) {
    size_t n = 0;
    MLIR_OpHandle *ops = ASR_ModuleStorageGetFieldOpSeq(op, "ops", &n);
    if (!ops) {
        return true;
    }
    for (size_t i = 0; i < n; ++i) {
        if (!verify_op_tree(ctx, ops[i], depth + 1)) {
            return false;
        }
    }
    return true;
}

#define ASR_VERIFY_MAX_SYMS 512

typedef struct {
    char names[ASR_VERIFY_MAX_SYMS][128];
    size_t n;
} ASR_VerifySymSet;

static bool verify_sym_name_eq(string a, const char *b) {
    size_t blen = strlen(b);
    return a.size == blen && a.str && memcmp(a.str, b, blen) == 0;
}

static bool verify_symset_has(const ASR_VerifySymSet *set, string name) {
    if (!name.str || name.size == 0) {
        return false;
    }
    for (size_t i = 0; i < set->n; ++i) {
        if (verify_sym_name_eq(name, set->names[i])) {
            return true;
        }
    }
    return false;
}

static bool verify_symset_add(ASR_VerifySymSet *set, string name) {
    if (!name.str || name.size == 0) {
        return false;
    }
    if (verify_symset_has(set, name)) {
        return false;
    }
    if (set->n >= ASR_VERIFY_MAX_SYMS) {
        return false;
    }
    size_t n = name.size < 127 ? name.size : 127;
    memcpy(set->names[set->n], name.str, n);
    set->names[set->n][n] = '\0';
    set->n++;
    return true;
}

static bool verify_collect_symtab(MLIR_OpHandle symtab, ASR_VerifySymSet *set) {
    size_t n = 0;
    MLIR_OpHandle *ops = asr_get_scope_region_ops(symtab, &n);
    if (!ops) {
        return true;
    }
    for (size_t i = 0; i < n; ++i) {
        ASR_DialectOpKind k = ASR_DialectGetOpKindNative(ops[i]);
        if (k == ASR_DIALECT_OP_SYMBOL_VARIABLE) {
            if (!verify_symset_add(set, asr_get_field_str(ops[i], "name"))) {
                return false;
            }
        } else if (k == ASR_DIALECT_OP_SYMBOL_FUNCTION) {
            MLIR_OpHandle fn_symtab = asr_get_scope_region(ops[i], "symtab");
            if (fn_symtab != MLIR_INVALID_HANDLE &&
                    !verify_collect_symtab(fn_symtab, set)) {
                return false;
            }
        }
    }
    return true;
}

static bool verify_var_refs_in_op(MLIR_OpHandle op, const ASR_VerifySymSet *set,
        int depth);

static bool verify_var_refs_in_scope_region(MLIR_OpHandle region,
        const ASR_VerifySymSet *set, int depth) {
    size_t n = 0;
    MLIR_OpHandle *ops = asr_get_scope_region_ops(region, &n);
    for (size_t i = 0; i < n; ++i) {
        if (!verify_var_refs_in_op(ops[i], set, depth)) {
            return false;
        }
    }
    return true;
}

static bool verify_var_refs_in_op(MLIR_OpHandle op, const ASR_VerifySymSet *set,
        int depth) {
    if (op == MLIR_INVALID_HANDLE || depth > 64) {
        return true;
    }
    ASR_DialectOpKind kind = ASR_DialectGetOpKindNative(op);
    if (kind == ASR_DIALECT_OP_EXPR_VAR) {
        string sym = asr_get_field_str(op, "v");
        return verify_symset_has(set, sym);
    }
    if (kind == ASR_DIALECT_OP_SYMBOL_PROGRAM ||
            kind == ASR_DIALECT_OP_SYMBOL_FUNCTION) {
        MLIR_OpHandle body = asr_get_scope_region(op, "body");
        if (body != MLIR_INVALID_HANDLE &&
                !verify_var_refs_in_scope_region(body, set, depth + 1)) {
            return false;
        }
        return true;
    }
    const ASR_DialectOpSchema *schema = ASR_DialectLookupSchema(kind);
    if (!schema) {
        if (is_scope_region_op(op)) {
            return verify_var_refs_in_scope_region(op, set, depth + 1);
        }
        return true;
    }
    for (size_t i = 0; i < schema->n_fields; ++i) {
        const ASR_DialectFieldDesc *fd = &schema->fields[i];
        if (field_is_op_ref(fd->kind)) {
            MLIR_OpHandle child = ASR_ModuleStorageGetFieldOp(op, fd->name);
            if (child != MLIR_INVALID_HANDLE &&
                    !verify_var_refs_in_op(child, set, depth + 1)) {
                return false;
            }
        } else if (field_is_op_seq(fd->kind)) {
            size_t n = 0;
            MLIR_OpHandle *children =
                ASR_ModuleStorageGetFieldOpSeq(op, fd->name, &n);
            for (size_t j = 0; j < n; ++j) {
                if (!verify_var_refs_in_op(children[j], set, depth + 1)) {
                    return false;
                }
            }
        }
    }
    return true;
}

static bool verify_scope_symbols(MLIR_OpHandle scope_op) {
    ASR_VerifySymSet syms = {};
    MLIR_OpHandle symtab = asr_get_scope_region(scope_op, "symtab");
    if (symtab == MLIR_INVALID_HANDLE) {
        return true;
    }
    if (!verify_collect_symtab(symtab, &syms)) {
        return false;
    }
    MLIR_OpHandle body = asr_get_scope_region(scope_op, "body");
    if (body != MLIR_INVALID_HANDLE &&
            !verify_var_refs_in_scope_region(body, &syms, 0)) {
        return false;
    }
    return true;
}

bool ASR_DialectVerifyNative(MLIR_Context *ctx, MLIR_OpHandle module) {
    if (MLIR_GetOpType(module) != OP_TYPE_MODULE) {
        return false;
    }
    MLIR_RegionHandle region = MLIR_GetOpRegion(module, 0);
    MLIR_BlockHandle block = MLIR_GetRegionBlock(region, 0);
    size_t n_ops = MLIR_GetBlockNumOps(block);
    for (size_t i = 0; i < n_ops; ++i) {
        MLIR_OpHandle op = MLIR_GetBlockOp(block, i);
        ASR_DialectOpKind kind = ASR_DialectGetOpKindNative(op);
        if (kind == ASR_DIALECT_OP_INVALID) {
            continue;
        }
        if (!verify_op_tree(ctx, op, 0)) {
            return false;
        }
        if (kind == ASR_DIALECT_OP_SYMBOL_PROGRAM ||
                kind == ASR_DIALECT_OP_SYMBOL_FUNCTION) {
            if (!verify_scope_symbols(op)) {
                return false;
            }
        }
    }
    return true;
}

extern string ASR_DialectPrintPretty(MLIR_Context *ctx, MLIR_OpHandle module);

string ASR_DialectPrintNative(MLIR_Context *ctx, MLIR_OpHandle module) {
    return ASR_DialectPrintPretty(ctx, module);
}

extern bool ASR_DialectLowerModuleNative(
    MLIR_Context *ctx, MLIR_OpHandle module, const ASR_DialectOptions *options);

bool ASR_DialectLowerToHighMLIRNative(
    MLIR_Context *ctx, MLIR_OpHandle module, const ASR_DialectOptions *options) {
    if (options && options->verify_asr_dialect) {
        if (!ASR_DialectVerifyNative(ctx, module)) {
            return false;
        }
    }
    return ASR_DialectLowerModuleNative(ctx, module, options);
}
