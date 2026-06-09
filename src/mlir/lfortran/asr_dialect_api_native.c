// Native ASR dialect op storage, print, verify, and lowering driver.
#include "asr_dialect_api.h"
#include "asr_dialect_storage.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <base/string.h>

extern void ASR_DialectFormatOpSummary(MLIR_Context *ctx, MLIR_OpHandle op,
        char *buf, size_t bufsz);

void ASR_AttachFortranLocAttrs(MLIR_Context *ctx, MLIR_OpHandle op,
        uint32_t loc_first, uint32_t loc_last) {
    if (ctx == NULL || op == MLIR_INVALID_HANDLE || loc_first == 0) {
        return;
    }
    if (loc_last < loc_first) {
        loc_last = loc_first;
    }
    MLIR_TypeHandle i64_ty = MLIR_CreateTypeInteger(ctx, 64, false);
    MLIR_AttributeHandle a_first = MLIR_CreateAttributeInteger(ctx,
        str_lit("asr.loc_first"), (int64_t)loc_first, i64_ty);
    MLIR_AttributeHandle a_last = MLIR_CreateAttributeInteger(ctx,
        str_lit("asr.loc_last"), (int64_t)loc_last, i64_ty);
    if (a_first != MLIR_INVALID_HANDLE) {
        MLIR_AppendOpAttribute(ctx, op, a_first);
    }
    if (a_last != MLIR_INVALID_HANDLE) {
        MLIR_AppendOpAttribute(ctx, op, a_last);
    }
}

bool ASR_ReadFortranLocAttrs(MLIR_OpHandle op,
        uint32_t *loc_first, uint32_t *loc_last) {
    if (op == MLIR_INVALID_HANDLE || !loc_first || !loc_last) {
        return false;
    }
    *loc_first = 0;
    *loc_last = 0;
    MLIR_AttributeHandle a_first =
        MLIR_GetOpAttributeByName(op, "asr.loc_first");
    MLIR_AttributeHandle a_last =
        MLIR_GetOpAttributeByName(op, "asr.loc_last");
    if (a_first == MLIR_INVALID_HANDLE) {
        return false;
    }
    *loc_first = (uint32_t)MLIR_GetAttributeInteger(a_first);
    if (a_last != MLIR_INVALID_HANDLE) {
        *loc_last = (uint32_t)MLIR_GetAttributeInteger(a_last);
    } else {
        *loc_last = *loc_first;
    }
    return *loc_first != 0;
}

#define ASR_META_PREFIX "asr."

static string asr_meta_attr_name(Arena *arena, const char *field_name) {
    size_t n = strlen(ASR_META_PREFIX) + strlen(field_name);
    char *buf = (char *)arena_alloc(arena, n + 1);
    memcpy(buf, ASR_META_PREFIX, strlen(ASR_META_PREFIX));
    memcpy(buf + strlen(ASR_META_PREFIX), field_name, strlen(field_name));
    buf[n] = '\0';
    return str_from_cstr_view(buf);
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

#define ASR_MAX_REGION_ATTACH 16
#define ASR_MAX_REGION_ATTACH_OPS 4096

typedef struct {
    int region_index;
    int attach_source;
    MLIR_RegionHandle src_region;
    MLIR_OpHandle *ops;
    size_t n_ops;
} RegionAttachPlan;

static int region_attach_source(const ASR_DialectField *in) {
    /* 0=empty, 1=op_seq, 3=single op
     * Do not use value.region (attach 2): it aliases op/op_seq in the union. */
    switch (in->kind) {
        case ASR_FIELD_EXPR_SEQ:
        case ASR_FIELD_OP_SEQ:
        case ASR_FIELD_NODE_SEQ:
        case ASR_FIELD_SYMBOL_SEQ:
        case ASR_FIELD_STMT_SEQ:
            if (in->value.op_seq.n_items > 0 && in->value.op_seq.items) {
                return 1;
            }
            return 0;
        case ASR_FIELD_OP:
            /* Multi-op scope fields (symtab/metadata) set n_items explicitly.
             * Single-op fields (do-loop head) only set value.op; n_items stays 0. */
            if (in->value.op_seq.n_items > 0) {
                return in->value.op_seq.items ? 1 : 0;
            }
            if (in->value.op != MLIR_INVALID_HANDLE) {
                return 3;
            }
            return 0;
        default:
            return 0;
    }
}

static void transfer_region_block_ops(MLIR_Context *ctx,
        MLIR_RegionHandle src, MLIR_BlockHandle dst_block) {
    if (src == MLIR_INVALID_HANDLE || MLIR_GetRegionNumBlocks(src) == 0) {
        return;
    }
    MLIR_BlockHandle src_block = MLIR_GetRegionBlock(src, 0);
    if (src_block == dst_block) {
        return;
    }
    size_t moved = 0;
    while (MLIR_GetBlockNumOps(src_block) > 0 &&
            moved < ASR_MAX_REGION_ATTACH_OPS) {
        MLIR_OpHandle child = MLIR_GetBlockOp(src_block, 0);
        if (child == MLIR_INVALID_HANDLE) {
            break;
        }
        MLIR_AppendBlockOp(ctx, dst_block, child);
        ++moved;
    }
}

static void attach_op_regions_post_create(MLIR_Context *ctx, MLIR_OpHandle op,
        size_t n_regions, const RegionAttachPlan *plans, size_t n_plans) {
    for (size_t ri = 0; ri < n_regions; ++ri) {
        MLIR_RegionHandle dst_r = MLIR_EnsureOpRegion(ctx, op, ri);
        MLIR_BlockHandle dst_b = MLIR_EnsureRegionEntryBlock(ctx, dst_r);
        const RegionAttachPlan *plan = NULL;
        for (size_t pi = 0; pi < n_plans; ++pi) {
            if (plans[pi].region_index == (int)ri) {
                plan = &plans[pi];
                break;
            }
        }
        if (!plan) {
            continue;
        }
        switch (plan->attach_source) {
            case 2:
                transfer_region_block_ops(ctx, plan->src_region, dst_b);
                break;
            case 1:
                if (plan->ops && plan->n_ops > 0) {
                    size_t n = plan->n_ops;
                    if (n > ASR_MAX_REGION_ATTACH_OPS) {
                        n = ASR_MAX_REGION_ATTACH_OPS;
                    }
                    for (size_t j = 0; j < n; ++j) {
                        if (plan->ops[j] != MLIR_INVALID_HANDLE) {
                            MLIR_AppendBlockOp(ctx, dst_b, plan->ops[j]);
                        }
                    }
                }
                break;
            case 3:
                if (plan->ops && plan->n_ops == 1 &&
                        plan->ops[0] != MLIR_INVALID_HANDLE) {
                    MLIR_AppendBlockOp(ctx, dst_b, plan->ops[0]);
                }
                break;
            default:
                break;
        }
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

static bool schema_sets_result_type(const ASR_DialectOpSchema *schema) {
    return schema->category == ASR_DIALECT_CATEGORY_EXPR ||
        schema->category == ASR_DIALECT_CATEGORY_TTYPE ||
        schema->category == ASR_DIALECT_CATEGORY_SYMBOL;
}

static const ASR_DialectField *find_input_field(
        const ASR_DialectField *fields, size_t n_fields, const char *name) {
    for (size_t i = 0; i < n_fields; ++i) {
        if (strcmp(fields[i].name, name) == 0) {
            return &fields[i];
        }
    }
    return NULL;
}

MLIR_OpHandle ASR_DialectCreateOpNative(
    MLIR_Context *ctx, ASR_DialectOpKind kind, MLIR_LocationHandle loc,
    const ASR_DialectField *fields, size_t n_fields) {
    const ASR_DialectOpSchema *schema = ASR_DialectLookupSchema(kind);
    if (!schema) {
        return MLIR_INVALID_HANDLE;
    }

    Arena *arena = ctx->arena;
    size_t max_attrs = n_fields + 4;
    MLIR_AttributeHandle *attrs = (MLIR_AttributeHandle *)arena_alloc(
        arena, max_attrs * sizeof(MLIR_AttributeHandle));
    size_t n_attrs = 0;

    size_t max_operands = n_fields + 8;
    MLIR_ValueHandle *operands = (MLIR_ValueHandle *)arena_alloc(
        arena, max_operands * sizeof(MLIR_ValueHandle));
    size_t n_operands = 0;

    size_t n_regions = schema->n_regions;
    RegionAttachPlan attach[ASR_MAX_REGION_ATTACH];
    size_t n_attach = 0;

    MLIR_TypeHandle result_ty = MLIR_INVALID_HANDLE;
    if (schema_sets_result_type(schema)) {
        for (size_t i = 0; i < schema->n_fields; ++i) {
            const ASR_DialectFieldDesc *fd = &schema->fields[i];
            if (fd->storage != ASR_STORAGE_RESULT_TYPE) {
                continue;
            }
            const ASR_DialectField *in = find_input_field(fields, n_fields, fd->name);
            if (in && in->value.type != MLIR_INVALID_HANDLE) {
                result_ty = in->value.type;
                break;
            }
        }
    }

    if (result_ty == MLIR_INVALID_HANDLE && schema_sets_result_type(schema)) {
        for (size_t i = 0; i < n_fields; ++i) {
            if (fields[i].kind == ASR_FIELD_TTYPE &&
                    fields[i].value.type != MLIR_INVALID_HANDLE) {
                result_ty = fields[i].value.type;
                break;
            }
        }
    }

    for (size_t si = 0; si < schema->n_fields; ++si) {
        const ASR_DialectFieldDesc *fd = &schema->fields[si];
        const ASR_DialectField *in = find_input_field(fields, n_fields, fd->name);
        if (!in) {
            continue;
        }
        switch (fd->storage) {
            case ASR_STORAGE_OMITTED:
                break;
            case ASR_STORAGE_ATTR:
            case ASR_STORAGE_SYMBOL_REF_ATTR:
            case ASR_STORAGE_SYMBOL_REF_ARRAY_ATTR:
            case ASR_STORAGE_TYPE_ATTR: {
                MLIR_AttributeHandle a = field_to_meta_attr(ctx, arena, in);
                if (a != MLIR_INVALID_HANDLE) {
                    attrs[n_attrs++] = a;
                }
                break;
            }
            case ASR_STORAGE_OPERAND:
            case ASR_STORAGE_OPTIONAL_OPERAND: {
                MLIR_ValueHandle v = op_result_value(in->value.op);
                if (v != MLIR_INVALID_HANDLE) {
                    operands[n_operands++] = v;
                }
                break;
            }
            case ASR_STORAGE_VARIADIC_OPERANDS: {
                if (in->value.op_seq.items) {
                    for (size_t j = 0; j < in->value.op_seq.n_items; ++j) {
                        MLIR_ValueHandle v = op_result_value(in->value.op_seq.items[j]);
                        if (v != MLIR_INVALID_HANDLE) {
                            operands[n_operands++] = v;
                        }
                    }
                }
                break;
            }
            case ASR_STORAGE_REGION:
            case ASR_STORAGE_OPTIONAL_REGION: {
                if (fd->region_index < 0 ||
                        (size_t)fd->region_index >= n_regions ||
                        n_attach >= ASR_MAX_REGION_ATTACH) {
                    break;
                }
                attach[n_attach].region_index = fd->region_index;
                attach[n_attach].attach_source = 0;
                attach[n_attach].src_region = MLIR_INVALID_HANDLE;
                attach[n_attach].ops = NULL;
                attach[n_attach].n_ops = 0;
                switch (region_attach_source(in)) {
                    case 1:
                        attach[n_attach].attach_source = 1;
                        attach[n_attach].ops = in->value.op_seq.items;
                        attach[n_attach].n_ops = in->value.op_seq.n_items;
                        break;
                    case 2:
                        attach[n_attach].attach_source = 2;
                        attach[n_attach].src_region = in->value.region;
                        break;
                    case 3: {
                        MLIR_OpHandle *one = (MLIR_OpHandle *)arena_alloc(
                            arena, sizeof(MLIR_OpHandle));
                        one[0] = in->value.op;
                        attach[n_attach].attach_source = 3;
                        attach[n_attach].ops = one;
                        attach[n_attach].n_ops = 1;
                        break;
                    }
                    default:
                        break;
                }
                n_attach++;
                break;
            }
            case ASR_STORAGE_RESULT_TYPE:
                break;
            default:
                break;
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
        operands, n_operands,
        NULL, n_regions,
        loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    if (op == MLIR_INVALID_HANDLE) {
        return MLIR_INVALID_HANDLE;
    }

    attach_op_regions_post_create(ctx, op, n_regions, attach, n_attach);
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

static bool verify_missing_field_allowed(
        ASR_DialectOpKind kind, const ASR_DialectFieldDesc *fd) {
    if (fd->presence != ASR_FIELD_REQUIRED) {
        return true;
    }
    if (fd->storage == ASR_STORAGE_OMITTED) {
        return true;
    }
    if (kind == ASR_DIALECT_OP_SYMBOL_VARIABLE &&
            strcmp(fd->name, "parent_symtab") == 0) {
        return true;
    }
    return false;
}

static bool verify_field_present(MLIR_OpHandle op, const ASR_DialectFieldDesc *fd) {
    switch (fd->storage) {
        case ASR_STORAGE_ATTR:
        case ASR_STORAGE_SYMBOL_REF_ATTR:
        case ASR_STORAGE_SYMBOL_REF_ARRAY_ATTR:
        case ASR_STORAGE_TYPE_ATTR: {
            char buf[128];
            snprintf(buf, sizeof(buf), "%s%s", ASR_META_PREFIX, fd->name);
            return MLIR_GetOpAttributeByName(op, buf) != MLIR_INVALID_HANDLE;
        }
        case ASR_STORAGE_OPERAND:
            return fd->operand_index >= 0 &&
                (size_t)fd->operand_index < MLIR_GetOpNumOperands(op);
        case ASR_STORAGE_OPTIONAL_OPERAND:
            return true;
        case ASR_STORAGE_VARIADIC_OPERANDS:
            if (fd->presence != ASR_FIELD_REQUIRED) {
                return true;
            }
            return fd->operand_index >= 0 &&
                (size_t)fd->operand_index < MLIR_GetOpNumOperands(op);
        case ASR_STORAGE_REGION:
            return fd->region_index >= 0 &&
                (size_t)fd->region_index < MLIR_GetOpNumRegions(op);
        case ASR_STORAGE_OPTIONAL_REGION:
            return true;
        case ASR_STORAGE_RESULT_TYPE:
            return MLIR_GetOpNumResults(op) > 0 ||
                asr_get_field_type(op, fd->name) != MLIR_INVALID_HANDLE;
        case ASR_STORAGE_OMITTED:
            return true;
        default:
            return true;
    }
}

static bool verify_op_matches_elem_kind(
        ASR_DialectOpKind kind, ASR_DialectRegionElementKind elem_kind) {
    if (elem_kind == ASR_DIALECT_REGION_ELEM_NONE ||
            elem_kind == ASR_DIALECT_REGION_ELEM_ANY ||
            elem_kind == ASR_DIALECT_REGION_ELEM_METADATA) {
        return true;
    }
    const ASR_DialectOpSchema *schema = ASR_DialectLookupSchema(kind);
    if (!schema) {
        return elem_kind == ASR_DIALECT_REGION_ELEM_ANY;
    }
    switch (elem_kind) {
        case ASR_DIALECT_REGION_ELEM_EXPR:
            return schema->category == ASR_DIALECT_CATEGORY_EXPR;
        case ASR_DIALECT_REGION_ELEM_STMT:
            return schema->category == ASR_DIALECT_CATEGORY_STMT;
        case ASR_DIALECT_REGION_ELEM_SYMBOL:
            return schema->category == ASR_DIALECT_CATEGORY_SYMBOL;
        default:
            return true;
    }
}

typedef struct {
    MLIR_OpHandle parent;
    const char *region_field;
    size_t stmt_index;
    MLIR_OpHandle stmt_op;
} ASR_VerifySite;

#if defined(__cplusplus)
#define ASR_VERIFY_THREAD_LOCAL thread_local
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define ASR_VERIFY_THREAD_LOCAL _Thread_local
#else
#define ASR_VERIFY_THREAD_LOCAL __thread
#endif

static ASR_VERIFY_THREAD_LOCAL ASR_DialectVerifyError g_asr_verify_error;
static ASR_VERIFY_THREAD_LOCAL bool g_asr_verify_error_set;

void ASR_DialectClearVerifyErrorNative(void) {
    memset(&g_asr_verify_error, 0, sizeof(g_asr_verify_error));
    g_asr_verify_error_set = false;
}

const ASR_DialectVerifyError *ASR_DialectGetLastVerifyErrorNative(void) {
    return g_asr_verify_error_set ? &g_asr_verify_error : NULL;
}

static void verify_copy_string_field(char *dst, size_t dstsz, string src) {
    if (!dst || dstsz == 0) {
        return;
    }
    if (!src.str || src.size == 0) {
        dst[0] = '\0';
        return;
    }
    size_t n = src.size < dstsz - 1 ? src.size : dstsz - 1;
    memcpy(dst, src.str, n);
    dst[n] = '\0';
}

static void verify_op_display_name(MLIR_OpHandle op, char *buf, size_t bufsz) {
    if (!buf || bufsz == 0) {
        return;
    }
    ASR_DialectOpKind kind = ASR_DialectGetOpKindNative(op);
    const ASR_DialectOpSchema *schema = ASR_DialectLookupSchema(kind);
    if (schema && schema->mlir_name) {
        if (kind == ASR_DIALECT_OP_SYMBOL_PROGRAM ||
                kind == ASR_DIALECT_OP_SYMBOL_FUNCTION) {
            string name = asr_get_field_str(op, "name");
            if (name.str && name.size > 0) {
                snprintf(buf, bufsz, "%s @%.*s", schema->mlir_name,
                    (int)(name.size < 96 ? name.size : 96), name.str);
                return;
            }
        }
        snprintf(buf, bufsz, "%s", schema->mlir_name);
        return;
    }
    string raw = MLIR_GetOpName(op);
    verify_copy_string_field(buf, bufsz, raw);
    if (buf[0] == '\0') {
        snprintf(buf, bufsz, "<invalid-op>");
    }
}

static void verify_fill_source_loc(MLIR_OpHandle op) {
    g_asr_verify_error.filename[0] = '\0';
    g_asr_verify_error.line = 0;
    g_asr_verify_error.column = 0;
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);
    if (loc == MLIR_INVALID_HANDLE) {
        return;
    }
    if (MLIR_GetLocationKind(loc) != MLIR_LOC_FILE) {
        return;
    }
    verify_copy_string_field(g_asr_verify_error.filename,
        sizeof(g_asr_verify_error.filename), MLIR_GetLocationFileFilename(loc));
    g_asr_verify_error.line = MLIR_GetLocationFileLine(loc);
    g_asr_verify_error.column = MLIR_GetLocationFileColumn(loc);
}

static void verify_format_region_path(MLIR_OpHandle parent,
        const char *region_field, char *buf, size_t bufsz) {
    if (!buf || bufsz == 0) {
        return;
    }
    if (parent == MLIR_INVALID_HANDLE || !region_field) {
        buf[0] = '\0';
        return;
    }
    char parent_label[128];
    verify_op_display_name(parent, parent_label, sizeof(parent_label));
    snprintf(buf, bufsz, "%s > %s", parent_label, region_field);
}

static const char *verify_elem_kind_label(
        ASR_DialectRegionElementKind elem_kind) {
    switch (elem_kind) {
        case ASR_DIALECT_REGION_ELEM_STMT:
            return "statement";
        case ASR_DIALECT_REGION_ELEM_EXPR:
            return "expression";
        case ASR_DIALECT_REGION_ELEM_SYMBOL:
            return "symbol";
        case ASR_DIALECT_REGION_ELEM_METADATA:
            return "metadata";
        default:
            return "region element";
    }
}

static void verify_log_fail_debug(const char *where, MLIR_OpHandle op) {
    const char *debug = getenv("ASR_DIALECT_VERIFY_DEBUG");
    if (!debug || debug[0] == '\0') {
        return;
    }
    char op_label[128];
    verify_op_display_name(op, op_label, sizeof(op_label));
    fprintf(stderr, "ASR verify fail at %s: op=%s kind=%d\n",
        where, op_label, (int)ASR_DialectGetOpKindNative(op));
}

static void verify_record_error(MLIR_Context *ctx, MLIR_OpHandle op,
        const char *reason, const ASR_VerifySite *site) {
    if (g_asr_verify_error_set || !reason) {
        return;
    }
    verify_op_display_name(op, g_asr_verify_error.op_name,
        sizeof(g_asr_verify_error.op_name));
    ASR_DialectFormatOpSummary(ctx, op, g_asr_verify_error.stmt_detail,
        sizeof(g_asr_verify_error.stmt_detail));
    if (g_asr_verify_error.stmt_detail[0] == '\0') {
        snprintf(g_asr_verify_error.stmt_detail,
            sizeof(g_asr_verify_error.stmt_detail), "%s",
            g_asr_verify_error.op_name);
    }
    verify_fill_source_loc(op);
    g_asr_verify_error.region_path[0] = '\0';
    g_asr_verify_error.dialect_line = 0;
    if (site && site->parent != MLIR_INVALID_HANDLE && site->region_field) {
        verify_format_region_path(site->parent, site->region_field,
            g_asr_verify_error.region_path,
            sizeof(g_asr_verify_error.region_path));
        g_asr_verify_error.dialect_line = site->stmt_index + 1;
    }
    if (g_asr_verify_error.region_path[0] != '\0' &&
            g_asr_verify_error.dialect_line > 0) {
        snprintf(g_asr_verify_error.message,
            sizeof(g_asr_verify_error.message),
            "in %s, ASR dialect line %zu, statement `%s`: %s",
            g_asr_verify_error.region_path,
            g_asr_verify_error.dialect_line,
            g_asr_verify_error.stmt_detail,
            reason);
    } else if (g_asr_verify_error.region_path[0] != '\0') {
        snprintf(g_asr_verify_error.message,
            sizeof(g_asr_verify_error.message),
            "in %s, statement `%s`: %s",
            g_asr_verify_error.region_path,
            g_asr_verify_error.stmt_detail,
            reason);
    } else {
        snprintf(g_asr_verify_error.message,
            sizeof(g_asr_verify_error.message),
            "statement `%s`: %s",
            g_asr_verify_error.stmt_detail,
            reason);
    }
    if (g_asr_verify_error.filename[0] != '\0' &&
            g_asr_verify_error.line > 0) {
        char suffix[128];
        if (g_asr_verify_error.column > 0) {
            snprintf(suffix, sizeof(suffix), " (source: %s:%d:%d)",
                g_asr_verify_error.filename,
                g_asr_verify_error.line,
                g_asr_verify_error.column);
        } else {
            snprintf(suffix, sizeof(suffix), " (source: %s:%d)",
                g_asr_verify_error.filename,
                g_asr_verify_error.line);
        }
        size_t used = strlen(g_asr_verify_error.message);
        if (used + 1 < sizeof(g_asr_verify_error.message)) {
            strncat(g_asr_verify_error.message, suffix,
                sizeof(g_asr_verify_error.message) - used - 1);
        }
    }
    g_asr_verify_error.fortran_loc_first = 0;
    g_asr_verify_error.fortran_loc_last = 0;
    MLIR_OpHandle loc_op = op;
    if (!ASR_ReadFortranLocAttrs(op, &g_asr_verify_error.fortran_loc_first,
            &g_asr_verify_error.fortran_loc_last) &&
            site && site->stmt_op != MLIR_INVALID_HANDLE) {
        loc_op = site->stmt_op;
        ASR_ReadFortranLocAttrs(loc_op, &g_asr_verify_error.fortran_loc_first,
            &g_asr_verify_error.fortran_loc_last);
    }
    g_asr_verify_error_set = true;
    verify_log_fail_debug("recorded", op);
}

static ASR_VERIFY_THREAD_LOCAL ASR_DialectCodegenError g_asr_codegen_error;
static ASR_VERIFY_THREAD_LOCAL bool g_asr_codegen_error_set;

void ASR_DialectClearCodegenErrorNative(void) {
    memset(&g_asr_codegen_error, 0, sizeof(g_asr_codegen_error));
    g_asr_codegen_error_set = false;
}

const ASR_DialectCodegenError *ASR_DialectGetLastCodegenErrorNative(void) {
    return g_asr_codegen_error_set ? &g_asr_codegen_error : NULL;
}

static void codegen_record_error(MLIR_Context *ctx, MLIR_OpHandle op,
        const char *reason, const ASR_LoweringContext *lc) {
    if (g_asr_codegen_error_set || !reason || !ctx) {
        return;
    }
    MLIR_OpHandle site_op = op;
    if (lc && lc->site_stmt_op != MLIR_INVALID_HANDLE) {
        site_op = lc->site_stmt_op;
    }
    verify_op_display_name(op, g_asr_codegen_error.op_name,
        sizeof(g_asr_codegen_error.op_name));
    ASR_DialectFormatOpSummary(ctx, site_op, g_asr_codegen_error.stmt_detail,
        sizeof(g_asr_codegen_error.stmt_detail));
    if (g_asr_codegen_error.stmt_detail[0] == '\0') {
        snprintf(g_asr_codegen_error.stmt_detail,
            sizeof(g_asr_codegen_error.stmt_detail), "%s",
            g_asr_codegen_error.op_name);
    }
    g_asr_codegen_error.expr_detail[0] = '\0';
    if (op != MLIR_INVALID_HANDLE && op != site_op) {
        ASR_DialectFormatOpSummary(ctx, op, g_asr_codegen_error.expr_detail,
            sizeof(g_asr_codegen_error.expr_detail));
    }
    g_asr_codegen_error.filename[0] = '\0';
    g_asr_codegen_error.line = 0;
    g_asr_codegen_error.column = 0;
    MLIR_LocationHandle mlir_loc = MLIR_GetOpLocation(site_op);
    if (mlir_loc != MLIR_INVALID_HANDLE &&
            MLIR_GetLocationKind(mlir_loc) == MLIR_LOC_FILE) {
        verify_copy_string_field(g_asr_codegen_error.filename,
            sizeof(g_asr_codegen_error.filename),
            MLIR_GetLocationFileFilename(mlir_loc));
        g_asr_codegen_error.line = MLIR_GetLocationFileLine(mlir_loc);
        g_asr_codegen_error.column = MLIR_GetLocationFileColumn(mlir_loc);
    }
    g_asr_codegen_error.region_path[0] = '\0';
    g_asr_codegen_error.dialect_line = 0;
    if (lc && lc->site_parent != MLIR_INVALID_HANDLE && lc->site_region_field) {
        verify_format_region_path(lc->site_parent, lc->site_region_field,
            g_asr_codegen_error.region_path,
            sizeof(g_asr_codegen_error.region_path));
        g_asr_codegen_error.dialect_line = lc->site_stmt_index + 1;
    }
    g_asr_codegen_error.fortran_loc_first = 0;
    g_asr_codegen_error.fortran_loc_last = 0;
    ASR_ReadFortranLocAttrs(site_op, &g_asr_codegen_error.fortran_loc_first,
        &g_asr_codegen_error.fortran_loc_last);
    if (g_asr_codegen_error.region_path[0] != '\0' &&
            g_asr_codegen_error.dialect_line > 0) {
        if (g_asr_codegen_error.expr_detail[0] != '\0') {
            snprintf(g_asr_codegen_error.message,
                sizeof(g_asr_codegen_error.message),
                "in %s, ASR dialect line %zu, statement `%s`: %s (`%s`)",
                g_asr_codegen_error.region_path,
                g_asr_codegen_error.dialect_line,
                g_asr_codegen_error.stmt_detail,
                reason,
                g_asr_codegen_error.expr_detail);
        } else {
            snprintf(g_asr_codegen_error.message,
                sizeof(g_asr_codegen_error.message),
                "in %s, ASR dialect line %zu, statement `%s`: %s",
                g_asr_codegen_error.region_path,
                g_asr_codegen_error.dialect_line,
                g_asr_codegen_error.stmt_detail,
                reason);
        }
    } else if (g_asr_codegen_error.region_path[0] != '\0') {
        snprintf(g_asr_codegen_error.message,
            sizeof(g_asr_codegen_error.message),
            "in %s, statement `%s`: %s",
            g_asr_codegen_error.region_path,
            g_asr_codegen_error.stmt_detail,
            reason);
    } else {
        snprintf(g_asr_codegen_error.message,
            sizeof(g_asr_codegen_error.message),
            "statement `%s`: %s",
            g_asr_codegen_error.stmt_detail,
            reason);
    }
    g_asr_codegen_error_set = true;
    if (getenv("ASR_DIALECT_VERIFY_DEBUG")) {
        fprintf(stderr, "ASR codegen error: %s\n", g_asr_codegen_error.message);
    }
}

bool ASR_LowerUnsupportedNative(ASR_LoweringContext *ctx, MLIR_OpHandle op,
        const char *message) {
    if (!ctx || !message) {
        return false;
    }
    codegen_record_error(ctx->ctx, op, message, ctx);
    return false;
}

static bool verify_op_tree(MLIR_Context *ctx, MLIR_OpHandle op, int depth,
        const ASR_VerifySite *site);

static bool verify_region_tree(MLIR_Context *ctx, MLIR_RegionHandle region,
        ASR_DialectRegionElementKind elem_kind, int depth,
        MLIR_OpHandle parent, const char *region_field) {
    if (region == MLIR_INVALID_HANDLE || depth > 64) {
        return true;
    }
    size_t n = asr_region_block_op_count(region);
    for (size_t i = 0; i < n; ++i) {
        MLIR_OpHandle child = asr_region_block_op(region, i);
        ASR_VerifySite child_site = {
            .parent = parent,
            .region_field = region_field,
            .stmt_index = i,
            .stmt_op = child,
        };
        if (elem_kind != ASR_DIALECT_REGION_ELEM_NONE &&
                elem_kind != ASR_DIALECT_REGION_ELEM_ANY &&
                elem_kind != ASR_DIALECT_REGION_ELEM_METADATA &&
                !verify_op_matches_elem_kind(
                    ASR_DialectGetOpKindNative(child), elem_kind)) {
            ASR_DialectOpKind child_kind = ASR_DialectGetOpKindNative(child);
            char reason[256];
            if (child_kind == ASR_DIALECT_OP_INVALID) {
                snprintf(reason, sizeof(reason),
                    "unknown ASR dialect op is not a valid %s",
                    verify_elem_kind_label(elem_kind));
            } else {
                snprintf(reason, sizeof(reason),
                    "op is not a valid %s for this region",
                    verify_elem_kind_label(elem_kind));
            }
            verify_record_error(ctx, child, reason, &child_site);
            (void)ctx;
            return false;
        }
        if (!verify_op_tree(ctx, child, depth + 1, &child_site)) {
            return false;
        }
    }
    return true;
}

static bool verify_op_tree(MLIR_Context *ctx, MLIR_OpHandle op, int depth,
        const ASR_VerifySite *site) {
    if (op == MLIR_INVALID_HANDLE || depth > 64) {
        return true;
    }
    ASR_DialectOpKind kind = ASR_DialectGetOpKindNative(op);
    const ASR_DialectOpSchema *schema = ASR_DialectLookupSchema(kind);
    if (!schema) {
        verify_record_error(ctx, op, "unknown ASR dialect op", site);
        return false;
    }

    for (size_t i = 0; i < schema->n_fields; ++i) {
        const ASR_DialectFieldDesc *fd = &schema->fields[i];
        if (verify_missing_field_allowed(kind, fd)) {
            continue;
        }
        if (!verify_field_present(op, fd)) {
            char reason[160];
            snprintf(reason, sizeof(reason),
                "missing required field '%s'", fd->name);
            verify_record_error(ctx, op, reason, site);
            (void)ctx;
            return false;
        }
    }

    for (size_t i = 0; i < schema->n_fields; ++i) {
        const ASR_DialectFieldDesc *fd = &schema->fields[i];
        if (fd->storage != ASR_STORAGE_REGION &&
                fd->storage != ASR_STORAGE_OPTIONAL_REGION) {
            continue;
        }
        if (fd->region_index < 0 ||
                (size_t)fd->region_index >= MLIR_GetOpNumRegions(op)) {
            continue;
        }
        MLIR_RegionHandle r = MLIR_GetOpRegion(op, (size_t)fd->region_index);
        if (!verify_region_tree(ctx, r, fd->region_element_kind, depth + 1,
                op, fd->name)) {
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

static bool verify_collect_symtab_region(MLIR_RegionHandle symtab,
        ASR_VerifySymSet *set) {
    size_t n = asr_region_block_op_count(symtab);
    for (size_t i = 0; i < n; ++i) {
        MLIR_OpHandle op = asr_region_block_op(symtab, i);
        ASR_DialectOpKind k = ASR_DialectGetOpKindNative(op);
        if (k == ASR_DIALECT_OP_SYMBOL_VARIABLE) {
            if (!verify_symset_add(set, asr_get_field_str(op, "name"))) {
                return false;
            }
        } else if (k == ASR_DIALECT_OP_SYMBOL_FUNCTION) {
            MLIR_RegionHandle fn_symtab = ASR_FunctionGetSymtabRegion(op);
            if (fn_symtab != MLIR_INVALID_HANDLE &&
                    !verify_collect_symtab_region(fn_symtab, set)) {
                return false;
            }
        }
    }
    return true;
}

static bool verify_var_refs_in_op(MLIR_Context *ctx, MLIR_OpHandle op,
        const ASR_VerifySymSet *set, int depth, const ASR_VerifySite *site);

static bool verify_var_refs_in_region(MLIR_Context *ctx, MLIR_RegionHandle region,
        const ASR_VerifySymSet *set, int depth,
        MLIR_OpHandle parent, const char *region_field) {
    size_t n = asr_region_block_op_count(region);
    for (size_t i = 0; i < n; ++i) {
        MLIR_OpHandle stmt = asr_region_block_op(region, i);
        ASR_VerifySite child_site = {
            .parent = parent,
            .region_field = region_field,
            .stmt_index = i,
            .stmt_op = stmt,
        };
        if (!verify_var_refs_in_op(ctx, stmt, set, depth, &child_site)) {
            return false;
        }
    }
    return true;
}

static bool verify_var_refs_in_op(MLIR_Context *ctx, MLIR_OpHandle op,
        const ASR_VerifySymSet *set, int depth, const ASR_VerifySite *site) {
    if (op == MLIR_INVALID_HANDLE || depth > 64) {
        return true;
    }
    ASR_DialectOpKind kind = ASR_DialectGetOpKindNative(op);
    if (kind == ASR_DIALECT_OP_EXPR_VAR) {
        string sym = ASR_VarV(op);
        if (verify_symset_has(set, sym)) {
            return true;
        }
        char reason[160];
        if (sym.str && sym.size > 0) {
            snprintf(reason, sizeof(reason),
                "undefined variable `@%.*s`",
                (int)(sym.size < 96 ? sym.size : 96), sym.str);
        } else {
            snprintf(reason, sizeof(reason), "undefined variable reference");
        }
        verify_record_error(ctx, op, reason, site);
        return false;
    }
    if (kind == ASR_DIALECT_OP_SYMBOL_PROGRAM ||
            kind == ASR_DIALECT_OP_SYMBOL_FUNCTION) {
        MLIR_RegionHandle body = (kind == ASR_DIALECT_OP_SYMBOL_PROGRAM)
            ? ASR_ProgramGetBodyRegion(op)
            : ASR_FunctionGetBodyRegion(op);
        if (body != MLIR_INVALID_HANDLE &&
                !verify_var_refs_in_region(ctx, body, set, depth + 1, op, "body")) {
            return false;
        }
        return true;
    }
    const ASR_DialectOpSchema *schema = ASR_DialectLookupSchema(kind);
    if (!schema) {
        return true;
    }
    for (size_t i = 0; i < schema->n_fields; ++i) {
        const ASR_DialectFieldDesc *fd = &schema->fields[i];
        if (fd->storage == ASR_STORAGE_OPERAND ||
                fd->storage == ASR_STORAGE_OPTIONAL_OPERAND) {
            MLIR_OpHandle child = asr_operand_defining_op(
                (size_t)fd->operand_index < MLIR_GetOpNumOperands(op)
                    ? MLIR_GetOpOperand(op, (size_t)fd->operand_index)
                    : MLIR_INVALID_HANDLE);
            if (child != MLIR_INVALID_HANDLE &&
                    !verify_var_refs_in_op(ctx, child, set, depth + 1, site)) {
                return false;
            }
        } else if (fd->storage == ASR_STORAGE_REGION &&
                fd->region_index >= 0 &&
                (size_t)fd->region_index < MLIR_GetOpNumRegions(op)) {
            MLIR_RegionHandle r = MLIR_GetOpRegion(op, (size_t)fd->region_index);
            if (!verify_var_refs_in_region(ctx, r, set, depth + 1, op, fd->name)) {
                return false;
            }
        }
    }
    return true;
}

static bool verify_scope_regions(MLIR_Context *ctx, MLIR_OpHandle scope_op) {
    ASR_DialectOpKind kind = ASR_DialectGetOpKindNative(scope_op);
    if (kind != ASR_DIALECT_OP_SYMBOL_PROGRAM &&
            kind != ASR_DIALECT_OP_SYMBOL_FUNCTION) {
        return true;
    }
    if (MLIR_GetOpNumRegions(scope_op) != 3) {
        verify_record_error(ctx, scope_op,
            "program/function is missing required symtab/metadata/body regions",
            NULL);
        return false;
    }
    MLIR_RegionHandle symtab = kind == ASR_DIALECT_OP_SYMBOL_PROGRAM
        ? ASR_ProgramGetSymtabRegion(scope_op)
        : ASR_FunctionGetSymtabRegion(scope_op);
    MLIR_RegionHandle metadata = kind == ASR_DIALECT_OP_SYMBOL_PROGRAM
        ? ASR_ProgramGetMetadataRegion(scope_op)
        : ASR_FunctionGetMetadataRegion(scope_op);
    MLIR_RegionHandle body = kind == ASR_DIALECT_OP_SYMBOL_PROGRAM
        ? ASR_ProgramGetBodyRegion(scope_op)
        : ASR_FunctionGetBodyRegion(scope_op);
    if (symtab == MLIR_INVALID_HANDLE ||
            metadata == MLIR_INVALID_HANDLE ||
            body == MLIR_INVALID_HANDLE) {
        verify_record_error(ctx, scope_op,
            "program/function has invalid symtab/metadata/body regions",
            NULL);
        return false;
    }
    return true;
}

static bool verify_scope_symbols(MLIR_Context *ctx, MLIR_OpHandle scope_op) {
    if (!verify_scope_regions(ctx, scope_op)) {
        return false;
    }
    ASR_VerifySymSet syms = {};
    MLIR_RegionHandle symtab = ASR_DialectGetOpKind(scope_op) ==
            ASR_DIALECT_OP_SYMBOL_PROGRAM
        ? ASR_ProgramGetSymtabRegion(scope_op)
        : ASR_FunctionGetSymtabRegion(scope_op);
    if (symtab == MLIR_INVALID_HANDLE) {
        return true;
    }
    if (!verify_collect_symtab_region(symtab, &syms)) {
        return false;
    }
    MLIR_RegionHandle body = ASR_DialectGetOpKind(scope_op) ==
            ASR_DIALECT_OP_SYMBOL_PROGRAM
        ? ASR_ProgramGetBodyRegion(scope_op)
        : ASR_FunctionGetBodyRegion(scope_op);
    if (body != MLIR_INVALID_HANDLE &&
            !verify_var_refs_in_region(ctx, body, &syms, 0, scope_op, "body")) {
        return false;
    }
    return true;
}

bool ASR_DialectVerifyNative(MLIR_Context *ctx, MLIR_OpHandle module) {
    ASR_DialectClearVerifyErrorNative();
    if (MLIR_GetOpType(module) != OP_TYPE_MODULE) {
        verify_record_error(ctx, module, "expected top-level MLIR module op", NULL);
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
        if (!verify_op_tree(ctx, op, 0, NULL)) {
            return false;
        }
        if (kind == ASR_DIALECT_OP_SYMBOL_PROGRAM ||
                kind == ASR_DIALECT_OP_SYMBOL_FUNCTION) {
            if (!verify_scope_symbols(ctx, op)) {
                return false;
            }
        }
        if (kind == ASR_DIALECT_OP_UNIT_TRANSLATIONUNIT) {
            MLIR_RegionHandle items = ASR_TranslationUnitGetItemsRegion(op);
            size_t n = asr_region_block_op_count(items);
            for (size_t j = 0; j < n; ++j) {
                MLIR_OpHandle item = asr_region_block_op(items, j);
                ASR_DialectOpKind ik = ASR_DialectGetOpKindNative(item);
                if (ik == ASR_DIALECT_OP_SYMBOL_PROGRAM ||
                        ik == ASR_DIALECT_OP_SYMBOL_FUNCTION) {
                    if (!verify_scope_symbols(ctx, item)) {
                        return false;
                    }
                }
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
    ASR_DialectClearCodegenErrorNative();
    if (options && options->verify_asr_dialect) {
        if (!ASR_DialectVerifyNative(ctx, module)) {
            return false;
        }
    }
    return ASR_DialectLowerModuleNative(ctx, module, options);
}
