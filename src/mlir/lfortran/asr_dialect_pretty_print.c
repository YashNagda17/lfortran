// Statement-first ASR dialect pretty printer (V1 structural storage).
#include "asr_dialect_api.h"
#include "asr_dialect_storage.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include <base/strbuf.h>
#include <base/string.h>

#include "generated/asr_dialect_enum_print.h"

#define ASR_PP_INDENT 2
#define ASR_PP_UNRESOLVED "<unresolved expr>"

typedef struct {
    MLIR_Context *ctx;
    strbuf out;
    int indent;
    bool in_program;
    bool in_translation_unit;
} ASR_PpCtx;

static void pp_expr(ASR_PpCtx *pp, MLIR_OpHandle op, char *buf, size_t bufsz);
static bool pp_type_spelling(ASR_PpCtx *pp, MLIR_TypeHandle ty, char *buf, size_t bufsz);
static void pp_format_index_op(ASR_PpCtx *pp, MLIR_OpHandle idx_op,
        char *buf, size_t bufsz);
static void pp_format_array_item(ASR_PpCtx *pp, MLIR_OpHandle op,
        char *buf, size_t bufsz);
static void pp_format_print_arg(ASR_PpCtx *pp, MLIR_OpHandle text,
        char *buf, size_t bufsz);

static void pp_indent(ASR_PpCtx *pp) {
    for (int i = 0; i < pp->indent; ++i) {
        strbuf_append_char(pp->ctx->arena, &pp->out, ' ');
    }
}

static void pp_line(ASR_PpCtx *pp, const char *s) {
    pp_indent(pp);
    strbuf_append_cstr(pp->ctx->arena, &pp->out, s);
    strbuf_append_char(pp->ctx->arena, &pp->out, '\n');
}

static void pp_fmt(ASR_PpCtx *pp, const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    pp_line(pp, buf);
}

static void pp_format_sym(string name, char *buf, size_t bufsz) {
    if (!name.str || name.size == 0) {
        snprintf(buf, bufsz, "@?");
        return;
    }
    snprintf(buf, bufsz, "@%.*s", (int)name.size, name.str);
}

static int64_t pp_parse_memref_static_len(ASR_PpCtx *pp, MLIR_TypeHandle ty) {
    if (!MLIR_IsTypeMemref(ty)) {
        return 0;
    }
    string ts = MLIR_GetTypeString(pp->ctx, ty);
    const char *prefix = "memref<";
    size_t plen = 7;
    if (ts.size < plen + 5 || memcmp(ts.str, prefix, plen) != 0) {
        return 0;
    }
    size_t i = plen;
    int64_t n = 0;
    while (i < ts.size && ts.str[i] >= '0' && ts.str[i] <= '9') {
        n = n * 10 + (int64_t)(ts.str[i] - '0');
        i++;
    }
    if (i + 4 >= ts.size || ts.str[i] != 'x') {
        return 0;
    }
    if (memcmp(ts.str + i + 1, "i32", 3) != 0 || ts.str[i + 4] != '>') {
        return 0;
    }
    return n > 0 ? n : 0;
}

static bool pp_variable_type_spelling(ASR_PpCtx *pp, MLIR_OpHandle var_op,
        char *buf, size_t bufsz) {
    int64_t asr_kind = asr_get_type_kind_attr(var_op);
    int64_t array_len = asr_get_array_len_attr(var_op);
    if (asr_kind > 0 && array_len > 0) {
        snprintf(buf, bufsz, "!asr.array<!asr.integer<%lld>, %lld>",
            (long long)asr_kind, (long long)array_len);
        return true;
    }
    if (asr_kind > 0) {
        snprintf(buf, bufsz, "!asr.integer<%lld>", (long long)asr_kind);
        return true;
    }
    return pp_type_spelling(pp, asr_get_field_type(var_op, "type"), buf, bufsz);
}

/* Returns true when a type suffix should be printed using !asr.* dialect types. */
static bool pp_type_spelling(ASR_PpCtx *pp, MLIR_TypeHandle ty, char *buf, size_t bufsz) {
    if (ty == MLIR_INVALID_HANDLE) {
        return false;
    }
    int64_t asr_kind = 0;
    bool is_array = false;
    int64_t array_len = 0;
    if (ASR_ModuleStorageGetTypeInfo(ty, &asr_kind, &is_array, &array_len)) {
        if (is_array && array_len > 0) {
            snprintf(buf, bufsz, "!asr.array<!asr.integer<%lld>, %lld>",
                (long long)asr_kind, (long long)array_len);
            return true;
        }
        if (MLIR_IsTypeInteger(ty)) {
            snprintf(buf, bufsz, "!asr.integer<%lld>", (long long)asr_kind);
            return true;
        }
        if (MLIR_IsTypeFloat(ty)) {
            snprintf(buf, bufsz, "!asr.real<%lld>", (long long)asr_kind);
            return true;
        }
    }
    if (MLIR_IsTypeInteger(ty)) {
        snprintf(buf, bufsz, "i32");
        return true;
    }
    if (MLIR_IsTypeFloat(ty)) {
        snprintf(buf, bufsz, "real(8)");
        return true;
    }
    if (MLIR_IsTypeMemref(ty)) {
        MLIR_TypeHandle elem = MLIR_GetTypeShapedElement(ty);
        char elem_ty[64];
        int64_t n = pp_parse_memref_static_len(pp, ty);
        if (n > 0 && pp_type_spelling(pp, elem, elem_ty, sizeof(elem_ty))) {
            snprintf(buf, bufsz, "%s[%lld]", elem_ty, (long long)n);
            return true;
        }
    }
    (void)pp;
    return false;
}

static MLIR_OpHandle pp_peel_cast(MLIR_OpHandle op) {
    while (op != MLIR_INVALID_HANDLE &&
            ASR_DialectGetOpKind(op) == ASR_DIALECT_OP_EXPR_CAST) {
        op = asr_get_field_op(op, "arg");
    }
    return op;
}

/* Index/base formatting without calling pp_expr (avoids mutual recursion). */
static void pp_format_simple_expr(ASR_PpCtx *pp, MLIR_OpHandle op,
        char *buf, size_t bufsz) {
    if (op == MLIR_INVALID_HANDLE) {
        snprintf(buf, bufsz, "?");
        return;
    }
    op = pp_peel_cast(op);
    ASR_DialectOpKind kind = ASR_DialectGetOpKind(op);
    if (kind == ASR_DIALECT_OP_EXPR_INTEGERCONSTANT) {
        snprintf(buf, bufsz, "%" PRId64, asr_get_field_i64(op, "n", 0));
        return;
    }
    if (kind == ASR_DIALECT_OP_EXPR_VAR) {
        pp_format_sym(asr_get_field_str(op, "v"), buf, bufsz);
        return;
    }
    pp_expr(pp, op, buf, bufsz);
}

static void pp_format_index_op(ASR_PpCtx *pp, MLIR_OpHandle idx_op,
        char *buf, size_t bufsz) {
    if (idx_op == MLIR_INVALID_HANDLE) {
        snprintf(buf, bufsz, "?");
        return;
    }
    if (ASR_DialectGetOpKind(idx_op) == ASR_DIALECT_OP_ARRAY_INDEX_ARRAY_INDEX) {
        MLIR_OpHandle right = asr_get_field_op(idx_op, "right");
        MLIR_OpHandle left = asr_get_field_op(idx_op, "left");
        if (right != MLIR_INVALID_HANDLE) {
            pp_format_simple_expr(pp, right, buf, bufsz);
            return;
        }
        if (left != MLIR_INVALID_HANDLE) {
            pp_format_simple_expr(pp, left, buf, bufsz);
            return;
        }
    }
    pp_format_simple_expr(pp, idx_op, buf, bufsz);
}

static bool pp_format_array_constant(ASR_PpCtx *pp, MLIR_OpHandle op,
        char *buf, size_t bufsz) {
    MLIR_OpHandle *elems = asr_get_field_op_seq(op, "elements");
    size_t n = asr_get_field_op_seq_count(op, "elements");
    if (!elems || n == 0) {
        return false;
    }
    size_t pos = 0;
    pos += (size_t)snprintf(buf + pos, bufsz - pos, "[");
    for (size_t i = 0; i < n && pos < bufsz - 16; ++i) {
        char ebuf[64];
        if (i > 0) {
            pos += (size_t)snprintf(buf + pos, bufsz - pos, ", ");
        }
        pp_expr(pp, elems[i], ebuf, sizeof(ebuf));
        pos += (size_t)snprintf(buf + pos, bufsz - pos, "%s", ebuf);
    }
    snprintf(buf + pos, bufsz - pos, "]");
    return true;
}

static bool pp_format_array_constructor(ASR_PpCtx *pp, MLIR_OpHandle op,
        char *buf, size_t bufsz) {
    MLIR_OpHandle *args = asr_get_field_op_seq(op, "args");
    size_t n = asr_get_seq_n_args_attr(op);
    if (!args || n == 0) {
        return false;
    }
    size_t pos = 0;
    pos += (size_t)snprintf(buf + pos, bufsz - pos, "[");
    for (size_t i = 0; i < n && pos < bufsz - 16; ++i) {
        char ebuf[64];
        if (i > 0) {
            pos += (size_t)snprintf(buf + pos, bufsz - pos, ", ");
        }
        pp_expr(pp, args[i], ebuf, sizeof(ebuf));
        pos += (size_t)snprintf(buf + pos, bufsz - pos, "%s", ebuf);
    }
    snprintf(buf + pos, bufsz - pos, "]");
    return true;
}

static bool pp_op_name_is(MLIR_OpHandle op, const char *name) {
    string nm = MLIR_GetOpName(op);
    size_t n = 0;
    while (name[n]) {
        n++;
    }
    return nm.size == n && nm.str && memcmp(nm.str, name, n) == 0;
}

static bool pp_is_array_item_op(MLIR_OpHandle op) {
    if (op == MLIR_INVALID_HANDLE) {
        return false;
    }
    if (ASR_DialectGetOpKind(op) == ASR_DIALECT_OP_EXPR_ARRAYITEM) {
        return true;
    }
    return pp_op_name_is(op, "asr.array_item");
}

static void pp_format_array_item(ASR_PpCtx *pp, MLIR_OpHandle op,
        char *buf, size_t bufsz) {
    MLIR_OpHandle base = asr_get_field_op(op, "v");
    char basebuf[128];
    char ibuf[64];
    if (ASR_DialectGetOpKind(base) == ASR_DIALECT_OP_EXPR_VAR) {
        pp_format_sym(asr_get_field_str(base, "v"), basebuf, sizeof(basebuf));
    } else {
        pp_expr(pp, base, basebuf, sizeof(basebuf));
    }
    MLIR_OpHandle *indices = asr_get_field_op_seq(op, "args");
    size_t n_idx = asr_get_seq_n_args_attr(op);
    if (indices && n_idx > 0) {
        pp_format_index_op(pp, indices[0], ibuf, sizeof(ibuf));
        snprintf(buf, bufsz, "%s(%s)", basebuf, ibuf);
        return;
    }
    if (indices && indices[0] != MLIR_INVALID_HANDLE) {
        pp_format_index_op(pp, indices[0], ibuf, sizeof(ibuf));
        snprintf(buf, bufsz, "%s(%s)", basebuf, ibuf);
        return;
    }
    snprintf(buf, bufsz, "%s(?)", basebuf);
}

static void pp_format_print_arg(ASR_PpCtx *pp, MLIR_OpHandle text,
        char *buf, size_t bufsz) {
    if (text == MLIR_INVALID_HANDLE) {
        snprintf(buf, bufsz, "%s", ASR_PP_UNRESOLVED);
        return;
    }
    if (ASR_DialectGetOpKind(text) == ASR_DIALECT_OP_EXPR_VAR) {
        pp_format_sym(asr_get_field_str(text, "v"), buf, bufsz);
        return;
    }
    if (pp_is_array_item_op(text)) {
        pp_format_array_item(pp, text, buf, bufsz);
        return;
    }
    if (ASR_DialectGetOpKind(text) == ASR_DIALECT_OP_EXPR_STRINGFORMAT) {
        MLIR_OpHandle *args = asr_get_field_op_seq(text, "args");
        size_t n = asr_get_seq_n_args_attr(text);
        if (args && n > 0) {
            pp_format_print_arg(pp, args[0], buf, bufsz);
            return;
        }
    }
    pp_expr(pp, text, buf, bufsz);
}

static MLIR_TypeHandle pp_expr_result_type(MLIR_OpHandle op) {
    MLIR_TypeHandle ty = asr_get_field_type(op, "type");
    if (ty != MLIR_INVALID_HANDLE) {
        return ty;
    }
    return MLIR_INVALID_HANDLE;
}

static bool pp_binop_infix_spelling(int64_t bop, char *op, size_t opsz) {
    switch (bop) {
        case 0: snprintf(op, opsz, "+"); return true;
        case 1: snprintf(op, opsz, "-"); return true;
        case 2: snprintf(op, opsz, "*"); return true;
        case 3: snprintf(op, opsz, "/"); return true;
        default: return false;
    }
}

static void pp_expr(ASR_PpCtx *pp, MLIR_OpHandle op, char *buf, size_t bufsz) {
    if (op == MLIR_INVALID_HANDLE) {
        snprintf(buf, bufsz, "%s", ASR_PP_UNRESOLVED);
        return;
    }

    op = pp_peel_cast(op);
    ASR_DialectOpKind kind = ASR_DialectGetOpKind(op);
    switch (kind) {
        case ASR_DIALECT_OP_EXPR_INTEGERCONSTANT:
            snprintf(buf, bufsz, "%" PRId64, asr_get_field_i64(op, "n", 0));
            return;
        case ASR_DIALECT_OP_EXPR_VAR: {
            string sym = asr_get_field_str(op, "v");
            pp_format_sym(sym, buf, bufsz);
            return;
        }
        case ASR_DIALECT_OP_EXPR_INTEGERBINOP: {
            MLIR_OpHandle left = asr_get_field_op(op, "left");
            MLIR_OpHandle right = asr_get_field_op(op, "right");
            int64_t bop = asr_get_field_i64(op, "op", 0);
            char lbuf[160];
            char rbuf[160];
            char opch[8];
            pp_expr(pp, left, lbuf, sizeof(lbuf));
            pp_expr(pp, right, rbuf, sizeof(rbuf));
            if (pp_binop_infix_spelling(bop, opch, sizeof(opch))) {
                snprintf(buf, bufsz, "%s %s %s", lbuf, opch, rbuf);
                return;
            }
            const char *opkw = asr_enum_binop_name(bop);
            if (!opkw) {
                opkw = "add";
            }
            snprintf(buf, bufsz, "asr.integer_bin_op %s(%s, %s)",
                opkw, lbuf, rbuf);
            return;
        }
        case ASR_DIALECT_OP_EXPR_ARRAYCONSTANT:
            if (pp_format_array_constant(pp, op, buf, bufsz)) {
                return;
            }
            snprintf(buf, bufsz, "[?]");
            return;
        case ASR_DIALECT_OP_EXPR_ARRAYCONSTRUCTOR:
            if (pp_format_array_constructor(pp, op, buf, bufsz)) {
                return;
            }
            snprintf(buf, bufsz, "[?]");
            return;
        case ASR_DIALECT_OP_EXPR_ARRAYITEM:
            pp_format_array_item(pp, op, buf, bufsz);
            return;
        default:
            break;
    }

    if (pp_is_array_item_op(op)) {
        pp_format_array_item(pp, op, buf, bufsz);
        return;
    }

    const ASR_DialectOpSchema *schema = ASR_DialectLookupSchema(kind);
    if (schema && schema->mlir_name) {
        snprintf(buf, bufsz, "%s", schema->mlir_name);
    } else {
        snprintf(buf, bufsz, "%s", ASR_PP_UNRESOLVED);
    }
}

static bool pp_is_default_step(MLIR_OpHandle step_op) {
    if (step_op == MLIR_INVALID_HANDLE) {
        return true;
    }
    if (ASR_DialectGetOpKind(step_op) == ASR_DIALECT_OP_EXPR_INTEGERCONSTANT) {
        return asr_get_field_i64(step_op, "n", 0) == 1;
    }
    return false;
}

static bool pp_variable_field_is_default(const char *fname, MLIR_OpHandle op) {
    if (strcmp(fname, "name") == 0) {
        return true;
    }
    if (strcmp(fname, "intent") == 0) {
        return asr_get_field_i64(op, fname, 0) == 0;
    }
    if (strcmp(fname, "storage") == 0) {
        return asr_get_field_i64(op, fname, 0) == 0;
    }
    if (strcmp(fname, "abi") == 0) {
        return asr_get_field_i64(op, fname, 0) == 0;
    }
    if (strcmp(fname, "access") == 0) {
        return asr_get_field_i64(op, fname, 0) == 0;
    }
    if (strcmp(fname, "presence") == 0) {
        return asr_get_field_i64(op, fname, 0) == 0;
    }
    if (strcmp(fname, "pass_attr") == 0) {
        return asr_get_field_i64(op, fname, 0) == 0;
    }
    if (strcmp(fname, "value_attr") == 0 || strcmp(fname, "target_attr") == 0 ||
            strcmp(fname, "contiguous_attr") == 0 || strcmp(fname, "is_volatile") == 0 ||
            strcmp(fname, "is_protected") == 0) {
        return !asr_get_field_bool(op, fname, false);
    }
    if (strcmp(fname, "bindc_name") == 0 || strcmp(fname, "self_argument") == 0) {
        return asr_get_field_str(op, fname).size == 0;
    }
    return false;
}

static bool pp_should_elide_field(
        ASR_DialectOpKind kind, const ASR_DialectFieldDesc *fd, MLIR_OpHandle op) {
    if (!fd) {
        return true;
    }
    if (asr_get_field_attr(op, fd->name) == MLIR_INVALID_HANDLE) {
        return true;
    }
    if (kind == ASR_DIALECT_OP_SYMBOL_VARIABLE) {
        return pp_variable_field_is_default(fd->name, op);
    }
    if (fd->kind == ASR_FIELD_TTYPE) {
        return true;
    }
    if (fd->kind == ASR_FIELD_BOOL || fd->kind == ASR_FIELD_BOOL_OPT) {
        return !asr_get_field_bool(op, fd->name, false);
    }
    if (fd->kind == ASR_FIELD_I64 || fd->kind == ASR_FIELD_I64_OPT) {
        return asr_get_field_i64(op, fd->name, 0) == 0;
    }
    if (fd->kind == ASR_FIELD_IDENTIFIER || fd->kind == ASR_FIELD_IDENTIFIER_OPT ||
            fd->kind == ASR_FIELD_STRING || fd->kind == ASR_FIELD_STRING_OPT) {
        return asr_get_field_str(op, fd->name).size == 0;
    }
    return false;
}

static const char *pp_enum_field_name(
        ASR_DialectOpKind kind, const char *field, int64_t v) {
    if (strcmp(field, "intent") == 0) {
        return asr_enum_intent_name(v);
    }
    if (strcmp(field, "storage") == 0) {
        return asr_enum_storage_type_name(v);
    }
    if (strcmp(field, "access") == 0) {
        return asr_enum_access_name(v);
    }
    if (strcmp(field, "abi") == 0) {
        return asr_enum_abi_name(v);
    }
    if (strcmp(field, "presence") == 0) {
        return asr_enum_presence_name(v);
    }
    if (strcmp(field, "pass_attr") == 0) {
        return asr_enum_pass_attr_name(v);
    }
    (void)kind;
    return NULL;
}

static const char *pp_enum_attr_category(const char *field) {
    if (strcmp(field, "intent") == 0) return "intent";
    if (strcmp(field, "storage") == 0) return "storage";
    if (strcmp(field, "abi") == 0) return "abi";
    if (strcmp(field, "access") == 0) return "access";
    if (strcmp(field, "presence") == 0) return "presence";
    if (strcmp(field, "pass_attr") == 0) return "pass_attr";
    return field;
}

static void pp_print_variable_full_attrs(ASR_PpCtx *pp, MLIR_OpHandle op) {
    const ASR_DialectOpSchema *schema =
        ASR_DialectLookupSchema(ASR_DIALECT_OP_SYMBOL_VARIABLE);
    if (!schema) {
        return;
    }

    pp->indent += ASR_PP_INDENT;

    for (size_t i = 0; i < schema->n_fields; ++i) {
        const ASR_DialectFieldDesc *fd = &schema->fields[i];
        const char *fname = fd->name;
        if (strcmp(fname, "name") == 0 || strcmp(fname, "type") == 0 ||
                strcmp(fname, "parent_symtab") == 0) {
            continue;
        }

        if (strcmp(fname, "dependencies") == 0) {
            string deps = asr_get_field_str(op, fname);
            if (deps.size == 0) {
                pp_fmt(pp, "dependencies = [],");
            } else {
                pp_fmt(pp, "dependencies = [%.*s],", (int)deps.size,
                    deps.str ? deps.str : "");
            }
            continue;
        }

        if (strcmp(fname, "codims") == 0) {
            size_t n_codims = asr_get_field_op_seq_count(op, fname);
            if (n_codims == 0) {
                pp_fmt(pp, "codims = [],");
            } else {
                pp_fmt(pp, "codims = [%zu],", n_codims);
            }
            continue;
        }

        if (fd->kind == ASR_FIELD_EXPR_OPT || fd->kind == ASR_FIELD_EXPR) {
            MLIR_OpHandle child = asr_get_field_op(op, fname);
            if (child == MLIR_INVALID_HANDLE) {
                pp_fmt(pp, "  %s = null,", fname);
            } else {
                char ebuf[160];
                pp_expr(pp, child, ebuf, sizeof(ebuf));
                pp_fmt(pp, "  %s = %s,", fname, ebuf);
            }
            continue;
        }

        if (fd->kind == ASR_FIELD_SYMBOL_REF_OPT || fd->kind == ASR_FIELD_SYMBOL_REF) {
            string sym = asr_get_field_str(op, fname);
            if (sym.size == 0) {
                pp_fmt(pp, "  %s = null,", fname);
            } else {
                char sbuf[128];
                pp_format_sym(sym, sbuf, sizeof(sbuf));
                pp_fmt(pp, "  %s = %s,", fname, sbuf);
            }
            continue;
        }

        if (fd->kind == ASR_FIELD_BOOL || fd->kind == ASR_FIELD_BOOL_OPT) {
            pp_fmt(pp, "  %s = %s,", fname,
                asr_get_field_bool(op, fname, false) ? "true" : "false");
            continue;
        }

        if (fd->kind == ASR_FIELD_I64 || fd->kind == ASR_FIELD_I64_OPT) {
            const char *en = pp_enum_field_name(
                ASR_DIALECT_OP_SYMBOL_VARIABLE, fname,
                asr_get_field_i64(op, fname, 0));
            if (en) {
                pp_fmt(pp, "  %s = #asr.%s<%s>,", fname,
                    pp_enum_attr_category(fname), en);
            } else {
                pp_fmt(pp, "  %s = %" PRId64 ",", fname,
                    asr_get_field_i64(op, fname, 0));
            }
            continue;
        }

        if (fd->kind == ASR_FIELD_STRING || fd->kind == ASR_FIELD_STRING_OPT ||
                fd->kind == ASR_FIELD_IDENTIFIER || fd->kind == ASR_FIELD_IDENTIFIER_OPT ||
                fd->kind == ASR_FIELD_IDENTIFIER_SEQ) {
            string s = asr_get_field_str(op, fname);
            pp_fmt(pp, "  %s = \"%.*s\",", fname, (int)s.size, s.str ? s.str : "");
        }
    }

    pp->indent -= ASR_PP_INDENT;
    pp_line(pp, "}");
}

static void pp_stmt(ASR_PpCtx *pp, MLIR_OpHandle op);
static void pp_stmt_block(ASR_PpCtx *pp, MLIR_OpHandle op);

static void pp_scope_region_ops(ASR_PpCtx *pp, MLIR_OpHandle region_op) {
    size_t n = 0;
    MLIR_OpHandle *ops = asr_get_scope_region_ops(region_op, &n);
    for (size_t i = 0; i < n; ++i) {
        pp_stmt_block(pp, ops[i]);
    }
}

static void pp_scope_regions(ASR_PpCtx *pp, MLIR_OpHandle scope_op) {
    MLIR_OpHandle symtab = asr_get_scope_region(scope_op, "symtab");
    MLIR_OpHandle metadata = asr_get_scope_region(scope_op, "metadata");
    MLIR_OpHandle body = asr_get_scope_region(scope_op, "body");

    if (symtab != MLIR_INVALID_HANDLE) {
        pp_line(pp, "asr.symtab {");
        pp->indent += ASR_PP_INDENT;
        pp_scope_region_ops(pp, symtab);
        pp->indent -= ASR_PP_INDENT;
        pp_line(pp, "}");
    }
    if (metadata != MLIR_INVALID_HANDLE) {
        pp_line(pp, "asr.metadata {");
        pp->indent += ASR_PP_INDENT;
        pp_scope_region_ops(pp, metadata);
        pp->indent -= ASR_PP_INDENT;
        pp_line(pp, "}");
    }
    if (body != MLIR_INVALID_HANDLE) {
        pp_line(pp, "asr.body {");
        pp->indent += ASR_PP_INDENT;
        pp_scope_region_ops(pp, body);
        pp->indent -= ASR_PP_INDENT;
        pp_line(pp, "}");
    }
}

static void pp_stmt(ASR_PpCtx *pp, MLIR_OpHandle op) {
    ASR_DialectOpKind kind = ASR_DialectGetOpKind(op);
    char buf[256];
    char ty[32];

    if (kind == ASR_DIALECT_OP_INVALID) {
        return;
    }

    if (kind == ASR_DIALECT_OP_UNIT_TRANSLATIONUNIT) {
        pp_line(pp, "asr.translation_unit {");
        pp->indent += ASR_PP_INDENT;
        pp->in_translation_unit = true;
        MLIR_OpHandle *items = asr_get_field_op_seq(op, "items");
        size_t n_items = asr_get_field_op_seq_count(op, "items");
        for (size_t i = 0; i < n_items; ++i) {
            pp_stmt(pp, items[i]);
        }
        pp->indent -= ASR_PP_INDENT;
        pp_line(pp, "}");
        pp->in_translation_unit = false;
        return;
    }

    if (kind == ASR_DIALECT_OP_SYMBOL_PROGRAM) {
        string name = asr_get_field_str(op, "name");
        pp_format_sym(name, buf, sizeof(buf));
        pp_fmt(pp, "asr.program %s {", buf);
        pp->indent += ASR_PP_INDENT;
        pp_scope_regions(pp, op);
        pp->indent -= ASR_PP_INDENT;
        pp_line(pp, "}");
        return;
    }

    if (kind == ASR_DIALECT_OP_SYMBOL_FUNCTION) {
        string name = asr_get_field_str(op, "name");
        pp_format_sym(name, buf, sizeof(buf));
        pp_fmt(pp, "asr.function %s {", buf);
        pp->indent += ASR_PP_INDENT;
        pp_scope_regions(pp, op);
        pp->indent -= ASR_PP_INDENT;
        pp_line(pp, "}");
        return;
    }

    if (asr_op_name_is(op, "asr.symtab") || asr_op_name_is(op, "asr.metadata") ||
            asr_op_name_is(op, "asr.body")) {
        return;
    }

    if (kind == ASR_DIALECT_OP_SYMBOL_VARIABLE) {
        string name = asr_get_field_str(op, "name");
        int64_t arr_len = asr_get_array_len_attr(op);
        pp_format_sym(name, buf, sizeof(buf));
        if (pp_variable_type_spelling(pp, op, ty, sizeof(ty))) {
            pp_fmt(pp, "asr.variable %s : %s {", buf, ty);
        } else if (pp_type_spelling(pp, asr_get_field_type(op, "type"), ty, sizeof(ty))) {
            pp_fmt(pp, "asr.variable %s : %s {", buf, ty);
        } else if (arr_len > 0) {
            pp_fmt(pp, "asr.variable %s : i32[%lld] {", buf, (long long)arr_len);
        } else {
            pp_fmt(pp, "asr.variable %s {", buf);
        }
        pp_print_variable_full_attrs(pp, op);
        return;
    }

    if (kind == ASR_DIALECT_OP_STMT_ASSIGNMENT) {
        MLIR_OpHandle target = asr_get_field_op(op, "target");
        MLIR_OpHandle value = asr_get_field_op(op, "value");
        char tbuf[160];
        char vbuf[160];
        if (ASR_DialectGetOpKind(target) == ASR_DIALECT_OP_EXPR_VAR) {
            pp_format_sym(asr_get_field_str(target, "v"), tbuf, sizeof(tbuf));
        } else {
            pp_expr(pp, target, tbuf, sizeof(tbuf));
        }
        pp_expr(pp, value, vbuf, sizeof(vbuf));
        if (pp_type_spelling(pp, pp_expr_result_type(value), ty, sizeof(ty))) {
            pp_fmt(pp, "asr.assign %s to %s : %s", vbuf, tbuf, ty);
        } else {
            pp_fmt(pp, "asr.assign %s to %s", vbuf, tbuf);
        }
        return;
    }

    if (kind == ASR_DIALECT_OP_STMT_DOLOOP) {
        MLIR_OpHandle head = asr_get_field_op(op, "head");
        MLIR_OpHandle body = asr_get_field_op(op, "body");
        MLIR_OpHandle orelse = asr_get_field_op(op, "orelse");
        MLIR_OpHandle v = asr_get_field_op(head, "v");
        MLIR_OpHandle start = asr_get_field_op(head, "start");
        MLIR_OpHandle end = asr_get_field_op(head, "end");
        MLIR_OpHandle step = asr_get_field_op(head, "increment");
        char iv[64];
        char sbuf[160];
        char ebuf[160];
        char stepbuf[160];

        if (ASR_DialectGetOpKind(v) == ASR_DIALECT_OP_EXPR_VAR) {
            pp_format_sym(asr_get_field_str(v, "v"), iv, sizeof(iv));
        } else {
            pp_expr(pp, v, iv, sizeof(iv));
        }
        pp_expr(pp, start, sbuf, sizeof(sbuf));
        pp_expr(pp, end, ebuf, sizeof(ebuf));

        if (!pp_is_default_step(step)) {
            pp_expr(pp, step, stepbuf, sizeof(stepbuf));
            pp_fmt(pp, "asr.do_loop %s = %s to %s step %s {",
                iv, sbuf, ebuf, stepbuf);
        } else {
            pp_fmt(pp, "asr.do_loop %s = %s to %s {", iv, sbuf, ebuf);
        }

        pp->indent += ASR_PP_INDENT;
        {
            size_t n_body = asr_get_body_count(op);
            if (n_body > 0) {
                for (size_t bi = 0; bi < n_body; ++bi) {
                    MLIR_OpHandle body_stmt = asr_get_body_op(op, bi);
                    pp_stmt_block(pp, body_stmt);
                }
            } else if (body != MLIR_INVALID_HANDLE) {
                pp_stmt_block(pp, body);
            }
        }
        pp->indent -= ASR_PP_INDENT;

        if (orelse != MLIR_INVALID_HANDLE) {
            pp_line(pp, "} else {");
            pp->indent += ASR_PP_INDENT;
            pp_stmt_block(pp, orelse);
            pp->indent -= ASR_PP_INDENT;
            pp_line(pp, "}");
        } else {
            pp_line(pp, "}");
        }
        return;
    }

    if (kind == ASR_DIALECT_OP_STMT_PRINT) {
        MLIR_OpHandle text = asr_get_field_op(op, "text");
        char pbuf[160];
        pp_format_print_arg(pp, text, pbuf, sizeof(pbuf));
        if (ASR_DialectGetOpKind(text) == ASR_DIALECT_OP_EXPR_VAR) {
            if (pp_type_spelling(pp, pp_expr_result_type(text), ty, sizeof(ty))) {
                pp_fmt(pp, "asr.print %s : %s", pbuf, ty);
            } else {
                pp_fmt(pp, "asr.print %s", pbuf);
            }
        } else {
            pp_fmt(pp, "asr.print %s", pbuf);
        }
        return;
    }

    if (kind == ASR_DIALECT_OP_STMT_RETURN) {
        pp_line(pp, "asr.return");
        return;
    }

    if (kind == ASR_DIALECT_OP_STMT_SUBROUTINECALL) {
        string name = asr_get_field_str(op, "name");
        pp_format_sym(name, buf, sizeof(buf));
        pp_fmt(pp, "asr.subroutine_call %s", buf);
        return;
    }

    /* Expression ops are not printed at module level — only inline in statements. */
    if (kind == ASR_DIALECT_OP_EXPR_INTEGERCONSTANT ||
            kind == ASR_DIALECT_OP_EXPR_INTEGERBINOP ||
            kind == ASR_DIALECT_OP_EXPR_VAR ||
            kind == ASR_DIALECT_OP_EXPR_ARRAYITEM ||
            kind == ASR_DIALECT_OP_EXPR_ARRAYCONSTANT ||
            kind == ASR_DIALECT_OP_EXPR_ARRAYCONSTRUCTOR) {
        return;
    }

    const ASR_DialectOpSchema *schema = ASR_DialectLookupSchema(kind);
    if (schema && schema->mlir_name) {
        pp_fmt(pp, "%s {", schema->mlir_name);
        pp->indent += ASR_PP_INDENT;
        for (size_t i = 0; i < schema->n_fields; ++i) {
            const ASR_DialectFieldDesc *fd = &schema->fields[i];
            if (pp_should_elide_field(kind, fd, op)) {
                continue;
            }
            if (fd->kind == ASR_FIELD_EXPR || fd->kind == ASR_FIELD_STMT ||
                    fd->kind == ASR_FIELD_OP) {
                char ebuf[160];
                pp_expr(pp, asr_get_field_op(op, fd->name), ebuf, sizeof(ebuf));
                pp_fmt(pp, "  %s = %s", fd->name, ebuf);
            }
        }
        pp->indent -= ASR_PP_INDENT;
        pp_line(pp, "}");
    }
}

static void pp_stmt_block(ASR_PpCtx *pp, MLIR_OpHandle op) {
    if (op == MLIR_INVALID_HANDLE) {
        return;
    }
    pp_stmt(pp, op);
}

static void pp_close_scopes(ASR_PpCtx *pp) {
    if (pp->in_program) {
        pp->indent -= ASR_PP_INDENT;
        pp_line(pp, "}");
        pp->in_program = false;
    }
    if (pp->in_translation_unit) {
        pp->indent -= ASR_PP_INDENT;
        pp_line(pp, "}");
        pp->in_translation_unit = false;
    }
}

string ASR_DialectPrintPretty(MLIR_Context *ctx, MLIR_OpHandle module) {
    ASR_PpCtx pp = {};
    pp.ctx = ctx;
    pp.out = strbuf_make_cap(ctx->arena, 65536);

    strbuf_append_cstr(ctx->arena, &pp.out, "module {\n");
    pp.indent = ASR_PP_INDENT;

    if (MLIR_GetOpType(module) != OP_TYPE_MODULE) {
        return strbuf_to_string(pp.out);
    }

    MLIR_RegionHandle region = MLIR_GetOpRegion(module, 0);
    MLIR_BlockHandle block = MLIR_GetRegionBlock(region, 0);
    size_t n_ops = MLIR_GetBlockNumOps(block);

    for (size_t i = 0; i < n_ops; ++i) {
        MLIR_OpHandle op = MLIR_GetBlockOp(block, i);
        if (ASR_DialectGetOpKind(op) == ASR_DIALECT_OP_INVALID) {
            continue;
        }
        pp_stmt(&pp, op);
    }

    pp_close_scopes(&pp);
    strbuf_append_cstr(ctx->arena, &pp.out, "}\n");
    return strbuf_to_string(pp.out);
}
