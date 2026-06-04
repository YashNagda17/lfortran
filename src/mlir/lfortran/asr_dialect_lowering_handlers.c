// Semantic lowering: ASR dialect ops -> func/memref/arith high-level MLIR (V1).
#include "asr_dialect_api.h"
#include "asr_dialect_storage.h"
#include "generated/asr_dialect_lowering_dispatch.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <base/string.h>


static string arena_ssa(ASR_LoweringContext *lc) {
    char *buf = (char *)arena_alloc(lc->ctx->arena, 32);
    snprintf(buf, 32, "v%d", lc->ssa_counter++);
    return str_from_cstr_view(buf);
}

#define get_i64 asr_get_field_i64
#define get_bool asr_get_field_bool
#define get_str asr_get_field_str

static ASR_SymSlot *lookup_sym(ASR_LoweringContext *lc, string name) {
    for (size_t i = 0; i < lc->n_sym_slots; ++i) {
        if (str_eq(lc->sym_slots[i].name, name)) {
            return &lc->sym_slots[i];
        }
    }
    return NULL;
}

static MLIR_OpHandle emit_op(ASR_LoweringContext *lc, MLIR_OpType ty, string name,
        MLIR_AttributeHandle *attrs, size_t n_attrs,
        MLIR_TypeHandle *rts, size_t n_rts, MLIR_ValueHandle *rs, size_t n_rs,
        MLIR_ValueHandle *ops, size_t n_ops) {
    return MLIR_CreateOp(lc->ctx, ty, name, attrs, n_attrs, rts, n_rts, rs, n_rs,
        ops, n_ops, NULL, 0, lc->loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
}

static void append_current(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    MLIR_AppendBlockOp(lc->ctx, lc->cur_block, op);
}

static MLIR_ValueHandle emit_const_i32(ASR_LoweringContext *lc, int64_t v) {
    MLIR_ValueHandle res = MLIR_CreateValueOpResult(lc->ctx, MLIR_INVALID_HANDLE,
        0, lc->i32_ty, arena_ssa(lc), lc->loc);
    MLIR_AttributeHandle val = MLIR_CreateAttributeInteger(lc->ctx,
        str_lit("value"), v, lc->i32_ty);
    MLIR_AttributeHandle as[1] = {val};
    MLIR_TypeHandle rts[1] = {lc->i32_ty};
    MLIR_ValueHandle rs[1] = {res};
    append_current(lc, emit_op(lc, OP_TYPE_ARITH_CONSTANT, str_lit("arith.constant"),
        as, 1, rts, 1, rs, 1, NULL, 0));
    return res;
}

static MLIR_ValueHandle emit_const_index(ASR_LoweringContext *lc, int64_t v) {
    MLIR_ValueHandle res = MLIR_CreateValueOpResult(lc->ctx, MLIR_INVALID_HANDLE,
        0, lc->index_ty, arena_ssa(lc), lc->loc);
    MLIR_AttributeHandle val = MLIR_CreateAttributeInteger(lc->ctx,
        str_lit("value"), v, lc->index_ty);
    MLIR_AttributeHandle as[1] = {val};
    MLIR_TypeHandle rts[1] = {lc->index_ty};
    MLIR_ValueHandle rs[1] = {res};
    append_current(lc, emit_op(lc, OP_TYPE_ARITH_CONSTANT, str_lit("arith.constant"),
        as, 1, rts, 1, rs, 1, NULL, 0));
    return res;
}

static MLIR_ValueHandle emit_memref_alloca(ASR_LoweringContext *lc, MLIR_TypeHandle ty) {
    MLIR_ValueHandle res = MLIR_CreateValueOpResult(lc->ctx, MLIR_INVALID_HANDLE,
        0, ty, arena_ssa(lc), lc->loc);
    MLIR_TypeHandle rts[1] = {ty};
    MLIR_ValueHandle rs[1] = {res};
    append_current(lc, emit_op(lc, OP_TYPE_UNREGISTERED, str_lit("memref.alloca"),
        NULL, 0, rts, 1, rs, 1, NULL, 0));
    return res;
}

static MLIR_TypeHandle memref_ty(ASR_LoweringContext *lc, int64_t len) {
    int64_t shape[1] = {len > 0 ? len : 1};
    return MLIR_CreateTypeMemref(lc->ctx, shape, 1, lc->i32_ty);
}

#define get_op_ref asr_get_field_op

static MLIR_ValueHandle lower_expr_value(ASR_LoweringContext *lc, MLIR_OpHandle op);
static MLIR_ValueHandle lower_expr_i1(ASR_LoweringContext *lc, MLIR_OpHandle op);
static bool lower_expr_op(ASR_LoweringContext *lc, MLIR_OpHandle op);
static bool lower_expr_i1_op(ASR_LoweringContext *lc, MLIR_OpHandle op);
static bool lower_stmt_ref(ASR_LoweringContext *lc, MLIR_OpHandle stmt_op);
static MLIR_ValueHandle emit_binop_i32(ASR_LoweringContext *lc, MLIR_OpType ty,
        string nm, MLIR_ValueHandle a, MLIR_ValueHandle b);

static MLIR_BlockHandle new_cfg_block(ASR_LoweringContext *lc) {
    MLIR_BlockHandle b = MLIR_CreateBlock(lc->ctx);
    MLIR_AppendRegionBlock(lc->ctx, lc->fn_region, b);
    return b;
}

static MLIR_OpHandle emit_branch_op(ASR_LoweringContext *lc, string opname, MLIR_OpType ty,
        MLIR_ValueHandle *operands, size_t n_operands,
        MLIR_BlockHandle *successors, size_t n_successors) {
    MLIR_ValueHandle **sops = NULL;
    size_t *sn = NULL;
    if (n_successors > 0) {
        sops = (MLIR_ValueHandle **)arena_alloc(
            lc->ctx->arena, n_successors * sizeof(MLIR_ValueHandle *));
        sn = (size_t *)arena_alloc(lc->ctx->arena, n_successors * sizeof(size_t));
        for (size_t i = 0; i < n_successors; ++i) {
            sops[i] = NULL;
            sn[i] = 0;
        }
    }
    return MLIR_CreateOpWithSuccessors(lc->ctx, ty, opname,
        NULL, 0, NULL, 0, NULL, 0, operands, n_operands,
        NULL, 0, successors, n_successors, sops, sn,
        lc->loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
}

static void emit_branch(ASR_LoweringContext *lc, MLIR_BlockHandle target) {
    if (lc->block_terminated) {
        return;
    }
    MLIR_BlockHandle succs[1] = {target};
    append_current(lc, emit_branch_op(lc, str_lit("cf.br"), OP_TYPE_CF_BR,
        NULL, 0, succs, 1));
    lc->block_terminated = true;
}

static void emit_cond_branch(ASR_LoweringContext *lc, MLIR_ValueHandle cond,
        MLIR_BlockHandle true_b, MLIR_BlockHandle false_b) {
    if (lc->block_terminated) {
        return;
    }
    MLIR_ValueHandle ops[1] = {cond};
    MLIR_BlockHandle succs[2] = {true_b, false_b};
    append_current(lc, emit_branch_op(lc, str_lit("cf.cond_br"), OP_TYPE_CF_COND_BR,
        ops, 1, succs, 2));
    lc->block_terminated = true;
}

static int64_t icmp_predicate_for(int64_t cmpop) {
    switch (cmpop) {
        case 0: return 0; /* Eq */
        case 1: return 1; /* NotEq */
        case 2: return 2; /* Lt */
        case 3: return 3; /* LtE */
        case 4: return 4; /* Gt */
        case 5: return 5; /* GtE */
        default: return -1;
    }
}

static MLIR_ValueHandle emit_index_cast(ASR_LoweringContext *lc, MLIR_ValueHandle v) {
    MLIR_ValueHandle res = MLIR_CreateValueOpResult(lc->ctx, MLIR_INVALID_HANDLE,
        0, lc->index_ty, arena_ssa(lc), lc->loc);
    MLIR_TypeHandle rts[1] = {lc->index_ty};
    MLIR_ValueHandle rs[1] = {res};
    MLIR_ValueHandle ops[1] = {v};
    append_current(lc, emit_op(lc, OP_TYPE_ARITH_INDEX_CAST,
        str_lit("arith.index_cast"), NULL, 0, rts, 1, rs, 1, ops, 1));
    return res;
}

static MLIR_ValueHandle fortran_index_to_memref_index(ASR_LoweringContext *lc,
        MLIR_OpHandle index_op, int64_t lbound) {
    MLIR_ValueHandle fortran_idx = lower_expr_value(lc, index_op);
    if (fortran_idx == MLIR_INVALID_HANDLE) {
        return MLIR_INVALID_HANDLE;
    }
    if (lbound == 0) {
        return emit_index_cast(lc, fortran_idx);
    }
    MLIR_ValueHandle lb = emit_const_i32(lc, lbound);
    MLIR_ValueHandle z = emit_binop_i32(lc, OP_TYPE_ARITH_SUBI,
        str_lit("arith.subi"), fortran_idx, lb);
    return emit_index_cast(lc, z);
}

static MLIR_ValueHandle emit_memref_load_at(ASR_LoweringContext *lc,
        ASR_SymSlot *slot, MLIR_ValueHandle idx) {
    MLIR_ValueHandle res = MLIR_CreateValueOpResult(lc->ctx, MLIR_INVALID_HANDLE,
        0, lc->i32_ty, arena_ssa(lc), lc->loc);
    MLIR_TypeHandle rts[1] = {lc->i32_ty};
    MLIR_ValueHandle rs[1] = {res};
    MLIR_ValueHandle ops[2] = {slot->memref, idx};
    append_current(lc, emit_op(lc, OP_TYPE_MEMREF_LOAD, str_lit("memref.load"),
        NULL, 0, rts, 1, rs, 1, ops, 2));
    return res;
}

static void emit_memref_store_at(ASR_LoweringContext *lc, MLIR_ValueHandle val,
        ASR_SymSlot *slot, MLIR_ValueHandle idx) {
    MLIR_ValueHandle ops[3] = {val, slot->memref, idx};
    append_current(lc, emit_op(lc, OP_TYPE_MEMREF_STORE, str_lit("memref.store"),
        NULL, 0, NULL, 0, NULL, 0, ops, 3));
}

static MLIR_ValueHandle emit_icmp_i32(ASR_LoweringContext *lc, int64_t predicate,
        MLIR_ValueHandle a, MLIR_ValueHandle b) {
    MLIR_ValueHandle res = MLIR_CreateValueOpResult(lc->ctx, MLIR_INVALID_HANDLE,
        0, lc->i1_ty, arena_ssa(lc), lc->loc);
    MLIR_TypeHandle rts[1] = {lc->i1_ty};
    MLIR_ValueHandle rs[1] = {res};
    MLIR_ValueHandle ops[2] = {a, b};
    MLIR_AttributeHandle pred = MLIR_CreateAttributeInteger(lc->ctx,
        str_lit("predicate"), predicate, lc->i64_ty);
    MLIR_AttributeHandle as[1] = {pred};
    append_current(lc, emit_op(lc, OP_TYPE_ARITH_CMPI, str_lit("arith.cmpi"),
        as, 1, rts, 1, rs, 1, ops, 2));
    return res;
}

static MLIR_ValueHandle emit_binop_i32(ASR_LoweringContext *lc, MLIR_OpType ty,
        string nm, MLIR_ValueHandle a, MLIR_ValueHandle b) {
    MLIR_ValueHandle res = MLIR_CreateValueOpResult(lc->ctx, MLIR_INVALID_HANDLE,
        0, lc->i32_ty, arena_ssa(lc), lc->loc);
    MLIR_TypeHandle rts[1] = {lc->i32_ty};
    MLIR_ValueHandle rs[1] = {res};
    MLIR_ValueHandle ops[2] = {a, b};
    append_current(lc, emit_op(lc, ty, nm, NULL, 0, rts, 1, rs, 1, ops, 2));
    return res;
}

static bool lower_stmt_ref(ASR_LoweringContext *lc, MLIR_OpHandle stmt_op) {
    if (stmt_op == MLIR_INVALID_HANDLE) {
        return true;
    }
    return ASR_DialectLowerOneOp(lc, stmt_op);
}

bool ASR_LowerVariable(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    string name = get_str(op, "name");
    if (lc->n_sym_slots >= ASR_MAX_SYMS) {
        return false;
    }
    ASR_SymSlot *slot = &lc->sym_slots[lc->n_sym_slots++];
    slot->name = name;
    slot->is_array = false;
    slot->array_len = 0;
    int64_t arr_len = asr_get_array_len_attr(op);
    if (arr_len > 0) {
        slot->is_array = true;
        slot->array_len = arr_len;
        slot->memref_ty = memref_ty(lc, arr_len);
    } else {
        slot->memref_ty = lc->memref_i32_1_ty;
    }
    slot->memref = emit_memref_alloca(lc, slot->memref_ty);
    return true;
}

static MLIR_ValueHandle lower_array_item_value(ASR_LoweringContext *lc,
        MLIR_OpHandle op) {
    MLIR_OpHandle base = get_op_ref(op, "v");
    if (ASR_DialectGetOpKind(base) != ASR_DIALECT_OP_EXPR_VAR) {
        return MLIR_INVALID_HANDLE;
    }
    string sym = get_str(base, "v");
    ASR_SymSlot *slot = lookup_sym(lc, sym);
    if (!slot || !slot->is_array) {
        return MLIR_INVALID_HANDLE;
    }
    MLIR_OpHandle *indices = asr_get_field_op_seq(op, "args");
    if (!indices) {
        return MLIR_INVALID_HANDLE;
    }
    MLIR_OpHandle index_op = get_op_ref(indices[0], "right");
    if (index_op == MLIR_INVALID_HANDLE) {
        index_op = get_op_ref(indices[0], "left");
    }
    if (index_op == MLIR_INVALID_HANDLE) {
        return MLIR_INVALID_HANDLE;
    }
    MLIR_ValueHandle idx = fortran_index_to_memref_index(lc, index_op, 1);
    return emit_memref_load_at(lc, slot, idx);
}

static bool store_array_item(ASR_LoweringContext *lc, MLIR_OpHandle item_op,
        MLIR_ValueHandle val) {
    MLIR_OpHandle base = get_op_ref(item_op, "v");
    if (ASR_DialectGetOpKind(base) != ASR_DIALECT_OP_EXPR_VAR) {
        return false;
    }
    ASR_SymSlot *slot = lookup_sym(lc, get_str(base, "v"));
    if (!slot || !slot->is_array) {
        return false;
    }
    MLIR_OpHandle *indices = asr_get_field_op_seq(item_op, "args");
    if (!indices) {
        return false;
    }
    MLIR_OpHandle index_op = get_op_ref(indices[0], "right");
    if (index_op == MLIR_INVALID_HANDLE) {
        index_op = get_op_ref(indices[0], "left");
    }
    if (index_op == MLIR_INVALID_HANDLE) {
        return false;
    }
    MLIR_ValueHandle idx = fortran_index_to_memref_index(lc, index_op, 1);
    emit_memref_store_at(lc, val, slot, idx);
    return true;
}

static bool store_array_constructor(ASR_LoweringContext *lc, ASR_SymSlot *slot,
        MLIR_OpHandle ctor_op) {
    MLIR_OpHandle *elems = asr_get_field_op_seq(ctor_op, "args");
    if (!elems) {
        return false;
    }
    size_t n = asr_get_seq_n_args_attr(ctor_op);
    if (n == 0) {
        n = (size_t)slot->array_len;
    }
    if (n > (size_t)slot->array_len) {
        n = (size_t)slot->array_len;
    }
    for (size_t i = 0; i < n; ++i) {
        MLIR_ValueHandle val = lower_expr_value(lc, elems[i]);
        if (val == MLIR_INVALID_HANDLE) {
            return false;
        }
        emit_memref_store_at(lc, val, slot, emit_const_index(lc, (int64_t)i));
    }
    return true;
}

static bool store_array_constant(ASR_LoweringContext *lc, ASR_SymSlot *slot,
        MLIR_OpHandle const_op) {
    MLIR_OpHandle *elems = asr_get_field_op_seq(const_op, "elements");
    size_t n = asr_get_field_op_seq_count(const_op, "elements");
    if (!elems || n == 0) {
        return ASR_LowerUnsupported(lc, const_op,
            "ArrayConstant: missing elements in dialect IR");
    }
    size_t limit = n;
    if ((size_t)slot->array_len > 0 && limit > (size_t)slot->array_len) {
        limit = (size_t)slot->array_len;
    }
    for (size_t i = 0; i < limit; ++i) {
        MLIR_ValueHandle val = lower_expr_value(lc, elems[i]);
        if (val == MLIR_INVALID_HANDLE) {
            return false;
        }
        emit_memref_store_at(lc, val, slot, emit_const_index(lc, (int64_t)i));
    }
    return true;
}

static MLIR_OpHandle peel_assignment_value(MLIR_OpHandle value_op) {
    while (value_op != MLIR_INVALID_HANDLE) {
        ASR_DialectOpKind k = ASR_DialectGetOpKind(value_op);
        if (k == ASR_DIALECT_OP_EXPR_CAST) {
            value_op = get_op_ref(value_op, "arg");
            continue;
        }
        break;
    }
    return value_op;
}

bool ASR_LowerAssignment(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    MLIR_OpHandle value_op = peel_assignment_value(get_op_ref(op, "value"));
    MLIR_OpHandle target = get_op_ref(op, "target");
    ASR_DialectOpKind tk = ASR_DialectGetOpKind(target);
    ASR_DialectOpKind vk = ASR_DialectGetOpKind(value_op);

    if (tk == ASR_DIALECT_OP_EXPR_VAR) {
        string sym = get_str(target, "v");
        ASR_SymSlot *slot = lookup_sym(lc, sym);
        if (!slot) {
            return false;
        }
        if (slot->is_array && vk == ASR_DIALECT_OP_EXPR_ARRAYCONSTRUCTOR) {
            return store_array_constructor(lc, slot, value_op);
        }
        if (slot->is_array && vk == ASR_DIALECT_OP_EXPR_ARRAYCONSTANT) {
            return store_array_constant(lc, slot, value_op);
        }
        if (slot->is_array) {
            return ASR_LowerUnsupported(lc, op,
                "Assignment: unsupported array value form");
        }
        MLIR_ValueHandle val = lower_expr_value(lc, value_op);
        if (val == MLIR_INVALID_HANDLE) {
            return false;
        }
        emit_memref_store_at(lc, val, slot, emit_const_index(lc, 0));
        return true;
    }
    if (tk == ASR_DIALECT_OP_EXPR_ARRAYITEM) {
        MLIR_ValueHandle val = lower_expr_value(lc, value_op);
        if (val == MLIR_INVALID_HANDLE) {
            return false;
        }
        return store_array_item(lc, target, val);
    }
    return ASR_LowerUnsupported(lc, op, "Assignment: unsupported target");
}

static bool lower_print_expr(ASR_LoweringContext *lc, MLIR_OpHandle text_op) {
    if (text_op == MLIR_INVALID_HANDLE) {
        return false;
    }
    if (ASR_DialectGetOpKind(text_op) == ASR_DIALECT_OP_EXPR_STRINGFORMAT) {
        MLIR_OpHandle *args = asr_get_field_op_seq(text_op, "args");
        size_t n = asr_get_seq_n_args_attr(text_op);
        if (!args || n == 0) {
            return false;
        }
        for (size_t i = 0; i < n; ++i) {
            MLIR_ValueHandle val = lower_expr_value(lc, args[i]);
            if (val == MLIR_INVALID_HANDLE) {
                return false;
            }
            MLIR_ValueHandle ops[1] = {val};
            append_current(lc, emit_op(lc, OP_TYPE_VECTOR_PRINT,
                str_lit("vector.print"), NULL, 0, NULL, 0, NULL, 0, ops, 1));
        }
        return true;
    }
    MLIR_ValueHandle val = lower_expr_value(lc, text_op);
    if (val == MLIR_INVALID_HANDLE) {
        return false;
    }
    MLIR_ValueHandle ops[1] = {val};
    append_current(lc, emit_op(lc, OP_TYPE_VECTOR_PRINT, str_lit("vector.print"),
        NULL, 0, NULL, 0, NULL, 0, ops, 1));
    return true;
}

bool ASR_LowerPrint(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_print_expr(lc, get_op_ref(op, "text"));
}

bool ASR_LowerFileWrite(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_print_expr(lc, get_op_ref(op, "values"));
}

bool ASR_LowerReturn(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    (void)op;
    MLIR_ValueHandle zero = emit_const_i32(lc, 0);
    MLIR_ValueHandle ops[1] = {zero};
    append_current(lc, emit_op(lc, OP_TYPE_FUNC_RETURN, str_lit("func.return"),
        NULL, 0, NULL, 0, NULL, 0, ops, 1));
    lc->block_terminated = true;
    return true;
}

static bool lower_scope_region_ops(ASR_LoweringContext *lc, MLIR_OpHandle region_op) {
    size_t n = 0;
    MLIR_OpHandle *ops = asr_get_scope_region_ops(region_op, &n);
    for (size_t i = 0; i < n; ++i) {
        if (!ASR_DialectLowerOneOp(lc, ops[i])) {
            return false;
        }
    }
    return true;
}

static bool lower_body_region_ops(ASR_LoweringContext *lc, MLIR_OpHandle region_op) {
    size_t n = 0;
    MLIR_OpHandle *ops = asr_get_scope_region_ops(region_op, &n);
    for (size_t i = 0; i < n; ++i) {
        if (!lower_stmt_ref(lc, ops[i])) {
            return false;
        }
    }
    return true;
}

static MLIR_OpHandle find_program_in_block(MLIR_BlockHandle block) {
    size_t n = MLIR_GetBlockNumOps(block);
    for (size_t i = 0; i < n; ++i) {
        MLIR_OpHandle op = MLIR_GetBlockOp(block, i);
        if (ASR_DialectGetOpKind(op) == ASR_DIALECT_OP_SYMBOL_PROGRAM) {
            return op;
        }
        if (ASR_DialectGetOpKind(op) == ASR_DIALECT_OP_UNIT_TRANSLATIONUNIT) {
            MLIR_OpHandle *items = asr_get_field_op_seq(op, "items");
            size_t n_items = asr_get_field_op_seq_count(op, "items");
            if (items && n_items > 0 &&
                    ASR_DialectGetOpKind(items[0]) == ASR_DIALECT_OP_SYMBOL_PROGRAM) {
                return items[0];
            }
        }
    }
    return MLIR_INVALID_HANDLE;
}

bool ASR_LowerProgram(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    lc->n_sym_slots = 0;
    lc->fn_region = MLIR_CreateRegion(lc->ctx);
    lc->cur_block = MLIR_CreateBlock(lc->ctx);
    MLIR_AppendRegionBlock(lc->ctx, lc->fn_region, lc->cur_block);
    lc->block_terminated = false;

    MLIR_OpHandle symtab = asr_get_scope_region(op, "symtab");
    if (symtab != MLIR_INVALID_HANDLE &&
            !lower_scope_region_ops(lc, symtab)) {
        return false;
    }

    MLIR_OpHandle body = asr_get_scope_region(op, "body");
    if (body != MLIR_INVALID_HANDLE &&
            !lower_body_region_ops(lc, body)) {
        return false;
    }
    return true;
}

bool ASR_LowerTranslationUnit(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    (void)lc;
    (void)op;
    return true;
}

static MLIR_ValueHandle lower_expr_value(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    if (op == MLIR_INVALID_HANDLE) {
        return MLIR_INVALID_HANDLE;
    }
    ASR_DialectOpKind kind = ASR_DialectGetOpKind(op);
    switch (kind) {
        case ASR_DIALECT_OP_EXPR_INTEGERCONSTANT: {
            int64_t n = get_i64(op, "n", 0);
            return emit_const_i32(lc, n);
        }
        case ASR_DIALECT_OP_EXPR_VAR: {
            string sym = get_str(op, "v");
            ASR_SymSlot *slot = lookup_sym(lc, sym);
            if (!slot) {
                return MLIR_INVALID_HANDLE;
            }
            if (slot->is_array) {
                return MLIR_INVALID_HANDLE;
            }
            return emit_memref_load_at(lc, slot, emit_const_index(lc, 0));
        }
        case ASR_DIALECT_OP_EXPR_INTEGERBINOP: {
            MLIR_ValueHandle lhs = lower_expr_value(lc, get_op_ref(op, "left"));
            MLIR_ValueHandle rhs = lower_expr_value(lc, get_op_ref(op, "right"));
            int64_t bop = get_i64(op, "op", 0);
            MLIR_OpType ty = OP_TYPE_ARITH_ADDI;
            string nm = str_lit("arith.addi");
            switch (bop) {
                case 0: break;
                case 1: ty = OP_TYPE_ARITH_SUBI; nm = str_lit("arith.subi"); break;
                case 2: ty = OP_TYPE_ARITH_MULI; nm = str_lit("arith.muli"); break;
                case 3: ty = OP_TYPE_ARITH_DIVSI; nm = str_lit("arith.divsi"); break;
                default:
                    if (lc->options && lc->options->allow_unimplemented_nodes) {
                        return emit_const_i32(lc, 0);
                    }
                    ASR_LowerUnsupported(lc, op,
                        "IntegerBinOp: unsupported binop");
                    return MLIR_INVALID_HANDLE;
            }
            MLIR_ValueHandle res = MLIR_CreateValueOpResult(lc->ctx,
                MLIR_INVALID_HANDLE, 0, lc->i32_ty, arena_ssa(lc), lc->loc);
            MLIR_TypeHandle rts[1] = {lc->i32_ty};
            MLIR_ValueHandle rs[1] = {res};
            MLIR_ValueHandle ops[2] = {lhs, rhs};
            append_current(lc, emit_op(lc, ty, nm, NULL, 0, rts, 1, rs, 1, ops, 2));
            return res;
        }
        case ASR_DIALECT_OP_EXPR_REALCONSTANT: {
            return emit_const_i32(lc, 0);
        }
        case ASR_DIALECT_OP_EXPR_REALBINOP: {
            MLIR_ValueHandle lhs = lower_expr_value(lc, get_op_ref(op, "left"));
            MLIR_ValueHandle rhs = lower_expr_value(lc, get_op_ref(op, "right"));
            return emit_binop_i32(lc, OP_TYPE_ARITH_ADDF, str_lit("arith.addf"),
                lhs, rhs);
        }
        case ASR_DIALECT_OP_EXPR_LOGICALCONSTANT: {
            return emit_const_i32(lc, get_i64(op, "value", 0) ? 1 : 0);
        }
        case ASR_DIALECT_OP_EXPR_LOGICALNOT: {
            MLIR_ValueHandle arg = lower_expr_value(lc, get_op_ref(op, "arg"));
            MLIR_ValueHandle one = emit_const_i32(lc, 1);
            return emit_binop_i32(lc, OP_TYPE_ARITH_XORI, str_lit("arith.xori"),
                arg, one);
        }
        case ASR_DIALECT_OP_EXPR_LOGICALBINOP: {
            MLIR_ValueHandle lhs = lower_expr_value(lc, get_op_ref(op, "left"));
            MLIR_ValueHandle rhs = lower_expr_value(lc, get_op_ref(op, "right"));
            int64_t bop = get_i64(op, "op", 0);
            if (bop == 0) {
                return emit_binop_i32(lc, OP_TYPE_ARITH_ANDI, str_lit("arith.andi"),
                    lhs, rhs);
            }
            return emit_binop_i32(lc, OP_TYPE_ARITH_ORI, str_lit("arith.ori"),
                lhs, rhs);
        }
        case ASR_DIALECT_OP_EXPR_CAST: {
            return lower_expr_value(lc, get_op_ref(op, "arg"));
        }
        case ASR_DIALECT_OP_EXPR_INTEGERUNARYMINUS: {
            MLIR_ValueHandle arg = lower_expr_value(lc, get_op_ref(op, "arg"));
            MLIR_ValueHandle zero = emit_const_i32(lc, 0);
            return emit_binop_i32(lc, OP_TYPE_ARITH_SUBI, str_lit("arith.subi"),
                zero, arg);
        }
        case ASR_DIALECT_OP_EXPR_STRINGFORMAT: {
            MLIR_OpHandle *args = asr_get_field_op_seq(op, "args");
            if (args && args[0] != MLIR_INVALID_HANDLE) {
                return lower_expr_value(lc, args[0]);
            }
            return emit_const_i32(lc, 0);
        }
        case ASR_DIALECT_OP_EXPR_ARRAYITEM:
            return lower_array_item_value(lc, op);
        case ASR_DIALECT_OP_EXPR_ARRAYCONSTRUCTOR:
            return emit_const_i32(lc, 0);
        case ASR_DIALECT_OP_EXPR_ARRAYCONSTANT: {
            MLIR_OpHandle *elems = asr_get_field_op_seq(op, "elements");
            size_t n = asr_get_field_op_seq_count(op, "elements");
            if (elems && n > 0) {
                return lower_expr_value(lc, elems[0]);
            }
            return MLIR_INVALID_HANDLE;
        }
        default:
            if (lc->options && lc->options->allow_unimplemented_nodes) {
                return emit_const_i32(lc, 0);
            }
            ASR_LowerUnsupported(lc, op, "expression lowering not implemented");
            return MLIR_INVALID_HANDLE;
    }
}

static MLIR_ValueHandle lower_expr_i1(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    if (op == MLIR_INVALID_HANDLE) {
        return MLIR_INVALID_HANDLE;
    }
    ASR_DialectOpKind kind = ASR_DialectGetOpKind(op);
    if (kind == ASR_DIALECT_OP_EXPR_INTEGERCOMPARE) {
        MLIR_ValueHandle lhs = lower_expr_value(lc, get_op_ref(op, "left"));
        MLIR_ValueHandle rhs = lower_expr_value(lc, get_op_ref(op, "right"));
        int64_t pred = icmp_predicate_for(get_i64(op, "op", 0));
        if (pred < 0) {
            ASR_LowerUnsupported(lc, op, "IntegerCompare: unsupported cmpop");
            return MLIR_INVALID_HANDLE;
        }
        return emit_icmp_i32(lc, pred, lhs, rhs);
    }
    ASR_LowerUnsupported(lc, op, "boolean expression lowering not implemented");
    return MLIR_INVALID_HANDLE;
}

static bool lower_expr_op(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_value(lc, op) != MLIR_INVALID_HANDLE;
}

static bool lower_expr_i1_op(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_i1(lc, op) != MLIR_INVALID_HANDLE;
}

static void init_types(ASR_LoweringContext *lc) {
    lc->i32_ty = MLIR_CreateTypeInteger(lc->ctx, 32, true);
    lc->i64_ty = MLIR_CreateTypeInteger(lc->ctx, 64, false);
    lc->i1_ty = MLIR_CreateTypeInteger(lc->ctx, 1, false);
    lc->index_ty = MLIR_CreateTypeIndex(lc->ctx);
    int64_t shape[1] = {1};
    lc->memref_i32_1_ty = MLIR_CreateTypeMemref(lc->ctx, shape, 1, lc->i32_ty);
}

// Expression dispatch: thin wrappers over lower_expr_value() / lower_expr_i1().
bool ASR_LowerIntegerConstant(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerVar(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerIntegerBinOp(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerIntegerCompare(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_i1_op(lc, op);
}
bool ASR_LowerIntegerUnaryMinus(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerIntegerBitNot(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerRealConstant(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerRealBinOp(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerRealCompare(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerRealUnaryMinus(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerCast(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerLogicalConstant(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerLogicalNot(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerLogicalBinOp(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerLogicalCompare(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerStringFormat(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerArrayItem(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerArrayConstant(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerArrayConstructor(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerArraySize(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerArrayBound(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerArraySection(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerArrayTranspose(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerArrayReshape(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerArrayPhysicalCast(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerIntrinsicElementalFunction(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerIntrinsicArrayFunction(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerIntrinsicImpureFunction(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerIntrinsicImpureSubroutine(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return lower_expr_op(lc, op);
}
bool ASR_LowerFunction(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "Function lowering pending");
}
bool ASR_LowerAllocate(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "Allocate lowering pending");
}
bool ASR_LowerDoLoop(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    MLIR_OpHandle head_op = get_op_ref(op, "head");
    if (head_op == MLIR_INVALID_HANDLE) {
        return ASR_LowerUnsupported(lc, op, "DoLoop: missing head");
    }
    MLIR_OpHandle v_op = get_op_ref(head_op, "v");
    MLIR_OpHandle start_op = get_op_ref(head_op, "start");
    MLIR_OpHandle end_op = get_op_ref(head_op, "end");
    MLIR_OpHandle inc_op = get_op_ref(head_op, "increment");
    if (v_op == MLIR_INVALID_HANDLE || start_op == MLIR_INVALID_HANDLE ||
            end_op == MLIR_INVALID_HANDLE) {
        return ASR_LowerUnsupported(lc, op, "DoLoop: incomplete head");
    }
    if (ASR_DialectGetOpKind(v_op) != ASR_DIALECT_OP_EXPR_VAR) {
        return ASR_LowerUnsupported(lc, op, "DoLoop: index must be a variable");
    }
    string sym = get_str(v_op, "v");
    ASR_SymSlot *slot = lookup_sym(lc, sym);
    if (!slot || slot->is_array) {
        return ASR_LowerUnsupported(lc, op, "DoLoop: scalar loop variable required");
    }

    MLIR_ValueHandle start_v = lower_expr_value(lc, start_op);
    emit_memref_store_at(lc, start_v, slot, emit_const_index(lc, 0));

    MLIR_BlockHandle header_b = new_cfg_block(lc);
    MLIR_BlockHandle body_b = new_cfg_block(lc);
    MLIR_BlockHandle step_b = new_cfg_block(lc);
    MLIR_BlockHandle exit_b = new_cfg_block(lc);
    emit_branch(lc, header_b);

    lc->cur_block = header_b;
    lc->block_terminated = false;
    MLIR_ValueHandle iv = emit_memref_load_at(lc, slot, emit_const_index(lc, 0));
    MLIR_ValueHandle endv = lower_expr_value(lc, end_op);
    MLIR_ValueHandle cond = emit_icmp_i32(lc, 3, iv, endv);
    emit_cond_branch(lc, cond, body_b, exit_b);

    lc->cur_block = body_b;
    lc->block_terminated = false;
    size_t n_body = asr_get_body_count(op);
    if (n_body > 0) {
        for (size_t bi = 0; bi < n_body; ++bi) {
            MLIR_OpHandle body_stmt = asr_get_body_op(op, bi);
            if (!lower_stmt_ref(lc, body_stmt)) {
                return false;
            }
            if (lc->block_terminated) {
                break;
            }
        }
    } else if (!lower_stmt_ref(lc, get_op_ref(op, "body"))) {
        return false;
    }
    if (!lc->block_terminated) {
        emit_branch(lc, step_b);
    }

    lc->cur_block = step_b;
    lc->block_terminated = false;
    iv = emit_memref_load_at(lc, slot, emit_const_index(lc, 0));
    MLIR_ValueHandle stepv;
    if (inc_op != MLIR_INVALID_HANDLE) {
        stepv = lower_expr_value(lc, inc_op);
    } else {
        stepv = emit_const_i32(lc, 1);
    }
    MLIR_ValueHandle next = emit_binop_i32(lc, OP_TYPE_ARITH_ADDI,
        str_lit("arith.addi"), iv, stepv);
    emit_memref_store_at(lc, next, slot, emit_const_index(lc, 0));
    emit_branch(lc, header_b);

    lc->cur_block = exit_b;
    lc->block_terminated = false;
    return true;
}
bool ASR_LowerIf(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    MLIR_OpHandle test_op = get_op_ref(op, "test");
    if (test_op == MLIR_INVALID_HANDLE) {
        return ASR_LowerUnsupported(lc, op, "If: missing test");
    }
    MLIR_ValueHandle cond = lower_expr_i1(lc, test_op);
    if (cond == MLIR_INVALID_HANDLE) {
        return false;
    }

    MLIR_BlockHandle then_b = new_cfg_block(lc);
    MLIR_BlockHandle else_b = new_cfg_block(lc);
    MLIR_BlockHandle merge_b = new_cfg_block(lc);
    emit_cond_branch(lc, cond, then_b, else_b);

    lc->cur_block = then_b;
    lc->block_terminated = false;
    if (!lower_stmt_ref(lc, get_op_ref(op, "body"))) {
        return false;
    }
    if (!lc->block_terminated) {
        emit_branch(lc, merge_b);
    }

    lc->cur_block = else_b;
    lc->block_terminated = false;
    if (!lower_stmt_ref(lc, get_op_ref(op, "orelse"))) {
        return false;
    }
    if (!lc->block_terminated) {
        emit_branch(lc, merge_b);
    }

    lc->cur_block = merge_b;
    lc->block_terminated = false;
    return true;
}
bool ASR_LowerErrorStop(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    (void)op;
    MLIR_ValueHandle code = emit_const_i32(lc, 1);
    MLIR_ValueHandle ops[1] = {code};
    append_current(lc, emit_op(lc, OP_TYPE_FUNC_RETURN, str_lit("func.return"),
        NULL, 0, NULL, 0, NULL, 0, ops, 1));
    lc->block_terminated = true;
    return true;
}
bool ASR_LowerInteger(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    (void)lc; (void)op; return true;
}
bool ASR_LowerReal(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    (void)lc; (void)op; return true;
}
bool ASR_LowerLogical(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    (void)lc; (void)op; return true;
}
bool ASR_LowerComplex(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    (void)lc; (void)op; return true;
}
bool ASR_LowerArray(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    (void)lc; (void)op; return true;
}

bool ASR_DialectLowerModuleNative(
    MLIR_Context *ctx, MLIR_OpHandle module, const ASR_DialectOptions *options) {
    if (MLIR_GetOpType(module) != OP_TYPE_MODULE) {
        return false;
    }
    ASR_LoweringContext lc = {};
    lc.ctx = ctx;
    lc.module = module;
    lc.options = options;
    lc.loc = MLIR_CreateLocationUnknown(ctx, str_lit("lfortran"));
    init_types(&lc);

    MLIR_RegionHandle mod_region = MLIR_GetOpRegion(module, 0);
    MLIR_BlockHandle asr_block = MLIR_GetRegionBlock(mod_region, 0);

    MLIR_RegionHandle new_mod_region = MLIR_CreateRegion(ctx);
    lc.module_block = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, new_mod_region, lc.module_block);

    lc.fn_region = MLIR_CreateRegion(ctx);
    lc.cur_block = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, lc.fn_region, lc.cur_block);
    lc.block_terminated = false;

    MLIR_OpHandle program = find_program_in_block(asr_block);
    if (program == MLIR_INVALID_HANDLE) {
        return false;
    }
    if (!ASR_LowerProgram(&lc, program)) {
        if (!options || !options->allow_unimplemented_nodes) {
            return false;
        }
    }

    if (!lc.block_terminated) {
        ASR_LowerReturn(&lc, MLIR_INVALID_HANDLE);
    }

    MLIR_TypeHandle res_tys[1] = {lc.i32_ty};
    MLIR_TypeHandle fn_ty = MLIR_CreateTypeFunction(ctx, NULL, 0, res_tys, 1);
    MLIR_AttributeHandle sym_name = MLIR_CreateAttributeString(ctx,
        str_lit("sym_name"), str_lit("main"));
    MLIR_AttributeHandle fn_ty_attr = MLIR_CreateAttributeType(ctx,
        str_lit("function_type"), fn_ty);
    MLIR_AttributeHandle attrs[2] = {sym_name, fn_ty_attr};
    MLIR_RegionHandle regs[1] = {lc.fn_region};
    MLIR_OpHandle fn = MLIR_CreateOp(ctx, OP_TYPE_FUNC_FUNC, str_lit("func.func"),
        attrs, 2, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        lc.loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    MLIR_AppendBlockOp(ctx, lc.module_block, fn);
    MLIR_SetOpRegion(ctx, module, 0, new_mod_region);
    return true;
}
