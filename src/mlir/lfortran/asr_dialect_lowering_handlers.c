// Semantic lowering: ASR dialect ops -> func/memref/arith high-level MLIR (native).
#include "asr_dialect_api.h"

#include <stdio.h>
#include <string.h>
#include <base/string.h>

struct ASR_LoweringContext {
    MLIR_Context *ctx;
    MLIR_LocationHandle loc;
    MLIR_OpHandle module;
    MLIR_BlockHandle module_block;
    MLIR_RegionHandle fn_region;
    MLIR_BlockHandle cur_block;
    MLIR_TypeHandle i32_ty;
    MLIR_TypeHandle i64_ty;
    MLIR_TypeHandle i1_ty;
    MLIR_TypeHandle index_ty;
    MLIR_TypeHandle memref_i32_1_ty;
    int ssa_counter;
    bool block_terminated;
    const ASR_DialectOptions *options;
};

#define MAX_SYMS 256
typedef struct {
    string name;
    MLIR_ValueHandle memref;
    MLIR_TypeHandle memref_ty;
    bool is_array;
    int64_t array_len;
} ASR_SymSlot;

static ASR_SymSlot sym_slots[MAX_SYMS];
static size_t n_sym_slots = 0;

static string arena_ssa(ASR_LoweringContext *lc) {
    char *buf = (char *)arena_alloc(lc->ctx->arena, 32);
    snprintf(buf, 32, "v%d", lc->ssa_counter++);
    return str_from_cstr_view(buf);
}

static MLIR_AttributeHandle get_attr(MLIR_OpHandle op, const char *field) {
    char buf[128];
    snprintf(buf, sizeof(buf), "asr.f.%s", field);
    return MLIR_GetOpAttributeByName(op, buf);
}

static int64_t get_i64(MLIR_OpHandle op, const char *field, int64_t def) {
    MLIR_AttributeHandle a = get_attr(op, field);
    if (a == MLIR_INVALID_HANDLE) return def;
    return MLIR_GetAttributeInteger(a);
}

static bool get_bool(MLIR_OpHandle op, const char *field, bool def) {
    MLIR_AttributeHandle a = get_attr(op, field);
    if (a == MLIR_INVALID_HANDLE) return def;
    return MLIR_GetAttributeBool(a);
}

static string get_str(MLIR_OpHandle op, const char *field) {
    MLIR_AttributeHandle a = get_attr(op, field);
    if (a == MLIR_INVALID_HANDLE) return str_lit("");
    return MLIR_GetAttributeString(a);
}

static MLIR_ValueHandle get_value_attr(MLIR_OpHandle op, const char *field) {
    MLIR_AttributeHandle a = get_attr(op, field);
    if (a == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    return (MLIR_ValueHandle)MLIR_GetAttributeInteger(a);
}

static ASR_SymSlot *lookup_sym(string name) {
    for (size_t i = 0; i < n_sym_slots; ++i) {
        if (str_eq(sym_slots[i].name, name)) {
            return &sym_slots[i];
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

static MLIR_OpHandle get_op_ref(MLIR_OpHandle op, const char *field) {
    return (MLIR_OpHandle)get_i64(op, field, 0);
}

static MLIR_ValueHandle lower_expr_value(ASR_LoweringContext *lc, MLIR_OpHandle op);

bool ASR_LowerIntegerConstant(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    int64_t n = get_i64(op, "n", 0);
    MLIR_ValueHandle v = emit_const_i32(lc, n);
    if (MLIR_GetOpNumResults(op) > 0) {
        (void)v;
    }
    return true;
}

bool ASR_LowerVar(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    string sym = get_str(op, "v");
    ASR_SymSlot *slot = lookup_sym(sym);
    if (!slot || slot->is_array) {
        return ASR_LowerUnsupported(lc, op, "Var: scalar symbol required");
    }
    MLIR_ValueHandle idx = emit_const_index(lc, 0);
    MLIR_ValueHandle res = MLIR_CreateValueOpResult(lc->ctx, MLIR_INVALID_HANDLE,
        0, lc->i32_ty, arena_ssa(lc), lc->loc);
    MLIR_TypeHandle rts[1] = {lc->i32_ty};
    MLIR_ValueHandle rs[1] = {res};
    MLIR_ValueHandle ops[2] = {slot->memref, idx};
    append_current(lc, emit_op(lc, OP_TYPE_MEMREF_LOAD, str_lit("memref.load"),
        NULL, 0, rts, 1, rs, 1, ops, 2));
    (void)res;
    return true;
}

bool ASR_LowerIntegerBinOp(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    MLIR_ValueHandle lhs = lower_expr_value(lc, get_op_ref(op, "left"));
    MLIR_ValueHandle rhs = lower_expr_value(lc, get_op_ref(op, "right"));
    int64_t bop = get_i64(op, "op", 0);
    MLIR_OpType ty = OP_TYPE_ARITH_ADDI;
    string nm = str_lit("arith.addi");
    switch (bop) {
        case 0: ty = OP_TYPE_ARITH_ADDI; nm = str_lit("arith.addi"); break;
        case 1: ty = OP_TYPE_ARITH_SUBI; nm = str_lit("arith.subi"); break;
        case 2: ty = OP_TYPE_ARITH_MULI; nm = str_lit("arith.muli"); break;
        case 3: ty = OP_TYPE_ARITH_DIVSI; nm = str_lit("arith.divsi"); break;
        default: return ASR_LowerUnsupported(lc, op, "IntegerBinOp: unsupported binop");
    }
    MLIR_ValueHandle res = MLIR_CreateValueOpResult(lc->ctx, MLIR_INVALID_HANDLE,
        0, lc->i32_ty, arena_ssa(lc), lc->loc);
    MLIR_TypeHandle rts[1] = {lc->i32_ty};
    MLIR_ValueHandle rs[1] = {res};
    MLIR_ValueHandle ops[2] = {lhs, rhs};
    append_current(lc, emit_op(lc, ty, nm, NULL, 0, rts, 1, rs, 1, ops, 2));
    (void)res;
    return true;
}

bool ASR_LowerVariable(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    string name = get_str(op, "name");
    if (n_sym_slots >= MAX_SYMS) {
        return false;
    }
    ASR_SymSlot *slot = &sym_slots[n_sym_slots++];
    slot->name = name;
    slot->is_array = false;
    slot->array_len = 0;
    slot->memref_ty = lc->memref_i32_1_ty;
    slot->memref = emit_memref_alloca(lc, slot->memref_ty);
    return true;
}

bool ASR_LowerAssignment(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    MLIR_ValueHandle val = lower_expr_value(lc, get_op_ref(op, "value"));
    MLIR_OpHandle target = get_op_ref(op, "target");
    ASR_DialectOpKind tk = ASR_DialectGetOpKind(target);
    if (tk == ASR_DIALECT_OP_EXPR_VAR) {
        string sym = get_str(target, "v");
        ASR_SymSlot *slot = lookup_sym(sym);
        if (!slot) return false;
        MLIR_ValueHandle idx = emit_const_index(lc, 0);
        MLIR_ValueHandle ops[3] = {val, slot->memref, idx};
        append_current(lc, emit_op(lc, OP_TYPE_MEMREF_STORE, str_lit("memref.store"),
            NULL, 0, NULL, 0, NULL, 0, ops, 3));
        return true;
    }
    return ASR_LowerUnsupported(lc, op, "Assignment: unsupported target");
}

bool ASR_LowerPrint(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    MLIR_ValueHandle val = lower_expr_value(lc, get_op_ref(op, "text"));
    MLIR_ValueHandle ops[1] = {val};
    append_current(lc, emit_op(lc, OP_TYPE_VECTOR_PRINT, str_lit("vector.print"),
        NULL, 0, NULL, 0, NULL, 0, ops, 1));
    return true;
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

bool ASR_LowerProgram(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    (void)op;
    lc->fn_region = MLIR_CreateRegion(lc->ctx);
    lc->cur_block = MLIR_CreateBlock(lc->ctx);
    MLIR_AppendRegionBlock(lc->ctx, lc->fn_region, lc->cur_block);
    lc->block_terminated = false;
    return true;
}

bool ASR_LowerTranslationUnit(ASR_LoweringContext *lc, MLIR_OpHandle op) {
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
            ASR_SymSlot *slot = lookup_sym(sym);
            if (!slot) return MLIR_INVALID_HANDLE;
            MLIR_ValueHandle idx = emit_const_index(lc, 0);
            MLIR_ValueHandle res = MLIR_CreateValueOpResult(lc->ctx,
                MLIR_INVALID_HANDLE, 0, lc->i32_ty, arena_ssa(lc), lc->loc);
            MLIR_TypeHandle rts[1] = {lc->i32_ty};
            MLIR_ValueHandle rs[1] = {res};
            MLIR_ValueHandle ops[2] = {slot->memref, idx};
            append_current(lc, emit_op(lc, OP_TYPE_MEMREF_LOAD, str_lit("memref.load"),
                NULL, 0, rts, 1, rs, 1, ops, 2));
            return res;
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
            }
            MLIR_ValueHandle res = MLIR_CreateValueOpResult(lc->ctx,
                MLIR_INVALID_HANDLE, 0, lc->i32_ty, arena_ssa(lc), lc->loc);
            MLIR_TypeHandle rts[1] = {lc->i32_ty};
            MLIR_ValueHandle rs[1] = {res};
            MLIR_ValueHandle ops[2] = {lhs, rhs};
            append_current(lc, emit_op(lc, ty, nm, NULL, 0, rts, 1, rs, 1, ops, 2));
            return res;
        }
        default:
            ASR_LowerUnsupported(lc, op, "expression lowering not implemented");
            return MLIR_INVALID_HANDLE;
    }
}

static void init_types(ASR_LoweringContext *lc) {
    lc->i32_ty = MLIR_CreateTypeInteger(lc->ctx, 32, true);
    lc->i64_ty = MLIR_CreateTypeInteger(lc->ctx, 64, false);
    lc->i1_ty = MLIR_CreateTypeInteger(lc->ctx, 1, false);
    lc->index_ty = MLIR_CreateTypeIndex(lc->ctx);
    int64_t shape[1] = {1};
    lc->memref_i32_1_ty = MLIR_CreateTypeMemref(lc->ctx, shape, 1, lc->i32_ty);
}

// Stub handlers for remaining focused ops (expand incrementally).
bool ASR_LowerIntegerCompare(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "IntegerCompare lowering pending");
}
bool ASR_LowerIntegerUnaryMinus(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "IntegerUnaryMinus lowering pending");
}
bool ASR_LowerIntegerBitNot(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "IntegerBitNot lowering pending");
}
bool ASR_LowerRealConstant(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "RealConstant lowering pending");
}
bool ASR_LowerRealBinOp(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "RealBinOp lowering pending");
}
bool ASR_LowerRealCompare(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "RealCompare lowering pending");
}
bool ASR_LowerRealUnaryMinus(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "RealUnaryMinus lowering pending");
}
bool ASR_LowerCast(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "Cast lowering pending");
}
bool ASR_LowerLogicalConstant(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "LogicalConstant lowering pending");
}
bool ASR_LowerLogicalNot(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "LogicalNot lowering pending");
}
bool ASR_LowerLogicalCompare(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "LogicalCompare lowering pending");
}
bool ASR_LowerArrayItem(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "ArrayItem lowering pending");
}
bool ASR_LowerArrayConstant(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "ArrayConstant lowering pending");
}
bool ASR_LowerArrayConstructor(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "ArrayConstructor lowering pending");
}
bool ASR_LowerArraySize(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "ArraySize lowering pending");
}
bool ASR_LowerArrayBound(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "ArrayBound lowering pending");
}
bool ASR_LowerArraySection(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "ArraySection lowering pending");
}
bool ASR_LowerArrayTranspose(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "ArrayTranspose lowering pending");
}
bool ASR_LowerArrayReshape(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "ArrayReshape lowering pending");
}
bool ASR_LowerArrayPhysicalCast(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "ArrayPhysicalCast lowering pending");
}
bool ASR_LowerIntrinsicElementalFunction(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "IntrinsicElementalFunction lowering pending");
}
bool ASR_LowerIntrinsicArrayFunction(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "IntrinsicArrayFunction lowering pending");
}
bool ASR_LowerIntrinsicImpureFunction(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "IntrinsicImpureFunction lowering pending");
}
bool ASR_LowerIntrinsicImpureSubroutine(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "IntrinsicImpureSubroutine lowering pending");
}
bool ASR_LowerFunction(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "Function lowering pending");
}
bool ASR_LowerAllocate(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "Allocate lowering pending");
}
bool ASR_LowerDoLoop(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "DoLoop lowering pending");
}
bool ASR_LowerIf(ASR_LoweringContext *lc, MLIR_OpHandle op) {
    return ASR_LowerUnsupported(lc, op, "If lowering pending");
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

#include "generated/asr_dialect_lowering_dispatch.inc"

bool ASR_DialectLowerModuleNative(
    MLIR_Context *ctx, MLIR_OpHandle module, const ASR_DialectOptions *options) {
    if (MLIR_GetOpType(module) != OP_TYPE_MODULE) {
        return false;
    }
    n_sym_slots = 0;

    ASR_LoweringContext lc = {};
    lc.ctx = ctx;
    lc.module = module;
    lc.options = options;
    lc.loc = MLIR_CreateLocationUnknown(ctx, str_lit("lfortran"));
    init_types(&lc);

    MLIR_RegionHandle mod_region = MLIR_GetOpRegion(module, 0);
    MLIR_BlockHandle asr_block = MLIR_GetRegionBlock(mod_region, 0);
    size_t n_asr_ops = MLIR_GetBlockNumOps(asr_block);

    MLIR_RegionHandle new_mod_region = MLIR_CreateRegion(ctx);
    lc.module_block = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, new_mod_region, lc.module_block);

    lc.fn_region = MLIR_CreateRegion(ctx);
    lc.cur_block = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, lc.fn_region, lc.cur_block);
    lc.block_terminated = false;

    for (size_t i = 0; i < n_asr_ops; ++i) {
        MLIR_OpHandle op = MLIR_GetBlockOp(asr_block, i);
        if (!ASR_DialectLowerOneOp(&lc, op)) {
            if (!options || !options->allow_unimplemented_nodes) {
                return false;
            }
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
    MLIR_AppendBlockOp(ctx, asr_block, fn);
    (void)new_mod_region;
    return true;
}
