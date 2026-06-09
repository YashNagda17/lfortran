// LFortran-specific MLIR lowering and LLVM IR translation hooks.
//
// Lowering: vector.print -> printf, memref<1xi32> alloca/load/store,
// arith.index_cast, index constants.
// Translation: GEP index types and block labels for LLVM IR emission.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <base/arena.h>
#include <base/string.h>

#include "mlir_api.h"
#include "mlir_lfortran_hooks.h"
#include "mlir_op_names.h"

static bool name_eq(string s, const char *cstr) {
    size_t n = 0;
    while (cstr[n]) n++;
    return s.size == n && memcmp(s.str, cstr, n) == 0;
}

static MLIR_TypeHandle ty_i64(MLIR_Context *ctx) {
    return MLIR_CreateTypeInteger(ctx, 64, false);
}
static MLIR_TypeHandle ty_i32(MLIR_Context *ctx) {
    return MLIR_CreateTypeInteger(ctx, 32, true);
}

static MLIR_ValueHandle make_result_value(MLIR_Context *ctx,
                                          MLIR_TypeHandle type,
                                          MLIR_LocationHandle loc) {
    return MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, type,
                                    str_lit(""), loc);
}

static MLIR_OpHandle create_simple_op(
        MLIR_Context *ctx, MLIR_OpType type, string opname,
        MLIR_AttributeHandle *attrs, size_t n_attrs,
        MLIR_TypeHandle *result_types, size_t n_result_types,
        MLIR_ValueHandle *results, size_t n_results,
        MLIR_ValueHandle *operands, size_t n_operands,
        MLIR_RegionHandle *regions, size_t n_regions,
        MLIR_LocationHandle loc) {
    return MLIR_CreateOp(ctx, type, opname,
                         attrs, n_attrs,
                         result_types, n_result_types,
                         results, n_results,
                         operands, n_operands,
                         regions, n_regions,
                         loc, MLIR_INVALID_HANDLE,
                         str_lit(""), -1);
}

typedef struct MLIR_LFortranLowerState LowerState;

void MLIR_LFortranLowerStateInit(MLIR_LFortranLowerState *st,
        MLIR_Context *ctx, MLIR_OpHandle module) {
    memset(st, 0, sizeof(*st));
    st->ctx = ctx;
    st->module = module;
    if (MLIR_GetOpNumRegions(module) > 0) {
        MLIR_RegionHandle body = MLIR_GetOpRegion(module, 0);
        if (MLIR_GetRegionNumBlocks(body) > 0) {
            st->module_body = MLIR_GetRegionBlock(body, 0);
        }
    }
}

static void append_printf_decl(LowerState *st) {
    if (st->printf_decl) return;
    MLIR_TypeHandle ptr = MLIR_CreateTypeLLVMPointer(st->ctx);
    MLIR_TypeHandle params[1] = { ptr };
    MLIR_TypeHandle i32r = ty_i32(st->ctx);
    MLIR_TypeHandle fn_ty =
        MLIR_CreateTypeLLVMFunction(st->ctx, i32r, params, 1, true);
    MLIR_AttributeHandle attrs[2];
    attrs[0] = MLIR_CreateAttributeString(st->ctx, str_lit("sym_name"),
                                          str_lit("printf"));
    attrs[1] = MLIR_CreateAttributeType(st->ctx, str_lit("function_type"),
                                        fn_ty);
    MLIR_RegionHandle empty_region = MLIR_CreateRegion(st->ctx);
    MLIR_OpHandle decl = create_simple_op(
        st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.func"),
        attrs, 2, NULL, 0, NULL, 0, NULL, 0,
        &empty_region, 1,
        MLIR_CreateLocationUnknown(st->ctx, str_lit("")));
    MLIR_AppendBlockOp(st->ctx, st->module_body, decl);
    st->printf_decl = true;
}

static void ensure_fmt_i64_nl(LowerState *st) {
    if (st->vp_fmt_i64_nl) return;
    static const char raw[] = "%lld\n";
    string bytes = { (char *)raw, sizeof(raw) };
    MLIR_LocationHandle loc =
        MLIR_CreateLocationUnknown(st->ctx, str_lit(""));
    MLIR_OpHandle g = MLIR_CreateLLVMGlobalString(
        st->ctx, str_lit("lfortran_vp_fmt_i64_nl"), bytes, loc);
    MLIR_AppendBlockOp(st->ctx, st->module_body, g);
    st->vp_fmt_i64_nl = true;
}

static void ensure_fmt_g_nl(LowerState *st) {
    if (st->vp_fmt_g_nl) return;
    static const char raw[] = "%g\n";
    string bytes = { (char *)raw, sizeof(raw) };
    MLIR_LocationHandle loc =
        MLIR_CreateLocationUnknown(st->ctx, str_lit(""));
    MLIR_OpHandle g = MLIR_CreateLLVMGlobalString(
        st->ctx, str_lit("lfortran_vp_fmt_g_nl"), bytes, loc);
    MLIR_AppendBlockOp(st->ctx, st->module_body, g);
    st->vp_fmt_g_nl = true;
}

static bool str_contains_substr(string s, const char *sub) {
    size_t n = 0;
    while (sub[n]) n++;
    if (s.size < n) return false;
    for (size_t i = 0; i + n <= s.size; i++) {
        if (memcmp(s.str + i, sub, n) == 0) return true;
    }
    return false;
}

static bool parse_memref_nxi32(string ts, uint64_t *n_out) {
    const char *prefix = "memref<";
    size_t plen = 7;
    if (ts.size < plen + 5 + 1) return false;
    if (memcmp(ts.str, prefix, plen) != 0) return false;
    size_t i = plen;
    uint64_t n = 0;
    while (i < ts.size && ts.str[i] >= '0' && ts.str[i] <= '9') {
        n = n * 10 + (uint64_t)(ts.str[i] - '0');
        i++;
    }
    if (n == 0) return false;
    if (i + 4 >= ts.size) return false;
    if (memcmp(ts.str + i, "xi32", 4) != 0) return false;
    if (ts.str[i + 4] != '>') return false;
    *n_out = n;
    return true;
}

static bool get_i32_array_type_for_mem(MLIR_Context *ctx, MLIR_ValueHandle mem,
                                       MLIR_TypeHandle i32ty,
                                       MLIR_TypeHandle *arrty_out) {
    MLIR_TypeHandle mty = MLIR_GetValueType(mem);
    string ts = MLIR_GetTypeString(ctx, mty);
    uint64_t n = 0;
    if (parse_memref_nxi32(ts, &n)) {
        *arrty_out = MLIR_CreateTypeLLVMArray(ctx, i32ty, n);
        return true;
    }
    MLIR_OpHandle def = MLIR_GetValueDefiningOp(mem);
    if (def == MLIR_INVALID_HANDLE) return false;
    if (!name_eq(MLIR_GetOpName(def), "llvm.alloca")) return false;
    MLIR_AttributeHandle eat = MLIR_GetOpAttributeByName(def, "elem_type");
    if (eat == MLIR_INVALID_HANDLE) return false;
    *arrty_out = MLIR_GetAttributeTypeValue(eat);
    return *arrty_out != MLIR_INVALID_HANDLE;
}

static bool lower_memref_alloca(LowerState *st, MLIR_OpHandle op,
                                MLIR_BlockHandle parent, size_t pos) {
    if (MLIR_GetOpNumResults(op) != 1) return false;
    MLIR_ValueHandle old_res = MLIR_GetOpResult(op, 0);
    MLIR_TypeHandle mty = MLIR_GetValueType(old_res);
    string ts = MLIR_GetTypeString(st->ctx, mty);
    uint64_t n = 0;
    if (!parse_memref_nxi32(ts, &n)) return false;

    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);
    MLIR_TypeHandle i32ty = ty_i32(st->ctx);
    MLIR_TypeHandle arrty = MLIR_CreateTypeLLVMArray(st->ctx, i32ty, n);
    MLIR_TypeHandle ptrty = MLIR_CreateTypeLLVMPointer(st->ctx);
    MLIR_TypeHandle i64ty = ty_i64(st->ctx);

    MLIR_ValueHandle cnt_res = make_result_value(st->ctx, i64ty, loc);
    MLIR_AttributeHandle vattr =
        MLIR_CreateAttributeInteger(st->ctx, str_lit("value"), 1, i64ty);
    MLIR_AttributeHandle cat[1] = { vattr };
    MLIR_TypeHandle crts[1] = { i64ty };
    MLIR_ValueHandle crs[1] = { cnt_res };
    MLIR_OpHandle cnt_op = create_simple_op(
        st->ctx, OP_TYPE_LLVM_MLIR_CONSTANT, str_lit("llvm.mlir.constant"),
        cat, 1, crts, 1, crs, 1, NULL, 0, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, cnt_op, pos);

    MLIR_ValueHandle alloca_res = make_result_value(st->ctx, ptrty, loc);
    MLIR_AttributeHandle eat =
        MLIR_CreateAttributeType(st->ctx, str_lit("elem_type"), arrty);
    MLIR_AttributeHandle aattrs[1] = { eat };
    MLIR_TypeHandle arts[1] = { ptrty };
    MLIR_ValueHandle ars[1] = { alloca_res };
    MLIR_ValueHandle aops[1] = { cnt_res };
    MLIR_OpHandle aop = create_simple_op(
        st->ctx, OP_TYPE_LLVM_ALLOCA, str_lit("llvm.alloca"),
        aattrs, 1, arts, 1, ars, 1, aops, 1, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, aop, pos + 1);

    MLIR_ReplaceAllUsesOfValue(st->ctx, old_res, alloca_res);
    return true;
}

static bool lower_memref_load(LowerState *st, MLIR_OpHandle op,
                              MLIR_BlockHandle parent, size_t pos) {
    if (MLIR_GetOpNumOperands(op) != 2) return false;
    if (MLIR_GetOpNumResults(op) != 1) return false;
    MLIR_ValueHandle mem = MLIR_GetOpOperand(op, 0);
    MLIR_ValueHandle idx = MLIR_GetOpOperand(op, 1);
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);
    MLIR_TypeHandle i32ty = ty_i32(st->ctx);
    MLIR_TypeHandle arrty = MLIR_INVALID_HANDLE;
    if (!get_i32_array_type_for_mem(st->ctx, mem, i32ty, &arrty)) return false;
    MLIR_TypeHandle ptrty = MLIR_CreateTypeLLVMPointer(st->ctx);

    MLIR_ValueHandle gep_res = make_result_value(st->ctx, ptrty, loc);
    int32_t raw_idx[2] = { 0, (int32_t)0x80000000 };
    MLIR_AttributeHandle raw_attr = MLIR_CreateAttributeDenseI32Array(
        st->ctx, str_lit("rawConstantIndices"), raw_idx, 2);
    MLIR_AttributeHandle elem_attr =
        MLIR_CreateAttributeType(st->ctx, str_lit("elem_type"), arrty);
    MLIR_AttributeHandle gattrs[2] = { raw_attr, elem_attr };
    MLIR_TypeHandle gep_rts[1] = { ptrty };
    MLIR_ValueHandle gep_rs[1] = { gep_res };
    MLIR_ValueHandle gep_ops[2] = { mem, idx };
    MLIR_OpHandle gep = create_simple_op(
        st->ctx, OP_TYPE_LLVM_GEP, str_lit("llvm.getelementptr"),
        gattrs, 2, gep_rts, 1, gep_rs, 1, gep_ops, 2, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, gep, pos);

    MLIR_ValueHandle old_res = MLIR_GetOpResult(op, 0);
    MLIR_ValueHandle new_load_res = make_result_value(st->ctx, i32ty, loc);
    MLIR_TypeHandle load_rts[1] = { i32ty };
    MLIR_ValueHandle load_rs[1] = { new_load_res };
    MLIR_ValueHandle load_ops[1] = { gep_res };
    MLIR_OpHandle load = create_simple_op(
        st->ctx, OP_TYPE_LLVM_LOAD, str_lit("llvm.load"),
        NULL, 0, load_rts, 1, load_rs, 1, load_ops, 1, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, load, pos + 1);

    MLIR_ReplaceAllUsesOfValue(st->ctx, old_res, new_load_res);
    return true;
}

static bool lower_memref_store(LowerState *st, MLIR_OpHandle op,
                               MLIR_BlockHandle parent, size_t pos) {
    if (MLIR_GetOpNumOperands(op) != 3) return false;
    MLIR_ValueHandle val = MLIR_GetOpOperand(op, 0);
    MLIR_ValueHandle mem = MLIR_GetOpOperand(op, 1);
    MLIR_ValueHandle idx = MLIR_GetOpOperand(op, 2);
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);
    MLIR_TypeHandle i32ty = ty_i32(st->ctx);
    MLIR_TypeHandle arrty = MLIR_INVALID_HANDLE;
    if (!get_i32_array_type_for_mem(st->ctx, mem, i32ty, &arrty)) return false;
    MLIR_TypeHandle ptrty = MLIR_CreateTypeLLVMPointer(st->ctx);

    MLIR_ValueHandle gep_res = make_result_value(st->ctx, ptrty, loc);
    int32_t raw_idx[2] = { 0, (int32_t)0x80000000 };
    MLIR_AttributeHandle raw_attr = MLIR_CreateAttributeDenseI32Array(
        st->ctx, str_lit("rawConstantIndices"), raw_idx, 2);
    MLIR_AttributeHandle elem_attr =
        MLIR_CreateAttributeType(st->ctx, str_lit("elem_type"), arrty);
    MLIR_AttributeHandle gattrs[2] = { raw_attr, elem_attr };
    MLIR_TypeHandle gep_rts[1] = { ptrty };
    MLIR_ValueHandle gep_rs[1] = { gep_res };
    MLIR_ValueHandle gep_ops[2] = { mem, idx };
    MLIR_OpHandle gep = create_simple_op(
        st->ctx, OP_TYPE_LLVM_GEP, str_lit("llvm.getelementptr"),
        gattrs, 2, gep_rts, 1, gep_rs, 1, gep_ops, 2, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, gep, pos);

    MLIR_ValueHandle st_ops[2] = { val, gep_res };
    MLIR_OpHandle store = create_simple_op(
        st->ctx, OP_TYPE_LLVM_STORE, str_lit("llvm.store"),
        NULL, 0, NULL, 0, NULL, 0, st_ops, 2, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, store, pos + 1);
    return true;
}

static bool lower_vector_print(LowerState *st, MLIR_OpHandle op,
                               MLIR_BlockHandle parent, size_t pos) {
    if (MLIR_GetOpNumOperands(op) != 1) return false;
    MLIR_ValueHandle arg = MLIR_GetOpOperand(op, 0);
    MLIR_TypeHandle xty = MLIR_GetValueType(arg);
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);

    string ts = MLIR_GetTypeString(st->ctx, xty);
    bool is_f32 = ts.size == 3 && ts.str[0] == 'f' &&
        ts.str[1] == '3' && ts.str[2] == '2';
    bool is_f64 = ts.size == 3 && ts.str[0] == 'f' &&
        ts.str[1] == '6' && ts.str[2] == '4';

    size_t p = pos;

    if (is_f32) {
        ensure_fmt_g_nl(st);
        MLIR_TypeHandle f64t = MLIR_CreateTypeFloat(st->ctx, 64, false);
        MLIR_ValueHandle res = make_result_value(st->ctx, f64t, loc);
        MLIR_TypeHandle rts[1] = { f64t };
        MLIR_ValueHandle results[1] = { res };
        MLIR_ValueHandle ops[1] = { arg };
        MLIR_OpHandle fpe = create_simple_op(
            st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.fpext"),
            NULL, 0, rts, 1, results, 1, ops, 1, NULL, 0, loc);
        MLIR_InsertBlockOpAtIndex(st->ctx, parent, fpe, p++);
        arg = res;
    } else if (is_f64) {
        ensure_fmt_g_nl(st);
    } else {
        ensure_fmt_i64_nl(st);
        if (MLIR_IsTypeInteger(xty) &&
            !(ts.size == 3 && ts.str[0] == 'i' && ts.str[1] == '6' &&
                ts.str[2] == '4')) {
            MLIR_TypeHandle i64t = ty_i64(st->ctx);
            MLIR_ValueHandle res = make_result_value(st->ctx, i64t, loc);
            MLIR_TypeHandle rts[1] = { i64t };
            MLIR_ValueHandle results[1] = { res };
            MLIR_ValueHandle ops[1] = { arg };
            MLIR_OpHandle sext = create_simple_op(
                st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.sext"),
                NULL, 0, rts, 1, results, 1, ops, 1, NULL, 0, loc);
            MLIR_InsertBlockOpAtIndex(st->ctx, parent, sext, p++);
            arg = res;
        }
    }

    append_printf_decl(st);

    MLIR_TypeHandle ptr_ty = MLIR_CreateTypeLLVMPointer(st->ctx);
    MLIR_TypeHandle i32r = ty_i32(st->ctx);
    MLIR_TypeHandle p0[1] = { ptr_ty };
    MLIR_TypeHandle printf_ty =
        MLIR_CreateTypeLLVMFunction(st->ctx, i32r, p0, 1, true);

    string gsym;
    if (is_f32 || is_f64) {
        gsym = str_lit("lfortran_vp_fmt_g_nl");
    } else {
        gsym = str_lit("lfortran_vp_fmt_i64_nl");
    }

    MLIR_ValueHandle fmt_ptr = make_result_value(st->ctx, ptr_ty, loc);
    MLIR_AttributeHandle addr_attrs[1];
    addr_attrs[0] = MLIR_CreateAttributeSymbolRef(
        st->ctx, str_lit("global_name"), gsym);
    MLIR_TypeHandle addr_rts[1] = { ptr_ty };
    MLIR_ValueHandle addr_results[1] = { fmt_ptr };
    MLIR_OpHandle addr_op = create_simple_op(
        st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.mlir.addressof"),
        addr_attrs, 1, addr_rts, 1, addr_results, 1, NULL, 0, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, addr_op, p++);

    MLIR_ValueHandle call_res = make_result_value(st->ctx, i32r, loc);
    MLIR_AttributeHandle call_attrs[2];
    call_attrs[0] = MLIR_CreateAttributeSymbolRef(
        st->ctx, str_lit("callee"), str_lit("printf"));
    call_attrs[1] = MLIR_CreateAttributeType(
        st->ctx, str_lit("var_callee_type"), printf_ty);
    MLIR_TypeHandle call_rts[1] = { i32r };
    MLIR_ValueHandle call_results[1] = { call_res };
    MLIR_ValueHandle call_ops[2] = { fmt_ptr, arg };
    MLIR_OpHandle call_op = create_simple_op(
        st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.call"),
        call_attrs, 2, call_rts, 1, call_results, 1, call_ops, 2, NULL, 0,
        loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, call_op, p++);
    (void)call_res;

    return true;
}

static bool lower_index_arith_constant(LowerState *st, MLIR_OpHandle op,
                                       MLIR_BlockHandle parent, size_t pos) {
    if (MLIR_GetOpNumResults(op) != 1) return false;
    MLIR_ValueHandle old_res = MLIR_GetOpResult(op, 0);
    MLIR_TypeHandle ty = MLIR_GetValueType(old_res);
    string ts = MLIR_GetTypeString(st->ctx, ty);
    if (!(ts.size == 5 && memcmp(ts.str, "index", 5) == 0)) return false;

    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);
    size_t na = MLIR_GetOpNumAttributes(op);
    MLIR_AttributeHandle val_attr = MLIR_INVALID_HANDLE;
    for (size_t i = 0; i < na; i++) {
        MLIR_AttributeHandle a = MLIR_GetOpAttribute(op, i);
        if (name_eq(MLIR_GetAttributeName(a), "value")) { val_attr = a; break; }
    }
    if (val_attr == MLIR_INVALID_HANDLE) return false;

    ty = ty_i64(st->ctx);
    int64_t iv = MLIR_GetAttributeInteger(val_attr);
    val_attr = MLIR_CreateAttributeInteger(st->ctx, str_lit("value"), iv, ty);

    MLIR_ValueHandle new_res = make_result_value(st->ctx, ty, loc);
    MLIR_AttributeHandle attrs[1] = { val_attr };
    MLIR_TypeHandle rts[1] = { ty };
    MLIR_ValueHandle results[1] = { new_res };
    MLIR_OpHandle nop = create_simple_op(
        st->ctx, OP_TYPE_LLVM_MLIR_CONSTANT, str_lit("llvm.mlir.constant"),
        attrs, 1, rts, 1, results, 1, NULL, 0, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, nop, pos);
    MLIR_ReplaceAllUsesOfValue(st->ctx, old_res, new_res);
    return true;
}

static bool lower_arith_index_cast(LowerState *st, MLIR_OpHandle op,
                                   MLIR_BlockHandle parent, size_t pos) {
    if (MLIR_GetOpNumOperands(op) != 1 || MLIR_GetOpNumResults(op) != 1)
        return false;
    MLIR_ValueHandle src = MLIR_GetOpOperand(op, 0);
    MLIR_TypeHandle src_ty = MLIR_GetValueType(src);
    MLIR_TypeHandle dst_ty = MLIR_GetOpResult_type(op, 0);
    string dst_ts = MLIR_GetTypeString(st->ctx, dst_ty);
    if (!(dst_ts.size == 5 && memcmp(dst_ts.str, "index", 5) == 0))
        return false;

    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);
    MLIR_TypeHandle i64ty = ty_i64(st->ctx);
    MLIR_ValueHandle old_res = MLIR_GetOpResult(op, 0);
    MLIR_ValueHandle new_res = make_result_value(st->ctx, i64ty, loc);
    MLIR_TypeHandle rts[1] = {i64ty};
    MLIR_ValueHandle results[1] = {new_res};
    MLIR_ValueHandle ops[1] = {src};
    string src_ts = MLIR_GetTypeString(st->ctx, src_ty);
    if (src_ts.size == 3 && src_ts.str[0] == 'i' && src_ts.str[1] == '6' &&
        src_ts.str[2] == '4') {
        MLIR_ReplaceAllUsesOfValue(st->ctx, old_res, src);
        return true;
    }
    if (!(src_ts.size == 3 && src_ts.str[0] == 'i' && src_ts.str[1] == '3' &&
          src_ts.str[2] == '2') && !MLIR_IsTypeInteger(src_ty)) {
        return false;
    }
    MLIR_OpHandle nop = create_simple_op(
        st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.sext"),
        NULL, 0, rts, 1, results, 1, ops, 1, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, nop, pos);
    MLIR_ReplaceAllUsesOfValue(st->ctx, old_res, new_res);
    return true;
}

static void walk_block_vector_print(LowerState *st, MLIR_BlockHandle block);

static void walk_op_vector_print(LowerState *st, MLIR_OpHandle op) {
    size_t nr = MLIR_GetOpNumRegions(op);
    for (size_t r = 0; r < nr; r++) {
        MLIR_RegionHandle reg = MLIR_GetOpRegion(op, r);
        for (size_t b = 0; b < MLIR_GetRegionNumBlocks(reg); b++) {
            walk_block_vector_print(st, MLIR_GetRegionBlock(reg, b));
        }
    }
}

static void walk_block_vector_print(LowerState *st, MLIR_BlockHandle block) {
    size_t i = 0;
    while (i < MLIR_GetBlockNumOps(block)) {
        MLIR_OpHandle op = MLIR_GetBlockOp(block, i);
        walk_op_vector_print(st, op);
        if (name_eq(MLIR_GetOpName(op), "vector.print")) {
            size_t before = MLIR_GetBlockNumOps(block);
            if (lower_vector_print(st, op, block, i)) {
                size_t inserted = MLIR_GetBlockNumOps(block) - before;
                MLIR_EraseOp(st->ctx, op);
                i += inserted;
                continue;
            }
        }
        i++;
    }
}

bool mlir_lower_vector_print_native(MLIR_LFortranLowerState *st) {
    if (st->module == MLIR_INVALID_HANDLE) return false;
    if (st->module_body == MLIR_INVALID_HANDLE) return false;
    walk_block_vector_print(st, st->module_body);
    return true;
}

bool MLIR_LFortranTryLowerOp(MLIR_LFortranLowerState *st, MLIR_OpHandle op,
        MLIR_BlockHandle parent, size_t pos) {
    string name = MLIR_GetOpName(op);
    if (name_eq(name, "memref.alloca"))
        return lower_memref_alloca(st, op, parent, pos);
    if (name_eq(name, "memref.load"))
        return lower_memref_load(st, op, parent, pos);
    if (name_eq(name, "memref.store"))
        return lower_memref_store(st, op, parent, pos);
    if (name_eq(name, "arith.constant"))
        return lower_index_arith_constant(st, op, parent, pos);
    if (name_eq(name, "arith.index_cast"))
        return lower_arith_index_cast(st, op, parent, pos);
    if (name_eq(name, "vector.print"))
        return lower_vector_print(st, op, parent, pos);
    return false;
}

bool MLIR_LFortranPrintGepIndexType(MLIR_LFortranBuf *out, MLIR_Context *ctx,
                                    MLIR_TypeHandle ty,
                                    MLIR_LFortranPrintTypeFn default_print) {
    (void)default_print;
    string s = MLIR_GetTypeString(ctx, ty);
    if (s.size == 5 && memcmp(s.str, "index", 5) == 0) {
        const char *lit = "i64";
        size_t n = 3;
        if (out->len + n > out->cap) {
            size_t nc = out->cap ? out->cap : 1024;
            while (out->len + n > nc) nc *= 2;
            char *nd = (char *)realloc(out->data, nc);
            if (!nd) return false;
            out->data = nd;
            out->cap = nc;
        }
        memcpy(out->data + out->len, lit, n);
        out->len += n;
        return true;
    }
    return false;
}

bool MLIR_LFortranShouldEmitBlockLabel(size_t block_index) {
    return block_index != 0;
}
