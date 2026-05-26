#include <libasr/codegen/asr_to_mlir_new.h>
#include <libasr/codegen/evaluator.h>
#include <libasr/diagnostics.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

extern "C" {
#include <mlir_api.h>
#include <base/arena.h>
#include <base/string.h>
#include <platform/platform.h>
}

namespace LCompilers {

std::once_flag mlir_corec_platform_init;

void ensure_mlir_corec_platform_initialized() {
    std::call_once(mlir_corec_platform_init,
                   []() { platform_init(0, nullptr); });
}

uint64_t node_hash(const ASR::asr_t *node) { return (uint64_t)node; }

class CodeGenError {
public:
    diag::Diagnostic d;
    explicit CodeGenError(const std::string &msg)
        : d{diag::Diagnostic(msg, diag::Level::Error, diag::Stage::CodeGen)} {}
    CodeGenError(const std::string &msg, const Location &loc)
        : d{diag::Diagnostic(msg, diag::Level::Error, diag::Stage::CodeGen,
              {diag::Label("", {loc})})} {}
};

static std::string copy_mlir_string(string s) {
    if (!s.str || s.size == 0) {
        return std::string();
    }
    return std::string(s.str, s.str + s.size);
}

static bool use_upstream_mlir_lowering() {
    const char *e = std::getenv("USE_MLIR_Upstream");
    return e && e[0] == '1' && e[1] == '\0';
}

static string arena_ssa_name(Arena *arena, int id) {
    char *buf = (char *)arena_alloc(arena, 48);
    int n = snprintf(buf, 48, "v%d", id);
    if (n < 0) {
        std::abort();
    }
    return str_from_cstr_len_view(buf, (uint64_t)n);
}

// arith.cmpi / llvm.icmp predicate (signed): eq=0, ne=1, slt=2, sle=3, sgt=4, sge=5
static int64_t icmp_predicate_for(ASR::cmpopType op) {
    switch (op) {
        case ASR::cmpopType::Eq:
            return 0;
        case ASR::cmpopType::NotEq:
            return 1;
        case ASR::cmpopType::Lt:
            return 2;
        case ASR::cmpopType::LtE:
            return 3;
        case ASR::cmpopType::Gt:
            return 4;
        case ASR::cmpopType::GtE:
            return 5;
        default:
            return -1;
    }
}

struct SymSlot {
    MLIR_ValueHandle memref = MLIR_INVALID_HANDLE;
    MLIR_TypeHandle memref_ty = MLIR_INVALID_HANDLE;
    bool is_array = false;
    int64_t array_len = 0;
};

struct MlirEmitter {
    Arena *arena = nullptr;
    MLIR_Context ctx{};
    MLIR_LocationHandle loc = MLIR_INVALID_HANDLE;
    MLIR_TypeHandle i32_ty = MLIR_INVALID_HANDLE;
    MLIR_TypeHandle i64_ty = MLIR_INVALID_HANDLE;
    MLIR_TypeHandle i1_ty = MLIR_INVALID_HANDLE;
    MLIR_TypeHandle index_ty = MLIR_INVALID_HANDLE;
    MLIR_TypeHandle memref_i32_1_ty = MLIR_INVALID_HANDLE;
    MLIR_BlockHandle module_block = MLIR_INVALID_HANDLE;
    MLIR_BlockHandle cur_block = MLIR_INVALID_HANDLE;
    MLIR_RegionHandle fn_body_r = MLIR_INVALID_HANDLE;
    int ssa_counter = 0;
    std::map<uint64_t, SymSlot> symtab;
    MLIR_ValueHandle last_value = MLIR_INVALID_HANDLE;
    MLIR_OpHandle module_op = MLIR_INVALID_HANDLE;
    bool block_terminated = false;

    static void check_mlir(bool ok, const char *stage) {
        if (!ok) {
            throw CodeGenError(std::string("mlir-new: failed at ") + stage);
        }
    }

    MLIR_OpHandle emit_simple_op(MLIR_OpType ty, string opname,
            MLIR_AttributeHandle *attrs, size_t n_attrs,
            MLIR_TypeHandle *result_types, size_t n_result_types,
            MLIR_ValueHandle *results, size_t n_results,
            MLIR_ValueHandle *operands, size_t n_operands,
            MLIR_RegionHandle *regions, size_t n_regions) {
        return MLIR_CreateOp(&ctx, ty, opname, attrs, n_attrs, result_types,
            n_result_types, results, n_results, operands, n_operands, regions,
            n_regions, loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    }

    MLIR_OpHandle emit_branch_op(string opname, MLIR_OpType ty,
            MLIR_ValueHandle *operands, size_t n_operands,
            MLIR_BlockHandle *successors, size_t n_successors) {
        MLIR_ValueHandle **sops = nullptr;
        size_t *sn = nullptr;
        if (n_successors > 0) {
            sops = (MLIR_ValueHandle **)arena_alloc(
                arena, n_successors * sizeof(MLIR_ValueHandle *));
            sn = (size_t *)arena_alloc(arena, n_successors * sizeof(size_t));
            for (size_t i = 0; i < n_successors; i++) {
                sops[i] = nullptr;
                sn[i] = 0;
            }
        }
        return MLIR_CreateOpWithSuccessors(&ctx, ty, opname,
            nullptr, 0, nullptr, 0, nullptr, 0, operands, n_operands,
            nullptr, 0, successors, n_successors, sops, sn,
            loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    }

    void append_current(MLIR_OpHandle op) {
        check_mlir(cur_block != MLIR_INVALID_HANDLE, "missing current block");
        check_mlir(!block_terminated, "appending to terminated block");
        MLIR_AppendBlockOp(&ctx, cur_block, op);
    }

    void append_module(MLIR_OpHandle op) {
        check_mlir(module_block != MLIR_INVALID_HANDLE, "missing module block");
        MLIR_AppendBlockOp(&ctx, module_block, op);
    }

    MLIR_BlockHandle new_cfg_block() {
        check_mlir(fn_body_r != MLIR_INVALID_HANDLE, "missing function region");
        MLIR_BlockHandle b = MLIR_CreateBlock(&ctx);
        MLIR_AppendRegionBlock(&ctx, fn_body_r, b);
        return b;
    }

    void emit_branch(MLIR_BlockHandle target) {
        if (block_terminated) {
            return;
        }
        MLIR_BlockHandle succs[1] = {target};
        MLIR_OpHandle op = emit_branch_op(str_lit("cf.br"), OP_TYPE_CF_BR,
            nullptr, 0, succs, 1);
        append_current(op);
        block_terminated = true;
    }

    void emit_cond_branch(MLIR_ValueHandle cond, MLIR_BlockHandle true_b,
            MLIR_BlockHandle false_b) {
        if (block_terminated) {
            return;
        }
        MLIR_ValueHandle ops[1] = {cond};
        MLIR_BlockHandle succs[2] = {true_b, false_b};
        MLIR_OpHandle op = emit_branch_op(str_lit("cf.cond_br"),
            OP_TYPE_CF_COND_BR, ops, 1, succs, 2);
        append_current(op);
        block_terminated = true;
    }

    void emit_module_skeleton() {
        arena = arena_create(131072);
        if (!arena) {
            throw CodeGenError("mlir-new: arena_create failed");
        }
        MLIR_SetArenaAllocator(&ctx, arena);
        loc = MLIR_CreateLocationUnknown(&ctx, str_lit("lfortran"));
        i32_ty = MLIR_CreateTypeInteger(&ctx, 32, true);
        i64_ty = MLIR_CreateTypeInteger(&ctx, 64, false);
        i1_ty = MLIR_CreateTypeInteger(&ctx, 1, false);
        index_ty = MLIR_CreateTypeIndex(&ctx);
        int64_t shape_one[1] = {1};
        memref_i32_1_ty = MLIR_CreateTypeMemref(&ctx, shape_one, 1, i32_ty);

        MLIR_RegionHandle mr = MLIR_CreateRegion(&ctx);
        module_block = MLIR_CreateBlock(&ctx);
        MLIR_AppendRegionBlock(&ctx, mr, module_block);
        MLIR_RegionHandle mregs[1] = {mr};
        module_op = MLIR_CreateOp(&ctx, OP_TYPE_MODULE, str_lit("module"),
            nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, mregs, 1,
            loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
        check_mlir(module_op != MLIR_INVALID_HANDLE, "builtin.module");
    }

    MLIR_TypeHandle memref_ty_for_shape(int64_t len) {
        int64_t shape[1] = {len};
        return MLIR_CreateTypeMemref(&ctx, shape, 1, i32_ty);
    }

    MLIR_ValueHandle emit_const_i32(int64_t value) {
        MLIR_ValueHandle res = MLIR_CreateValueOpResult(
            &ctx, MLIR_INVALID_HANDLE, 0, i32_ty,
            arena_ssa_name(arena, ssa_counter++), loc);
        MLIR_TypeHandle rts[1] = {i32_ty};
        MLIR_ValueHandle rs[1] = {res};
        MLIR_AttributeHandle val =
            MLIR_CreateAttributeInteger(&ctx, str_lit("value"), value, i32_ty);
        MLIR_AttributeHandle as[1] = {val};
        MLIR_OpHandle op = emit_simple_op(OP_TYPE_ARITH_CONSTANT,
            str_lit("arith.constant"), as, 1, rts, 1, rs, 1, nullptr, 0,
            nullptr, 0);
        append_current(op);
        return res;
    }

    MLIR_ValueHandle emit_const_index(int64_t value) {
        MLIR_ValueHandle res = MLIR_CreateValueOpResult(
            &ctx, MLIR_INVALID_HANDLE, 0, index_ty,
            arena_ssa_name(arena, ssa_counter++), loc);
        MLIR_TypeHandle rts[1] = {index_ty};
        MLIR_ValueHandle rs[1] = {res};
        MLIR_AttributeHandle val =
            MLIR_CreateAttributeInteger(&ctx, str_lit("value"), value, index_ty);
        MLIR_AttributeHandle as[1] = {val};
        MLIR_OpHandle op = emit_simple_op(OP_TYPE_ARITH_CONSTANT,
            str_lit("arith.constant"), as, 1, rts, 1, rs, 1, nullptr, 0,
            nullptr, 0);
        append_current(op);
        return res;
    }

    MLIR_ValueHandle emit_const_i1(bool value) {
        MLIR_ValueHandle res = MLIR_CreateValueOpResult(
            &ctx, MLIR_INVALID_HANDLE, 0, i1_ty,
            arena_ssa_name(arena, ssa_counter++), loc);
        MLIR_TypeHandle rts[1] = {i1_ty};
        MLIR_ValueHandle rs[1] = {res};
        MLIR_AttributeHandle val = MLIR_CreateAttributeInteger(
            &ctx, str_lit("value"), value ? 1 : 0, i1_ty);
        MLIR_AttributeHandle as[1] = {val};
        MLIR_OpHandle op = emit_simple_op(OP_TYPE_ARITH_CONSTANT,
            str_lit("arith.constant"), as, 1, rts, 1, rs, 1, nullptr, 0,
            nullptr, 0);
        append_current(op);
        return res;
    }

    MLIR_ValueHandle i32_to_index(MLIR_ValueHandle v) {
        MLIR_ValueHandle res = MLIR_CreateValueOpResult(&ctx,
            MLIR_INVALID_HANDLE, 0, index_ty,
            arena_ssa_name(arena, ssa_counter++), loc);
        MLIR_TypeHandle rts[1] = {index_ty};
        MLIR_ValueHandle rs[1] = {res};
        MLIR_ValueHandle ops[1] = {v};
        MLIR_OpHandle op = emit_simple_op(OP_TYPE_ARITH_INDEX_CAST,
            str_lit("arith.index_cast"), nullptr, 0, rts, 1, rs, 1, ops, 1,
            nullptr, 0);
        append_current(op);
        return res;
    }

    MLIR_ValueHandle emit_memref_alloca(MLIR_TypeHandle memref_ty) {
        MLIR_ValueHandle res = MLIR_CreateValueOpResult(&ctx,
            MLIR_INVALID_HANDLE, 0, memref_ty,
            arena_ssa_name(arena, ssa_counter++), loc);
        MLIR_TypeHandle rts[1] = {memref_ty};
        MLIR_ValueHandle rs[1] = {res};
        MLIR_OpHandle op = emit_simple_op(OP_TYPE_UNREGISTERED,
            str_lit("memref.alloca"), nullptr, 0, rts, 1, rs, 1, nullptr, 0,
            nullptr, 0);
        append_current(op);
        return res;
    }

    void emit_memref_store_i32(MLIR_ValueHandle val, MLIR_ValueHandle memref,
            MLIR_ValueHandle idx) {
        MLIR_ValueHandle ops[3] = {val, memref, idx};
        MLIR_OpHandle op = emit_simple_op(OP_TYPE_MEMREF_STORE,
            str_lit("memref.store"), nullptr, 0, nullptr, 0, nullptr, 0, ops, 3,
            nullptr, 0);
        append_current(op);
    }

    MLIR_ValueHandle emit_memref_load_i32(MLIR_ValueHandle memref,
            MLIR_ValueHandle idx) {
        MLIR_ValueHandle res = MLIR_CreateValueOpResult(&ctx,
            MLIR_INVALID_HANDLE, 0, i32_ty,
            arena_ssa_name(arena, ssa_counter++), loc);
        MLIR_TypeHandle rts[1] = {i32_ty};
        MLIR_ValueHandle rs[1] = {res};
        MLIR_ValueHandle ops[2] = {memref, idx};
        MLIR_OpHandle op = emit_simple_op(OP_TYPE_MEMREF_LOAD,
            str_lit("memref.load"), nullptr, 0, rts, 1, rs, 1, ops, 2, nullptr,
            0);
        append_current(op);
        return res;
    }

    void emit_memref_store_scalar(MLIR_ValueHandle val, MLIR_ValueHandle memref) {
        MLIR_ValueHandle c0 = emit_const_index(0);
        emit_memref_store_i32(val, memref, c0);
    }

    MLIR_ValueHandle emit_memref_load_scalar(MLIR_ValueHandle memref) {
        MLIR_ValueHandle c0 = emit_const_index(0);
        return emit_memref_load_i32(memref, c0);
    }

    MLIR_ValueHandle emit_binop_i32(MLIR_OpType ty, string opname,
            MLIR_ValueHandle a, MLIR_ValueHandle b) {
        MLIR_ValueHandle res = MLIR_CreateValueOpResult(&ctx,
            MLIR_INVALID_HANDLE, 0, i32_ty,
            arena_ssa_name(arena, ssa_counter++), loc);
        MLIR_TypeHandle rts[1] = {i32_ty};
        MLIR_ValueHandle rs[1] = {res};
        MLIR_ValueHandle ops[2] = {a, b};
        MLIR_OpHandle op = emit_simple_op(ty, opname, nullptr, 0, rts, 1, rs,
            1, ops, 2, nullptr, 0);
        append_current(op);
        return res;
    }

    MLIR_ValueHandle emit_binop_i1(MLIR_OpType ty, string opname,
            MLIR_ValueHandle a, MLIR_ValueHandle b) {
        MLIR_ValueHandle res = MLIR_CreateValueOpResult(&ctx,
            MLIR_INVALID_HANDLE, 0, i1_ty,
            arena_ssa_name(arena, ssa_counter++), loc);
        MLIR_TypeHandle rts[1] = {i1_ty};
        MLIR_ValueHandle rs[1] = {res};
        MLIR_ValueHandle ops[2] = {a, b};
        MLIR_OpHandle op = emit_simple_op(ty, opname, nullptr, 0, rts, 1, rs,
            1, ops, 2, nullptr, 0);
        append_current(op);
        return res;
    }

    MLIR_ValueHandle emit_icmp_i32(int64_t predicate, MLIR_ValueHandle a,
            MLIR_ValueHandle b) {
        MLIR_ValueHandle res = MLIR_CreateValueOpResult(&ctx,
            MLIR_INVALID_HANDLE, 0, i1_ty,
            arena_ssa_name(arena, ssa_counter++), loc);
        MLIR_TypeHandle rts[1] = {i1_ty};
        MLIR_ValueHandle rs[1] = {res};
        MLIR_ValueHandle ops[2] = {a, b};
        MLIR_AttributeHandle pred = MLIR_CreateAttributeInteger(&ctx,
            str_lit("predicate"), predicate, i64_ty);
        MLIR_AttributeHandle as[1] = {pred};
        MLIR_OpHandle op = emit_simple_op(OP_TYPE_ARITH_CMPI,
            str_lit("arith.cmpi"), as, 1, rts, 1, rs, 1, ops, 2, nullptr, 0);
        append_current(op);
        return res;
    }

    void emit_vector_print(MLIR_ValueHandle v) {
        MLIR_ValueHandle ops[1] = {v};
        MLIR_OpHandle op = emit_simple_op(OP_TYPE_VECTOR_PRINT,
            str_lit("vector.print"), nullptr, 0, nullptr, 0, nullptr, 0, ops,
            1, nullptr, 0);
        append_current(op);
    }

    void emit_func_return_i32(MLIR_ValueHandle ret) {
        MLIR_ValueHandle ops[1] = {ret};
        MLIR_OpHandle op = emit_simple_op(OP_TYPE_FUNC_RETURN,
            str_lit("func.return"), nullptr, 0, nullptr, 0, nullptr, 0, ops, 1,
            nullptr, 0);
        append_current(op);
        block_terminated = true;
    }

    bool is_supported_integer_type(ASR::ttype_t *type) {
        type = ASRUtils::type_get_past_allocatable(
            ASRUtils::type_get_past_pointer(type));
        return ASRUtils::is_integer(*type);
    }

    bool is_fixed_integer_array(ASR::ttype_t *type, int64_t &len) {
        type = ASRUtils::type_get_past_allocatable(
            ASRUtils::type_get_past_pointer(type));
        if (!ASR::is_a<ASR::Array_t>(*type)) {
            return false;
        }
        if (!ASRUtils::is_fixed_size_array(type)) {
            return false;
        }
        ASR::ttype_t *elem = ASRUtils::type_get_past_array(type);
        if (!ASRUtils::is_integer(*elem)) {
            return false;
        }
        len = ASRUtils::get_fixed_size_of_array(type);
        return len > 0;
    }

    SymSlot &require_slot(uint64_t h, const Location &loc) {
        auto it = symtab.find(h);
        if (it == symtab.end()) {
            throw CodeGenError("symbol not found", loc);
        }
        return it->second;
    }

    SymSlot &slot_for_var(ASR::Variable_t *v) {
        return require_slot(node_hash(reinterpret_cast<const ASR::asr_t *>(v)),
            v->base.base.loc);
    }

    MLIR_ValueHandle fortran_to_zero_based_index(ASR::expr_t *idx_expr,
            ASR::dimension_t *dim) {
        visit_expr_value(*idx_expr);
        MLIR_ValueHandle fortran_idx = last_value;
        int64_t lbound = 1;
        if (dim && dim->m_start) {
            if (!ASRUtils::extract_value(ASRUtils::expr_value(dim->m_start),
                    lbound)) {
                throw CodeGenError(
                    "mlir-new: array lower bound must be constant for now",
                    idx_expr->base.loc);
            }
        }
        if (lbound == 0) {
            return i32_to_index(fortran_idx);
        }
        MLIR_ValueHandle lb = emit_const_i32(lbound);
        MLIR_ValueHandle z = emit_binop_i32(OP_TYPE_ARITH_SUBI,
            str_lit("arith.subi"), fortran_idx, lb);
        return i32_to_index(z);
    }

    void emit_variable_storage(const ASR::Variable_t &x) {
        uint64_t h = node_hash(reinterpret_cast<const ASR::asr_t *>(&x));
        int64_t array_len = 0;
        if (is_fixed_integer_array(x.m_type, array_len)) {
            SymSlot slot;
            slot.is_array = true;
            slot.array_len = array_len;
            slot.memref_ty = memref_ty_for_shape(array_len);
            slot.memref = emit_memref_alloca(slot.memref_ty);
            symtab[h] = slot;
            return;
        }
        if (!is_supported_integer_type(x.m_type)) {
            throw CodeGenError(
                "mlir-new: only integer scalars and fixed-size integer arrays "
                "are supported for now",
                x.base.base.loc);
        }
        SymSlot slot;
        slot.memref_ty = memref_i32_1_ty;
        slot.memref = emit_memref_alloca(memref_i32_1_ty);
        symtab[h] = slot;
    }

    void init_array_from_constant(const ASR::Variable_t &x, SymSlot &slot,
            const ASR::ArrayConstant_t &ac) {
        if (!slot.is_array) {
            throw CodeGenError("mlir-new: internal array init mismatch",
                x.base.base.loc);
        }
        int64_t n_elem = ac.m_n_data;
        if (n_elem > slot.array_len) {
            n_elem = slot.array_len;
        }
        ASR::ttype_t *elem_type = ASRUtils::type_get_past_array(ac.m_type);
        if (!ASRUtils::is_integer(*elem_type)) {
            throw CodeGenError(
                "mlir-new: only integer array constants are supported",
                x.base.base.loc);
        }
        int64_t kind = ASRUtils::extract_kind_from_ttype_t(elem_type);
        for (int64_t i = 0; i < n_elem; i++) {
            int64_t value = 0;
            if (kind == 4) {
                value = ((int32_t *)ac.m_data)[i];
            } else if (kind == 8) {
                value = ((int64_t *)ac.m_data)[i];
            } else {
                throw CodeGenError(
                    "mlir-new: unsupported integer kind in array constant",
                    x.base.base.loc);
            }
            MLIR_ValueHandle idx = emit_const_index(i);
            MLIR_ValueHandle c = emit_const_i32(value);
            emit_memref_store_i32(c, slot.memref, idx);
        }
    }

    void emit_variable_static_init(const ASR::Variable_t &x) {
        if (!x.m_symbolic_value) {
            return;
        }
        SymSlot &slot = slot_for_var(const_cast<ASR::Variable_t *>(&x));
        if (ASR::is_a<ASR::ArrayConstant_t>(*x.m_symbolic_value)) {
            init_array_from_constant(x, slot,
                *ASR::down_cast<ASR::ArrayConstant_t>(x.m_symbolic_value));
            return;
        }
        visit_expr_value(*x.m_symbolic_value);
        if (slot.is_array) {
            throw CodeGenError(
                "mlir-new: scalar init for array not supported",
                x.base.base.loc);
        }
        emit_memref_store_scalar(last_value, slot.memref);
    }

    void visit_IntegerBinOp(const ASR::IntegerBinOp_t &x) {
        visit_expr_value(*x.m_left);
        MLIR_ValueHandle lhs = last_value;
        visit_expr_value(*x.m_right);
        MLIR_ValueHandle rhs = last_value;
        switch (x.m_op) {
            case ASR::binopType::Add:
                last_value = emit_binop_i32(OP_TYPE_ARITH_ADDI,
                    str_lit("arith.addi"), lhs, rhs);
                return;
            case ASR::binopType::Sub:
                last_value = emit_binop_i32(OP_TYPE_ARITH_SUBI,
                    str_lit("arith.subi"), lhs, rhs);
                return;
            case ASR::binopType::Mul:
                last_value = emit_binop_i32(OP_TYPE_ARITH_MULI,
                    str_lit("arith.muli"), lhs, rhs);
                return;
            case ASR::binopType::Div:
                last_value = emit_binop_i32(OP_TYPE_ARITH_DIVSI,
                    str_lit("arith.divsi"), lhs, rhs);
                return;
            default:
                throw CodeGenError(
                    "mlir-new: integer binary operator not supported yet",
                    x.base.base.loc);
        }
    }

    void visit_expr_value(ASR::expr_t &e) {
        if (ASR::is_a<ASR::IntegerConstant_t>(e)) {
            const ASR::IntegerConstant_t *ic =
                ASR::down_cast<ASR::IntegerConstant_t>(&e);
            last_value = emit_const_i32(ic->m_n);
            return;
        }
        if (ASR::is_a<ASR::Var_t>(e)) {
            ASR::Variable_t *v = ASRUtils::EXPR2VAR(&e);
            SymSlot &slot = slot_for_var(v);
            if (slot.is_array) {
                throw CodeGenError(
                    "mlir-new: use array element in expression context",
                    e.base.loc);
            }
            last_value = emit_memref_load_scalar(slot.memref);
            return;
        }
        if (ASR::is_a<ASR::ArrayItem_t>(e)) {
            const ASR::ArrayItem_t &ai =
                *ASR::down_cast<ASR::ArrayItem_t>(&e);
            if (!ASR::is_a<ASR::Var_t>(*ai.m_v)) {
                throw CodeGenError(
                    "mlir-new: array base must be a variable for now",
                    e.base.loc);
            }
            ASR::Variable_t *v = ASRUtils::EXPR2VAR(ai.m_v);
            SymSlot &slot = slot_for_var(v);
            if (!slot.is_array) {
                throw CodeGenError("mlir-new: indexing a non-array variable",
                    e.base.loc);
            }
            if (ai.n_args != 1) {
                throw CodeGenError(
                    "mlir-new: only rank-1 array indexing is supported for now",
                    e.base.loc);
            }
            ASR::dimension_t *dims = nullptr;
            size_t n_dims = ASRUtils::extract_dimensions_from_ttype(v->m_type, dims);
            ASR::dimension_t *dim = (n_dims > 0) ? &dims[0] : nullptr;
            MLIR_ValueHandle idx = fortran_to_zero_based_index(
                ai.m_args[0].m_right, dim);
            last_value = emit_memref_load_i32(slot.memref, idx);
            return;
        }
        if (ASR::is_a<ASR::IntegerBinOp_t>(e)) {
            visit_IntegerBinOp(*ASR::down_cast<ASR::IntegerBinOp_t>(&e));
            return;
        }
        throw CodeGenError(
            "mlir-new: expression kind not supported yet", e.base.loc);
    }

    MLIR_ValueHandle visit_expr_bool(ASR::expr_t &e) {
        if (ASR::is_a<ASR::LogicalConstant_t>(e)) {
            const ASR::LogicalConstant_t *lc =
                ASR::down_cast<ASR::LogicalConstant_t>(&e);
            return emit_const_i1(lc->m_value);
        }
        if (ASR::is_a<ASR::IntegerCompare_t>(e)) {
            const ASR::IntegerCompare_t *ic =
                ASR::down_cast<ASR::IntegerCompare_t>(&e);
            int64_t pred = icmp_predicate_for(ic->m_op);
            if (pred < 0) {
                throw CodeGenError(
                    "mlir-new: integer compare operator not supported",
                    e.base.loc);
            }
            visit_expr_value(*ic->m_left);
            MLIR_ValueHandle lhs = last_value;
            visit_expr_value(*ic->m_right);
            MLIR_ValueHandle rhs = last_value;
            return emit_icmp_i32(pred, lhs, rhs);
        }
        if (ASR::is_a<ASR::LogicalBinOp_t>(e)) {
            const ASR::LogicalBinOp_t *lb =
                ASR::down_cast<ASR::LogicalBinOp_t>(&e);
            MLIR_ValueHandle lhs = visit_expr_bool(*lb->m_left);
            MLIR_ValueHandle rhs = visit_expr_bool(*lb->m_right);
            switch (lb->m_op) {
                case ASR::logicalbinopType::And:
                    return emit_binop_i1(OP_TYPE_ARITH_ANDI,
                        str_lit("arith.andi"), lhs, rhs);
                case ASR::logicalbinopType::Or:
                    return emit_binop_i1(OP_TYPE_ARITH_ORI,
                        str_lit("arith.ori"), lhs, rhs);
                default:
                    throw CodeGenError(
                        "mlir-new: logical binary operator not supported",
                        e.base.loc);
            }
        }
        throw CodeGenError(
            "mlir-new: boolean expression kind not supported yet", e.base.loc);
    }

    void store_to_target(ASR::expr_t *target) {
        if (ASR::is_a<ASR::Var_t>(*target)) {
            ASR::Variable_t *v = ASRUtils::EXPR2VAR(target);
            SymSlot &slot = slot_for_var(v);
            if (slot.is_array) {
                throw CodeGenError(
                    "mlir-new: assign whole array not supported yet",
                    target->base.loc);
            }
            emit_memref_store_scalar(last_value, slot.memref);
            return;
        }
        if (ASR::is_a<ASR::ArrayItem_t>(*target)) {
            const ASR::ArrayItem_t &ai =
                *ASR::down_cast<ASR::ArrayItem_t>(target);
            if (!ASR::is_a<ASR::Var_t>(*ai.m_v)) {
                throw CodeGenError(
                    "mlir-new: array base must be a variable for now",
                    target->base.loc);
            }
            ASR::Variable_t *v = ASRUtils::EXPR2VAR(ai.m_v);
            SymSlot &slot = slot_for_var(v);
            if (!slot.is_array || ai.n_args != 1) {
                throw CodeGenError(
                    "mlir-new: only rank-1 array indexing is supported",
                    target->base.loc);
            }
            ASR::dimension_t *dims = nullptr;
            size_t n_dims = ASRUtils::extract_dimensions_from_ttype(v->m_type, dims);
            ASR::dimension_t *dim = (n_dims > 0) ? &dims[0] : nullptr;
            MLIR_ValueHandle idx = fortran_to_zero_based_index(
                ai.m_args[0].m_right, dim);
            emit_memref_store_i32(last_value, slot.memref, idx);
            return;
        }
        throw CodeGenError("mlir-new: assignment target not supported",
            target->base.loc);
    }

    void visit_Assignment(const ASR::Assignment_t &x) {
        visit_expr_value(*x.m_value);
        store_to_target(x.m_target);
    }

    void handle_print_formatter(ASR::expr_t *x) {
        if (ASR::is_a<ASR::StringFormat_t>(*x)) {
            const ASR::StringFormat_t *sf =
                ASR::down_cast<ASR::StringFormat_t>(x);
            for (size_t i = 0; i < sf->n_args; i++) {
                visit_expr_value(*sf->m_args[i]);
                if (!ASRUtils::is_integer(
                        *ASRUtils::expr_type(sf->m_args[i]))) {
                    throw CodeGenError(
                        "mlir-new: print only supports integers in this "
                        "minimal pipeline",
                        sf->m_args[i]->base.loc);
                }
                emit_vector_print(last_value);
            }
            return;
        }
        ASR::ttype_t *t = ASRUtils::expr_type(x);
        if (ASRUtils::is_integer(*t)) {
            visit_expr_value(*x);
            emit_vector_print(last_value);
            return;
        }
        throw CodeGenError("mlir-new: print formatter not supported",
            x->base.loc);
    }

    void visit_Print(const ASR::Print_t &x) {
        if (!x.m_text) {
            throw CodeGenError("mlir-new: print with no format expression",
                x.base.base.loc);
        }
        handle_print_formatter(x.m_text);
    }

    void visit_FileWrite(const ASR::FileWrite_t &x) {
        bool default_unit = false;
        if (x.m_unit) {
            int uv = -1;
            if (ASRUtils::extract_value(x.m_unit, uv) && uv == 6) {
                default_unit = true;
            }
        } else {
            default_unit = true;
        }
        if (!default_unit) {
            throw CodeGenError(
                "mlir-new: only write to default unit (*,*) is supported",
                x.base.base.loc);
        }
        if (x.n_values != 1) {
            throw CodeGenError("mlir-new: expected a single format/value in "
                               "write(*,*) for minimal pipeline",
                x.base.base.loc);
        }
        handle_print_formatter(x.m_values[0]);
    }

    void visit_DoLoop(const ASR::DoLoop_t &x) {
        if (x.n_orelse > 0) {
            throw CodeGenError("mlir-new: do-loop else not supported",
                x.base.base.loc);
        }
        if (!x.m_head.m_v || !x.m_head.m_start || !x.m_head.m_end) {
            throw CodeGenError("mlir-new: incomplete do-loop header",
                x.base.base.loc);
        }
        if (!ASR::is_a<ASR::Var_t>(*x.m_head.m_v)) {
            throw CodeGenError("mlir-new: do-loop index must be a variable",
                x.base.base.loc);
        }
        ASR::Variable_t *loop_var = ASRUtils::EXPR2VAR(x.m_head.m_v);
        SymSlot &slot = slot_for_var(loop_var);
        if (slot.is_array) {
            throw CodeGenError("mlir-new: do-loop index must be scalar",
                x.base.base.loc);
        }

        visit_expr_value(*x.m_head.m_start);
        emit_memref_store_scalar(last_value, slot.memref);

        MLIR_BlockHandle header_b = new_cfg_block();
        MLIR_BlockHandle body_b = new_cfg_block();
        MLIR_BlockHandle step_b = new_cfg_block();
        MLIR_BlockHandle exit_b = new_cfg_block();
        emit_branch(header_b);

        cur_block = header_b;
        block_terminated = false;
        MLIR_ValueHandle iv = emit_memref_load_scalar(slot.memref);
        visit_expr_value(*x.m_head.m_end);
        MLIR_ValueHandle endv = last_value;
        MLIR_ValueHandle cond = emit_icmp_i32(
            icmp_predicate_for(ASR::cmpopType::LtE), iv, endv);
        emit_cond_branch(cond, body_b, exit_b);

        cur_block = body_b;
        block_terminated = false;
        for (size_t i = 0; i < x.n_body; i++) {
            visit_stmt(*x.m_body[i]);
            if (block_terminated) {
                break;
            }
        }
        if (!block_terminated) {
            emit_branch(step_b);
        }

        cur_block = step_b;
        block_terminated = false;
        iv = emit_memref_load_scalar(slot.memref);
        MLIR_ValueHandle stepv;
        if (x.m_head.m_increment) {
            visit_expr_value(*x.m_head.m_increment);
            stepv = last_value;
        } else {
            stepv = emit_const_i32(1);
        }
        MLIR_ValueHandle next = emit_binop_i32(OP_TYPE_ARITH_ADDI,
            str_lit("arith.addi"), iv, stepv);
        emit_memref_store_scalar(next, slot.memref);
        emit_branch(header_b);

        cur_block = exit_b;
        block_terminated = false;
    }

    void visit_WhileLoop(const ASR::WhileLoop_t &x) {
        if (x.n_orelse > 0) {
            throw CodeGenError("mlir-new: while-loop else not supported",
                x.base.base.loc);
        }
        if (!x.m_test) {
            throw CodeGenError("mlir-new: while-loop missing test expression",
                x.base.base.loc);
        }

        MLIR_BlockHandle header_b = new_cfg_block();
        MLIR_BlockHandle body_b = new_cfg_block();
        MLIR_BlockHandle exit_b = new_cfg_block();
        emit_branch(header_b);

        cur_block = header_b;
        block_terminated = false;
        MLIR_ValueHandle cond = visit_expr_bool(*x.m_test);
        emit_cond_branch(cond, body_b, exit_b);

        cur_block = body_b;
        block_terminated = false;
        for (size_t i = 0; i < x.n_body; i++) {
            visit_stmt(*x.m_body[i]);
            if (block_terminated) {
                break;
            }
        }
        if (!block_terminated) {
            emit_branch(header_b);
        }

        cur_block = exit_b;
        block_terminated = false;
    }

    void visit_stmt(ASR::stmt_t &s) {
        if (ASR::is_a<ASR::Assignment_t>(s)) {
            visit_Assignment(
                *ASR::down_cast<ASR::Assignment_t>(&s));
        } else if (ASR::is_a<ASR::Print_t>(s)) {
            visit_Print(*ASR::down_cast<ASR::Print_t>(&s));
        } else if (ASR::is_a<ASR::FileWrite_t>(s)) {
            visit_FileWrite(*ASR::down_cast<ASR::FileWrite_t>(&s));
        } else if (ASR::is_a<ASR::DoLoop_t>(s)) {
            visit_DoLoop(*ASR::down_cast<ASR::DoLoop_t>(&s));
        } else if (ASR::is_a<ASR::WhileLoop_t>(s)) {
            visit_WhileLoop(*ASR::down_cast<ASR::WhileLoop_t>(&s));
        } else {
            throw CodeGenError(
                "mlir-new: statement not supported yet", s.base.loc);
        }
    }

    void visit_Program(const ASR::Program_t &x) {
        fn_body_r = MLIR_CreateRegion(&ctx);
        cur_block = MLIR_CreateBlock(&ctx);
        MLIR_AppendRegionBlock(&ctx, fn_body_r, cur_block);
        block_terminated = false;

        for (auto &item : x.m_symtab->get_scope()) {
            if (ASR::is_a<ASR::Variable_t>(*item.second)) {
                emit_variable_storage(
                    *ASR::down_cast<ASR::Variable_t>(item.second));
            }
        }
        for (auto &item : x.m_symtab->get_scope()) {
            if (ASR::is_a<ASR::Variable_t>(*item.second)) {
                emit_variable_static_init(
                    *ASR::down_cast<ASR::Variable_t>(item.second));
            }
        }
        for (size_t i = 0; i < x.n_body; i++) {
            visit_stmt(*x.m_body[i]);
            if (block_terminated) {
                break;
            }
        }
        if (!block_terminated) {
            MLIR_ValueHandle zero = emit_const_i32(0);
            emit_func_return_i32(zero);
        }

        string main_sym = str_lit("main");
        MLIR_AttributeHandle sym_name =
            MLIR_CreateAttributeString(&ctx, str_lit("sym_name"), main_sym);
        MLIR_TypeHandle res_tys[1] = {i32_ty};
        MLIR_TypeHandle fn_ty =
            MLIR_CreateTypeFunction(&ctx, nullptr, 0, res_tys, 1);
        MLIR_AttributeHandle fn_ty_attr =
            MLIR_CreateAttributeType(&ctx, str_lit("function_type"), fn_ty);
        MLIR_AttributeHandle attrs[2] = {sym_name, fn_ty_attr};
        MLIR_RegionHandle regs[1] = {fn_body_r};
        MLIR_OpHandle fn = MLIR_CreateOp(&ctx, OP_TYPE_FUNC_FUNC,
            str_lit("func.func"), attrs, 2,
            nullptr, 0, nullptr, 0, nullptr, 0, regs, 1,
            loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
        check_mlir(fn != MLIR_INVALID_HANDLE, "func.func main");
        append_module(fn);
    }

    void visit_TranslationUnit(const ASR::TranslationUnit_t &x) {
        emit_module_skeleton();
        for (auto &item : x.m_symtab->get_scope()) {
            if (ASR::is_a<ASR::Program_t>(*item.second)) {
                visit_Program(
                    *ASR::down_cast<ASR::Program_t>(item.second));
                return;
            }
        }
        throw CodeGenError(
            "mlir-new: no program unit found (expected a single program)");
    }
};

Result<std::unique_ptr<MLIRModule>> asr_to_mlir_new(Allocator &al,
    ASR::asr_t &asr, diag::Diagnostics &diagnostics) {
    ensure_mlir_corec_platform_initialized();
    (void)al;
    if (!ASR::is_a<ASR::unit_t>(asr)) {
        diagnostics.diagnostics.push_back(diag::Diagnostic(
            "asr_to_mlir_new: expected a translation unit (asr unit)",
            diag::Level::Error, diag::Stage::CodeGen));
        Error err;
        return err;
    }
    const ASR::unit_t *u = ASR::down_cast<ASR::unit_t>(&asr);
    if (u->type != ASR::unitType::TranslationUnit) {
        diagnostics.diagnostics.push_back(diag::Diagnostic(
            "asr_to_mlir_new: expected unit type translation unit",
            diag::Level::Error, diag::Stage::CodeGen));
        Error err;
        return err;
    }
    const ASR::TranslationUnit_t *tu =
        ASR::down_cast<ASR::TranslationUnit_t>(u);

    MlirEmitter emit;
    try {
        emit.visit_TranslationUnit(*tu);
    } catch (const CodeGenError &e) {
        diagnostics.diagnostics.push_back(e.d);
        if (emit.arena) {
            arena_destroy(emit.arena);
        }
        return Error();
    }

    std::string high_mlir =
        copy_mlir_string(MLIR_PrintOperationUpstream(&emit.ctx, emit.module_op));

    bool use_upstream = use_upstream_mlir_lowering();
    bool lowered = use_upstream
        ? MLIR_LowerToLLVMDialectUpstream(&emit.ctx, emit.module_op)
        : MLIR_LowerToLLVMDialect(&emit.ctx, emit.module_op);
    if (!lowered) {
        diagnostics.diagnostics.push_back(diag::Diagnostic(
            "asr_to_mlir_new: MLIR_LowerToLLVMDialect failed "
            "(see stderr from mlir passes)",
            diag::Level::Error, diag::Stage::CodeGen));
        arena_destroy(emit.arena);
        return Error();
    }

    std::string llvm_dialect_mlir =
        copy_mlir_string(MLIR_PrintOperationUpstream(&emit.ctx, emit.module_op));

    string llvm_s = use_upstream
        ? MLIR_TranslateModuleToLLVMIRUpstream(&emit.ctx, emit.module_op)
        : MLIR_TranslateModuleToLLVMIR(&emit.ctx, emit.module_op);
    std::string llvm_ir = copy_mlir_string(llvm_s);
    if (llvm_ir.empty()) {
        diagnostics.diagnostics.push_back(diag::Diagnostic(
            "asr_to_mlir_new: MLIR_TranslateModuleToLLVMIR returned empty "
            "output",
            diag::Level::Error, diag::Stage::CodeGen));
        arena_destroy(emit.arena);
        return Error();
    }

    arena_destroy(emit.arena);
    return Result<std::unique_ptr<MLIRModule>>(std::unique_ptr<MLIRModule>(
        new MLIRModule(std::move(high_mlir), std::move(llvm_dialect_mlir),
            std::move(llvm_ir))));
}

} // namespace LCompilers
