#ifndef MLIR_LFORTRAN_NEW_BACKEND_H
#define MLIR_LFORTRAN_NEW_BACKEND_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum MlirNewBackendKind {
    MLIR_NEW_BACKEND_NATIVE = 0,
    MLIR_NEW_BACKEND_UPSTREAM = 1,
} MlirNewBackendKind;

#ifdef __cplusplus
}
#endif

// C++ headers only need MlirNewBackendKind (avoid corec's array_size macro via
// mlir_api.h). Translation units that need MlirNewApi define this before include.
#if !defined(__cplusplus) || defined(MLIR_NEW_BACKEND_INCLUDE_API)

#include <stddef.h>
#include <stdint.h>

#include <base/arena.h>
#include <base/string.h>
#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creation / mutation API used by LlvmEmitter. Print and translate are
// backend-specific pipeline entry points (native C vs upstream MLIR).
typedef struct MlirNewApi {
    void (*SetArenaAllocator)(MLIR_Context *ctx, Arena *arena);
    MLIR_LocationHandle (*CreateLocationUnknown)(MLIR_Context *ctx, string loc);
    MLIR_TypeHandle (*CreateTypeInteger)(MLIR_Context *ctx, uint32_t width,
                                         bool is_signed);
    MLIR_TypeHandle (*CreateTypeLLVMPointer)(MLIR_Context *ctx);
    MLIR_TypeHandle (*CreateTypeLLVMVoid)(MLIR_Context *ctx);
    MLIR_TypeHandle (*CreateTypeLLVMFunction)(MLIR_Context *ctx,
            MLIR_TypeHandle result, const MLIR_TypeHandle *inputs,
            size_t n_inputs, bool is_vararg);
    MLIR_TypeHandle (*CreateTypeLLVMArray)(MLIR_Context *ctx,
            MLIR_TypeHandle elem, uint64_t count);
    MLIR_RegionHandle (*CreateRegion)(MLIR_Context *ctx);
    MLIR_BlockHandle (*CreateBlock)(MLIR_Context *ctx);
    void (*AppendRegionBlock)(MLIR_Context *ctx, MLIR_RegionHandle region,
                              MLIR_BlockHandle block);
    MLIR_OpHandle (*CreateOp)(MLIR_Context *ctx, MLIR_OpType type, string opname,
            MLIR_AttributeHandle *attrs, size_t n_attrs,
            MLIR_TypeHandle *result_types, size_t n_result_types,
            MLIR_ValueHandle *results, size_t n_results,
            MLIR_ValueHandle *operands, size_t n_operands,
            MLIR_RegionHandle *regions, size_t n_regions,
            MLIR_LocationHandle location, MLIR_LocationHandle unnumbered_loc_def,
            string trailing_comment, int64_t source_line_start);
    MLIR_OpHandle (*CreateOpWithSuccessors)(MLIR_Context *ctx, MLIR_OpType type,
            string opname, MLIR_AttributeHandle *attrs, size_t n_attrs,
            MLIR_TypeHandle *result_types, size_t n_result_types,
            MLIR_ValueHandle *results, size_t n_results,
            MLIR_ValueHandle *operands, size_t n_operands,
            MLIR_RegionHandle *regions, size_t n_regions,
            MLIR_BlockHandle *successors, size_t n_successors,
            MLIR_ValueHandle **succ_operands, size_t *n_succ_operands,
            MLIR_LocationHandle location, MLIR_LocationHandle unnumbered_loc_def,
            string trailing_comment, int64_t source_line_start);
    void (*AppendBlockOp)(MLIR_Context *ctx, MLIR_BlockHandle block,
                          MLIR_OpHandle op);
    MLIR_ValueHandle (*CreateValueOpResult)(MLIR_Context *ctx, MLIR_OpHandle owner,
            uint32_t result_index, MLIR_TypeHandle type, string name,
            MLIR_LocationHandle loc);
    MLIR_AttributeHandle (*CreateAttributeInteger)(MLIR_Context *ctx, string name,
            int64_t value, MLIR_TypeHandle type);
    MLIR_AttributeHandle (*CreateAttributeType)(MLIR_Context *ctx, string name,
            MLIR_TypeHandle type);
    MLIR_AttributeHandle (*CreateAttributeDenseI32Array)(MLIR_Context *ctx,
            string name, const int32_t *values, size_t n_values);
    MLIR_AttributeHandle (*CreateAttributeString)(MLIR_Context *ctx, string name,
            string value);
    MLIR_AttributeHandle (*CreateAttributeSymbolRef)(MLIR_Context *ctx, string name,
            string symbol);
    MLIR_OpHandle (*CreateLLVMGlobalString)(MLIR_Context *ctx, string sym_name,
            string bytes, MLIR_LocationHandle loc);
    string (*PrintModule)(MLIR_Context *ctx, MLIR_OpHandle module);
    string (*TranslateToLLVMIR)(MLIR_Context *ctx, MLIR_OpHandle module);
} MlirNewApi;

extern const MlirNewApi lfortran_mlir_new_native_api;
extern const MlirNewApi lfortran_mlir_new_upstream_api;

const MlirNewApi *mlir_new_api_for(MlirNewBackendKind backend);

#ifdef __cplusplus
}
#endif

#endif /* !__cplusplus || MLIR_NEW_BACKEND_INCLUDE_API */

#endif
