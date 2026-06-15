#include "mlir_new_backend.h"

// Prefixed upstream core API (compiled directly with the rename header forced in by src/mlir/CMakeLists.txt).
MLIR_OpHandle MLIR_Upstream_CreateOp(
        MLIR_Context *ctx, MLIR_OpType type, string opname,
        MLIR_AttributeHandle *attrs, size_t n_attrs,
        MLIR_TypeHandle *result_types, size_t n_result_types,
        MLIR_ValueHandle *results, size_t n_results,
        MLIR_ValueHandle *operands, size_t n_operands,
        MLIR_RegionHandle *regions, size_t n_regions,
        MLIR_LocationHandle location, MLIR_LocationHandle unnumbered_loc_def,
        string trailing_comment, int64_t source_line_start);
MLIR_OpHandle MLIR_Upstream_CreateOpWithSuccessors(
        MLIR_Context *ctx, MLIR_OpType type, string opname,
        MLIR_AttributeHandle *attrs, size_t n_attrs,
        MLIR_TypeHandle *result_types, size_t n_result_types,
        MLIR_ValueHandle *results, size_t n_results,
        MLIR_ValueHandle *operands, size_t n_operands,
        MLIR_RegionHandle *regions, size_t n_regions,
        MLIR_BlockHandle *successors, size_t n_successors,
        MLIR_ValueHandle **succ_operands, size_t *n_succ_operands,
        MLIR_LocationHandle location, MLIR_LocationHandle unnumbered_loc_def,
        string trailing_comment, int64_t source_line_start);
void MLIR_Upstream_SetArenaAllocator(MLIR_Context *ctx, Arena *arena);
MLIR_LocationHandle MLIR_Upstream_CreateLocationUnknown(MLIR_Context *ctx,
        string loc);
MLIR_TypeHandle MLIR_Upstream_CreateTypeInteger(MLIR_Context *ctx,
        uint32_t width, bool is_signed);
MLIR_TypeHandle MLIR_Upstream_CreateTypeLLVMPointer(MLIR_Context *ctx);
MLIR_TypeHandle MLIR_Upstream_CreateTypeLLVMVoid(MLIR_Context *ctx);
MLIR_TypeHandle MLIR_Upstream_CreateTypeLLVMFunction(MLIR_Context *ctx,
        MLIR_TypeHandle result, const MLIR_TypeHandle *inputs,
        size_t n_inputs, bool is_vararg);
MLIR_TypeHandle MLIR_Upstream_CreateTypeLLVMArray(MLIR_Context *ctx,
        MLIR_TypeHandle elem, uint64_t count);
MLIR_RegionHandle MLIR_Upstream_CreateRegion(MLIR_Context *ctx);
MLIR_BlockHandle MLIR_Upstream_CreateBlock(MLIR_Context *ctx);
void MLIR_Upstream_AppendRegionBlock(MLIR_Context *ctx, MLIR_RegionHandle region,
        MLIR_BlockHandle block);
void MLIR_Upstream_AppendBlockOp(MLIR_Context *ctx, MLIR_BlockHandle block,
        MLIR_OpHandle op);
MLIR_ValueHandle MLIR_Upstream_CreateValueOpResult(MLIR_Context *ctx,
        MLIR_OpHandle owner, uint32_t result_index, MLIR_TypeHandle type,
        string name, MLIR_LocationHandle loc);
MLIR_AttributeHandle MLIR_Upstream_CreateAttributeInteger(MLIR_Context *ctx,
        string name, int64_t value, MLIR_TypeHandle type);
MLIR_AttributeHandle MLIR_Upstream_CreateAttributeType(MLIR_Context *ctx,
        string name, MLIR_TypeHandle type);
MLIR_AttributeHandle MLIR_Upstream_CreateAttributeDenseI32Array(
        MLIR_Context *ctx, string name, const int32_t *values, size_t n_values);
MLIR_AttributeHandle MLIR_Upstream_CreateAttributeString(MLIR_Context *ctx,
        string name, string value);
MLIR_AttributeHandle MLIR_Upstream_CreateAttributeSymbolRef(MLIR_Context *ctx,
        string name, string symbol);
MLIR_OpHandle MLIR_Upstream_CreateLLVMGlobalString(MLIR_Context *ctx,
        string sym_name, string bytes, MLIR_LocationHandle loc);
string MLIR_Upstream_PrintOperationUpstream(MLIR_Context *ctx, MLIR_OpHandle h);

const MlirNewApi lfortran_mlir_new_native_api = {
    .SetArenaAllocator = MLIR_SetArenaAllocator,
    .CreateLocationUnknown = MLIR_CreateLocationUnknown,
    .CreateTypeInteger = MLIR_CreateTypeInteger,
    .CreateTypeLLVMPointer = MLIR_CreateTypeLLVMPointer,
    .CreateTypeLLVMVoid = MLIR_CreateTypeLLVMVoid,
    .CreateTypeLLVMFunction = MLIR_CreateTypeLLVMFunction,
    .CreateTypeLLVMArray = MLIR_CreateTypeLLVMArray,
    .CreateRegion = MLIR_CreateRegion,
    .CreateBlock = MLIR_CreateBlock,
    .AppendRegionBlock = MLIR_AppendRegionBlock,
    .CreateOp = MLIR_CreateOp,
    .CreateOpWithSuccessors = MLIR_CreateOpWithSuccessors,
    .AppendBlockOp = MLIR_AppendBlockOp,
    .CreateValueOpResult = MLIR_CreateValueOpResult,
    .CreateAttributeInteger = MLIR_CreateAttributeInteger,
    .CreateAttributeType = MLIR_CreateAttributeType,
    .CreateAttributeDenseI32Array = MLIR_CreateAttributeDenseI32Array,
    .CreateAttributeString = MLIR_CreateAttributeString,
    .CreateAttributeSymbolRef = MLIR_CreateAttributeSymbolRef,
    .CreateLLVMGlobalString = MLIR_CreateLLVMGlobalString,
    .PrintModule = MLIR_PrintOperationGeneric,
    .TranslateToLLVMIR = MLIR_TranslateModuleToLLVMIR,
};

const MlirNewApi lfortran_mlir_new_upstream_api = {
    .SetArenaAllocator = MLIR_Upstream_SetArenaAllocator,
    .CreateLocationUnknown = MLIR_Upstream_CreateLocationUnknown,
    .CreateTypeInteger = MLIR_Upstream_CreateTypeInteger,
    .CreateTypeLLVMPointer = MLIR_Upstream_CreateTypeLLVMPointer,
    .CreateTypeLLVMVoid = MLIR_Upstream_CreateTypeLLVMVoid,
    .CreateTypeLLVMFunction = MLIR_Upstream_CreateTypeLLVMFunction,
    .CreateTypeLLVMArray = MLIR_Upstream_CreateTypeLLVMArray,
    .CreateRegion = MLIR_Upstream_CreateRegion,
    .CreateBlock = MLIR_Upstream_CreateBlock,
    .AppendRegionBlock = MLIR_Upstream_AppendRegionBlock,
    .CreateOp = MLIR_Upstream_CreateOp,
    .CreateOpWithSuccessors = MLIR_Upstream_CreateOpWithSuccessors,
    .AppendBlockOp = MLIR_Upstream_AppendBlockOp,
    .CreateValueOpResult = MLIR_Upstream_CreateValueOpResult,
    .CreateAttributeInteger = MLIR_Upstream_CreateAttributeInteger,
    .CreateAttributeType = MLIR_Upstream_CreateAttributeType,
    .CreateAttributeDenseI32Array = MLIR_Upstream_CreateAttributeDenseI32Array,
    .CreateAttributeString = MLIR_Upstream_CreateAttributeString,
    .CreateAttributeSymbolRef = MLIR_Upstream_CreateAttributeSymbolRef,
    .CreateLLVMGlobalString = MLIR_Upstream_CreateLLVMGlobalString,
    .PrintModule = MLIR_Upstream_PrintOperationUpstream,
    .TranslateToLLVMIR = MLIR_TranslateModuleToLLVMIRUpstream,
};

const MlirNewApi *mlir_new_api_for(MlirNewBackendKind backend) {
    switch (backend) {
        case MLIR_NEW_BACKEND_UPSTREAM:
            return &lfortran_mlir_new_upstream_api;
        case MLIR_NEW_BACKEND_NATIVE:
        default:
            return &lfortran_mlir_new_native_api;
    }
}
