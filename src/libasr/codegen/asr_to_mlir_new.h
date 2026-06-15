#ifndef LFORTRAN_ASR_TO_MLIR_NEW_H
#define LFORTRAN_ASR_TO_MLIR_NEW_H

#include <libasr/asr.h>
#include <libasr/asr_utils.h>
#include <libasr/codegen/evaluator.h>

extern "C" {
#include <mlir_new_backend.h>
}

namespace LCompilers {

    // Final ASR -> LLVM dialect MLIR (direct emission) -> LLVM IR.
    // `backend` selects native (mlir_api_impl.c) vs upstream core API.
    Result<std::unique_ptr<MLIRModule>> asr_to_mlir_new(Allocator &al,
        ASR::asr_t &asr, diag::Diagnostics &diagnostics,
        MlirNewBackendKind backend);

} // namespace LCompilers

#endif // LFORTRAN_ASR_TO_MLIR_NEW_H
