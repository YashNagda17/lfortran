#ifndef LFORTRAN_ASR_TO_MLIR_NEW_H
#define LFORTRAN_ASR_TO_MLIR_NEW_H

#include <libasr/asr.h>
#include <libasr/asr_utils.h>
#include <libasr/codegen/evaluator.h>

namespace LCompilers {

    Result<std::unique_ptr<MLIRModule>> asr_to_mlir_new(Allocator &al,
        ASR::asr_t &asr, diag::Diagnostics &diagnostics);

} // namespace LCompilers

#endif // LFORTRAN_ASR_TO_MLIR_NEW_H
