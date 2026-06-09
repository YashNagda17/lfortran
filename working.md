# `mlir-new` Backend Working Notes

This document describes the current uncommitted git diff for the new MLIR
backend in this checkout.  The command-line name implemented by the diff is:

```console
lfortran --backend=mlir-new ...
```

The user-facing inspection flags added by the diff are:

```console
lfortran --backend=mlir-new --show-mlir-asr-dialect file.f90
lfortran --backend=mlir-new --show-mlir-high-dialect file.f90
lfortran --backend=mlir-new --show-mlir-llvm-dialect file.f90
lfortran --backend=mlir-new --show-llvm-from-mlir file.f90
lfortran --backend=mlir-new -c file.f90
```

The notes below are based on the current `git diff` in
`/home/yash/Desktop/GSOC/lfortran_mlir/3_lfortran/lfortran`.

## Scope of the Diff

The diff adds a complete experimental pipeline that starts from LFortran ASR and
passes through several inspectable stages:

```text
Fortran source
  -> LFortran AST
  -> LFortran ASR
  -> generated/native ASR dialect module
  -> high-level MLIR using func/memref/arith/cf/vector
  -> LLVM dialect MLIR
  -> textual LLVM IR
  -> parsed LLVM module / object file path
```

The core design is intentionally prototype-shaped:

- `ASR.asdl` remains the source of truth for the dialect schema.
- Python generation creates the repetitive C/C++ API wrappers, accessors,
  schema tables, pretty-print policies, and lowering dispatch.
- Handwritten C++ converts live ASR nodes into the ASR dialect.
- Handwritten C verifies, prints, and lowers the ASR dialect to high-level MLIR.
- Handwritten native lowering handles the subset needed by the current examples.
- An upstream MLIR lowering path remains available for comparison through
  `USE_MLIR_Upstream=1`.

The diff is large because it introduces both the compiler driver plumbing and a
new MLIR support layer.  Current diff stat:

```text
48 files changed, 34842 insertions(+), 26 deletions(-)
```

## Repository Notes Read

The requested `../.AGENTS.md` and `../.claude.md` files were not present relative
to `3_lfortran`.  Inside the actual working repository, `lfortran/AGENTS.md`
exists and `lfortran/CLAUDE.md` points to the same guidance.  The relevant
constraints for this work are:

- keep `libasr` boundaries clean;
- use existing LFortran style and C++17;
- avoid broad unrelated refactors;
- route new backend behavior through the existing command-line, evaluator, and
  codegen boundaries.

## Diff Inventory

| Status | File | Lines | Role in `mlir-new` |
| --- | --- | ---: | --- |
| M | `CMakeLists.txt` | changed | top-level MLIR/LLVM link selection and optional wasm lowering components | -> approved
| A | `cmake/LFortranMLIRLink.cmake` | 50 | gathers static MLIR libraries safely for the new MLIR C API shim |
| M | `src/CMakeLists.txt` | changed | adds `src/mlir` subdirectory when LLVM and MLIR are enabled |
| M | `src/bin/lfortran.cpp` | changed | adds `mlir-new` backend enum and routes show/compile modes |
| M | `src/bin/lfortran_command_line_parser.cpp` | changed | adds stage-specific MLIR show flags and help text |
| M | `src/bin/lfortran_command_line_parser.h` | changed | stores the new show-flag booleans |
| M | `src/lfortran/fortran_evaluator.cpp` | changed | evaluates ASR through `asr_to_mlir_new` |
| M | `src/lfortran/fortran_evaluator.h` | changed | declares `get_mlir_new` |
| M | `src/libasr/CMakeLists.txt` | changed | compiles and links the new ASR-to-MLIR sources |
| A | `src/libasr/asdl_to_asr_dialect.py` | 1520 | generates ASR dialect schema/API/accessor/lowering/print files |
| A | `src/libasr/codegen/asr_to_asr_dialect.cpp` | 2598 | emits ASR dialect operations from LFortran ASR nodes |
| A | `src/libasr/codegen/asr_to_asr_dialect.h` | 287 | declares the ASR dialect visitor |
| A | `src/libasr/codegen/asr_to_mlir_new.cpp` | 254 | orchestrates the end-to-end `mlir-new` pipeline |
| A | `src/libasr/codegen/asr_to_mlir_new.h` | 53 | declares stage requests and public pipeline entrypoints |
| M | `src/libasr/codegen/evaluator.cpp` | changed | stores string-backed MLIR/LLVM snapshots and parses LLVM IR text |
| M | `src/libasr/codegen/evaluator.h` | changed | extends `MLIRModule` with new stage strings |
| A | `src/mlir/CMakeLists.txt` | 243 | builds the C MLIR API, dialect runtime, and generated dialect files |
| A | `src/mlir/generated/asr_dialect_accessors.h` | 5769 | generated field accessors for every ASR dialect op |
| A | `src/mlir/generated/asr_dialect_api_generated.h` | 5558 | generated `ASR_Create*Op` wrapper API |
| A | `src/mlir/generated/asr_dialect_enum_print.h` | 365 | generated ASR enum keyword printers |
| A | `src/mlir/generated/asr_dialect_ids.h` | 928 | generated op and field ids |
| A | `src/mlir/generated/asr_dialect_layout_fields.h` | 12 | generated synthetic layout field ids |
| A | `src/mlir/generated/asr_dialect_lowering_dispatch.h` | 74 | generated switch from op kind to lowering handler |
| A | `src/mlir/generated/asr_dialect_print_policy.h` | 213 | generated pretty-print policy |
| A | `src/mlir/generated/asr_dialect_schema.h` | 4423 | generated ASR dialect schema table and name lookup |
| A | `src/mlir/generated/asr_dialect_storage_merged.inc` | 926 | generated merged field storage policy |
| A | `src/mlir/lfortran/asr_dialect_api.c` | 104 | public C API dispatch layer |
| A | `src/mlir/lfortran/asr_dialect_api.h` | 158 | public ASR dialect C API |
| A | `src/mlir/lfortran/asr_dialect_api_native.c` | 1147 | native op creation, verification, diagnostics, lowering driver |
| D | `src/mlir/lfortran/asr_dialect_format.h` | 0 | removed empty placeholder; replaced by concrete printer implementation |
| A | `src/mlir/lfortran/asr_dialect_lowering_handlers.c` | 909 | ASR dialect to high-level MLIR lowering handlers |
| A | `src/mlir/lfortran/asr_dialect_pretty_print.c` | 848 | ASR dialect dump printer and diagnostic summaries |
| A | `src/mlir/lfortran/asr_dialect_storage.h` | 276 | helpers for reading stored ASR dialect fields |
| A | `src/mlir/lfortran/asr_dialect_storage_policy.h` | 113 | handwritten storage overrides and region layout policy |
| A | `src/mlir/lfortran/corec/base/buddy.c` | 536 | hosted buddy allocator support for corec/MLIR API |
| A | `src/mlir/lfortran/corec/base/string.c` | 94 | hosted string helper implementation |
| A | `src/mlir/lfortran/corec/base/string.h` | 41 | string helper declarations |
| A | `src/mlir/lfortran/corec/platform/platform.h` | 223 | platform abstraction declarations |
| A | `src/mlir/lfortran/corec/platform/platform_linux.c` | 503 | Linux hosted platform implementation |
| A | `src/mlir/lfortran/corec/platform/platform_macos.c` | 364 | macOS hosted platform implementation |
| A | `src/mlir/lfortran/corec/platform/platform_windows.c` | 639 | Windows hosted platform implementation |
| A | `src/mlir/lfortran/mlir_api_impl_upstream.cpp` | 2534 | C API bridge backed by upstream MLIR C++ APIs |
| A | `src/mlir/lfortran/mlir_lower_lfortran_hooks.c` | 495 | LFortran-specific native lowering hooks |
| A | `src/mlir/lfortran/mlir_lower_lfortran_hooks.h` | 33 | hook declarations |
| A | `src/mlir/lfortran/mlir_lower_to_llvm.c` | 791 | native high-MLIR to LLVM-dialect lowering |
| A | `src/mlir/lfortran/mlir_translate_lfortran_hooks.c` | 38 | LLVM IR translator hook helpers |
| A | `src/mlir/lfortran/mlir_translate_lfortran_hooks.h` | 27 | translator hook declarations |
| A | `src/mlir/lfortran/mlir_translate_to_llvm_ir.c` | 1468 | native LLVM dialect to LLVM IR text translator |

## High-Level Flow From Source to Execution

This section is the shortest complete path from a user's command to executed or
linked code. It names the code blocks involved at each translation boundary.

### Stage 0: User Selects a Journey

The user chooses one of these journeys:

```console
# inspect ASR dialect
lfortran --backend=mlir-new --show-mlir-asr-dialect file.f90

# inspect high-level MLIR
lfortran --backend=mlir-new --show-mlir-high-dialect file.f90

# inspect LLVM dialect MLIR
lfortran --backend=mlir-new --show-mlir-llvm-dialect file.f90

# inspect LLVM IR translated from MLIR
lfortran --backend=mlir-new --show-llvm-from-mlir file.f90

# compile object
lfortran --backend=mlir-new -c file.f90

# compile and link executable
lfortran --backend=mlir-new file.f90
```

Driver code involved:

```text
src/bin/lfortran_command_line_parser.cpp
  lines 302-313: registers the new --show-mlir-* flags
  line 333: documents mlir-new in backend help

src/bin/lfortran_command_line_parser.h
  lines 47-49: stores stage-specific show booleans

src/bin/lfortran.cpp
  lines 98-114: collects show options
  line 120: adds Backend::mlir_new
  lines 2702-2705: parses --backend=mlir-new
  lines 2828-2834: sends all MLIR dump flags to handle_mlir
  lines 2909-2919: sends -c backend=mlir-new to handle_mlir
  lines 2980-2990: sends multi-file mlir-new compiles to handle_mlir
```

### Stage 1: Source to ASR

All user journeys first go through the normal LFortran frontend path. The new
backend does not replace parsing, semantic analysis, or ASR construction.

The driver code block is `handle_mlir` in `src/bin/lfortran.cpp`:

```cpp
// Src -> AST -> ASR
LCompilers::Result<LCompilers::ASR::TranslationUnit_t*>
    result = fe.get_asr2(input, lm, diagnostics);
```

Code grounding:

- `src/bin/lfortran.cpp` line 1094 defines `handle_mlir`.
- Lines 1100-1119 read the file, create a `FortranEvaluator`, create a
  `LocationManager`, and call `fe.get_asr2`.
- The ASR result is a normal `ASR::TranslationUnit_t *`.

Why this matters:

Any parse or semantic error is still owned by the existing frontend. The
`mlir-new` path begins only after `get_asr2` succeeds.

### Stage 2: CLI Flags Become `MlirNewRequest`

After ASR is available, `handle_mlir` decides whether this is the old MLIR path
or the new one:

```cpp
std::optional<LCompilers::MlirNewRequest> mlir_new_req =
    LCompilers::get_mlir_new_request(backend, show);

LCompilers::Result<std::unique_ptr<LCompilers::MLIRModule>> res =
    mlir_new_req
        ? fe.get_mlir_new(*(LCompilers::ASR::asr_t *)asr, diagnostics,
            *mlir_new_req)
        : fe.get_mlir(*(LCompilers::ASR::asr_t *)asr, diagnostics);
```

Code grounding:

- `src/bin/lfortran.cpp` lines 1127-1133 perform this selection.
- `src/libasr/codegen/asr_to_mlir_new.cpp` lines 98-123 build the request.

The target mapping is:

| User input | Target enum | Dump-only? | Next code path |
| --- | --- | --- | --- |
| `--show-mlir-asr-dialect` | `AsrDialect` | yes | stop after ASR dialect print |
| `--show-mlir-high-dialect` | `HighMlir` | yes | stop after ASR dialect to high MLIR |
| `--show-mlir` with `mlir-new` | `HighMlir` | yes | compatibility high-MLIR dump |
| `--show-mlir-llvm-dialect` | `LlvmDialect` | yes | stop after LLVM dialect lowering |
| `--show-llvm-from-mlir` | `LlvmIr` | yes | stop after LLVM IR translation and parse |
| no show flag, `--backend=mlir-new` | `ObjectFile` | no | produce object/executable path |

The request builder also reads:

```cpp
const char *upstream = std::getenv("USE_MLIR_Upstream");
request.use_upstream_lowering = upstream && upstream[0] == '1';
```

So this command switches the lower/translate part to the upstream comparison
path:

```console
USE_MLIR_Upstream=1 lfortran --backend=mlir-new --show-mlir-llvm-dialect file.f90
```

### Stage 3: Evaluator Enters the New Backend

The evaluator entrypoint is `FortranEvaluator::get_mlir_new`:

```cpp
Result<std::unique_ptr<MLIRModule>> FortranEvaluator::get_mlir_new(
        ASR::asr_t &asr, diag::Diagnostics &diagnostics,
        const MlirNewRequest &request) {
    // Initial ASR only: no default ASR passes before mlir-new lowering.
    Result<std::unique_ptr<MLIRModule>> res = asr_to_mlir_new(al,
        (ASR::asr_t &)asr, diagnostics, request);

    if (request.target == MlirNewPipelineTarget::ObjectFile
            || request.target == MlirNewPipelineTarget::LlvmIr) {
        m->mlir_to_llvm(*m->llvm_ctx);
    }
    return m;
}
```

Code grounding:

- `src/lfortran/fortran_evaluator.cpp` line 625 defines `get_mlir_new`.
- Lines 637-638 call `asr_to_mlir_new`.
- Lines 646-648 parse generated LLVM IR text into an LLVM module for object and
  LLVM-IR journeys.

Why no default ASR passes:

This path intentionally starts from the initial ASR. The comment in
`get_mlir_new` says:

```cpp
// Initial ASR only: no default ASR passes before mlir-new lowering.
```

That keeps this backend focused on "can we represent and lower ASR itself?"
rather than depending on a pre-lowered or heavily rewritten ASR shape.

### Stage 4: ASR to ASR Dialect

The main orchestration function is `asr_to_mlir_new`:

```cpp
Result<std::unique_ptr<MLIRModule>> asr_to_mlir_new(Allocator &al,
    ASR::asr_t &asr, diag::Diagnostics &diagnostics,
    const MlirNewRequest &request)
```

Code grounding:

- `src/libasr/codegen/asr_to_mlir_new.cpp` line 145 defines it.
- Lines 151-160 reject non-translation-unit ASR.
- Lines 163-172 create `ASRToAsrDialectVisitor` and visit the translation unit.

The visitor code block is:

```text
src/libasr/codegen/asr_to_asr_dialect.cpp
  visit_TranslationUnit: lines 2575-2596
  visit_Program:         lines 2508-2573
  visit_Function:        lines 2393-2506
  visit_Var:             lines 2379-2391
  visit_Assignment:      lines 1765-1775
  visit_IntegerBinOp:    lines 1019-1028
  visit_IntegerCompare:  lines 1048-1057
  visit_IntegerConstant: lines 1059-1065
  visit_DoLoop:          lines 457-489
  visit_If:              lines 457-489
  visit_Return:          lines 2267-2271
```

What this stage creates:

```text
ASR::TranslationUnit_t
  -> ASR_CreateTranslationUnitOp
ASR::Program_t
  -> ASR_CreateProgramOp
ASR::Variable_t
  -> ASR_CreateVariableOp
ASR::Assignment_t
  -> ASR_CreateAssignmentOp
ASR::IntegerBinOp_t
  -> ASR_CreateIntegerBinOpOp
ASR::Var_t
  -> ASR_CreateVarOp
```

The generated wrapper code is in:

```text
src/mlir/generated/asr_dialect_api_generated.h
  ASR_CreateIntegerBinOpOp: line 1247
  ASR_CreateIntegerConstantOp: line 1339
  ASR_CreateVarOp: line 2910
  ASR_CreateAssignmentOp: line 3197
  ASR_CreateDoLoopOp: line 3387
  ASR_CreateIfOp: line 4083
  ASR_CreatePrintOp: line 4263
  ASR_CreateProgramOp: line 4862
  ASR_CreateVariableOp: line 5090
  ASR_CreateTranslationUnitOp: line 5543
```

The runtime operation creation code is:

```text
src/mlir/lfortran/asr_dialect_api_native.c
  ASR_DialectCreateOpNative: line 233
```

Why this stage exists:

It converts C++ ASR nodes into an MLIR-side tree with fields, regions,
attributes, symbol refs, and source-location attributes. After this point, later
code does not need C++ ASR node pointers.

### Stage 5: ASR Dialect Verification and Printing

`asr_to_mlir_new` verifies the ASR dialect before lowering:

```cpp
if (!ASR_DialectVerify(&emit.ctx, emit.module_op)) {
    if (target == MlirNewPipelineTarget::AsrDialect) {
        // warning, continue for dump
    } else {
        // error, stop compile/lowering
    }
}
```

Code grounding:

- `src/libasr/codegen/asr_to_mlir_new.cpp` lines 176-190 verify and decide
  whether to continue.
- `src/mlir/lfortran/asr_dialect_api_native.c` lines 793-882 structurally verify
  the operation tree.
- `src/mlir/lfortran/asr_dialect_api_native.c` lines 884-1027 verify symbol sets
  and `Var` references.
- `src/mlir/lfortran/asr_dialect_api_native.c` lines 1029-1059 verify
  `Program` and `Function` region layout.
- `src/mlir/lfortran/asr_dialect_pretty_print.c` line 822 defines
  `ASR_DialectPrintPretty`.

Why dumps can continue with verification warnings:

If the user asked for `--show-mlir-asr-dialect`, the best debugging artifact is
often the partially invalid dialect dump itself. For compile/lowering targets,
the same verification failure is fatal because later stages should not process a
known-invalid dialect tree.

### Stage 6: ASR Dialect to High-Level MLIR

The lowering call is:

```cpp
ASR_DialectOptions opts{};
opts.verify_asr_dialect = true;
opts.allow_unimplemented_nodes = false;
if (!ASR_DialectLowerToHighMLIR(&emit.ctx, emit.module_op, &opts)) {
    // report lowering diagnostic
}
```

Code grounding:

- `src/libasr/codegen/asr_to_mlir_new.cpp` lines 204-213 call this lowering.
- `src/mlir/lfortran/asr_dialect_api_native.c` lines 1135-1147 verify then
  dispatch to native lowering.
- `src/mlir/lfortran/asr_dialect_lowering_handlers.c` line 857 defines
  `ASR_DialectLowerModuleNative`.
- `src/mlir/lfortran/asr_dialect_lowering_handlers.c` line 585 defines
  `ASR_LowerTranslationUnit`.
- `src/mlir/lfortran/asr_dialect_lowering_handlers.c` line 553 defines
  `ASR_LowerProgram`.
- `src/mlir/lfortran/asr_dialect_lowering_handlers.c` line 379 defines
  `ASR_LowerAssignment`.
- `src/mlir/lfortran/asr_dialect_lowering_handlers.c` line 741 defines
  `ASR_LowerDoLoop`.
- `src/mlir/lfortran/asr_dialect_lowering_handlers.c` line 807 defines
  `ASR_LowerIf`.
- `src/mlir/lfortran/asr_dialect_lowering_handlers.c` line 847 defines
  `ASR_LowerErrorStop`.

What this stage emits:

```text
ASR dialect program
  -> func.func @main() -> i32

ASR variable
  -> memref.alloca

ASR assignment
  -> arith.constant / expression ops
  -> memref.store

ASR var expression
  -> memref.load

ASR print
  -> vector.print

ASR do loop
  -> cf.br / cf.cond_br CFG blocks

ASR if
  -> cf.cond_br plus then/else/merge blocks
```

### Stage 7: High-Level MLIR to LLVM Dialect

The pipeline chooses native or upstream lowering:

```cpp
bool lowered = request.use_upstream_lowering
    ? MLIR_LowerToLLVMDialectUpstream(&emit.ctx, emit.module_op)
    : MLIR_LowerToLLVMDialect(&emit.ctx, emit.module_op);
```

Code grounding:

- `src/libasr/codegen/asr_to_mlir_new.cpp` lines 224-233 make this choice.
- `src/mlir/lfortran/mlir_lower_to_llvm.c` line 753 defines
  `MLIR_LowerToLLVMDialect`.
- `src/mlir/lfortran/mlir_lower_to_llvm.c` line 649 defines `try_lower_op`.
- `src/mlir/lfortran/mlir_lower_to_llvm.c` line 665 calls
  `MLIR_LFortranTryLowerOp` before generic lowering.
- `src/mlir/lfortran/mlir_lower_lfortran_hooks.c` line 479 defines
  `MLIR_LFortranTryLowerOp`.

What this stage lowers:

```text
func.func       -> llvm.func
func.return     -> llvm.return
arith.constant  -> llvm.mlir.constant
cf.br           -> llvm.br
cf.cond_br      -> llvm.cond_br
memref.alloca   -> llvm.alloca      through LFortran hook
memref.load     -> llvm.load        through LFortran hook
memref.store    -> llvm.store       through LFortran hook
vector.print    -> llvm.call printf through LFortran hook
```

Why LFortran hooks run first:

Generic lowering does not know this prototype's simplified memref and print
conventions. The hook layer rewrites those operations before the generic lowering
dispatch handles the rest.

### Stage 8: LLVM Dialect to LLVM IR Text

The translation call is:

```cpp
string llvm_s = request.use_upstream_lowering
    ? MLIR_TranslateModuleToLLVMIRUpstream(&emit.ctx, emit.module_op)
    : MLIR_TranslateModuleToLLVMIR(&emit.ctx, emit.module_op);
```

Code grounding:

- `src/libasr/codegen/asr_to_mlir_new.cpp` lines 244-251 translate to LLVM IR.
- `src/mlir/lfortran/mlir_translate_to_llvm_ir.c` line 1396 defines
  `MLIR_TranslateModuleToLLVMIR`.
- `src/mlir/lfortran/mlir_translate_to_llvm_ir.c` line 732 emits individual
  operations.
- `src/mlir/lfortran/mlir_translate_to_llvm_ir.c` line 1052 emits functions.
- `src/mlir/lfortran/mlir_translate_to_llvm_ir.c` line 1145 emits globals.
- `src/mlir/lfortran/mlir_translate_lfortran_hooks.c` lines 12-34 print `index`
  types as `i64`.

What this stage emits:

```llvm
declare i32 @printf(ptr, ...)

define i32 @main() {
entry:
  ...
  ret i32 0
}
```

Why this text is still parsed:

The object-file path expects an LLVM module. The new C-side translator returns
text, so `MLIRModule::mlir_to_llvm()` parses the text back into LLVM IR objects.

### Stage 9: LLVM IR Text to Object or Execution

For object and executable journeys, `FortranEvaluator::get_mlir_new` calls:

```cpp
m->mlir_to_llvm(*m->llvm_ctx);
```

Then `handle_mlir` either prints a dump or saves an object:

```cpp
if (mlir_new_req->dump_only) {
    LCompilers::write_mlir_new_dump(std::cout, *m, mlir_new_req->target);
} else {
    e.save_object_file(*(m->llvm_m), outfile);
}
```

Code grounding:

- `src/lfortran/fortran_evaluator.cpp` lines 646-648 call `mlir_to_llvm`.
- `src/libasr/codegen/evaluator.cpp` line 234 defines `MLIRModule::mlir_to_llvm`.
- `src/bin/lfortran.cpp` lines 1141-1147 print dumps or save the object file.
- `src/bin/lfortran.cpp` lines 2030-2031 allow `mlir_new` in executable linking.

Execution path:

```text
--backend=mlir-new -c file.f90
  -> object file saved by LLVMEvaluator::save_object_file

--backend=mlir-new file.f90
  -> object file produced
  -> normal LFortran executable link path handles final executable

./a.out
  -> operating system executes the linked native binary
```

The new backend therefore owns the IR production up to an LLVM module/object.
The existing LFortran driver still owns final linking and running the resulting
program is outside the compiler.

## User Journeys

### Journey 1: See whether ASR was captured correctly

Command:

```console
lfortran --backend=mlir-new --show-mlir-asr-dialect example.f90
```

Code path:

```text
handle_mlir
  -> fe.get_asr2
  -> get_mlir_new_request(target=AsrDialect, dump_only=true)
  -> FortranEvaluator::get_mlir_new
  -> asr_to_mlir_new
  -> ASRToAsrDialectVisitor::visit_TranslationUnit
  -> ASR_DialectVerify
  -> ASR_DialectPrint
  -> write_mlir_new_dump prints mlir_asr_dialect()
```

Files involved:

- `src/bin/lfortran.cpp`
- `src/lfortran/fortran_evaluator.cpp`
- `src/libasr/codegen/asr_to_mlir_new.cpp`
- `src/libasr/codegen/asr_to_asr_dialect.cpp`
- `src/mlir/lfortran/asr_dialect_api_native.c`
- `src/mlir/lfortran/asr_dialect_pretty_print.c`

What the user learns:

- whether the ASR node appears in the dialect;
- whether symbol table regions contain expected variables;
- whether statement body regions contain expected assignments, prints, loops,
  and returns;
- whether source-location-backed diagnostics can point to the right Fortran
  site.

Example:

```fortran
program main
  integer :: x
  x = 7
end program
```

Expected ASR dialect concepts:

```text
asr.translation_unit
  asr.program @main
    asr.symtab
      asr.variable @x
    asr.body
      asr.assignment target=@x value=7
      asr.return
```

If `asr.variable @x` is missing, the issue is in program/symbol emission. If the
variable exists but assignment references a different name, the issue is in
`visit_Var` or symbol-reference construction.

### Journey 2: See whether ASR dialect lowers to normal MLIR

Command:

```console
lfortran --backend=mlir-new --show-mlir-high-dialect example.f90
```

Code path:

```text
handle_mlir
  -> fe.get_asr2
  -> target=HighMlir
  -> ASR dialect build
  -> ASR_DialectLowerToHighMLIR
  -> MLIR_PrintOperationUpstream
  -> write_mlir_new_dump prints mlir_high_dialect()
```

Primary file:

- `src/mlir/lfortran/asr_dialect_lowering_handlers.c`

What the user learns:

- whether declarations became allocations;
- whether variable reads became loads;
- whether assignments became stores;
- whether loops/ifs became CFG;
- whether prints became `vector.print`.

Example:

```fortran
integer :: x
x = 1 + 2
print *, x
```

Expected lowering concepts:

```text
ASR_LowerVariable
  -> memref.alloca

ASR_LowerIntegerBinOp
  -> arith.addi

ASR_LowerAssignment
  -> memref.store

ASR_LowerPrint
  -> vector.print
```

If `asr.assignment` is visible in the high-level dump, lowering did not replace
the ASR dialect operation. Check generated dispatch and `ASR_LowerAssignment`.

### Journey 3: See whether high MLIR becomes LLVM dialect

Command:

```console
lfortran --backend=mlir-new --show-mlir-llvm-dialect example.f90
```

Code path:

```text
handle_mlir
  -> target=LlvmDialect
  -> ASR dialect build
  -> ASR dialect lowering
  -> MLIR_LowerToLLVMDialect
  -> MLIR_PrintOperationUpstream
  -> write_mlir_new_dump prints mlir_llvm_dialect()
```

Primary files:

- `src/mlir/lfortran/mlir_lower_to_llvm.c`
- `src/mlir/lfortran/mlir_lower_lfortran_hooks.c`

What the user learns:

- whether `func.func` became `llvm.func`;
- whether `memref.*` was removed;
- whether `vector.print` became `printf`-style LLVM calls;
- whether CFG ops became LLVM branch ops.

Example:

```text
vector.print %x : i32
  -> MLIR_LFortranTryLowerOp
  -> ensure printf declaration
  -> create format string global
  -> llvm.call @printf
```

If `vector.print` survives in the LLVM dialect dump, inspect
`MLIR_LFortranTryLowerOp` and the vector-print prepass.

### Journey 4: Get LLVM IR text from the new MLIR path

Command:

```console
lfortran --backend=mlir-new --show-llvm-from-mlir example.f90
```

Code path:

```text
handle_mlir
  -> target=LlvmIr
  -> ASR dialect build
  -> high MLIR lowering
  -> LLVM dialect lowering
  -> MLIR_TranslateModuleToLLVMIR
  -> MLIRModule::mlir_to_llvm parses text
  -> write_mlir_new_dump prints llvm_ir_from_mlir_api
```

Primary files:

- `src/mlir/lfortran/mlir_translate_to_llvm_ir.c`
- `src/mlir/lfortran/mlir_translate_lfortran_hooks.c`
- `src/libasr/codegen/evaluator.cpp`

What the user learns:

- whether LLVM dialect translation emits parseable LLVM IR;
- whether function/global declarations are well-formed;
- whether `index` and pointer-like types were translated to valid LLVM types.

Example expected concepts:

```llvm
declare i32 @printf(ptr, ...)

define i32 @main() {
entry:
  %0 = alloca i32, i64 1
  store i32 3, ptr %0
  %1 = load i32, ptr %0
  call i32 (ptr, ...) @printf(ptr @.fmt_i32, i32 %1)
  ret i32 0
}
```

If this stage fails after emitting text, the error usually belongs either to
`mlir_translate_to_llvm_ir.c` or to `MLIRModule::mlir_to_llvm` parsing and
verification.

### MLIR OpTypes
MLIR_Context
    │
    └── MLIR_OpHandle (module)
            └── MLIR_RegionHandle
                    └── MLIR_BlockHandle
                            ├── MLIR_ValueHandle[]  (block args)
                            └── MLIR_OpHandle[]     (ops in order)
                                    ├── MLIR_ValueHandle[]     (operands)
                                    ├── MLIR_ValueHandle[]     (results)
                                    ├── MLIR_TypeHandle[]      (result types)
                                    ├── MLIR_AttributeHandle[] (attrs)
                                    ├── MLIR_RegionHandle[]    (nested regions)
                                    ├── MLIR_BlockHandle[]     (successors, for branches)
                                    └── MLIR_LocationHandle    (where in source)

### Journey 5: Produce an object file

Command:

```console
lfortran --backend=mlir-new -c example.f90 -o example.o
```

Code path:

```text
handle_mlir
  -> target=ObjectFile
  -> full mlir-new pipeline through LLVM IR text
  -> MLIRModule::mlir_to_llvm
  -> LLVMEvaluator::save_object_file
```

Files involved:

- `src/bin/lfortran.cpp` for routing and object save;
- `src/lfortran/fortran_evaluator.cpp` for `get_mlir_new`;
- `src/libasr/codegen/evaluator.cpp` for parsing generated LLVM IR text.

What the user gets:

An object file produced through the new MLIR path, using the existing LLVM object
emission support after the new backend has created an `llvm::Module`.

### Journey 6: Produce and run an executable

Command:

```console
lfortran --backend=mlir-new example.f90 -o example
./example
```

Code path:

```text
handle_mlir for object creation
  -> save object
  -> normal LFortran link_executable flow
  -> native executable
```

Code grounding:

- `src/bin/lfortran.cpp` lines 2030-2031 allow `Backend::mlir_new` in the link
  path.

For a simple print program:

```fortran
program main
  integer :: x
  x = 42
  print *, x
end program
```

runtime output should come from the native LLVM IR path that lowered
`vector.print` to `printf`.

### Journey 7: Compare native lowering with upstream MLIR lowering

Command:

```console
USE_MLIR_Upstream=1 lfortran --backend=mlir-new --show-mlir-llvm-dialect example.f90
```

Code path:

```text
get_mlir_new_request
  -> request.use_upstream_lowering = true
asr_to_mlir_new
  -> MLIR_LowerToLLVMDialectUpstream
  -> MLIR_TranslateModuleToLLVMIRUpstream when LLVM IR is requested
```

Primary file:

- `src/mlir/lfortran/mlir_api_impl_upstream.cpp`

Why this journey exists:

The native path is deliberately small and prototype-oriented. The upstream path
lets a developer compare behavior against upstream MLIR passes and isolate
whether a bug is in ASR dialect lowering or in native high-MLIR-to-LLVM lowering.

### Journey 8: Diagnose an unsupported ASR node

Example input:

```fortran
program main
  real :: x
  x = sqrt(2.0)
  print *, x
end program
```

Possible behavior:

- ASR dialect dump may succeed because many ASR nodes are representable.
- High-level lowering or compile may fail because real/intrinsic lowering is
  outside the current compile-supported subset.

Code path:

```text
generated lowering dispatch
  -> supported op kind calls ASR_Lower* handler
  -> unsupported op kind calls ASR_LowerUnsupported
```

Files involved:

- `src/libasr/asdl_to_asr_dialect.py` lines 43-55 define
  `COMPILE_SUPPORTED_OPS`.
- `src/mlir/generated/asr_dialect_lowering_dispatch.h` routes unsupported nodes.
- `src/mlir/lfortran/asr_dialect_api_native.c` lines 685-791 manage codegen
  errors and unsupported diagnostics.

Use the ASR dialect dump first:

```console
lfortran --backend=mlir-new --show-mlir-asr-dialect failing.f90
```

If the ASR dialect dump looks correct, the missing work is a lowering handler
rather than ASR serialization.

### Journey 9: Add support for another ASR operation

This is the developer journey for extending the backend.

Steps:

1. Check whether the ASR node already exists in `ASR.asdl`.
2. Check whether the generator already emits a schema and wrapper for it.
3. If it is representable but not compilable, add it to `COMPILE_SUPPORTED_OPS`
   in `src/libasr/asdl_to_asr_dialect.py`.
4. Regenerate the generated headers through the CMake custom command or the
   generator script.
5. Add a lowering handler in
   `src/mlir/lfortran/asr_dialect_lowering_handlers.c`.
6. Make sure generated dispatch calls the handler.
7. Add or update ASR dialect emission only if the generated visitor is not
   enough.
8. Test all stage dumps for a small Fortran example.

Example support path:

```text
ASR.asdl
  -> generator emits ASR_OP_NEW_EXPR, schema, accessor, create wrapper
  -> add op to COMPILE_SUPPORTED_OPS
  -> generated dispatch calls ASR_LowerNewExpr
  -> implement ASR_LowerNewExpr in asr_dialect_lowering_handlers.c
  -> verify high MLIR contains the expected arith op
```

Example stage checks:

```console
lfortran --backend=mlir-new --show-mlir-asr-dialect new_expr.f90
lfortran --backend=mlir-new --show-mlir-high-dialect new_expr.f90
lfortran --backend=mlir-new --show-mlir-llvm-dialect new_expr.f90
lfortran --backend=mlir-new --show-llvm-from-mlir new_expr.f90
lfortran --backend=mlir-new -c new_expr.f90
```

### Journey 10: Change ASR.asdl or storage policy

Files involved:

- `src/libasr/ASR.asdl`
- `src/libasr/asdl_to_asr_dialect.py`
- `src/mlir/lfortran/asr_dialect_storage_policy.h`
- `src/mlir/generated/*`
- `src/libasr/codegen/asr_to_asr_dialect.cpp`
- `src/libasr/codegen/asr_to_asr_dialect.h`

Build/codegen path:

```text
src/mlir/CMakeLists.txt
  custom command lines 185-209
  -> invokes asdl_to_asr_dialect.py
  -> writes src/mlir/generated/*
  -> patches generated sections in asr_to_asr_dialect.*
```

What to verify:

```console
git diff src/mlir/generated
git diff src/libasr/codegen/asr_to_asr_dialect.cpp
git diff src/libasr/codegen/asr_to_asr_dialect.h
```

If generated files changed unexpectedly, inspect:

- field kind classification in `asdl_to_asr_dialect.py`;
- storage overrides in `asr_dialect_storage_policy.h`;
- placeholder/unsupported-field validation in the generator.

## End-to-End User Flow

### Basic Example

Input:

```fortran
program main
  integer :: x
  x = 1 + 2
  print *, x
end program
```

ASR is still produced by the normal frontend and semantic passes.  The new
backend starts after ASR exists.

With:

```console
lfortran --backend=mlir-new --show-mlir-asr-dialect example.f90
```

the new path emits an ASR dialect dump.  Conceptually, the dump contains:

```mlir
module {
  asr.translation_unit {
    asr.program @main {
      asr.symtab {
        asr.variable @x : !asr.integer<4>
      }
      asr.metadata {
      }
      asr.body {
        asr.assign (asr.integer_bin_op + 1, 2) to @x : !asr.integer<4>
        asr.print @x
        asr.return
      }
    }
  }
}
```

The exact textual form is produced by
`src/mlir/lfortran/asr_dialect_pretty_print.c`, but the structure above shows the
important processing model:

- declarations become entries in a symbol table region;
- executable statements become body-region operations;
- expression operands are stored as nested ops, operands, attributes, or regions
  according to the generated storage policy;
- `Var` uses a symbol-reference string, not a raw C++ ASR pointer.

With:

```console
lfortran --backend=mlir-new --show-mlir-high-dialect example.f90
```

the ASR dialect lowers into high-level MLIR such as:

```mlir
module {
  func.func @main() -> i32 {
    %x = memref.alloca : memref<1xi32>
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %sum = arith.addi %c1, %c2 : i32
    %zero = arith.constant 0 : index
    memref.store %sum, %x[%zero] : memref<1xi32>
    %loaded = memref.load %x[%zero] : memref<1xi32>
    vector.print %loaded : i32
    func.return %zero_i32 : i32
  }
}
```

The names above are illustrative.  The implementation creates SSA names with a
local counter.  The important details are:

- variables become stack-like `memref.alloca`;
- scalar assignment becomes a store to index zero;
- scalar variable use becomes a load from index zero;
- `print` initially becomes `vector.print`;
- function return is normalized to `func.return` with an integer status.

With:

```console
lfortran --backend=mlir-new --show-mlir-llvm-dialect example.f90
```

the high-level MLIR lowers to LLVM dialect.  LFortran-specific hooks rewrite
operations that the prototype needs immediately:

```mlir
llvm.func @main() -> i32 {
  %buffer = llvm.alloca ...
  %ptr = llvm.getelementptr ...
  llvm.store ...
  %value = llvm.load ...
  llvm.call @printf(...)
  llvm.return ...
}
```

With:

```console
lfortran --backend=mlir-new --show-llvm-from-mlir example.f90
```

the LLVM dialect is translated into textual LLVM IR.  The resulting string is
stored in `MLIRModule::llvm_ir_from_mlir_api` and then parsed back into an LLVM
module when the normal object-file path needs an `llvm::Module`.

## Build and Link Changes

### `CMakeLists.txt`

What changed:

- Lines 156-157 add cache option `LFORTRAN_MLIR_WASM_LOWERING`.
- Lines 282-291 add optional WebAssembly LLVM components when `WITH_MLIR`,
  `LFORTRAN_MLIR_WASM_LOWERING`, and not `WITH_TARGET_WASM`.
- Lines 308-311 replace a hard-coded small `mlir_libs` list with
  `include(LFortranMLIRLink)`, MLIR include directories, and
  `${LFORTRAN_MLIR_LINK_LIBS}`.

Why:

The new backend brings in more MLIR and LLVM surface area than the previous
minimal integration.  Static MLIR builds often require many interdependent
`libMLIR*.a` archives.  Linking only a few named libraries is fragile because a
new pass, dialect, or translation hook can silently add another archive
dependency.

How:

The top-level CMake file delegates MLIR library discovery to
`cmake/LFortranMLIRLink.cmake`.  It also exposes wasm lowering as a cache option
so the normal native `mlir-new` path does not have to force wasm components into
all builds.

Example:

```cmake
option(LFORTRAN_MLIR_WASM_LOWERING
    "Enable optional MLIR-to-WASM lowering support in the mlir-new prototype"
    OFF)
```

This keeps the ordinary native backend path smaller, while still allowing a
developer to opt into wasm-specific lowering experiments.

### `cmake/LFortranMLIRLink.cmake`

What it does:

- Lines 1-50 add a new CMake helper.
- It searches `LLVM_LIBRARY_DIR`, `MLIR_LIBRARY_DIR`, or derives a lib directory
  from `MLIR_DIR`.
- It globs `libMLIR*.a`.
- On Linux/FreeBSD it wraps static MLIR archives with
  `-Wl,--start-group` and `-Wl,--end-group`.
- If no static archives are found, it falls back to a small named list:
  `MLIRIR`, `MLIRLLVMToLLVMIRTranslation`,
  `MLIRBuiltinToLLVMIRTranslation`, `MLIRLLVMDialect`,
  `MLIROpenMPToLLVMIRTranslation`, and `MLIROpenMPDialect`.

Why:

The upstream MLIR C++ bridge in `src/mlir/lfortran/mlir_api_impl_upstream.cpp`
can touch many dialects and translations.  Static library order matters on Unix
linkers.  Grouping the MLIR archives lets the linker resolve cyclic references
without hand-maintaining a long and brittle library list.

Simplification:

The helper does not try to model every MLIR target precisely.  For this
prototype it prefers linking all available static MLIR archives when present.
That is heavier than a minimal dependency graph, but much simpler and more
robust while the backend surface is still moving.

### `src/CMakeLists.txt`

What changed:

- Lines 1-3 add:

```cmake
if (WITH_LLVM AND WITH_MLIR)
    add_subdirectory(mlir)
endif()
```

Why:

All new MLIR C API, ASR dialect runtime, generator outputs, native lowering, and
upstream shims live under `src/mlir`.  The directory should only be built when
both LLVM and MLIR support are enabled.

How it connects:

`src/mlir/CMakeLists.txt` builds libraries that are then linked into `libasr` by
`src/libasr/CMakeLists.txt`.

### `src/libasr/CMakeLists.txt`

What changed:

- Lines 97-101 add:
  - `codegen/asr_to_mlir_new.cpp`
  - `codegen/asr_to_asr_dialect.cpp`
- Lines 159-164 link the `asr` target against:
  - `lfortran_mlir_c_api`
  - `lfortran_asr_dialect`
- The include path now reaches:
  - `../mlir/lfortran`
  - `../mlir`

Why:

The public C++ entrypoint for the backend lives in `libasr/codegen`, but the
backend implementation uses a C MLIR API and ASR dialect runtime under
`src/mlir/lfortran`.  `libasr` therefore needs to compile the C++ frontend of the
pipeline and link the C runtime that owns dialect creation, verification,
printing, and lowering.

Example flow:

```text
src/libasr/codegen/asr_to_mlir_new.cpp
  includes src/mlir/lfortran/asr_dialect_api.h
  calls ASR_DialectCreate()
  calls ASR_DialectVerifyModule()
  calls ASR_DialectLowerToHighMLIR()
  calls MLIR_LowerToLLVMDialect()
  calls MLIR_TranslateModuleToLLVMIR()
```

### `src/mlir/CMakeLists.txt`

What it does:

- Lines 1-14 document that this directory builds the `mlir-new` support layer.
- Lines 16-18 return unless `WITH_MLIR` and `WITH_LLVM` are enabled.
- Lines 20-29 define `lf_mlir_source`, an override helper:
  - prefer `src/mlir/lfortran/<relative path>` if present;
  - otherwise use the upstream `src/mlir/<relative path>` source.
- Lines 31-35 fail early if the upstream MLIR submodule source root is missing.
- Lines 37-74 build `lfortran_corec`, including local hosted platform sources.
- Lines 76-90 select C API core source overrides.
- Lines 91-102 collect optional wasm lowering sources.
- Lines 104-111 collect LFortran-specific lowering and translation hooks.
- Lines 113-117 choose the local `mlir_api_impl_upstream.cpp` override.
- Lines 119-130 build `lfortran_mlir_c_api_core`.
- Lines 132-153 build `lfortran_mlir_upstream_shim` and link it with
  `p::mlir` and `p::llvm`.
- Lines 154-170 optionally build wasm lowering support.
- Lines 172-179 expose interface target `lfortran_mlir_c_api`.
- Lines 185-209 run `src/libasr/asdl_to_asr_dialect.py` as a custom command.
- Lines 211-220 create target `generate_asr_dialect`.
- Lines 222-243 build `lfortran_asr_dialect`.

Why:

The new backend needs both generated files and handwritten runtime files.  The
generator depends on `ASR.asdl` and the storage policy header.  The C runtime
depends on the generated schema, ids, accessors, and lowering dispatch.  CMake
has to encode that dependency order explicitly:

```text
ASR.asdl + asr_dialect_storage_policy.h
  -> asdl_to_asr_dialect.py
  -> src/mlir/generated/*
  -> lfortran_asr_dialect
  -> libasr
  -> lfortran executable
```

Simplification:

Instead of introducing a full MLIR TableGen/ODS dialect in this prototype, the
ASR dialect is represented with unregistered MLIR operations plus generated C
metadata.  That lets the backend iterate from `ASR.asdl` without having to
maintain parallel ODS files.

## Command-Line and Evaluator Changes

### `src/bin/lfortran_command_line_parser.h`

What changed:

- Lines 47-49 add these booleans:

```cpp
bool show_mlir_high_dialect;
bool show_mlir_asr_dialect;
bool show_mlir_llvm_dialect;
```

Why:

The old `--show-mlir` had only one MLIR dump meaning.  The new backend has three
useful MLIR-stage dumps:

- ASR dialect;
- high-level MLIR;
- LLVM dialect MLIR.

The parser needs separate state for each stage so `lfortran.cpp` can choose the
right `MlirNewPipelineTarget`.

### `src/bin/lfortran_command_line_parser.cpp`

What changed:

- Lines 302-313 expand `--show-mlir` help text.
- Lines 302-313 add:
  - `--show-mlir-high-dialect`
  - `--show-mlir-asr-dialect`
  - `--show-mlir-llvm-dialect`
- Line 333 adds `mlir-new` to the backend help string.

Why:

Developers need to inspect each stage while validating the new backend.  The ASR
dialect stage answers "did ASR serialize correctly?", the high-level MLIR stage
answers "did ASR lower into normal MLIR control/data flow?", and the LLVM
dialect stage answers "did high-level MLIR lower toward LLVM correctly?".

Example debugging sequence:

```console
lfortran --backend=mlir-new --show-mlir-asr-dialect loop.f90
lfortran --backend=mlir-new --show-mlir-high-dialect loop.f90
lfortran --backend=mlir-new --show-mlir-llvm-dialect loop.f90
lfortran --backend=mlir-new --show-llvm-from-mlir loop.f90
```

This lets a developer identify whether a failure belongs to:

- ASR dialect emission;
- ASR dialect verification;
- ASR dialect to high-level MLIR lowering;
- high-level MLIR to LLVM dialect lowering;
- LLVM dialect to LLVM IR translation;
- LLVM parsing/object emission.

### `src/bin/lfortran.cpp`

What changed:

- Line 28 includes `libasr/codegen/asr_to_mlir_new.h`.
- Line 59 includes `<optional>`.
- Lines 98-114 add helper functions:
  - `mlir_show_from_cli()`
  - `mlir_show_for_compile()`
- Line 120 adds `Backend::mlir_new`.
- Lines 1097-1098 update `handle_mlir` to accept:
  - backend string;
  - `LCompilers::MlirShowOptions`.
- Lines 1127-1133 create an optional `MlirNewRequest` with
  `get_mlir_new_request()`.
- Lines 1127-1133 route selected requests to `fe.get_mlir_new`.
- Lines 1141-1149 choose `write_mlir_new_dump()` for the new path and keep
  classic `show_mlir` / `show_llvm_from_mlir` for the old path.
- Lines 2030-2031 allow `Backend::mlir_new` in `link_executable`.
- Lines 2702-2705 parse `--backend=mlir-new`; backend error text includes it.
- Lines 2762-2767 use `.mlir` output for new dump flags.
- Lines 2828-2834 route all MLIR show flags through `handle_mlir`.
- Lines 2909-2916 support `-c --backend=mlir-new`.
- Lines 2980-2987 support multi-file compile path for `mlir-new`.

Why:

The new backend must behave like other LFortran backends from the user's point
of view.  It needs to work for:

- stage dumps;
- `-c` object generation;
- executable link flow;
- multi-file compile flow;
- normal backend selection errors.

How:

The driver converts CLI flags into a `MlirNewRequest`.  That request tells the
pipeline what target to produce:

```text
--show-mlir-asr-dialect   -> MlirNewPipelineTarget::AsrDialect
--show-mlir-high-dialect  -> MlirNewPipelineTarget::HighMlir
--show-mlir-llvm-dialect  -> MlirNewPipelineTarget::LlvmDialect
--show-llvm-from-mlir     -> MlirNewPipelineTarget::LlvmIr
-c / normal compile       -> MlirNewPipelineTarget::ObjectFile
```

Important behavior:

- `--show-mlir` still exists.  For `--backend=mlir-new`, it maps to high-level
  MLIR because that is the closest analog to the old "show MLIR" behavior.
- The old MLIR backend remains available as `--backend=mlir`.
- New stage flags can select the new backend even when the backend string is
  not explicitly `mlir-new`, because `get_mlir_new_request()` checks the show
  flags too.

### `src/lfortran/fortran_evaluator.h`

What changed:

- Line 15 includes `libasr/codegen/asr_to_mlir_new.h`.
- Lines 117-119 declare:

```cpp
std::unique_ptr<LLVMEvaluator> get_mlir_new(
    ASR::TranslationUnit_t &asr, diag::Diagnostics &diagnostics,
    const MlirNewRequest &request);
```

Why:

`FortranEvaluator` is the existing boundary where the driver asks for compiled
representations.  Adding `get_mlir_new` keeps the new backend behind the same
evaluation abstraction as the old MLIR and LLVM paths.

### `src/lfortran/fortran_evaluator.cpp`

What changed:

- Line 29 includes the new backend header.
- Lines 625-655 implement `get_mlir_new`.

How:

`get_mlir_new` creates an `MLIRModule` through `asr_to_mlir_new`.  When the
request target is `ObjectFile` or `LlvmIr`, it calls:

```cpp
m->mlir_to_llvm(*m->llvm_ctx);
```

That converts the textual LLVM IR string produced by the new MLIR C API into an
LLVM module that can use the existing LFortran object generation path.

Why no default ASR passes here:

The new backend wants to inspect the ASR shape close to what the frontend
produces and then lower it through the ASR dialect.  The function therefore does
not run the normal default ASR pass pipeline before conversion.  That keeps the
prototype focused on dialect correctness instead of depending on additional ASR
rewrites.

### `src/libasr/codegen/evaluator.h`

What changed:

- Lines 55-62 add string snapshots:
  - `mlir_asr_dialect_text`
  - `mlir_high_dialect_str`
  - `mlir_llvm_dialect_text`
  - `llvm_ir_from_mlir_api`
- Lines 65-66 add a string-backed constructor.
- Lines 69-71 add dump getters:
  - `mlir_high_dialect()`
  - `mlir_asr_dialect()`
  - `mlir_llvm_dialect()`

Why:

The new pipeline currently produces text from the C MLIR API.  It needs to carry
multiple text snapshots at once so the CLI can print exactly the requested
stage.  The old `MLIRModule` model was centered on one upstream MLIR module
object and one dump.

### `src/libasr/codegen/evaluator.cpp`

What changed:

- Lines 174-181 implement the string-backed constructor.
- Lines 188-190 make `mlir_str()` return `mlir_high_dialect_str` when present.
- Lines 197-226 implement the three new dialect dump getters.
- Lines 235-250 update `mlir_to_llvm()` to parse
  `llvm_ir_from_mlir_api` with LLVM's `parseAssemblyString()`.

Why:

The native `mlir-new` translator returns LLVM IR text.  Existing downstream
LFortran code wants an `llvm::Module`.  Parsing the generated text is a compact
bridge that avoids rewriting the whole object emission path for the prototype.

How errors are handled:

If LLVM cannot parse or verify the generated IR, the code throws a codegen
error.  That pushes the error through existing compiler diagnostics instead of
silently continuing with a missing module.

## Public `mlir-new` Pipeline Entry

### `src/libasr/codegen/asr_to_mlir_new.h`

What it defines:

- Lines 14-22 define `MlirNewPipelineTarget`:
  - `AsrDialect`
  - `HighMlir`
  - `LlvmDialect`
  - `LlvmIr`
  - `ObjectFile`
- Lines 24-30 define `MlirShowOptions`.
- Lines 32-36 define `MlirNewRequest`:
  - target stage;
  - whether to use upstream lowering;
  - whether the request is dump-only.
- Lines 40-49 declare:
  - `get_mlir_new_request`
  - `write_mlir_new_dump`
  - `asr_to_mlir_new`

Why:

This header is the small public contract between the driver/evaluator and the
new backend.  Everything else is implementation detail.

Example request:

```cpp
MlirShowOptions show;
show.show_mlir_asr_dialect = true;

auto request = get_mlir_new_request("mlir-new", show, false);
// request->target == MlirNewPipelineTarget::AsrDialect
// request->dump_only == true
```

### `src/libasr/codegen/asr_to_mlir_new.cpp`

What it does:

- Lines 13-18 include the ASR dialect C API, MLIR API, and corec platform/arena
  support.
- Lines 22-27 initialize the hosted corec platform once.
- Lines 29-34 copy `MLIR` C API strings into C++ strings.
- Lines 36-38 clean up the ASR dialect emitter arena.
- Lines 40-49 build a string-backed `MLIRModule`.
- Lines 51-87 convert native verify/lowering errors into LFortran diagnostics.
- Lines 89-96 decide whether a CLI/backend selection uses `mlir-new`.
- Lines 98-123 build a `MlirNewRequest`.
- Lines 125-143 print the selected dump string.
- Lines 145-252 implement the complete pipeline.

How the pipeline runs:

1. Initialize hosted platform support.
2. Require the ASR root to be a `TranslationUnit`.
3. Build an ASR dialect module using `ASRToAsrDialectVisitor`.
4. Verify the ASR dialect module.
5. Print the ASR dialect snapshot.
6. If the target is `AsrDialect`, return immediately.
7. Lower the ASR dialect to high-level MLIR with `ASR_DialectLowerToHighMLIR`.
8. Print the high-level MLIR snapshot.
9. If the target is `HighMlir`, return immediately.
10. Lower high-level MLIR to LLVM dialect.
11. Print the LLVM dialect snapshot.
12. If the target is `LlvmDialect`, return immediately.
13. Translate LLVM dialect to LLVM IR.
14. Return an `MLIRModule` containing all stage strings.

Why verification happens before lowering:

ASR dialect creation is a serialization boundary.  Verification checks that the
serialized representation has the expected fields, regions, symbol references,
and body layout before lowering starts.  That makes failures more local and
easier to diagnose.

Example failure split:

```text
Bad ASR dialect field layout
  -> ASR_DialectVerifyModule reports an ASR dialect verification error.

Valid ASR dialect but unsupported expression lowering
  -> ASR_DialectLowerToHighMLIR reports a codegen/lowering error.

Valid high-level MLIR but missing native LLVM conversion
  -> MLIR_LowerToLLVMDialect reports a lowering problem.
```

Upstream toggle:

```console
USE_MLIR_Upstream=1 lfortran --backend=mlir-new --show-mlir-llvm-dialect file.f90
```

When that environment variable is set, the pipeline chooses the upstream MLIR
lowering/translation entrypoints where available.  Without it, it uses the
native C lowering path.

## ASDL-Driven Dialect Generation

### `src/libasr/asdl_to_asr_dialect.py`

What it does:

This generator reads `src/libasr/ASR.asdl` and emits the generated ASR dialect
headers under `src/mlir/generated`.  It also patches generated visitor sections
inside `src/libasr/codegen/asr_to_asr_dialect.h` and
`src/libasr/codegen/asr_to_asr_dialect.cpp`.

Major sections:

- Lines 1-13 describe outputs and source patching.
- Lines 23-41 categorize ASDL enums, operation categories, and product ops.
- Lines 43-55 define `COMPILE_SUPPORTED_OPS`.
- Lines 57-62 define skipped fields.
- Lines 72-144 map ASDL field kinds to dialect field kinds.
- Lines 156-178 generate `ASR_DialectField` assignments.
- Lines 182-191 define stored dialect fields and normalize `ArrayConstant`.
- Lines 194-244 define storage classification and synthetic layout fields.
- Lines 251-294 parse storage overrides from `asr_dialect_storage_policy.h`.
- Lines 297-372 model field storage policy.
- Lines 375-439 merge default and overridden storage decisions.
- Lines 446-483 validate overrides and placeholder field handling.
- Lines 485-521 generate merged storage and layout headers.
- Lines 524-655 generate ids, schema field entries, and accessors.
- Lines 658-682 collect ASDL ops.
- Lines 685-692 collect simple enums.
- Lines 695-831 generate `asr_dialect_schema.h`.
- Lines 834-884 generate inline `ASR_Create*Op` wrappers.
- Lines 902-983 generate C++ visitor field emission snippets.
- Lines 986-996 list handwritten visitor ops that should not be generated.
- Lines 1025-1065 generate visitor declarations and implementations.
- Lines 1068-1119 replace marked generated sections in the C++ visitor files.
- Lines 1122-1155 generate lowering dispatch.
- Lines 1158-1190 generate enum-print helpers.
- Lines 1193-1419 generate pretty-print policy.
- Lines 1422-1450 write all outputs and update visitor sources.
- Lines 1453-1487 implement `--check` mode.
- Lines 1490-1520 parse CLI arguments and run generation.

Why:

ASR has many node kinds.  Handwriting schema, wrapper constructors, accessors,
printer policies, and lowering dispatch for every node would be repetitive and
error-prone.  Since ASR already has a declarative ASDL file, this generator uses
that file as the stable source of truth.

How fields are mapped:

Example ASDL-ish idea:

```text
IntegerBinOp(expr left, binop op, expr right, ttype type, expr? value)
```

The generator classifies:

- `left`: expression operand or nested op field;
- `op`: enum attribute;
- `right`: expression operand or nested op field;
- `type`: type descriptor field;
- `value`: optional expression field.

It then emits:

- schema entries describing those fields;
- generated accessors like `ASR_IntegerBinOpLeftOp`;
- generated constructor wrapper `ASR_CreateIntegerBinOpOp`;
- pretty-print policy for enum names;
- lowering dispatch entry if the op is in `COMPILE_SUPPORTED_OPS`.

Compile-supported subset:

The generator explicitly separates "can be represented in the ASR dialect" from
"can be lowered by the prototype compiler path".  `COMPILE_SUPPORTED_OPS`
currently focuses on:

- integer expressions;
- casts;
- variables;
- a subset of arrays;
- `TranslationUnit`;
- `Program`;
- `Variable`;
- `Assignment`;
- `Return`;
- `Print`;
- `FileWrite`;
- `DoLoop`;
- `If`;
- `ErrorStop`.

This means the ASR dialect dump can grow broadly while the compile path stays
narrow and explicit.  Unsupported compile-lowering cases should fail with a
diagnostic instead of becoming placeholder LLVM.

Important simplification:

`ArrayConstant` is normalized.  The ASR node may point at raw ASR heap storage,
but the dialect representation exposes an `elements` expression sequence.  This
avoids storing raw C++ ASR pointers inside the MLIR-side dialect.

### Generated file: `src/mlir/generated/asr_dialect_ids.h`

What it does:

- Defines generated enum ids for every ASR dialect operation and field.
- Gives stable numeric identifiers to operation kinds such as program,
  assignment, integer constant, integer binary op, variable, print, and return.
- Gives stable field ids used by schema lookup, storage lookup, and generated
  accessors.

Why:

The C API needs compact switchable ids instead of repeatedly comparing strings.
For example, verification and lowering can switch on:

```c
ASR_OP_ASSIGNMENT
ASR_OP_INTEGER_BIN_OP
ASR_OP_PRINT
ASR_OP_DO_LOOP
```

This also keeps generated accessors and handwritten code speaking the same ids.

### Generated file: `src/mlir/generated/asr_dialect_schema.h`

What it does:

- Defines operation categories.
- Defines field kinds.
- Defines field descriptors.
- Defines the schema table for all generated ASR dialect operations.
- Defines the name lookup table for `asr.*` operation names.

How it is used:

- `asr_dialect_api_native.c` uses schema entries while constructing ops.
- `asr_dialect_api_native.c` uses schema entries while verifying fields.
- `asr_dialect_storage.h` uses field descriptors to read operands, regions, and
  attributes.
- The pretty printer uses schema names for fallback output.

Example:

When a native op named `asr.assignment` is created, the runtime can use the
schema to know:

- it is a statement op;
- it expects target and value fields;
- target/value are stored as operation references or nested op fields;
- optional overloaded fields can be omitted.

### Generated file: `src/mlir/generated/asr_dialect_accessors.h`

What it does:

Generates static inline field readers for every operation field.  Examples from
the generated API include:

- `ASR_AssignmentTargetOp`
- `ASR_AssignmentValueOp`
- `ASR_IfTestOp`
- `ASR_DoLoopGetBodyRegion`
- `ASR_IntegerBinOpLeftOp`
- `ASR_IntegerBinOpRightOp`
- `ASR_VarMOp`
- `ASR_PrintValuesRegion`

Why:

Lowering and printing code should not duplicate storage mechanics.  For example,
the lowering handler for assignment can ask for the target and value through
generated accessors, regardless of whether a field is stored as:

- an MLIR operand;
- an MLIR attribute;
- a nested region;
- a metadata attribute.

### Generated file: `src/mlir/generated/asr_dialect_api_generated.h`

What it does:

Generates inline constructor wrappers for ASR dialect operations.  Useful
anchors include:

- `ASR_CreateArrayConstantOp` at line 246
- `ASR_CreateArrayItemOp` at line 315
- `ASR_CreateIntegerBinOpOp` at line 1247
- `ASR_CreateIntegerCompareOp` at line 1312
- `ASR_CreateIntegerConstantOp` at line 1339
- `ASR_CreateVarOp` at line 2910
- `ASR_CreateAssignmentOp` at line 3197
- `ASR_CreateDoLoopOp` at line 3387
- `ASR_CreateFileWriteOp` at line 3893
- `ASR_CreateIfOp` at line 4083
- `ASR_CreatePrintOp` at line 4263
- `ASR_CreateFunctionOp` at line 4667
- `ASR_CreateProgramOp` at line 4862
- `ASR_CreateVariableOp` at line 5090
- `ASR_CreateTranslationUnitOp` at line 5543

Why:

The C++ ASR visitor should create a `Program` op by calling a strongly named
wrapper, not by manually assembling a field array and operation name each time.
The wrappers also encode field order from the generated schema.

Example:

The handwritten visitor can construct:

```cpp
ASR_CreateAssignmentOp(ctx, loc, target, value, type, overloaded);
```

instead of manually writing:

```cpp
ASR_DialectField fields[] = {
    ... target ...,
    ... value ...,
    ... type ...,
    ... overloaded ...
};
ASR_DialectCreateOp(ctx, loc, "asr.assignment", fields, 4);
```

### Generated file: `src/mlir/generated/asr_dialect_lowering_dispatch.h`

What it does:

- Declares lowering handlers for supported operations.
- Switches from generated op kind to a handwritten lowering handler.
- Routes unsupported op kinds to `ASR_LowerUnsupported`.

Why:

This keeps the compile-supported subset explicit.  The ASR dialect can represent
many nodes, but lowering only succeeds for handlers that have been implemented
and intentionally exposed.

Example:

```c
case ASR_OP_INTEGER_BIN_OP:
    return ASR_LowerIntegerBinOp(ctx, op);
default:
    return ASR_LowerUnsupported(ctx, op);
```

### Generated file: `src/mlir/generated/asr_dialect_enum_print.h`

What it does:

Maps ASR enum values to stable textual names for pretty printing.  This includes
operation-specific enums such as binary operators, comparison operators, storage
types, intent, access, ABI, array physical types, and other ASR enum classes.

Why:

Without generated enum printers, dumps would show only numeric values.  Numeric
enum output is difficult to review and unstable for humans.  The ASR dialect
dump should show meaningful text like `Add`, `Sub`, `Eq`, `In`, or `Local`.

### Generated file: `src/mlir/generated/asr_dialect_print_policy.h`

What it does:

Defines generated rules for how fields should be printed in the ASR dialect
pretty printer.  It includes enum mappings and variable-field handling rules.

Why:

The printer needs to know which fields are useful in a compact dump and how to
format them.  Keeping this generated from ASDL and policy data avoids duplicating
field names in handwritten printer logic.

### Generated file: `src/mlir/generated/asr_dialect_storage_merged.inc`

What it does:

Contains the merged storage table generated from:

- default ASDL-derived storage rules;
- handwritten overrides in `asr_dialect_storage_policy.h`.

The file is included through a macro list.  Each row has the shape:

```c
X(OpName, field_name, storage_kind, region_kind, region_element_kind)
```

Why:

The generator can choose reasonable defaults, but some ASR nodes need
handwritten layout choices.  For example:

- `Program` needs symbol table, metadata, and body regions.
- `Function` needs symbol table, metadata, and body regions.
- `DoLoop` needs head, body, and optional else regions.
- `If` needs body and optional else regions.
- `Var` should store a symbol reference, not a raw symbol pointer.

### Generated file: `src/mlir/generated/asr_dialect_layout_fields.h`

What it does:

Defines ids for synthetic layout fields such as:

- `Program.metadata`
- `Function.metadata`
- `ArrayConstant.elements`

Why:

These fields do not map one-to-one to normal ASDL storage, but they are needed
for a regular MLIR-side layout.  Giving them generated ids lets all runtime
helpers treat them like first-class fields.

## ASR to ASR Dialect Emission

### `src/libasr/codegen/asr_to_asr_dialect.h`

What it declares:

- Lines 23-31 define `AsrDialectError`.
- Lines 33-47 define `ASRToAsrDialectVisitor` state:
  - arena;
  - ASR allocator;
  - MLIR context;
  - source location;
  - module block and module op;
  - last emitted dialect value;
  - termination state;
  - suppression state.
- Lines 48-68 declare type conversion and generic emit helpers.
- Lines 69-82 declare module and scope-region helpers.
- Lines 73-78 define `ScopeRegion`, containing:
  - `symtab`;
  - `metadata`;
  - `body`.
- Lines 84-92 declare helpers for expression, statement, do-loop head, array
  index, and print emission.
- Lines 94-105 declare handwritten overrides.
- Lines 106-276 contain generated visitor declarations.
- Lines 278-282 declare top-level program visit functions.

Why:

The ASR visitor is the bridge from C++ ASR objects into the C ASR dialect API.
It needs enough state to emit nested regions, remember the last expression op,
and attach source locations for diagnostics.

### `src/libasr/codegen/asr_to_asr_dialect.cpp`

What it does:

- Lines 9-14 convert ASR strings to C strings.
- Lines 25-33 map ASR type kind to integer bit width.
- Lines 35-68 convert ASR types to MLIR API type handles.
- Lines 70-148 emit expression, statement, and array-index sequences.
- Lines 150-218 create placeholder product/type sequence helpers where the
  prototype does not yet carry full lowering semantics.
- Lines 220-235 create the MLIR module skeleton.
- Lines 264-283 create statement regions.
- Lines 285-321 append ops to scopes or module and attach source-location
  attributes.
- Lines 323-350 dispatch expression and statement sequence emission.
- Lines 352-386 handle do-loop heads and array index product ops.
- Lines 388-455 handle `Print` and `FileWrite`.
- Lines 457-489 handle `DoLoop`, `If`, and `WhileLoop` with regions.
- Lines 492-518 handle `Variable`.
- Lines 520-580 handle arrays:
  - `ArrayConstructor`;
  - `ArrayItem`;
  - `ArrayConstant`.
- Lines 583-2377 contain generated visitor implementations for many ASR nodes.
- Lines 2379-2391 handle `Var`.
- Lines 2393-2506 handle `Function`.
- Lines 2508-2573 handle `Program`.
- Lines 2575-2596 handle `TranslationUnit`.

Why:

The visitor serializes existing ASR into a dialect form that can be verified,
printed, and lowered without depending on C++ ASR object lifetimes.  This creates
a clear boundary between "LFortran ASR world" and "MLIR C API world".

How simple expressions are emitted:

For:

```fortran
x = 1 + 2
```

ASR contains an assignment whose value is an integer binary op.  The visitor
emits roughly:

```text
ASR visit Assignment
  -> emit target expression
      -> visit Var(x)
      -> ASR_CreateVarOp(symbol_ref="@x", type=i32)
  -> emit value expression
      -> visit IntegerBinOp
          -> visit IntegerConstant(1)
          -> ASR_CreateIntegerConstantOp(...)
          -> visit IntegerConstant(2)
          -> ASR_CreateIntegerConstantOp(...)
          -> ASR_CreateIntegerBinOpOp(left, Add, right, type, value)
  -> ASR_CreateAssignmentOp(target, value, type, overloaded)
```

How `Program` is emitted:

For:

```fortran
program main
  integer :: x
  x = 1
end program
```

the visitor creates:

```text
asr.translation_unit
  asr.program @main
    asr.symtab
      asr.variable @x
    asr.metadata
    asr.body
      asr.assignment
      asr.return
```

The body gets an automatic return when needed so the later `func.func @main() ->
i32` lowering has a terminating operation.

How loops are emitted:

For:

```fortran
do i = 1, 3
  print *, i
end do
```

the ASR dialect keeps loop structure as regions:

```text
asr.do_loop
  head:
    start = 1
    end = 3
    increment = 1
  body:
    asr.print @i
  orelse:
    empty
```

That makes the later lowering handler responsible for choosing a CFG shape.

How `Print` and `FileWrite` are simplified:

- `StringFormat` print arguments are flattened into multiple `asr.print`
  operations.
- `FileWrite` supports only stdout-like/default output, such as unit `6`.
- Multiple print values become multiple simple prints.

This is enough for examples like:

```fortran
print *, x, y
write(6, *) x
```

but not a full Fortran formatted I/O implementation.

How arrays are simplified:

- `ArrayConstructor`, `ArrayItem`, and `ArrayConstant` have handwritten
  emission.
- `ArrayConstant` uses `ASRUtils::fetch_ArrayConstant_value` to expose values as
  element operations.
- The dialect stores elements as an explicit region/sequence rather than raw ASR
  heap memory.

## ASR Dialect C API and Runtime

### `src/mlir/lfortran/asr_dialect_api.h`

What it defines:

- Lines 1-2 establish this as the stable public C API.
- Lines 14-15 define `ASR_SymbolRef` as `string`.
- Lines 17-27 define options and backend selection.
- Lines 29-33 include generated schema and accessors.
- Lines 35-55 define `ASR_DialectField`.
- Lines 57-89 define `ASR_LoweringContext`.
- Lines 91-94 define source-location attribute helpers.
- Lines 96-107 declare create/get/lookup functions.
- Lines 109-140 define verification and codegen error structs.
- Lines 142-154 declare verify, lower, print, and unsupported entrypoints.

Why:

The C++ codegen side should not know how the C MLIR API stores every field.
This header gives C++ a compact stable API:

```c
ASR_Dialect *dialect = ASR_DialectCreate(options);
MLIR_Op *op = ASR_CreateProgramOp(...);
ASR_DialectVerifyModule(dialect, module);
ASR_DialectPrintPretty(dialect, module);
ASR_DialectLowerToHighMLIR(dialect, module, options);
```

Important simplification:

`ASR_LoweringContext` uses a fixed symbol slot count through `ASR_MAX_SYMS`.
That is simple and predictable for prototype examples, but it is not a scalable
symbol-table design for the full compiler.

### `src/mlir/lfortran/asr_dialect_api.c`

What it does:

- Lines 1-3 describe it as a dispatch layer.
- Lines 10-22 declare native implementation entrypoints.
- Lines 24-29 create a native dialect instance.
- Lines 31-34 return native op kind.
- Lines 36-43 look up schema by kind.
- Lines 45-64 binary-search the generated name lookup table.
- Lines 66-104 dispatch errors, verification, lowering, printing, and
  unsupported handling to native implementations.

Why:

This separates the public API from the native implementation.  Today the
dispatch target is native-only, but the structure leaves room for other backends
or implementations without changing C++ call sites.

### `src/mlir/lfortran/asr_dialect_storage_policy.h`

What it does:

- Lines 1-10 describe the storage model:
  - ASDL is the source of truth;
  - simple expressions become operands where possible;
  - structural sequences become regions;
  - scalars/enums/strings/symbol names become attributes.
- Lines 16-28 define field storage kinds.
- Lines 30-47 define region and element kinds.
- Lines 49-59 define type kinds.
- Lines 63-64 include generated merged storage and layout fields.
- Lines 65-92 define handwritten storage overrides.
- Lines 95-110 define explicit region indexes.
- Line 113 defines metadata region index.

Why:

Default rules cannot express every useful layout.  For example, a `Program`
needs three distinct regions:

```text
symtab   -> declarations
metadata -> prototype metadata or derived layout notes
body     -> executable statements
```

Likewise, an `If` needs separate body and else regions, and a `DoLoop` needs
head/body/else layout.

How it simplifies the prototype:

The policy explicitly omits or remaps hard-to-serialize raw fields.  For
example:

- `Variable.parent_symtab` is omitted because it is a C++ ASR symbol-table
  pointer;
- `Var` stores a symbol reference string;
- `Assignment` stores target and value as operation fields;
- `Print` stores values in a region-like sequence.

### `src/mlir/lfortran/asr_dialect_storage.h`

What it does:

- Lines 24-26 define metadata constants and a max variadic op count.
- Lines 35-48 read attributes.
- Lines 50-63 map operation name to generated kind.
- Lines 65-78 look up field descriptors.
- Lines 80-109 read op fields through operands or defining ops.
- Lines 111-149 read field attributes.
- Lines 155-185 read regions and blocks.
- Lines 187-222 read operation sequences from regions.
- Lines 224-247 parse static memref lengths.
- Lines 249-264 read variable type/array helpers.
- Lines 264-273 count args/elements.

Why:

The verifier, pretty printer, and lowering handlers all need to read the same
stored dialect representation.  Centralizing the storage readers avoids each
consumer having its own copy of "field X is in operand 2" or "region Y is region
1".

Example:

An assignment lowering handler should be able to say:

```c
MLIR_Op *target = ASR_AssignmentTargetOp(ctx, op);
MLIR_Op *value = ASR_AssignmentValueOp(ctx, op);
```

without knowing whether target and value came from operands, nested regions, or
metadata.

### `src/mlir/lfortran/asr_dialect_api_native.c`

What it does:

- Lines 15-57 attach and read Fortran source-location attributes.
- Lines 61-68 build metadata attribute names.
- Lines 70-78 get an operation result value.
- Lines 80-182 plan and attach operation sequences/single ops to regions.
- Lines 184-215 convert fields to metadata attributes.
- Lines 217-221 choose result-type categories.
- Lines 233-390 implement `ASR_DialectCreateOpNative`.
- Lines 393-406 return native op kind.
- Lines 408-480 verify missing fields, presence, and region element kind.
- Lines 482-683 manage verification diagnostics.
- Lines 685-791 manage codegen/lowering diagnostics and unsupported handling.
- Lines 793-882 structurally verify the operation tree.
- Lines 884-1027 verify symbol sets and `Var` references.
- Lines 1029-1059 verify `Program` and `Function` region layout.
- Lines 1061-1085 verify scope symbols.
- Lines 1087-1127 verify the top-level module.
- Lines 1129-1133 dispatch pretty printing.
- Lines 1135-1147 verify then lower the module.

Why:

This file is the runtime authority for the ASR dialect.  It turns generated
schema data plus handwritten policy into concrete MLIR operations and checks
that the resulting tree is coherent before lowering.

How operation creation works:

```text
ASR_CreateAssignmentOp(...)
  -> generated wrapper packs fields
  -> ASR_DialectCreateOp(...)
  -> ASR_DialectCreateOpNative(...)
  -> schema lookup for asr.assignment
  -> fields classified as attrs, operands, or regions
  -> unregistered MLIR op is created
  -> region fields are attached
  -> source-location attrs are attached
```

How verification helps:

For:

```fortran
x = y
```

if `y` appears as an `asr.var @y` but there is no matching symbol in the current
program/function symbol table, the native verifier can report a symbol-reference
error before high-level MLIR lowering tries to create a load from an unknown
allocation.

How diagnostics remain source-aware:

The C++ visitor attaches source-location attributes.  Native verification and
lowering errors can read those attributes and report diagnostics at the original
Fortran source site where possible.

### `src/mlir/lfortran/asr_dialect_pretty_print.c`

What it does:

- Lines 1-5 state that this file is formatting-only.
- Lines 23-119 define display helpers for types and symbols.
- Lines 121-154 define printer context and line output helpers.
- Lines 160-169 peel casts for nicer expression printing.
- Lines 172-210 format simple expressions and array indexes.
- Lines 212-251 format array constants and constructors.
- Lines 254-307 format array items and print arguments.
- Lines 319-385 print expressions.
- Lines 397-454 print full variable attributes.
- Lines 458-497 print `asr.symtab`, `asr.metadata`, and `asr.body`.
- Lines 499-699 print statements and structural ops.
- Lines 701-820 format operation summaries for diagnostics.
- Lines 822-848 implement `ASR_DialectPrintPretty`.

Why:

Raw unregistered MLIR operations would be too noisy for debugging the dialect.
The printer provides a stable human-facing syntax for stage dumps and diagnostic
summaries.

Example ASR dialect dump shape:

```mlir
module {
  asr.translation_unit {
    asr.program @main {
      asr.symtab {
        asr.variable @x : !asr.integer<4>
      }
      asr.metadata {
      }
      asr.body {
        asr.assign 1 to @x : !asr.integer<4>
        asr.print @x
        asr.return
      }
    }
  }
}
```

Why this matters:

This dump is the easiest way to debug the first new backend boundary.  If an ASR
node is missing, stored in the wrong region, or printed with a placeholder, the
problem is visible before any MLIR lowering happens.

### Deleted file: `src/mlir/lfortran/asr_dialect_format.h`

What changed:

The diff deletes this empty placeholder header.  It is replaced in practice by:

- `src/mlir/lfortran/asr_dialect_pretty_print.c`
- `ASR_DialectPrintPretty`
- `ASR_DialectFormatOpSummary`

Why:

The printer is now implemented as a concrete C source file rather than a stub
format header.

## ASR Dialect to High-Level MLIR

### `src/mlir/lfortran/asr_dialect_lowering_handlers.c`

What it does:

- Lines 1-5 describe ASR dialect to high-level MLIR lowering.
- Lines 16-18 declare entry prototypes.
- Lines 20-24 generate SSA names.
- Lines 31-48 look up symbols by flat `ASR_VarV` strings.
- Lines 50-60 create and append MLIR operations.
- Lines 62-100 create constants, `memref.alloca`, and memref types.
- Lines 111-157 create CFG blocks and branches.
- Lines 159-169 map comparison predicates.
- Lines 171-195 handle index casts and Fortran 1-based to 0-based indexing.
- Lines 197-240 handle memref load/store and i32 binary operations.
- Lines 249-268 lower variables to allocations.
- Lines 271-320 lower array item load/store.
- Lines 322-365 initialize array constructors/constants.
- Lines 367-415 lower scalar and array assignments.
- Lines 417-478 lower print expressions to `vector.print`.
- Lines 480-486 lower `Print` and `FileWrite`.
- Lines 488-496 lower `Return`.
- Lines 498-518 lower regions with diagnostic site tracking.
- Lines 520-551 dispatch statements.
- Lines 553-583 lower `Program`.
- Lines 585-600 lower `TranslationUnit`.
- Lines 602-671 lower expressions.
- Lines 674-691 lower boolean comparisons.
- Lines 701-708 initialize common types.
- Lines 710-737 expose generated dispatch wrappers.
- Lines 738-740 leave `Function` unsupported.
- Lines 741-806 lower `DoLoop`.
- Lines 807-846 lower `If`.
- Lines 847-855 lower `ErrorStop`.
- Lines 857-909 lower a module into a `func.func @main() -> i32`.

Why:

This file is the semantic bridge from ASR dialect into common MLIR dialects.
The result is no longer ASR-shaped.  It uses regular operations such as:

- `func.func`
- `memref.alloca`
- `memref.load`
- `memref.store`
- `arith.constant`
- `arith.addi`
- `arith.cmpi`
- `cf.br`
- `cf.cond_br`
- `vector.print`

How scalar variables are represented:

For:

```fortran
integer :: x
x = 42
print *, x
```

the current lowering model is:

```text
asr.variable @x
  -> %x = memref.alloca : memref<1xi32>

asr.assignment @x, 42
  -> %c42 = arith.constant 42 : i32
  -> %zero = arith.constant 0 : index
  -> memref.store %c42, %x[%zero] : memref<1xi32>

asr.print @x
  -> %loaded = memref.load %x[%zero] : memref<1xi32>
  -> vector.print %loaded : i32
```

Why memref is used for scalars:

Using a one-element memref gives the prototype a single storage model for
variables.  Loads and stores are explicit, and array lowering can extend the same
model.  This is not the final optimized representation, but it is simple for
correctness experiments.

How arrays are represented:

For:

```fortran
integer :: a(3)
a = [1, 2, 3]
print *, a(2)
```

the intended lowering shape is:

```text
asr.variable @a : memref<3xi32>
  -> %a = memref.alloca : memref<3xi32>

asr.array_constant [1, 2, 3]
  -> stores each element into %a

asr.array_item @a(2)
  -> convert Fortran index 2 to zero-based index 1
  -> memref.load %a[%idx] : memref<3xi32>
```

The index conversion is explicitly handled because Fortran source indexing is
1-based in these simple examples, while MLIR memref indexes are zero-based.

How loops are lowered:

For:

```fortran
do i = 1, 3
  print *, i
end do
```

the handler emits explicit control-flow blocks:

```text
entry:
  store start into i
  cf.br ^loop_header

loop_header:
  load i
  compare i <= end
  cf.cond_br %cond, ^loop_body, ^loop_exit

loop_body:
  lower body statements
  load i
  add increment
  store i
  cf.br ^loop_header

loop_exit:
  continue after loop
```

Simplification:

The prototype uses explicit `cf` control flow rather than producing `scf.for`.
This keeps lowering direct and avoids needing a structured-loop construction
layer in the ASR dialect lowering pass.

How conditionals are lowered:

For:

```fortran
if (x > 0) then
  print *, x
else
  print *, 0
end if
```

the handler emits:

```text
condition
  -> arith.cmpi
  -> cf.cond_br ^then, ^else

then:
  lower then body
  cf.br ^merge

else:
  lower else body
  cf.br ^merge

merge:
  continue
```

Unsupported cases:

`Function` lowering is explicitly unsupported at lines 738-740.  The current
compile path focuses on top-level programs and `main`-like lowering.

## High-Level MLIR to LLVM Dialect

### `src/mlir/lfortran/mlir_lower_lfortran_hooks.h`

What it declares:

- Lines 1-3 explain that hooks are called from `mlir_lower_to_llvm.c` before
  generic lowering.
- Lines 13-20 define hook state.
- Lines 22-31 declare:
  - state initialization;
  - try-lower hook;
  - vector-print prepass.

Why:

The native generic lowering file knows how to lower broad operation families,
but LFortran needs special handling for prototype operations such as
`vector.print` and simple memref operations.

### `src/mlir/lfortran/mlir_lower_lfortran_hooks.c`

What it does:

- Lines 1-2 describe the hook scope:
  - `vector.print`;
  - `memref<1xi32>`;
  - index constants/casts.
- Lines 56-67 initialize state.
- Lines 69-113 ensure `printf` declaration and format strings.
- Lines 125-142 parse `memref<Nxi32>`.
- Lines 144-161 build array types for memref storage.
- Lines 163-203 lower `memref.alloca`.
- Lines 205-244 lower `memref.load`.
- Lines 246-279 lower `memref.store`.
- Lines 281-370 lower `vector.print`.
- Lines 373-404 lower index `arith.constant`.
- Lines 406-440 lower `arith.index_cast`.
- Lines 442-477 run a vector-print prepass.
- Lines 479-495 dispatch hook lowering.

Why:

The ASR dialect high-level lowering intentionally uses common MLIR operations
that are easy to inspect.  The native LLVM lowering then needs a small set of
LFortran-specific rewrites.

Example:

```mlir
vector.print %x : i32
```

becomes a call-like LLVM dialect sequence:

```text
create or reuse printf declaration
create or reuse "%d\n" global format string
possibly extend/cast the value
llvm.call @printf(format_ptr, value)
```

Example memref lowering:

```mlir
%a = memref.alloca : memref<3xi32>
memref.store %v, %a[%idx] : memref<3xi32>
%loaded = memref.load %a[%idx] : memref<3xi32>
```

becomes:

```text
llvm.alloca for 3 i32 values
llvm.getelementptr for indexed address
llvm.store
llvm.getelementptr for indexed address
llvm.load
```

Simplification:

The hook parser focuses on simple static memrefs such as `memref<Nxi32>`.  This
is enough for current integer scalar/array examples, but it is not a full memref
descriptor lowering.

### `src/mlir/lfortran/mlir_lower_to_llvm.c`

What it does:

- Lines 1-18 explain the native lowering pass.
- Lines 82-93 define `LowerState`.
- Lines 104-130 lower `arith.constant`.
- Lines 133-148 lower `func.return`.
- Lines 152-255 lower `func.func` to `llvm.func`.
- Lines 258-305 lower `func.call`.
- Lines 311-348 generically rename operations.
- Lines 353-395 lower `cf.br` and `cf.cond_br`.
- Lines 398-436 lower `func.constant`.
- Lines 443-453 lower `unrealized_conversion_cast`.
- Lines 458-504 lower indirect calls.
- Lines 507-530 rewrite `scf.yield`.
- Lines 541-630 lower `scf.if` to CFG.
- Lines 649-713 dispatch lowering, calling LFortran hooks first.
- Lines 715-751 recursively walk and lower the operation tree.
- Lines 753-767 implement `MLIR_LowerToLLVMDialect`.
- Lines 775-791 implement `MLIR_LowerToLLVMDialectForWasm`.

Why:

The prototype needs a native route from high-level MLIR to LLVM dialect without
depending on all upstream pass-pipeline behavior.  This file implements the
minimum useful lowering set directly in the C MLIR API.

How dispatch works:

```text
for each op:
  try LFortran-specific hook first
  else lower known func/arith/cf/scf operation
  else use a generic rename or leave a diagnostic/fallback path
```

Why hooks run first:

Operations such as `vector.print` are not generic LLVM lowering cases in this
prototype.  They need LFortran runtime-style behavior (`printf`) before the
generic pass rewrites function/control-flow structure.

Wasm note:

The wasm-specific entrypoint lifts CF to SCF and keeps SCF where needed for the
experimental wasm path.  Native `mlir-new` does not require this unless the wasm
option is enabled.

## LLVM Dialect to LLVM IR Text

### `src/mlir/lfortran/mlir_translate_lfortran_hooks.h`

What it declares:

- Lines 1-3 explain that the layout matches the translator buffer.
- Lines 21-25 declare hook functions for LLVM IR translation.

Why:

The native LLVM IR translator needs a couple of LFortran-specific formatting
decisions while staying mostly generic.

### `src/mlir/lfortran/mlir_translate_lfortran_hooks.c`

What it does:

- Lines 12-34 print GEP index type `index` as `i64`.
- Lines 36-38 skip label output for block 0.

Why:

LLVM IR does not have MLIR's `index` type.  The translator needs to pick a real
LLVM integer type; this prototype uses `i64` for index-like values.

### `src/mlir/lfortran/mlir_translate_to_llvm_ir.c`

What it does:

- Lines 1-10 describe this as a native LLVM dialect to LLVM IR text translator.
- Lines 31-66 implement a growable buffer.
- Lines 72-109 define maps and string helpers.
- Lines 115-144 define name/type hooks.
- Lines 146-220 begin type text translation.
- Line 313 implements inline literal handling.
- Line 434 folds aggregate initializer chains.
- Line 580 emits calls.
- Lines 646 and 651 handle phi emission support.
- Line 702 maps LLVM binary op names.
- Line 717 maps LLVM cast op names.
- Lines 732-1031 emit individual operations.
- Lines 1052-1139 emit functions.
- Lines 1145-1293 emit globals.
- Lines 1295-1390 collect named structs.
- Lines 1396-1468 implement `MLIR_TranslateModuleToLLVMIR`.

Why:

The `mlir-new` native path needs textual LLVM IR without requiring the complete
upstream MLIR translation stack for every experiment.  The existing downstream
LFortran object path can then parse that LLVM IR string into an `llvm::Module`.

Operation examples:

High-level print lowering may create LLVM dialect like:

```text
llvm.mlir.addressof @.fmt_i32
llvm.getelementptr ...
llvm.call @printf(...)
```

The translator emits LLVM IR text like:

```llvm
@.fmt_i32 = private constant [4 x i8] c"%d\0A\00"

declare i32 @printf(ptr, ...)

define i32 @main() {
entry:
  %0 = alloca i32, i64 1
  store i32 42, ptr %0
  %1 = load i32, ptr %0
  %2 = call i32 (ptr, ...) @printf(ptr @.fmt_i32, i32 %1)
  ret i32 0
}
```

The exact output depends on the MLIR C API values and generated names, but the
translator owns that conceptual conversion.

Simplification:

The translator is text-based.  It does not build LLVM IR objects directly.  That
keeps the C-side prototype small, but it means textual correctness and parse
verification in `MLIRModule::mlir_to_llvm()` are important.

## Upstream MLIR Bridge

### `src/mlir/lfortran/mlir_api_impl_upstream.cpp`

What it does:

- Lines 1-20 explain that this is the local LFortran override of the upstream
  MLIR C API implementation.
- Lines 97-118 initialize `UpstreamCtx`, registering dialects such as:
  - arith;
  - affine;
  - func;
  - gpu;
  - index;
  - scf;
  - cf;
  - memref;
  - tensor;
  - vector;
  - LLVM;
  - linalg;
  - math.
- Lines 124-130 define `ValueBox`.
- Lines 137-166 maintain side maps for user attributes and by-name attributes.
- Lines 190-211 map operation names to upstream operation types.
- Lines 230-305 maintain a block operation iterator cache.
- Lines 312-319 manage lifecycle arena setup.
- Lines 325-360 implement region/block operations.
- Lines 1205-1275 implement attribute builders.
- Lines 1285-1320 begin LLVM type construction helpers.
- Lines 2002-2050 rewrite wasm memory intrinsics.
- Lines 2064-2092 lift wasm import/export attributes.
- Lines 2094-2115 implement vector-print prepass.
- Lines 2117-2192 implement upstream lowering to LLVM dialect.
- Lines 2194-2280 implement custom CFG-to-SCF handling for wasm.
- Lines 2290-2323 route `MLIR_LiftCfToScf`.
- Lines 2328-2388 implement optional wasm upstream lowering.
- Lines 2404-2435 implement upstream translation to LLVM IR.
- Lines 2444-2520 implement optional wasm object translation.
- Lines 2392-2400 and 2524-2532 provide stubs when wasm lowering is disabled.

Why:

The project already has an upstream MLIR integration under `src/mlir`.  The new
backend needs a local override that can:

- allow unregistered ASR dialect operations;
- support local region and block helpers;
- run vector-print prepasses;
- preserve LFortran-specific wasm attributes;
- expose an upstream comparison path for lowering and translation.

How it is selected:

`src/mlir/CMakeLists.txt` prefers the local file when present:

```text
src/mlir/lfortran/mlir_api_impl_upstream.cpp
  overrides
src/mlir/mlir_api_impl_upstream.cpp
```

How it is used:

The main `mlir-new` pipeline chooses native lowering by default.  If:

```console
USE_MLIR_Upstream=1
```

is set, `asr_to_mlir_new.cpp` selects upstream lowering/translation entrypoints
where applicable.  This allows side-by-side comparison while native lowering is
still under development.

## CoreC and Platform Support Files

The files under `src/mlir/lfortran/corec` provide hosted support code needed by
the C MLIR API runtime and generated C dialect implementation.  They are built
into `lfortran_corec` by `src/mlir/CMakeLists.txt`.

### `src/mlir/lfortran/corec/base/buddy.c`

What it does:

- Defines a buddy allocator over a hosted platform heap.
- Key anchors:
  - line 17: `struct buddy_block`
  - line 24: `struct list_head`
  - line 53: `add_memory`
  - line 80: `buddy_init`
  - line 110: `buddy_print_stats`
  - line 406: `buddy_alloc_order`
  - line 467: `buddy_alloc`
  - line 492: `buddy_free`

Why:

The MLIR C API layer and corec arena-style code need deterministic allocation
support outside of the original upstream environment.  The hosted buddy
allocator supplies that memory layer.

### `src/mlir/lfortran/corec/base/string.c`

What it does:

- Line 18 implements `str_to_cstr_copy`.
- Line 25 implements `str_eq`.

Why:

Generated and C-side APIs use the lightweight `string` struct.  C++ and platform
boundaries often need null-terminated C strings or string equality checks.

### `src/mlir/lfortran/corec/base/string.h`

What it does:

- Line 15 defines the `string` struct.
- Lines 25-26 declare string helpers.

Why:

The ASR dialect API uses `string` for symbol references and textual attributes.
This header keeps the type available to generated and handwritten C code.

### `src/mlir/lfortran/corec/platform/platform.h`

What it does:

- Declares platform heap functions:
  - `platform_heap_base`
  - `platform_heap_size`
  - `platform_heap_grow`
- Declares process and file operations:
  - `platform_exit`
  - `platform_fd_close`
  - `platform_fd_read`
  - `platform_fd_seek`
  - `platform_fd_tell`
- Declares argument access:
  - `platform_args_sizes_get`
  - `platform_args_get`
- Declares platform initialization:
  - `platform_init`
- Declares mmap helpers:
  - `platform_read_file_mmap`
  - `platform_file_unmap`

Why:

The core C layer wants a small platform API rather than using libc directly in
every file.  This makes the same C runtime build on Linux, macOS, and Windows.

### `src/mlir/lfortran/corec/platform/platform_linux.c`

What it does:

- Implements hosted Linux platform support.
- Key anchors:
  - line 124: syscall wrapper;
  - line 149: `platform_exit`;
  - line 156: heap initialization;
  - line 201: `platform_heap_grow`;
  - line 252: `platform_init`;
  - lines 320-345: file descriptor helpers;
  - line 356: args size query;
  - line 370: mmap file read;
  - line 437: unmap;
  - line 451: args copy;
  - line 500: `_start_c`.

Why:

This lets the C runtime run in a hosted Linux compiler process while preserving
the corec-style platform boundary.

### `src/mlir/lfortran/corec/platform/platform_macos.c`

What it does:

- Implements hosted macOS platform support.
- Key anchors:
  - line 63: heap initialization;
  - line 91: `platform_exit`;
  - line 108: `platform_heap_grow`;
  - line 146: `platform_init`;
  - lines 213-238: file descriptor helpers;
  - line 249: args size query;
  - line 276: mmap file read;
  - line 333: unmap;
  - line 360: `_start`.

Why:

The new MLIR C support layer is intended to build on macOS as well as Linux.

### `src/mlir/lfortran/corec/platform/platform_windows.c`

What it does:

- Implements hosted Windows platform support.
- Key anchors:
  - line 128: heap initialization;
  - line 150: `platform_heap_grow`;
  - line 202: `platform_exit`;
  - line 251: `platform_init`;
  - lines 322-383: file descriptor helpers;
  - lines 401-442: wide-string/UTF-8 argument handling helpers;
  - line 485: args size query;
  - line 516: mmap-style file read;
  - line 598: unmap;
  - line 635: `_start`.

Why:

The hosted corec platform layer must also cover Windows builds.  Argument
conversion is more involved because Windows process arguments are wide strings.

## ASR Dialect Storage and Runtime Examples

### Example 1: Assignment

Input:

```fortran
program main
  integer :: x
  x = 10
end program
```

Processing:

```text
ASR Variable(x)
  -> ASR_CreateVariableOp
  -> stored in Program.symtab region

ASR IntegerConstant(10)
  -> ASR_CreateIntegerConstantOp
  -> expression op

ASR Assignment(Var(x), IntegerConstant(10))
  -> ASR_CreateAssignmentOp
  -> stored in Program.body region

Verifier
  -> sees Var(x)
  -> confirms @x exists in current scope symbols

Lowering
  -> creates memref allocation for x
  -> stores constant 10 into x[0]
```

Why this shape:

The dialect keeps declarations and executable statements in different regions.
That mirrors the compiler's need to allocate storage before lowering body
statements.

### Example 2: If Statement

Input:

```fortran
if (x > 0) then
  print *, x
else
  print *, 0
end if
```

ASR dialect shape:

```text
asr.if
  test: asr.integer_compare
  body:
    asr.print @x
  orelse:
    asr.print 0
```

High-level MLIR shape:

```text
%cond = arith.cmpi sgt, %x, %zero : i32
cf.cond_br %cond, ^then, ^else
^then:
  vector.print %x : i32
  cf.br ^merge
^else:
  vector.print %zero : i32
  cf.br ^merge
^merge:
```

Why:

The ASR dialect remains structured enough for inspection, while the high-level
MLIR lowering chooses explicit CFG blocks that are easy for the native LLVM
lowering to handle.

### Example 3: Do Loop

Input:

```fortran
do i = 1, 5
  print *, i
end do
```

ASR dialect shape:

```text
asr.do_loop
  head:
    variable: @i
    start: 1
    end: 5
    increment: 1
  body:
    asr.print @i
```

High-level MLIR shape:

```text
store 1 into i
cf.br ^loop_header

^loop_header:
  load i
  compare i <= 5
  cf.cond_br %cond, ^loop_body, ^loop_exit

^loop_body:
  print i
  i = i + 1
  cf.br ^loop_header

^loop_exit:
```

Why:

This lowering does not need `scf.for` construction.  It can go directly to
`cf.br` and `cf.cond_br`, which the native LLVM dialect lowering already knows
how to rewrite.

### Example 4: Array Constant and Array Item

Input:

```fortran
integer :: a(3)
a = [1, 2, 3]
print *, a(2)
```

ASR dialect shape:

```text
asr.variable @a : memref<3xi32>
asr.array_constant elements:
  asr.integer_constant 1
  asr.integer_constant 2
  asr.integer_constant 3
asr.array_item @a indices:
  asr.array_index 2
```

High-level MLIR shape:

```text
%a = memref.alloca : memref<3xi32>
store 1 into %a[0]
store 2 into %a[1]
store 3 into %a[2]
%idx = arith.constant 1 : index
%v = memref.load %a[%idx] : memref<3xi32>
vector.print %v : i32
```

Why:

The ASR dialect uses Fortran-facing index values and explicit array item
structure.  The high-level lowering converts to zero-based memref indexes.

## More Code-Grounded Construct Examples

The examples below connect specific Fortran constructs to the concrete files and
functions that process them. They are intentionally small because each one is a
unit test shape for one lowering feature.

### Scalar Arithmetic Journey

Input:

```fortran
program main
  integer :: x
  x = (1 + 2) * 3
  print *, x
end program
```

ASR dialect emission:

```text
visit_IntegerConstant(1)
  -> ASR_CreateIntegerConstantOp
visit_IntegerConstant(2)
  -> ASR_CreateIntegerConstantOp
visit_IntegerBinOp(1 + 2)
  -> ASR_CreateIntegerBinOpOp(op=Add)
visit_IntegerConstant(3)
  -> ASR_CreateIntegerConstantOp
visit_IntegerBinOp((1 + 2) * 3)
  -> ASR_CreateIntegerBinOpOp(op=Mul)
visit_Assignment
  -> ASR_CreateAssignmentOp
```

Code grounding:

- `src/libasr/codegen/asr_to_asr_dialect.cpp` lines 1019-1028 emit
  `IntegerBinOp`.
- `src/libasr/codegen/asr_to_asr_dialect.cpp` lines 1059-1065 emit
  `IntegerConstant`.
- `src/libasr/codegen/asr_to_asr_dialect.cpp` lines 1765-1775 emit
  `Assignment`.
- `src/mlir/lfortran/asr_dialect_lowering_handlers.c` lines 602-671 lower
  expression values.
- `src/mlir/lfortran/asr_dialect_lowering_handlers.c` lines 231-240 build i32
  binary arithmetic operations.

High-level MLIR concept:

```text
%c1 = arith.constant 1 : i32
%c2 = arith.constant 2 : i32
%sum = arith.addi %c1, %c2 : i32
%c3 = arith.constant 3 : i32
%prod = arith.muli %sum, %c3 : i32
memref.store %prod, %x[%zero] : memref<1xi32>
```

The key point is that nested ASR expressions stay nested until high-level MLIR
lowering asks `lower_expr_value()` for an SSA value. At that point each nested
expression emits its own MLIR operation and returns a value handle.

### Comparison and If Journey

Input:

```fortran
program main
  integer :: x
  x = 5
  if (x > 0) then
    print *, x
  else
    print *, 0
  end if
end program
```

ASR dialect emission:

```text
visit_IntegerCompare
  -> ASR_CreateIntegerCompareOp
visit_If
  -> ASR_CreateIfOp with body and orelse regions
```

Code grounding:

- `src/libasr/codegen/asr_to_asr_dialect.cpp` lines 1048-1057 emit
  `IntegerCompare`.
- `src/libasr/codegen/asr_to_asr_dialect.cpp` lines 457-489 emit `If` regions.
- `src/mlir/lfortran/asr_dialect_lowering_handlers.c` lines 159-169 map compare
  predicates.
- `src/mlir/lfortran/asr_dialect_lowering_handlers.c` lines 674-691 lower
  integer comparisons to an i1 value.
- `src/mlir/lfortran/asr_dialect_lowering_handlers.c` lines 807-846 lower `If`
  to CFG blocks.

High-level MLIR concept:

```text
%loaded = memref.load %x[%zero] : memref<1xi32>
%c0 = arith.constant 0 : i32
%cond = arith.cmpi sgt, %loaded, %c0 : i32
cf.cond_br %cond, ^then, ^else
^then:
  vector.print %loaded : i32
  cf.br ^merge
^else:
  vector.print %c0 : i32
  cf.br ^merge
^merge:
```

This example is useful because it touches expression lowering, predicate mapping,
region lowering, CFG block creation, branch insertion, and print lowering.

### `error stop` Journey

Input:

```fortran
program main
  error stop
end program
```

ASR dialect emission:

```text
visit_ErrorStop
  -> generated ASR dialect op for error_stop
```

High-level MLIR lowering:

```text
ASR_LowerErrorStop
  -> func.return 1
```

Code grounding:

- `src/libasr/codegen/asr_to_asr_dialect.cpp` lines 1861-1867 emit
  `ErrorStop`.
- `src/mlir/lfortran/asr_dialect_lowering_handlers.c` lines 847-855 lower
  `ErrorStop`.

Why it matters:

This is the current simple failure-exit model. A normal implicit or explicit
return uses status `0`; `error stop` uses status `1`. That keeps process status
visible without implementing full Fortran stop-code semantics.

### `write(6, *)` Journey

Input:

```fortran
program main
  integer :: x
  x = 9
  write(6, *) x
end program
```

ASR dialect emission:

```text
visit_FileWrite
  -> check stdout-like unit
  -> flatten values
  -> ASR_CreateFileWriteOp or print-like value sequence
```

High-level MLIR lowering:

```text
ASR_LowerFileWrite
  -> lower_print_expr
  -> vector.print
```

LLVM dialect lowering:

```text
vector.print
  -> MLIR_LFortranTryLowerOp
  -> printf declaration and format global
  -> llvm.call @printf
```

Code grounding:

- `src/libasr/codegen/asr_to_asr_dialect.cpp` lines 388-455 handle `Print` and
  `FileWrite`.
- `src/mlir/lfortran/asr_dialect_lowering_handlers.c` lines 417-478 lower print
  expressions.
- `src/mlir/lfortran/asr_dialect_lowering_handlers.c` lines 480-486 lower
  `Print` and `FileWrite`.
- `src/mlir/lfortran/mlir_lower_lfortran_hooks.c` lines 281-370 lower
  `vector.print` to `printf`-style LLVM dialect.

Simplification:

Only stdout-like writing is modeled. This is enough for `write(6, *) x` and
similar examples, but not enough for formatted files, internal files, named
units, `iostat`, `iomsg`, or full Fortran I/O behavior.

### Array Element Assignment Journey

Input:

```fortran
program main
  integer :: a(3)
  a = [1, 2, 3]
  a(2) = 8
  print *, a(2)
end program
```

ASR dialect emission:

```text
ArrayConstant
  -> normalized elements region
ArrayItem
  -> array symbol plus index sequence
Assignment(ArrayItem, IntegerConstant)
  -> assignment op whose target is an array item
```

High-level MLIR lowering:

```text
ASR_LowerVariable
  -> %a = memref.alloca : memref<3xi32>
store_array_constant
  -> stores each constant into %a[0], %a[1], %a[2]
store_array_item
  -> convert Fortran index 2 to memref index 1
  -> memref.store 8, %a[%idx]
lower_array_item_value
  -> memref.load %a[%idx]
```

Code grounding:

- `src/libasr/codegen/asr_to_asr_dialect.cpp` lines 520-580 emit
  `ArrayConstructor`, `ArrayItem`, and `ArrayConstant`.
- `src/mlir/lfortran/asr_dialect_lowering_handlers.c` lines 182-195 convert
  Fortran indexes to memref indexes.
- `src/mlir/lfortran/asr_dialect_lowering_handlers.c` lines 271-320 lower array
  item load/store behavior.
- `src/mlir/lfortran/asr_dialect_lowering_handlers.c` lines 322-365 store array
  constructor/constant values.

Why this example matters:

It shows why `ArrayConstant.elements` was added as a synthetic layout field. The
lowerer needs a normal sequence of element operations, not a pointer into ASR
constant storage.

### Unsupported Function Call Journey

Input:

```fortran
integer function f()
  f = 1
end function

program main
  print *, f()
end program
```

Current expected behavior:

```text
ASR dialect representation may contain function-related nodes
high-level compile lowering should reject function lowering
```

Code grounding:

- `src/libasr/codegen/asr_to_asr_dialect.cpp` lines 2393-2506 emit `Function`.
- `src/mlir/generated/asr_dialect_api_generated.h` line 4667 contains
  `ASR_CreateFunctionOp`.
- `src/mlir/lfortran/asr_dialect_lowering_handlers.c` lines 738-740 make
  `ASR_LowerFunction` unsupported.

Why this is deliberate:

Function representation is useful for the ASR dialect dump and for testing the
schema, but compiling functions requires ABI, call, argument, return-value, and
nested symbol-scope handling. The current compile path is intentionally limited
to one top-level program lowered to `func.func @main() -> i32`.

### Multi-File Compile Journey

Command:

```console
lfortran --backend=mlir-new a.f90 b.f90 -o app
```

Driver behavior:

```text
for each input file:
  handle_mlir(arg_file, tmp_o, ..., backend="mlir-new", show_options)
  -> produce one temporary object
normal driver link path
  -> link temporary objects into final executable
```

Code grounding:

- `src/bin/lfortran.cpp` lines 2980-2990 route each file through `handle_mlir`
  when backend is `mlir-new`.

Boundary:

This driver path exists, but the current ASR dialect compile subset is still
single-program shaped. Multi-file support at the driver level does not imply that
modules, cross-file symbols, or all separate-compilation semantics are complete
inside the new backend.

## Simplifications and Current Boundaries

These are deliberate simplifications in the current diff.  They should be read
as prototype boundaries, not accidental omissions.

### 1. ASDL is source of truth, not MLIR ODS

The dialect schema is generated from `ASR.asdl`.  There is no full TableGen/ODS
ASR dialect definition in this diff.

Why:

LFortran already describes ASR in ASDL.  Reusing it avoids maintaining two
schemas while the prototype is changing quickly.

Tradeoff:

The MLIR operations are unregistered `asr.*` operations with generated C schema
metadata, not first-class upstream MLIR dialect ops.

### 2. Representation is broader than compilation

Many ASR nodes can be represented in the ASR dialect because the generator emits
schema and wrappers broadly.  Only `COMPILE_SUPPORTED_OPS` are wired into the
compile-lowering dispatch.

Why:

This makes `--show-mlir-asr-dialect` useful for more ASR shapes while keeping
`--backend=mlir-new -c` honest about unsupported lowering.

Example:

An unsupported ASR expression may still appear in an ASR dialect dump, but
compilation should fail through `ASR_LowerUnsupported` instead of generating
incorrect MLIR.

### 3. Symbol references are flat strings

`Var` stores a symbol reference string such as `@x`, not a raw ASR symbol
pointer.

Why:

Raw ASR pointers are not stable across the dialect boundary.  String symbol refs
are printable, verifiable, and independent of C++ object lifetime.

Tradeoff:

The current verifier uses a simple symbol-set model and fixed storage.  This is
not a complete nested symbol table implementation.

### 4. `ASR_MAX_SYMS` is fixed

`ASR_LoweringContext` has fixed symbol slots.

Why:

It is simple and deterministic for current examples.

Tradeoff:

Large programs or deeply nested scopes will need a dynamic symbol-table model.

### 5. Type lowering is narrow

`asr_to_asr_dialect.cpp` maps key scalar types and simple arrays:

- integer;
- logical;
- real;
- array/memref-like types.

Fallbacks exist for unsupported or unknown types.

Why:

The current compile path focuses on integer examples, simple arrays, control
flow, and printing.

Tradeoff:

Full Fortran type coverage is not implemented here.

### 6. Only the first program unit is lowered

`TranslationUnit` emission selects the first program unit and errors if no
program unit exists.

Why:

The prototype target is a single `main`-like program path.

Tradeoff:

Modules, multiple program units, and full separate compilation need additional
design.

### 7. Function lowering is represented but not compiled

The ASR dialect visitor has a handwritten `Function` emitter, and generated
files include function schema/wrappers.  The high-level MLIR lowering handler
currently returns unsupported for `Function`.

Why:

Function representation is needed for dialect completeness and symbol-table
shape, but function code generation requires call ABI, arguments, return values,
and nested symbol handling that are outside the current minimal compile subset.

### 8. Print is lowered through `vector.print`, then `printf`

The high-level ASR dialect lowering emits `vector.print`.  The LLVM lowering
hook rewrites it to `printf`.

Why:

`vector.print` is easy to inspect in high-level MLIR and avoids introducing a
custom runtime print operation in the ASR-to-high stage.

Tradeoff:

This is not full Fortran I/O.  Formatting, files, units, and advanced output are
mostly unsupported.

### 9. `FileWrite` only supports stdout-like output

`FileWrite` lowering is simplified to the same print path for unit/default
stdout-like cases.

Why:

That supports examples such as:

```fortran
write(6, *) x
```

without implementing full Fortran file I/O semantics.

### 10. Native LLVM IR translation is text-based

The native translator emits LLVM IR text, then the C++ `MLIRModule` path parses
that text into an `llvm::Module`.

Why:

This is a compact integration path that reuses LFortran's existing object-file
generation.

Tradeoff:

Bad textual IR is caught later during parse/verify.  A direct LLVM IR builder
would be more structured, but much larger.

### 11. Static MLIR linking is broad

`LFortranMLIRLink.cmake` prefers all static `libMLIR*.a` archives when found.

Why:

The new bridge uses a moving set of MLIR dialects, passes, and translations.
Broad static linking avoids repeatedly chasing archive-order failures.

Tradeoff:

The link is heavier than a minimal library list.

## File-by-File Change Details

### `CMakeLists.txt`

Purpose:

Top-level build integration for MLIR link libraries and optional wasm lowering.

Main additions:

- `LFORTRAN_MLIR_WASM_LOWERING` option at lines 156-157.
- Conditional WebAssembly component handling at lines 282-291.
- Inclusion of `LFortranMLIRLink` and use of `${LFORTRAN_MLIR_LINK_LIBS}` at
  lines 308-311.

Why needed:

The new backend links both the native C MLIR support code and an upstream MLIR
C++ shim.  That needs a more robust MLIR link setup than the previous minimal
list.

### `cmake/LFortranMLIRLink.cmake`

Purpose:

Centralize MLIR library discovery for `mlir-new`.

Main additions:

- locate the MLIR library directory;
- glob static MLIR archives;
- group static libraries on ELF-like linkers;
- provide a fallback named-library list.

Why needed:

The upstream shim uses many MLIR dialects and transformations.  Static archive
link order can otherwise fail unpredictably.

### `src/CMakeLists.txt`

Purpose:

Enable the new `src/mlir` support subtree.

Main additions:

- `add_subdirectory(mlir)` under `WITH_LLVM AND WITH_MLIR`.

Why needed:

The new generated dialect runtime and MLIR API bridge must be compiled before
`libasr` links against them.

### `src/bin/lfortran.cpp`

Purpose:

Driver-level integration for the new backend and dump stages.

Main additions:

- include `asr_to_mlir_new.h`;
- add `Backend::mlir_new`;
- route stage-specific show flags;
- call `FortranEvaluator::get_mlir_new`;
- support `-c`, executable link, and multi-file compile paths.

Why needed:

Users should access the backend through normal LFortran CLI flows, not through a
separate test executable.

### `src/bin/lfortran_command_line_parser.cpp`

Purpose:

Expose user-visible backend and dump flags.

Main additions:

- `--show-mlir-high-dialect`;
- `--show-mlir-asr-dialect`;
- `--show-mlir-llvm-dialect`;
- `mlir-new` in backend help.

Why needed:

The backend has multiple important internal stages.  Each stage needs a stable
inspection flag.

### `src/bin/lfortran_command_line_parser.h`

Purpose:

Store parser state for the new show flags.

Main additions:

- three booleans for ASR/high/LLVM MLIR stage dumps.

Why needed:

`lfortran.cpp` converts these booleans into `MlirNewRequest`.

### `src/lfortran/fortran_evaluator.cpp`

Purpose:

Evaluator integration for the backend.

Main additions:

- `get_mlir_new`;
- call `asr_to_mlir_new`;
- convert LLVM IR text into an LLVM module for object/IR targets.

Why needed:

The evaluator is the right boundary between frontend ASR production and backend
module/object production.

### `src/lfortran/fortran_evaluator.h`

Purpose:

Expose the evaluator entrypoint.

Main additions:

- declaration of `get_mlir_new`.

Why needed:

The driver needs a public evaluator method for the new backend path.

### `src/libasr/CMakeLists.txt`

Purpose:

Compile and link new backend C++ sources.

Main additions:

- `asr_to_mlir_new.cpp`;
- `asr_to_asr_dialect.cpp`;
- link to `lfortran_mlir_c_api`;
- link to `lfortran_asr_dialect`.

Why needed:

The C++ pipeline entrypoints live in `libasr`, while runtime dialect processing
lives in `src/mlir/lfortran`.

### `src/libasr/asdl_to_asr_dialect.py`

Purpose:

Generate ASR dialect schema, wrappers, accessors, printer policy, storage
tables, and lowering dispatch from `ASR.asdl`.

Main additions:

- ASDL parsing and op classification;
- compile-supported op list;
- storage policy merge;
- schema/accessor/API generation;
- visitor section patching;
- generated lowering dispatch;
- `--check` mode.

Why needed:

The ASR dialect has many node kinds.  Generation keeps the dialect in sync with
`ASR.asdl` and keeps handwritten code focused on policy and lowering.

### `src/libasr/codegen/asr_to_asr_dialect.cpp`

Purpose:

Serialize C++ ASR into ASR dialect MLIR operations.

Main additions:

- module creation;
- type conversion;
- expression and statement sequence emission;
- handwritten structural nodes;
- generated visitors for many ASR nodes;
- top-level `TranslationUnit`/`Program` handling.

Why needed:

This is the first actual backend conversion step.  It gives the MLIR side a
complete operation tree to verify, print, and lower.

### `src/libasr/codegen/asr_to_asr_dialect.h`

Purpose:

Declare the ASR dialect visitor and its helpers.

Main additions:

- visitor state;
- scope-region model;
- emit helpers;
- handwritten visitor declarations;
- generated visitor declarations.

Why needed:

The C++ visitor needs a stable shape that the generator can patch without
overwriting handwritten policy code.

### `src/libasr/codegen/asr_to_mlir_new.cpp`

Purpose:

Own the public `mlir-new` pipeline.

Main additions:

- CLI/backend request mapping;
- ASR dialect build;
- ASR dialect verify and dump;
- ASR dialect to high-level MLIR lowering;
- high-level MLIR dump;
- high-level MLIR to LLVM dialect lowering;
- LLVM dialect dump;
- LLVM dialect to LLVM IR translation.

Why needed:

This file coordinates all individual layers into one user-facing backend.

### `src/libasr/codegen/asr_to_mlir_new.h`

Purpose:

Public backend request and entrypoint declarations.

Main additions:

- `MlirNewPipelineTarget`;
- `MlirShowOptions`;
- `MlirNewRequest`;
- helper and conversion function declarations.

Why needed:

The driver and evaluator need a small C++ contract without depending on C API
implementation details.

### `src/libasr/codegen/evaluator.cpp`

Purpose:

Store and consume new MLIR/LLVM stage strings.

Main additions:

- string-backed `MLIRModule` constructor;
- stage dump getters;
- parse `llvm_ir_from_mlir_api` into an LLVM module.

Why needed:

The new backend crosses a text boundary for dumps and LLVM IR translation.

### `src/libasr/codegen/evaluator.h`

Purpose:

Extend `MLIRModule` state.

Main additions:

- ASR dialect text;
- high MLIR text;
- LLVM dialect text;
- LLVM IR text;
- getters.

Why needed:

Multiple stage dumps can coexist in one pipeline result.

### `src/mlir/CMakeLists.txt`

Purpose:

Build all new MLIR support libraries and generated files.

Main additions:

- source override helper;
- corec library;
- C MLIR API core;
- upstream shim;
- optional wasm lowering;
- ASR dialect generator custom command;
- `lfortran_asr_dialect` library.

Why needed:

This is the build root for the new MLIR support layer.

### `src/mlir/generated/asr_dialect_accessors.h`

Purpose:

Generated accessors for all ASR dialect fields.

Why needed:

Lowering, verification, and printing need field access without duplicating
storage layout rules.

### `src/mlir/generated/asr_dialect_api_generated.h`

Purpose:

Generated `ASR_Create*Op` wrappers.

Why needed:

C++ emission code can call named wrappers instead of manually constructing
generic field arrays.

### `src/mlir/generated/asr_dialect_enum_print.h`

Purpose:

Generated ASR enum value printers.

Why needed:

Human-readable dumps and diagnostics need enum names.

### `src/mlir/generated/asr_dialect_ids.h`

Purpose:

Generated operation and field ids.

Why needed:

C code can switch on compact ids and share ids across schema/accessor/runtime
code.

### `src/mlir/generated/asr_dialect_layout_fields.h`

Purpose:

Generated ids for synthetic layout fields.

Why needed:

Synthetic fields such as metadata and normalized array elements must participate
in the same storage system as ASDL fields.

### `src/mlir/generated/asr_dialect_lowering_dispatch.h`

Purpose:

Generated lowering dispatch for supported operation kinds.

Why needed:

Compile lowering stays explicit and unsupported ops fail cleanly.

### `src/mlir/generated/asr_dialect_print_policy.h`

Purpose:

Generated pretty-print policy.

Why needed:

The printer needs generated knowledge of field names, enums, and compact
printing rules.

### `src/mlir/generated/asr_dialect_schema.h`

Purpose:

Generated ASR dialect schema and name lookup.

Why needed:

Creation, verification, storage lookup, and fallback printing all depend on a
single generated schema table.

### `src/mlir/generated/asr_dialect_storage_merged.inc`

Purpose:

Generated merged field storage policy.

Why needed:

Default ASDL-derived storage and handwritten storage overrides need one runtime
table.

### `src/mlir/lfortran/asr_dialect_api.c`

Purpose:

Public API dispatch layer.

Why needed:

Keeps call sites stable while native implementation details live elsewhere.

### `src/mlir/lfortran/asr_dialect_api.h`

Purpose:

Public C API for ASR dialect creation, verification, printing, and lowering.

Why needed:

C++ `libasr` code needs a stable C boundary into the MLIR API layer.

### `src/mlir/lfortran/asr_dialect_api_native.c`

Purpose:

Native ASR dialect runtime implementation.

Why needed:

This file creates unregistered `asr.*` operations, attaches fields and regions,
verifies structure and symbols, builds diagnostics, and starts lowering.

### `src/mlir/lfortran/asr_dialect_format.h`

Purpose:

Deleted empty placeholder.

Why changed:

The actual formatting implementation now lives in
`asr_dialect_pretty_print.c`.

### `src/mlir/lfortran/asr_dialect_lowering_handlers.c`

Purpose:

Lower ASR dialect operations into high-level MLIR.

Why needed:

This is the semantic lowering pass from ASR-shaped operations to common MLIR
dialects.

### `src/mlir/lfortran/asr_dialect_pretty_print.c`

Purpose:

Print readable ASR dialect dumps and operation summaries.

Why needed:

Debugging and diagnostics require a readable form of the generated/native ASR
dialect tree.

### `src/mlir/lfortran/asr_dialect_storage.h`

Purpose:

Read stored ASR dialect fields.

Why needed:

Generated accessors and handwritten runtime code need consistent storage
decoding.

### `src/mlir/lfortran/asr_dialect_storage_policy.h`

Purpose:

Define handwritten storage overrides and structural region layout.

Why needed:

ASDL defaults are not enough for program/function bodies, symbol tables, loop
regions, if regions, and symbol references.

### `src/mlir/lfortran/corec/base/buddy.c`

Purpose:

Hosted buddy allocator for corec runtime support.

Why needed:

The C MLIR layer needs arena/platform allocation support.

### `src/mlir/lfortran/corec/base/string.c`

Purpose:

String helper implementation.

Why needed:

The C API uses lightweight `string` values for names and symbol references.

### `src/mlir/lfortran/corec/base/string.h`

Purpose:

String helper declarations.

Why needed:

Generated and handwritten C files share the same string representation.

### `src/mlir/lfortran/corec/platform/platform.h`

Purpose:

Platform abstraction declarations.

Why needed:

Corec runtime support needs a small cross-platform hosted API.

### `src/mlir/lfortran/corec/platform/platform_linux.c`

Purpose:

Linux hosted platform implementation.

Why needed:

Provides heap growth, args, file, mmap, and entry support for Linux.

### `src/mlir/lfortran/corec/platform/platform_macos.c`

Purpose:

macOS hosted platform implementation.

Why needed:

Provides equivalent hosted platform behavior for macOS.

### `src/mlir/lfortran/corec/platform/platform_windows.c`

Purpose:

Windows hosted platform implementation.

Why needed:

Provides hosted platform support and Windows argument conversion.

### `src/mlir/lfortran/mlir_api_impl_upstream.cpp`

Purpose:

Local upstream MLIR C++ bridge override.

Why needed:

The new backend needs local C API behavior, unregistered op support,
LFortran-specific prepasses, and optional upstream lowering/translation.

### `src/mlir/lfortran/mlir_lower_lfortran_hooks.c`

Purpose:

LFortran-specific lowering hooks for native LLVM dialect lowering.

Why needed:

Handles `vector.print`, simple memrefs, and index operations before generic
lowering.

### `src/mlir/lfortran/mlir_lower_lfortran_hooks.h`

Purpose:

Hook declarations.

Why needed:

`mlir_lower_to_llvm.c` calls these hooks before generic lowering.

### `src/mlir/lfortran/mlir_lower_to_llvm.c`

Purpose:

Native high-level MLIR to LLVM dialect lowering.

Why needed:

Provides a local lowering route for the prototype without requiring all upstream
pass machinery.

### `src/mlir/lfortran/mlir_translate_lfortran_hooks.c`

Purpose:

LFortran-specific LLVM IR translation helpers.

Why needed:

Handles `index` printing as `i64` and suppresses the first block label where
appropriate.

### `src/mlir/lfortran/mlir_translate_lfortran_hooks.h`

Purpose:

Translator hook declarations.

Why needed:

The native LLVM IR translator uses these helpers while emitting text.

### `src/mlir/lfortran/mlir_translate_to_llvm_ir.c`

Purpose:

Native LLVM dialect to LLVM IR text translator.

Why needed:

Produces textual LLVM IR that `MLIRModule::mlir_to_llvm()` can parse into an
LLVM module for the existing object-generation path.

## Practical Debugging Map

Use this map to decide which file owns a failure.

### CLI flag is not recognized

Look at:

- `src/bin/lfortran_command_line_parser.cpp`
- `src/bin/lfortran_command_line_parser.h`
- `src/bin/lfortran.cpp`

Example:

```console
lfortran --show-mlir-asr-dialect file.f90
```

If the flag is rejected, the parser change is incomplete.

### Backend name is rejected

Look at:

- `src/bin/lfortran.cpp` around backend parsing;
- `src/bin/lfortran_command_line_parser.cpp` help text.

Example:

```console
lfortran --backend=mlir-new file.f90
```

If the backend is rejected, the driver did not map the string to
`Backend::mlir_new`.

### ASR dialect dump is missing a node

Look at:

- `src/libasr/codegen/asr_to_asr_dialect.cpp`
- generated visitor sections in `asr_to_asr_dialect.cpp`
- `src/libasr/asdl_to_asr_dialect.py`
- `src/mlir/generated/asr_dialect_api_generated.h`

Example:

If an `IntegerBinOp` does not appear for `1 + 2`, check the generated visitor
for `visit_IntegerBinOp` and the wrapper `ASR_CreateIntegerBinOpOp`.

### ASR dialect verification fails

Look at:

- `src/mlir/lfortran/asr_dialect_api_native.c`
- `src/mlir/lfortran/asr_dialect_storage_policy.h`
- `src/mlir/generated/asr_dialect_schema.h`
- `src/mlir/generated/asr_dialect_storage_merged.inc`

Example:

If a `Var` references `@x` but `@x` is not in the program `symtab`, symbol
verification in `asr_dialect_api_native.c` should produce the failure.

### High-level MLIR lowering fails

Look at:

- `src/mlir/lfortran/asr_dialect_lowering_handlers.c`
- `src/mlir/generated/asr_dialect_lowering_dispatch.h`

Example:

If an unsupported `RealBinOp` reaches lowering, generated dispatch should route
it to `ASR_LowerUnsupported` unless a handler was intentionally added.

### LLVM dialect lowering fails

Look at:

- `src/mlir/lfortran/mlir_lower_to_llvm.c`
- `src/mlir/lfortran/mlir_lower_lfortran_hooks.c`

Example:

If `vector.print` survives into a later stage, the LFortran hook prepass or
dispatch probably did not run.

### LLVM IR parse fails

Look at:

- `src/mlir/lfortran/mlir_translate_to_llvm_ir.c`
- `src/mlir/lfortran/mlir_translate_lfortran_hooks.c`
- `src/libasr/codegen/evaluator.cpp`

Example:

If the generated text contains an MLIR `index` type in LLVM IR, the translator
hook that prints index as `i64` may not have handled that site.

### Static MLIR link fails

Look at:

- `CMakeLists.txt`
- `cmake/LFortranMLIRLink.cmake`
- `src/mlir/CMakeLists.txt`

Example:

If a symbol from an MLIR dialect is missing at link time, check whether the
static archive was included in `${LFORTRAN_MLIR_LINK_LIBS}` and whether archive
grouping was enabled for the platform.

## Summary of the Design Intent

The current diff adds an experimental but end-to-end backend path named
`mlir-new`.  Its main goal is not to optimize code yet.  Its goal is to make the
ASR-to-MLIR path explicit, inspectable, and generated from LFortran's existing
ASR schema.

The most important change is the new boundary:

```text
C++ ASR nodes
  -> generated/native ASR dialect
  -> verified and pretty-printed dialect tree
  -> high-level MLIR
  -> LLVM dialect
  -> LLVM IR text
```

That boundary gives developers a way to see and test each stage separately.  It
also makes unsupported cases visible.  Instead of silently lowering unknown ASR
nodes to placeholder LLVM, the generated dispatch and native runtime can report
where the unsupported operation entered the pipeline.

The largest simplification is that the ASR dialect is not a full upstream MLIR
ODS dialect yet.  It is an unregistered `asr.*` operation tree with generated C
schema metadata and handwritten runtime policy.  That is a reasonable prototype
shape because it keeps `ASR.asdl` as the source of truth while the lowering rules
are still being discovered.

