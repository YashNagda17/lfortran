# `3_lfortran/lfortran` — mlir-new backend

LFortran fork with **`--backend=mlir-new`**: **Initial ASR** → **high-level MLIR** → **`MLIR_LowerToLLVMDialect`** → **LLVM IR** → object code.

Repo-level docs: [`../../README.md`](../../README.md), [`../../description.md`](../../description.md), [`../../Implementation2.md`](../../Implementation2.md).

## Pipeline

```
Fortran
  → Initial ASR (get_asr2; get_mlir_new skips default passes)
  → asr_to_mlir_new.cpp  (func, memref, arith, cf, vector.print)
  → MLIR_LowerToLLVMDialect[*Upstream]
  → llvm-dialect MLIR
  → MLIR_TranslateModuleToLLVMIR[*Upstream]
  → llvm::Module → -c / link
```

### Lowering sources

| Component | Location |
|-----------|----------|
| ASR emitter | [`src/libasr/codegen/asr_to_mlir_new.cpp`](src/libasr/codegen/asr_to_mlir_new.cpp) |
| MLIR C API + passes | [`src/mlir/`](src/mlir/) (synced from [`../mlir/`](../mlir/)) |
| Hosted platform | [`src/mlir/corec/platform/`](src/mlir/corec/platform/) from [`../../lfortran/src/mlir/corec/platform/`](../../lfortran/src/mlir/corec/platform/) — **not** bare-metal [`../mlir/corec/platform/`](../mlir/corec/platform/) |

Native passes in `src/mlir/`: `mlir_lift_cf_to_scf.c`, `mlir_lower_to_llvm.c`, `mlir_translate_to_llvm_ir.c`, wasm pipeline, `mlir_api_impl_upstream.cpp`.

## Build

Requires conda **`mlir19`**. From this directory:

```bash
conda activate mlir19

# Clean configure (especially after switching from 2_lfortran):
rm -rf CMakeCache.txt CMakeFiles/
rm -rf src/runtime/CMakeCache.txt src/runtime/CMakeFiles/ src/runtime/build.ninja

./build0.sh
cmake . -GNinja \
  -DWITH_LLVM=yes \
  -DWITH_MLIR=yes \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
./build1.sh

export PATH="$(pwd)/src/bin:$PATH"
```

See also [`../../mlir_build.txt`](../../mlir_build.txt).

| Flag | Effect |
|------|--------|
| `-DWITH_LLVM=yes` | Required for mlir-new object emission |
| `-DWITH_MLIR=yes` | Builds `lfortran_mlir_c_api` + enables `--backend=mlir-new` |

## Environment

| Variable | Lowering | Translate |
|----------|----------|-----------|
| `USE_MLIR_Upstream=1` | `MLIR_LowerToLLVMDialectUpstream` | `MLIR_TranslateModuleToLLVMIRUpstream` |
| unset / `0` | `MLIR_LowerToLLVMDialect` (native C) | `MLIR_TranslateModuleToLLVMIR` |

`vector.print` is lowered to libc **`printf`** on both paths (upstream VectorToLLVM would otherwise emit `printI64`/`printNewline`, which are not in `lfortran_runtime`).

## CLI

| Command | Output |
|---------|--------|
| `--backend=mlir-new` | Full pipeline; compile / link |
| `--show-mlir` | High-level MLIR (`func`/`memref`/`arith`) |
| `--show-mlir-asr-dialect` | Same as `--show-mlir` for mlir-new |
| `--show-mlir-llvm-dialect` | MLIR after lowering to LLVM dialect |
| `--show-llvm-from-mlir` | LLVM IR text |
| `-c -o file.o` | Object file |

Examples:

```bash
lfortran program.f90 --backend=mlir-new
lfortran program.f90 --backend=mlir-new --show-mlir
lfortran program.f90 --backend=mlir-new --show-mlir-llvm-dialect
lfortran program.f90 --backend=mlir-new --show-llvm-from-mlir
lfortran -c program.f90 -o program.o --backend=mlir-new

USE_MLIR_Upstream=0 lfortran program.f90 --backend=mlir-new --show-llvm-from-mlir
USE_MLIR_Upstream=1 lfortran program.f90 --backend=mlir-new --show-llvm-from-mlir
```

## Codegen (asr_to_mlir_new)

| ASR | High-level MLIR |
|-----|-----------------|
| Integer locals | `memref<1xi32>`, `memref.alloca`, load/store at index 0 |
| Rank-1 integer arrays | `memref.alloca`, `arith` index + load/store |
| Integer binops | `arith.addi`, `subi`, `muli`, `divsi`, … |
| `do` / `while` | `cf.br`, `cf.cond_br` |
| Print | `vector.print` → `printf` after lowering |

**Platform:** `platform_init(0, nullptr)` via `std::call_once` in the emitter (corec buddy heap). CMake: `PLATFORM_SKIP_ENTRY` + `PLATFORM_HOSTED` on `lfortran_corec`.

## vs `2_lfortran`

[`../../2_lfortran/lfortran/`](../../2_lfortran/lfortran/) uses **translate-only** emission (LLVM dialect directly from ASR, no high-level lowering pass). **`3_lfortran`** is the full [`../mlir/`](../mlir/) pipeline from Initial ASR.

## Verification

```bash
lfortran --backend=mlir-new --show-mlir a.f90
lfortran --backend=mlir-new --show-mlir-llvm-dialect a.f90
USE_MLIR_Upstream=0 lfortran --backend=mlir-new --show-llvm-from-mlir a.f90
lfortran --backend=mlir-new -c a.f90 -o a.o
```
