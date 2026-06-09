# `mlir-new` Backend in This Checkout

This README explains the experimental `mlir-new` backend in this lfortran checkout. 

The description below is based on the current checkout, the last 13 commit
messages in this repo, and the current `working.md` notes.

## 1. What This Checkout Adds

This checkout adds a new backend path:

```console
lfortran --backend=mlir-new file.f90
```

The main idea is simple:

- keep the normal LFortran frontend;
- start from ASR;
- convert ASR into a new ASR dialect;
- lower that dialect step by step;
- let the user inspect every stage;
- still reuse the existing LLVM object-file path at the end.

In short:

```text
Fortran source
  -> AST
  -> ASR
  -> ASR dialect
  -> high-level MLIR
  -> LLVM dialect MLIR
  -> LLVM IR text
  -> llvm::Module
  -> object file / executable
```

## 2. Small Words First

If you are new to this backend, these are the only words you need first:

- `AST`: the parsed source tree.
- `ASR`: LFortran's typed compiler IR. Think of it as a cleaned-up program
  representation after semantic checks.
- `ASR dialect`: an MLIR-side tree that represents ASR nodes as `asr.*`
  operations.
- `high-level MLIR`: normal MLIR operations like `func`, `memref`, `arith`,
  `cf`, and `vector`.
- `LLVM dialect MLIR`: MLIR operations in the `llvm.*` dialect.
- `LLVM IR`: the textual LLVM form that `llc` and LLVM understand.

## 3. Quick Usage

These commands need a build with MLIR enabled.

Use this small example:

```fortran
program main
  integer :: x
  x = 1 + 2
  print *, x
end program
```

### Show the ASR dialect

```console
lfortran --backend=mlir-new --show-mlir-asr-dialect example.f90
```

Use this when you want to answer:

- Did the backend capture the ASR structure?
- Did variables, statements, and expressions become `asr.*` operations?

### Show high-level MLIR

```console
lfortran --backend=mlir-new --show-mlir-high-dialect example.f90
```

You can also use:

```console
lfortran --backend=mlir-new --show-mlir example.f90
```

Use this when you want to answer:

- Did variables become `memref.alloca`?
- Did arithmetic become `arith.*`?
- Did `print` become `vector.print`?

### Show LLVM dialect MLIR

```console
lfortran --backend=mlir-new --show-mlir-llvm-dialect example.f90
```

Use this when you want to answer:

- Did `func.func` become `llvm.func`?
- Did `memref.*` disappear?
- Did `vector.print` become `llvm.call @printf`?

### Show LLVM IR produced from MLIR

```console
lfortran --backend=mlir-new --show-llvm-from-mlir example.f90
```

Use this when you want to answer:

- Is the generated LLVM IR valid text?
- Does it look like normal LLVM IR?

### Produce an object file

```console
lfortran --backend=mlir-new -c example.f90 -o example.o
```

### Produce an executable

```console
lfortran --backend=mlir-new example.f90 -o example
./example
```

### Compare against the upstream MLIR path

```console
USE_MLIR_Upstream=1 lfortran --backend=mlir-new --show-mlir-llvm-dialect example.f90
```

This keeps the same frontend and ASR-dialect stages, but switches the later
lowering and translation stage to the upstream MLIR-backed path.

## 4. One Small Example Through the Pipeline

Take the same input:

```fortran
program main
  integer :: x
  x = 1 + 2
  print *, x
end program
```

### Step A: ASR dialect

Conceptually, the backend builds something like this:

```mlir
module {
  asr.translation_unit {
    asr.program @main {
      asr.symtab {
        asr.variable @x : !asr.integer<4>
      }
      asr.body {
        asr.assignment ...
        asr.print ...
        asr.return
      }
    }
  }
}
```

Simple meaning:

- declarations go into a symbol-table region;
- statements go into a body region;
- expressions become nested `asr.*` operations.

### Step B: High-level MLIR

Conceptually, the backend lowers that to something like this:

```mlir
module {
  func.func @main() -> i32 {
    %x = memref.alloca : memref<1xi32>
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %sum = arith.addi %c1, %c2 : i32
    %i0 = arith.constant 0 : index
    memref.store %sum, %x[%i0] : memref<1xi32>
    %loaded = memref.load %x[%i0] : memref<1xi32>
    vector.print %loaded : i32
    func.return ...
  }
}
```

Simple meaning:

- a scalar variable becomes `memref<1xi32>`;
- scalar assignment becomes store-to-index-zero;
- scalar read becomes load-from-index-zero;
- `print` becomes `vector.print` first.

### Step C: LLVM dialect MLIR

Conceptually, that becomes something like this:

```mlir
llvm.func @main() -> i32 {
  %buf = llvm.alloca ...
  %ptr = llvm.getelementptr ...
  llvm.store ...
  %val = llvm.load ...
  llvm.call @printf(...)
  llvm.return ...
}
```

Simple meaning:

- `func.func` becomes `llvm.func`;
- `memref` operations become `llvm.alloca`, `llvm.load`, `llvm.store`;
- `vector.print` becomes a `printf` call through an LFortran hook.

### Step D: LLVM IR text

Conceptually, the final text looks like this:

```llvm
declare i32 @printf(ptr, ...)

define i32 @main() {
entry:
  ...
  ret i32 0
}
```

The exact names can differ, but the shape should look familiar to anyone who
has seen LLVM IR before.

## 5. Main Lowering Pipelines

There are two ways to think about this backend.

### 5.1 Inspection pipeline

Use this when you want to stop at a chosen stage and inspect it.

```text
source -> AST -> ASR -> ASR dialect -> stop
source -> AST -> ASR -> ASR dialect -> high-level MLIR -> stop
source -> AST -> ASR -> ASR dialect -> high-level MLIR -> LLVM dialect -> stop
source -> AST -> ASR -> ASR dialect -> high-level MLIR -> LLVM dialect -> LLVM IR text -> stop
```

### 5.2 Compile pipeline

Use this when you want an object file or executable.

```text
source
  -> AST
  -> ASR
  -> ASR dialect
  -> high-level MLIR
  -> LLVM dialect MLIR
  -> LLVM IR text
  -> parsed llvm::Module
  -> object file
  -> executable
```

## 6. Stage-by-Stage File Map

| Stage | Main files | What happens |
| --- | --- | --- |
| CLI and request selection | `src/bin/lfortran_command_line_parser.cpp`, `src/bin/lfortran.cpp` | Adds `--backend=mlir-new` and the stage dump flags |
| Source to ASR | normal LFortran frontend | No new parser or semantic frontend was added |
| ASR to ASR dialect | `src/libasr/codegen/asr_to_mlir_new.cpp`, `src/libasr/codegen/asr_to_asr_dialect.cpp` | Visits C++ ASR nodes and creates `asr.*` operations |
| Generated dialect schema | `src/libasr/asdl_to_asr_dialect.py`, `src/mlir/generated/*` | Generates wrappers, schema tables, accessors, enum printers, dispatch tables |
| Verify and print ASR dialect | `src/mlir/lfortran/asr_dialect_api_native.c`, `src/mlir/lfortran/asr_dialect_pretty_print.c` | Verifies the dialect tree and prints readable dumps |
| ASR dialect to high-level MLIR | `src/mlir/lfortran/asr_dialect_lowering_handlers.c` | Lowers `asr.*` operations to `func`, `memref`, `arith`, `cf`, `vector` |
| High-level MLIR to LLVM dialect | `src/mlir/lfortran/mlir_lower_to_llvm.c`, `src/mlir/lfortran/mlir_lfortran_hooks.c` | Generic lowering plus LFortran-specific hook lowering |
| LLVM dialect to LLVM IR text | `src/mlir/lfortran/mlir_translate_to_llvm_ir.c` | Emits LLVM IR as text |
| Upstream comparison path | `src/mlir/lfortran/mlir_api_impl_upstream.cpp` | Uses upstream MLIR C++ APIs for lowering and translation |
| Final object-file path | `src/libasr/codegen/evaluator.cpp` | Parses the generated LLVM IR text back into an `llvm::Module` and reuses the existing object emission path |

## 7. Summary of Changes

### 7.1 Driver and user-facing changes

These files make the backend visible to the user:

- `src/bin/lfortran_command_line_parser.cpp`
- `src/bin/lfortran_command_line_parser.h`
- `src/bin/lfortran.cpp`
- `src/lfortran/fortran_evaluator.cpp`
- `src/lfortran/fortran_evaluator.h`

What changed:

- new backend name: `mlir-new`;
- new dump flags for each stage;
- object-file and executable flow now accepts `mlir-new`;
- evaluator entrypoint `get_mlir_new()` was added.

### 7.2 ASR dialect generation and emission

These files build the new ASR dialect layer:

- `src/libasr/asdl_to_asr_dialect.py`
- `src/libasr/codegen/asr_to_asr_dialect.cpp`
- `src/libasr/codegen/asr_to_asr_dialect.h`
- `src/mlir/generated/*`

What changed:

- `ASR.asdl` is reused as the source of truth;
- repetitive MLIR-side schema code is generated;
- handwritten C++ visits live ASR nodes and emits the new dialect.

### 7.3 ASR dialect runtime

These files make the ASR dialect usable after it has been emitted:

- `src/mlir/lfortran/asr_dialect_api.h`
- `src/mlir/lfortran/asr_dialect_api.c`
- `src/mlir/lfortran/asr_dialect_api_native.c`
- `src/mlir/lfortran/asr_dialect_storage.h`
- `src/mlir/lfortran/asr_dialect_storage_policy.h`
- `src/mlir/lfortran/asr_dialect_pretty_print.c`

What changed:

- native C API to create and inspect dialect operations;
- verification for structure and symbol references;
- pretty-printing for readable dumps;
- storage rules for where each field lives.

### 7.4 Lowering and translation

These files do the real backend work after the ASR dialect exists:

- `src/mlir/lfortran/asr_dialect_lowering_handlers.c`
- `src/mlir/lfortran/mlir_lower_to_llvm.c`
- `src/mlir/lfortran/mlir_lfortran_hooks.c`
- `src/mlir/lfortran/mlir_translate_to_llvm_ir.c`

What changed:

- ASR dialect lowers to normal MLIR;
- high-level MLIR lowers to LLVM dialect;
- LFortran-specific hook logic handles the special prototype cases;
- LLVM dialect is translated to LLVM IR text.

### 7.5 Upstream comparison bridge

Main file:

- `src/mlir/lfortran/mlir_api_impl_upstream.cpp`

What changed:

- this gives a second path for lowering and translation;
- it lets developers compare the small native path against upstream MLIR;
- it keeps the same public C API surface.

### 7.6 Build and platform support

These files make the backend build and link:

- `CMakeLists.txt`
- `cmake/LFortranMLIRLink.cmake`
- `src/CMakeLists.txt`
- `src/libasr/CMakeLists.txt`
- `src/mlir/CMakeLists.txt`
- `src/mlir/lfortran/corec/*`

What changed:

- MLIR is enabled in the build;
- static MLIR archives are linked safely;
- the new `src/mlir` subtree is built;
- hosted `corec` support was added for this MLIR use case.

## 8. Native Path vs Upstream Path

By default, the later lowering path is the native path in this checkout.

```text
ASR dialect
  -> high-level MLIR
  -> native LLVM-dialect lowering
  -> native LLVM IR text translation
```

If you set:

```console
USE_MLIR_Upstream=1
```

then the backend keeps the same frontend and ASR-dialect steps, but switches the
later lowering and translation part to the upstream MLIR-backed path.

This is useful for debugging because it helps answer a simple question:

- Is the problem in ASR-dialect generation or early lowering?
- Or is the problem in the native high-MLIR to LLVM path?

## 9. Current Limits

These are important. This backend is experimental and deliberately narrow.

- The ASR dialect is generated from `ASR.asdl`. It is not a full upstream MLIR
  ODS dialect yet.
- The backend can represent more ASR nodes than it can compile.
- Symbol references are stored as strings like `@x`, not as raw ASR pointers.
- The current lowering context uses a fixed symbol table size.
- Type support is still narrow. Integer and a small set of common cases are the
  main focus.
- Only the first program unit is lowered right now.
- Function nodes are represented in the dialect, but full function codegen is
  still incomplete.
- `print` is lowered through `vector.print`, and then later through `printf`.
- `write(6, *)`-style output is simplified to the same print path.
- The native LLVM IR translator emits text first, then the C++ side parses that
  text back into an `llvm::Module`.

## 10. Last 13 Commits, in Simple Words

Read from bottom to top, the work started with MLIR infrastructure and ended at
user-facing CLI wiring.

| Commit | Commit message | Plain meaning |
| --- | --- | --- |
| `885213953` | `mlir: register certik/mlir as git submodule at src/mlir/upstream` | Brought the upstream MLIR code into the repo as a submodule |
| `e358e9477` | `mlir: add hosted LFortran corec platform overrides` | Added hosted platform support so the MLIR side can run inside LFortran |
| `81d1f0880` | `asdl: add ASR dialect generator script` | Added the Python generator that turns `ASR.asdl` into dialect support code |
| `d39585cd6` | `asdl: add generated ASR dialect schema and tables` | Added generated schema tables, ids, and related metadata |
| `557bbf067` | `mlir: add ASR dialect storage policy and field readers` | Added rules for where dialect fields are stored and helpers to read them |
| `5d102ced2` | `mlir: make buddy_init idempotent for hosted MLIR use` | Made the hosted allocator safe to initialize more than once |
| `d210ad9a3` | `mlir: add ASR dialect C API and native create/verify backend` | Added the runtime C API for creating, verifying, and using ASR dialect operations |
| `fc06c202f` | `build: add MLIR static archive link helper` | Added build support for linking the MLIR libraries cleanly |
| `20c7228b7` | `build: enable MLIR in root and src CMake` | Turned on MLIR support in the main build system |
| `51a5e0c98` | `build: add corec and ASR dialect MLIR CMake targets` | Added concrete build targets for the new runtime and dialect pieces |
| `18f5c09db` | `libasr: wire ASR dialect libraries into codegen build` | Connected the new MLIR pieces to the existing codegen build |
| `2d4bec6df` | `codegen: add ASR-to-ASR-dialect visitor` | Added the C++ visitor that turns live ASR nodes into `asr.*` operations |
| `f2e43b68b` | `cli: wire mlir-new backend flags and MLIRModule stage dumps` | Exposed the new backend and stage dumps to the user-facing CLI |

## 11. Good Files to Read First

If you are new and want the shortest reading path, start here:

1. `src/bin/lfortran_command_line_parser.cpp`
   This shows the new flags.
2. `src/bin/lfortran.cpp`
   This shows how the driver chooses the `mlir-new` path.
3. `src/libasr/codegen/asr_to_mlir_new.cpp`
   This is the main pipeline entry.
4. `src/libasr/codegen/asr_to_asr_dialect.cpp`
   This shows how C++ ASR nodes become ASR dialect operations.
5. `src/mlir/lfortran/asr_dialect_lowering_handlers.c`
   This shows how the ASR dialect becomes high-level MLIR.
6. `src/mlir/lfortran/mlir_lfortran_hooks.c`
   This shows the special prototype cases in later lowering.
7. `src/mlir/lfortran/mlir_translate_to_llvm_ir.c`
   This shows how LLVM dialect becomes LLVM IR text.
8. `src/mlir/lfortran/mlir_api_impl_upstream.cpp`
   This shows the upstream comparison path.

## 12. Final Picture

The most important idea in this checkout is not optimization.

The most important idea is visibility.

This backend makes the ASR-to-MLIR path explicit and inspectable:

```text
C++ ASR nodes
  -> generated/native ASR dialect
  -> verified dialect tree
  -> high-level MLIR
  -> LLVM dialect MLIR
  -> LLVM IR text
```

That is why the backend is useful even in places where it is still incomplete.
It lets you see where a case is supported, where it stops, and which file owns
that stage.

Compilation Steps:
rm -rf CMakeCache.txt CMakeFiles/

# Reconfigure with Ninja generator
./build0.sh

cmake . -GNinja -DWITH_LLVM=yes -DWITH_MLIR=yes -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"

# Build with Ninja
./build1.sh
