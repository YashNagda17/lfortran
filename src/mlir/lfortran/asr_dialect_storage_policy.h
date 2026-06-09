// Checked-in ASR dialect field storage policy (reviewed source, not runtime-generated).
//
// ASR.asdl defines node/field shapes; this header records how each lowered field is
// stored in MLIR (attributes, operands, regions, symbol refs). The generator merges
// these overrides with default classification rules and emits schema/accessor tables.
//
// Prototype rule (see simplify.md):
//   single expr fields -> operands
//   expr/stmt/op/node sequences and structural nodes -> regions
//   scalars/enums/strings/symbol names -> attrs
#pragma once

#include <stddef.h>
#include <stdint.h>

typedef enum {
    ASR_STORAGE_ATTR,
    ASR_STORAGE_TYPE_ATTR,
    ASR_STORAGE_RESULT_TYPE,
    ASR_STORAGE_OPERAND,
    ASR_STORAGE_OPTIONAL_OPERAND,
    ASR_STORAGE_VARIADIC_OPERANDS,
    ASR_STORAGE_REGION,
    ASR_STORAGE_OPTIONAL_REGION,
    ASR_STORAGE_SYMBOL_REF_ATTR,
    ASR_STORAGE_SYMBOL_REF_ARRAY_ATTR,
    ASR_STORAGE_OMITTED,
} ASR_FieldStorageKind;

typedef enum {
    ASR_DIALECT_REGION_NONE,
    ASR_DIALECT_REGION_SYMTAB,
    ASR_DIALECT_REGION_METADATA,
    ASR_DIALECT_REGION_BODY,
    ASR_DIALECT_REGION_ORELSE,
    ASR_DIALECT_REGION_HEAD,
    ASR_DIALECT_REGION_ITEMS,
} ASR_DialectRegionKind;

typedef enum {
    ASR_DIALECT_REGION_ELEM_NONE,
    ASR_DIALECT_REGION_ELEM_ANY,
    ASR_DIALECT_REGION_ELEM_SYMBOL,
    ASR_DIALECT_REGION_ELEM_STMT,
    ASR_DIALECT_REGION_ELEM_EXPR,
    ASR_DIALECT_REGION_ELEM_METADATA,
} ASR_DialectRegionElementKind;

typedef enum {
    ASR_DIALECT_TYPE_INTEGER,
    ASR_DIALECT_TYPE_REAL,
    ASR_DIALECT_TYPE_LOGICAL,
    ASR_DIALECT_TYPE_COMPLEX,
    ASR_DIALECT_TYPE_CHARACTER,
    ASR_DIALECT_TYPE_ARRAY,
    ASR_DIALECT_TYPE_FUNCTION,
    ASR_DIALECT_TYPE_STRUCT,
    ASR_DIALECT_TYPE_POINTER,
} ASR_DialectTypeKind;

// Explicit storage overrides: (ASR op name, field name) -> storage policy.
// Unlisted fields use defaults merged in generated/asr_dialect_storage_merged.h.
#define ASR_FIELD_STORAGE_OVERRIDES(X) \
    X(Program, symtab, ASR_STORAGE_REGION, ASR_DIALECT_REGION_SYMTAB, ASR_DIALECT_REGION_ELEM_SYMBOL) \
    X(Program, metadata, ASR_STORAGE_REGION, ASR_DIALECT_REGION_METADATA, ASR_DIALECT_REGION_ELEM_METADATA) \
    X(Program, body, ASR_STORAGE_REGION, ASR_DIALECT_REGION_BODY, ASR_DIALECT_REGION_ELEM_STMT) \
    X(Program, start_name, ASR_STORAGE_OMITTED, ASR_DIALECT_REGION_NONE, ASR_DIALECT_REGION_ELEM_NONE) \
    X(Program, end_name, ASR_STORAGE_OMITTED, ASR_DIALECT_REGION_NONE, ASR_DIALECT_REGION_ELEM_NONE) \
    X(Function, symtab, ASR_STORAGE_REGION, ASR_DIALECT_REGION_SYMTAB, ASR_DIALECT_REGION_ELEM_SYMBOL) \
    X(Function, metadata, ASR_STORAGE_REGION, ASR_DIALECT_REGION_METADATA, ASR_DIALECT_REGION_ELEM_METADATA) \
    X(Function, body, ASR_STORAGE_REGION, ASR_DIALECT_REGION_BODY, ASR_DIALECT_REGION_ELEM_STMT) \
    X(Function, start_name, ASR_STORAGE_OMITTED, ASR_DIALECT_REGION_NONE, ASR_DIALECT_REGION_ELEM_NONE) \
    X(Function, end_name, ASR_STORAGE_OMITTED, ASR_DIALECT_REGION_NONE, ASR_DIALECT_REGION_ELEM_NONE) \
    X(TranslationUnit, symtab, ASR_STORAGE_OMITTED, ASR_DIALECT_REGION_NONE, ASR_DIALECT_REGION_ELEM_NONE) \
    X(TranslationUnit, items, ASR_STORAGE_REGION, ASR_DIALECT_REGION_ITEMS, ASR_DIALECT_REGION_ELEM_ANY) \
    X(DoLoop, head, ASR_STORAGE_REGION, ASR_DIALECT_REGION_HEAD, ASR_DIALECT_REGION_ELEM_ANY) \
    X(DoLoop, body, ASR_STORAGE_REGION, ASR_DIALECT_REGION_BODY, ASR_DIALECT_REGION_ELEM_STMT) \
    X(DoLoop, orelse, ASR_STORAGE_REGION, ASR_DIALECT_REGION_ORELSE, ASR_DIALECT_REGION_ELEM_STMT) \
    X(If, body, ASR_STORAGE_REGION, ASR_DIALECT_REGION_BODY, ASR_DIALECT_REGION_ELEM_STMT) \
    X(If, orelse, ASR_STORAGE_REGION, ASR_DIALECT_REGION_ORELSE, ASR_DIALECT_REGION_ELEM_STMT) \
    X(WhileLoop, test, ASR_STORAGE_OPERAND, ASR_DIALECT_REGION_NONE, ASR_DIALECT_REGION_ELEM_EXPR) \
    X(WhileLoop, body, ASR_STORAGE_REGION, ASR_DIALECT_REGION_BODY, ASR_DIALECT_REGION_ELEM_STMT) \
    X(WhileLoop, orelse, ASR_STORAGE_REGION, ASR_DIALECT_REGION_ORELSE, ASR_DIALECT_REGION_ELEM_STMT) \
    X(Assignment, target, ASR_STORAGE_OPERAND, ASR_DIALECT_REGION_NONE, ASR_DIALECT_REGION_ELEM_EXPR) \
    X(Assignment, value, ASR_STORAGE_OPERAND, ASR_DIALECT_REGION_NONE, ASR_DIALECT_REGION_ELEM_EXPR) \
    X(Assignment, overloaded, ASR_STORAGE_OPTIONAL_REGION, ASR_DIALECT_REGION_BODY, ASR_DIALECT_REGION_ELEM_STMT) \
    X(If, test, ASR_STORAGE_OPERAND, ASR_DIALECT_REGION_NONE, ASR_DIALECT_REGION_ELEM_EXPR) \
    X(Print, text, ASR_STORAGE_OPERAND, ASR_DIALECT_REGION_NONE, ASR_DIALECT_REGION_ELEM_EXPR) \
    X(Var, v, ASR_STORAGE_SYMBOL_REF_ATTR, ASR_DIALECT_REGION_NONE, ASR_DIALECT_REGION_ELEM_NONE) \
    X(Variable, parent_symtab, ASR_STORAGE_OMITTED, ASR_DIALECT_REGION_NONE, ASR_DIALECT_REGION_ELEM_NONE)

// Explicit region slot assignment: (op name, field name) -> region index.
#define ASR_REGION_LAYOUT_LIST(X) \
    X(Program, symtab, 0) \
    X(Program, metadata, 1) \
    X(Program, body, 2) \
    X(Function, symtab, 0) \
    X(Function, metadata, 1) \
    X(Function, body, 2) \
    X(TranslationUnit, items, 0) \
    X(DoLoop, head, 0) \
    X(DoLoop, body, 1) \
    X(DoLoop, orelse, 2) \
    X(If, body, 0) \
    X(If, orelse, 1) \
    X(WhileLoop, body, 0) \
    X(WhileLoop, orelse, 1) \
    X(Assignment, overloaded, 0)

// Scope-owning ops always carry a metadata region at index 1 (may be empty).
#define ASR_DIALECT_SCOPE_METADATA_REGION_INDEX 1
