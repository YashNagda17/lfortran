#!/usr/bin/env python3
"""
Generate ASR MLIR dialect artifacts from ASR.asdl (Design A: 1 constructor -> 1 asr.* op).

Outputs:
  - asr_dialect_schema.inc
  - asr_dialect_api_generated.h
  - asr_dialect_ops.td.inc
  - asr_to_asr_dialect_visitor.inc
  - asr_dialect_lowering_dispatch.inc
  - asr_dialect_coverage.csv
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import asdl


# ASDL types that are simple enum sums (not dialect ops).
SIMPLE_SUM_NAMES = {
    "cast_kind", "storage_type", "access", "intent", "deftype", "presence",
    "pass_attr", "abi", "codimension_type", "enumtype", "array_physical_type",
    "string_physical_type", "string_length_kind", "binop", "reduction_op",
    "logicalbinop", "cmpop", "integerboz", "arraybound", "arraystorage",
    "string_format_kind", "omp_region_type", "map_type", "schedule_type",
}

# Categories that produce dialect ops (top-level sum names in ASR.asdl).
OP_CATEGORIES = {"unit", "symbol", "stmt", "expr", "ttype"}

# Product-type sums that also become ops (records used as node payloads).
PRODUCT_OP_CATEGORIES = {
    "dimension", "codimension", "alloc_arg", "attribute", "attribute_arg",
    "call_arg", "reduction_expr", "tbind", "array_index", "do_loop_head",
    "case_stmt", "type_stmt", "rank_stmt", "require_instantiation", "omp_clause",
}

# Focused families with hand-written lowering (coverage CSV).
LOWERED_OPS = {
    # Arithmetic
    "IntegerConstant", "IntegerBinOp", "IntegerCompare", "IntegerUnaryMinus",
    "IntegerBitNot", "RealConstant", "RealBinOp", "RealCompare", "RealUnaryMinus",
    "Cast", "Var", "LogicalConstant", "LogicalNot", "LogicalCompare",
    # Array
    "ArrayItem", "ArrayConstant", "ArrayConstructor", "ArraySize", "ArrayBound",
    "ArraySection", "ArrayTranspose", "ArrayReshape", "ArrayPhysicalCast",
    # Intrinsics
    "IntrinsicElementalFunction", "IntrinsicArrayFunction", "IntrinsicImpureFunction",
    "IntrinsicImpureSubroutine",
    # Scaffolding
    "TranslationUnit", "Program", "Function", "Variable", "Assignment",
    "Return", "Print", "DoLoop", "If", "ErrorStop", "Allocate",
    # Types
    "Integer", "Real", "Logical", "Complex", "Array",
}

# Fields omitted from dialect op storage (handled by emitter policy).
SKIP_FIELDS = {"location", "symbol_table"}


def is_simple_sum(sum_node: asdl.Sum) -> bool:
    for constructor in sum_node.types:
        if constructor.fields:
            return False
    return True


def snake_case(name: str) -> str:
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def enum_kind(category: str, name: str) -> str:
    return f"ASR_DIALECT_OP_{category.upper()}_{name.upper()}"


def mlir_op_name(name: str) -> str:
    return f"asr.{snake_case(name)}"


CPP_KEYWORDS = {
    "inline", "static", "class", "virtual", "public", "private", "protected",
    "template", "operator", "new", "delete", "this", "friend", "register",
}


def c_param_name(name: str) -> str:
    if name in CPP_KEYWORDS:
        return name + "_"
    return name


def field_kind(asdl_type: str, seq: bool, opt: bool) -> str:
    if asdl_type in SIMPLE_SUM_NAMES:
        base = "I64"
    elif asdl_type in ("expr", "stmt", "symbol", "ttype", "node"):
        base = asdl_type.upper()
    elif asdl_type == "identifier":
        base = "IDENTIFIER"
    elif asdl_type == "string":
        base = "STRING"
    elif asdl_type == "int":
        base = "I64"
    elif asdl_type == "bool":
        base = "BOOL"
    elif asdl_type == "float":
        base = "F64"
    elif asdl_type == "void":
        base = "VOID"
    elif asdl_type.endswith("Type"):
        base = "I64"
    else:
        base = "OP"
    if seq:
        return f"ASR_FIELD_{base}_SEQ"
    if opt:
        return f"ASR_FIELD_{base}_OPT"
    return f"ASR_FIELD_{base}"


def gen_field_assign(i: int, f: asdl.Field, param: str) -> list[str]:
    fname = f.name or f.type
    fk = field_kind(f.type, f.seq, f.opt)
    lines = [
        f"    fields[{i}].kind = {fk};",
        f"    fields[{i}].name = \"{fname}\";",
    ]
    if f.type in ("identifier", "string"):
        lines.append(f"    fields[{i}].value.str = {param};")
    elif f.type == "bool":
        lines.append(f"    fields[{i}].value.b = {param};")
    elif f.type == "float":
        lines.append(f"    fields[{i}].value.f64 = {param};")
    elif f.type == "int" or is_enum_field(f.type) or f.type == "void":
        lines.append(f"    fields[{i}].value.i64 = {param};")
    elif f.type == "ttype":
        lines.append(f"    fields[{i}].value.type = {param};")
    else:
        lines.append(f"    fields[{i}].value.op = {param};")
    return lines


def collect_ops(mod: asdl.Module) -> list[dict]:
    ops = []
    for dfn in mod.dfns:
        cat = dfn.name
        val = dfn.value
        if cat in OP_CATEGORIES or cat in PRODUCT_OP_CATEGORIES:
            if isinstance(val, asdl.Sum):
                for cons in val.types:
                    ops.append({
                        "category": cat,
                        "name": cons.name,
                        "fields": cons.fields,
                        "kind": enum_kind(cat, cons.name),
                        "mlir_name": mlir_op_name(cons.name),
                    })
            elif isinstance(val, asdl.Product):
                ops.append({
                    "category": cat,
                    "name": cat,
                    "fields": val.fields,
                    "kind": enum_kind(cat, cat),
                    "mlir_name": mlir_op_name(cat),
                })
        elif isinstance(val, asdl.Sum) and is_simple_sum(val):
            pass  # enum, not op
        elif isinstance(val, asdl.Product):
            ops.append({
                "category": cat,
                "name": cat,
                "fields": val.fields,
                "kind": enum_kind(cat, cat),
                "mlir_name": mlir_op_name(cat),
            })
    ops.sort(key=lambda o: (o["category"], o["name"]))
    return ops


def collect_simple_enums(mod: asdl.Module) -> dict[str, list[str]]:
    enums = {}
    for dfn in mod.dfns:
        if dfn.name in SIMPLE_SUM_NAMES and isinstance(dfn.value, asdl.Sum):
            enums[dfn.name] = [c.name for c in dfn.value.types]
    return enums


def asdl_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gen_schema_inc(ops: list[dict], asdl_hash: str) -> str:
    lines = [
        "// Generated by asdl_to_asr_dialect.py — do not edit.",
        f"#define ASR_DIALECT_ASDL_SHA256 \"{asdl_hash}\"",
        "",
        "typedef enum {",
        "    ASR_DIALECT_OP_INVALID = 0,",
    ]
    for op in ops:
        lines.append(f"    {op['kind']},")
    lines.append("    ASR_DIALECT_OP_COUNT")
    lines.append("} ASR_DialectOpKind;")
    lines.append("")
    lines.append("typedef enum {")
    lines.append("    ASR_DIALECT_CATEGORY_UNIT,")
    lines.append("    ASR_DIALECT_CATEGORY_SYMBOL,")
    lines.append("    ASR_DIALECT_CATEGORY_STMT,")
    lines.append("    ASR_DIALECT_CATEGORY_EXPR,")
    lines.append("    ASR_DIALECT_CATEGORY_TTYPE,")
    lines.append("    ASR_DIALECT_CATEGORY_PRODUCT,")
    lines.append("} ASR_DialectCategory;")
    lines.append("")
    lines.append("typedef enum {")
    lines.append("    ASR_FIELD_I64, ASR_FIELD_F64, ASR_FIELD_BOOL,")
    lines.append("    ASR_FIELD_STRING, ASR_FIELD_IDENTIFIER,")
    lines.append("    ASR_FIELD_EXPR, ASR_FIELD_STMT, ASR_FIELD_SYMBOL,")
    lines.append("    ASR_FIELD_TTYPE, ASR_FIELD_NODE, ASR_FIELD_OP,")
    lines.append("    ASR_FIELD_VOID, ASR_FIELD_I64_SEQ, ASR_FIELD_F64_SEQ,")
    lines.append("    ASR_FIELD_BOOL_SEQ, ASR_FIELD_STRING_SEQ,")
    lines.append("    ASR_FIELD_IDENTIFIER_SEQ, ASR_FIELD_EXPR_SEQ,")
    lines.append("    ASR_FIELD_STMT_SEQ, ASR_FIELD_SYMBOL_SEQ,")
    lines.append("    ASR_FIELD_TTYPE_SEQ, ASR_FIELD_NODE_SEQ, ASR_FIELD_OP_SEQ,")
    lines.append("    ASR_FIELD_EXPR_OPT, ASR_FIELD_STMT_OPT, ASR_FIELD_SYMBOL_OPT,")
    lines.append("    ASR_FIELD_TTYPE_OPT, ASR_FIELD_NODE_OPT, ASR_FIELD_I64_OPT,")
    lines.append("    ASR_FIELD_F64_OPT, ASR_FIELD_STRING_OPT,")
    lines.append("    ASR_FIELD_IDENTIFIER_OPT, ASR_FIELD_BOOL_OPT,")
    lines.append("} ASR_DialectFieldKind;")
    lines.append("")
    lines.append("typedef enum { ASR_FIELD_REQUIRED, ASR_FIELD_OPTIONAL } ASR_FieldPresence;")
    lines.append("")
    lines.append("typedef struct {")
    lines.append("    const char *name;")
    lines.append("    ASR_DialectFieldKind kind;")
    lines.append("    ASR_FieldPresence presence;")
    lines.append("} ASR_DialectFieldDesc;")
    lines.append("")
    lines.append("typedef struct {")
    lines.append("    ASR_DialectOpKind kind;")
    lines.append("    ASR_DialectCategory category;")
    lines.append("    const char *asr_name;")
    lines.append("    const char *mlir_name;")
    lines.append("    size_t n_fields;")
    lines.append("    const ASR_DialectFieldDesc *fields;")
    lines.append("} ASR_DialectOpSchema;")
    lines.append("")

    cat_map = {
        "unit": "ASR_DIALECT_CATEGORY_UNIT",
        "symbol": "ASR_DIALECT_CATEGORY_SYMBOL",
        "stmt": "ASR_DIALECT_CATEGORY_STMT",
        "expr": "ASR_DIALECT_CATEGORY_EXPR",
        "ttype": "ASR_DIALECT_CATEGORY_TTYPE",
    }

    for op in ops:
        stored = [f for f in op["fields"] if f.name not in SKIP_FIELDS]
        lines.append(f"static const ASR_DialectFieldDesc {op['kind']}_FIELDS[] = {{")
        for f in stored:
            fk = field_kind(f.type, f.seq, f.opt)
            pres = "ASR_FIELD_OPTIONAL" if f.opt else "ASR_FIELD_REQUIRED"
            fname = f.name or f.type
            lines.append(f'    {{"{fname}", {fk}, {pres}}},')
        lines.append("};")
        lines.append("")

    lines.append("static const ASR_DialectOpSchema ASR_DIALECT_SCHEMA[] = {")
    for op in ops:
        stored = [f for f in op["fields"] if f.name not in SKIP_FIELDS]
        cat = cat_map.get(op["category"], "ASR_DIALECT_CATEGORY_PRODUCT")
        lines.append("    {")
        lines.append(f"        .kind = {op['kind']},")
        lines.append(f"        .category = {cat},")
        lines.append(f'        .asr_name = "{op["name"]}",')
        lines.append(f'        .mlir_name = "{op["mlir_name"]}",')
        lines.append(f"        .n_fields = {len(stored)},")
        lines.append(f"        .fields = {op['kind']}_FIELDS,")
        lines.append("    },")
    lines.append("};")
    lines.append("")
    lines.append("static const size_t ASR_DIALECT_SCHEMA_COUNT = sizeof(ASR_DIALECT_SCHEMA)"
                 " / sizeof(ASR_DIALECT_SCHEMA[0]);")
    lines.append("")
    return "\n".join(lines) + "\n"


def gen_api_generated_h(ops: list[dict]) -> str:
    lines = [
        "// Generated by asdl_to_asr_dialect.py — do not edit.",
        "// Include from C/C++ translation units only (not from asr_dialect_api.h).",
        "#pragma once",
        "",
        "#include \"asr_dialect_api.h\"",
        "",
    ]
    for op in ops:
        stored = [f for f in op["fields"] if f.name not in SKIP_FIELDS]
        params = ["MLIR_Context *ctx", "MLIR_LocationHandle loc"]
        for f in stored:
            fname = c_param_name(f.name or f.type)
            if f.type in ("expr", "stmt", "symbol", "node"):
                params.append(f"MLIR_OpHandle {fname}")
            elif f.type == "ttype":
                params.append(f"MLIR_TypeHandle {fname}")
            elif f.type in ("identifier", "string"):
                params.append(f"string {fname}")
            elif f.type == "int" or is_enum_field(f.type):
                params.append(f"int64_t {fname}")
            elif f.type == "bool":
                params.append(f"bool {fname}")
            elif f.type == "float":
                params.append(f"double {fname}")
            elif f.seq:
                params.append(f"MLIR_OpHandle *{fname}")
                params.append(f"size_t n_{fname}")
            else:
                params.append(f"MLIR_OpHandle {fname}")
        fn = f"ASR_Create{op['name']}Op"
        lines.append(f"static inline MLIR_OpHandle {fn}(")
        lines.append("        " + ",\n        ".join(params) + ") {")
        n = len(stored)
        if n == 0:
            lines.append("    (void)ctx; (void)loc;")
            lines.append(f"    return ASR_DialectCreateOp(ctx, {op['kind']}, loc, NULL, 0);")
        else:
            lines.append(f"    ASR_DialectField fields[{n}];")
            for i, f in enumerate(stored):
                param = c_param_name(f.name or f.type)
                lines.extend(gen_field_assign(i, f, param))
            lines.append(f"    return ASR_DialectCreateOp(ctx, {op['kind']}, loc, fields, {n});")
        lines.append("}")
        lines.append("")
    return "\n".join(lines)


def gen_ops_td_inc(ops: list[dict]) -> str:
    lines = ["// Generated by asdl_to_asr_dialect.py — do not edit.", ""]
    for op in ops:
        stored = [f for f in op["fields"] if f.name not in SKIP_FIELDS]
        op_class = f"ASR_{op['name']}Op"
        lines.append(f"def {op_class} : ASR_Op<\"{snake_case(op['name'])}\"> {{")
        lines.append(f'  let summary = "Generated from ASR {op["name"]}";')
        if stored:
            args = []
            for f in stored:
                fname = f.name or f.type
                suffix = ":$" + fname
                if f.type == "bool":
                    args.append(f"BoolAttr{suffix}")
                elif f.type == "int":
                    args.append(f"I64Attr{suffix}")
                elif f.type == "float":
                    args.append(f"F64Attr{suffix}")
                elif f.type in ("identifier", "string"):
                    args.append(f"StrAttr{suffix}")
                else:
                    args.append(f"ASR_Any{suffix}")
            lines.append("  let arguments = (ins " + ", ".join(args) + ");")
        lines.append("}")
        lines.append("")
    return "\n".join(lines)


def asr_member(field: asdl.Field) -> str:
    if field.name:
        return f"m_{field.name}"
    return field.type


def asr_count_member(field: asdl.Field) -> str:
    assert field.name
    return f"n_{field.name}"


def is_enum_field(asdl_type: str) -> bool:
    return asdl_type in SIMPLE_SUM_NAMES or asdl_type.endswith("Type")


def cpp_emit_field(field: asdl.Field) -> str:
    name = field.name
    asdl_type = field.type
    if name in SKIP_FIELDS:
        return ""
    fname = name or asdl_type
    member = asr_member(field)
    if asdl_type == "expr":
        if field.seq:
            count = asr_count_member(field)
            return (f"        size_t n_{fname} = x.{count};\n"
                    f"        MLIR_ValueHandle {fname} = emit_expr_seq_value(x.{member}, n_{fname});")
        if field.opt:
            return (f"        MLIR_ValueHandle {fname} = MLIR_INVALID_HANDLE;\n"
                    f"        if (x.{member}) {{ {fname} = emit_expr(*x.{member}); }}")
        return f"        MLIR_ValueHandle {fname} = emit_expr(*x.{member});"
    if asdl_type == "stmt":
        if field.seq:
            return f"        MLIR_ValueHandle {fname} = emit_stmt_seq_value(x.{member}, x.{asr_count_member(field)});"
        if field.opt:
            return (f"        MLIR_ValueHandle {fname} = MLIR_INVALID_HANDLE;\n"
                    f"        if (x.{member}) {{ {fname} = emit_stmt(*x.{member}); }}")
        return f"        MLIR_ValueHandle {fname} = emit_stmt(*x.{member});"
    if asdl_type == "symbol":
        if field.seq:
            return f"        string {fname} = emit_symbol_seq_ref(x.{member}, x.{asr_count_member(field)});"
        if field.opt:
            return (f"        string {fname} = str_lit(\"\");\n"
                    f"        if (x.{member}) {{ {fname} = emit_symbol_ref(*x.{member}); }}")
        return f"        string {fname} = emit_symbol_ref(*x.{member});"
    if asdl_type == "ttype":
        if field.seq:
            return f"        MLIR_TypeHandle {fname} = emit_type_seq_value(x.{member}, x.{asr_count_member(field)});"
        if field.opt:
            return (f"        MLIR_TypeHandle {fname} = MLIR_INVALID_HANDLE;\n"
                    f"        if (x.{member}) {{ {fname} = convert_type(*x.{member}); }}")
        return f"        MLIR_TypeHandle {fname} = convert_type(*x.{member});"
    if asdl_type == "identifier":
        if field.seq:
            return f"        string {fname} = emit_identifier_seq(x.{member}, x.{asr_count_member(field)});"
        return f"        string {fname} = asr_cstr(x.{member});"
    if asdl_type == "string":
        if field.opt:
            return (f"        string {fname} = str_lit(\"\");\n"
                    f"        if (x.{member}) {{ {fname} = asr_cstr(x.{member}); }}")
        return f"        string {fname} = asr_cstr(x.{member});"
    if asdl_type == "int":
        return f"        int64_t {fname} = x.{member};"
    if asdl_type == "bool":
        return f"        bool {fname} = x.{member};"
    if asdl_type == "float":
        return f"        double {fname} = x.{member};"
    if asdl_type == "void":
        return f"        int64_t {fname} = (int64_t)x.{member};"
    if is_enum_field(asdl_type):
        return f"        int64_t {fname} = (int64_t)x.{member};"
    if field.seq:
        return (f"        MLIR_ValueHandle {fname} = emit_product_seq_value("
                f"x.{member}, x.{asr_count_member(field)});")
    if field.opt:
        return (f"        MLIR_ValueHandle {fname} = MLIR_INVALID_HANDLE;\n"
                f"        if (x.{member}) {{ {fname} = emit_product_value(*x.{member}); }}")
    return f"        MLIR_ValueHandle {fname} = emit_product_value(*x.{member});"


def gen_visitor_inc(ops: list[dict]) -> str:
    lines = ["// Generated by asdl_to_asr_dialect.py — do not edit.", ""]
    for op in ops:
        if op["category"] not in OP_CATEGORIES:
            continue
        stored = [f for f in op["fields"] if f.name not in SKIP_FIELDS]
        lines.append(f"    void visit_{op['name']}(const ASR::{op['name']}_t &x) {{")
        for f in stored:
            code = cpp_emit_field(f)
            if code:
                lines.append(code)
        if op["category"] in ("expr", "ttype"):
            lines.append("        MLIR_LocationHandle op_loc = loc(x.base.base.loc);")
        elif op["category"] == "stmt":
            lines.append("        MLIR_LocationHandle op_loc = loc(x.base.loc);")
        elif op["category"] == "symbol":
            lines.append("        MLIR_LocationHandle op_loc = loc(x.base.base.loc);")
        else:
            lines.append("        MLIR_LocationHandle op_loc = default_loc();")
        args = []
        for f in stored:
            fname = f.name or f.type
            args.append(fname)
        fn = f"ASR_Create{op['name']}Op"
        if stored:
            lines.append(f"        last_value = {fn}(&ctx, op_loc, {', '.join(args)});")
        else:
            lines.append(f"        last_value = {fn}(&ctx, op_loc);")
        if op["category"] == "stmt":
            lines.append("        append_current_stmt(last_value);")
        lines.append("    }")
        lines.append("")
    return "\n".join(lines)


def gen_lowering_dispatch_inc(ops: list[dict]) -> str:
    lines = [
        "// Generated by asdl_to_asr_dialect.py — do not edit.",
        "",
        "bool ASR_DialectLowerOneOp(ASR_LoweringContext *ctx, MLIR_OpHandle op) {",
        "    ASR_DialectOpKind kind = ASR_DialectGetOpKind(op);",
        "    switch (kind) {",
    ]
    for op in ops:
        if op["name"] in LOWERED_OPS:
            handler = f"ASR_Lower{op['name']}"
            lines.append(f"    case {op['kind']}:")
            lines.append(f"        return {handler}(ctx, op);")
    lines.append("    default:")
    lines.append('        return ASR_LowerUnsupported(ctx, op, "ASR dialect op has no lowering yet");')
    lines.append("    }")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def gen_enum_print_inc(enums: dict[str, list[str]]) -> str:
    """C helpers: map semantic enum integers to keyword names for hybrid dump."""
    lines = [
        "// Generated by asdl_to_asr_dialect.py — do not edit.",
        "#pragma once",
        "",
        "#include <stdint.h>",
        "",
        "static inline const char *asr_enum_name_or_unknown(",
        "        const char *const *names, size_t n_names, int64_t v) {",
        "    if (v >= 0 && (size_t)v < n_names) {",
        "        return names[(size_t)v];",
        "    }",
        "    return NULL;",
        "}",
        "",
    ]
    for enum_name, variants in sorted(enums.items()):
        c_names = f"ASR_ENUM_{enum_name.upper()}_NAMES"
        kw = [snake_case(v) for v in variants]
        lines.append(f"static const char *const {c_names}[] = {{")
        for k in kw:
            lines.append(f'    "{k}",')
        lines.append("};")
        lines.append(f"static const size_t {c_names}_COUNT = {len(kw)};")
        lines.append("")
        fn = f"asr_enum_{snake_case(enum_name)}_name"
        lines.append(f"static inline const char *{fn}(int64_t v) {{")
        lines.append(
            f"    return asr_enum_name_or_unknown({c_names}, {c_names}_COUNT, v);")
        lines.append("}")
        lines.append("")
    return "\n".join(lines) + "\n"


def gen_coverage_csv(ops: list[dict]) -> str:
    lines = ["category,name,kind,mlir_name,generated,emitted,lowered,test"]
    for op in ops:
        lowered = "implemented" if op["name"] in LOWERED_OPS else "unsupported"
        test = "focus" if op["name"] in LOWERED_OPS else "no"
        lines.append(
            f"{op['category']},{op['name']},{op['kind']},{op['mlir_name']},"
            f"yes,yes,{lowered},{test}"
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    ops: list[dict],
    enums: dict[str, list[str]],
    asdl_hash: str,
    out_mlir_dir: Path,
    out_lfortran_dir: Path,
) -> None:
    out_mlir_dir.mkdir(parents=True, exist_ok=True)
    out_lfortran_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        out_mlir_dir / "asr_dialect_schema.inc": gen_schema_inc(ops, asdl_hash),
        out_mlir_dir / "asr_dialect_api_generated.h": gen_api_generated_h(ops),
        out_mlir_dir / "asr_dialect_ops.td.inc": gen_ops_td_inc(ops),
        out_mlir_dir / "asr_dialect_lowering_dispatch.inc": gen_lowering_dispatch_inc(ops),
        out_mlir_dir / "asr_dialect_enum_print.inc": gen_enum_print_inc(enums),
        out_mlir_dir / "asr_dialect_coverage.csv": gen_coverage_csv(ops),
        out_lfortran_dir / "asr_to_asr_dialect_visitor.inc": gen_visitor_inc(ops),
    }
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {path} ({len(content)} bytes)")


def check_outputs(
    ops: list[dict],
    enums: dict[str, list[str]],
    asdl_hash: str,
    out_mlir_dir: Path,
    out_lfortran_dir: Path,
) -> bool:
    expected = {
        out_mlir_dir / "asr_dialect_schema.inc": gen_schema_inc(ops, asdl_hash),
        out_mlir_dir / "asr_dialect_api_generated.h": gen_api_generated_h(ops),
        out_mlir_dir / "asr_dialect_ops.td.inc": gen_ops_td_inc(ops),
        out_mlir_dir / "asr_dialect_lowering_dispatch.inc": gen_lowering_dispatch_inc(ops),
        out_mlir_dir / "asr_dialect_enum_print.inc": gen_enum_print_inc(enums),
        out_mlir_dir / "asr_dialect_coverage.csv": gen_coverage_csv(ops),
        out_lfortran_dir / "asr_to_asr_dialect_visitor.inc": gen_visitor_inc(ops),
    }
    ok = True
    for path, content in expected.items():
        if not path.exists():
            print(f"MISSING: {path}", file=sys.stderr)
            ok = False
            continue
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            print(f"STALE: {path}", file=sys.stderr)
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asdl", type=Path, required=True)
    parser.add_argument("--out-mlir-dir", type=Path, required=True)
    parser.add_argument("--out-lfortran-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true",
                        help="Verify committed generated files match ASR.asdl")
    args = parser.parse_args()

    mod = asdl.parse(str(args.asdl))
    ops = collect_ops(mod)
    enums = collect_simple_enums(mod)
    asdl_hash = asdl_sha256(args.asdl)
    print(f"ASR.asdl SHA256: {asdl_hash}")
    print(f"Generated {len(ops)} dialect ops")

    if args.check:
        return 0 if check_outputs(ops, enums, asdl_hash, args.out_mlir_dir, args.out_lfortran_dir) else 1

    write_outputs(ops, enums, asdl_hash, args.out_mlir_dir, args.out_lfortran_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
