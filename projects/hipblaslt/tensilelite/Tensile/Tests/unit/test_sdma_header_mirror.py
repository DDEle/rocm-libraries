#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Scrape test: pins the C++ SDMA packet header and the Python packet emitter
# to the SAME wire format -- op/sub_op constants, packet lengths, field
# widths, and field offsets (16/13/25/29).
#
# Scrapes text and fails loudly, never skips, if a pattern stops matching.
# The golden dword vectors live in test_sdma_packet_emitter.py /
# tests/SdmaPktSubwin_test.cpp; this pins the spec those goldens are
# instances of.
################################################################################

import ast
import os
import re

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

CPP_HEADER = os.path.join(TENSILE_ROOT, "client", "src", "SdmaPktSubwin.hpp")
PY_EMITTER = os.path.join(TENSILE_ROOT, "Tensile", "Components", "SdmaPacketEmitter.py")


# ---------------------------------------------------------------------------
# Scrapers: every one raises (never returns a default) if it finds nothing.
# ---------------------------------------------------------------------------

def _read(path):
    if not os.path.isfile(path):
        raise AssertionError(
            "%s is missing. This test exists to keep the C++ packet header and "
            "the Python emitter in lockstep; if one of them moved or was "
            "deleted, that is the failure, not a reason to skip." % path)
    with open(path) as f:
        return f.read()


def _cpp_constexprs(text):
    """{name: int} for every `constexpr unsigned int NAME = <int>;`."""
    return {m.group(1): int(m.group(2)) for m in
            re.finditer(r"constexpr\s+unsigned\s+int\s+(\w+)\s*=\s*(\d+)\s*;", text)}


def _cpp_packet_bodies(text):
    """Split the header into the COPY and ATOMIC struct bodies."""
    copy_at = text.index("SDMA_PKT_COPY_LINEAR_SUBWIN_TAG")
    atomic_at = text.index("SDMA_PKT_ATOMIC_TAG")
    assert copy_at < atomic_at, "expected COPY to be declared before ATOMIC"
    return text[copy_at:atomic_at], text[atomic_at:]


def _cpp_bitfields(body):
    """{union_name: [(field, width), ...]} in DECLARATION ORDER."""
    unions = {}
    pattern = re.compile(
        r"struct\s*\{(?P<fields>.*?)\}\s*;\s*"
        r"unsigned\s+int\s+DW_\d+_DATA\s*;\s*\}\s*(?P<name>\w+)\s*;",
        re.S)
    for m in pattern.finditer(body):
        fields = re.findall(r"unsigned\s+int\s+(\w+)\s*:\s*(\d+)\s*;", m.group("fields"))
        unions[m.group("name")] = [(n, int(w)) for n, w in fields]
    if not unions:
        raise AssertionError(
            "found no bit-field unions in the C++ packet body -- the "
            "declaration shape this test scrapes has changed, so the test is "
            "no longer checking anything. Fix the scraper, do not delete it.")
    return unions


def _width(unions, union_name, field):
    for name, width in unions.get(union_name, []):
        if name == field:
            return width
    raise AssertionError("no bit-field %s in union %s (found: %s)"
                         % (field, union_name, unions.get(union_name)))


def _offset(unions, union_name, field):
    """Bit offset of `field` == sum of the widths declared before it."""
    offset = 0
    for name, width in unions.get(union_name, []):
        if name == field:
            return offset
        offset += width
    raise AssertionError("no bit-field %s in union %s (found: %s)"
                         % (field, union_name, unions.get(union_name)))


def _cpp_packet_dwords(text, struct_name):
    """The N in `static_assert(sizeof(<struct>) == N * sizeof(unsigned int)`."""
    m = re.search(r"sizeof\(%s\)\s*==\s*(\d+)\s*\*\s*sizeof" % re.escape(struct_name), text)
    if not m:
        raise AssertionError("no packet-size static_assert for %s" % struct_name)
    return int(m.group(1))


def _py_constants(text):
    """{name: value} for module-level literal assignments, parsed via ast
    (not import)."""
    consts = {}
    for node in ast.parse(text).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            consts[target.id] = ast.literal_eval(node.value)
        except ValueError:
            pass  # computed expression (e.g. COPY_HEADER_DW0); not a pinned literal
    return consts


def _py_dword_exprs(text, func_name):
    """{index: source of the RHS} for every `dw[<index>] = ...` in `func_name`,
    scoped to one function and one dword."""
    for node in ast.walk(ast.parse(text)):
        if not (isinstance(node, ast.FunctionDef) and node.name == func_name):
            continue
        exprs = {}
        for stmt in node.body:
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
                continue
            target = stmt.targets[0]
            if not (isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "dw"):
                continue
            try:
                exprs[ast.literal_eval(target.slice)] = ast.unparse(stmt.value)
            except ValueError:
                pass  # non-literal index; not something we can pin
        if not exprs:
            raise AssertionError(
                "%s assigns no dw[<literal>] entries -- the encoder shape this "
                "test scrapes has changed. Fix the scraper, do not delete it."
                % func_name)
        return exprs
    raise AssertionError("no function %s in %s" % (func_name, PY_EMITTER))


def _assert_shifted_by(exprs, index, bits, what):
    src = exprs.get(index)
    assert src is not None, "no dw[%d] assignment to check for %s" % (index, what)
    assert re.search(r"<<\s*%d\b" % bits, src), (
        "dw[%d] (%s) must shift by %d to match the C++ bit-field layout, got: %s"
        % (index, what, bits, src))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cpp():
    text = _read(CPP_HEADER)
    copy_body, atomic_body = _cpp_packet_bodies(text)
    return dict(text=text,
                consts=_cpp_constexprs(text),
                copy=_cpp_bitfields(copy_body),
                atomic=_cpp_bitfields(atomic_body))


@pytest.fixture(scope="module")
def py():
    text = _read(PY_EMITTER)
    return dict(text=text, consts=_py_constants(text))


# ---------------------------------------------------------------------------
# 1. Opcode constants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "SDMA_OP_COPY_SUBWIN",
    "SDMA_SUBOP_COPY_LINEAR_RECT",
    "SDMA_OP_ATOMIC",
    "SDMA_ATOMIC_ADD_RTN_32",
])
def test_opcode_constants_match(cpp, py, name):
    assert name in cpp["consts"], "%s not declared in %s" % (name, CPP_HEADER)
    assert name in py["consts"], "%s not declared in %s" % (name, PY_EMITTER)
    assert cpp["consts"][name] == py["consts"][name], (
        "%s disagrees: C++ %d vs Python %d. These two values are put on the wire "
        "by different code paths; a mismatch means the assembly emits a packet "
        "the header does not describe."
        % (name, cpp["consts"][name], py["consts"][name]))


# ---------------------------------------------------------------------------
# 2. Packet lengths
# ---------------------------------------------------------------------------

def test_copy_packet_dword_count_matches(cpp, py):
    assert _cpp_packet_dwords(cpp["text"], "SDMA_PKT_COPY_LINEAR_SUBWIN") \
        == py["consts"]["COPY_PACKET_DWORDS"] == 13


def test_atomic_packet_dword_count_matches(cpp, py):
    assert _cpp_packet_dwords(cpp["text"], "SDMA_PKT_ATOMIC") \
        == py["consts"]["ATOMIC_PACKET_DWORDS"] == 8


# ---------------------------------------------------------------------------
# 3. Field widths
# ---------------------------------------------------------------------------
# Python collapses each family of same-width fields into ONE constant; the
# check is that every C++ member of the family has that width.

_XY_FIELDS = [("DW_3_UNION", "src_x"), ("DW_3_UNION", "src_y"),
              ("DW_8_UNION", "dst_x"), ("DW_8_UNION", "dst_y"),
              ("DW_11_UNION", "rect_x"), ("DW_11_UNION", "rect_y")]
_Z_FIELDS = [("DW_4_UNION", "src_z"), ("DW_9_UNION", "dst_z"),
             ("DW_12_UNION", "rect_z")]
_PITCH_FIELDS = [("DW_4_UNION", "src_pitch"), ("DW_9_UNION", "dst_pitch")]
_SLICE_FIELDS = [("DW_5_UNION", "src_slice_pitch"), ("DW_10_UNION", "dst_slice_pitch")]


@pytest.mark.parametrize("py_name,fields", [
    ("_XY_BITS", _XY_FIELDS),
    ("_Z_BITS", _Z_FIELDS),
    ("_PITCH_BITS", _PITCH_FIELDS),
    ("_SLICE_BITS", _SLICE_FIELDS),
], ids=["xy", "z", "pitch", "slice"])
def test_field_widths_match(cpp, py, py_name, fields):
    expected = py["consts"][py_name]
    for union_name, field in fields:
        got = _width(cpp["copy"], union_name, field)
        assert got == expected, (
            "%s.%s is %d bits in C++ but Python's %s says %d. The emitter masks "
            "to the Python width, so the narrower of the two silently truncates."
            % (union_name, field, got, py_name, expected))


# ---------------------------------------------------------------------------
# 4. Field offsets
# ---------------------------------------------------------------------------
# Each assertion recomputes an offset from the C++ bit-field declaration order
# and compares it to the literal the Python emitter shifts by.

def test_y_offset_is_16(cpp, py):
    for union_name, field in [("DW_3_UNION", "src_y"), ("DW_8_UNION", "dst_y"),
                              ("DW_11_UNION", "rect_y")]:
        got = _offset(cpp["copy"], union_name, field)
        assert got == 16, (
            "%s.%s sits at bit %d in C++, but the Python emitter shifts the y "
            "term by 16." % (union_name, field, got))
    exprs = _py_dword_exprs(py["text"], "encodeCopyDwords")
    for index, what in [(3, "src_x|src_y"), (8, "dst_x|dst_y"), (11, "rect_x|rect_y")]:
        _assert_shifted_by(exprs, index, 16, what)


def test_pitch_offset_is_13(cpp, py):
    for union_name, field in _PITCH_FIELDS:
        got = _offset(cpp["copy"], union_name, field)
        assert got == 13, (
            "%s.%s sits at bit %d in C++, but the Python emitter shifts pitch "
            "by 13." % (union_name, field, got))
    exprs = _py_dword_exprs(py["text"], "encodeCopyDwords")
    for index, what in [(4, "src_pitch-1"), (9, "dst_pitch-1")]:
        _assert_shifted_by(exprs, index, 13, what)


def test_copy_header_elementsize_offset_is_29(cpp, py):
    assert _offset(cpp["copy"], "HEADER_UNION", "elementsize") == 29
    _assert_shifted_by(_py_dword_exprs(py["text"], "encodeCopyDwords"),
                       0, 29, "elementsize")


def test_atomic_header_operation_offset_is_25(cpp, py):
    assert _offset(cpp["atomic"], "HEADER_UNION", "operation") == 25
    _assert_shifted_by(_py_dword_exprs(py["text"], "encodeAtomicDwords"),
                       0, 25, "ADD_RTN_32 operation code")


# ---------------------------------------------------------------------------
# 5. The scrapers themselves found something
# ---------------------------------------------------------------------------
# Guards against a declaration-shape refactor making the regexes above match
# nothing, so every test up to this point would pass vacuously.

def test_scrape_actually_found_the_declarations(cpp, py):
    assert len(cpp["consts"]) >= 4, cpp["consts"]
    assert len(cpp["copy"]) >= 6, list(cpp["copy"])
    assert "HEADER_UNION" in cpp["atomic"]
    for name in ("_XY_BITS", "_Z_BITS", "_PITCH_BITS", "_SLICE_BITS",
                 "COPY_PACKET_DWORDS", "ATOMIC_PACKET_DWORDS"):
        assert name in py["consts"], "%s vanished from the Python emitter" % name


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
