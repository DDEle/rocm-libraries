# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Unit test for tensilelite_sanitize_hsakmt_link_interface().
#
# Runs standalone: `cmake -P test_hsakmt_link_interface.cmake`. Needs no ROCm,
# no compiler and no configured build tree.
#
# The cases below regression-test the CMAKE_MATCH_1 aliasing bug described in
# HsakmtLinkInterface.cmake.

cmake_minimum_required(VERSION 3.16)

get_filename_component(_here "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
include("${_here}/../HsakmtLinkInterface.cmake")

# Real directories, so the IS_DIRECTORY checks exercise the real filesystem
# rather than a stub. LIVE and SYSDEPS are distinct existing dirs.
set(LIVE "${_here}")
get_filename_component(SYSDEPS "${_here}/.." ABSOLUTE)
set(DEAD "${_here}/__no_such_search_dir__")
set(DEAD_SYSDEPS "${_here}/__no_such_sysdeps_dir__")

if(NOT DEFINED TEST_TMP_DIR)
    if(DEFINED ENV{TMPDIR})
        set(TEST_TMP_DIR "$ENV{TMPDIR}/tensilelite_hsakmt_link_test")
    else()
        set(TEST_TMP_DIR "/tmp/tensilelite_hsakmt_link_test")
    endif()
endif()
file(REMOVE_RECURSE "${TEST_TMP_DIR}")
file(MAKE_DIRECTORY "${TEST_TMP_DIR}")
set(LIVE_LIBC "${TEST_TMP_DIR}/libc.so")
file(TOUCH "${LIVE_LIBC}")
set(DEAD_LIBC "${TEST_TMP_DIR}/__gone__/libc.so")

set(_failures 0)

# A macro, not a function: it has to bump _failures in the caller's scope.
macro(expect_eq _label _actual _expected)
    if("${_actual}" STREQUAL "${_expected}")
        message(STATUS "PASS  ${_label}")
    else()
        message(STATUS "FAIL  ${_label}")
        message(STATUS "        expected: [${_expected}]")
        message(STATUS "        actual:   [${_actual}]")
        math(EXPR _failures "${_failures} + 1")
    endif()
endmacro()

# --- the CMAKE_MATCH_1-aliasing regression cases -----------------------------

set(_in "-L${LIVE}" "$<LINK_ONLY:-ldrm>")
tensilelite_sanitize_hsakmt_link_interface(_out "${SYSDEPS}" _in)
expect_eq("a live -L search dir is left alone"
          "${_out}" "-L${LIVE};$<LINK_ONLY:-ldrm>")

set(_in "-L${DEAD}" "-L${LIVE}")
tensilelite_sanitize_hsakmt_link_interface(_out "${SYSDEPS}" _in)
expect_eq("dead then live: only the dead one is repointed"
          "${_out}" "-L${SYSDEPS};-L${LIVE}")

set(_in "-L${LIVE}" "-L${DEAD}")
tensilelite_sanitize_hsakmt_link_interface(_out "${SYSDEPS}" _in)
expect_eq("live then dead: only the dead one is repointed"
          "${_out}" "-L${LIVE};-L${SYSDEPS}")

# --- behaviour the existing workaround already had ---------------------------

set(_in "-L${DEAD}")
tensilelite_sanitize_hsakmt_link_interface(_out "${SYSDEPS}" _in)
expect_eq("a lone dead -L is repointed at the vendored dir"
          "${_out}" "-L${SYSDEPS}")

set(_in "${DEAD_LIBC}" "$<LINK_ONLY:pthread>")
tensilelite_sanitize_hsakmt_link_interface(_out "${SYSDEPS}" _in)
expect_eq("a nonexistent libc.so entry is dropped"
          "${_out}" "$<LINK_ONLY:pthread>")

set(_in "${LIVE_LIBC}" "$<LINK_ONLY:pthread>")
tensilelite_sanitize_hsakmt_link_interface(_out "${SYSDEPS}" _in)
expect_eq("an existing libc.so entry is kept"
          "${_out}" "${LIVE_LIBC};$<LINK_ONLY:pthread>")

set(_in "$<LINK_ONLY:pthread>" "$<LINK_ONLY:rt>" "/usr/lib/libnuma.so" "$<LINK_ONLY:dl>")
tensilelite_sanitize_hsakmt_link_interface(_out "${SYSDEPS}" _in)
expect_eq("entries that are neither libc nor -L pass through untouched"
          "${_out}" "$<LINK_ONLY:pthread>;$<LINK_ONLY:rt>;/usr/lib/libnuma.so;$<LINK_ONLY:dl>")

# --- the vendored-dir master gate --------------------------------------------

set(_in "-L${DEAD}" "$<LINK_ONLY:-ldrm>")
tensilelite_sanitize_hsakmt_link_interface(_out "${DEAD_SYSDEPS}" _in)
expect_eq("no vendored dir to repoint at: the -L is left as it was"
          "${_out}" "-L${DEAD};$<LINK_ONLY:-ldrm>")

# --- the shape ROCm 7.2.0 actually ships -------------------------------------
#
# Two -L entries, both pointing at the same live system dir that genuinely
# supplies -ldrm/-ldrm_amdgpu.

set(_in "-L${LIVE}" "$<LINK_ONLY:-ldrm>"
        "-L${LIVE}" "$<LINK_ONLY:-ldrm_amdgpu>"
        "$<LINK_ONLY:pthread>" "${LIVE_LIBC}")
tensilelite_sanitize_hsakmt_link_interface(_out "${SYSDEPS}" _in)
expect_eq("a fully live interface is returned unchanged"
          "${_out}"
          "-L${LIVE};$<LINK_ONLY:-ldrm>;-L${LIVE};$<LINK_ONLY:-ldrm_amdgpu>;$<LINK_ONLY:pthread>;${LIVE_LIBC}")

file(REMOVE_RECURSE "${TEST_TMP_DIR}")

if(_failures GREATER 0)
    message(FATAL_ERROR "${_failures} assertion(s) failed")
endif()
message(STATUS "all assertions passed")
