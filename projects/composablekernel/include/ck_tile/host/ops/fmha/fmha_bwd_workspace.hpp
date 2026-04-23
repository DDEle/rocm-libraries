// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/host/hip_check_error.hpp"

#include <cstddef>
#include <stdexcept>
#include <hip/hip_runtime.h>

namespace ck_tile {

// Plain-data description of the FMHA backward workspace contract.
//
// All fields are runtime values so this struct can be filled by host-only
// drivers that do not (and should not) instantiate CK Tile pipeline templates.
// In-tree callers obtain a populated instance via make_fmha_bwd_workspace_spec
// (defined in the device template header).
//
// Field semantics mirror the previous template-parameterized FmhaBwdWorkspaceManager:
//   use_qr_qtr_dor_pipeline  - true if the QrQtrDor pipeline is used (no workspace at all)
//   is_group_mode            - true if group/varlen mode (vs batch mode)
//   is_deterministic         - true if deterministic kernel variant is used
//   has_mask                 - true if any attention mask is enabled at the call site
//   n0                       - BlockFmhaShape::kN0 from the dq_dk_dv pipeline
//   acc_element_size         - sizeof(AccDataType), used to size the dq_acc buffer
//   alignment                - workspace sub-buffer alignment (currently 16)
//   needs_zero_dq_acc        - whether the device dq_acc region must be zeroed
//                              before the kernel runs (atomic-add or skipped blocks)
struct fmha_bwd_workspace_spec
{
    bool   use_qr_qtr_dor_pipeline;
    bool   is_group_mode;
    bool   is_deterministic;
    bool   has_mask;
    int    n0;
    size_t acc_element_size;
    size_t alignment;
    bool   needs_zero_dq_acc;
};

// --- Workspace size / offset helpers ---------------------------------------
// All return values in bytes. Layout (offsets are cumulative):
//   [0)                                            nsplits[batch or 1]    (index_t)
//   [dq_acc_offsets_offset)  (group+det only)      dq_acc_offsets[batch]  (long_index_t)
//   [host_size = dq_acc_data_offset)               dq_acc[total_elements] (AccDataType)

CK_TILE_HOST size_t fmha_bwd_dq_acc_splits_size(const fmha_bwd_workspace_spec& spec, int batch)
{
    if(spec.use_qr_qtr_dor_pipeline)
        return 0;
    const auto elems = (spec.is_group_mode && spec.is_deterministic)
                           ? static_cast<size_t>(batch)
                           : static_cast<size_t>(1);
    return integer_least_multiple(sizeof(index_t) * elems, spec.alignment);
}

CK_TILE_HOST size_t fmha_bwd_dq_acc_offsets_size(const fmha_bwd_workspace_spec& spec, int batch)
{
    const auto elems = (spec.is_group_mode && spec.is_deterministic)
                           ? static_cast<size_t>(batch)
                           : static_cast<size_t>(0);
    return integer_least_multiple(sizeof(long_index_t) * elems, spec.alignment);
}

CK_TILE_HOST size_t fmha_bwd_workspace_host_size(const fmha_bwd_workspace_spec& spec, int batch)
{
    if(spec.use_qr_qtr_dor_pipeline)
        return 0;
    const size_t raw =
        fmha_bwd_dq_acc_splits_size(spec, batch) + fmha_bwd_dq_acc_offsets_size(spec, batch);
    // Pad to 4K so dq_acc buffer always starts on a page-aligned boundary.
    return integer_least_multiple(raw, static_cast<size_t>(4096));
}

// --- Workspace preparation -------------------------------------------------

// Fill the CPU-prepared portion of the workspace and return the size of the
// remaining device-side portion (dq_acc buffer) in bytes.
CK_TILE_HOST size_t prepare_fmha_bwd_workspace_host(const fmha_bwd_workspace_spec& spec,
                                                    void* cpu_ws,
                                                    int batch_size,
                                                    int hdim_q,
                                                    int nhead_q,
                                                    int seqlen_q,
                                                    int seqlen_k,
                                                    const int* seqstart_qs,
                                                    const int* seqstart_ks)
{
    if(spec.use_qr_qtr_dor_pipeline)
    {
        // QrQtrDor writes dq directly; no workspace is allocated so cpu_ws is nullptr.
        throw std::logic_error(
            "prepare_fmha_bwd_workspace_host: QrQtrDor pipeline does not use workspace");
    }
    if(spec.is_group_mode && (!seqstart_qs || !seqstart_ks))
    {
        throw std::runtime_error(
            "prepare_fmha_bwd_workspace_host: seqstart_qs and seqstart_ks are required for "
            "group mode");
    }

    const auto nsplits = reinterpret_cast<index_t*>(cpu_ws);
    // Guarded above: when we reach this point, use_qr_qtr_dor_pipeline is false, so the
    // splits buffer always exists. Compute its size with the spec's group/det flags.
    const size_t nsplits_count = (spec.is_group_mode && spec.is_deterministic)
                                     ? static_cast<size_t>(batch_size)
                                     : static_cast<size_t>(1);
    const size_t splits_bytes =
        integer_least_multiple(sizeof(index_t) * nsplits_count, spec.alignment);
    const auto offsets =
        reinterpret_cast<long_index_t*>(reinterpret_cast<char*>(cpu_ws) + splits_bytes);

    if(!spec.is_deterministic)
    {
        nsplits[0] = 1;
        if(!spec.is_group_mode)
            return spec.acc_element_size * static_cast<long_index_t>(batch_size) * nhead_q *
                   seqlen_q * hdim_q;
        else
            return spec.acc_element_size * static_cast<long_index_t>(nhead_q) *
                   seqstart_qs[batch_size] * hdim_q;
    }
    else if(spec.is_group_mode) // deterministic group mode
    {
        offsets[0] = 0;
        index_t i  = 0;
        for(; i < batch_size - 1; ++i)
        {
            nsplits[i] = integer_divide_ceil(seqstart_ks[i + 1] - seqstart_ks[i], spec.n0);
            offsets[i + 1] = offsets[i] + static_cast<long_index_t>(nhead_q) * nsplits[i] *
                                              (seqstart_qs[i + 1] - seqstart_qs[i]) * hdim_q;
        }
        nsplits[i] = integer_divide_ceil(seqstart_ks[i + 1] - seqstart_ks[i], spec.n0);
        return spec.acc_element_size *
               (offsets[i] + static_cast<long_index_t>(nhead_q) * nsplits[i] *
                                 (seqstart_qs[i + 1] - seqstart_qs[i]) * hdim_q);
    }
    else // deterministic non-group mode (kUsePersistent)
    {
        const index_t dqdqkdv_workers = static_cast<index_t>(get_num_cus());
        const index_t jobs_per_head   = integer_divide_ceil(seqlen_k, spec.n0);
        const index_t total_jobs      = batch_size * nhead_q * jobs_per_head;
        const index_t jobs_per_worker = integer_divide_ceil(total_jobs, dqdqkdv_workers);
        if(jobs_per_head % jobs_per_worker == 0)
            nsplits[0] = jobs_per_head / jobs_per_worker;
        else if(jobs_per_worker % jobs_per_head == 0)
            nsplits[0] = 1;
        else
            nsplits[0] = 1 + integer_divide_ceil(jobs_per_head - 1, jobs_per_worker);
        return spec.acc_element_size * static_cast<long_index_t>(batch_size) * nhead_q *
               nsplits[0] * seqlen_q * hdim_q;
    }
}

// Copy the host-prepared bytes into the device workspace and (optionally)
// zero the dq_acc region. Throws on HIP runtime failure via HIP_CHECK_ERROR.
CK_TILE_HOST void prepare_fmha_bwd_workspace_device(const fmha_bwd_workspace_spec& spec,
                                                    void* device_ws,
                                                    const void* host_ws,
                                                    size_t device_ws_size,
                                                    size_t host_ws_size)
{
    if(host_ws_size > 0)
        HIP_CHECK_ERROR(hipMemcpy(device_ws, host_ws, host_ws_size, hipMemcpyHostToDevice));
    if(spec.needs_zero_dq_acc)
        HIP_CHECK_ERROR(
            hipMemset(reinterpret_cast<char*>(device_ws) + host_ws_size, 0, device_ws_size));
}

// --- Factory ---------------------------------------------------------------
// Build a populated spec from a kernel struct (FmhaBwdDQDKDVKernel<...>).
//
// Template-only: the caller's translation unit must already have the kernel
// type's full definition in scope. This header intentionally does not include
// any ops/kernel header, so device-only translation units can use the rest of
// this file without dragging in HIP-template-heavy code paths.
template <typename Kernel>
CK_TILE_HOST constexpr fmha_bwd_workspace_spec make_fmha_bwd_workspace_spec()
{
    constexpr bool use_qr     = Kernel::kUseQrQtrDorPipeline;
    constexpr bool is_grp     = Kernel::kIsGroupMode;
    constexpr bool is_det     = Kernel::kIsDeterministic;
    constexpr bool has_msk    = Kernel::kHasMask;
    constexpr bool persistent = !use_qr && is_det && !is_grp;
    // non-deterministic and persistent kernels use atomic-add to write dq;
    // masked kernels may skip blocks so dq must be pre-zeroed.
    constexpr bool needs_zero = (persistent || !is_det) || has_msk;
    return fmha_bwd_workspace_spec{
        /*use_qr_qtr_dor_pipeline=*/use_qr,
        /*is_group_mode=*/is_grp,
        /*is_deterministic=*/is_det,
        /*has_mask=*/has_msk,
        /*n0=*/static_cast<int>(Kernel::FmhaPipeline::BlockFmhaShape::kN0),
        /*acc_element_size=*/sizeof(typename Kernel::AccDataType),
        /*alignment=*/static_cast<size_t>(16),
        /*needs_zero_dq_acc=*/needs_zero,
    };
}

} // namespace ck_tile
