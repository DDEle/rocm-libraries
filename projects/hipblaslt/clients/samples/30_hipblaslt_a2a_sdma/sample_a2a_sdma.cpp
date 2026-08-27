// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Fused GEMM+A2A over the public API with a real SDMA queue: a single-rank
// (world 1) loopback that runs the A2A epilogue to completion and checks the
// recv buffer against local D.
//
// The fused epilogue submits SDMA packets unconditionally, so the queue set is
// mandatory.
//
// Needs a device library holding a FusedGemmA2A solution for this shape --
// tensilelite/Tensile/Tests/common/gemm/gfx950/fused_a2a_logic_disabled.yaml
// builds one. Run with:
//
//   HIP_VISIBLE_DEVICES=<free card> \
//   HIPBLASLT_TENSILE_LIBPATH=<devlib>/library/gfx950 ./sample_a2a_sdma
//
// Exits 0 and prints a reason when the machine or the loaded library cannot
// host the run.

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <SdmaQueue.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <vector>

namespace
{
    // kAM carries the same standing as M/N/K in solution selection, and must
    // satisfy kAM % W == 0, (kAM / W) % MT0 == 0, kAM % MT0 == 0 and kAM <= kM.
    // At MT0 256 that puts the smallest world-4 value at 1024.
    constexpr int64_t  kM        = 1024;
    constexpr int64_t  kN        = 1024;
    constexpr int64_t  kK        = 1024;
    constexpr int64_t  kAM       = 1024;
    constexpr uint32_t kWorld    = 1;
    constexpr uint32_t kChannels = 1;

    constexpr size_t kWorkspaceSize = 32ull * 1024 * 1024;

    constexpr int kStatusOk      = 0;
    constexpr int kStatusFailed  = 1;
    constexpr int kStatusSkipped = 2;

    uint16_t toBf16(float f)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &f, sizeof(bits));
        return uint16_t(bits >> 16);
    }

    float fromBf16(uint16_t h)
    {
        const uint32_t bits = uint32_t(h) << 16;
        float          f    = 0.0f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

    hipblasStatus_t memcpyAllgather(void*       userData,
                                    const void* sendbuf,
                                    void*       recvbuf,
                                    size_t      bytesPerRank)
    {
        static_cast<void>(userData);
        std::memcpy(recvbuf, sendbuf, bytesPerRank);
        return HIPBLAS_STATUS_SUCCESS;
    }

    bool deviceIsGfx950()
    {
        hipDeviceProp_t props{};
        if(hipGetDeviceProperties(&props, 0) != hipSuccess)
            return false;
        return std::strncmp(props.gcnArchName, "gfx950", 6) == 0;
    }
}

#define CHECK_HIP(expr)                                                       \
    do                                                                        \
    {                                                                         \
        const hipError_t _e = (expr);                                         \
        if(_e != hipSuccess)                                                  \
        {                                                                     \
            std::printf("failed: %s -> %s\n", #expr, hipGetErrorString(_e));  \
            return kStatusFailed;                                             \
        }                                                                     \
    } while(0)

#define CHECK_LT(expr)                                                        \
    do                                                                        \
    {                                                                         \
        const hipblasStatus_t _s = (expr);                                    \
        if(_s != HIPBLAS_STATUS_SUCCESS)                                      \
        {                                                                     \
            std::printf("failed: %s -> status %d\n", #expr, int(_s));         \
            return kStatusFailed;                                             \
        }                                                                     \
    } while(0)

int main()
{
    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount < 1)
    {
        std::printf("skipped: no HIP device visible\n");
        return kStatusOk;
    }
    if(!deviceIsGfx950())
    {
        std::printf("skipped: fused GEMM+A2A is wired for gfx950 only\n");
        return kStatusOk;
    }
    CHECK_HIP(hipSetDevice(0));

    // One loopback queue: entry j targets rank j, and j == this rank is loopback.
    std::vector<hipblasLtSdmaQueue_t>                            queueSet(kWorld);
    std::vector<std::unique_ptr<TensileLite::Client::SdmaQueue>> owned;
    try
    {
        const uint32_t node = TensileLite::Client::sdmaNodeIdForDevice(0);
        owned.reserve(kWorld);
        for(uint32_t j = 0; j < kWorld; ++j)
        {
            owned.push_back(std::make_unique<TensileLite::Client::SdmaQueue>(
                node, TensileLite::Client::sdmaSelectEngine(node, node)));
            const HsaQueueResource& r = owned.back()->queueResource();
            queueSet[j]               = {owned.back()->ringBase(),
                                         (void*)r.Queue_read_ptr_aql,
                                         (void*)r.Queue_write_ptr_aql,
                                         (void*)r.Queue_DoorBell_aql};
        }
    }
    catch(const std::exception& e)
    {
        std::printf("skipped: cannot create an SDMA queue (%s)\n", e.what());
        return kStatusOk;
    }

    hipblasLtHandle_t handle = nullptr;
    CHECK_LT(hipblasLtCreate(&handle));
    CHECK_LT(hipblasLtSetDeviceComm(handle, 0, kWorld, kChannels, memcpyAllgather, nullptr));

    // D[m][n] = ((m % 7) + 1) * ((n % 5) + 1), exact in bf16.
    std::vector<uint16_t> hA(size_t(kK) * kM, toBf16(0.0f));
    std::vector<uint16_t> hB(size_t(kK) * kN, toBf16(0.0f));
    for(int64_t m = 0; m < kM; ++m)
        hA[size_t(m) * kK] = toBf16(float((m % 7) + 1));
    for(int64_t n = 0; n < kN; ++n)
        hB[size_t(n) * kK] = toBf16(float((n % 5) + 1));

    const size_t bytesA    = hA.size() * sizeof(uint16_t);
    const size_t bytesB    = hB.size() * sizeof(uint16_t);
    const size_t bytesCD   = size_t(kM) * kN * sizeof(uint16_t);
    const size_t bytesRecv = size_t(kAM) * kN * sizeof(uint16_t);

    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr;
    void *dRecv = nullptr, *dWorkspace = nullptr;
    CHECK_HIP(hipMalloc(&dA, bytesA));
    CHECK_HIP(hipMalloc(&dB, bytesB));
    CHECK_HIP(hipMalloc(&dC, bytesCD));
    CHECK_HIP(hipMalloc(&dD, bytesCD));
    CHECK_HIP(hipMalloc(&dRecv, bytesRecv));
    CHECK_HIP(hipMalloc(&dWorkspace, kWorkspaceSize));
    CHECK_HIP(hipMemcpy(dA, hA.data(), bytesA, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, hB.data(), bytesB, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(dC, 0, bytesCD));
    CHECK_HIP(hipMemset(dD, 0, bytesCD));
    CHECK_HIP(hipMemset(dRecv, 0, bytesRecv));

    void* recvPtrs[kWorld] = {dRecv};

    hipblasLtFusedEpilogueDescriptor_t fused = nullptr;
    CHECK_LT(hipblasLtFusedEpilogueCreate(&fused));
    CHECK_LT(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_A2A_PREFIX));

    const void* queuePtr = queueSet.data();
    CHECK_LT(hipblasLtFusedEpilogueSetAttribute(
        fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_SDMA_QUEUES, &queuePtr, sizeof(queuePtr)));
    void* const* recvArg = recvPtrs;
    CHECK_LT(hipblasLtFusedEpilogueSetAttribute(
        fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_RECV_PTRS, &recvArg, sizeof(recvArg)));
    const int64_t extent = kAM;
    CHECK_LT(hipblasLtFusedEpilogueSetAttribute(
        fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_EXTENT, &extent, sizeof(extent)));
    const hipblasLtA2ACompletionMode_t mode = HIPBLASLT_A2A_COMPLETION_IN_KERNEL;
    CHECK_LT(hipblasLtFusedEpilogueSetAttribute(
        fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_COMPLETION_MODE, &mode, sizeof(mode)));
    const uint32_t channel = 0;
    CHECK_LT(hipblasLtFusedEpilogueSetAttribute(
        fused, HIPBLASLT_FUSED_EPILOGUE_COMM_CHANNEL, &channel, sizeof(channel)));

    hipblasLtMatrixLayout_t layA = nullptr, layB = nullptr, layC = nullptr, layD = nullptr;
    CHECK_LT(hipblasLtMatrixLayoutCreate(&layA, HIP_R_16BF, kK, kM, kK));
    CHECK_LT(hipblasLtMatrixLayoutCreate(&layB, HIP_R_16BF, kK, kN, kK));
    CHECK_LT(hipblasLtMatrixLayoutCreate(&layC, HIP_R_16BF, kM, kN, kM));
    CHECK_LT(hipblasLtMatrixLayoutCreate(&layD, HIP_R_16BF, kM, kN, kM));

    hipblasLtMatmulDesc_t mm = nullptr;
    CHECK_LT(hipblasLtMatmulDescCreate(&mm, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    const hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    CHECK_LT(hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    CHECK_LT(hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
    CHECK_LT(hipblasLtMatmulDescSetAttribute(
        mm, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE, &fused, sizeof(fused)));

    hipblasLtMatmulPreference_t pref = nullptr;
    CHECK_LT(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_LT(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceSize, sizeof(kWorkspaceSize)));

    hipblasLtMatmulHeuristicResult_t heur[1];
    int                              algoCount = 0;
    CHECK_LT(hipblasLtMatmulAlgoGetHeuristic(
        handle, mm, layA, layB, layC, layD, pref, 1, heur, &algoCount));
    if(algoCount == 0)
    {
        std::printf("skipped: loaded device library carries no fused GEMM+A2A solution\n");
        return kStatusOk;
    }

    const float alpha = 1.0f, beta = 0.0f;
    CHECK_LT(hipblasLtMatmul(handle,
                             mm,
                             &alpha,
                             dA,
                             layA,
                             dB,
                             layB,
                             &beta,
                             dC,
                             layC,
                             dD,
                             layD,
                             &heur[0].algo,
                             dWorkspace,
                             kWorkspaceSize,
                             nullptr));
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<uint16_t> hD(size_t(kM) * kN);
    std::vector<uint16_t> hRecv(size_t(kAM) * kN);
    CHECK_HIP(hipMemcpy(hD.data(), dD, bytesCD, hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(hRecv.data(), dRecv, bytesRecv, hipMemcpyDeviceToHost));

    // Source 0 is this rank's own band; its token t / feature f sits at
    // f * extent + t, i.e. the same column-major order D itself carries.
    size_t  mismatches = 0;
    int64_t firstT = -1, firstF = -1;
    for(int64_t t = 0; t < kAM && mismatches == 0; ++t)
        for(int64_t f = 0; f < kN; ++f)
            if(hRecv[size_t(f) * kAM + t] != hD[size_t(f) * kM + t])
            {
                if(mismatches++ == 0)
                {
                    firstT = t;
                    firstF = f;
                }
                break;
            }

    if(mismatches != 0)
    {
        std::printf("FAILED: recv/D mismatch at token %lld feature %lld (recv %.1f, D %.1f)\n",
                    (long long)firstT,
                    (long long)firstF,
                    fromBf16(hRecv[size_t(firstF) * kAM + firstT]),
                    fromBf16(hD[size_t(firstF) * kM + firstT]));
        return kStatusFailed;
    }

    std::printf("A2A SDMA loopback verified over %u rank, %lldx%lld tile\n",
                kWorld,
                (long long)kAM,
                (long long)kN);
    return kStatusOk;
}
