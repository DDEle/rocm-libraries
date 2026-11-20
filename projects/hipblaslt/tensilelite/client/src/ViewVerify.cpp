// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ViewVerify.hpp"
#include "DeviceContext.hpp"

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <stdexcept>
#include <string>
#include <tuple>

namespace TensileLite
{
    namespace Client
    {
        __global__ void verifyOpKernel(VerifyOp op, uint32_t* block)
        {
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            const VerifySpan t = verifySpan(blockIdx.x, threadIdx.x, gridDim.x, blockDim.x);
            verifyRunOp(op, t.first, t.step, block, t.lead);
            if(op.fence)
                __threadfence_system();
        }

        // Single thread.
        __global__ void verifySkewKernel(uint64_t ticks)
        {
            const uint64_t start = (uint64_t)wall_clock64();
            while((uint64_t)wall_clock64() - start < ticks)
                __builtin_amdgcn_s_sleep(1);
        }

        namespace
        {
            bool signMagnitude(rocisa::DataType type)
            {
                switch(type)
                {
                case rocisa::DataType::Float:
                case rocisa::DataType::Half:
                case rocisa::DataType::BFloat16:
                case rocisa::DataType::Float8:
                case rocisa::DataType::BFloat8:
                    return true;
                default:
                    return false;
                }
            }

            VerifyTensor tensorOf(TensorDescriptor const& t, size_t dim)
            {
                VerifyTensor v;
                v.elementBytes   = (uint32_t)t.elementBytes();
                v.signMagnitude  = signMagnitude(t.dataType());
                v.allocatedBytes = t.totalAllocatedBytes();
                v.padded         = t.totalAllocatedElements() != t.totalLogicalElements();
                std::tie(v.stride, v.period) = verifyStridePeriod(t.strides(), dim);
                return v;
            }

            ContractionInputs const& slotInputs(DeviceContext const& ctx, size_t slot)
            {
                return dynamic_cast<ContractionInputs const&>(*ctx.inputArr.at(slot));
            }

            VerifySlot slotOf(DeviceContext const& ctx, size_t slot)
            {
                auto const& in = slotInputs(ctx, slot);
                return {in.a, in.b, in.d, in.a2aBuf};
            }

            void const* need(void const* p, char const* name, DeviceContext const& ctx)
            {
                if(p == nullptr)
                    throw std::runtime_error(concatenate(
                        "[fused-a2a] no verify ", name, " on device ", ctx.deviceId, "."));
                return p;
            }

            VerifyGoldens goldensOf(DeviceContext const& ctx, VerifyLayout const& layout)
            {
                try
                {
                    return verifyGoldensOf(ctx.golden, layout);
                }
                catch(std::runtime_error const& err)
                {
                    throw std::runtime_error(
                        concatenate(err.what(), " (device ", ctx.deviceId, ")"));
                }
            }

            void run(DeviceContext& ctx, std::vector<VerifyOp> const& ops)
            {
                for(auto const& op : ops)
                {
                    hipLaunchKernelGGL(verifyOpKernel,
                                       ::dim3(verifyBlocks(op.count)),
                                       ::dim3(VERIFY_THREADS),
                                       0,
                                       ctx.stream,
                                       op,
                                       static_cast<uint32_t*>(ctx.verifyBlock));
                    HIP_CHECK_EXC(hipGetLastError());
                }
            }
        }

        VerifyLayout verifyLayout(ContractionProblem const& problem, size_t world)
        {
            auto const* gemm = dynamic_cast<ContractionProblemGemm const*>(&problem);
            if(gemm == nullptr || gemm->fusedA2A() == FusedA2A::Disabled)
                throw std::runtime_error("[fused-a2a] --fused-a2a-verify needs a fused problem.");
            auto const& tensors = gemm->tensors();
            VerifyShape s;
            s.gemmA2A         = gemm->fusedA2A() == FusedA2A::GemmA2A;
            s.world           = (uint32_t)world;
            s.extent          = (uint64_t)gemm->fusedA2AExtent();
            s.tokens          = gemm->freeSizeB(0);
            const size_t none = (size_t)-1;
            s.a   = tensorOf(gemm->a(),
                             s.gemmA2A ? gemm->freeIndicesA()[0].i : gemm->boundIndices()[0].a);
            s.b   = tensorOf(gemm->b(), none);
            s.d   = tensorOf(tensors[ContractionProblemGemm::TENSOR::D],
                             s.gemmA2A ? gemm->freeIndices()[0].d : none);
            s.buf = tensorOf(tensors[ContractionProblemGemm::TENSOR::A2A_BUF], none);
            return verifyLayoutOf(s);
        }

        void syncAll(std::vector<std::shared_ptr<DeviceContext>> const& devices)
        {
            forEachDevice(devices, [](DeviceContext& ctx) {
                HIP_CHECK_EXC(hipStreamSynchronize(ctx.stream));
            });
        }

        uint64_t verifyProduce(DeviceContext&       ctx,
                               VerifyLayout const&  layout,
                               size_t               slot,
                               uint32_t             launch,
                               VerifyOptions const& options,
                               uint32_t             rank)
        {
            const VerifySlot    s     = slotOf(ctx, slot);
            const VerifyGoldens g     = goldensOf(ctx, layout);
            uint64_t            ticks = 0;
            if(verifySkews(options, launch, rank, layout.world))
            {
                int wallClockKHz = 0;
                HIP_CHECK_EXC(hipDeviceGetAttribute(
                    &wallClockKHz, hipDeviceAttributeWallClockRate, ctx.deviceId));
                ticks = verifySkewTicks(options.skewUs, (uint64_t)wallClockKHz);
                hipLaunchKernelGGL(verifySkewKernel, ::dim3(1), ::dim3(1), 0, ctx.stream, ticks);
                HIP_CHECK_EXC(hipGetLastError());
            }
            run(ctx, verifyProduceOps(layout, s, g, launch, options.fence));
            return ticks;
        }

        void verifyCheck(DeviceContext&      ctx,
                         VerifyLayout const& layout,
                         size_t              slot,
                         uint32_t            launch)
        {
            need(ctx.verifyBlock, "check block", ctx);
            run(ctx, verifyCheckOps(layout, slotOf(ctx, slot), goldensOf(ctx, layout), launch));
        }

        void verifyCaptureInputs(DeviceContext& ctx, VerifyLayout const& layout)
        {
            run(ctx, verifyCaptureInputOps(layout, slotOf(ctx, 0), goldensOf(ctx, layout)));
        }

        void verifyCaptureGolden(DeviceContext& ctx, VerifyLayout const& layout)
        {
            run(ctx, verifyCaptureGoldenOps(layout, slotOf(ctx, 0), goldensOf(ctx, layout)));
        }

        void verifyNormalize(DeviceContext&      ctx,
                             VerifyLayout const& layout,
                             size_t              slot,
                             uint32_t            launch)
        {
            run(ctx, verifyNormalizeOps(layout, slotOf(ctx, slot), launch));
        }

        void verifyRestore(DeviceContext& ctx, VerifyLayout const& layout)
        {
            const VerifyGoldens g = goldensOf(ctx, layout);
            for(size_t slot = 0; slot < ctx.inputArr.size(); slot++)
                run(ctx, verifyRestoreOps(layout, slotOf(ctx, slot), g));
        }

        VerifyCounts verifyReadBlock(DeviceContext& ctx)
        {
            uint32_t words[VERIFY_BLOCK_WORDS];
            HIP_CHECK_EXC(hipMemcpy(words,
                                    need(ctx.verifyBlock, "check block", ctx),
                                    sizeof(words),
                                    hipMemcpyDeviceToHost));
            verifyResetBlock(ctx.verifyBlock);
            return verifyDecodeBlock(words);
        }

        void verifyResetBlock(void* block)
        {
            uint32_t words[VERIFY_BLOCK_WORDS];
            verifyInitialBlock(words);
            HIP_CHECK_EXC(hipMemcpy(block, words, sizeof(words), hipMemcpyHostToDevice));
        }
    } // namespace Client
} // namespace TensileLite
