// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <Tensile/Tensile_fwd.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <hip/hip_runtime.h>

#include <memory>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        class DataInitialization;

        class ScopedDevice
        {
        public:
            explicit ScopedDevice(int deviceId)
            {
                HIP_CHECK_EXC(hipGetDevice(&m_prevDeviceId));
                HIP_CHECK_EXC(hipSetDevice(deviceId));
            }

            ~ScopedDevice()
            {
                static_cast<void>(hipSetDevice(m_prevDeviceId));
            }

            ScopedDevice(ScopedDevice const&)            = delete;
            ScopedDevice& operator=(ScopedDevice const&) = delete;

        private:
            int m_prevDeviceId = 0;
        };

        struct DeviceContext
        {
            DeviceContext(int deviceId, bool useDefaultStream, size_t fusedA2AFlagBytes = 0)
                : deviceId(deviceId)
                , adapter(std::make_shared<hip::SolutionAdapter>())
                , fusedA2AFlagBytes(fusedA2AFlagBytes)
            {
                if(!useDefaultStream)
                {
                    ScopedDevice guard(deviceId);
                    HIP_CHECK_EXC(hipStreamCreate(&stream));
                }
                if(fusedA2AFlagBytes != 0)
                {
                    ScopedDevice guard(deviceId);
                    // Fine-grained: written by remote peers, must bypass stale L2.
                    HIP_CHECK_EXC(hipExtMallocWithFlags(
                        &fusedA2AFlags, fusedA2AFlagBytes, hipDeviceMallocFinegrained));
                    HIP_CHECK_EXC(hipMemset(fusedA2AFlags, 0, fusedA2AFlagBytes));
                }
            }

            ~DeviceContext()
            {
                if(fusedA2AFlags != nullptr)
                    static_cast<void>(hipFree(fusedA2AFlags));
                if(stream != nullptr)
                    static_cast<void>(hipStreamDestroy(stream));
            }

            DeviceContext(DeviceContext const&)            = delete;
            DeviceContext& operator=(DeviceContext const&) = delete;

            int                                         deviceId;
            hipStream_t                                 stream = nullptr;
            std::shared_ptr<hip::SolutionAdapter>       adapter;
            std::shared_ptr<DataInitialization>         dataInit;
            std::shared_ptr<ProblemInputs>              inputs;
            std::vector<std::shared_ptr<ProblemInputs>> inputArr;
            size_t                                      fusedA2AFlagBytes;
            void*                                       fusedA2AFlags = nullptr;
        };

        template <typename F>
        void forEachDevice(std::vector<std::shared_ptr<DeviceContext>> const& devices, F&& f)
        {
            for(auto const& ctx : devices)
            {
                ScopedDevice guard(ctx->deviceId);
                f(*ctx);
            }
        }
    } // namespace Client
} // namespace TensileLite
