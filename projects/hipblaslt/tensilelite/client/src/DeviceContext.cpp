// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DeviceContext.hpp"

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Utils.hpp>

#ifdef TENSILELITE_ENABLE_SDMA
#include "SdmaQueue.hpp"
#endif

#include <stdexcept>

namespace TensileLite
{
    namespace Client
    {
        namespace
        {
            // Counter regions at the head of the Synchronizer, each on a 64-byte line:
            //
            //   +0    cursors   MAX_RANKS * 2 * u64   SDMA cursor pair per queue
            //   +128  counter2  MAX_RANKS * u32       counter2[dst_rank]
            //   +192  counter3  1 * u32
            //   +256  counter1  W * tokenTiles * u32  counter1[dst_rank*tokenTiles + j]
            //
            // Twinned with FUSED_A2A_COUNTER*_OFFSET in Tensile/Components/Signature.py.
            constexpr size_t kCursorsPerQueue = 2;
            constexpr size_t kCounter2Offset  = fusedA2AAlignLine(
                (size_t)FUSED_A2A_MAX_RANKS * kCursorsPerQueue * sizeof(uint64_t));
            constexpr size_t kCounter3Offset = fusedA2AAlignLine(
                kCounter2Offset + (size_t)FUSED_A2A_MAX_RANKS * sizeof(uint32_t));
            constexpr size_t kCounter1Offset
                = fusedA2AAlignLine(kCounter3Offset + sizeof(uint32_t));
        }

        DeviceContext::DeviceContext(int                     deviceId,
                                     bool                    useDefaultStream,
                                     size_t                  commFlagsBytes,
                                     std::vector<int> const& sdmaPeers)
            : deviceId(deviceId)
            , adapter(std::make_shared<hip::SolutionAdapter>())
            , commFlagsBytes(commFlagsBytes)
        {
            if(!useDefaultStream)
            {
                ScopedDevice guard(deviceId);
                HIP_CHECK_EXC(hipStreamCreate(&stream));
            }
            if(commFlagsBytes != 0)
            {
                ScopedDevice guard(deviceId);
                // Fine-grained: written by remote peers, must bypass stale L2.
                HIP_CHECK_EXC(
                    hipExtMallocWithFlags(&commFlags, commFlagsBytes, hipDeviceMallocFinegrained));
                HIP_CHECK_EXC(hipMemset(commFlags, 0, commFlagsBytes));
            }
            if(!sdmaPeers.empty())
            {
#ifdef TENSILELITE_ENABLE_SDMA
                ScopedDevice guard(deviceId);
                for(int peer : sdmaPeers)
                {
                    if(peer == deviceId)
                        continue;
                    hipError_t pe = hipDeviceEnablePeerAccess(peer, 0);
                    if(pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled)
                        HIP_CHECK_EXC(pe);
                }
                const uint32_t node = sdmaNodeIdForDevice(deviceId);
                for(int peer : sdmaPeers)
                    sdmaQueues.push_back(std::make_shared<SdmaQueue>(
                        node, sdmaSelectEngine(node, sdmaNodeIdForDevice(peer))));
#else
                throw std::runtime_error(
                    "DeviceContext: SDMA queues requested, but this client was built without "
                    "TENSILELITE_ENABLE_SDMA.");
#endif
            }
        }

        DeviceContext::~DeviceContext()
        {
            if(commFlags != nullptr)
                static_cast<void>(hipFree(commFlags));
            if(stream != nullptr)
                static_cast<void>(hipStreamDestroy(stream));
        }

        size_t problemWorld(ContractionProblem const& problem)
        {
            auto gemm = dynamic_cast<ContractionProblemGemm const*>(&problem);
            return gemm && gemm->fusedGemmA2A() ? gemm->fusedA2AWorld() : 1;
        }

        void fillFusedA2AInputs(ContractionInputs&                                 inputs,
                                ContractionProblemGemm const&                      problem,
                                ContractionSolution const&                         solution,
                                Hardware const&                                    hardware,
                                std::vector<std::shared_ptr<DeviceContext>> const& devices,
                                size_t                                             rank,
                                size_t                                             slot,
                                size_t                                             channel)
        {
            const size_t world        = devices.size();
            const size_t macroTileN   = solution.sizeMapping.macroTile.y;
            const size_t tokenTiles   = (problem.freeSizeB(0) + macroTileN - 1) / macroTileN;
            const size_t counterBytes = kCounter1Offset + world * tokenTiles * sizeof(uint32_t);
            const size_t synchronizerBytes
                = problem.tensors()[ContractionProblemGemm::TENSOR::Synchronizer]
                      .totalAllocatedBytes();
            if(counterBytes > synchronizerBytes)
                throw std::runtime_error(concatenate("[fused-a2a] the counter needs ",
                                                     counterBytes,
                                                     " bytes; the Synchronizer holds ",
                                                     synchronizerBytes,
                                                     "."));
            if(solution.requiredWorkspaceSize(problem, hardware) != 0)
                throw std::runtime_error(
                    "[fused-a2a] solutions that need a workspace are not supported.");

            inputs.fusedA2APeers.clear();
#ifdef TENSILELITE_ENABLE_SDMA
            for(size_t j = 0; j < world; j++)
            {
                if(slot >= devices[j]->inputArr.size())
                    throw std::runtime_error(concatenate("[fused-a2a] device ",
                                                         j,
                                                         " has ",
                                                         devices[j]->inputArr.size(),
                                                         " rotating buffers; slot ",
                                                         slot,
                                                         " was requested."));
                auto const& peer
                    = dynamic_cast<ContractionInputs const&>(*devices[j]->inputArr[slot]);
                auto const&             queue = *devices[rank]->sdmaQueues[j];
                const HsaQueueResource& q     = queue.queueResource();
                inputs.fusedA2APeers.push_back(
                    {(char*)devices[j]->commFlags + channel * FUSED_A2A_FLAG_BLOCK_BYTES,
                     peer.a2aRecv,
                     queue.ringBase(),
                     (void*)q.Queue_read_ptr_aql,
                     (void*)q.Queue_write_ptr_aql,
                     (void*)q.Queue_DoorBell_aql});
            }
#endif
            inputs.fusedA2ACounter = inputs.Synchronizer;
            inputs.fusedA2AMyRank  = (uint32_t)rank;
            inputs.fusedA2ADrain   = FUSED_A2A_DRAIN_RECV | FUSED_A2A_DRAIN_SEND;
        }
    } // namespace Client
} // namespace TensileLite
