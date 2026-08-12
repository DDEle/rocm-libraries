// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-side SDMA queue management for the fused GEMM+AllToAll SDMA offload
// route: allocates the ring, creates the KFD SDMA queue, and exports the
// device-visible handle(s) that the GPU assembly reads to fill packets and
// ring the doorbell.
//
// Ported from MORI's anvil (src/application/transport/sdma/anvil.cpp
// SdmaQueue::SdmaQueue and include/mori/core/transport/sdma/anvil_device.hpp
// SdmaQueueDeviceHandle).
//
// Header-only, and therefore hsakmt-DEPENDENT: including it requires the
// hsakmt/hsa headers on the include path. Only TUs in tensilelite-client-common
// have that (the dependency is PRIVATE to that target), and the sole includer
// gates itself on TENSILELITE_ENABLE_SDMA_A2A.

#pragma once

#include <hip/hip_runtime.h>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsakmt/hsakmt.h"
#include "hsakmt/hsakmttypes.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        // 256KB SDMA ring, matching MORI's SDMA_QUEUE_SIZE. wptr/rptr/doorbell
        // are monotonically increasing BYTE counts; wrap happens only when
        // indexing into the ring (index % SDMA_QUEUE_SIZE).
        constexpr uint32_t SDMA_QUEUE_SIZE = 256 * 1024;

        // Device-visible handle. THE FIELD LAYOUT IS A CONTRACT: the GPU
        // assembly reads these by fixed offset, so field order/type must not
        // change (the static_asserts below lock every offset).
        //
        // The first six fields are pointers into producer-SHARED memory. The
        // seventh, cachedHwReadIndex, is a VALUE and a per-producer PRIVATE
        // cache seed (the hardware read pointer at construction), not shared
        // state: each producer copies it into its own local and mutates that
        // copy, never writing back to this memory (see MORI CanWriteUpto).
        struct SdmaQueueDeviceHandle
        {
            // Producer-shared pointers; plain uint64_t* (not hsakmt's
            // HSAuint64*) so the layout is stated in fixed-width types --
            // same layout, since HSAuint64 is itself a uint64_t typedef.
            uint32_t* queueBuf; // ring base (Uncached)
            uint64_t* rptr; // hardware read pointer (byte count)
            uint64_t* wptr; // hardware write pointer (byte count)
            uint64_t* doorbell; // doorbell (byte count)
            uint64_t* cachedWptr; // software producer cursor (shared, uncached)
            uint64_t* committedWptr; // software committed cursor (shared, uncached)

            uint64_t cachedHwReadIndex; // per-producer private cache seed, see above
        };

        // Size only, not per-field offsets. The asm side strides this array by a
        // hardcoded 7*8 (GlobalWriteBatch.py handleBytes) that nothing links to
        // this type, so an ADDED field is the failure worth catching -- it moves
        // the stride while leaving every existing offset intact.
        static_assert(sizeof(SdmaQueueDeviceHandle) == 7 * sizeof(uint64_t),
                      "SdmaQueueDeviceHandle must be exactly 7 x 8 bytes -- "
                      "GlobalWriteBatch.py strides the handle array by that value, "
                      "and SdmaRingEmitter.py reads the fields at OFF_* 0..48");

        // A NAMED namespace, not an anonymous one: the state below must be one
        // object per PROCESS, and an anonymous namespace in a header gives each
        // TU its own copy -- `inline` would not merge them either, since each
        // TU's entity is distinct.
        namespace detail
        {
            inline void checkHip(hipError_t e, const char* what, const char* file, int line)
            {
                if(e != hipSuccess)
                    throw std::runtime_error(std::string("HIP error at ") + file + ":"
                                             + std::to_string(line) + " - " + what + " ("
                                             + hipGetErrorString(e) + ")");
            }

            inline void checkHsakmt(HSAKMT_STATUS s, const char* what, const char* file, int line)
            {
                if(s != HSAKMT_STATUS_SUCCESS)
                    throw std::runtime_error(std::string("HSAKMT error ") + std::to_string((int)s)
                                             + " at " + file + ":" + std::to_string(line) + " - "
                                             + what);
            }

            inline void checkHsa(hsa_status_t s, const char* what, const char* file, int line)
            {
                if(s != HSA_STATUS_SUCCESS && s != HSA_STATUS_INFO_BREAK)
                {
                    const char* msg = nullptr;
                    hsa_status_string(s, &msg);
                    throw std::runtime_error(std::string("HSA error at ") + file + ":"
                                             + std::to_string(line) + " - " + what + " ("
                                             + (msg ? msg : "?") + ")");
                }
            }

            // HSA + KFD are process-global; initialize once. GPU agents are
            // captured in HIP-device order via the iterate-agents callback.
            inline std::once_flag           gHsaInitFlag;
            inline std::vector<hsa_agent_t> gGpuAgents;

            inline hsa_status_t gpuAgentCb(hsa_agent_t agent, void* data)
            {
                auto*             agents = static_cast<std::vector<hsa_agent_t>*>(data);
                hsa_device_type_t type{};
                hsa_status_t      st = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
                if(st != HSA_STATUS_SUCCESS)
                    return st;
                if(type == HSA_DEVICE_TYPE_GPU)
                    agents->push_back(agent);
                return HSA_STATUS_SUCCESS;
            }
        } // namespace detail

// Scoped to this header: #undef'd at the bottom, after the last use, so a
// header-only include does not leak three very generic names.
#define CHK_HIP(cmd) ::TensileLite::Client::detail::checkHip((cmd), #cmd, __FILE__, __LINE__)
#define CHK_KMT(cmd) ::TensileLite::Client::detail::checkHsakmt((cmd), #cmd, __FILE__, __LINE__)
#define CHK_HSA(cmd) ::TensileLite::Client::detail::checkHsa((cmd), #cmd, __FILE__, __LINE__)

        namespace detail
        {
            inline void ensureHsaKfd()
            {
                std::call_once(gHsaInitFlag, [] {
                    CHK_HSA(hsa_init());
                    CHK_HSA(hsa_iterate_agents(&gpuAgentCb, &gGpuAgents));
                    CHK_KMT(hsaKmtOpenKFD());
                    HsaSystemProperties props{};
                    CHK_KMT(hsaKmtAcquireSystemProperties(&props));
                });
            }
        } // namespace detail

        // ---- Topology helpers ---------------------------------------------
        // KFD topology node id for a HIP device ordinal (via the HSA agent's
        // NODE info, mirroring MORI). Initializes HSA + KFD on first call.
        inline uint32_t sdmaNodeIdForDevice(int hipDeviceId)
        {
            detail::ensureHsaKfd();
            if(hipDeviceId < 0 || hipDeviceId >= (int)detail::gGpuAgents.size())
                throw std::runtime_error("sdmaNodeIdForDevice: HIP device "
                                         + std::to_string(hipDeviceId) + " out of range ("
                                         + std::to_string(detail::gGpuAgents.size())
                                         + " GPU agents)");
            uint32_t node = 0;
            CHK_HSA(
                hsa_agent_get_info(detail::gGpuAgents[hipDeviceId], HSA_AGENT_INFO_NODE, &node));
            return node;
        }

        // SDMA engine id to use for the srcNode->dstNode link. Prefers the
        // first engine in KFD's RecSdmaEngIdMask for that io-link; for a
        // loopback (srcNode==dstNode, no io-link) returns a general engine (0),
        // matching MORI's loopback handling.
        inline uint32_t sdmaSelectEngine(uint32_t srcNode, uint32_t dstNode)
        {
            detail::ensureHsaKfd();
            // Loopback (self) has no io-link and no recommended engine; use a
            // general (non-xGMI) SDMA engine, matching MORI's loopback path.
            if(srcNode == dstNode)
                return 0;

            HsaNodeProperties props{};
            if(hsaKmtGetNodeProperties(srcNode, &props) != HSAKMT_STATUS_SUCCESS
               || props.NumIOLinks == 0)
                return 0;

            std::vector<HsaIoLinkProperties> links(props.NumIOLinks);
            if(hsaKmtGetNodeIoLinkProperties(srcNode, props.NumIOLinks, links.data())
               != HSAKMT_STATUS_SUCCESS)
                return 0;

            for(const auto& link : links)
            {
                if(link.NodeTo == dstNode)
                {
                    uint32_t mask = link.RecSdmaEngIdMask;
                    // First engine set in the recommended mask (one queue per
                    // peer -- no fan-out over multiple engines).
                    for(uint32_t b = 0; b < 32; ++b)
                        if(mask & (1u << b))
                            return b;
                    break;
                }
            }
            return 0; // fall back to a general engine if KFD reports no mask
        }

        // One SDMA queue: owns a 256KB Uncached ring, the KFD queue resource,
        // the two software cursors (uncached device memory), and a device copy
        // of its SdmaQueueDeviceHandle. Non-copyable (owns HW resources).
        //
        // localNode / engineId are KFD topology ids; use sdmaNodeIdForDevice()
        // and sdmaSelectEngine() above to derive them from a HIP device id.
        class SdmaQueue
        {
        public:
            inline SdmaQueue(uint32_t localNode, uint32_t engineId);
            inline ~SdmaQueue();

            SdmaQueue(const SdmaQueue&)            = delete;
            SdmaQueue& operator=(const SdmaQueue&) = delete;

            // Device pointer to this queue's handle (for single-queue device use).
            SdmaQueueDeviceHandle* deviceHandle() const
            {
                return deviceHandle_;
            }
            // Host-visible copy of the same handle (for host-side driving/packing).
            const SdmaQueueDeviceHandle& hostHandle() const
            {
                return hostHandle_;
            }

            // ---- Host-side driving (smoke / bring-up only) -----------------
            // The production producer is the GPU kernel. These helpers let the
            // host enqueue a packet and drive the doorbell so a queue can be
            // exercised end-to-end without any device code.
            //
            // Copies `bytes` of `pkt` into the ring at the current write
            // cursor, advances wptr, and rings the doorbell. `bytes` must be a
            // multiple of 4 and fit without wrapping. Returns the submitted
            // (post-increment) wptr byte count.
            inline uint64_t submitPacketHost(const void* pkt, size_t bytes);

            // Spin until the engine's read pointer catches up to the last
            // submitted write pointer (queue fully drained). Returns false on
            // timeout.
            inline bool waitIdleHost(uint64_t timeoutSpins = (1ull << 34));

        private:
            // Best-effort release, shared by the destructor and the ctor's
            // failure path (a throw mid-ctor means ~SdmaQueue never runs).
            inline void teardown() noexcept;

            void*            queueBuffer_ = nullptr; // ring (Uncached)
            HsaQueueResource queue_{}; // KFD queue resource

            uint64_t*              cachedWptr_    = nullptr; // uncached device mem
            uint64_t*              committedWptr_ = nullptr; // uncached device mem
            SdmaQueueDeviceHandle* deviceHandle_  = nullptr; // device copy
            SdmaQueueDeviceHandle  hostHandle_{}; // host copy
            uint64_t               hostWptr_ = 0; // host-side write cursor
        };

        inline SdmaQueue::SdmaQueue(uint32_t localNode, uint32_t engineId)
        {
            detail::ensureHsaKfd();

            // Ring: NonPaged + HostAccess + ExecuteAccess + Uncached, 4KB pages.
            // Uncached is load-bearing (packet writes bypass L2 -> no flush).
            HsaMemFlags memFlags{};
            memFlags.ui32.NonPaged      = 1;
            memFlags.ui32.HostAccess    = 1;
            memFlags.ui32.PageSize      = HSA_PAGE_SIZE_4KB;
            memFlags.ui32.NoNUMABind    = 1;
            memFlags.ui32.ExecuteAccess = 1;
            memFlags.ui32.Uncached      = 1;

            // ~SdmaQueue() will not run on a throw here, so run the same
            // teardown on any exception before rethrowing.
            try
            {
                CHK_KMT(hsaKmtAllocMemory(localNode, SDMA_QUEUE_SIZE, memFlags, &queueBuffer_));
                CHK_KMT(hsaKmtMapMemoryToGPU(queueBuffer_, SDMA_QUEUE_SIZE, nullptr));

                std::memset(&queue_, 0, sizeof(HsaQueueResource));
                CHK_KMT(hsaKmtCreateQueueExt(localNode,
                                             HSA_QUEUE_SDMA_BY_ENG_ID,
                                             100, // queue percentage
                                             HSA_QUEUE_PRIORITY_MAXIMUM,
                                             engineId,
                                             queueBuffer_,
                                             SDMA_QUEUE_SIZE,
                                             nullptr,
                                             &queue_));

                // Software cursors in uncached device memory (shared producer state).
                CHK_HIP(hipMalloc(&deviceHandle_, sizeof(SdmaQueueDeviceHandle)));
                CHK_HIP(hipExtMallocWithFlags(
                    (void**)&cachedWptr_, sizeof(uint64_t), hipDeviceMallocUncached));
                CHK_HIP(hipExtMallocWithFlags(
                    (void**)&committedWptr_, sizeof(uint64_t), hipDeviceMallocUncached));

                // Seed the cursors to the current HARDWARE write pointer so the
                // first reserved index is contiguous with whatever the queue was
                // created at (MORI does exactly this).
                const uint64_t hwWptr = (uint64_t) * (queue_.Queue_write_ptr_aql);
                const uint64_t hwRptr = (uint64_t) * (queue_.Queue_read_ptr_aql);
                hostWptr_             = hwWptr;

                hostHandle_ = SdmaQueueDeviceHandle{
                    /*queueBuf*/ static_cast<uint32_t*>(queueBuffer_),
                    /*rptr*/ (uint64_t*)queue_.Queue_read_ptr_aql,
                    /*wptr*/ (uint64_t*)queue_.Queue_write_ptr_aql,
                    /*doorbell*/ (uint64_t*)queue_.Queue_DoorBell_aql,
                    /*cachedWptr*/ cachedWptr_,
                    /*committedWptr*/ committedWptr_,
                    // Per-producer private cache seed, see the struct above.
                    /*cachedHwReadIndex*/ hwRptr,
                };

                CHK_HIP(hipMemcpy(deviceHandle_,
                                  &hostHandle_,
                                  sizeof(SdmaQueueDeviceHandle),
                                  hipMemcpyHostToDevice));
                CHK_HIP(hipMemcpy(cachedWptr_, &hwWptr, sizeof(uint64_t), hipMemcpyHostToDevice));
                CHK_HIP(
                    hipMemcpy(committedWptr_, &hwWptr, sizeof(uint64_t), hipMemcpyHostToDevice));
            }
            catch(...)
            {
                teardown();
                throw;
            }
        }

        inline void SdmaQueue::teardown() noexcept
        {
            // Best-effort release, shared by the destructor and the ctor's
            // failure path; safe to call after a partial construction.
            if(queue_.QueueId)
            {
                (void)hsaKmtDestroyQueue(queue_.QueueId);
                queue_.QueueId = 0;
            }
            if(deviceHandle_)
            {
                (void)hipFree(deviceHandle_);
                deviceHandle_ = nullptr;
            }
            if(cachedWptr_)
            {
                (void)hipFree(cachedWptr_);
                cachedWptr_ = nullptr;
            }
            if(committedWptr_)
            {
                (void)hipFree(committedWptr_);
                committedWptr_ = nullptr;
            }
            if(queueBuffer_)
            {
                (void)hsaKmtUnmapMemoryToGPU(queueBuffer_);
                (void)hsaKmtFreeMemory(queueBuffer_, SDMA_QUEUE_SIZE);
                queueBuffer_ = nullptr;
            }
        }

        inline SdmaQueue::~SdmaQueue()
        {
            teardown();
        }

        inline uint64_t SdmaQueue::submitPacketHost(const void* pkt, size_t bytes)
        {
            if(bytes == 0 || (bytes % sizeof(uint32_t)) != 0)
                throw std::runtime_error("submitPacketHost: bytes must be a nonzero multiple of 4");
            if(bytes > SDMA_QUEUE_SIZE)
                throw std::runtime_error("submitPacketHost: packet larger than ring");

            // Byte offset into the ring for the current write cursor.
            const uint64_t offset = hostWptr_ % SDMA_QUEUE_SIZE;
            if(offset + bytes > SDMA_QUEUE_SIZE)
                throw std::runtime_error("submitPacketHost: packet would wrap the ring "
                                         "(host smoke path does not implement wrap)");

            // Ring is Uncached, so this is visible to the engine with no flush.
            std::memcpy(static_cast<uint8_t*>(queueBuffer_) + offset, pkt, bytes);

            hostWptr_ += bytes;

            // Publish the new write pointer, then ring the doorbell.
            *(queue_.Queue_write_ptr_aql) = hostWptr_;
            // Ensure the wptr store lands before the doorbell store.
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
            *(queue_.Queue_DoorBell_aql) = hostWptr_;

            return hostWptr_;
        }

        inline bool SdmaQueue::waitIdleHost(uint64_t timeoutSpins)
        {
            for(uint64_t i = 0; i < timeoutSpins; ++i)
            {
                const uint64_t rp = (uint64_t) * (volatile HSAuint64*)(queue_.Queue_read_ptr_aql);
                if(rp >= hostWptr_)
                    return true;
            }
            return false;
        }

        // A set of W queues for one local device -- one queue per peer. The W
        // device handles are packed contiguously into a single device array so
        // a kernel can index them by destination rank.
        class SdmaQueueSet
        {
        public:
            // localNode is this device's KFD node; targetNodes[j] is peer j's
            // KFD node (use localNode for a loopback/self entry). One queue is
            // created per target, with its engine chosen by sdmaSelectEngine().
            inline SdmaQueueSet(uint32_t localNode, const std::vector<uint32_t>& targetNodes);
            inline ~SdmaQueueSet();

            SdmaQueueSet(const SdmaQueueSet&)            = delete;
            SdmaQueueSet& operator=(const SdmaQueueSet&) = delete;

            size_t size() const
            {
                return queues_.size();
            }
            SdmaQueue& queue(size_t i)
            {
                return *queues_[i];
            }

            // Device pointer to the W-element SdmaQueueDeviceHandle array.
            SdmaQueueDeviceHandle* deviceHandles() const
            {
                return dHandles_;
            }

        private:
            std::vector<std::unique_ptr<SdmaQueue>> queues_;
            SdmaQueueDeviceHandle*                  dHandles_ = nullptr;
        };

        inline SdmaQueueSet::SdmaQueueSet(uint32_t                     localNode,
                                          const std::vector<uint32_t>& targetNodes)
        {
            detail::ensureHsaKfd();

            std::vector<SdmaQueueDeviceHandle> handles;
            handles.reserve(targetNodes.size());
            for(uint32_t dstNode : targetNodes)
            {
                uint32_t engine = sdmaSelectEngine(localNode, dstNode);
                queues_.emplace_back(std::make_unique<SdmaQueue>(localNode, engine));
                handles.push_back(queues_.back()->hostHandle());
            }

            const size_t bytes = handles.size() * sizeof(SdmaQueueDeviceHandle);

            // Hold the allocation in a local owner until the copy succeeds, so a
            // failing hipMemcpy doesn't leak it (CHK_HIP throws, and a throw here
            // means ~SdmaQueueSet never runs).
            SdmaQueueDeviceHandle* raw = nullptr;
            CHK_HIP(hipMalloc(&raw, bytes));
            auto hipFreeDeleter = [](SdmaQueueDeviceHandle* p) { (void)hipFree(p); };
            std::unique_ptr<SdmaQueueDeviceHandle, decltype(hipFreeDeleter)> owned(raw,
                                                                                   hipFreeDeleter);
            CHK_HIP(hipMemcpy(owned.get(), handles.data(), bytes, hipMemcpyHostToDevice));
            dHandles_ = owned.release();
        }

        inline SdmaQueueSet::~SdmaQueueSet()
        {
            if(dHandles_)
                (void)hipFree(dHandles_);
        }

    } // namespace Client
} // namespace TensileLite

// Last use is above; do not let these escape to includers.
#undef CHK_HIP
#undef CHK_KMT
#undef CHK_HSA
