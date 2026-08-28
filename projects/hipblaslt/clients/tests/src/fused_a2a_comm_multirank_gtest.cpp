// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Multi-rank tests for the per-peer resolution in hipblasLtSetDeviceComm: one thread per
// rank, each with its own handle on its own device, exercising the same-process peer branch.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "../../../library/src/amd_detail/rocblaslt/src/include/handle.h"
#include "../../../library/src/amd_detail/rocblaslt/src/include/rocblaslt_fused_a2a_peers.hpp"

__global__ void fusedA2AStoreWord(uint32_t* dst, uint32_t value)
{
    *dst = value;
}

namespace
{
    constexpr std::chrono::seconds kBarrierTimeout{30};
    constexpr uint32_t             kWorld    = 2;
    constexpr uint32_t             kChannels = 3;

    // Collects one record per rank and releases every rank once all have arrived. Waiting is
    // bounded by kBarrierTimeout; a rank that times out gets false.
    class RecordExchange
    {
    public:
        explicit RecordExchange(uint32_t world)
            : m_records(world)
        {
        }

        bool contributeAndWait(const RocblasltFusedA2APeerRecord& record)
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            if(record.rank >= m_records.size())
                return false;
            m_records[record.rank] = record;
            if(++m_arrived == m_records.size())
            {
                m_ready = true;
                m_cv.notify_all();
                return true;
            }
            return m_cv.wait_for(lock, kBarrierTimeout, [this] { return m_ready; });
        }

        const RocblasltFusedA2APeerRecord* data() const
        {
            return m_records.data();
        }

        size_t world() const
        {
            return m_records.size();
        }

    private:
        std::vector<RocblasltFusedA2APeerRecord> m_records;
        std::mutex                               m_mutex;
        std::condition_variable                  m_cv;
        uint32_t                                 m_arrived = 0;
        bool                                     m_ready   = false;
    };

    // Takes the slot index from the record rather than from a parameter.
    hipblasStatus_t barrierAllgather(void*       userData,
                                     const void* sendbuf,
                                     void*       recvbuf,
                                     size_t      bytesPerRank)
    {
        auto* exchange = static_cast<RecordExchange*>(userData);
        if(bytesPerRank != sizeof(RocblasltFusedA2APeerRecord))
            return HIPBLAS_STATUS_INVALID_VALUE;

        RocblasltFusedA2APeerRecord record{};
        std::memcpy(&record, sendbuf, sizeof(record));
        if(!exchange->contributeAndWait(record))
            return HIPBLAS_STATUS_INTERNAL_ERROR;

        std::memcpy(recvbuf, exchange->data(), bytesPerRank * exchange->world());
        return HIPBLAS_STATUS_SUCCESS;
    }

    class FusedA2ACommMultiRank : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            int deviceCount = 0;
            ASSERT_EQ(hipGetDeviceCount(&deviceCount), hipSuccess);
            if(deviceCount < int(kWorld))
                GTEST_SKIP() << "needs " << kWorld << " devices";

            int forward = 0;
            int backward = 0;
            ASSERT_EQ(hipDeviceCanAccessPeer(&forward, 0, 1), hipSuccess);
            ASSERT_EQ(hipDeviceCanAccessPeer(&backward, 1, 0), hipSuccess);
            if(forward == 0 || backward == 0)
                GTEST_SKIP() << "devices 0 and 1 cannot reach each other";

            RecordExchange               exchange(kWorld);
            std::vector<hipblasStatus_t> created(kWorld, HIPBLAS_STATUS_NOT_INITIALIZED);
            std::vector<hipblasStatus_t> registered(kWorld, HIPBLAS_STATUS_NOT_INITIALIZED);

            std::vector<std::thread> ranks;
            for(uint32_t rank = 0; rank < kWorld; ++rank)
            {
                ranks.emplace_back([&, rank] {
                    if(hipSetDevice(int(rank)) != hipSuccess)
                        return;
                    created[rank] = hipblasLtCreate(&m_handles[rank]);
                    if(created[rank] != HIPBLAS_STATUS_SUCCESS)
                        return;
                    registered[rank] = hipblasLtSetDeviceComm(
                        m_handles[rank], rank, kWorld, kChannels, barrierAllgather, &exchange);
                });
            }
            for(auto& rank : ranks)
                rank.join();

            for(uint32_t rank = 0; rank < kWorld; ++rank)
            {
                ASSERT_EQ(created[rank], HIPBLAS_STATUS_SUCCESS) << "rank " << rank;
                ASSERT_EQ(registered[rank], HIPBLAS_STATUS_SUCCESS) << "rank " << rank;
            }
        }

        void TearDown() override
        {
            for(auto& handle : m_handles)
            {
                if(handle != nullptr)
                    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
                handle = nullptr;
            }
        }

        _rocblaslt_handle* rank(uint32_t index) const
        {
            return reinterpret_cast<_rocblaslt_handle*>(m_handles[index]);
        }

        std::vector<hipblasLtHandle_t> m_handles = std::vector<hipblasLtHandle_t>(kWorld, nullptr);
    };

    TEST_F(FusedA2ACommMultiRank, eachRankResolvesPeerToThePeersOwnRegion)
    {
        auto* rank0 = rank(0);
        auto* rank1 = rank(1);

        ASSERT_NE(rank0->comm_flag_base, nullptr);
        ASSERT_NE(rank1->comm_flag_base, nullptr);
        ASSERT_NE(rank0->comm_flag_base, rank1->comm_flag_base);

        EXPECT_EQ(int(rank0->comm_peer_kind[0]), int(rocblaslt_comm_peer_self));
        EXPECT_EQ(rank0->comm_peer_flag[0], rank0->comm_flag_base);
        EXPECT_EQ(int(rank0->comm_peer_kind[1]), int(rocblaslt_comm_peer_local));
        EXPECT_EQ(rank0->comm_peer_flag[1], rank1->comm_flag_base);

        EXPECT_EQ(int(rank1->comm_peer_kind[1]), int(rocblaslt_comm_peer_self));
        EXPECT_EQ(rank1->comm_peer_flag[1], rank1->comm_flag_base);
        EXPECT_EQ(int(rank1->comm_peer_kind[0]), int(rocblaslt_comm_peer_local));
        EXPECT_EQ(rank1->comm_peer_flag[0], rank0->comm_flag_base);
    }

    TEST_F(FusedA2ACommMultiRank, resolvedPeersFeedTheKernargGroups)
    {
        const auto peers = rocblaslt::buildFusedA2APeerFields(
            rank(0)->comm_peer_flag, nullptr, kWorld, 0, nullptr);

        ASSERT_EQ(peers.size(), size_t(kWorld));
        EXPECT_EQ(peers[0][rocblaslt::kFusedA2AFlagSlot], rank(0)->comm_flag_base);
        EXPECT_EQ(peers[1][rocblaslt::kFusedA2AFlagSlot], rank(1)->comm_flag_base);
    }

    TEST_F(FusedA2ACommMultiRank, peerFlagIsWritableFromTheResolvingDevice)
    {
        constexpr uint32_t kSentinel = 0x0A2A0A2Au;

        ASSERT_EQ(hipSetDevice(0), hipSuccess);
        fusedA2AStoreWord<<<1, 1>>>(static_cast<uint32_t*>(rank(0)->comm_peer_flag[1]), kSentinel);
        ASSERT_EQ(hipGetLastError(), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        uint32_t readback = 0;
        ASSERT_EQ(hipSetDevice(1), hipSuccess);
        ASSERT_EQ(hipMemcpy(&readback,
                            rank(1)->comm_flag_base,
                            sizeof(readback),
                            hipMemcpyDeviceToHost),
                  hipSuccess);
        EXPECT_EQ(readback, kSentinel);
    }
}
