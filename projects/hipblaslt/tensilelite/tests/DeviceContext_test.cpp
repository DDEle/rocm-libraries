// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "A2AGemmHost.hpp"
#include "DeviceContext.hpp"
#include "FusedA2ACounterSentinel.hpp"
#include "TimingEvents.hpp"

#include <Tensile/ContractionProblem.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <memory>
#include <vector>

using TensileLite::Client::DeviceContext;
using TensileLite::Client::ScopedDevice;
using TensileLite::Client::TimingEvents;

namespace
{
    int deviceCount()
    {
        int n = 0;
        if(hipGetDeviceCount(&n) != hipSuccess)
            return 0;
        return n;
    }

    int currentDevice()
    {
        int d = -1;
        EXPECT_EQ(hipGetDevice(&d), hipSuccess);
        return d;
    }
}

TEST(ScopedDevice, SwitchesAndRestores)
{
    if(deviceCount() < 2)
        GTEST_SKIP() << "needs two devices";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    {
        ScopedDevice guard(1);
        EXPECT_EQ(currentDevice(), 1);
    }
    EXPECT_EQ(currentDevice(), 0);
}

TEST(DeviceContext, StreamLivesOnItsDevice)
{
    if(deviceCount() < 2)
        GTEST_SKIP() << "needs two devices";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    DeviceContext ctx(1, false);
    EXPECT_EQ(currentDevice(), 0);
    ASSERT_NE(ctx.stream, nullptr);
    hipDevice_t dev = -1;
    ASSERT_EQ(hipStreamGetDevice(ctx.stream, &dev), hipSuccess);
    EXPECT_EQ(dev, 1);
    EXPECT_NE(ctx.adapter.get(), nullptr);
}

TEST(DeviceContext, DefaultStreamIsNull)
{
    if(deviceCount() < 1)
        GTEST_SKIP() << "needs a device";
    DeviceContext ctx(0, true);
    EXPECT_EQ(ctx.stream, nullptr);
}

TEST(TimingEvents, CountsInvocationsWithoutCards)
{
    TimingEvents events({}, 4, 1);
    EXPECT_EQ(events.numInvocations(), 4u);
    EXPECT_EQ(events.numCards(), 0u);
}

TEST(TimingEvents, CreatesEachCardsEventsOnItsDevice)
{
    if(deviceCount() < 2)
        GTEST_SKIP() << "needs two devices";
    std::vector<std::shared_ptr<DeviceContext>> devices{std::make_shared<DeviceContext>(0, false),
                                                        std::make_shared<DeviceContext>(1, false)};
    TimingEvents                                events(devices, 3, 2);
    ASSERT_EQ(events.numCards(), 2u);
    for(size_t d = 0; d < devices.size(); d++)
    {
        ASSERT_EQ(events.card(d).size(), 3u);
        ASSERT_EQ(events.card(d)[2].size(), 2u);
        ScopedDevice device(devices[d]->deviceId);
        EXPECT_EQ(hipEventRecord(events.card(d)[2][1], devices[d]->stream), hipSuccess);
        EXPECT_EQ(hipEventSynchronize(events.card(d)[2][1]), hipSuccess);
    }
}

TEST(DeviceContext, ArmA2AGemmPoisonsEverySlot)
{
    if(deviceCount() < 1)
        GTEST_SKIP() << "needs a device";
    using namespace TensileLite;
    using namespace TensileLite::Client;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    DeviceContext ctx(0, true, fusedA2ACommFlagsBytes(FusedA2A::A2AGemm));

    constexpr size_t bBytes = 4096;
    std::vector<void*> bufs(2);
    for(auto& p : bufs)
    {
        ASSERT_EQ(hipMalloc(&p, bBytes), hipSuccess);
        ASSERT_EQ(hipMemset(p, 0x11, bBytes), hipSuccess);
        auto in = std::make_shared<ContractionInputs>();
        in->b   = p;
        ctx.inputArr.push_back(in);
    }
    ASSERT_EQ(hipMemset(ctx.commFlags, 0x55, ctx.commFlagsBytes), hipSuccess);

    auto layout = a2aGemmCounterLayout(3072, 2048, 256, 256, 256, 4, 1);
    ctx.launches = 7;
    ctx.armA2AGemm(layout.allocBytes, layout.payloadBytes, bBytes);

    for(auto p : bufs)
    {
        std::vector<uint8_t> host(bBytes);
        ASSERT_EQ(hipMemcpy(host.data(), p, bBytes, hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_EQ(std::count(host.begin(), host.end(), uint8_t(A2A_GEMM_POISON_BYTE)), (long)bBytes);
        static_cast<void>(hipFree(p));
    }
    std::vector<uint8_t> comm(ctx.commFlagsBytes);
    ASSERT_EQ(hipMemcpy(comm.data(), ctx.commFlags, comm.size(), hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(std::count(comm.begin(), comm.end(), uint8_t(0)), (long)comm.size());

    std::vector<uint8_t> counter(layout.allocBytes);
    ASSERT_GE(ctx.a2aCounterBytes, layout.allocBytes);
    ASSERT_EQ(hipMemcpy(counter.data(), ctx.a2aCounter, counter.size(), hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(std::count(counter.begin(), counter.begin() + layout.payloadBytes, uint8_t(0)),
              (long)layout.payloadBytes);
    EXPECT_EQ(fusedA2ACounterSentinelFirstBad(
                  reinterpret_cast<uint32_t const*>(counter.data() + layout.payloadBytes)),
              -1);
    EXPECT_EQ(ctx.a2aPayloadBytes, layout.payloadBytes);
    EXPECT_EQ(ctx.launches, 0u);
}

TEST(DeviceContext, CommBytesFollowTheMode)
{
    using namespace TensileLite;
    using namespace TensileLite::Client;
    EXPECT_EQ(fusedA2ACommFlagsBytes(FusedA2A::None), 0u);
    EXPECT_EQ(fusedA2ACommFlagsBytes(FusedA2A::GemmA2A), 2 * FUSED_A2A_FLAG_BLOCK_BYTES);
    EXPECT_EQ(fusedA2ACommFlagsBytes(FusedA2A::A2AGemm), FUSED_A2A_FLAG_BLOCK_BYTES);
}
