// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "DeviceContext.hpp"

#include <hip/hip_runtime.h>

using TensileLite::Client::DeviceContext;
using TensileLite::Client::ScopedDevice;

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
