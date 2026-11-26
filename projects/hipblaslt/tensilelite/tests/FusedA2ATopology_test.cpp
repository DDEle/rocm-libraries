// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "FusedA2ATopology.hpp"

#include <numeric>
#include <set>
#include <vector>

using namespace TensileLite::Client;

namespace
{
    PciBusId parse(char const* text)
    {
        PciBusId id;
        EXPECT_TRUE(parsePciBusId(text, id)) << text;
        return id;
    }

    // Devices 2c and 2c+1 are functions 0 and 1 of card c.
    std::vector<PciBusId> dpxCards(size_t devices)
    {
        std::vector<PciBusId> ids;
        for(size_t d = 0; d < devices; d++)
            ids.push_back({0, (unsigned)(0x10 + 0x10 * (d / 2)), 0, (unsigned)(d % 2)});
        return ids;
    }

    std::vector<int> iota(int n)
    {
        std::vector<int> v(n);
        std::iota(v.begin(), v.end(), 0);
        return v;
    }
}

TEST(FusedA2ATopology, ParsesPciBusId)
{
    const PciBusId id = parse("0001:c5:1f.1");
    EXPECT_EQ(id.domain, 0x1u);
    EXPECT_EQ(id.bus, 0xc5u);
    EXPECT_EQ(id.device, 0x1fu);
    EXPECT_EQ(id.function, 0x1u);

    PciBusId bad;
    EXPECT_FALSE(parsePciBusId("0000:c5:00", bad));
    EXPECT_FALSE(parsePciBusId("", bad));
}

TEST(FusedA2ATopology, ParsesDirectMode)
{
    EXPECT_EQ(parseFusedA2ADirect("none"), FusedA2ADirect::None);
    EXPECT_EQ(parseFusedA2ADirect("self"), FusedA2ADirect::Self);
    EXPECT_EQ(parseFusedA2ADirect("card"), FusedA2ADirect::Card);
    for(char const* bad : {"true", "false", "", "Card"})
    {
        try
        {
            parseFusedA2ADirect(bad);
            ADD_FAILURE() << "accepted '" << bad << "'";
        }
        catch(std::runtime_error const& e)
        {
            EXPECT_NE(std::string(e.what()).find("none, self or card"), std::string::npos)
                << e.what();
        }
    }
}

TEST(FusedA2ATopology, DirectMaskOfEachMode)
{
    const std::vector<PciBusId> peers = {parse("0000:05:00.0"),
                                         parse("0000:05:00.1"),
                                         parse("0000:05:01.0"),
                                         parse("0000:06:00.0"),
                                         parse("0001:05:00.0")};
    const uint32_t              card[5] = {0x3, 0x3, 0x4, 0x8, 0x10};
    for(size_t d = 0; d < peers.size(); d++)
    {
        EXPECT_EQ(fusedA2ADirectMask(FusedA2ADirect::None, peers[d], peers), 0u) << d;
        EXPECT_EQ(fusedA2ADirectMask(FusedA2ADirect::Self, peers[d], peers), 1u << d) << d;
        EXPECT_EQ(fusedA2ADirectMask(FusedA2ADirect::Card, peers[d], peers), card[d]) << d;
    }
}

TEST(FusedA2ATopology, DirectMaskOfDpxPairs)
{
    const auto peers = dpxCards(8);
    for(size_t d = 0; d < peers.size(); d++)
    {
        EXPECT_EQ(fusedA2ADirectMask(FusedA2ADirect::None, peers[d], peers), 0u) << d;
        EXPECT_EQ(fusedA2ADirectMask(FusedA2ADirect::Self, peers[d], peers), 1u << d) << d;
        EXPECT_EQ(fusedA2ADirectMask(FusedA2ADirect::Card, peers[d], peers),
                  0x3u << (d & ~size_t(1)))
            << d;
    }
}

TEST(FusedA2ATopology, MaskEngineKeepsEngineZero)
{
    for(size_t k = 0; k < 12; k++)
        EXPECT_EQ(sdmaMaskEngine(0x3f, k), k % 6) << "k " << k;
    EXPECT_EQ(sdmaMaskEngine(0x1, 3), 0u);
    EXPECT_EQ(sdmaMaskEngine(0xa4, 0), 2u);
    EXPECT_EQ(sdmaMaskEngine(0xa4, 1), 5u);
    EXPECT_EQ(sdmaMaskEngine(0xa4, 2), 7u);
    EXPECT_EQ(sdmaMaskEngine(0xa4, 3), 2u);
    EXPECT_EQ(sdmaMaskEngine(0, 4), 0u);
}

TEST(FusedA2ATopology, PlanWithoutDirectIsRankOrder)
{
    const auto plan = sdmaQueuePlan(iota(8), 3, 0);
    ASSERT_EQ(plan.size(), 8u);
    const size_t remote[8] = {0, 1, 2, 0, 3, 4, 5, 6};
    for(size_t i = 0; i < plan.size(); i++)
    {
        EXPECT_EQ(plan[i].peer, i);
        EXPECT_FALSE(plan[i].spread) << i;
        EXPECT_EQ(plan[i].k, remote[i]) << i;
    }
}

TEST(FusedA2ATopology, PlanSkipsDirectPeers)
{
    const auto peers = dpxCards(8);
    for(FusedA2ADirect mode : {FusedA2ADirect::Self, FusedA2ADirect::Card})
        for(int self = 0; self < 8; self++)
        {
            const uint32_t mask = fusedA2ADirectMask(mode, peers[self], peers);
            const auto     plan = sdmaQueuePlan(iota(8), self, mask);
            ASSERT_EQ(plan.size(), mode == FusedA2ADirect::Card ? 6u : 7u) << self;

            std::set<uint32_t> engines;
            size_t             i = 0;
            for(size_t j = 0; j < 8; j++)
                if(((mask >> j) & 1u) == 0)
                {
                    EXPECT_EQ(plan[i].peer, j) << self;
                    EXPECT_TRUE(plan[i].spread);
                    EXPECT_EQ(plan[i].k, i);
                    engines.insert(sdmaMaskEngine(0x3f, plan[i].k));
                    i++;
                }
            if(mode == FusedA2ADirect::Card)
                EXPECT_EQ(engines.size(), 6u) << self;
        }
}

TEST(FusedA2ATopology, PlanOfOneCardIsEmpty)
{
    const auto peers = dpxCards(2);
    for(int self = 0; self < 2; self++)
        EXPECT_TRUE(
            sdmaQueuePlan(
                iota(2), self, fusedA2ADirectMask(FusedA2ADirect::Card, peers[self], peers))
                .empty());
}
