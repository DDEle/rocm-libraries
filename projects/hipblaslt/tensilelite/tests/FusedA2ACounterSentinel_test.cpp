// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Guard-tail detector for the fused GEMM.A2A counter allocation: a 64-byte
// tail is appended past the three-level counter payload (counter[dst_rank*
// tokenTiles+j] over W*tokenTiles slots, a W-entry counter2[dst_rank], and a
// single counter3), filled with a pattern of distinct words, and re-checked
// after each launch. These tests pin the payload/alloc size formulas and
// confirm the detector names the first corrupted word across a range of
// realistic overrun shapes (single word, counter-like values, misaligned
// bytes, wholesale clobber).

#include <cstdint>
#include <cstring>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include <FusedA2ACounterSentinel.hpp>

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    // A filled guard tail, as the client holds it right after allocation.
    std::vector<uint32_t> freshGuard()
    {
        std::vector<uint32_t> g(FUSED_A2A_COUNTER_SENTINEL_WORDS, 0);
        fusedA2ACounterSentinelFill(g.data());
        return g;
    }
}

// ---- Layout ---------------------------------------------------------------

// Pins the payload formula against counterBytes in FusedA2AClient.cpp.
TEST(FusedA2ACounterSentinel, PayloadMatchesTheCounterLayout)
{
    // Three levels ride one allocation, in this order:
    //   [0, W*tokenTiles)              counter[dst_rank*tokenTiles + j]
    //   [W*tokenTiles, W*tokenTiles+W) counter2[dst_rank]
    //   [W*tokenTiles+W]               counter3, the single grid-wide WG tally
    EXPECT_EQ(fusedA2ACounterPayloadBytes(4, 16), (size_t)(4 * 16 + 4 + 1) * sizeof(uint32_t));
    EXPECT_EQ(fusedA2ACounterPayloadBytes(8, 1), (size_t)(8 * 1 + 8 + 1) * sizeof(uint32_t));
    EXPECT_EQ(fusedA2ACounterPayloadBytes(1, 1), (size_t)(1 * 1 + 1 + 1) * sizeof(uint32_t));
}

TEST(FusedA2ACounterSentinel, Counter3SitsImmediatelyAfterCounter2)
{
    // Pins the host-side payload formula only; the kernel's SGPR arithmetic
    // that computes the matching index is not cross-checked here.
    for(uint32_t w : {1u, 2u, 4u, 8u})
    {
        for(uint32_t t : {1u, 8u, 16u})
        {
            const size_t words = fusedA2ACounterPayloadBytes(w, t) / sizeof(uint32_t);
            // One assertion only: `words - 1 == w*t + w` is implied by this.
            EXPECT_EQ(words, (size_t)w * t + w + 1) << "W=" << w << " tokenTiles=" << t;
        }
    }
}

TEST(FusedA2ACounterSentinel, AllocIsPayloadPlusGuardTail)
{
    EXPECT_EQ(FUSED_A2A_COUNTER_SENTINEL_BYTES, (size_t)64);
    EXPECT_EQ(FUSED_A2A_COUNTER_SENTINEL_WORDS,
              FUSED_A2A_COUNTER_SENTINEL_BYTES / sizeof(uint32_t));

    for(uint32_t w : {1u, 2u, 4u, 8u})
    {
        for(uint32_t t : {1u, 3u, 16u, 257u})
        {
            EXPECT_EQ(fusedA2ACounterAllocBytes(w, t),
                      fusedA2ACounterPayloadBytes(w, t) + FUSED_A2A_COUNTER_SENTINEL_BYTES)
                << "W=" << w << " tokenTiles=" << t;
        }
    }
}

// ---- The detector must not cry wolf ---------------------------------------

TEST(FusedA2ACounterSentinel, FilledGuardReadsIntact)
{
    auto g = freshGuard();
    EXPECT_EQ(fusedA2ACounterSentinelFirstBad(g.data()), -1);
}

// ---- The detector must bite ------------------------------------------------

TEST(FusedA2ACounterSentinel, EveryGuardWordIsDistinct)
{
    auto               g = freshGuard();
    std::set<uint32_t> seen(g.begin(), g.end());
    EXPECT_EQ(seen.size(), g.size());
}

// Corrupts each word in turn; the detector must name that word.
TEST(FusedA2ACounterSentinel, DetectsEachCorruptedWord)
{
    for(size_t i = 0; i < FUSED_A2A_COUNTER_SENTINEL_WORDS; i++)
    {
        auto g = freshGuard();
        g[i]   = ~g[i];
        EXPECT_EQ(fusedA2ACounterSentinelFirstBad(g.data()), (int)i)
            << "corrupted word " << i << " went unnoticed";
    }
}

// Realistic overrun values: per-launch memset (zero) or atomic-increment tile counts.
TEST(FusedA2ACounterSentinel, DetectsCounterLikeWrites)
{
    for(uint32_t v : {0u, 1u, 2u, 3u, 16u, 64u, 255u, 4096u})
    {
        for(size_t i = 0; i < FUSED_A2A_COUNTER_SENTINEL_WORDS; i++)
        {
            auto g = freshGuard();
            g[i]   = v;
            EXPECT_EQ(fusedA2ACounterSentinelFirstBad(g.data()), (int)i)
                << "guard word " << i << " overwritten with " << v << " went unnoticed";
        }
    }
}

// An overrun need not be word-aligned (byte-granular or misaligned copies can clip the guard).
TEST(FusedA2ACounterSentinel, DetectsSingleByteCorruption)
{
    for(size_t byteIdx = 0; byteIdx < FUSED_A2A_COUNTER_SENTINEL_BYTES; byteIdx++)
    {
        auto  g = freshGuard();
        auto* b = reinterpret_cast<uint8_t*>(g.data());
        b[byteIdx] ^= 0xFF;
        EXPECT_EQ(fusedA2ACounterSentinelFirstBad(g.data()), (int)(byteIdx / sizeof(uint32_t)))
            << "byte " << byteIdx << " went unnoticed";
    }
}

// A wholesale clobber must report the FIRST bad word, not merely "something is wrong".
TEST(FusedA2ACounterSentinel, ReportsFirstBadWordWhenManyAreCorrupt)
{
    auto g = freshGuard();
    for(size_t i = 5; i < FUSED_A2A_COUNTER_SENTINEL_WORDS; i++)
        g[i] = 0;
    EXPECT_EQ(fusedA2ACounterSentinelFirstBad(g.data()), 5);
}
