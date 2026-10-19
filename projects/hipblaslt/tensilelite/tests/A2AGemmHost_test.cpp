// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "A2AGemmHost.hpp"

#include <cstdint>
#include <vector>

using namespace TensileLite::Client;

TEST(A2AGemmHost, CounterLayoutMatchesTheBypassSizing)
{
    // M=3072, N=2048, MT 256x256, 256 CUs, W=4, S=1: F=12, tokenBlocks=8, batches=(7*12)/256+1=1.
    auto l = a2aGemmCounterLayout(3072, 2048, 256, 256, 256, 4, 1);
    EXPECT_EQ(l.featureBlocks, 12u);
    EXPECT_EQ(l.tokenBlocks, 8u);
    EXPECT_EQ(l.batches, 1u);
    EXPECT_EQ(l.payloadBytes, fusedA2AMode1PayloadBytes(4, 8, 1, 1));
    EXPECT_EQ(l.allocBytes, l.payloadBytes + FUSED_A2A_COUNTER_SENTINEL_BYTES);
    EXPECT_GT(a2aGemmCounterLayout(3072, 2048, 256, 256, 256, 4, 2).payloadBytes, l.payloadBytes);
}

TEST(A2AGemmHost, InterleavePutsRankRShardAtColumnRTimesKLocal)
{
    // W=2, N=2, K=4 (kLocal=2), 2-byte elements; x_r[kl, n, j] = 100*r + 10*j + 2*n + kl.
    const size_t W = 2, N = 2, K = 4, kLocal = 2;
    std::vector<std::vector<uint16_t>> x(W, std::vector<uint16_t>(kLocal * N * W));
    for(size_t r = 0; r < W; r++)
        for(size_t j = 0; j < W; j++)
            for(size_t n = 0; n < N; n++)
                for(size_t kl = 0; kl < kLocal; kl++)
                    x[r][kl + n * kLocal + j * kLocal * N] = uint16_t(100 * r + 10 * j + 2 * n + kl);
    std::vector<void const*> xs{x[0].data(), x[1].data()};
    std::vector<uint16_t>    b(N * K);
    a2aGemmInterleaveB(b.data(), xs, 1, N, K, sizeof(uint16_t));
    // rank 1's B[n, r*kLocal + kl] = x_r[kl, n, 1]
    EXPECT_EQ(b, (std::vector<uint16_t>{10, 11, 110, 111, 12, 13, 112, 113}));
}

TEST(A2AGemmHost, SegmentJOfRankDComesFromRankDPlusJ)
{
    EXPECT_EQ(a2aGemmSegmentSource(0, 1, 4), 1u);
    EXPECT_EQ(a2aGemmSegmentSource(3, 1, 4), 0u);
    EXPECT_EQ(a2aGemmSegmentSource(2, 3, 4), 1u);
}

TEST(A2AGemmHost, CommResidueChecksSeqAndEveryPeersDone)
{
    std::vector<uint32_t> words(TensileLite::FUSED_A2A_FLAG_BLOCK_BYTES / sizeof(uint32_t), 0);
    const size_t seq  = FUSED_A2A_MODE1_SEQ_OFFSET / sizeof(uint32_t);
    const size_t done = FUSED_A2A_MODE1_DONE_OFFSET / sizeof(uint32_t);
    words[seq]      = 5;
    words[done + 0] = 5;
    words[done + 2] = 5;
    EXPECT_EQ(a2aGemmCommResidue(words.data(), 1, 3, 5), "");
    EXPECT_NE(a2aGemmCommResidue(words.data(), 1, 3, 6), "");
    words[done + 2] = 4;
    EXPECT_NE(a2aGemmCommResidue(words.data(), 1, 3, 5), "");
}

TEST(A2AGemmHost, CommResidueRejectsAStrayWordOutsideDoneAndSeq)
{
    std::vector<uint32_t> words(TensileLite::FUSED_A2A_FLAG_BLOCK_BYTES / sizeof(uint32_t), 0);
    const size_t seq  = FUSED_A2A_MODE1_SEQ_OFFSET / sizeof(uint32_t);
    const size_t done = FUSED_A2A_MODE1_DONE_OFFSET / sizeof(uint32_t);
    words[seq]      = 5;
    words[done + 0] = 5;
    words[done + 2] = 5;
    for(size_t stray : {done + TensileLite::FUSED_A2A_MAX_RANKS, seq - 1, seq + 1, words.size() - 1})
    {
        words[stray] = 1;
        EXPECT_NE(a2aGemmCommResidue(words.data(), 1, 3, 5), "") << "word " << stray;
        words[stray] = 0;
    }
    EXPECT_EQ(a2aGemmCommResidue(words.data(), 1, 3, 5), "");
}

TEST(A2AGemmHost, CommResidueRejectsADoneSlotBeyondTheWorld)
{
    std::vector<uint32_t> words(TensileLite::FUSED_A2A_FLAG_BLOCK_BYTES / sizeof(uint32_t), 0);
    const size_t seq  = FUSED_A2A_MODE1_SEQ_OFFSET / sizeof(uint32_t);
    const size_t done = FUSED_A2A_MODE1_DONE_OFFSET / sizeof(uint32_t);
    words[seq]      = 5;
    words[done + 0] = 5;
    words[done + 2] = 5;
    for(size_t s = 3; s < TensileLite::FUSED_A2A_MAX_RANKS; s++)
    {
        words[done + s] = 5;
        EXPECT_NE(a2aGemmCommResidue(words.data(), 1, 3, 5), "") << "done[" << s << "]";
        words[done + s] = 0;
    }
}

TEST(A2AGemmHost, CommResidueRequiresOwnDoneToStayZero)
{
    std::vector<uint32_t> words(TensileLite::FUSED_A2A_FLAG_BLOCK_BYTES / sizeof(uint32_t), 0);
    const size_t seq  = FUSED_A2A_MODE1_SEQ_OFFSET / sizeof(uint32_t);
    const size_t done = FUSED_A2A_MODE1_DONE_OFFSET / sizeof(uint32_t);
    words[seq]      = 5;
    words[done + 0] = 5;
    words[done + 2] = 5;
    words[done + 1] = 5;
    EXPECT_NE(a2aGemmCommResidue(words.data(), 1, 3, 5), "");
}

TEST(A2AGemmHost, CommResidueRequiresASingleRankBlockToStayZero)
{
    std::vector<uint32_t> words(TensileLite::FUSED_A2A_FLAG_BLOCK_BYTES / sizeof(uint32_t), 0);
    const size_t seq  = FUSED_A2A_MODE1_SEQ_OFFSET / sizeof(uint32_t);
    const size_t done = FUSED_A2A_MODE1_DONE_OFFSET / sizeof(uint32_t);
    EXPECT_EQ(a2aGemmCommResidue(words.data(), 0, 1, 5), "");
    for(size_t stray : {seq, done, done + 1, size_t(0), words.size() - 1})
    {
        words[stray] = 5;
        EXPECT_NE(a2aGemmCommResidue(words.data(), 0, 1, 5), "") << "word " << stray;
        words[stray] = 0;
    }
}
