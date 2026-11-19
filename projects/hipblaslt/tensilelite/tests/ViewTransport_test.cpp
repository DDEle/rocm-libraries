// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ViewTransport.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    constexpr uint64_t kSrc  = 0x100000000ull;
    constexpr uint64_t kDst  = 0x200000000ull;
    constexpr uint64_t kFlag = 0x300000040ull;
    constexpr uint64_t kView = 0x400000000ull;

    template <typename T>
    T* fakePointer(uint64_t address)
    {
        return reinterpret_cast<T*>(static_cast<uintptr_t>(address));
    }

    uint32_t lo(uint64_t v)
    {
        return static_cast<uint32_t>(v);
    }

    uint32_t hi(uint64_t v)
    {
        return static_cast<uint32_t>(v >> 32);
    }

    // W=3, rank 1, queue 0 (source 2); ldb 512 bf16 = 64 packet elements, S=2, MT1 128,
    // 200 tokens: two token blocks, the second 72 rows.
    ViewArmArgs a2aGemmArgs(uint32_t coalesce)
    {
        ViewArmArgs a{};
        a.mode              = VIEW_MODE_A2A_GEMM;
        a.rank              = 1;
        a.world             = 3;
        a.numQueues         = 2;
        a.subRounds         = 2;
        a.tokenBlocks       = 2;
        a.featureTiles      = 1;
        a.numCu             = 256;
        a.mt1               = 128;
        a.nTokens           = 200;
        a.srcPitch          = 64;
        a.dstPitch          = 64;
        a.slice             = 64 * 128;
        a.rectX             = 32;
        a.srcBlockStride    = 128 * 1024;
        a.dstBlockStride    = 128 * 1024;
        a.subRoundStride    = 512;
        a.coalesce          = coalesce;
        a.view              = fakePointer<uint32_t>(kView);
        a.queues[0].srcBase = kSrc;
        a.queues[0].dstBase = kDst;
        a.queues[0].flag    = fakePointer<uint32_t>(kFlag);
        return a;
    }

    // W=2, rank 0, queue 1; nShard 256 bf16 = 32 packet elements, ldd 3072 = 384 packet
    // elements, MT0 128 (tilesPerRank 2), MT1 256, 511 tokens: T=2, the second tile 255 rows.
    ViewArmArgs gemmA2AArgs()
    {
        ViewArmArgs a{};
        a.mode              = VIEW_MODE_GEMM_A2A;
        a.rank              = 0;
        a.world             = 2;
        a.numQueues         = 2;
        a.subRounds         = 1;
        a.tokenBlocks       = 2;
        a.mt1               = 256;
        a.nTokens           = 511;
        a.srcPitch          = 384;
        a.dstPitch          = 32;
        a.slice             = 256 * 32;
        a.rectX             = 32;
        a.srcBlockStride    = 256 * 3072 * 2;
        a.dstBlockStride    = 256 * 256 * 2;
        a.tilesPerRank      = 2;
        a.coalesce          = 1;
        a.view              = fakePointer<uint32_t>(kView);
        a.queues[1].srcBase = kSrc;
        a.queues[1].dstBase = kDst;
        a.queues[1].flag    = fakePointer<uint32_t>(kFlag);
        return a;
    }

    std::vector<uint32_t> build(ViewArmArgs const& a, uint32_t q, uint32_t epoch)
    {
        std::vector<uint32_t> ring(1024, 0xDEADBEEFu);
        ViewRingWriter        w{ring.data(), ring.size(), 0};
        viewBuildChain(a, q, epoch, w);
        ViewRingWriter count{nullptr, ring.size(), 0};
        viewBuildChain(a, q, epoch, count);
        EXPECT_EQ(count.pos, w.pos);
        ring.resize(w.pos);
        return ring;
    }

    void expectCopy(std::vector<uint32_t> const& r,
                    size_t                       at,
                    uint64_t                     src,
                    uint64_t                     dst,
                    uint32_t                     srcPitch,
                    uint32_t                     dstPitch,
                    uint32_t                     slice,
                    uint32_t                     rectX,
                    uint32_t                     rectY)
    {
        ASSERT_LE(at + VIEW_COPY_DWORDS, r.size());
        std::vector<uint32_t> expected = {0x80000401u,
                                          lo(src),
                                          hi(src),
                                          0u,
                                          (srcPitch - 1) << 13,
                                          slice - 1,
                                          lo(dst),
                                          hi(dst),
                                          0u,
                                          (dstPitch - 1) << 13,
                                          slice - 1,
                                          ((rectY - 1) << 16) | (rectX - 1),
                                          0u};
        EXPECT_EQ(std::vector<uint32_t>(r.begin() + at, r.begin() + at + VIEW_COPY_DWORDS),
                  expected)
            << "COPY at dword " << at;
    }

    void expectAtomic(std::vector<uint32_t> const& r, size_t at, uint64_t addr, uint32_t addend)
    {
        ASSERT_LE(at + VIEW_ATOMIC_DWORDS, r.size());
        std::vector<uint32_t> expected = {0x1E00000Au, lo(addr), hi(addr), addend, 0u, 0u, 0u, 0u};
        EXPECT_EQ(std::vector<uint32_t>(r.begin() + at, r.begin() + at + VIEW_ATOMIC_DWORDS),
                  expected)
            << "ATOMIC at dword " << at;
    }

    void expectPoll(std::vector<uint32_t> const& r, size_t at, uint64_t addr, uint32_t ref)
    {
        ASSERT_LE(at + VIEW_POLL_DWORDS, r.size());
        std::vector<uint32_t> expected
            = {0xD0000008u, lo(addr), hi(addr), ref, 0xFFFFFFFFu, 0x0FFF000Au};
        EXPECT_EQ(std::vector<uint32_t>(r.begin() + at, r.begin() + at + VIEW_POLL_DWORDS),
                  expected)
            << "POLL at dword " << at;
    }

    struct FakeQueue
    {
        std::vector<uint32_t> ring;
        uint64_t              rptr = 0, wptr = 0, doorbell = 0;

        FakeQueue(size_t dwords, uint64_t start)
            : ring(dwords, 0xDEADBEEFu)
            , rptr(start)
            , wptr(start)
        {
        }

        void attach(ViewArmArgs& a, uint32_t q)
        {
            a.queues[q].ring      = ring.data();
            a.queues[q].ringBytes = ring.size() * sizeof(uint32_t);
            a.queues[q].rptr      = &rptr;
            a.queues[q].wptr      = &wptr;
            a.queues[q].doorbell  = &doorbell;
        }
    };

    using Adds = std::vector<std::pair<uint64_t, uint32_t>>;

    auto recordTo(Adds& adds)
    {
        return [&adds](uint32_t* p, uint32_t v) { adds.push_back({viewAddress(p), v}); };
    }

    uint64_t a2aGemmSrc(uint32_t b, uint32_t h)
    {
        return kSrc + b * 128 * 1024 + h * 512;
    }

    uint64_t a2aGemmDst(uint32_t b, uint32_t h)
    {
        return kDst + b * 128 * 1024 + h * 512;
    }

    constexpr uint64_t kBlockStride = 1ull << 20;
    constexpr uint64_t kSubStride   = 1ull << 8;
    constexpr uint32_t kAll         = 1u << 30;

    // A2AGemm with one-row token blocks and every queue's flag at kFlag + 4 * q.
    ViewArmArgs chainArgs(uint32_t world,
                          uint32_t blocks,
                          uint32_t features,
                          uint32_t numCu,
                          uint32_t subRounds,
                          uint32_t coalesce)
    {
        ViewArmArgs a{};
        a.mode           = VIEW_MODE_A2A_GEMM;
        a.world          = world;
        a.numQueues      = world - 1;
        a.subRounds      = subRounds;
        a.tokenBlocks    = blocks;
        a.featureTiles   = features;
        a.numCu          = numCu;
        a.mt1            = 1;
        a.nTokens        = blocks;
        a.srcPitch       = 1;
        a.dstPitch       = 1;
        a.slice          = 1;
        a.rectX          = 1;
        a.srcBlockStride = kBlockStride;
        a.dstBlockStride = kBlockStride;
        a.subRoundStride = kSubStride;
        a.coalesce       = coalesce;
        a.view           = fakePointer<uint32_t>(kView);
        for(uint32_t q = 0; q + 1 < world; q++)
        {
            a.queues[q].srcBase = kSrc;
            a.queues[q].dstBase = kDst;
            a.queues[q].flag    = fakePointer<uint32_t>(kFlag + 4 * q);
        }
        return a;
    }

    // Queue q's A2AGemm chain: C<b>[.<h>] per COPY, A+<addend> per ATOMIC.
    std::string chainText(ViewArmArgs const& a, uint32_t q)
    {
        ViewRingWriter count{nullptr, 1ull << 40, 0};
        viewBuildUnits(a, q, 0, count);
        uint64_t dwords = 1;
        while(dwords <= count.pos)
            dwords *= 2;
        std::vector<uint32_t> ring(dwords, 0xDEADBEEFu);
        ViewRingWriter        w{ring.data(), dwords, 0};
        viewBuildUnits(a, q, 0, w);
        std::string out;
        for(uint64_t at = 0; at < w.pos;)
        {
            out += out.empty() ? "" : " ";
            if(ring[at] == VIEW_COPY_HEADER)
            {
                const uint64_t off = ((uint64_t)ring[at + 2] << 32 | ring[at + 1]) - a.queues[q].srcBase;
                out += "C" + std::to_string(off / a.srcBlockStride);
                if(a.subRounds > 1)
                    out += "." + std::to_string(off % a.srcBlockStride / a.subRoundStride);
                at += VIEW_COPY_DWORDS;
            }
            else if(ring[at] == VIEW_ATOMIC_HEADER)
            {
                out += "A+" + std::to_string(ring[at + 3]);
                at += VIEW_ATOMIC_DWORDS;
            }
            else
            {
                ADD_FAILURE() << "dword " << at << " is no COPY or ATOMIC header";
                break;
            }
        }
        return out;
    }
}

TEST(ViewTransport, A2AGemmOneAtomicPerUnit)
{
    auto const r = build(a2aGemmArgs(1), 0, 0);
    ASSERT_EQ(r.size(), 4 * (VIEW_COPY_DWORDS + VIEW_ATOMIC_DWORDS));
    size_t at = 0;
    for(uint32_t b = 0; b < 2; b++)
        for(uint32_t h = 0; h < 2; h++)
        {
            expectCopy(r, at, a2aGemmSrc(b, h), a2aGemmDst(b, h), 64, 64, 8192, 32, b ? 72 : 128);
            at += VIEW_COPY_DWORDS;
            expectAtomic(r, at, kFlag, 1);
            at += VIEW_ATOMIC_DWORDS;
        }
}

// W=3, B=2, S=2, coalesce 3: groups {0} {1}; coalescing stops at every sub-round.
TEST(ViewTransport, A2AGemmCoalescedFlushesEachSubRound)
{
    auto const r = build(a2aGemmArgs(3), 0, 0);
    ASSERT_EQ(r.size(), 4 * (VIEW_COPY_DWORDS + VIEW_ATOMIC_DWORDS));
    size_t at = 0;
    for(uint32_t b = 0; b < 2; b++)
        for(uint32_t h = 0; h < 2; h++)
        {
            expectCopy(r, at, a2aGemmSrc(b, h), a2aGemmDst(b, h), 64, 64, 8192, 32, b ? 72 : 128);
            at += VIEW_COPY_DWORDS;
            expectAtomic(r, at, kFlag, 1);
            at += VIEW_ATOMIC_DWORDS;
        }
}

// W=3, B=3, S=1, Qb=2: groups {0, 1} {2}. Queue 1 serves group 1, whose ATOMIC also carries the
// missing unit, then group 0; queue 0 serves group 0, then group 1, whose ATOMIC ends the chain.
TEST(ViewTransport, A2AGemmStaggersQueuesAndPadsShortRounds)
{
    ViewArmArgs a    = a2aGemmArgs(1);
    a.subRounds      = 1;
    a.tokenBlocks    = 3;
    a.nTokens        = 3 * 128;
    a.queues[1]      = a.queues[0];
    using Step       = std::pair<uint32_t, uint32_t>;
    auto expectSteps = [](std::vector<uint32_t> const& r, std::vector<Step> const& steps) {
        ASSERT_EQ(r.size(), steps.size() * (VIEW_COPY_DWORDS + VIEW_ATOMIC_DWORDS));
        size_t at = 0;
        for(auto [b, addend] : steps)
        {
            expectCopy(r, at, a2aGemmSrc(b, 0), a2aGemmDst(b, 0), 64, 64, 8192, 32, 128);
            at += VIEW_COPY_DWORDS;
            expectAtomic(r, at, kFlag, addend);
            at += VIEW_ATOMIC_DWORDS;
        }
    };
    expectSteps(build(a, 1, 0), {{2, 2}, {0, 1}, {1, 1}});
    expectSteps(build(a, 0, 0), {{0, 1}, {1, 1}, {2, 2}});

    a.tokenBlocks = 1;
    a.nTokens     = 128;
    expectSteps(build(a, 1, 0), {{0, 2}});
}

// W=8, B=1, S=2, coalesce 1: queue 0 copies block 0 in round 0; the ATOMIC after its last
// unit carries rounds 1..6, so nothing follows it.
TEST(ViewTransport, A2AGemmFoldsTrailingEmptyRoundsIntoTheLastAtomic)
{
    ViewArmArgs a = a2aGemmArgs(1);
    a.world       = 8;
    a.numQueues   = 7;
    a.tokenBlocks = 1;
    a.nTokens     = 128;
    auto const r  = build(a, 0, 0);
    ASSERT_EQ(r.size(), 2 * (VIEW_COPY_DWORDS + VIEW_ATOMIC_DWORDS));
    expectCopy(r, 0, a2aGemmSrc(0, 0), a2aGemmDst(0, 0), 64, 64, 8192, 32, 128);
    expectAtomic(r, VIEW_COPY_DWORDS, kFlag, 1);
    expectCopy(r,
               VIEW_COPY_DWORDS + VIEW_ATOMIC_DWORDS,
               a2aGemmSrc(0, 1),
               a2aGemmDst(0, 1),
               64,
               64,
               8192,
               32,
               128);
    expectAtomic(r, 2 * VIEW_COPY_DWORDS + VIEW_ATOMIC_DWORDS, kFlag, 7 * 2 - 1);
}

// W=8, B=16, F=16, S=1, coalesce >= Qb: groups {0,1,2} .. {12,13,14} {15} {} (GK=Qb=3). Queue q
// serves groups q, q-1, .., 0, 6, .., q+1 with one ATOMIC each; the empty group's Qb rides on the
// ATOMIC before it, on queue 6 on the first one.
TEST(ViewTransport, A2AGemmServesContiguousGroupsInInKernelOrder)
{
    ViewArmArgs const a = chainArgs(8, 16, 16, 256, 1, kAll);
    EXPECT_EQ(chainText(a, 0),
              "C0 C1 C2 A+6 C15 A+3 C12 C13 C14 A+3 C9 C10 C11 A+3 C6 C7 C8 A+3 C3 C4 C5 A+3");
    EXPECT_EQ(chainText(a, 1),
              "C3 C4 C5 A+3 C0 C1 C2 A+6 C15 A+3 C12 C13 C14 A+3 C9 C10 C11 A+3 C6 C7 C8 A+3");
    EXPECT_EQ(chainText(a, 5),
              "C15 A+3 C12 C13 C14 A+3 C9 C10 C11 A+3 C6 C7 C8 A+3 C3 C4 C5 A+3 C0 C1 C2 A+6");
    EXPECT_EQ(chainText(a, 6),
              "C15 A+6 C12 C13 C14 A+3 C9 C10 C11 A+3 C6 C7 C8 A+3 C3 C4 C5 A+3 C0 C1 C2 A+3");
}

// W=4, B=4, S=2: groups {0, 1} {2, 3} {} (Qb=2), each sent once per sub-round, sub-round outer.
TEST(ViewTransport, A2AGemmSendsEachGroupPerSubRoundAndFoldsEmptyGroups)
{
    ViewArmArgs const a = chainArgs(4, 4, 1, 256, 2, kAll);
    EXPECT_EQ(chainText(a, 0), "C0.0 C1.0 A+2 C0.1 C1.1 A+6 C2.0 C3.0 A+2 C2.1 C3.1 A+2");
    EXPECT_EQ(chainText(a, 1), "C2.0 C3.0 A+2 C2.1 C3.1 A+2 C0.0 C1.0 A+2 C0.1 C1.1 A+6");
    EXPECT_EQ(chainText(a, 2), "C2.0 C3.0 A+6 C2.1 C3.1 A+2 C0.0 C1.0 A+2 C0.1 C1.1 A+2");
}

// W=4, B=4, S=1: queue 2's first round has no block; its Qb rides on the chain's first ATOMIC,
// also when that ATOMIC comes before the group's last unit.
TEST(ViewTransport, A2AGemmLeadingEmptyRoundRidesOnTheFirstAtomic)
{
    EXPECT_EQ(chainText(chainArgs(4, 4, 1, 256, 1, kAll), 2), "C2 C3 A+4 C0 C1 A+2");
    EXPECT_EQ(chainText(chainArgs(4, 4, 1, 256, 1, 1), 2), "C2 A+3 C3 A+1 C0 A+1 C1 A+1");
    EXPECT_EQ(chainText(chainArgs(4, 4, 1, 256, 2, 1), 2),
              "C2.0 A+5 C3.0 A+1 C2.1 A+1 C3.1 A+1 C0.0 A+1 C1.0 A+1 C0.1 A+1 C1.1 A+1");
}

// numCu = 256: InKernel's batches where numCu / F does not divide, and F > numCu.
TEST(ViewTransport, A2AGemmBatchesFollowTheInKernelPartition)
{
    struct Case
    {
        uint32_t                                   world, blocks, features, numCu, qb;
        std::vector<std::pair<uint32_t, uint32_t>> spans;
    };
    const std::vector<Case> cases = {{3, 64, 4, 256, 32, {{0, 64}}},
                                     {8, 44, 12, 256, 4, {{0, 22}, {22, 43}, {43, 44}}},
                                     {4, 16, 24, 256, 4, {{0, 11}, {11, 16}}},
                                     {2, 3, 300, 256, 1, {{0, 1}, {1, 2}, {2, 3}}}};
    for(auto const& c : cases)
    {
        SCOPED_TRACE(testing::Message() << "W=" << c.world << " B=" << c.blocks << " F=" << c.features);
        ViewArmArgs const a = chainArgs(c.world, c.blocks, c.features, c.numCu, 1, 1);
        ASSERT_EQ(viewA2AGemmBatches(a), c.spans.size());
        EXPECT_EQ(viewA2AGemmBatchBlocksMax(a), c.spans[0].second);
        EXPECT_EQ(viewA2AGemmRoundUnits(a), c.qb);
        for(uint32_t iB = 0; iB < c.spans.size(); iB++)
        {
            EXPECT_EQ(viewA2AGemmBatchLo(a, iB), c.spans[iB].first) << "batch " << iB;
            EXPECT_EQ(viewA2AGemmBatchHi(a, iB), c.spans[iB].second) << "batch " << iB;
        }
    }
}

// B = 0: no batch, no packet, nothing booked.
TEST(ViewTransport, A2AGemmEmptyTokensBuildsNothing)
{
    ViewArmArgs a = a2aGemmArgs(1);
    a.tokenBlocks = 0;
    a.nTokens     = 0;
    for(uint32_t features : {1u, 12u, 300u})
    {
        SCOPED_TRACE(testing::Message() << "F=" << features);
        a.featureTiles = features;
        EXPECT_EQ(viewA2AGemmBatches(a), 0u);
        EXPECT_EQ(viewA2AGemmChainUnits(a), 0u);
        EXPECT_TRUE(build(a, 0, 0).empty());
        Adds adds;
        viewBookSkipped(a, 0, recordTo(adds));
        EXPECT_EQ(adds, (Adds{{kFlag, 0u}}));
    }
}

TEST(ViewTransport, A2AGemmEmptyFeaturesBuildsNothing)
{
    ViewArmArgs a  = a2aGemmArgs(1);
    a.featureTiles = 0;
    for(uint32_t blocks : {0u, 1u, 5u, 300u})
    {
        SCOPED_TRACE(testing::Message() << "B=" << blocks);
        a.tokenBlocks = blocks;
        a.nTokens     = blocks * a.mt1;
        EXPECT_EQ(viewA2AGemmBatches(a), 0u);
        EXPECT_EQ(viewA2AGemmBatchBlocksMax(a), 0u);
        EXPECT_EQ(viewA2AGemmRoundUnits(a), 0u);
        EXPECT_EQ(viewA2AGemmChainUnits(a), 0u);
        EXPECT_TRUE(build(a, 0, 0).empty());
        Adds adds;
        viewBookSkipped(a, 0, recordTo(adds));
        EXPECT_EQ(adds, (Adds{{kFlag, 0u}}));
    }
}

// Walks every queue's chain for many (W, B, S, coalesce, F, numCu) and follows each work-group's
// segment switches as the kernel does: F' = min(F, numCu), batch iB = b * F' / numCu, groups of
// GK = ceil(n / G) blocks of the batch, Gb = (b - bLo) / GK, pos = (b - bLo) % GK; queue
// i = A2AShardIdx - 1 starting at Gb, ordinal K starting at S * iB * G * Qb + pos and stepping
// by Qb at every switch. Against InKernel's a2aEnqueue order (per non-empty group, per h, the
// group's COPYs, then one ATOMIC) no unit is released after more COPYs of its queue; with
// coalesce >= Qb each one after as many. Each unit is released within coalesce - 1 COPYs after
// its own, and each queue issues ceil(|group| / coalesce) ATOMICs per non-empty (group, h).
TEST(ViewTransport, A2AGemmChainReleasesEveryUnitAfterItsCopy)
{
    // (F, numCu): numCu / F of 1, 2, 3 and 8 blocks, not dividing, numCu = 256 and numCu < F.
    const std::vector<std::pair<uint32_t, uint32_t>> grids = {{8, 8},
                                                              {8, 16},
                                                              {8, 24},
                                                              {8, 64},
                                                              {3, 4},
                                                              {3, 7},
                                                              {3, 17},
                                                              {7, 31},
                                                              {4, 256},
                                                              {11, 256},
                                                              {12, 256},
                                                              {16, 256},
                                                              {24, 256},
                                                              {32, 31},
                                                              {7, 3}};
    for(uint32_t world : {2u, 3u, 4u, 5u, 6u, 7u, 8u})
        for(uint32_t blocks : {0u, 1u, 2u, 3u, 5u, 7u, 11u, 16u, 37u, 44u, 64u})
        {
            auto fn = grids;
            fn.push_back({8, 8 * std::max(blocks, 1u)});
            fn.push_back({8, 8 * (blocks + 5)});
            for(uint32_t subRounds : {1u, 2u, 4u})
                for(uint32_t coalesce : {1u, 3u, 4u, kAll})
                    for(auto [features, numCu] : fn)
                    {
                        SCOPED_TRACE(testing::Message()
                                     << "W=" << world << " B=" << blocks << " S=" << subRounds
                                     << " c=" << coalesce << " F=" << features
                                     << " numCu=" << numCu);
                        const uint32_t g  = world - 1;
                        const uint32_t fp = std::min(features, numCu);
                        // InKernel's batches, a2aBatchSpan on F.
                        std::vector<std::pair<uint32_t, uint32_t>> spans;
                        uint32_t                                   nbMax = 0;
                        for(uint32_t b = 0; b < blocks; b++)
                        {
                            const uint32_t iB = b * features / numCu;
                            const uint32_t lo = (iB * numCu + features - 1) / features;
                            const uint32_t hi = std::min(
                                (iB * numCu + numCu + features - 1) / features, blocks);
                            ASSERT_TRUE(lo <= b && b < hi);
                            if(spans.empty() || spans.back().first != lo)
                                spans.push_back({lo, hi});
                            nbMax = std::max(nbMax, hi - lo);
                        }
                        const uint32_t qb    = (nbMax + g - 1) / g;
                        const uint32_t total = spans.size() * g * qb * subRounds;
                        ViewArmArgs    a
                            = chainArgs(world, blocks, features, numCu, subRounds, coalesce);
                        ASSERT_EQ(viewA2AGemmBatches(a), spans.size());
                        ASSERT_EQ(viewA2AGemmBatchBlocksMax(a), nbMax);
                        for(uint32_t iB = 0; iB < spans.size(); iB++)
                        {
                            ASSERT_EQ(viewA2AGemmBatchLo(a, iB), spans[iB].first);
                            ASSERT_EQ(viewA2AGemmBatchHi(a, iB), spans[iB].second);
                        }
                        ASSERT_EQ(viewA2AGemmRoundUnits(a), qb * subRounds);
                        ASSERT_EQ(viewA2AGemmChainUnits(a), total);

                        // Per queue: packet index and COPY count through each unit's COPY, and
                        // the (packet index, flag[i] after it, COPYs before it) of each ATOMIC.
                        struct Atomic
                        {
                            size_t   at;
                            uint32_t flag;
                            uint32_t copies;
                        };
                        std::vector<std::vector<size_t>>   copyAt(g);
                        std::vector<std::vector<uint32_t>> copyNo(g);
                        std::vector<std::vector<Atomic>>   atomics(g);
                        std::vector<size_t>                packets(g, 0);
                        for(uint32_t i = 0; i < g; i++)
                        {
                            std::vector<uint32_t> ring(16384, 0xDEADBEEFu);
                            ViewRingWriter        w{ring.data(), ring.size(), 0};
                            viewBuildChain(a, i, 0, w);
                            ASSERT_LT(w.pos, ring.size());
                            std::vector<uint32_t> copied(blocks * subRounds, 0);
                            copyAt[i].assign(blocks * subRounds, 0);
                            copyNo[i].assign(blocks * subRounds, 0);
                            uint32_t counter = 0, copies = 0;
                            for(uint64_t at = 0; at < w.pos; packets[i]++)
                            {
                                if(ring[at] == VIEW_COPY_HEADER)
                                {
                                    const uint64_t off
                                        = ((uint64_t)ring[at + 2] << 32 | ring[at + 1]) - kSrc;
                                    const uint32_t b = (uint32_t)(off / kBlockStride);
                                    const uint32_t h
                                        = (uint32_t)(off % kBlockStride / kSubStride);
                                    ASSERT_LT(b, blocks);
                                    ASSERT_LT(h, subRounds);
                                    copied[b * subRounds + h]++;
                                    copyAt[i][b * subRounds + h] = packets[i];
                                    copies++;
                                    copyNo[i][b * subRounds + h] = copies;
                                    at += VIEW_COPY_DWORDS;
                                }
                                else
                                {
                                    ASSERT_EQ(ring[at], VIEW_ATOMIC_HEADER) << "dword " << at;
                                    ASSERT_EQ(((uint64_t)ring[at + 2] << 32 | ring[at + 1]),
                                              kFlag + 4 * i);
                                    ASSERT_GT(ring[at + 3], 0u);
                                    counter += ring[at + 3];
                                    atomics[i].push_back({packets[i], counter, copies});
                                    at += VIEW_ATOMIC_DWORDS;
                                }
                            }
                            EXPECT_EQ(copied, std::vector<uint32_t>(blocks * subRounds, 1))
                                << "queue " << i;
                            EXPECT_EQ(counter, total) << "queue " << i;
                        }

                        Adds booked;
                        viewBookSkipped(a, 0, recordTo(booked));
                        EXPECT_EQ(booked, (Adds{{kFlag, total}}));
                        if(blocks == 0)
                        {
                            for(uint32_t i = 0; i < g; i++)
                                EXPECT_EQ(packets[i], 0u) << "queue " << i;
                            continue;
                        }

                        // InKernel's release position (COPYs on queue i up to the unit's ATOMIC)
                        // and ATOMIC count, a2aEnqueue's ring order per batch; the ATOMIC count
                        // with one ATOMIC per coalesce units of each (group, h).
                        std::vector<std::vector<uint32_t>> inkRelease(
                            g, std::vector<uint32_t>(blocks * subRounds, 0));
                        std::vector<uint32_t> inkAtomics(g, 0), coalescedAtomics(g, 0);
                        for(uint32_t i = 0; i < g; i++)
                        {
                            uint32_t copies = 0;
                            for(auto [lo, hi] : spans)
                            {
                                const int64_t n  = hi - lo;
                                const int64_t gk = (n + g - 1) / g;
                                int64_t       left = n, row = lo + i * gk;
                                while(left > 0)
                                {
                                    if(row < lo + n)
                                    {
                                        const int64_t nBlk = std::min<int64_t>(lo + n - row, gk);
                                        for(uint32_t h = 0; h < subRounds; h++)
                                        {
                                            copies += nBlk;
                                            for(int64_t b = row; b < row + nBlk; b++)
                                                inkRelease[i][b * subRounds + h] = copies;
                                            inkAtomics[i]++;
                                            coalescedAtomics[i] += (nBlk + coalesce - 1) / coalesce;
                                        }
                                        left -= nBlk;
                                    }
                                    row -= gk;
                                    if(row < lo)
                                        row += g * gk;
                                }
                            }
                        }

                        std::vector<size_t> lastRelease(g, 0);
                        for(uint32_t b = 0; b < blocks; b++)
                        {
                            const uint32_t iB = b * fp / numCu;
                            ASSERT_LT(iB, spans.size());
                            const uint32_t bLo = spans[iB].first;
                            ASSERT_TRUE(bLo <= b && b < spans[iB].second);
                            const uint32_t gk    = (spans[iB].second - bLo + g - 1) / g;
                            const uint32_t gb    = (b - bLo) / gk;
                            const uint32_t pos   = (b - bLo) % gk;
                            uint32_t       shard = gb + 1;
                            uint32_t       k     = subRounds * iB * g * qb + pos;
                            for(uint32_t j = 0; j < g; j++)
                                for(uint32_t h = 0; h < subRounds; h++)
                                {
                                    SCOPED_TRACE(testing::Message()
                                                 << "b=" << b << " j=" << j << " h=" << h);
                                    const uint32_t i = shard - 1;
                                    ASSERT_EQ(i, (gb + j) % g);
                                    ASSERT_EQ(k, qb * (subRounds * (iB * g + j) + h) + pos);
                                    const size_t copy = copyAt[i][b * subRounds + h];
                                    auto const&  at   = atomics[i];
                                    auto const   next = std::find_if(
                                        at.begin(), at.end(), [&](auto const& e) {
                                            return e.at > copy;
                                        });
                                    ASSERT_NE(next, at.end()) << "no ATOMIC after the COPY";
                                    ASSERT_GE(next->flag, k + 1);
                                    auto const release = std::find_if(
                                        at.begin(), at.end(), [&](auto const& e) {
                                            return e.flag >= k + 1;
                                        });
                                    EXPECT_GT(release->at, copy) << "released before its COPY";
                                    const uint32_t ink = inkRelease[i][b * subRounds + h];
                                    ASSERT_LE(release->copies, ink) << "released after InKernel";
                                    if(coalesce >= qb)
                                        ASSERT_EQ(release->copies, ink) << "released before InKernel";
                                    ASSERT_LE(release->copies - copyNo[i][b * subRounds + h],
                                              coalesce - 1)
                                        << "released more than coalesce - 1 COPYs late";
                                    lastRelease[i] = std::max(lastRelease[i], release->at);
                                    k += qb;
                                    if(h == subRounds - 1)
                                        shard = shard + 1 < world ? shard + 1 : 1;
                                }
                        }
                        for(uint32_t i = 0; i < g; i++)
                        {
                            EXPECT_EQ(lastRelease[i] + 1, packets[i])
                                << "queue " << i << ": packets after the last awaited ATOMIC";
                            ASSERT_FALSE(atomics[i].empty());
                            EXPECT_EQ(atomics[i].back().at + 1, packets[i])
                                << "queue " << i << ": the chain does not end with an ATOMIC";
                            if(coalesce >= qb)
                                EXPECT_EQ(atomics[i].size(), inkAtomics[i])
                                    << "queue " << i << ": ATOMICs against InKernel";
                            EXPECT_EQ(atomics[i].size(), coalescedAtomics[i])
                                << "queue " << i << ": ATOMICs against one per coalesce units";
                        }
                    }
        }
}

TEST(ViewTransport, A2AGemmBooksRoundedRoundsAndAdvancesOnlyTheEpoch)
{
    ViewArmArgs a = a2aGemmArgs(1);
    a.world       = 4;
    a.numQueues   = 3;
    a.tokenBlocks = 5;
    Adds adds;
    viewBookSkipped(a, 0, recordTo(adds));
    EXPECT_EQ(adds, (Adds{{kFlag, 3 * 2 * 2}}));
    uint32_t header[FUSED_A2A_VIEW_SIG_OFFSET / 4] = {};
    viewAdvance(a, 0, header);
    viewAdvance(a, 1, header);
    uint32_t expected[FUSED_A2A_VIEW_SIG_OFFSET / 4] = {};
    expected[FUSED_A2A_VIEW_EPOCH_OFFSET / 4]        = 2;
    EXPECT_TRUE(std::equal(std::begin(header), std::end(header), std::begin(expected)));
}

// A2AGemm queue 0 is drained: flag[0] goes from the previous launch's NB*G*Qb*S to 0 before any
// ring dword, wptr or doorbell is written, and the chain is the one viewBuildChain builds.
TEST(ViewTransport, A2AGemmArmResetsTheFlagBeforeTheChain)
{
    ViewArmArgs a = a2aGemmArgs(1);
    FakeQueue   aq(1024, 400);
    aq.attach(a, 0);
    uint32_t flag    = 2 * 1 * 2;
    a.queues[0].flag = &flag;
    Adds stores;
    bool ringUntouched = false;
    auto store         = [&](uint32_t* p, uint32_t v) {
        ringUntouched = aq.wptr == 400 && aq.doorbell == 0
                        && aq.ring == std::vector<uint32_t>(1024, 0xDEADBEEFu);
        stores.push_back({viewAddress(p), v});
        viewStore32(p, v);
    };
    Adds adds;
    EXPECT_EQ(viewArmOrBook(a, 0, 0, 0, recordTo(adds), store), 0u);
    EXPECT_EQ(stores, (Adds{{viewAddress(&flag), 0u}}));
    EXPECT_TRUE(ringUntouched);
    EXPECT_TRUE(adds.empty());
    EXPECT_EQ(flag, 0u);
    auto const chain = build(a, 0, 0);
    EXPECT_EQ(aq.wptr, 400 + chain.size() * 4);
    EXPECT_EQ(aq.doorbell, aq.wptr);
    EXPECT_EQ(std::vector<uint32_t>(aq.ring.begin() + 100, aq.ring.begin() + 100 + chain.size()),
              chain);
}

// W=4, B=5, S=2, one batch: NB*G*Qb*S = 12. A full ring or a queue with packets pending is not armed, and
// flag[0] ends at exactly 12 from any previous value.
TEST(ViewTransport, A2AGemmBookingReachesExactlyTheChainTotal)
{
    for(bool stuck : {false, true})
        for(uint32_t previous : {0u, 12u, 0xFFFFFFF0u})
        {
            SCOPED_TRACE(testing::Message() << "stuck " << stuck << " previous " << previous);
            ViewArmArgs a = a2aGemmArgs(1);
            a.world       = 4;
            a.numQueues   = 3;
            a.tokenBlocks = 5;
            FakeQueue aq(stuck ? 1024 : 16, 400);
            if(stuck)
                aq.wptr += 64;
            aq.attach(a, 0);
            uint32_t flag       = previous;
            a.queues[0].flag    = &flag;
            const uint64_t wptr = aq.wptr;
            EXPECT_EQ(viewArmOrBook(a, 0, 0, 0, viewAdd32, viewStore32),
                      stuck ? VIEW_ERROR_STUCK : VIEW_ERROR_RING_FULL);
            EXPECT_EQ(flag, 12u);
            EXPECT_EQ(aq.wptr, wptr);
            EXPECT_EQ(aq.doorbell, 0u);
            EXPECT_EQ(aq.ring, std::vector<uint32_t>(aq.ring.size(), 0xDEADBEEFu));
        }
}

// GemmA2A: no flag store and no drain check; a queue with packets pending still gets exactly
// the chain viewBuildChain builds.
TEST(ViewTransport, GemmA2AArmStoresNothingAndWritesTheSameChain)
{
    for(uint32_t epoch : {0u, 1u, 5u})
    {
        SCOPED_TRACE(testing::Message() << "epoch " << epoch);
        ViewArmArgs g = gemmA2AArgs();
        FakeQueue   gq(1024, 400);
        gq.wptr += 64;
        gq.attach(g, 1);
        Adds adds, stores;
        EXPECT_EQ(viewArmOrBook(g, 1, epoch, 0, recordTo(adds), recordTo(stores)), 0u);
        EXPECT_TRUE(adds.empty());
        EXPECT_TRUE(stores.empty());
        auto const chain = build(g, 1, epoch);
        ASSERT_EQ(chain.size(),
                  2 * (VIEW_POLL_DWORDS + VIEW_COPY_DWORDS + VIEW_ATOMIC_DWORDS)
                      + VIEW_ATOMIC_DWORDS);
        EXPECT_EQ(gq.wptr, 464 + chain.size() * 4);
        EXPECT_EQ(
            std::vector<uint32_t>(gq.ring.begin() + 116, gq.ring.begin() + 116 + chain.size()),
            chain);
    }
}

TEST(ViewTransport, GemmA2APollThresholdsFollowTheEpoch)
{
    const uint64_t sig = kView + FUSED_A2A_VIEW_SIG_OFFSET + 4 * (1 * 2);
    for(uint32_t epoch : {0u, 1u})
    {
        auto const r = build(gemmA2AArgs(), 1, epoch);
        ASSERT_EQ(r.size(),
                  2 * (VIEW_POLL_DWORDS + VIEW_COPY_DWORDS + VIEW_ATOMIC_DWORDS)
                      + VIEW_ATOMIC_DWORDS);
        const uint32_t ref = epoch == 0 ? 2 : 4;
        size_t         at  = 0;
        for(uint32_t j = 0; j < 2; j++)
        {
            expectPoll(r, at, sig + 4 * j, ref);
            at += VIEW_POLL_DWORDS;
            expectCopy(r,
                       at,
                       kSrc + j * 256 * 3072 * 2,
                       kDst + j * 256 * 256 * 2,
                       384,
                       32,
                       8192,
                       32,
                       j ? 255 : 256);
            at += VIEW_COPY_DWORDS;
            expectAtomic(r, at, kFlag, 1);
            at += VIEW_ATOMIC_DWORDS;
        }
        expectAtomic(r, at, kView + FUSED_A2A_VIEW_OUTBOUND_OFFSET, 1);
    }
}

TEST(ViewTransport, PacketsDoNotStraddleTheRingEnd)
{
    ViewArmArgs a = a2aGemmArgs(1);
    a.world       = 2;
    a.numQueues   = 1;
    a.tokenBlocks = 1;
    a.subRounds   = 1;
    std::vector<uint32_t> ring(64, 0xDEADBEEFu);
    const uint64_t        start = 5 * 64 + 60;
    ViewRingWriter        w{ring.data(), ring.size(), start};
    viewBuildChain(a, 0, 0, w);
    EXPECT_EQ((w.pos - start) * sizeof(uint32_t), (4 + VIEW_COPY_DWORDS + VIEW_ATOMIC_DWORDS) * 4);
    for(size_t k = 60; k < 64; k++)
        EXPECT_EQ(ring[k], 0u) << "pad dword " << k;
    expectCopy(ring, 0, a2aGemmSrc(0, 0), a2aGemmDst(0, 0), 64, 64, 8192, 32, 128);
    expectAtomic(ring, VIEW_COPY_DWORDS, kFlag, 1);
    EXPECT_EQ(ring[VIEW_COPY_DWORDS + VIEW_ATOMIC_DWORDS], 0xDEADBEEFu);
}

TEST(ViewTransport, ArmQueuePublishesWptrAndDoorbell)
{
    std::vector<uint32_t> ring(1024, 0xDEADBEEFu);
    uint64_t              rptr = 400, wptr = 400, doorbell = 0;
    ViewArmArgs           a = a2aGemmArgs(1);
    a.queues[0].ring        = ring.data();
    a.queues[0].ringBytes   = ring.size() * sizeof(uint32_t);
    a.queues[0].rptr        = &rptr;
    a.queues[0].wptr        = &wptr;
    a.queues[0].doorbell    = &doorbell;
    ASSERT_TRUE(viewArmQueue(a, 0, 0));
    EXPECT_EQ(wptr, 400 + 4 * (VIEW_COPY_DWORDS + VIEW_ATOMIC_DWORDS) * 4);
    EXPECT_EQ(doorbell, wptr);
    expectCopy(ring, 100, a2aGemmSrc(0, 0), a2aGemmDst(0, 0), 64, 64, 8192, 32, 128);
}

TEST(ViewTransport, ArmQueueSkipsWhenTheRingIsFull)
{
    std::vector<uint32_t> ring(64, 0xDEADBEEFu);
    uint64_t              rptr = 0, wptr = 0, doorbell = 0;
    ViewArmArgs           a = a2aGemmArgs(1);
    a.queues[0].ring        = ring.data();
    a.queues[0].ringBytes   = ring.size() * sizeof(uint32_t);
    a.queues[0].rptr        = &rptr;
    a.queues[0].wptr        = &wptr;
    a.queues[0].doorbell    = &doorbell;
    EXPECT_FALSE(viewArmQueue(a, 0, 0));
    EXPECT_EQ(wptr, 0u);
    EXPECT_EQ(doorbell, 0u);
    EXPECT_EQ(ring, std::vector<uint32_t>(64, 0xDEADBEEFu));
}

// A 512-byte ring; the chain is 4 * (COPY + ATOMIC) = 336 bytes, from wptr 512 (ring offset 0) to 848.
TEST(ViewTransport, ArmQueueFitsOnlyWhileTheEndStaysBelowRptrPlusTheRing)
{
    const uint64_t chainBytes = 4 * (VIEW_COPY_DWORDS + VIEW_ATOMIC_DWORDS) * 4;
    for(uint64_t gap : {0u, 4u})
    {
        SCOPED_TRACE(testing::Message() << "end - rptr = ringBytes - " << gap);
        FakeQueue   queue(128, 512);
        ViewArmArgs a = a2aGemmArgs(1);
        queue.attach(a, 0);
        queue.rptr = 512 + chainBytes + gap - queue.ring.size() * sizeof(uint32_t);
        EXPECT_EQ(viewQueueFits(a, 0, 0), gap != 0);
        EXPECT_EQ(viewArmQueue(a, 0, 0), gap != 0);
        if(gap == 0)
        {
            EXPECT_EQ(queue.wptr, 512u);
            EXPECT_EQ(queue.doorbell, 0u);
            EXPECT_EQ(queue.ring, std::vector<uint32_t>(128, 0xDEADBEEFu));
        }
        else
        {
            EXPECT_EQ(queue.wptr, 512 + chainBytes);
            EXPECT_EQ(queue.doorbell, queue.wptr);
            expectCopy(queue.ring, 0, a2aGemmSrc(0, 0), a2aGemmDst(0, 0), 64, 64, 8192, 32, 128);
        }
    }
}

// A COPY that ends exactly at the ring end is not padded.
TEST(ViewTransport, PacketEndingAtTheRingEndIsNotPadded)
{
    ViewArmArgs a = a2aGemmArgs(1);
    a.world       = 2;
    a.numQueues   = 1;
    a.tokenBlocks = 1;
    a.subRounds   = 1;
    std::vector<uint32_t> ring(64, 0xDEADBEEFu);
    const uint64_t        start = 64 - VIEW_COPY_DWORDS;
    ViewRingWriter        w{ring.data(), ring.size(), start};
    viewBuildChain(a, 0, 0, w);
    EXPECT_EQ(w.pos - start, VIEW_COPY_DWORDS + VIEW_ATOMIC_DWORDS);
    expectCopy(ring, start, a2aGemmSrc(0, 0), a2aGemmDst(0, 0), 64, 64, 8192, 32, 128);
    expectAtomic(ring, 0, kFlag, 1);
}

TEST(ViewTransport, AdvanceMovesBaseToExpected)
{
    uint32_t    header[FUSED_A2A_VIEW_SIG_OFFSET / 4] = {};
    ViewArmArgs a                                     = a2aGemmArgs(1);
    viewAdvance(a, 0, header);
    viewAdvance(a, 1, header);
    for(uint32_t i = 0; i < FUSED_A2A_MAX_RANKS; i++)
    {
        EXPECT_EQ(header[FUSED_A2A_VIEW_BASE_OFFSET / 4 + i], 0u);
        EXPECT_EQ(header[FUSED_A2A_VIEW_EXPECTED_OFFSET / 4 + i], 0u);
    }
    EXPECT_EQ(header[FUSED_A2A_VIEW_EPOCH_OFFSET / 4], 2u);

    uint32_t    gemmHeader[FUSED_A2A_VIEW_SIG_OFFSET / 4] = {};
    ViewArmArgs g                                         = gemmA2AArgs();
    for(uint32_t epoch = 0; epoch < 3; epoch++)
        viewAdvance(g, epoch, gemmHeader);
    EXPECT_EQ(gemmHeader[FUSED_A2A_VIEW_BASE_OFFSET / 4 + 1], 4u);
    EXPECT_EQ(gemmHeader[FUSED_A2A_VIEW_EXPECTED_OFFSET / 4 + 1], 6u);
    EXPECT_EQ(gemmHeader[FUSED_A2A_VIEW_OUTBOUND_BASE_OFFSET / 4], 4u);
    EXPECT_EQ(gemmHeader[FUSED_A2A_VIEW_EPOCH_OFFSET / 4], 3u);
}

TEST(ViewTransport, ArmHaltsOnlyGemmA2AAndBooksItsQueue)
{
    ViewArmArgs g = gemmA2AArgs();
    FakeQueue   gq(1024, 400);
    gq.attach(g, 1);
    Adds gAdds, gStores;
    EXPECT_EQ(viewArmOrBook(g, 1, 0, VIEW_ERROR_TIMEOUT, recordTo(gAdds), recordTo(gStores)), 0u);
    EXPECT_TRUE(gStores.empty());
    EXPECT_EQ(gq.wptr, 400u);
    EXPECT_EQ(gq.doorbell, 0u);
    EXPECT_EQ(gq.ring, std::vector<uint32_t>(1024, 0xDEADBEEFu));
    EXPECT_EQ(gAdds, (Adds{{kFlag, 2}, {kView + FUSED_A2A_VIEW_OUTBOUND_OFFSET, 1}}));

    ViewArmArgs a = a2aGemmArgs(1);
    FakeQueue   aq(1024, 400);
    aq.attach(a, 0);
    Adds aAdds, aStores;
    EXPECT_EQ(viewArmOrBook(a, 0, 0, VIEW_ERROR_TIMEOUT, recordTo(aAdds), recordTo(aStores)), 0u);
    EXPECT_EQ(aq.wptr, 400 + 4 * (VIEW_COPY_DWORDS + VIEW_ATOMIC_DWORDS) * 4);
    EXPECT_TRUE(aAdds.empty());
    EXPECT_EQ(aStores, (Adds{{kFlag, 0u}}));
}

TEST(ViewTransport, RingFullBooksTheSkippedQueue)
{
    ViewArmArgs g = gemmA2AArgs();
    FakeQueue   gq(16, 0);
    gq.attach(g, 1);
    Adds gAdds, stores;
    EXPECT_EQ(viewArmOrBook(g, 1, 0, 0, recordTo(gAdds), recordTo(stores)), VIEW_ERROR_RING_FULL);
    EXPECT_EQ(gq.wptr, 0u);
    EXPECT_EQ(gAdds, (Adds{{kFlag, 2}, {kView + FUSED_A2A_VIEW_OUTBOUND_OFFSET, 1}}));
    EXPECT_TRUE(stores.empty());

    ViewArmArgs a = a2aGemmArgs(1);
    FakeQueue   aq(16, 0);
    aq.attach(a, 0);
    Adds aAdds;
    EXPECT_EQ(viewArmOrBook(a, 0, 0, 0, recordTo(aAdds), recordTo(stores)), VIEW_ERROR_RING_FULL);
    EXPECT_EQ(aAdds, (Adds{{kFlag, 4}}));
    EXPECT_EQ(stores, (Adds{{kFlag, 0u}}));

    FakeQueue roomy(1024, 0);
    roomy.attach(g, 1);
    gAdds.clear();
    EXPECT_EQ(viewArmOrBook(g, 1, 0, 0, recordTo(gAdds), recordTo(stores)), 0u);
    EXPECT_NE(roomy.wptr, 0u);
    EXPECT_TRUE(gAdds.empty());
}

TEST(ViewTransport, BookingAddsAtomically)
{
    uint32_t    block[FUSED_A2A_VIEW_SIG_OFFSET / 4] = {};
    uint32_t    peerFlag                             = 7;
    ViewArmArgs g                                    = gemmA2AArgs();
    g.view                                           = block;
    g.queues[1].flag                                 = &peerFlag;
    viewBookSkipped(g, 1, viewAdd32);
    EXPECT_EQ(peerFlag, 9u);
    EXPECT_EQ(block[FUSED_A2A_VIEW_OUTBOUND_OFFSET / 4], 1u);
}

TEST(ViewTransport, ReleaseSigKeepsTheNextLaunchParked)
{
    constexpr uint32_t kWords                                        = 4 + 8;
    uint32_t           block[FUSED_A2A_VIEW_SIG_OFFSET / 4 + kWords] = {};
    uint32_t* const    sig = block + FUSED_A2A_VIEW_SIG_OFFSET / 4;
    for(uint32_t i = 0; i < kWords; i++)
        sig[i] = 0xC0DE0000u + i;
    ViewArmArgs g = gemmA2AArgs();
    g.view        = block;
    sig[0]        = 4;
    viewReleaseSig(g, 3);
    for(uint32_t i = 0; i < 4; i++)
        EXPECT_EQ(sig[i], 6u) << "sig " << i;
    for(uint32_t i = 4; i < kWords; i++)
        EXPECT_EQ(sig[i], 0xC0DE0000u + i) << "sig " << i;

    g.view                 = fakePointer<uint32_t>(kView);
    auto const     next    = build(g, 1, 3);
    const uint32_t nextRef = next[3];
    EXPECT_LT(sig[1 * 2 + 0], nextRef);
    EXPECT_EQ(sig[1 * 2 + 0] + g.tilesPerRank, nextRef);
}

TEST(ViewTransport, QueuesDrainedComparesEveryQueue)
{
    ViewArmArgs g = gemmA2AArgs();
    FakeQueue   q0(64, 128), q1(64, 256);
    q0.attach(g, 0);
    q1.attach(g, 1);
    EXPECT_TRUE(viewQueuesDrained(g));
    q1.wptr += 32;
    EXPECT_FALSE(viewQueuesDrained(g));
    q1.rptr = q1.wptr;
    EXPECT_TRUE(viewQueuesDrained(g));
}

TEST(ViewTransport, TimelineAddsOneTimestampAtEachEndOfTheChain)
{
    constexpr uint64_t kStats = 0x500000000ull;
    constexpr uint32_t kEpoch = VIEW_STATS_RECORDS + 1;
    std::vector<std::pair<ViewArmArgs, uint32_t>> cases
        = {{a2aGemmArgs(1), 0}, {a2aGemmArgs(3), 0}, {gemmA2AArgs(), 1}};
    for(auto [a, q] : cases)
    {
        SCOPED_TRACE(testing::Message() << "mode " << a.mode << " coalesce " << a.coalesce);
        auto const plain = build(a, q, kEpoch);
        a.stats          = fakePointer<uint64_t>(kStats);
        auto const timed = build(a, q, kEpoch);
        ASSERT_EQ(timed.size(), plain.size() + 2 * VIEW_TIMESTAMP_DWORDS);
        const uint64_t record = kStats + 1 * VIEW_STATS_FIELDS * 32;
        const uint64_t start  = record + (4 + 2 * q) * 32;
        const uint64_t end    = record + (5 + 2 * q) * 32;
        EXPECT_EQ(std::vector<uint32_t>(timed.begin(), timed.begin() + VIEW_TIMESTAMP_DWORDS),
                  (std::vector<uint32_t>{0x0000020Du, lo(start), hi(start)}));
        EXPECT_EQ(std::vector<uint32_t>(timed.begin() + VIEW_TIMESTAMP_DWORDS,
                                        timed.end() - VIEW_TIMESTAMP_DWORDS),
                  plain);
        EXPECT_EQ(std::vector<uint32_t>(timed.end() - VIEW_TIMESTAMP_DWORDS, timed.end()),
                  (std::vector<uint32_t>{0x0000020Du, lo(end), hi(end)}));
    }
}

TEST(ViewTransport, NoPollDropsOnlyTheGemmA2APolls)
{
    ViewArmArgs g     = gemmA2AArgs();
    auto const  plain = build(g, 1, 1);
    g.debug           = VIEW_DEBUG_NO_POLL;
    auto const            noPoll = build(g, 1, 1);
    std::vector<uint32_t> expected;
    uint32_t              polls = 0;
    for(size_t at = 0; at < plain.size();)
    {
        const uint32_t header = plain[at];
        const uint32_t size   = header == VIEW_POLL_HEADER   ? VIEW_POLL_DWORDS
                                : header == VIEW_COPY_HEADER ? VIEW_COPY_DWORDS
                                                             : VIEW_ATOMIC_DWORDS;
        ASSERT_TRUE(header == VIEW_POLL_HEADER || header == VIEW_COPY_HEADER
                    || header == VIEW_ATOMIC_HEADER)
            << "dword " << at;
        if(header == VIEW_POLL_HEADER)
            polls++;
        else
            expected.insert(expected.end(), plain.begin() + at, plain.begin() + at + size);
        at += size;
    }
    EXPECT_EQ(polls, g.tokenBlocks);
    EXPECT_EQ(noPoll, expected);

    ViewArmArgs a = a2aGemmArgs(1);
    auto const  a2aGemm = build(a, 0, 0);
    a.debug             = VIEW_DEBUG_NO_POLL;
    EXPECT_EQ(build(a, 0, 0), a2aGemm);
}
