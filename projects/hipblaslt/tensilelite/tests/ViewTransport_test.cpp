// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ViewTransport.hpp"
#include "ViewVerify.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
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

namespace
{
    struct Packet
    {
        uint32_t              header;
        std::vector<uint32_t> words;
    };

    // The packets of a chain without TIMESTAMPs and zero padding.
    std::vector<Packet> packets(std::vector<uint32_t> const& r)
    {
        std::vector<Packet> out;
        for(size_t at = 0; at < r.size();)
        {
            const uint32_t header = r[at];
            const uint32_t size   = header == VIEW_POLL_HEADER   ? VIEW_POLL_DWORDS
                                    : header == VIEW_COPY_HEADER ? VIEW_COPY_DWORDS
                                                                 : VIEW_ATOMIC_DWORDS;
            EXPECT_TRUE(header == VIEW_POLL_HEADER || header == VIEW_COPY_HEADER
                        || header == VIEW_ATOMIC_HEADER)
                << "dword " << at;
            out.push_back({header, std::vector<uint32_t>(r.begin() + at, r.begin() + at + size)});
            at += size;
        }
        return out;
    }

    // build() with a ring that holds the whole chain.
    std::vector<uint32_t> buildWhole(ViewArmArgs const& a, uint32_t q, uint32_t epoch)
    {
        ViewRingWriter count{nullptr, 1ull << 40, 0};
        viewBuildChain(a, q, epoch, count);
        uint64_t dwords = 1;
        while(dwords <= count.pos)
            dwords *= 2;
        std::vector<uint32_t> ring(dwords, 0xDEADBEEFu);
        ViewRingWriter        w{ring.data(), dwords, 0};
        viewBuildChain(a, q, epoch, w);
        ring.resize(w.pos);
        return ring;
    }

    std::vector<Packet> only(std::vector<Packet> const& p, uint32_t header)
    {
        std::vector<Packet> out;
        for(auto const& k : p)
            if(k.header == header)
                out.push_back(k);
        return out;
    }

    bool operator==(Packet const& x, Packet const& y)
    {
        return x.words == y.words;
    }

    uint64_t fnv1a(std::vector<uint32_t> const& words)
    {
        uint64_t h = 0xcbf29ce484222325ull;
        for(uint32_t w : words)
            for(int b = 0; b < 4; b++)
            {
                h ^= (w >> (8 * b)) & 0xFFu;
                h *= 0x100000001b3ull;
            }
        return h;
    }

    // The configs of the chain hashes DebugBitsZeroKeepTheBuilderBytes stores.
    ViewArmArgs dumpedA2AGemmArgs()
    {
        ViewArmArgs a = chainArgs(4, 5, 3, 8, 2, 2);
        a.rank        = 2;
        return a;
    }

    ViewArmArgs dumpedGemmA2AArgs()
    {
        ViewArmArgs g = gemmA2AArgs();
        for(uint32_t q = 0; q < 2; q++)
        {
            g.queues[q].srcBase = kSrc + q * 0x1000;
            g.queues[q].dstBase = kDst + q * 0x1000;
            g.queues[q].flag    = fakePointer<uint32_t>(kFlag + 4 * q);
        }
        return g;
    }

    std::vector<ViewArmArgs> faultCases()
    {
        std::vector<ViewArmArgs> cases = {a2aGemmArgs(1), a2aGemmArgs(3), dumpedGemmA2AArgs()};
        for(uint32_t c : {1u, 2u, kAll})
        {
            cases.push_back(chainArgs(4, 5, 3, 8, 2, c));
            cases.push_back(chainArgs(3, 7, 1, 64, 1, c));
            cases.push_back(chainArgs(8, 16, 16, 256, 4, c));
        }
        return cases;
    }

    uint64_t atomicSum(std::vector<Packet> const& p)
    {
        uint64_t sum = 0;
        for(auto const& k : only(p, VIEW_ATOMIC_HEADER))
            sum += k.words[3];
        return sum;
    }
}

TEST(ViewTransport, DebugBitsZeroKeepTheBuilderBytes)
{
    ViewArmArgs const a = dumpedA2AGemmArgs();
    ViewArmArgs const g = dumpedGemmA2AArgs();
    const std::vector<std::pair<std::vector<uint32_t>, std::pair<size_t, uint64_t>>> cases
        = {{build(a, 0, 0), {210, 0xabac02f7ed9f5213ull}},
           {build(a, 1, 7), {210, 0x9897dc5c36713cfbull}},
           {build(g, 0, 5), {62, 0xb2abe52fd03dbb4full}},
           {build(g, 1, 5), {62, 0x50ae60652e042a97ull}}};
    for(auto const& [ring, want] : cases)
    {
        EXPECT_EQ(ring.size(), want.first);
        EXPECT_EQ(fnv1a(ring), want.second);
    }
}

TEST(ViewTransport, DropCopyOmitsOnlyQueue0sFirstCopy)
{
    for(auto a : faultCases())
    {
        SCOPED_TRACE(testing::Message() << "mode " << a.mode << " W " << a.world << " B "
                                        << a.tokenBlocks << " coalesce " << a.coalesce);
        const uint32_t                     queues = a.mode == VIEW_MODE_A2A_GEMM ? a.numQueues : 2;
        std::vector<std::vector<uint32_t>> plain;
        for(uint32_t k = 0; k < queues; k++)
            plain.push_back(buildWhole(a, k, 3));
        a.debug = VIEW_DEBUG_DROP_COPY;
        for(uint32_t k = 1; k < queues; k++)
            EXPECT_EQ(buildWhole(a, k, 3), plain[k]) << "queue " << k;
        auto const before = packets(plain[0]);
        auto const after  = packets(buildWhole(a, 0, 3));
        auto       copies = only(before, VIEW_COPY_HEADER);
        ASSERT_FALSE(copies.empty());
        copies.erase(copies.begin());
        EXPECT_EQ(only(after, VIEW_COPY_HEADER), copies);
        EXPECT_EQ(only(after, VIEW_ATOMIC_HEADER), only(before, VIEW_ATOMIC_HEADER));
        EXPECT_EQ(only(after, VIEW_POLL_HEADER), only(before, VIEW_POLL_HEADER));
        EXPECT_EQ(atomicSum(after), atomicSum(before));
        EXPECT_EQ(after.size(), before.size() - 1);
    }
}

TEST(ViewTransport, EarlyReleaseMovesQueue0sFirstCopyBeforeTheLastAtomic)
{
    for(auto a : faultCases())
    {
        SCOPED_TRACE(testing::Message() << "mode " << a.mode << " W " << a.world << " B "
                                        << a.tokenBlocks << " coalesce " << a.coalesce);
        const uint32_t                     queues = a.mode == VIEW_MODE_A2A_GEMM ? a.numQueues : 2;
        std::vector<std::vector<uint32_t>> plain;
        for(uint32_t k = 0; k < queues; k++)
            plain.push_back(buildWhole(a, k, 3));
        a.debug = VIEW_DEBUG_EARLY_RELEASE;
        for(uint32_t k = 1; k < queues; k++)
            EXPECT_EQ(buildWhole(a, k, 3), plain[k]) << "queue " << k;
        auto       expected = packets(plain[0]);
        auto const after    = packets(buildWhole(a, 0, 3));
        auto       first    = std::find_if(expected.begin(), expected.end(), [](Packet const& k) {
            return k.header == VIEW_COPY_HEADER;
        });
        ASSERT_NE(first, expected.end());
        const Packet moved = *first;
        expected.erase(first);
        auto last = std::find_if(expected.rbegin(), expected.rend(), [](Packet const& k) {
            return k.header == VIEW_ATOMIC_HEADER;
        });
        ASSERT_NE(last, expected.rend());
        expected.insert(last.base() - 1, moved);
        EXPECT_EQ(after, expected);
        EXPECT_EQ(atomicSum(after), atomicSum(packets(plain[0])));
    }
}

namespace
{
    // Value of a sign-magnitude float with `ebits` exponent and `mbits` mantissa bits; `ocp`
    // is OCP E4M3 (NaN only at all-ones, no infinity).
    double minifloat(uint32_t bits, int ebits, int mbits, bool ocp)
    {
        const int      bias = (1 << (ebits - 1)) - 1;
        const uint32_t e    = (bits >> mbits) & ((1u << ebits) - 1);
        const uint32_t m    = bits & ((1u << mbits) - 1);
        const double   s    = (bits >> (ebits + mbits)) & 1u ? -1.0 : 1.0;
        const uint32_t eMax = (1u << ebits) - 1;
        if(ocp && e == eMax && m == (1u << mbits) - 1)
            return std::nan("");
        if(!ocp && e == eMax)
            return m == 0 ? s * INFINITY : std::nan("");
        if(e == 0)
            return s * std::ldexp((double)m, 1 - bias - mbits);
        return s * std::ldexp(1.0 + std::ldexp((double)m, -mbits), (int)e - bias);
    }

    double fp32(uint32_t bits)
    {
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

    // Whether lane value `negated` is -`value`, sign of zero included.
    bool isNegation(double value, double negated)
    {
        if(std::isnan(value) || std::isnan(negated))
            return std::isnan(value) && std::isnan(negated);
        return negated == -value && std::signbit(negated) != std::signbit(value);
    }
}

TEST(ViewTransport, SignMaskNegatesEveryElement)
{
    struct Format
    {
        const char*                     name;
        uint32_t                        bytes;
        std::function<double(uint32_t)> value;
    };
    const std::vector<Format> formats = {
        {"bf16", 2, [](uint32_t b) { return fp32(b << 16); }},
        {"fp16", 2, [](uint32_t b) { return minifloat(b, 5, 10, false); }},
        {"fp8", 1, [](uint32_t b) { return minifloat(b, 4, 3, true); }},
        {"bf8", 1, [](uint32_t b) { return minifloat(b, 5, 2, false); }},
    };
    for(auto const& f : formats)
    {
        const uint32_t mask  = viewSignMask(f.bytes);
        const uint32_t bits  = 8 * f.bytes;
        const uint32_t lanes = 4 / f.bytes;
        uint32_t       bad   = 0;
        for(uint32_t p = 0; p < (1u << bits); p++)
            for(uint32_t lane = 0; lane < lanes; lane++)
            {
                const uint32_t other
                    = (0x9E3779B9u * (p + 1)) & ~(((1u << bits) - 1) << (bits * lane));
                const uint32_t word = other | (p << (bits * lane));
                const uint32_t neg  = word ^ mask;
                const uint32_t q    = (neg >> (bits * lane)) & ((1u << bits) - 1);
                if(q == p || !isNegation(f.value(p), f.value(q)))
                    bad++;
                for(uint32_t l = 0; l < lanes; l++)
                    if(l != lane
                       && !isNegation(f.value((word >> (bits * l)) & ((1u << bits) - 1)),
                                      f.value((neg >> (bits * l)) & ((1u << bits) - 1))))
                        bad++;
            }
        EXPECT_EQ(bad, 0u) << f.name;
    }
    const uint32_t mask = viewSignMask(4);
    EXPECT_EQ(mask, 0x80000000u);
    for(uint32_t p : {0x00000000u,
                      0x80000000u,
                      0x00000001u,
                      0x3F800000u,
                      0xC0490FDBu,
                      0x7F7FFFFFu,
                      0x7F800000u,
                      0xFF800000u,
                      0x7FC00000u,
                      0x00800000u,
                      0x12345678u})
    {
        EXPECT_NE(p ^ mask, p);
        EXPECT_TRUE(isNegation(fp32(p), fp32(p ^ mask))) << std::hex << p;
    }
}

TEST(ViewTransport, FaultFreeBuilderIgnoresTheCopyFaultBits)
{
    for(auto a : faultCases())
    {
        SCOPED_TRACE(testing::Message() << "mode " << a.mode << " W " << a.world);
        const auto plain = buildWhole(a, 0, 3);
        a.debug          = VIEW_DEBUG_COPY_FAULTS;
        ViewRingWriter count{nullptr, 1ull << 40, 0};
        viewBuildUnits<0>(a, 0, 3, count);
        std::vector<uint32_t> ring(count.pos + 1, 0xDEADBEEFu);
        ViewRingWriter        w{ring.data(), 1ull << 40, 0};
        viewBuildUnits<0>(a, 0, 3, w);
        ring.resize(w.pos);
        EXPECT_EQ(ring, plain);
        EXPECT_NE(buildWhole(a, 0, 3), plain);
    }
}

TEST(ViewTransport, VerifyCompareMatchesTheGoldenPerElement)
{
    for(uint32_t bytes : {1u, 2u, 4u})
    {
        SCOPED_TRACE(testing::Message() << "element bytes " << bytes);
        const uint32_t mask   = viewSignMask(bytes);
        const uint32_t golden = 0x3F7A1C42u;
        EXPECT_TRUE(viewVerifyCompare(golden, golden, mask));
        EXPECT_FALSE(viewVerifyCompare(golden ^ mask, golden, mask));
        for(uint32_t bit = 0; bit < 32; bit++)
            EXPECT_FALSE(viewVerifyCompare(golden ^ (1u << bit), golden, mask)) << bit;
        EXPECT_FALSE(viewVerifyCompare(VERIFY_POISON_WORD, golden, mask));
        // Zero elements match either sign; the others still compare.
        const uint32_t lowLane = bytes == 4 ? 0xFFFFFFFFu : (1u << (8 * bytes)) - 1;
        const uint32_t zeroLow = golden & ~lowLane;
        const uint32_t sign    = mask & lowLane;
        EXPECT_TRUE(viewVerifyCompare(zeroLow ^ sign, zeroLow, mask));
        EXPECT_FALSE(viewVerifyCompare(zeroLow ^ sign ^ 1u, zeroLow, mask));
        if(bytes != 4)
            EXPECT_FALSE(viewVerifyCompare(zeroLow ^ mask, zeroLow, mask));
        // verifyCheckWord: the golden with the flipped sign bits.
        EXPECT_TRUE(verifyCheckWord(golden ^ sign, golden, bytes, sign));
        EXPECT_FALSE(verifyCheckWord(golden, golden, bytes, sign));
        EXPECT_TRUE(verifyCheckWord(zeroLow, zeroLow, bytes, sign));
    }
}

TEST(ViewTransport, VerifyPatternMaskNegatesTheElementsOfTheSetBits)
{
    for(uint32_t bytes : {1u, 2u, 4u})
        for(VerifyPattern p : {VerifyPattern{1, 0, 8, bytes},
                               VerifyPattern{4, 0, 8, bytes},
                               VerifyPattern{1, 12, 3, bytes},
                               VerifyPattern{3, 48, 40, bytes},
                               VerifyPattern{5, 0, 64, bytes}})
            for(uint32_t e : {1u, 2u, 7u, 300u})
            {
                SCOPED_TRACE(testing::Message()
                             << "bytes " << bytes << " stride " << p.stride << " period "
                             << p.period << " shard " << p.shard << " e " << e);
                const uint32_t bits  = verifyPatternBits(e);
                const uint32_t lanes = 4 / bytes;
                for(size_t word = 0; word < 64; word++)
                {
                    uint32_t want = 0;
                    for(uint32_t l = 0; l < lanes; l++)
                    {
                        const uint64_t i = word * lanes + l;
                        const uint64_t c = ((p.period ? i % p.period : i) / p.stride) % p.shard;
                        if((bits >> (c % 32)) & 1u)
                            want |= 1u << (8 * bytes * (l + 1) - 1);
                    }
                    EXPECT_EQ(verifyPatternMask(p, bits, word), want) << word;
                    EXPECT_EQ(verifyPatternMask(p, bits, word) & ~viewSignMask(bytes), 0u);
                }
            }
    EXPECT_EQ(verifyPatternBits(0), 0u);
    EXPECT_EQ(verifyPatternMask(VerifyPattern{1, 0, 0, 4}, 0xFFFFFFFFu, 3), 0u);
    EXPECT_EQ(verifyPatternMask(VerifyPattern{1, 0, 8, 4}, 0u, 3), 0u);
    // The low b bits of the pattern bits of e = 0 .. 2^b - 1 are distinct.
    for(uint32_t b = 1; b <= 10; b++)
    {
        std::set<uint32_t> seen;
        for(uint32_t e = 0; e < (1u << b); e++)
            seen.insert(verifyPatternBits(e) & ((1u << b) - 1));
        EXPECT_EQ(seen.size(), 1u << b) << b;
    }
}

TEST(ViewTransport, VerifyCheckBlockLayoutAndTotals)
{
    const std::set<uint32_t> index
        = {VERIFY_D_MISMATCHES, VERIFY_RECV_MISMATCHES, VERIFY_FIRST_BAD, VERIFY_CHECKED};
    EXPECT_EQ(index.size(), (size_t)VERIFY_BLOCK_WORDS);
    EXPECT_LT(*index.rbegin(), VERIFY_BLOCK_WORDS);
    EXPECT_LE(VERIFY_BLOCK_WORDS * sizeof(uint32_t), VERIFY_BLOCK_BYTES);

    uint32_t words[VERIFY_BLOCK_WORDS];
    std::fill(std::begin(words), std::end(words), 0xA5A5A5A5u);
    verifyInitialBlock(words);
    VerifyCounts c = verifyDecodeBlock(words);
    EXPECT_EQ(c.dMismatches, 0u);
    EXPECT_EQ(c.recvMismatches, 0u);
    EXPECT_EQ(c.firstBad, VERIFY_NO_BAD_LAUNCH);
    EXPECT_EQ(c.checked, 0u);
    words[VERIFY_D_MISMATCHES]    = 11;
    words[VERIFY_RECV_MISMATCHES] = 22;
    words[VERIFY_FIRST_BAD]       = 3;
    words[VERIFY_CHECKED]         = 44;
    c                             = verifyDecodeBlock(words);
    EXPECT_EQ(c.dMismatches, 11u);
    EXPECT_EQ(c.recvMismatches, 22u);
    EXPECT_EQ(c.firstBad, 3u);
    EXPECT_EQ(c.checked, 44u);

    VerifyCounts total, clean, later;
    clean.checked     = 7;
    later.dMismatches = 2;
    later.firstBad    = 9;
    later.checked     = 7;
    for(auto const& x : {c, clean, later})
        verifyAccumulate(total, x);
    EXPECT_EQ(total.dMismatches, 13u);
    EXPECT_EQ(total.recvMismatches, 22u);
    EXPECT_EQ(total.firstBad, 3u);
    EXPECT_EQ(total.checked, 58u);
}

TEST(ViewTransport, VerifyCheckPassesAccumulateAcrossThreadsAndLaunches)
{
    std::vector<uint32_t> golden(10, 0x3F800000u), data = golden;
    data[2] = data[7] = 0x40000000u;
    uint32_t block[VERIFY_BLOCK_WORDS];
    verifyInitialBlock(block);
    for(uint32_t launch : {5u, 3u})
    {
        VerifyOp op    = verifyOp(VerifyOpKind::Check, data.data(), golden.data(), data.size());
        op.launch      = launch;
        op.counter     = VERIFY_RECV_MISMATCHES;
        op.countLaunch = true;
        for(size_t t = 0; t < 2; t++)
            verifyRunOp(op, t, 2, block, t == 0);
    }
    const VerifyCounts c = verifyDecodeBlock(block);
    EXPECT_EQ(c.recvMismatches, 4u);
    EXPECT_EQ(c.dMismatches, 0u);
    EXPECT_EQ(c.firstBad, 3u);
    EXPECT_EQ(c.checked, 2u);
}

namespace
{
    // A2AGemm: A [M, K] column-major, K = W * kL; B and a2aBuf [W][N][kL]; D [M, N].
    // GemmA2A: A [M, kKg] column-major, M = W * kL; D [M, N]; recv [W][N][kL]. f32.
    constexpr uint32_t kM  = 4;
    constexpr uint32_t kN  = 3;
    constexpr uint32_t kL  = 8;
    constexpr uint32_t kKg = 2;

    VerifyTensor f32Tensor(uint64_t elements, uint64_t stride = 1, uint64_t period = 0)
    {
        return {4, true, elements * 4, false, stride, period};
    }

    VerifyShape simShape(bool gemmA2A, uint32_t w)
    {
        VerifyShape s;
        s.gemmA2A = gemmA2A;
        s.world   = w;
        s.extent  = w * kL;
        s.tokens  = kN;
        s.buf     = f32Tensor(w * kN * kL);
        if(gemmA2A)
        {
            s.a = f32Tensor(w * kL * kKg, 1, w * kL);
            s.b = f32Tensor(kKg * kN);
            s.d = f32Tensor(w * kL * kN, 1, w * kL);
        }
        else
        {
            s.a = f32Tensor(kM * w * kL, kM);
            s.b = f32Tensor(w * kN * kL);
            s.d = f32Tensor(kM * kN, 1, kM);
        }
        return s;
    }

    void runOps(std::vector<VerifyOp> const& ops, uint32_t* block)
    {
        for(auto const& op : ops)
            for(size_t t = 0; t < 3; t++)
                verifyRunOp(op, t, 3, block, t == 0);
    }
}

TEST(ViewTransport, VerifyLayoutPlacesThePeerSegments)
{
    for(uint32_t w = 2; w <= FUSED_A2A_MAX_RANKS; w++)
    {
        const VerifyLayout l = verifyLayoutOf(simShape(false, w));
        EXPECT_EQ(l.peerLo, (size_t)kN * kL);
        EXPECT_EQ(l.peerHi, (size_t)kN * kL * w);
        EXPECT_EQ(l.peerHi, l.bufWords);
        EXPECT_EQ(l.aWords, (size_t)kM * w * kL);
        EXPECT_EQ(l.bWords, (size_t)w * kN * kL);
        EXPECT_EQ(l.dWords, (size_t)kM * kN);
        EXPECT_EQ(l.a.stride, kM);
        EXPECT_EQ(l.a.shard, kL);
        EXPECT_EQ(l.b.shard, kL);
        EXPECT_EQ(l.d.shard, 0u);
        const VerifyLayout g = verifyLayoutOf(simShape(true, w));
        EXPECT_EQ(g.peerHi, 0u);
        EXPECT_EQ(g.recv.shard, kL);
        EXPECT_EQ(g.d.shard, kL);
        EXPECT_EQ(g.d.period, (uint64_t)w * kL);
        EXPECT_EQ(g.bufWords, (size_t)w * kN * kL);
    }
    auto rejects = [](VerifyShape s, char const* what) {
        EXPECT_THROW(verifyLayoutOf(s), std::runtime_error) << what;
    };
    for(bool gemmA2A : {false, true})
    {
        VerifyShape s     = simShape(gemmA2A, 4);
        auto        t     = s;
        t.a.signMagnitude = false;
        rejects(t, "A type");
        t = s;
        t.extent += 1;
        rejects(t, "extent");
        t = s;
        t.d.allocatedBytes += 2;
        rejects(t, "D bytes");
        t = s;
        t.buf.allocatedBytes -= 2;
        rejects(t, "a2aBuf bytes");
    }
    VerifyShape t = simShape(false, 4);
    t.b.allocatedBytes += 4;
    rejects(t, "A2AGemm B size");
    t                 = simShape(false, 4);
    t.b.signMagnitude = false;
    rejects(t, "A2AGemm B type");
    t = simShape(false, 4);
    t.buf.allocatedBytes -= 4;
    rejects(t, "a2aBuf too small");
    t          = simShape(true, 4);
    t.d.padded = true;
    rejects(t, "GemmA2A D padding");
    t                 = simShape(true, 4);
    t.d.signMagnitude = false;
    rejects(t, "GemmA2A D type");
    t          = simShape(false, 4);
    t.d.padded = true;
    EXPECT_NO_THROW(verifyLayoutOf(t));
}

TEST(ViewTransport, VerifyArgsErrorRejectsEachUnsupportedOption)
{
    VerifyArgs ok;
    ok.verify = VERIFY_PRODUCE;
    EXPECT_EQ(verifyArgsError(ok), "");
    ok.verify = VERIFY_SKEW;
    EXPECT_EQ(verifyArgsError(ok), "");
    ok.gemmA2A = true;
    EXPECT_EQ(verifyArgsError(ok), "");
    std::vector<std::pair<const char*, std::function<void(VerifyArgs&)>>> bad = {
        {"verify 3", [](VerifyArgs& a) { a.verify = 3; }},
        {"verify -1", [](VerifyArgs& a) { a.verify = -1; }},
        {"barrier 2", [](VerifyArgs& a) { a.barrier = 2; }},
        {"fence -1", [](VerifyArgs& a) { a.fence = -1; }},
        {"skew -1", [](VerifyArgs& a) { a.skewUs = -1; }},
        {"validate 0", [](VerifyArgs& a) { a.numElementsToValidate = 0; }},
        {"warmups 0", [](VerifyArgs& a) { a.numWarmups = 0; }},
        {"c-equal-d", [](VerifyArgs& a) { a.cEqualD = true; }},
        {"sparse", [](VerifyArgs& a) { a.sparse = true; }},
        {"bounds-check 1", [](VerifyArgs& a) { a.boundsCheckNaN = true; }},
        {"scales", [](VerifyArgs& a) { a.scales = true; }},
        {"swizzle", [](VerifyArgs& a) { a.swizzled = true; }},
        {"beta", [](VerifyArgs& a) { a.betaZero = false; }},
        {"activation", [](VerifyArgs& a) { a.activation = true; }},
        {"bias", [](VerifyArgs& a) { a.bias = true; }},
        {"not fused", [](VerifyArgs& a) { a.fusedA2A = false; }},
    };
    const std::set<std::string> gemmA2AOnly = {"beta", "activation", "bias"};
    const std::set<std::string> ranges
        = {"verify 3", "verify -1", "barrier 2", "fence -1", "skew -1"};
    for(auto const& [name, set] : bad)
        for(int verify : {VERIFY_OFF, VERIFY_PRODUCE, VERIFY_SKEW})
            for(bool gemmA2A : {false, true})
            {
                VerifyArgs a;
                a.verify  = verify;
                a.gemmA2A = gemmA2A;
                set(a);
                const bool want
                    = ranges.count(name)
                      || (verify != VERIFY_OFF && (gemmA2A || !gemmA2AOnly.count(name)));
                EXPECT_EQ(!verifyArgsError(a).empty(), want)
                    << name << " verify " << verify << " GemmA2A " << gemmA2A;
            }
    VerifyArgs warm;
    warm.verify     = VERIFY_PRODUCE;
    warm.numWarmups = 1;
    EXPECT_EQ(verifyArgsError(warm), "");
}

TEST(ViewTransport, VerifyScheduleStateAndGoldenGate)
{
    using G = VerifyGolden;
    EXPECT_EQ(verifyGoldenAfterValidation(G::Pending, true, false), G::Captured);
    EXPECT_EQ(verifyGoldenAfterValidation(G::Pending, true, true), G::Disabled);
    EXPECT_EQ(verifyGoldenAfterValidation(G::Pending, false, false), G::Pending);
    EXPECT_EQ(verifyGoldenAfterValidation(G::Captured, true, true), G::Captured);
    EXPECT_EQ(verifyGoldenAfterValidation(G::Disabled, true, false), G::Disabled);

    VerifyState st;
    st.reset(3);
    EXPECT_FALSE(st.checks());
    for(uint32_t e = 0; e < 5; e++)
        EXPECT_EQ(st.beginLaunch(e % 3), e);
    using L = std::vector<std::pair<size_t, uint32_t>>;
    EXPECT_EQ(st.takeLaunched(), (L{{0, 3}, {1, 4}, {2, 2}}));
    EXPECT_TRUE(st.takeLaunched().empty());
    EXPECT_EQ(st.beginLaunch(1), 5u);
    EXPECT_EQ(st.takeLaunched(), (L{{1, 5}}));
    EXPECT_THROW(st.beginLaunch(3), std::out_of_range);
    st.golden = G::Captured;
    EXPECT_TRUE(st.checks());
    st.reset(2);
    EXPECT_EQ(st.launches, 0u);
    EXPECT_EQ(st.golden, G::Pending);

    // From launch 1 on, 2W + 6 launches make every rank the skew victim at least twice.
    for(uint32_t w = 1; w <= FUSED_A2A_MAX_RANKS; w++)
        for(uint32_t r = 0; r < w; r++)
        {
            EXPECT_FALSE(verifySkewVictim(0, r, w));
            uint32_t hits = 0;
            for(uint32_t e = 1; e < 2 * w + 6; e++)
            {
                EXPECT_EQ(verifySkewVictim(e, r, w), e % w == r);
                hits += verifySkewVictim(e, r, w);
            }
            EXPECT_GE(hits, 2u) << "W " << w << " rank " << r;
        }

    VerifyOptions on, off;
    off.barrier = false;
    for(uint32_t e = 0; e < 4; e++)
    {
        EXPECT_TRUE(verifyBarrier(on, e));
        EXPECT_EQ(verifyBarrier(off, e), e == 0);
        for(uint32_t debug = 0; debug <= VIEW_DEBUG_ALL; debug++)
            EXPECT_EQ(verifyViewDebug(debug, e),
                      e == 0 ? debug
                                   & ~(uint32_t)(VIEW_DEBUG_NO_POLL | VIEW_DEBUG_DROP_COPY
                                                 | VIEW_DEBUG_EARLY_RELEASE)
                             : debug);
    }
}

TEST(ViewTransport, VerifyOpsTouchExactlyTheirRanges)
{
    for(bool gemmA2A : {false, true})
        for(uint32_t w : {2u, 4u})
        {
            SCOPED_TRACE(testing::Message() << (gemmA2A ? "GemmA2A" : "A2AGemm") << " W " << w);
            const VerifyLayout l      = verifyLayoutOf(simShape(gemmA2A, w));
            auto               filled = [](size_t n, uint32_t base) {
                std::vector<uint32_t> v(n);
                for(size_t i = 0; i < n; i++)
                    v[i] = base + (uint32_t)i;
                return v;
            };
            auto gA = filled(l.aWords, 0x3F000000u), gB = filled(l.bWords, 0x3E000000u),
                 gD = filled(l.dWords, 0x3D000000u), gR = filled(l.bufWords, 0x3C000000u);
            auto a = filled(l.aWords, 1), b = filled(l.bWords, 2), d = filled(l.dWords, 3),
                 buf = filled(l.bufWords, 4);
            const VerifySlot    s{a.data(), b.data(), d.data(), buf.data()};
            const VerifyGoldens g{gA.data(), gB.data(), gD.data(), gR.data()};
            uint32_t            block[VERIFY_BLOCK_WORDS];
            verifyInitialBlock(block);
            const uint32_t e    = 5;
            const uint32_t bits = verifyPatternBits(e);
            const auto     b0 = b, buf0 = buf;
            runOps(verifyProduceOps(l, s, g, e, true), block);
            for(size_t i = 0; i < l.aWords; i++)
                ASSERT_EQ(a[i], gA[i] ^ verifyPatternMask(l.a, bits, i)) << i;
            for(auto x : d)
                EXPECT_EQ(x, VERIFY_POISON_WORD);
            if(!gemmA2A)
            {
                for(size_t i = 0; i < l.bWords; i++)
                    ASSERT_EQ(b[i], gB[i] ^ verifyPatternMask(l.b, bits, i)) << i;
                for(size_t i = 0; i < l.bufWords; i++)
                    EXPECT_EQ(buf[i], i < l.peerLo ? buf0[i] : VERIFY_POISON_WORD) << i;
            }
            else
            {
                EXPECT_EQ(b, b0);
                for(auto x : buf)
                    EXPECT_EQ(x, VERIFY_POISON_WORD);
            }
            EXPECT_NE(a, gA);

            // The expected outputs of launch e, then one bad last word of each checked buffer.
            for(size_t i = 0; i < l.dWords; i++)
                d[i] = gD[i] ^ verifyPatternMask(l.d, bits, i);
            for(size_t i = 0; i < l.bufWords; i++)
                buf[i] = gR[i] ^ verifyPatternMask(l.recv, bits, i);
            runOps(verifyCheckOps(l, s, g, e), block);
            VerifyCounts c = verifyDecodeBlock(block);
            EXPECT_EQ(c.dMismatches + c.recvMismatches, 0u);
            EXPECT_EQ(c.checked, 1u);
            d.back() ^= 1u;
            buf.back() ^= 1u;
            runOps(verifyCheckOps(l, s, g, e), block);
            c = verifyDecodeBlock(block);
            EXPECT_EQ(c.dMismatches, 1u);
            EXPECT_EQ(c.recvMismatches, gemmA2A ? 1u : 0u);
            EXPECT_EQ(c.firstBad, e);
            EXPECT_EQ(c.checked, 2u);
            d.back() ^= 1u;
            buf.back() ^= 1u;

            for(size_t i = 0; i < l.bufWords; i++)
                buf[i] ^= gemmA2A ? 0u : 1u;
            const auto d1 = d, buf1 = buf;
            runOps(verifyNormalizeOps(l, s, e), block);
            EXPECT_EQ(d, gemmA2A ? gD : d1);
            EXPECT_EQ(buf, gemmA2A ? gR : buf1);
            runOps(verifyRestoreOps(l, s, g), block);
            EXPECT_EQ(a, gA);
            if(!gemmA2A)
                EXPECT_EQ(b, gB);

            auto a2 = filled(l.aWords, 9), b2 = filled(l.bWords, 8), d2 = filled(l.dWords, 7),
                 r2 = filled(l.bufWords, 6);
            const VerifySlot fresh{a2.data(), b2.data(), d2.data(), r2.data()};
            runOps(verifyCaptureInputOps(l, fresh, g), block);
            EXPECT_EQ(gA, a2);
            if(!gemmA2A)
                EXPECT_EQ(gB, b2);
            runOps(verifyCaptureGoldenOps(l, fresh, g), block);
            EXPECT_EQ(gD, d2);
            if(gemmA2A)
                EXPECT_EQ(gR, r2);
            EXPECT_EQ(verifyDecodeBlock(block).checked, 2u);
        }
}

TEST(ViewTransport, VerifyBlockVerdictReportsAndFailsPerDevice)
{
    struct Case
    {
        const char*  name;
        VerifyCounts c;
        bool         report, failed;
    };
    VerifyCounts dOnly, recvOnly, checkedOnly, clean;
    dOnly.dMismatches       = 3;
    dOnly.firstBad          = 4;
    recvOnly.recvMismatches = 5;
    recvOnly.firstBad       = 2;
    checkedOnly.checked     = 7;
    for(auto const& [name, c, report, failed] : {Case{"D only", dOnly, true, true},
                                                 Case{"recv only", recvOnly, true, true},
                                                 Case{"checked only", checkedOnly, true, false},
                                                 Case{"clean", clean, false, false}})
    {
        VerifyCounts total;
        total.checked                   = 1;
        const VerifyBlockVerdict v      = verifyTakeBlock(total, c);
        VerifyCounts             summed = {};
        summed.checked                  = 1;
        verifyAccumulate(summed, c);
        EXPECT_EQ(v.report, report) << name;
        EXPECT_EQ(v.failed, failed) << name;
        EXPECT_EQ(total.dMismatches, summed.dMismatches) << name;
        EXPECT_EQ(total.recvMismatches, summed.recvMismatches) << name;
        EXPECT_EQ(total.checked, summed.checked) << name;
        EXPECT_EQ(total.firstBad, summed.firstBad) << name;
    }
}

TEST(ViewTransport, VerifySkewsOnlyInSkewModeWithTicksOfTheWallClock)
{
    for(int mode : {VERIFY_OFF, VERIFY_PRODUCE, VERIFY_SKEW})
    {
        VerifyOptions o;
        o.mode = mode;
        for(uint32_t w : {2u, 4u, 8u})
            for(uint32_t e = 0; e < 3 * w; e++)
                for(uint32_t r = 0; r < w; r++)
                    EXPECT_EQ(verifySkews(o, e, r, w),
                              mode == VERIFY_SKEW && verifySkewVictim(e, r, w))
                        << mode << " " << w << " " << e << " " << r;
    }
    EXPECT_EQ(verifySkewTicks(1000, 100000), 100000u);
    EXPECT_EQ(verifySkewTicks(1, 100000), 100u);
    EXPECT_EQ(verifySkewTicks(5000, 25000), 125000u);
    EXPECT_EQ(verifySkewTicks(0, 100000), 0u);

    VerifyState st;
    st.reset(2);
    st.addSkew(0);
    EXPECT_EQ(st.skewed, 0u);
    st.addSkew(300);
    st.addSkew(0);
    st.addSkew(500);
    EXPECT_EQ(st.skewed, 2u);
    EXPECT_EQ(st.skewTicks, 500u);
    st.iterations = 2;
    st.reset(1);
    EXPECT_EQ(st.skewed, 0u);
    EXPECT_EQ(st.skewTicks, 0u);
    EXPECT_EQ(st.iterations, 0u);
}

TEST(ViewTransport, VerifyKernelGridRunsEveryWordOnceAndCountsTheLaunchOnce)
{
    const size_t full = (size_t)VERIFY_THREADS * VERIFY_MAX_BLOCKS;
    for(size_t count : {(size_t)1,
                        (size_t)VERIFY_THREADS - 1,
                        (size_t)VERIFY_THREADS,
                        (size_t)VERIFY_THREADS + 1,
                        (size_t)5 * VERIFY_THREADS + 3,
                        full,
                        full + 17})
    {
        SCOPED_TRACE(count);
        const uint32_t blocks = verifyBlocks(count);
        EXPECT_EQ(
            blocks,
            std::min<size_t>((count + VERIFY_THREADS - 1) / VERIFY_THREADS, VERIFY_MAX_BLOCKS));
        std::vector<uint32_t> golden(count, 0x3F800000u), data(count, 0x40000000u);
        std::vector<uint32_t> flip(count, 0x3F800000u);
        VerifyOp check    = verifyOp(VerifyOpKind::Check, data.data(), golden.data(), count);
        check.countLaunch = true;
        check.launch      = 1;
        VerifyOp pattern  = verifyOp(
            VerifyOpKind::Pattern, flip.data(), flip.data(), count, VerifyPattern{1, 0, 1, 4}, 1u);
        uint32_t block[VERIFY_BLOCK_WORDS];
        verifyInitialBlock(block);
        uint32_t leads = 0;
        for(uint32_t b = 0; b < blocks; b++)
            for(uint32_t t = 0; t < VERIFY_THREADS; t++)
            {
                const VerifySpan sp = verifySpan(b, t, blocks, VERIFY_THREADS);
                leads += sp.lead;
                verifyRunOp(check, sp.first, sp.step, block, sp.lead);
                verifyRunOp(pattern, sp.first, sp.step, block, sp.lead);
            }
        const VerifyCounts c = verifyDecodeBlock(block);
        EXPECT_EQ(leads, 1u);
        EXPECT_EQ(c.dMismatches, count);
        EXPECT_EQ(c.checked, 1u);
        EXPECT_EQ(std::count(flip.begin(), flip.end(), 0xBF800000u), (std::ptrdiff_t)count);
    }
    EXPECT_EQ(verifyBlocks(0), 1u);
}

TEST(ViewTransport, VerifyStridePeriodPicksTheDimAndTheNextLargerStride)
{
    using P                            = std::pair<uint64_t, uint64_t>;
    const std::vector<size_t> colMajor = {1, 4, 4 * 64, 4 * 64 * 2};
    EXPECT_EQ(verifyStridePeriod(colMajor, 0), (P{1, 4}));
    EXPECT_EQ(verifyStridePeriod(colMajor, 1), (P{4, 256}));
    EXPECT_EQ(verifyStridePeriod(colMajor, 2), (P{256, 512}));
    EXPECT_EQ(verifyStridePeriod(colMajor, 3), (P{512, 0}));
    EXPECT_EQ(verifyStridePeriod(colMajor, 4), (P{1, 0}));
    const std::vector<size_t> transposed = {64, 1, 64 * 4};
    EXPECT_EQ(verifyStridePeriod(transposed, 1), (P{1, 64}));
    EXPECT_EQ(verifyStridePeriod(transposed, 0), (P{64, 256}));
    const std::vector<size_t> unitDims = {1, 1, 32, 32};
    EXPECT_EQ(verifyStridePeriod(unitDims, 0), (P{1, 32}));
    EXPECT_EQ(verifyStridePeriod(unitDims, 2), (P{32, 0}));
    EXPECT_EQ(verifyStridePeriod({}, 0), (P{1, 0}));
}

TEST(ViewTransport, VerifyGoldensGrowToTheLayoutAndCoverEveryOp)
{
    std::map<void*, size_t>                 live;
    std::vector<std::unique_ptr<uint8_t[]>> storage;
    size_t                                  allocs = 0, frees = 0;
    auto                                    release = [&](void* p) {
        ASSERT_EQ(live.count(p), 1u);
        live.erase(p);
        frees++;
    };
    auto allocate = [&](size_t bytes) {
        storage.emplace_back(new uint8_t[bytes]);
        live[storage.back().get()] = bytes;
        allocs++;
        return static_cast<void*>(storage.back().get());
    };
    for(bool gemmA2A : {false, true})
    {
        SCOPED_TRACE(gemmA2A ? "GemmA2A" : "A2AGemm");
        const VerifyLayout  small = verifyLayoutOf(simShape(gemmA2A, 2));
        const VerifyLayout  large = verifyLayoutOf(simShape(gemmA2A, 4));
        VerifyGoldenBuffers g;
        EXPECT_THROW(verifyGoldensOf(g, small), std::runtime_error);
        verifyGrowGoldens(g, small, release, allocate);
        const VerifyGoldenBytes n = verifyGoldenBytes(small);
        EXPECT_EQ(n.a, small.aWords * 4);
        EXPECT_EQ(n.d, small.dWords * 4);
        EXPECT_EQ(n.b, gemmA2A ? 0 : small.bWords * 4);
        EXPECT_EQ(n.recv, gemmA2A ? small.bufWords * 4 : 0);
        for(auto [buf, bytes] : {std::pair{g.a, n.a}, {g.b, n.b}, {g.d, n.d}, {g.recv, n.recv}})
        {
            EXPECT_EQ(buf.bytes, bytes);
            EXPECT_EQ(buf.p != nullptr, bytes != 0);
            if(buf.p != nullptr)
                EXPECT_EQ(live.at(buf.p), bytes);
        }
        const size_t a0 = allocs;
        verifyGrowGoldens(g, small, release, allocate);
        EXPECT_EQ(allocs, a0);
        const VerifyGoldens gs = verifyGoldensOf(g, small);
        EXPECT_EQ(gs.b == nullptr, gemmA2A);
        EXPECT_EQ(gs.recv == nullptr, !gemmA2A);
        EXPECT_THROW(verifyGoldensOf(g, large), std::runtime_error);
        const size_t f0 = frees;
        verifyGrowGoldens(g, large, release, allocate);
        EXPECT_EQ(frees - f0, (size_t)(gemmA2A ? 3 : 2));
        EXPECT_EQ(g.a.bytes, large.aWords * 4);
        EXPECT_NO_THROW(verifyGoldensOf(g, large));
        const size_t a1 = allocs;
        verifyGrowGoldens(g, small, release, allocate);
        EXPECT_EQ(allocs, a1);
        EXPECT_EQ(g.a.bytes, large.aWords * 4);
        VerifyGoldenBuffers shortD = g;
        shortD.d.bytes -= 4;
        EXPECT_THROW(verifyGoldensOf(shortD, large), std::runtime_error);

        // Every op that reads or writes a golden stays inside verifyGoldenBytes.
        for(uint32_t w : {2u, 4u})
        {
            const VerifyLayout      l  = verifyLayoutOf(simShape(gemmA2A, w));
            const VerifyGoldenBytes nb = verifyGoldenBytes(l);
            std::vector<uint32_t>   a(l.aWords), b(l.bWords), d(l.dWords), buf(l.bufWords);
            std::vector<uint32_t>   ga(nb.a / 4), gb(nb.b / 4), gd(nb.d / 4), gr(nb.recv / 4);
            const VerifySlot        s{a.data(), b.data(), d.data(), buf.data()};
            const VerifyGoldens     gg{ga.data(), gb.data(), gd.data(), gr.data()};
            std::vector<VerifyOp>   ops;
            for(auto const& v : {verifyProduceOps(l, s, gg, 3, true),
                                 verifyCheckOps(l, s, gg, 3),
                                 verifyRestoreOps(l, s, gg),
                                 verifyCaptureInputOps(l, s, gg),
                                 verifyCaptureGoldenOps(l, s, gg)})
                ops.insert(ops.end(), v.begin(), v.end());
            size_t touched = 0;
            for(auto const& op : ops)
                for(void const* p : {(void const*)op.dst, (void const*)op.src})
                    for(auto [golden, bytes] : {std::pair{(void const*)ga.data(), nb.a},
                                                {(void const*)gb.data(), nb.b},
                                                {(void const*)gd.data(), nb.d},
                                                {(void const*)gr.data(), nb.recv}})
                        if(p != nullptr && p == golden && bytes != 0)
                        {
                            EXPECT_EQ(op.count * 4, bytes) << w;
                            touched++;
                        }
            EXPECT_EQ(touched, (size_t)(gemmA2A ? 7 : 8)) << w;
        }
    }
}

// Host model of verify on exact f32 data with the real layout, ops, schedule helpers and state,
// run in main.cpp's order. A2AGemm: D_r = A_r * gathered B, gathered segment j from rank
// (r + j) mod W, segment 0 the own one. GemmA2A: D_s = A_s * B_s, recv_p segment s = rows
// [p*kL, (p+1)*kL) of D_s. Faults hit receiver 0's COPY from rank 1 mod W.
namespace
{
    enum class Fault
    {
        None,
        Missing,
        WrongSource,
        Early, // A2AGemm: before the source's producer; GemmA2A: before the source's GEMM
        EarlyRecv, // GemmA2A: before the receiver's producer
        Late, // after the receiver's GEMM (A2AGemm) or check (GemmA2A)
        NextProducer, // A2AGemm: after the source's producer of the slot's next use
        Stale1, // A2AGemm: the slot at its previous use; GemmA2A: launch e-1
        Stale2, // GemmA2A: launch e-2
        NextSlot,
        PrevSlot,
        Slot0,
        WrongSlotDest,
        PinnedSlot0, // Slot0 at every launch from the fault launch on
        DebugDrop, // Missing at every launch that arms VIEW_DEBUG_DROP_COPY
        GemmSkipsD, // rank 0's GEMM does not write D
        GemmWrongSlot, // rank 0's GEMM reads and writes slot (k + 1) mod S instead of k
    };

    const char* faultName(Fault f)
    {
        static const char* names[] = {"none",
                                      "missing",
                                      "wrong-source",
                                      "early",
                                      "early-recv",
                                      "late",
                                      "next-producer",
                                      "stale-1",
                                      "stale-2",
                                      "next-slot",
                                      "prev-slot",
                                      "slot-0",
                                      "wrong-slot-dest",
                                      "pinned-slot-0",
                                      "debug-drop",
                                      "gemm-skips-d",
                                      "gemm-wrong-slot"};
        return names[(int)f];
    }

    using Words = std::vector<uint32_t>;

    uint32_t bitsOf(float f)
    {
        uint32_t b;
        std::memcpy(&b, &f, sizeof(b));
        return b;
    }

    float floatOf(uint32_t b)
    {
        float f;
        std::memcpy(&f, &b, sizeof(f));
        return f;
    }

    struct SimConfig
    {
        uint32_t world;
        uint32_t slots;
        uint32_t launchesPerIteration;
        uint32_t iterations;
        uint32_t enqueuesPerSync;
        bool     reset; // slow-path reset of slot 0's inputs before every later iteration
        bool     gemmA2A;
    };

    struct SimResult
    {
        VerifyCounts total;
        uint64_t     launches    = 0;
        uint64_t     badLaunches = 0;
        uint64_t     faulted     = 0; // launches the fault changed
        bool         cpuOk       = true;
        bool         applied     = false;
        bool         differs     = false; // receiver 0's data at its consumer differed
        bool         restored    = false; // every slot's inputs equal the pristine ones at the end
        VerifyGolden golden      = VerifyGolden::Pending;
    };

    struct Block
    {
        uint32_t words[VERIFY_BLOCK_WORDS] = {};
    };

    class Sim
    {
    public:
        Sim(SimConfig c, Fault fault = Fault::None, uint32_t faultLaunch = 0)
            : c(c)
            , w(c.world)
            , fault(fault)
            , faultLaunch(faultLaunch)
            , l(verifyLayoutOf(simShape(c.gemmA2A, c.world)))
            , seg(kN * kL)
        {
            for(uint32_t r = 0; r < w; r++)
            {
                Words a(l.aWords), b(c.gemmA2A ? kKg * kN : l.bWords);
                for(size_t i = 0; i < a.size(); i++)
                    a[i] = bitsOf(1.0f + (float)(((i + 1) * 2654435761u + r * 40503u) >> 28 & 3));
                for(size_t i = 0; i < b.size(); i++)
                    b[i] = bitsOf(1.0f
                                  + (float)(((i + 7) * 2246822519u + r * 3266489917u) >> 28 & 3));
                pristineA.push_back(a);
                pristineB.push_back(b);
            }
            A.assign(w, std::vector<Words>(c.slots));
            B.assign(w, std::vector<Words>(c.slots));
            D.assign(w, std::vector<Words>(c.slots, Words(l.dWords, 0)));
            buf.assign(w, std::vector<Words>(c.slots, Words(l.bufWords, 0)));
            for(uint32_t r = 0; r < w; r++)
                for(uint32_t k = 0; k < c.slots; k++)
                {
                    A[r][k] = pristineA[r];
                    B[r][k] = c.gemmA2A ? Words() : pristineB[r];
                }
            gA.assign(w, Words(l.aWords));
            gB.assign(w, Words(l.bWords));
            gD.assign(w, Words(l.dWords));
            gR.assign(w, Words(l.bufWords));
            blocks.assign(w, Block{});
            history.assign(w, std::map<uint32_t, Words>());
            slotHistory.assign(w, std::vector<Words>(c.slots));
        }

        // One solution. The inputs of the problem stay in place for the next call.
        SimResult run()
        {
            res = SimResult{};
            for(auto& b : blocks)
                verifyInitialBlock(b.words);
            state.reset(c.slots);
            for(uint32_t it = 0; it < c.iterations; it++)
            {
                if(it > 0 && c.reset)
                    for(uint32_t r = 0; r < w; r++)
                    {
                        A[r][0] = pristineA[r];
                        if(!c.gemmA2A)
                            B[r][0] = pristineB[r];
                    }
                std::vector<size_t> synced;
                for(uint32_t j = 0; j < c.launchesPerIteration; j++)
                {
                    const uint32_t k = j % c.slots;
                    launch(k);
                    synced.push_back(k);
                    if(synced.size() == c.enqueuesPerSync || j + 1 == c.launchesPerIteration)
                    {
                        for(auto const& [slot, e] : state.takeLaunched())
                            for(uint32_t r = 0; r < w; r++)
                                runOps(verifyNormalizeOps(l, slotOf(r, slot), e), block(r));
                        for(size_t s : synced)
                            res.cpuOk = res.cpuOk && cpuValid(s);
                        synced.clear();
                    }
                }
            }
            for(uint32_t r = 0; r < w; r++)
                for(uint32_t k = 0; k < c.slots; k++)
                    runOps(verifyRestoreOps(l, slotOf(r, k), goldensOf(r)), block(r));
            res.restored = true;
            for(uint32_t r = 0; r < w; r++)
                for(uint32_t k = 0; k < c.slots; k++)
                    res.restored = res.restored && A[r][k] == pristineA[r]
                                   && (c.gemmA2A || B[r][k] == pristineB[r]);
            for(auto const& b : blocks)
                verifyAccumulate(res.total, verifyDecodeBlock(b.words));
            res.golden = state.golden;
            return res;
        }

        // The next problem: rotating slots copied from slot 0, goldens taken again.
        void nextProblem()
        {
            for(uint32_t r = 0; r < w; r++)
                for(uint32_t k = 1; k < c.slots; k++)
                {
                    A[r][k] = A[r][0];
                    B[r][k] = B[r][0];
                }
            inputsCaptured = false;
        }

    private:
        SimConfig                              c;
        uint32_t                               w;
        Fault                                  fault;
        uint32_t                               faultLaunch;
        VerifyLayout                           l;
        size_t                                 seg;
        std::vector<Words>                     pristineA, pristineB, gA, gB, gD, gR;
        std::vector<std::vector<Words>>        A, B, D, buf; // [rank][slot]
        std::vector<Block>                     blocks;
        std::vector<std::map<uint32_t, Words>> history; // GemmA2A: D_s after launch e
        std::vector<std::vector<Words>>        slotHistory; // A2AGemm: B_s after its last producer
        VerifyState                            state;
        SimResult                              res;
        bool                                   inputsCaptured = false;

        uint32_t* block(uint32_t r)
        {
            return blocks[r].words;
        }

        VerifySlot slotOf(uint32_t r, size_t k)
        {
            return {A[r][k].data(), B[r][k].data(), D[r][k].data(), buf[r][k].data()};
        }

        VerifyGoldens goldensOf(uint32_t r)
        {
            return {gA[r].data(), gB[r].data(), gD[r].data(), gR[r].data()};
        }

        static Words segment(Words const& v, size_t i, size_t n)
        {
            return Words(v.begin() + i * n, v.begin() + (i + 1) * n);
        }

        static void setSegment(Words& v, size_t i, Words const& x)
        {
            std::copy(x.begin(), x.end(), v.begin() + i * x.size());
        }

        Words a2aGemmD(Words const& a, std::vector<Words> const& gathered, uint32_t r) const
        {
            Words d(l.dWords);
            for(uint32_t n = 0; n < kN; n++)
                for(uint32_t m = 0; m < kM; m++)
                {
                    float sum = 0;
                    for(uint32_t j = 0; j < w; j++)
                        for(uint32_t kl = 0; kl < kL; kl++)
                            sum += floatOf(a[m + (((r + j) % w) * kL + kl) * kM])
                                   * floatOf(gathered[j][n * kL + kl]);
                    d[m + n * kM] = bitsOf(sum);
                }
            return d;
        }

        Words gemmA2AD(Words const& a, uint32_t s) const
        {
            const uint32_t M = w * kL;
            Words          d(l.dWords);
            for(uint32_t n = 0; n < kN; n++)
                for(uint32_t m = 0; m < M; m++)
                {
                    float sum = 0;
                    for(uint32_t q = 0; q < kKg; q++)
                        sum += floatOf(a[m + q * M]) * floatOf(pristineB[s][q + n * kKg]);
                    d[m + n * M] = bitsOf(sum);
                }
            return d;
        }

        // Rows [p*kL, (p+1)*kL) of D as recv segment [N][kL].
        Words shard(Words const& d, uint32_t p) const
        {
            const uint32_t M = w * kL;
            Words          v(seg);
            for(uint32_t n = 0; n < kN; n++)
                for(uint32_t f = 0; f < kL; f++)
                    v[n * kL + f] = d[p * kL + f + n * M];
            return v;
        }

        Words referenceD(uint32_t r) const
        {
            if(c.gemmA2A)
                return gemmA2AD(pristineA[r], r);
            std::vector<Words> g;
            for(uint32_t j = 0; j < w; j++)
                g.push_back(segment(pristineB[(r + j) % w], r, seg));
            return a2aGemmD(pristineA[r], g, r);
        }

        Words referenceRecv(uint32_t p) const
        {
            Words v(l.bufWords);
            for(uint32_t s = 0; s < w; s++)
                setSegment(v, s, shard(gemmA2AD(pristineA[s], s), p));
            return v;
        }

        bool cpuValid(size_t k) const
        {
            for(uint32_t r = 0; r < w; r++)
                if(D[r][k] != referenceD(r) || (c.gemmA2A && buf[r][k] != referenceRecv(r)))
                    return false;
            return true;
        }

        bool active(uint32_t e) const
        {
            if(fault == Fault::DebugDrop)
                return (verifyViewDebug(VIEW_DEBUG_DROP_COPY, e) & VIEW_DEBUG_DROP_COPY) != 0;
            if(fault == Fault::PinnedSlot0)
                return e >= faultLaunch;
            return fault != Fault::None && e == faultLaunch;
        }

        void launch(uint32_t k)
        {
            const uint32_t e = state.beginLaunch(k);
            res.launches++;
            if(!inputsCaptured)
            {
                for(uint32_t r = 0; r < w; r++)
                    runOps(verifyCaptureInputOps(l, slotOf(r, 0), goldensOf(r)), block(r));
                inputsCaptured = true;
            }
            std::vector<Words> before(w);
            for(uint32_t r = 0; r < w; r++)
            {
                before[r] = c.gemmA2A ? D[r][k] : B[r][k];
                runOps(verifyProduceOps(l, slotOf(r, k), goldensOf(r), e, true), block(r));
            }
            uint64_t bad = 0;
            for(uint32_t r = 0; r < w; r++)
                bad += block(r)[VERIFY_D_MISMATCHES] + block(r)[VERIFY_RECV_MISMATCHES];
            if(c.gemmA2A)
                launchGemmA2A(k, e, active(e));
            else
                launchA2AGemm(k, e, active(e), before);
            uint64_t after = 0;
            for(uint32_t r = 0; r < w; r++)
                after += block(r)[VERIFY_D_MISMATCHES] + block(r)[VERIFY_RECV_MISMATCHES];
            res.badLaunches += after != bad;
            if(e == 0)
            {
                res.cpuOk    = res.cpuOk && cpuValid(0);
                state.golden = verifyGoldenAfterValidation(state.golden, true, !cpuValid(0));
                if(state.golden == VerifyGolden::Captured)
                    for(uint32_t r = 0; r < w; r++)
                        runOps(verifyCaptureGoldenOps(l, slotOf(r, 0), goldensOf(r)), block(r));
            }
        }

        // Rank r's gathered B of slot k: its own segment, then a2aBuf segments 1..W-1.
        std::vector<Words> gatheredOf(uint32_t r, uint32_t k) const
        {
            std::vector<Words> g = {segment(B[r][k], r, seg)};
            for(uint32_t j = 1; j < w; j++)
                g.push_back(segment(buf[r][k], j, seg));
            return g;
        }

        bool gemmFaulty(bool faulty) const
        {
            return faulty && (fault == Fault::GemmSkipsD || fault == Fault::GemmWrongSlot);
        }

        // Rank 0's GEMM of a launch on slot k under GemmSkipsD or GemmWrongSlot; `right`: the D
        // it should write, gemm(j): its D on slot j.
        template <class Gemm>
        void gemmFaultOnRank0(uint32_t k, Words const& right, Gemm gemm)
        {
            if(fault == Fault::GemmWrongSlot)
            {
                const uint32_t j = (k + 1) % c.slots;
                if(j == k)
                {
                    D[0][k] = right;
                    return;
                }
                D[0][j] = gemm(j);
            }
            res.applied  = true;
            const bool d = D[0][k] != right;
            res.differs  = res.differs || d;
            res.faulted += d;
        }

        void check(uint32_t k, uint32_t e)
        {
            if(state.checks())
                for(uint32_t r = 0; r < w; r++)
                    runOps(verifyCheckOps(l, slotOf(r, k), goldensOf(r), e), block(r));
        }

        void launchA2AGemm(uint32_t k, uint32_t e, bool faulty, std::vector<Words> const& before)
        {
            const uint32_t src = 1 % w;
            Words          late;
            bool           lateCopy = false;
            bool           applied  = false;
            for(uint32_t r = 0; r < w; r++)
                for(uint32_t i = 0; i + 1 < w; i++)
                {
                    const uint32_t s    = (r + 1 + i) % w;
                    Words          data = segment(B[s][k], r, seg);
                    uint32_t       dstK = k;
                    if(faulty && r == 0 && i == 0)
                    {
                        applied = true;
                        switch(fault)
                        {
                        case Fault::Missing:
                        case Fault::DebugDrop:
                            continue;
                        case Fault::WrongSource:
                            data = w > 2 ? segment(B[(src + 1) % w][k], 0, seg)
                                         : segment(B[src][k], 1, seg);
                            break;
                        case Fault::Early:
                            data = segment(before[s], r, seg);
                            break;
                        case Fault::Late:
                            late     = data;
                            lateCopy = true;
                            continue;
                        case Fault::NextProducer:
                        {
                            Words next(l.bWords);
                            runOps({verifyOp(VerifyOpKind::Pattern,
                                             next.data(),
                                             gB[s].data(),
                                             l.bWords,
                                             l.b,
                                             verifyPatternBits(e + c.slots))},
                                   block(0));
                            data = segment(next, r, seg);
                            break;
                        }
                        case Fault::Stale1:
                            if(slotHistory[s][k].empty())
                                applied = false;
                            else
                                data = segment(slotHistory[s][k], r, seg);
                            break;
                        case Fault::NextSlot:
                        case Fault::PrevSlot:
                        case Fault::Slot0:
                        case Fault::PinnedSlot0:
                        {
                            const uint32_t j = fault == Fault::NextSlot ? (k + 1) % c.slots
                                               : fault == Fault::PrevSlot
                                                   ? (k + c.slots - 1) % c.slots
                                                   : 0;
                            applied          = j != k;
                            data             = segment(B[s][j], r, seg);
                            break;
                        }
                        case Fault::WrongSlotDest:
                            applied = c.slots > 1;
                            dstK    = (k + 1) % c.slots;
                            break;
                        default:
                            applied = false;
                        }
                    }
                    setSegment(buf[r][dstK], i + 1, data);
                }
            for(uint32_t s = 0; s < w; s++)
                slotHistory[s][k] = B[s][k];
            std::vector<std::vector<Words>> gathered(w);
            for(uint32_t r = 0; r < w; r++)
                gathered[r] = gatheredOf(r, k);
            if(applied)
            {
                res.applied  = true;
                const bool d = gathered[0][1] != segment(B[src][k], 0, seg);
                res.differs  = res.differs || d;
                res.faulted += d;
            }
            const bool gemmFault = gemmFaulty(faulty);
            for(uint32_t r = gemmFault ? 1 : 0; r < w; r++)
                D[r][k] = a2aGemmD(A[r][k], gathered[r], r);
            if(gemmFault)
                gemmFaultOnRank0(k, a2aGemmD(A[0][k], gathered[0], 0), [&](uint32_t j) {
                    return a2aGemmD(A[0][j], gatheredOf(0, j), 0);
                });
            if(lateCopy)
                setSegment(buf[0][k], 1, late);
            check(k, e);
        }

        void launchGemmA2A(uint32_t k, uint32_t e, bool faulty)
        {
            const uint32_t src = 1 % w;
            Words          pending;
            bool           late = false, earlyRecv = false, applied = false;
            const Words    early = shard(D[src][k], 0);
            const bool     gemmFault = gemmFaulty(faulty);
            for(uint32_t s = gemmFault ? 1 : 0; s < w; s++)
                D[s][k] = gemmA2AD(A[s][k], s);
            if(gemmFault)
                gemmFaultOnRank0(
                    k, gemmA2AD(A[0][k], 0), [&](uint32_t j) { return gemmA2AD(A[0][j], 0); });
            for(uint32_t s = 0; s < w; s++)
                history[s][e] = D[s][k];
            for(uint32_t s = 0; s < w; s++)
                for(uint32_t p = 0; p < w; p++)
                {
                    Words    v    = shard(D[s][k], p);
                    uint32_t dest = k;
                    if(faulty && s == src && p == 0)
                    {
                        applied = true;
                        switch(fault)
                        {
                        case Fault::Missing:
                        case Fault::DebugDrop:
                            continue;
                        case Fault::WrongSource:
                            v = shard(D[s][k], (p + 1) % w);
                            break;
                        case Fault::Early:
                            v = early;
                            break;
                        case Fault::EarlyRecv:
                            earlyRecv = true;
                            break;
                        case Fault::Late:
                            pending = v;
                            late    = true;
                            continue;
                        case Fault::Stale1:
                        case Fault::Stale2:
                        {
                            const uint32_t back = fault == Fault::Stale1 ? 1 : 2;
                            if(e < back)
                                applied = false;
                            else
                                v = shard(history[s][e - back], p);
                            break;
                        }
                        case Fault::NextSlot:
                        case Fault::PrevSlot:
                        case Fault::Slot0:
                        case Fault::PinnedSlot0:
                        {
                            const uint32_t j = fault == Fault::NextSlot ? (k + 1) % c.slots
                                               : fault == Fault::PrevSlot
                                                   ? (k + c.slots - 1) % c.slots
                                                   : 0;
                            applied          = j != k;
                            v                = shard(D[s][j], p);
                            break;
                        }
                        case Fault::WrongSlotDest:
                            applied = c.slots > 1;
                            dest    = (k + 1) % c.slots;
                            break;
                        default:
                            applied = false;
                        }
                    }
                    setSegment(buf[p][dest], s, v);
                }
            if(earlyRecv)
                setSegment(buf[0][k], src, Words(seg, VERIFY_POISON_WORD));
            if(applied)
            {
                res.applied  = true;
                const bool d = segment(buf[0][k], src, seg) != shard(D[src][k], 0);
                res.differs  = res.differs || d;
                res.faulted += d;
            }
            check(k, e);
            if(late)
                setSegment(buf[0][k], src, pending);
        }
    };

    std::vector<SimConfig> simConfigs(bool gemmA2A)
    {
        std::vector<SimConfig> out;
        for(uint32_t w : {2u, 3u, 4u})
            for(uint32_t slots : {1u, 2u, 5u})
                for(bool reset : {false, true})
                    out.push_back({w, slots, 7, 2, 3, reset, gemmA2A});
        return out;
    }
}

TEST(ViewTransport, VerifySimCleanRunsHaveNoMismatch)
{
    for(bool gemmA2A : {false, true})
        for(auto const& c : simConfigs(gemmA2A))
        {
            SCOPED_TRACE(testing::Message() << (gemmA2A ? "GemmA2A" : "A2AGemm") << " W " << c.world
                                            << " slots " << c.slots << " reset " << c.reset);
            const SimResult r = Sim(c).run();
            EXPECT_EQ(r.golden, VerifyGolden::Captured);
            EXPECT_EQ(r.total.dMismatches + r.total.recvMismatches, 0u);
            EXPECT_TRUE(r.cpuOk);
            EXPECT_TRUE(r.restored);
            EXPECT_EQ(r.launches, c.launchesPerIteration * c.iterations);
            // V3: every launch but the solution's first is checked on every rank.
            EXPECT_EQ(r.total.checked, c.world * (r.launches - 1));
        }
}

TEST(ViewTransport, VerifySimDetectsEveryTransportFault)
{
    std::map<std::string, uint64_t> runs;
    for(bool gemmA2A : {false, true})
        for(auto const& c : simConfigs(gemmA2A))
            for(Fault f : {Fault::Missing,
                           Fault::WrongSource,
                           Fault::Early,
                           Fault::EarlyRecv,
                           Fault::Late,
                           Fault::NextProducer,
                           Fault::Stale1,
                           Fault::Stale2,
                           Fault::NextSlot,
                           Fault::PrevSlot,
                           Fault::Slot0,
                           Fault::WrongSlotDest,
                           Fault::GemmSkipsD,
                           Fault::GemmWrongSlot})
                for(uint32_t e = 1; e < c.launchesPerIteration * c.iterations; e++)
                {
                    const SimResult r = Sim(c, f, e).run();
                    if(!r.applied)
                        continue;
                    const std::string key
                        = std::string(gemmA2A ? "GemmA2A " : "A2AGemm ") + faultName(f);
                    runs[key]++;
                    SCOPED_TRACE(testing::Message()
                                 << key << " W " << c.world << " slots " << c.slots << " reset "
                                 << c.reset << " launch " << e);
                    EXPECT_TRUE(r.differs);
                    EXPECT_GT(r.total.dMismatches + r.total.recvMismatches, 0u);
                    EXPECT_EQ(r.total.firstBad, e);
                    EXPECT_EQ(r.badLaunches, 1u);
                    EXPECT_EQ(r.golden, VerifyGolden::Captured);
                    EXPECT_TRUE(r.restored);
                }
    uint64_t total = 0;
    for(auto const& [key, n] : runs)
    {
        RecordProperty(key, (int)n);
        total += n;
    }
    EXPECT_GT(total, 2000u);
    for(char const* key : {"A2AGemm prev-slot",
                           "A2AGemm slot-0",
                           "A2AGemm next-slot",
                           "A2AGemm stale-1",
                           "GemmA2A stale-2",
                           "GemmA2A prev-slot",
                           "GemmA2A slot-0",
                           "A2AGemm gemm-skips-d",
                           "A2AGemm gemm-wrong-slot",
                           "GemmA2A gemm-skips-d",
                           "GemmA2A gemm-wrong-slot"})
        EXPECT_GT(runs[key], 0u) << key;
}

TEST(ViewTransport, VerifySimSourcePinnedToSlot0IsSeenInOneRunIteration)
{
    for(bool gemmA2A : {false, true})
        for(uint32_t w : {2u, 4u})
            for(uint32_t slots : {2u, 5u, 13u, 22u})
            {
                SCOPED_TRACE(testing::Message() << (gemmA2A ? "GemmA2A" : "A2AGemm") << " W " << w
                                                << " slots " << slots);
                const SimConfig c{w, slots, 23, 1, 22, false, gemmA2A};
                const SimResult r    = Sim(c, Fault::PinnedSlot0, 1).run();
                uint64_t        off0 = 0;
                for(uint32_t e = 1; e < 23; e++)
                    off0 += e % slots != 0;
                EXPECT_EQ(r.golden, VerifyGolden::Captured);
                EXPECT_EQ(r.total.firstBad, 1u);
                EXPECT_EQ(r.badLaunches, off0);
                EXPECT_EQ(r.faulted, off0);
            }
}

TEST(ViewTransport, VerifySimDebugFaultsSkipLaunch0AndHitEveryLaterLaunch)
{
    for(bool gemmA2A : {false, true})
        for(auto const& c : simConfigs(gemmA2A))
        {
            SCOPED_TRACE(testing::Message() << (gemmA2A ? "GemmA2A" : "A2AGemm") << " W " << c.world
                                            << " slots " << c.slots);
            const SimResult r = Sim(c, Fault::DebugDrop).run();
            EXPECT_EQ(r.golden, VerifyGolden::Captured);
            EXPECT_EQ(r.total.firstBad, 1u);
            EXPECT_EQ(r.badLaunches, r.launches - 1);
            EXPECT_EQ(r.total.checked, c.world * (r.launches - 1));
        }
}

TEST(ViewTransport, VerifySimFaultAtLaunch0DisablesTheGolden)
{
    for(bool gemmA2A : {false, true})
        for(auto const& c : simConfigs(gemmA2A))
        {
            const SimResult r = Sim(c, Fault::Missing, 0).run();
            EXPECT_EQ(r.golden, VerifyGolden::Disabled);
            EXPECT_FALSE(r.cpuOk);
            EXPECT_EQ(r.total.checked, 0u);
            EXPECT_EQ(r.total.dMismatches + r.total.recvMismatches, 0u);
        }
}

TEST(ViewTransport, VerifySimNextProblemStartsFromPristineInputs)
{
    for(bool gemmA2A : {false, true})
        for(auto [slots, launches, iterations] : {std::tuple{3u, 4u, 1u},
                                                  std::tuple{21u, 22u, 1u},
                                                  std::tuple{1u, 23u, 1u},
                                                  std::tuple{3u, 5u, 2u}})
        {
            SCOPED_TRACE(testing::Message() << (gemmA2A ? "GemmA2A" : "A2AGemm") << " slots "
                                            << slots << " launches " << launches);
            Sim sim({3, slots, launches, iterations, launches, false, gemmA2A});
            for(int problem = 0; problem < 3; problem++)
            {
                const SimResult r = sim.run();
                EXPECT_EQ(r.golden, VerifyGolden::Captured) << problem;
                EXPECT_TRUE(r.cpuOk) << problem;
                EXPECT_TRUE(r.restored) << problem;
                EXPECT_EQ(r.total.dMismatches + r.total.recvMismatches, 0u) << problem;
                sim.nextProblem();
            }
        }
}
