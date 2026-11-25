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
                          uint32_t coalesce,
                          uint32_t peerGroups = 0)
    {
        ViewArmArgs a{};
        a.mode           = VIEW_MODE_A2A_GEMM;
        a.world          = world;
        a.numQueues      = world - 1;
        a.subRounds      = subRounds;
        a.peerGroups     = peerGroups;
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

// W=8, B=16, F=16, S=1, coalesce >= Qb: 6 groups {0,1,2} .. {9,10,11} {12,13} {14,15} (GK=Qb=3).
// Queue q serves groups q, q-1, .., 0, 6, .., q+1 with one ATOMIC each; group 6 is empty, and its
// Qb rides on the ATOMIC before it, on queue 6 on the first one.
TEST(ViewTransport, A2AGemmServesContiguousGroupsInInKernelOrder)
{
    ViewArmArgs const a = chainArgs(8, 16, 16, 256, 1, kAll);
    EXPECT_EQ(chainText(a, 0),
              "C0 C1 C2 A+6 C14 C15 A+3 C12 C13 A+3 C9 C10 C11 A+3 C6 C7 C8 A+3 C3 C4 C5 A+3");
    EXPECT_EQ(chainText(a, 1),
              "C3 C4 C5 A+3 C0 C1 C2 A+6 C14 C15 A+3 C12 C13 A+3 C9 C10 C11 A+3 C6 C7 C8 A+3");
    EXPECT_EQ(chainText(a, 5),
              "C14 C15 A+3 C12 C13 A+3 C9 C10 C11 A+3 C6 C7 C8 A+3 C3 C4 C5 A+3 C0 C1 C2 A+6");
    EXPECT_EQ(chainText(a, 6),
              "C14 C15 A+6 C12 C13 A+3 C9 C10 C11 A+3 C6 C7 C8 A+3 C3 C4 C5 A+3 C0 C1 C2 A+3");
}

// W=4, B=7, F=1, numCu=7, A2APeerGroups 0: groups {0,1,2} {3,4} {5,6}, the larger first; each
// short group's ATOMIC brings the flag to the end of its Qb = 3 span.
TEST(ViewTransport, A2AGemmBalancesGroupSizesLargerFirst)
{
    ViewArmArgs const a = chainArgs(4, 7, 1, 7, 1, kAll);
    EXPECT_EQ(viewA2AGemmRoundUnits(a), 3u);
    EXPECT_EQ(chainText(a, 0), "C0 C1 C2 A+3 C5 C6 A+3 C3 C4 A+3");
    EXPECT_EQ(chainText(a, 1), "C3 C4 A+3 C0 C1 C2 A+3 C5 C6 A+3");
    EXPECT_EQ(chainText(a, 2), "C5 C6 A+3 C3 C4 A+3 C0 C1 C2 A+3");
    EXPECT_EQ(chainText(chainArgs(4, 7, 1, 7, 1, 1), 1), "C3 A+1 C4 A+2 C0 A+1 C1 A+1 C2 A+1 C5 A+1 C6 A+2");
}

// W=4, B=8, F=64, S=4, A2APeerGroups 1: one group per batch, Qb = nbMax = 4.
// Queues 1 and 2 are idle in rounds 0..q-1 of batch 0; those rounds ride on their first ATOMIC.
TEST(ViewTransport, A2AGemmOnePeerGroupLeavesQueuesIdle)
{
    ViewArmArgs const a = chainArgs(4, 8, 64, 256, 4, kAll, 1);
    EXPECT_EQ(viewA2AGemmRoundUnits(a), 16u);
    EXPECT_EQ(viewA2AGemmChainUnits(a), 96u);
    const std::string sub0 = "C0.0 C1.0 C2.0 C3.0 A+";
    const std::string rest0
        = " C0.1 C1.1 C2.1 C3.1 A+4 C0.2 C1.2 C2.2 C3.2 A+4 C0.3 C1.3 C2.3 C3.3 A+36 "
          "C4.0 C5.0 C6.0 C7.0 A+4 C4.1 C5.1 C6.1 C7.1 A+4 C4.2 C5.2 C6.2 C7.2 A+4 C4.3 C5.3 C6.3 C7.3 A+";
    EXPECT_EQ(chainText(a, 0), sub0 + "4" + rest0 + "36");
    EXPECT_EQ(chainText(a, 1), sub0 + "20" + rest0 + "20");
    EXPECT_EQ(chainText(a, 2), sub0 + "36" + rest0 + "4");
}

// W=8, B=16, F=32, S=1, A2APeerGroups 2: two groups of 4 per batch, Qb = 4.
TEST(ViewTransport, A2AGemmTwoPeerGroups)
{
    ViewArmArgs const a = chainArgs(8, 16, 32, 256, 1, kAll, 2);
    EXPECT_EQ(viewA2AGemmRoundUnits(a), 4u);
    EXPECT_EQ(chainText(a, 0), "C0 C1 C2 C3 A+24 C4 C5 C6 C7 A+4 C8 C9 C10 C11 A+24 C12 C13 C14 C15 A+4");
    EXPECT_EQ(chainText(a, 1), "C4 C5 C6 C7 A+4 C0 C1 C2 C3 A+24 C12 C13 C14 C15 A+4 C8 C9 C10 C11 A+24");
    EXPECT_EQ(chainText(a, 3), "C4 C5 C6 C7 A+12 C0 C1 C2 C3 A+24 C12 C13 C14 C15 A+4 C8 C9 C10 C11 A+16");
    EXPECT_EQ(chainText(a, 6), "C4 C5 C6 C7 A+24 C0 C1 C2 C3 A+24 C12 C13 C14 C15 A+4 C8 C9 C10 C11 A+4");
}

// W=8, B=16, F=32, S=2, A2APeerGroups 3: groups {0,1,2} {3,4,5} {6,7} per
// batch, Qb = 3, 84 per flag. Queues 2..6 all start with group 2; queue q >= 3 is idle in rounds
// 0..q-3 of batch 0 and in rounds q+1..6 of the last batch.
TEST(ViewTransport, A2AGemmThreePeerGroupsStartQueuesOnTheLastGroup)
{
    ViewArmArgs const a = chainArgs(8, 16, 32, 256, 2, kAll, 3);
    EXPECT_EQ(viewA2AGemmRoundUnits(a), 6u);
    EXPECT_EQ(viewA2AGemmChainUnits(a), 84u);
    EXPECT_EQ(chainText(a, 0),
              "C0.0 C1.0 C2.0 A+3 C0.1 C1.1 C2.1 A+27 C6.0 C7.0 A+3 C6.1 C7.1 A+3 C3.0 C4.0 C5.0 A+3 "
              "C3.1 C4.1 C5.1 A+3 C8.0 C9.0 C10.0 A+3 C8.1 C9.1 C10.1 A+27 C14.0 C15.0 A+3 "
              "C14.1 C15.1 A+3 C11.0 C12.0 C13.0 A+3 C11.1 C12.1 C13.1 A+3");
    const std::string mid = " C6.1 C7.1 A+3 C3.0 C4.0 C5.0 A+3 C3.1 C4.1 C5.1 A+3 C0.0 C1.0 C2.0 A+3 "
                            "C0.1 C1.1 C2.1 A+27 C14.0 C15.0 A+3 C14.1 C15.1 A+3 C11.0 C12.0 C13.0 A+3 "
                            "C11.1 C12.1 C13.1 A+3 C8.0 C9.0 C10.0 A+3 C8.1 C9.1 C10.1 A+";
    EXPECT_EQ(chainText(a, 2), "C6.0 C7.0 A+3" + mid + "27");
    EXPECT_EQ(chainText(a, 3), "C6.0 C7.0 A+9" + mid + "21");
    EXPECT_EQ(chainText(a, 6), "C6.0 C7.0 A+27" + mid + "3");
}

// W=8, B=8, F=32, A2APeerGroups 7: 7 groups {0,1} {2} .. {7}, Qb = 2, no idle queue.
TEST(ViewTransport, A2AGemmSevenPeerGroupsSplitEveryQueue)
{
    ViewArmArgs const a = chainArgs(8, 8, 32, 256, 1, kAll, 7);
    EXPECT_EQ(viewA2AGemmRoundUnits(a), 2u);
    EXPECT_EQ(chainText(a, 0), "C0 C1 A+2 C7 A+2 C6 A+2 C5 A+2 C4 A+2 C3 A+2 C2 A+2");
    EXPECT_EQ(chainText(a, 1), "C2 A+2 C0 C1 A+2 C7 A+2 C6 A+2 C5 A+2 C4 A+2 C3 A+2");
    EXPECT_EQ(chainText(a, 6), "C7 A+2 C6 A+2 C5 A+2 C4 A+2 C3 A+2 C2 A+2 C0 C1 A+2");
}

// viewA2AGemmShape: k = 0 keeps ceil(n / G) blocks per group; k >= 1 caps the group count.
TEST(ViewTransport, A2AGemmShapeFollowsA2APeerGroups)
{
    struct Case
    {
        uint32_t world, peerGroups, n, ng, gk;
    };
    const std::vector<Case> cases = {{8, 0, 16, 6, 3},  {8, 7, 16, 7, 3},  {8, 3, 8, 3, 3},
                                     {8, 1, 8, 1, 8},   {8, 2, 3, 2, 2},   {8, 4, 2, 2, 1},
                                     {4, 0, 7, 3, 3},   {4, 7, 7, 3, 3},   {8, 0, 1, 1, 1},
                                     {8, 3, 1, 1, 1},   {2, 0, 5, 1, 5},   {2, 3, 5, 1, 5},
                                     {8, 0, 0, 0, 0},   {8, 3, 0, 0, 0}};
    for(auto const& c : cases)
    {
        SCOPED_TRACE(testing::Message() << "W=" << c.world << " k=" << c.peerGroups << " n=" << c.n);
        uint32_t ng = 99, gk = 99;
        viewA2AGemmShape(chainArgs(c.world, 1, 1, 1, 1, 1, c.peerGroups), c.n, ng, gk);
        EXPECT_EQ(ng, c.ng);
        EXPECT_EQ(gk, c.gk);
    }
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

// Walks every queue's chain for many (W, B, S, coalesce, F, numCu, A2APeerGroups k) and follows
// each work-group's segment switches as the kernel does: F' = min(F, numCu), batch
// iB = b * F' / numCu, the batch's n blocks laid out as NG groups, the first n + NG - NG * GK of
// GK blocks and the rest of GK - 1 (k = 0: GK = ceil(n / G), NG = ceil(n / GK); else
// NG = min(G, n, k), GK = ceil(n / NG)), Gb the group of b, pos = b - its first block, Qb the GK
// of nbMax blocks; queue i = A2AShardIdx - 1 starting at Gb, ordinal K starting at
// S * iB * G * Qb + pos and stepping by Qb at every switch. Against InKernel's a2aEnqueue order
// (groups min(i, NG-1), .., 0, NG-1, .., min(i, NG-1)+1; per group, per h, the group's COPYs,
// then one ATOMIC) no unit is released after more COPYs of its queue; with coalesce >= Qb each
// one after as many. Each unit is released within coalesce - 1 COPYs after its own, and each
// queue issues ceil(|group| / coalesce) ATOMICs per (group, h).
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
                    for(uint32_t peerGroups : {0u, 1u, 2u, 3u, 4u, 7u})
                    for(auto [features, numCu] : fn)
                    {
                        SCOPED_TRACE(testing::Message()
                                     << "W=" << world << " B=" << blocks << " S=" << subRounds
                                     << " c=" << coalesce << " k=" << peerGroups
                                     << " F=" << features << " numCu=" << numCu);
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
                        // InKernel's groups of a batch of n blocks: (first offset, size) each.
                        auto groupsOf = [&](int64_t n) {
                            int64_t ng = 0, gk = 0;
                            if(peerGroups == 0)
                            {
                                gk = (n + g - 1) / g;
                                ng = (n + gk - 1) / gk;
                            }
                            else
                            {
                                ng = std::min<int64_t>({(int64_t)g, n, (int64_t)peerGroups});
                                gk = (n + ng - 1) / ng;
                            }
                            std::vector<std::pair<int64_t, int64_t>> out;
                            for(int64_t x = 0, at = 0; x < ng; x++)
                            {
                                const int64_t size = x < ng - (ng * gk - n) ? gk : gk - 1;
                                out.push_back({at, size});
                                at += size;
                            }
                            return out;
                        };
                        const uint32_t qb
                            = nbMax == 0 ? 0 : (uint32_t)groupsOf(nbMax).front().second;
                        const uint32_t total = spans.size() * g * qb * subRounds;
                        ViewArmArgs    a     = chainArgs(
                            world, blocks, features, numCu, subRounds, coalesce, peerGroups);
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
                                const auto    groups = groupsOf(hi - lo);
                                const int64_t ng     = groups.size();
                                const int64_t first  = std::min<int64_t>(i, ng - 1);
                                for(int64_t x = 0; x < ng; x++)
                                {
                                    auto const [off, nBlk] = groups[(first - x + ng) % ng];
                                    ASSERT_GT(nBlk, 0);
                                    ASSERT_LE(nBlk, (int64_t)qb);
                                    const int64_t row = lo + off;
                                    for(uint32_t h = 0; h < subRounds; h++)
                                    {
                                        copies += nBlk;
                                        for(int64_t b = row; b < row + nBlk; b++)
                                            inkRelease[i][b * subRounds + h] = copies;
                                        inkAtomics[i]++;
                                        coalescedAtomics[i] += (nBlk + coalesce - 1) / coalesce;
                                    }
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
                            const auto groups = groupsOf(spans[iB].second - bLo);
                            uint32_t   gb     = 0;
                            while(gb + 1 < groups.size() && groups[gb + 1].first <= b - bLo)
                                gb++;
                            const uint32_t pos   = b - bLo - groups[gb].first;
                            ASSERT_LT(pos, groups[gb].second);
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

TEST(ViewTransport, ArmHaltsBothModesAndBooksTheQueue)
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

    for(uint32_t error : {VIEW_ERROR_RING_FULL, VIEW_ERROR_TIMEOUT, VIEW_ERROR_ABORTED})
    {
        ViewArmArgs a = a2aGemmArgs(1);
        FakeQueue   aq(1024, 400);
        aq.attach(a, 0);
        Adds aAdds, aStores;
        EXPECT_EQ(viewArmOrBook(a, 0, 0, error, recordTo(aAdds), recordTo(aStores)), 0u);
        EXPECT_EQ(aq.wptr, 400u);
        EXPECT_EQ(aq.doorbell, 0u);
        EXPECT_EQ(aAdds, (Adds{{kFlag, 4}}));
        EXPECT_EQ(aStores, (Adds{{kFlag, 0u}}));
    }
    EXPECT_FALSE(viewArmHalted(0));
    EXPECT_TRUE(viewArmHalted(VIEW_ERROR_STUCK));
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
            for(uint32_t k : {1u, 2u, 3u, 7u})
            {
                cases.push_back(chainArgs(8, 16, 16, 256, 4, c, k));
                cases.push_back(chainArgs(8, 8, 32, 256, 1, c, k));
                cases.push_back(chainArgs(4, 7, 1, 7, 1, c, k));
                cases.push_back(chainArgs(5, 11, 3, 8, 2, c, k));
            }
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
                                        << a.tokenBlocks << " coalesce " << a.coalesce
                                        << " peerGroups " << a.peerGroups);
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
                                        << a.tokenBlocks << " coalesce " << a.coalesce
                                        << " peerGroups " << a.peerGroups);
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
        SCOPED_TRACE(testing::Message() << "mode " << a.mode << " W " << a.world << " B "
                                        << a.tokenBlocks << " coalesce " << a.coalesce
                                        << " peerGroups " << a.peerGroups);
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

namespace
{
    constexpr uint64_t kTail    = 0x500000000ull;
    constexpr uint64_t kScratch = 0x600000000ull;

    // Rank x's view-block tail.
    uint64_t tailAt(uint32_t x)
    {
        return kTail + x * 0x1000ull;
    }

    uint32_t peerOf(ViewArmArgs const& a, uint32_t q)
    {
        return a.mode == VIEW_MODE_A2A_GEMM ? (a.rank + 1 + q) % a.world : q;
    }

    // The readiness words of every queue, as makeViewArmArgs sets them.
    ViewArmArgs withTail(ViewArmArgs a)
    {
        a.tail = fakePointer<uint32_t>(tailAt(a.rank));
        for(uint32_t q = 0; q < a.numQueues; q++)
        {
            const uint32_t peer = peerOf(a, q);
            if(peer == a.rank)
                continue;
            a.queues[q].ready
                = fakePointer<uint32_t>(tailAt(a.rank) + VIEW_TAIL_ENTRY_OFFSET + 4 * peer);
            a.queues[q].entryOut
                = fakePointer<uint32_t>(tailAt(peer) + VIEW_TAIL_ENTRY_OFFSET + 4 * a.rank);
            if(a.mode == VIEW_MODE_A2A_GEMM)
                a.queues[q].pulled
                    = fakePointer<uint32_t>(tailAt(peer) + VIEW_TAIL_PULLED_OFFSET + 4 * a.rank);
        }
        return a;
    }

    uint64_t packetAddress(Packet const& p)
    {
        return (uint64_t)p.words[2] << 32 | p.words[1];
    }

    std::vector<ViewArmArgs> readinessCases()
    {
        std::vector<ViewArmArgs> cases;
        for(auto a : faultCases())
            for(uint32_t r = 0; r < a.world; r++)
            {
                if(a.mode == VIEW_MODE_GEMM_A2A && r != a.rank)
                    continue;
                if(a.mode == VIEW_MODE_A2A_GEMM)
                {
                    a.rank = r;
                    for(uint32_t q = 0; q < a.numQueues; q++)
                        if(a.queues[q].flag == nullptr)
                            a.queues[q] = a.queues[0];
                }
                cases.push_back(a);
            }
        for(uint32_t blocks : {0u, 1u})
            for(uint32_t features : {0u, 1u})
                cases.push_back(chainArgs(4, blocks, features, 8, 2, 1));
        ViewArmArgs g  = dumpedGemmA2AArgs();
        g.tokenBlocks  = 0;
        cases.push_back(g);
        return cases;
    }
}

// R-bytes and the head/tail layout: a chain with a unit and a ready word starts with POLL(ready,
// e + 1); an A2AGemm chain with a unit has one ATOMIC(pulled, 1), directly before its last ATOMIC,
// which is a flag ATOMIC; without the two packets, and with NO_READY | NO_EXITWAIT, the chain is
// the step-1 chain.
TEST(ViewTransport, ReadinessHeadPollAndPulledAtomicPlacement)
{
    for(auto const& base : readinessCases())
        for(uint32_t q = 0; q < base.numQueues; q++)
            for(uint32_t epoch : {0u, 5u, 0x7FFFFFFEu})
            {
                SCOPED_TRACE(testing::Message()
                             << "mode " << base.mode << " W " << base.world << " rank " << base.rank
                             << " B " << base.tokenBlocks << " F " << base.featureTiles << " c "
                             << base.coalesce << " k " << base.peerGroups << " q " << q
                             << " epoch " << epoch);
                ViewArmArgs const full  = withTail(base);
                auto const        plain = buildWhole(base, q, epoch);
                auto              p     = packets(buildWhole(full, q, epoch));
                const bool        head  = viewChainHasUnit(base, q) && peerOf(base, q) != base.rank;
                const bool        tail  = head && base.mode == VIEW_MODE_A2A_GEMM;
                if(head)
                {
                    ASSERT_FALSE(p.empty());
                    EXPECT_EQ(p[0].header, VIEW_POLL_HEADER);
                    EXPECT_EQ(packetAddress(p[0]), viewAddress(full.queues[q].ready));
                    EXPECT_EQ(p[0].words[3], epoch + 1u);
                    p.erase(p.begin());
                }
                else if(!p.empty() && p[0].header == VIEW_POLL_HEADER)
                    EXPECT_NE(packetAddress(p[0]), viewAddress(full.queues[q].ready));
                size_t pulledAt = p.size(), pulledCount = 0;
                for(size_t i = 0; i < p.size(); i++)
                    if(p[i].header == VIEW_ATOMIC_HEADER && full.queues[q].pulled != nullptr
                       && packetAddress(p[i]) == viewAddress(full.queues[q].pulled))
                    {
                        pulledAt = i;
                        pulledCount++;
                        EXPECT_EQ(p[i].words[3], 1u);
                    }
                EXPECT_EQ(pulledCount, tail ? 1u : 0u);
                if(tail)
                {
                    ASSERT_EQ(pulledAt + 2, p.size());
                    EXPECT_EQ(p.back().header, VIEW_ATOMIC_HEADER);
                    EXPECT_EQ(packetAddress(p.back()), viewAddress(full.queues[q].flag));
                    p.erase(p.begin() + pulledAt);
                    uint64_t flagSum = 0;
                    for(auto const& k : only(p, VIEW_ATOMIC_HEADER))
                        flagSum += k.words[3];
                    EXPECT_EQ(flagSum, viewA2AGemmChainUnits(base));
                }
                EXPECT_TRUE(p == packets(plain));

                ViewArmArgs off = full;
                off.debug |= VIEW_DEBUG_NO_READY | VIEW_DEBUG_NO_EXITWAIT;
                EXPECT_EQ(buildWhole(off, q, epoch), plain);
                ViewArmArgs noReady = full;
                noReady.debug |= VIEW_DEBUG_NO_READY;
                auto const nr = packets(buildWhole(noReady, q, epoch));
                EXPECT_EQ(nr.size() + (head ? 1 : 0), packets(buildWhole(full, q, epoch)).size());
                ViewArmArgs noExit = full;
                noExit.debug |= VIEW_DEBUG_NO_EXITWAIT;
                auto const ne = packets(buildWhole(noExit, q, epoch));
                EXPECT_EQ(ne.size() + (tail ? 1 : 0), packets(buildWhole(full, q, epoch)).size());
            }
}

// The step-1 hashes of DebugBitsZeroKeepTheBuilderBytes, with every readiness word set and
// VIEW_DEBUG_NO_READY | VIEW_DEBUG_NO_EXITWAIT.
TEST(ViewTransport, ReadinessOffKeepsTheStep1Bytes)
{
    ViewArmArgs a = withTail(dumpedA2AGemmArgs());
    for(uint32_t q = 1; q < a.numQueues; q++)
    {
        a.queues[q].srcBase = a.queues[0].srcBase;
        a.queues[q].dstBase = a.queues[0].dstBase;
    }
    ViewArmArgs g = withTail(dumpedGemmA2AArgs());
    a.debug = g.debug = VIEW_DEBUG_NO_READY | VIEW_DEBUG_NO_EXITWAIT;
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
    a.debug = g.debug = 0;
    EXPECT_NE(fnv1a(build(a, 0, 0)), 0xabac02f7ed9f5213ull);
    EXPECT_NE(fnv1a(build(g, 1, 5)), 0x50ae60652e042a97ull);
    EXPECT_EQ(fnv1a(build(g, 0, 5)), 0xb2abe52fd03dbb4full);
}

// VIEW_DEBUG_DELAY_CHAIN: queue 0 of delayRank gets delayCopies COPYs from delayScratch to
// delayScratch + srcBlockStride right before its last COPY; nothing else changes.
TEST(ViewTransport, DelayChainInsertsCopiesBeforeQueue0sLastCopy)
{
    for(auto base : readinessCases())
    {
        if(base.mode != VIEW_MODE_A2A_GEMM)
            continue;
        for(int debug : {0, VIEW_DEBUG_NO_READY, VIEW_DEBUG_NO_EXITWAIT})
            for(uint32_t delayRank : {0u, 1u})
                for(uint32_t q = 0; q < base.numQueues; q++)
                {
                    SCOPED_TRACE(testing::Message()
                                 << "W " << base.world << " rank " << base.rank << " B "
                                 << base.tokenBlocks << " c " << base.coalesce << " k "
                                 << base.peerGroups << " q " << q << " delayRank " << delayRank
                                 << " debug " << debug);
                    ViewArmArgs a = withTail(base);
                    a.debug       = debug;
                    auto const on = buildWhole(a, q, 3);
                    a.debug |= VIEW_DEBUG_DELAY_CHAIN;
                    a.delayRank   = delayRank;
                    a.delayCopies = 3;
                    EXPECT_EQ(buildWhole(a, q, 3), on);
                    a.delayScratch = kScratch;
                    auto       p     = packets(buildWhole(a, q, 3));
                    const bool delay = q == 0 && a.rank == delayRank && viewChainHasUnit(a, q);
                    size_t     first = p.size(), count = 0;
                    for(size_t i = 0; i < p.size(); i++)
                        if(p[i].header == VIEW_COPY_HEADER && packetAddress(p[i]) == kScratch)
                        {
                            EXPECT_EQ((uint64_t)p[i].words[7] << 32 | p[i].words[6],
                                      kScratch + a.srcBlockStride);
                            first = std::min(first, i);
                            count++;
                        }
                    EXPECT_EQ(count, delay ? 3u : 0u);
                    if(delay)
                    {
                        size_t lastCopy = 0;
                        for(size_t i = 0; i < p.size(); i++)
                            if(p[i].header == VIEW_COPY_HEADER)
                                lastCopy = i;
                        EXPECT_EQ(first + 3, lastCopy);
                        p.erase(p.begin() + first, p.begin() + first + 3);
                    }
                    EXPECT_TRUE(p == packets(on));
                }
    }
}

// Booking adds the chain's pulled word once (not under NO_EXITWAIT); an armed chain without a
// unit books it too; the arm thread publishes first, also when it books.
TEST(ViewTransport, ReadinessBookingAndPublish)
{
    using Event = std::tuple<char, uint64_t, uint32_t>;
    for(bool gemmA2A : {false, true})
        for(int debug : {0, VIEW_DEBUG_NO_READY, VIEW_DEBUG_NO_EXITWAIT})
            for(uint32_t error : {0u, VIEW_ERROR_TIMEOUT})
                for(bool bookOnly : {false, true})
                {
                    SCOPED_TRACE(testing::Message() << "gemmA2A " << gemmA2A << " debug " << debug
                                                    << " error " << error << " bookOnly "
                                                    << bookOnly);
                    ViewArmArgs a = withTail(gemmA2A ? gemmA2AArgs() : a2aGemmArgs(1));
                    const uint32_t q = gemmA2A ? 1 : 0;
                    a.debug          = debug;
                    FakeQueue fq(1024, 400);
                    fq.attach(a, q);
                    std::vector<Event> ev;
                    auto add = [&](uint32_t* p, uint32_t v) { ev.push_back({'a', viewAddress(p), v}); };
                    auto store
                        = [&](uint32_t* p, uint32_t v) { ev.push_back({'s', viewAddress(p), v}); };
                    auto rel
                        = [&](uint32_t* p, uint32_t v) { ev.push_back({'r', viewAddress(p), v}); };
                    const uint32_t bits = viewArmThread(a, q, 6, error, bookOnly, add, store, rel);
                    EXPECT_EQ(bits, 0u);
                    const bool books   = error != 0 || bookOnly;
                    const bool publish = (debug & VIEW_DEBUG_NO_READY) == 0;
                    const bool pulled  = !gemmA2A && (debug & VIEW_DEBUG_NO_EXITWAIT) == 0;
                    std::vector<Event> want;
                    if(publish)
                        want.push_back({'r', viewAddress(a.queues[q].entryOut), 7u});
                    if(!gemmA2A)
                        want.push_back({'s', kFlag, 0u});
                    if(books)
                    {
                        if(gemmA2A)
                        {
                            want.push_back({'a', kFlag, 2u});
                            want.push_back({'a', kView + FUSED_A2A_VIEW_OUTBOUND_OFFSET, 1u});
                        }
                        else
                        {
                            want.push_back({'a', kFlag, viewA2AGemmChainUnits(a)});
                            if(pulled)
                                want.push_back({'a', viewAddress(a.queues[q].pulled), 1u});
                        }
                    }
                    EXPECT_EQ(ev, want);
                    EXPECT_EQ(fq.wptr != 400, !books);
                }

    // An empty A2AGemm chain is armed with nothing in it and books pulled once.
    for(uint32_t features : {0u, 1u})
    {
        ViewArmArgs a = withTail(chainArgs(3, features == 0 ? 4 : 0, features, 8, 1, 1));
        a.rank        = 1;
        a             = withTail(a);
        FakeQueue fq(1024, 400);
        fq.attach(a, 1);
        Adds adds, stores;
        EXPECT_EQ(viewArmOrBook(a, 1, 0, 0, recordTo(adds), recordTo(stores)), 0u);
        EXPECT_EQ(fq.wptr, 400u);
        EXPECT_EQ(adds, (Adds{{viewAddress(a.queues[1].pulled), 1u}}));
        a.debug = VIEW_DEBUG_NO_EXITWAIT;
        adds.clear();
        EXPECT_EQ(viewArmOrBook(a, 1, 0, 0, recordTo(adds), recordTo(stores)), 0u);
        EXPECT_TRUE(adds.empty());
    }
}

// pulled[r] - epoch as int32: the exit wait across the u32 wrap.
TEST(ViewTransport, ExitDoneComparesWrapSafe)
{
    uint32_t    tail[VIEW_TAIL_BYTES / 4] = {};
    ViewArmArgs a                         = chainArgs(4, 4, 1, 8, 1, 1);
    a.rank                                = 1;
    a.tail                                = tail;
    uint32_t* const pulled                = tail + VIEW_TAIL_PULLED_OFFSET / 4;
    for(uint32_t epoch : {1u, 5u, 0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFFu, 0u})
    {
        SCOPED_TRACE(testing::Message() << "epoch " << epoch);
        for(uint32_t r = 0; r < 4; r++)
            pulled[r] = epoch;
        pulled[1] = epoch - 100;
        EXPECT_TRUE(viewExitDone(a, epoch));
        for(uint32_t r : {0u, 2u, 3u})
        {
            for(uint32_t lag : {1u, 0x7FFFFFFFu, 0x80000000u})
            {
                pulled[r] = epoch - lag;
                EXPECT_FALSE(viewExitDone(a, epoch)) << "r " << r << " lag " << lag;
            }
            pulled[r] = epoch + 0x7FFFFFFFu;
            EXPECT_TRUE(viewExitDone(a, epoch));
            pulled[r] = epoch;
        }
        pulled[2] = epoch - 1;
        a.debug   = VIEW_DEBUG_NO_EXITWAIT;
        EXPECT_TRUE(viewExitDone(a, epoch));
        EXPECT_FALSE(viewExitWaits(a));
        a.debug = 0;
        EXPECT_FALSE(viewExitDone(a, epoch));
        EXPECT_TRUE(viewExitWaits(a));
        pulled[2] = epoch;
    }
    ViewArmArgs one = chainArgs(2, 4, 1, 8, 1, 1);
    one.world       = 1;
    EXPECT_FALSE(viewExitWaits(one));
    EXPECT_FALSE(viewExitWaits(gemmA2AArgs()));
    // VIEW_TAIL_RELEASED >= every u32 head POLL reference.
    for(uint32_t ref : {1u, 0x80000000u, 0xFFFFFFFFu})
        EXPECT_GE(VIEW_TAIL_RELEASED, ref);
}

// NO_READY keeps the exit wait and the pulled word; only NO_EXITWAIT drops them.
TEST(ViewTransport, NoReadyKeepsTheExitWait)
{
    uint32_t    tail[VIEW_TAIL_BYTES / 4] = {};
    ViewArmArgs a                         = chainArgs(4, 4, 1, 8, 1, 1);
    a.rank                                = 1;
    a.tail                                = tail;
    a.queues[0].pulled                    = tail + VIEW_TAIL_PULLED_OFFSET / 4;
    tail[VIEW_TAIL_PULLED_OFFSET / 4]     = 2;
    for(int debug : {0, VIEW_DEBUG_NO_READY, VIEW_DEBUG_NO_EXITWAIT,
                     VIEW_DEBUG_NO_READY | VIEW_DEBUG_NO_EXITWAIT})
    {
        a.debug          = (uint32_t)debug;
        const bool waits = (debug & VIEW_DEBUG_NO_EXITWAIT) == 0;
        EXPECT_EQ(viewExitWaits(a), waits) << debug;
        EXPECT_EQ(viewExitDone(a, 3), !waits) << debug;
        EXPECT_EQ(viewPulled(a, 0), waits ? a.queues[0].pulled : nullptr) << debug;
    }
}

namespace
{
    struct ReleaseCalls
    {
        size_t copyBytes = 0, fillCount = 0;
        uint32_t fillValue = 0;
    };

    // viewReleaseTail of `tail` by `path` with host fakes of the three paths.
    bool releaseTail(uint32_t* tail, int path, ReleaseCalls& calls, bool ok = true)
    {
        std::vector<uint32_t> source(VIEW_TAIL_RELEASE_WORDS + 2, 0u);
        viewFillReleaseSource(source.data());
        EXPECT_EQ(source[VIEW_TAIL_RELEASE_WORDS], 0u);
        return viewReleaseTail(
            tail,
            path,
            [](uint32_t* p, uint32_t v) { *p = v; },
            [&](uint32_t* to, size_t bytes) {
                calls.copyBytes = bytes;
                std::memcpy(to, source.data(), bytes);
                return ok;
            },
            [&](uint32_t* to, size_t count, uint32_t value) {
                calls.fillCount = count;
                calls.fillValue = value;
                for(size_t i = 0; i < count; i++)
                    to[i] = value;
                return ok;
            });
    }
}

// Every release path stores VIEW_TAIL_RELEASED into entry[] and the released word and leaves
// pulled[] alone; the released word ends every exit wait, also after later pulled ATOMICs, and
// entry[] satisfies every head POLL.
TEST(ViewTransport, HostReleaseEndsTheExitWaitAndKeepsPulled)
{
    static_assert(VIEW_TAIL_RELEASED_OFFSET / 4 < VIEW_TAIL_RELEASE_WORDS);
    static_assert(VIEW_TAIL_PULLED_OFFSET >= VIEW_TAIL_RELEASE_BYTES);
    for(int path : {VIEW_RELEASE_HOST_STORE, VIEW_RELEASE_MEMCPY, VIEW_RELEASE_KERNEL})
        for(uint32_t world : {2u, 4u, 8u})
            for(uint32_t epoch : {1u, 6u, 1000u, 0x7FFFFFFFu})
            {
                SCOPED_TRACE(testing::Message()
                             << "path " << path << " W " << world << " epoch " << epoch);
                uint32_t    tail[VIEW_TAIL_BYTES / 4] = {};
                ViewArmArgs a                         = chainArgs(world, 4, 1, 8, 1, 1);
                a.rank                                = world - 1;
                a.tail                                = tail;
                uint32_t* const entry                 = tail + VIEW_TAIL_ENTRY_OFFSET / 4;
                uint32_t* const pulled                = tail + VIEW_TAIL_PULLED_OFFSET / 4;
                for(uint32_t r = 0; r < world; r++)
                {
                    entry[r]  = epoch - 1;
                    pulled[r] = epoch - 1;
                }
                ASSERT_FALSE(viewExitDone(a, epoch));
                std::vector<uint32_t> before(tail, tail + VIEW_TAIL_BYTES / 4);
                ReleaseCalls          calls;
                ASSERT_TRUE(releaseTail(tail, path, calls));
                for(uint32_t i = 0; i < VIEW_TAIL_BYTES / 4; i++)
                    EXPECT_EQ(tail[i], i < VIEW_TAIL_RELEASE_WORDS ? VIEW_TAIL_RELEASED : before[i])
                        << "word " << i;
                if(path == VIEW_RELEASE_MEMCPY)
                    EXPECT_EQ(calls.copyBytes, (size_t)VIEW_TAIL_RELEASE_BYTES);
                if(path == VIEW_RELEASE_KERNEL)
                {
                    EXPECT_EQ(calls.fillCount, (size_t)VIEW_TAIL_RELEASE_WORDS);
                    EXPECT_EQ(calls.fillValue, VIEW_TAIL_RELEASED);
                }
                EXPECT_TRUE(viewExitDone(a, epoch));
                for(uint32_t late = 0; late < 3; late++)
                {
                    for(uint32_t r = 0; r < world; r++)
                        pulled[r] += 1;
                    EXPECT_TRUE(viewExitDone(a, epoch + late));
                }
                for(uint32_t r = 0; r < world; r++)
                    EXPECT_GE(entry[r], epoch + 1);
            }
    uint32_t     tail[VIEW_TAIL_BYTES / 4] = {};
    ReleaseCalls calls;
    EXPECT_FALSE(releaseTail(tail, VIEW_RELEASE_MEMCPY, calls, false));
    EXPECT_FALSE(releaseTail(tail, VIEW_RELEASE_KERNEL, calls, false));
}

namespace
{
    struct FakeStream
    {
        uint64_t   now = 0, queries = 0, syncs = 0, idleAt = ~0ull;
        hipError_t busyError = hipErrorNotReady, syncResult = hipSuccess;

        hipError_t query()
        {
            queries++;
            if(queries > 1000)
                return hipErrorUnknown;
            return queries >= idleAt ? hipSuccess : busyError;
        }
    };
}

// viewPollIdle returns hipErrorNotReady once the clock passes the deadline, the first result
// other than hipErrorNotReady otherwise; viewSyncWithin synchronizes only an idle stream.
TEST(ViewTransport, BoundedSyncStopsAtTheDeadline)
{
    static_assert(kViewSyncBoundMs > 2 * kViewWaitTimeoutMs);
    auto poll = [](FakeStream& f, uint64_t deadline) {
        return viewPollIdle([&] { return f.query(); },
                            [&] { return f.now; },
                            [&] { f.now++; },
                            deadline);
    };
    FakeStream busy;
    EXPECT_EQ(poll(busy, 10), hipErrorNotReady);
    EXPECT_EQ(busy.queries, 12u);
    FakeStream idle;
    idle.idleAt = 3;
    EXPECT_EQ(poll(idle, 10), hipSuccess);
    EXPECT_EQ(idle.queries, 3u);
    FakeStream faulted;
    faulted.busyError = hipErrorIllegalAddress;
    EXPECT_EQ(poll(faulted, 10), hipErrorIllegalAddress);
    EXPECT_EQ(faulted.queries, 1u);

    auto sync = [](FakeStream& f, uint64_t bound) {
        return viewSyncWithin([&] { return f.query(); },
                              [&] {
                                  f.syncs++;
                                  return f.syncResult;
                              },
                              [&] { return f.now; },
                              [&] { f.now++; },
                              bound);
    };
    FakeStream stuck;
    stuck.now = 100;
    EXPECT_EQ(sync(stuck, 10), hipErrorNotReady);
    EXPECT_EQ(stuck.queries, 12u);
    EXPECT_EQ(stuck.syncs, 0u);
    FakeStream done;
    done.idleAt     = 2;
    done.syncResult = hipErrorLaunchFailure;
    EXPECT_EQ(sync(done, 10), hipErrorLaunchFailure);
    EXPECT_EQ(done.syncs, 1u);
    FakeStream lost;
    lost.busyError = hipErrorIllegalAddress;
    EXPECT_EQ(sync(lost, 10), hipErrorIllegalAddress);
    EXPECT_EQ(lost.syncs, 0u);
}

TEST(ViewTransport, ReleaseEntryRaisesOnlyThePeersEntries)
{
    uint32_t    tail[VIEW_TAIL_BYTES / 4] = {};
    ViewArmArgs g                         = gemmA2AArgs();
    g.world = g.numQueues = 4;
    g.rank                = 2;
    g.tail                = tail;
    uint32_t* const entry = tail + VIEW_TAIL_ENTRY_OFFSET / 4;
    entry[0]              = 3;
    entry[1]              = 9;
    entry[2]              = 1;
    entry[3]              = 0;
    viewReleaseEntry(g, 5, viewMax32);
    EXPECT_EQ(std::vector<uint32_t>(entry, entry + 4), (std::vector<uint32_t>{5, 9, 1, 5}));
    EXPECT_EQ(std::vector<uint32_t>(tail + 4, tail + VIEW_TAIL_BYTES / 4),
              std::vector<uint32_t>(VIEW_TAIL_BYTES / 4 - 4, 0u));
}

TEST(ViewTransport, TailCountsMatchEveryPeer)
{
    uint32_t tail[VIEW_TAIL_BYTES / 4] = {};
    for(uint32_t x = 0; x < 4; x++)
        tail[VIEW_TAIL_ENTRY_OFFSET / 4 + x] = tail[VIEW_TAIL_PULLED_OFFSET / 4 + x] = 7;
    tail[VIEW_TAIL_ENTRY_OFFSET / 4 + 1] = tail[VIEW_TAIL_PULLED_OFFSET / 4 + 1] = 0;
    for(uint32_t mode : {VIEW_MODE_A2A_GEMM, VIEW_MODE_GEMM_A2A})
    {
        EXPECT_TRUE(viewTailCountsMatch(mode, 1, 4, 7, tail, true, true));
        EXPECT_FALSE(viewTailCountsMatch(mode, 0, 4, 7, tail, true, true));
        EXPECT_FALSE(viewTailCountsMatch(mode, 1, 4, 8, tail, true, false));
    }
    tail[VIEW_TAIL_PULLED_OFFSET / 4 + 3] = 6;
    EXPECT_FALSE(viewTailCountsMatch(VIEW_MODE_A2A_GEMM, 1, 4, 7, tail, true, true));
    EXPECT_TRUE(viewTailCountsMatch(VIEW_MODE_A2A_GEMM, 1, 4, 7, tail, true, false));
    EXPECT_TRUE(viewTailCountsMatch(VIEW_MODE_GEMM_A2A, 1, 4, 7, tail, true, true));
    tail[VIEW_TAIL_ENTRY_OFFSET / 4 + 2] = 6;
    EXPECT_TRUE(viewTailCountsMatch(VIEW_MODE_GEMM_A2A, 1, 4, 7, tail, false, true));
    EXPECT_FALSE(viewTailCountsMatch(VIEW_MODE_GEMM_A2A, 1, 4, 7, tail, true, false));
}

TEST(ViewTransport, ThrowAtParsesSiteDeviceAndLaunch)
{
    EXPECT_FALSE(viewParseThrowAt("").set);
    const char* names[] = {"start", "arm", "launch", "wait", "stop", "check"};
    for(int i = 0; i < 6; i++)
    {
        const ViewThrowAt t = viewParseThrowAt(std::string(names[i]) + ":3:12");
        EXPECT_TRUE(t.set);
        EXPECT_EQ((int)t.site, i);
        EXPECT_EQ(t.device, 3u);
        EXPECT_EQ(t.launch, 12u);
    }
    for(auto bad : {"arm", "arm:1", "arm:1:", "x:1:2", "arm:-1:2", "arm:1:2x", ":1:2"})
        EXPECT_THROW(viewParseThrowAt(bad), std::invalid_argument) << bad;
}

// Every rank arms every launch with the same readiness bits; launch 0 of verify keeps them.
TEST(ViewTransport, ReadinessBitsAndTimelineFields)
{
    constexpr int readiness
        = VIEW_DEBUG_NO_READY | VIEW_DEBUG_NO_EXITWAIT | VIEW_DEBUG_DELAY_CHAIN;
    static_assert((VERIFY_LAUNCH0_CLEARED_DEBUG & readiness) == 0);
    static_assert((readiness & 31) == 0 && VIEW_DEBUG_ALL == (31 | readiness));
    EXPECT_EQ(verifyViewDebug(readiness, 0), (uint32_t)readiness);
    static_assert(VIEW_STATS_TS_READY >= VIEW_STATS_TS_END + 2 * (FUSED_A2A_MAX_RANKS - 1) + 1);
    static_assert(VIEW_STATS_TS_READY + FUSED_A2A_MAX_RANKS == VIEW_STATS_FIELDS);

    ViewArmArgs g = withTail(dumpedGemmA2AArgs());
    g.stats       = fakePointer<uint64_t>(0x700000000ull);
    auto const r  = buildWhole(g, 1, 2);
    ASSERT_GE(r.size(), 2 * VIEW_TIMESTAMP_DWORDS + VIEW_POLL_DWORDS);
    EXPECT_EQ(r[0], VIEW_TIMESTAMP_HEADER);
    EXPECT_EQ(r[VIEW_TIMESTAMP_DWORDS], VIEW_POLL_HEADER);
    const size_t ts = VIEW_TIMESTAMP_DWORDS + VIEW_POLL_DWORDS;
    EXPECT_EQ(r[ts], VIEW_TIMESTAMP_HEADER);
    EXPECT_EQ((uint64_t)r[ts + 2] << 32 | r[ts + 1],
              viewAddress(g.stats + viewStatsIndex(2, VIEW_STATS_TS_READY + 1)));
}

// The first match throws and clears the setting; a mismatch in any field does not. Replays
// launchAll's calls: 6 solutions alternating InKernel (no calls) and View, 4 launches each (comm
// yaml) or 23 (verify yaml), devices 0-3, every site; a throw ends its solution.
TEST(ViewTransport, ThrowAtFiresOnceAtTheFirstMatch)
{
    for(uint64_t perSolution : {4u, 23u})
        for(uint64_t at : {0u, 3u, 5u})
        {
            SCOPED_TRACE(testing::Message() << "launches " << perSolution << " at " << at);
            ViewThrowAt           t = viewParseThrowAt("check:1:" + std::to_string(at));
            std::vector<uint32_t> threw, ran;
            for(uint32_t solution = 1; solution < 6; solution += 2)
            {
                bool thrown = false;
                for(uint64_t launch = 0; launch < perSolution && !thrown; launch++)
                    for(uint32_t d = 0; d < 4 && !thrown; d++)
                        for(int site = 0; site < 6 && !thrown; site++)
                            thrown = viewThrowNow(t, (ViewThrowSite)site, d, launch);
                (thrown ? threw : ran).push_back(solution);
            }
            if(at < perSolution)
            {
                EXPECT_EQ(threw, (std::vector<uint32_t>{1}));
                EXPECT_EQ(ran, (std::vector<uint32_t>{3, 5}));
            }
            else
                EXPECT_TRUE(threw.empty());
            EXPECT_EQ(t.set, at >= perSolution);
        }
    ViewThrowAt t = viewParseThrowAt("wait:2:7");
    EXPECT_FALSE(viewThrowNow(t, ViewThrowSite::Stop, 2, 7));
    EXPECT_FALSE(viewThrowNow(t, ViewThrowSite::Wait, 1, 7));
    EXPECT_FALSE(viewThrowNow(t, ViewThrowSite::Wait, 2, 6));
    EXPECT_TRUE(t.set);
    EXPECT_TRUE(viewThrowNow(t, ViewThrowSite::Wait, 2, 7));
    EXPECT_FALSE(viewThrowNow(t, ViewThrowSite::Wait, 2, 7));
    ViewThrowAt none;
    EXPECT_FALSE(viewThrowNow(none, ViewThrowSite::Start, 0, 0));
}

namespace
{
    using ArmEvent = std::tuple<char, uint64_t, uint32_t>;

    // One FakeQueue per queue of a, each with room for its chain at `epoch`.
    std::vector<std::unique_ptr<FakeQueue>> attachEvery(ViewArmArgs& a, uint32_t epoch)
    {
        std::vector<std::unique_ptr<FakeQueue>> out;
        for(uint32_t q = 0; q < a.numQueues; q++)
        {
            ViewRingWriter count{nullptr, 1ull << 40, 0};
            viewBuildChain(a, q, epoch, count);
            size_t dwords = 16;
            while(dwords <= count.pos)
                dwords *= 2;
            out.push_back(std::make_unique<FakeQueue>(dwords, 400));
            out.back()->attach(a, q);
        }
        return out;
    }
}

// viewArmHostQueues makes the writes and adds of viewArmThread over every queue, with the same
// error bits and rings; it publishes every entry before any other write and rings no doorbell
// before its last write. Cases: readinessCases x debug {0, 32, 64} x error {0, TIMEOUT} x queue 0
// {fits, ring full, stuck}.
TEST(ViewTransport, HostArmBooksAsTheDeviceArm)
{
    const uint32_t epoch    = 6;
    uint64_t       unitFree = 0, pulledAtomics = 0;
    for(auto const& base : readinessCases())
        for(int debug : {0, VIEW_DEBUG_NO_READY, VIEW_DEBUG_NO_EXITWAIT})
            for(uint32_t error : {0u, VIEW_ERROR_TIMEOUT})
                for(int variant = 0; variant < 3; variant++)
                {
                    SCOPED_TRACE(testing::Message()
                                 << "mode " << base.mode << " W " << base.world << " rank "
                                 << base.rank << " B " << base.tokenBlocks << " F "
                                 << base.featureTiles << " debug " << debug << " error " << error
                                 << " variant " << variant);
                    ViewArmArgs host = withTail(base), dev = withTail(base);
                    host.debug = dev.debug = debug;
                    auto hostQs            = attachEvery(host, epoch);
                    auto devQs             = attachEvery(dev, epoch);
                    if(variant == 1)
                        host.queues[0].ringBytes = dev.queues[0].ringBytes = 4;
                    if(variant == 2)
                        hostQs[0]->rptr = devQs[0]->rptr = 300;

                    std::vector<ArmEvent> hostEv, devEv;
                    auto                  rung = [&] {
                        for(auto const& f : hostQs)
                            if(f->doorbell != 0)
                                return true;
                        return false;
                    };
                    auto hostAdd = [&](uint32_t* p, uint32_t v) {
                        EXPECT_FALSE(rung());
                        hostEv.push_back({'a', viewAddress(p), v});
                    };
                    auto hostStore = [&](uint32_t* p, uint32_t v) {
                        EXPECT_FALSE(rung());
                        hostEv.push_back({'w', viewAddress(p), v});
                    };
                    const uint32_t hostBits
                        = viewArmHostQueues(host, epoch, error, hostAdd, hostStore);

                    auto devAdd = [&](uint32_t* p, uint32_t v) {
                        devEv.push_back({'a', viewAddress(p), v});
                    };
                    auto devWrite = [&](uint32_t* p, uint32_t v) {
                        devEv.push_back({'w', viewAddress(p), v});
                    };
                    uint32_t devBits = 0;
                    for(uint32_t q = 0; q < dev.numQueues; q++)
                        devBits |= viewArmThread(
                            dev, q, epoch, error, false, devAdd, devWrite, devWrite);

                    EXPECT_EQ(hostBits, devBits);
                    std::set<uint64_t> entries;
                    for(uint32_t q = 0; q < host.numQueues; q++)
                    {
                        if(host.queues[q].entryOut != nullptr
                           && (debug & VIEW_DEBUG_NO_READY) == 0)
                            entries.insert(viewAddress(host.queues[q].entryOut));
                        EXPECT_EQ(hostQs[q]->wptr, devQs[q]->wptr) << "q " << q;
                        EXPECT_EQ(hostQs[q]->doorbell, devQs[q]->doorbell) << "q " << q;
                        EXPECT_TRUE(hostQs[q]->ring == devQs[q]->ring) << "q " << q;
                        if(error == 0 && variant == 0 && viewPulled(host, q) != nullptr)
                            (viewChainHasUnit(host, q) ? pulledAtomics : unitFree)++;
                    }
                    for(size_t i = 0; i < hostEv.size(); i++)
                    {
                        const bool publish = std::get<0>(hostEv[i]) == 'w'
                                             && entries.count(std::get<1>(hostEv[i])) != 0;
                        EXPECT_EQ(publish, i < entries.size()) << "event " << i;
                    }
                    std::sort(hostEv.begin(), hostEv.end());
                    std::sort(devEv.begin(), devEv.end());
                    EXPECT_EQ(hostEv, devEv);
                }
    EXPECT_GT(unitFree, 0u);
    EXPECT_GT(pulledAtomics, 0u);

    // A2AGemm W3 rank 1, debug 0: without units every queue books pulled at the arm; with units
    // none does.
    for(uint32_t blocks : {0u, 2u})
    {
        ViewArmArgs a = chainArgs(3, blocks, 1, 8, 1, 1);
        a.rank        = 1;
        a             = withTail(a);
        auto qs       = attachEvery(a, 0);
        Adds adds, stores;
        EXPECT_EQ(viewArmHostQueues(a, 0, 0, recordTo(adds), recordTo(stores)), 0u);
        Adds want;
        if(blocks == 0)
            want = {{viewAddress(a.queues[0].pulled), 1u}, {viewAddress(a.queues[1].pulled), 1u}};
        EXPECT_EQ(adds, want) << "blocks " << blocks;
        EXPECT_EQ(stores,
                  (Adds{{viewAddress(a.queues[0].entryOut), 1u},
                        {viewAddress(a.queues[1].entryOut), 1u},
                        {kFlag, 0u},
                        {kFlag + 4, 0u}}));
        EXPECT_NE(qs[0]->doorbell, 0u);
        EXPECT_EQ(qs[0]->wptr != 400, blocks != 0);
    }
}

// viewWireTails sets the words withTail sets, at W 2, 3 and 8, both modes, every rank;
// viewTailOf adds the tail offset.
TEST(ViewTransport, WireTailsPointEveryQueueAtTheTails)
{
    EXPECT_EQ(viewAddress(viewTailOf(fakePointer<void>(kTail), 0x60)), kTail + 0x60);
    for(uint32_t w : {2u, 3u, 8u})
        for(bool gemmA2A : {false, true})
            for(uint32_t r = 0; r < w; r++)
            {
                SCOPED_TRACE(testing::Message()
                             << "W " << w << " GemmA2A " << gemmA2A << " rank " << r);
                ViewArmArgs a = gemmA2A ? gemmA2AArgs() : chainArgs(w, 2, 1, 8, 1, 1);
                a.world       = w;
                a.numQueues   = gemmA2A ? w : w - 1;
                a.rank        = r;
                uint32_t* tails[FUSED_A2A_MAX_RANKS];
                for(uint32_t x = 0; x < w; x++)
                    tails[x] = fakePointer<uint32_t>(tailAt(x));
                ViewArmArgs wired = a;
                viewWireTails(wired, tails);
                ViewArmArgs const want = withTail(a);
                EXPECT_EQ(wired.tail, want.tail);
                for(uint32_t q = 0; q < a.numQueues; q++)
                {
                    EXPECT_EQ(wired.queues[q].ready, want.queues[q].ready) << "q " << q;
                    EXPECT_EQ(wired.queues[q].entryOut, want.queues[q].entryOut) << "q " << q;
                    EXPECT_EQ(wired.queues[q].pulled, want.queues[q].pulled) << "q " << q;
                }
            }
    ViewArmArgs a = chainArgs(3, 2, 1, 8, 1, 1);
    a.rank        = 1;
    uint32_t* tails[3];
    for(uint32_t x = 0; x < 3; x++)
        tails[x] = fakePointer<uint32_t>(tailAt(x));
    viewWireTails(a, tails);
    EXPECT_EQ(viewAddress(a.tail), tailAt(1));
    EXPECT_EQ(viewAddress(a.queues[0].ready), tailAt(1) + VIEW_TAIL_ENTRY_OFFSET + 8);
    EXPECT_EQ(viewAddress(a.queues[0].entryOut), tailAt(2) + VIEW_TAIL_ENTRY_OFFSET + 4);
    EXPECT_EQ(viewAddress(a.queues[0].pulled), tailAt(2) + VIEW_TAIL_PULLED_OFFSET + 4);
    EXPECT_EQ(viewAddress(a.queues[1].pulled), tailAt(0) + VIEW_TAIL_PULLED_OFFSET + 4);
}

// viewTailChecks: entry unless NO_READY, pulled unless NO_EXITWAIT. viewCheckTails compares each
// rank with its own header epoch, reports the failing ranks, and checks nothing when an error
// word is set.
TEST(ViewTransport, ValidatorChecksEveryRankAgainstItsOwnEpoch)
{
    for(uint32_t debug = 0; debug <= VIEW_DEBUG_ALL; debug++)
    {
        const ViewTailChecks c = viewTailChecks(debug);
        EXPECT_EQ(c.entry, (debug & VIEW_DEBUG_NO_READY) == 0) << debug;
        EXPECT_EQ(c.pulled, (debug & VIEW_DEBUG_NO_EXITWAIT) == 0) << debug;
    }
    constexpr uint32_t W = 4, T = VIEW_TAIL_BYTES / 4;
    constexpr uint32_t E = VIEW_TAIL_ENTRY_OFFSET / 4, P = VIEW_TAIL_PULLED_OFFSET / 4;
    using Bad            = std::vector<uint32_t>;
    for(uint32_t mode : {VIEW_MODE_A2A_GEMM, VIEW_MODE_GEMM_A2A})
    {
        SCOPED_TRACE(testing::Message() << "mode " << mode);
        const bool            a2aGemm = mode == VIEW_MODE_A2A_GEMM;
        std::vector<uint32_t> headers(W * kViewHeaderWords, 0), tails(W * T, 0);
        for(uint32_t d = 0; d < W; d++)
        {
            headers[d * kViewHeaderWords + FUSED_A2A_VIEW_EPOCH_OFFSET / 4] = 10 + 3 * d;
            for(uint32_t x = 0; x < W; x++)
                tails[d * T + E + x] = tails[d * T + P + x] = x == d ? 0 : 10 + 3 * d;
        }
        auto run = [&](ViewTailChecks c, bool checked) {
            Bad bad;
            EXPECT_EQ(viewCheckTails(mode, W, headers.data(), tails.data(), c, bad), checked);
            return bad;
        };
        EXPECT_EQ(run({true, true}, true), Bad{});
        tails[2 * T + E + 0]--;
        EXPECT_EQ(run({true, true}, true), Bad{2});
        EXPECT_EQ(run({false, true}, true), Bad{});
        tails[2 * T + E + 0]++;
        tails[3 * T + P + 1]++;
        EXPECT_EQ(run({true, true}, true), a2aGemm ? Bad{3} : Bad{});
        EXPECT_EQ(run({true, false}, true), Bad{});
        tails[0 * T + E + 1]--;
        EXPECT_EQ(run({true, true}, true), a2aGemm ? (Bad{0, 3}) : (Bad{0}));
        headers[1 * kViewHeaderWords + FUSED_A2A_VIEW_ERROR_OFFSET / 4] = VIEW_ERROR_TIMEOUT;
        EXPECT_EQ(run({true, true}, false), Bad{});
    }
}
