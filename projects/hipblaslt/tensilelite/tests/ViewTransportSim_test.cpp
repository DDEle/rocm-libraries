// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host model of FusedA2ATransport=View readiness across launches: W ranks with their streams
// (producer, arm, GEMM, wait, consumer) and SDMA queues that execute the ring bytes the client's
// arm code writes, under random, biased and exhaustive interleavings.

#include <gtest/gtest.h>

#include "ViewTransport.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    // Every view block, ring and queue pointer of the live Sim, at a fixed address: a copy of
    // a Sim plus a copy of the used bytes is a complete state.
    struct Arena
    {
        static constexpr size_t kBytes = 32u << 20;
        uint8_t*                base   = nullptr;
        size_t                  used   = 0;

        Arena()
        {
            base = static_cast<uint8_t*>(std::aligned_alloc(4096, kBytes));
        }

        template <typename T>
        T* take(size_t n)
        {
            used = (used + 63) & ~size_t(63);
            T* p = reinterpret_cast<T*>(base + used);
            used += n * sizeof(T);
            if(used > kBytes)
                throw std::runtime_error("sim arena exhausted");
            std::memset(static_cast<void*>(p), 0, n * sizeof(T));
            return p;
        }
    };

    Arena& arena()
    {
        static Arena a;
        return a;
    }

    // COPY operand addresses: 1 << 47 | kind << 44 | rank << 40 | slot << 36 | seg << 28,
    // plus block * kBlockStride + h * kSubStride.
    constexpr uint64_t kKindB       = 1; // A2AGemm B
    constexpr uint64_t kKindBuf     = 2; // A2AGemm a2aBuf
    constexpr uint64_t kKindD       = 3; // GemmA2A D
    constexpr uint64_t kKindRecv    = 4; // GemmA2A a2aBuf
    constexpr uint64_t kKindScratch = 5;
    constexpr uint64_t kBlockStride = 1ull << 12;
    constexpr uint64_t kSubStride   = 1ull << 4;

    uint64_t operand(uint64_t kind, uint64_t rank, uint64_t slot, uint64_t seg)
    {
        return 1ull << 47 | kind << 44 | rank << 40 | slot << 36 | seg << 28;
    }

    struct Operand
    {
        uint32_t kind, rank, slot, seg, block, h;
    };

    Operand decode(uint64_t a)
    {
        return {(uint32_t)(a >> 44 & 7),
                (uint32_t)(a >> 40 & 15),
                (uint32_t)(a >> 36 & 15),
                (uint32_t)(a >> 28 & 255),
                (uint32_t)((a & ((1ull << 28) - 1)) / kBlockStride),
                (uint32_t)((a & (kBlockStride - 1)) / kSubStride)};
    }

    constexpr int64_t kPoison = -2;
    constexpr int64_t kTorn   = -3;

    enum class Mutant
    {
        None,
        NoHeadA2AGemm, // A2AGemm chains without the head POLL
        NoHeadGemmA2A, // GemmA2A chains without the head POLL
        HeadRefE, // head POLL reference e instead of e + 1
        NoExitWait, // no A2AGemm exit wait in the stream
        ExitEpochMinus1, // exit wait against epoch - 1
        PulledEarly, // pulled ATOMIC right after the head
        PulledLast, // pulled ATOMIC after the last flag ATOMIC
        PublishBeforeProducer, // entry published when the producer starts
        NoPulledAtomic, // chains without the pulled ATOMIC, exit wait kept
        BookWithoutPulled, // bookings skip the pulled word
        PublishOnlyUnhalted, // a halted arm publishes nothing
        NoUnarmedAbort, // a throw before the arm books nothing
        TryArmLaunchOnly, // step 1's try: arm and GEMM launch only, GemmA2A abort only
        NoHostRelease, // the error path skips the host release
        NoTimeoutRelease, // the GemmA2A timeout path releases no entry
        // The host release stores VIEW_TAIL_RELEASED into every tail word and the exit wait
        // ignores the released word.
        ReleaseOverwritesPulled,
        ExitIgnoresRelease, // the exit wait ignores the released word
        NoArmedAbortA2AGemm, // an armed A2AGemm abort sets no error bit
    };

    const char* mutantName(Mutant m)
    {
        static const char* names[] = {"none",
                                      "no-head-a2agemm",
                                      "no-head-gemma2a",
                                      "head-ref-e",
                                      "no-exit-wait",
                                      "exit-epoch-minus-1",
                                      "pulled-early",
                                      "pulled-last",
                                      "publish-before-producer",
                                      "no-pulled-atomic",
                                      "book-without-pulled",
                                      "publish-only-unhalted",
                                      "no-unarmed-abort",
                                      "try-arm-launch-only",
                                      "no-host-release",
                                      "no-timeout-release",
                                      "release-overwrites-pulled",
                                      "exit-ignores-release",
                                      "no-armed-abort-a2agemm"};
        return names[(int)m];
    }

    struct Scenario
    {
        uint32_t mode         = VIEW_MODE_A2A_GEMM;
        uint32_t world        = 2;
        uint32_t blocks       = 1; // A2AGemm B, GemmA2A T
        uint32_t features     = 1;
        uint32_t numCu        = 8;
        uint32_t subRounds    = 1;
        uint32_t coalesce     = 1;
        uint32_t peerGroups   = 0;
        uint32_t tilesPerRank = 3;
        uint32_t launches     = 3;
        uint32_t slots        = 1;
        uint32_t ringDwords   = 1u << 12;
        int      debug        = 0;
        uint32_t delayRank    = 1;
        uint32_t delayCopies  = 2;
        uint32_t epoch0       = 0;
        bool     barrier      = false; // I1's host barrier around the producers
        bool     produceB     = true; // A2AGemm: the producer rewrites B
        Mutant   mutant       = Mutant::None;
        int      ringFullRank = -1; // that rank's queue ringFullQueue does not fit at ringFullLaunch
        uint32_t ringFullQueue = 0, ringFullLaunch = 0;
        int      spuriousRank = -1; // that rank's wait of spuriousLaunch times out at once
        uint32_t spuriousLaunch = 0;
        int      throwSite      = -1; // (int)ViewThrowSite
        uint32_t throwDevice = 0, throwLaunch = 0;
        int      deadRank   = -1; // stream and queues stop at deadLaunch
        uint32_t deadLaunch = 0;
        bool     deadThrows = true; // the host throws at the dead rank's first site
        bool     hostArm    = false; // arms through viewArmHostQueues; book-only stays viewArmThread

        bool gemmA2A() const
        {
            return mode == VIEW_MODE_GEMM_A2A;
        }

        bool errorFree() const
        {
            return ringFullRank < 0 && spuriousRank < 0 && throwSite < 0 && deadRank < 0;
        }

        std::string name() const
        {
            std::ostringstream o;
            o << (gemmA2A() ? "GemmA2A" : "A2AGemm") << " W" << world << " B" << blocks << " F"
              << features << " S" << subRounds << " c" << coalesce << " k" << peerGroups << " L"
              << launches << " slots" << slots << " ring" << ringDwords << " debug" << debug
              << " e0 " << epoch0 << (barrier ? " barrier" : "") << (produceB ? "" : " static-B")
              << " mutant " << mutantName(mutant) << (hostArm ? " host-arm" : "");
            if(ringFullRank >= 0)
                o << " ring-full " << ringFullRank << ":" << ringFullQueue << "@" << ringFullLaunch;
            if(spuriousRank >= 0)
                o << " spurious-timeout " << spuriousRank << "@" << spuriousLaunch;
            if(throwSite >= 0)
                o << " throw " << throwSite << ":" << throwDevice << "@" << throwLaunch;
            if(deadRank >= 0)
                o << " dead " << deadRank << "@" << deadLaunch << (deadThrows ? " throws" : "");
            return o.str();
        }
    };

    struct Outcome
    {
        uint64_t    ra = 0, rb = 0, rc = 0, late = 0, skew = 0, mismatches = 0;
        bool        complete = false, drained = false, countsOk = true;
        uint32_t    timeouts = 0, errors = 0;
        uint64_t    steps    = 0;
        std::string first;

        uint64_t safety() const
        {
            return ra + rb + rc + late + skew;
        }
    };

    enum class OpKind : uint8_t
    {
        ProduceStart,
        ProduceEnd,
        Arm,
        BookOnly,
        Gemm,
        Wait,
        AbortWait,
        Consume,
        Die
    };

    struct Op
    {
        OpKind   kind;
        uint32_t launch;
    };

    enum class ActKind : uint8_t
    {
        Stream,
        Sdma,
        HostRelease
    };

    struct Action
    {
        ActKind  kind;
        uint32_t rank;
        uint32_t queue;
    };

    class Sim
    {
    public:
        explicit Sim(Scenario const& s)
            : sc(s)
            , W(s.world)
            , nq(s.gemmA2A() ? s.world : s.world - 1)
        {
            arena().used = 0;
            tailOffset   = fusedA2AViewBlockBytes(sc.gemmA2A() ? sc.blocks : 0);
            blockWords   = (tailOffset + VIEW_TAIL_BYTES) / sizeof(uint32_t);
            for(uint32_t r = 0; r < W; r++)
            {
                block[r] = arena().take<uint32_t>(blockWords);
                for(uint32_t q = 0; q < nq; q++)
                {
                    ring[r][q] = arena().take<uint32_t>(sc.ringDwords);
                    qptr[r][q] = arena().take<uint64_t>(3);
                }
            }
            initEpoch();
            const uint32_t B = sc.blocks, S = sc.gemmA2A() ? 1 : sc.subRounds;
            bTag.assign(W * sc.slots, -1);
            bProducing.assign(W * sc.slots, -1);
            bufTag.assign(W * sc.slots * W * B * S, -1);
            dTag.assign(W * sc.slots * W * B, -1);
            recvTag.assign(W * sc.slots * W * B, -1);
            producedEnd.assign(W, -1);
            consumed.assign(W, -1);
            pc.assign(W, 0);
            phase.assign(W, 0);
            tile.assign(W, 0);
            armed.assign(W, 0);
            dead.assign(W, 0);
            segs.assign(W * nq, std::deque<Seg>());
            buildOps();
        }

        Scenario const& scenario() const
        {
            return sc;
        }

        std::vector<Action> enabled() const
        {
            std::vector<Action> out;
            for(uint32_t r = 0; r < W; r++)
            {
                if(dead[r])
                    continue;
                if(streamEnabled(r))
                    out.push_back({ActKind::Stream, r, 0});
                for(uint32_t q = 0; q < nq; q++)
                    if(sdmaEnabled(r, q))
                        out.push_back({ActKind::Sdma, r, q});
            }
            if(releasePending)
                out.push_back({ActKind::HostRelease, 0, 0});
            return out;
        }

        // Ranks whose current op is a wait that is blocked and can time out.
        std::vector<uint32_t> timeoutable() const
        {
            std::vector<uint32_t> out;
            for(uint32_t r = 0; r < W; r++)
            {
                if(dead[r] || pc[r] >= ops[r].size() || streamEnabled(r))
                    continue;
                const Op op = ops[r][pc[r]];
                if((op.kind == OpKind::Wait && phase[r] != 0) || op.kind == OpKind::AbortWait)
                    out.push_back(r);
            }
            return out;
        }

        void apply(Action const& a)
        {
            out.steps++;
            if(a.kind == ActKind::Stream)
                stream(a.rank);
            else if(a.kind == ActKind::Sdma)
                sdma(a.rank, a.queue);
            else
                hostRelease();
        }

        void fireTimeout(uint32_t r)
        {
            out.timeouts++;
            ViewArmArgs    a     = args(r, ops[r][pc[r]].launch);
            const uint32_t epoch = header(r, FUSED_A2A_VIEW_EPOCH_OFFSET);
            if(phase[r] == 2)
            {
                errorWord(r) |= VIEW_ERROR_STUCK;
                done(r);
                return;
            }
            errorWord(r) |= VIEW_ERROR_TIMEOUT;
            if(!sc.gemmA2A())
            {
                done(r);
                return;
            }
            viewReleaseSig(a, epoch);
            if(sc.mutant != Mutant::NoTimeoutRelease)
                viewReleaseEntry(a, epoch, viewMax32);
            phase[r] = 2;
        }

        // After a throw: the host release of every live rank's tail becomes due.
        void scheduleHostRelease()
        {
            if(sc.mutant != Mutant::NoHostRelease)
                releasePending = true;
        }

        // The host release becomes due when the dead rank dies.
        bool releaseOnDeath = false;

        bool finished() const
        {
            for(uint32_t r = 0; r < W; r++)
                if(!dead[r] && pc[r] < ops[r].size())
                    return false;
            return true;
        }

        bool drained() const
        {
            for(uint32_t r = 0; r < W; r++)
                for(uint32_t q = 0; !dead[r] && q < nq; q++)
                    if(qptr[r][q][0] != qptr[r][q][1])
                        return false;
            return true;
        }

        Outcome result() const
        {
            Outcome o  = out;
            o.complete = finished();
            o.drained  = drained();
            for(uint32_t r = 0; r < W; r++)
                o.errors |= header(r, FUSED_A2A_VIEW_ERROR_OFFSET);
            if(o.complete && sc.errorFree())
                for(uint32_t r = 0; r < W; r++)
                {
                    const uint32_t epoch = header(r, FUSED_A2A_VIEW_EPOCH_OFFSET);
                    o.countsOk           = o.countsOk && epoch == sc.epoch0 + sc.launches
                                 && viewTailCountsMatch(sc.mode,
                                                        r,
                                                        W,
                                                        epoch,
                                                        tail(r),
                                                        (sc.debug & VIEW_DEBUG_NO_READY) == 0,
                                                        (sc.debug & VIEW_DEBUG_NO_EXITWAIT) == 0);
                }
            return o;
        }

        uint64_t hash() const
        {
            uint64_t h   = 0xcbf29ce484222325ull;
            auto     mix = [&](void const* p, size_t n) {
                auto const* b = static_cast<uint8_t const*>(p);
                for(size_t i = 0; i < n; i++)
                {
                    h ^= b[i];
                    h *= 0x100000001b3ull;
                }
            };
            auto vec = [&](auto const& v) { mix(v.data(), v.size() * sizeof(v[0])); };
            mix(arena().base, arena().used);
            vec(pc);
            vec(phase);
            vec(tile);
            vec(armed);
            vec(dead);
            vec(bTag);
            vec(bProducing);
            vec(bufTag);
            vec(dTag);
            vec(recvTag);
            vec(producedEnd);
            vec(consumed);
            mix(&releasePending, sizeof(releasePending));
            return h;
        }

        // Runs to quiescence with `pick` choosing among the enabled actions; with timeouts,
        // fires one blocked wait's timeout whenever nothing else is enabled.
        template <typename Pick>
        void run(Pick&& pick, bool timeouts, uint64_t budget = 2000000)
        {
            while(out.steps < budget)
            {
                auto acts = enabled();
                if(!acts.empty())
                {
                    apply(acts[pick(acts)]);
                    continue;
                }
                if(!timeouts)
                    return;
                auto blocked = timeoutable();
                if(blocked.empty())
                    return;
                fireTimeout(blocked[0]);
            }
        }

    private:
        struct Seg
        {
            uint64_t lo, hi;
            uint32_t epoch;
        };

        Scenario                     sc;
        uint32_t                     W, nq;
        size_t                       tailOffset = 0, blockWords = 0;
        uint32_t*                    block[FUSED_A2A_MAX_RANKS]                      = {};
        uint32_t*                    ring[FUSED_A2A_MAX_RANKS][FUSED_A2A_MAX_RANKS] = {};
        uint64_t*                    qptr[FUSED_A2A_MAX_RANKS][FUSED_A2A_MAX_RANKS] = {};
        std::vector<std::vector<Op>> ops;
        std::vector<uint32_t>        pc, tile, armed;
        std::vector<uint8_t>         phase, dead;
        std::vector<std::deque<Seg>> segs;
        std::vector<int64_t>         bTag, bProducing, bufTag, dTag, recvTag, producedEnd, consumed;
        bool                         releasePending = false;
        Outcome                      out;

        uint32_t header(uint32_t r, size_t offset) const
        {
            return block[r][offset / sizeof(uint32_t)];
        }

        uint32_t& errorWord(uint32_t r)
        {
            return block[r][FUSED_A2A_VIEW_ERROR_OFFSET / sizeof(uint32_t)];
        }

        uint32_t* tail(uint32_t r) const
        {
            return block[r] + tailOffset / sizeof(uint32_t);
        }

        uint32_t slotOf(uint32_t launch) const
        {
            return launch % sc.slots;
        }

        void violation(uint64_t& counter, std::string const& what)
        {
            counter++;
            if(out.first.empty())
                out.first = what;
        }

        void initEpoch()
        {
            const uint32_t e0 = sc.epoch0;
            for(uint32_t r = 0; r < W; r++)
            {
                uint32_t* v = block[r];
                v[FUSED_A2A_VIEW_EPOCH_OFFSET / 4] = e0;
                for(uint32_t x = 0; x < W; x++)
                {
                    tail(r)[VIEW_TAIL_ENTRY_OFFSET / 4 + x]  = e0;
                    tail(r)[VIEW_TAIL_PULLED_OFFSET / 4 + x] = e0;
                }
                if(!sc.gemmA2A())
                    continue;
                for(uint32_t s = 0; s < W; s++)
                {
                    v[FUSED_A2A_VIEW_FLAG_OFFSET / 4 + s]     = e0 * sc.blocks;
                    v[FUSED_A2A_VIEW_EXPECTED_OFFSET / 4 + s] = e0 * sc.blocks;
                }
                v[FUSED_A2A_VIEW_OUTBOUND_OFFSET / 4] = e0 * W;
                for(uint32_t i = 0; i < W * sc.blocks; i++)
                    v[FUSED_A2A_VIEW_SIG_OFFSET / 4 + i] = e0 * sc.tilesPerRank;
            }
        }

        // The rank's ops of one launch without a throw.
        void fullLaunch(std::vector<Op>& o, uint32_t e) const
        {
            if(sc.gemmA2A() || sc.produceB)
            {
                o.push_back({OpKind::ProduceStart, e});
                o.push_back({OpKind::ProduceEnd, e});
            }
            o.push_back({OpKind::Arm, e});
            o.push_back({OpKind::Gemm, e});
            if(sc.gemmA2A() || sc.mutant != Mutant::NoExitWait)
                o.push_back({OpKind::Wait, e});
            if(sc.gemmA2A())
                o.push_back({OpKind::Consume, e});
        }

        void buildOps()
        {
            ops.assign(W, std::vector<Op>());
            const bool dies   = sc.deadRank >= 0;
            const bool throws = sc.throwSite >= 0 || (dies && sc.deadThrows);
            const uint32_t td = sc.throwSite >= 0 ? sc.throwDevice : (uint32_t)sc.deadRank;
            const uint32_t tl = sc.throwSite >= 0 ? sc.throwLaunch : sc.deadLaunch;
            const ViewThrowSite site
                = sc.throwSite >= 0 ? (ViewThrowSite)sc.throwSite : ViewThrowSite::Start;
            for(uint32_t r = 0; r < W; r++)
                for(uint32_t e = 0; e < sc.launches; e++)
                {
                    auto& o = ops[r];
                    if(dies && r == (uint32_t)sc.deadRank && e == sc.deadLaunch)
                    {
                        o.push_back({OpKind::Die, e});
                        break;
                    }
                    if(!throws || e < tl || (e == tl && r < td))
                    {
                        fullLaunch(o, e);
                        continue;
                    }
                    if(e > tl)
                        break;
                    std::vector<Op> body;
                    fullLaunch(body, e);
                    const bool produce = sc.gemmA2A() || sc.produceB;
                    if(produce)
                        o.insert(o.end(), body.begin(), body.begin() + 2);
                    const bool old = sc.mutant == Mutant::TryArmLaunchOnly;
                    if(r > td)
                    {
                        if(!old && sc.mutant != Mutant::NoUnarmedAbort)
                            o.push_back({OpKind::BookOnly, e});
                        break;
                    }
                    const size_t at = produce ? 2 : 0;
                    // body[at] = Arm, body[at + 1] = Gemm, then Wait and Consume.
                    std::vector<Op> rest(body.begin() + at, body.end());
                    auto            has = [&](OpKind k) {
                        return std::any_of(
                            rest.begin(), rest.end(), [&](Op const& x) { return x.kind == k; });
                    };
                    bool armedHere = false;
                    switch(site)
                    {
                    case ViewThrowSite::Start:
                    case ViewThrowSite::Arm:
                        break;
                    case ViewThrowSite::Launch:
                        o.push_back(rest[0]);
                        armedHere = true;
                        break;
                    case ViewThrowSite::Wait:
                        o.push_back(rest[0]);
                        o.push_back(rest[1]);
                        armedHere = true;
                        break;
                    case ViewThrowSite::Stop:
                    case ViewThrowSite::Check:
                        o.push_back(rest[0]);
                        o.push_back(rest[1]);
                        if(has(OpKind::Wait))
                            o.push_back({OpKind::Wait, e});
                        armedHere = true;
                        break;
                    }
                    if(old)
                    {
                        const bool inTry
                            = site == ViewThrowSite::Arm || site == ViewThrowSite::Launch;
                        if(inTry && sc.gemmA2A() && site == ViewThrowSite::Launch)
                            o.push_back({OpKind::AbortWait, e});
                        break;
                    }
                    if(armedHere)
                    {
                        if(sc.gemmA2A() || sc.mutant != Mutant::NoArmedAbortA2AGemm)
                            o.push_back({OpKind::AbortWait, e});
                    }
                    else if(sc.mutant != Mutant::NoUnarmedAbort)
                        o.push_back({OpKind::BookOnly, e});
                    break;
                }
        }

        ViewArmArgs args(uint32_t r, uint32_t launch) const
        {
            const uint32_t k = slotOf(launch);
            ViewArmArgs    a{};
            a.mode           = sc.mode;
            a.rank           = r;
            a.world          = W;
            a.numQueues      = nq;
            a.subRounds      = sc.gemmA2A() ? 1 : sc.subRounds;
            a.peerGroups     = sc.peerGroups;
            a.tokenBlocks    = sc.blocks;
            a.featureTiles   = sc.features;
            a.numCu          = sc.numCu;
            a.mt1            = 1;
            a.nTokens        = sc.blocks;
            a.srcPitch       = 1;
            a.dstPitch       = 1;
            a.slice          = 1;
            a.rectX          = 1;
            a.srcBlockStride = kBlockStride;
            a.dstBlockStride = kBlockStride;
            a.subRoundStride = sc.gemmA2A() ? 0 : kSubStride;
            a.coalesce       = sc.coalesce;
            a.tilesPerRank   = sc.tilesPerRank;
            a.view           = block[r];
            a.debug          = (uint32_t)sc.debug;
            a.tail           = tail(r);
            a.delayRank      = sc.delayRank;
            a.delayCopies    = sc.delayCopies;
            if((sc.debug & VIEW_DEBUG_DELAY_CHAIN) != 0)
                a.delayScratch = operand(kKindScratch, r, 0, 0);
            for(uint32_t q = 0; q < nq; q++)
            {
                ViewQueueArgs& qa = a.queues[q];
                qa.ring           = ring[r][q];
                qa.ringBytes      = sc.ringDwords * sizeof(uint32_t);
                qa.rptr           = &qptr[r][q][0];
                qa.wptr           = &qptr[r][q][1];
                qa.doorbell       = &qptr[r][q][2];
                const uint32_t peer = sc.gemmA2A() ? q : (r + 1 + q) % W;
                if(sc.gemmA2A())
                {
                    qa.srcBase = operand(kKindD, r, k, q);
                    qa.dstBase = operand(kKindRecv, q, k, r);
                    qa.flag    = block[q] + FUSED_A2A_VIEW_FLAG_OFFSET / 4 + r;
                }
                else
                {
                    qa.srcBase = operand(kKindB, peer, k, r);
                    qa.dstBase = operand(kKindBuf, r, k, q + 1);
                    qa.flag    = block[r] + FUSED_A2A_VIEW_FLAG_OFFSET / 4 + q;
                }
            }
            uint32_t* tails[FUSED_A2A_MAX_RANKS];
            for(uint32_t x = 0; x < W; x++)
                tails[x] = tail(x);
            viewWireTails(a, tails);
            for(uint32_t q = 0; q < nq; q++)
            {
                ViewQueueArgs& qa = a.queues[q];
                if((sc.gemmA2A() ? q : (r + 1 + q) % W) == r)
                    continue;
                if((sc.mutant == Mutant::NoHeadA2AGemm && !sc.gemmA2A())
                   || (sc.mutant == Mutant::NoHeadGemmA2A && sc.gemmA2A()))
                    qa.ready = nullptr;
                if(sc.mutant == Mutant::NoPulledAtomic)
                    qa.pulled = nullptr;
            }
            if(sc.ringFullRank == (int)r && launch == sc.ringFullLaunch)
                a.queues[sc.ringFullQueue].ringBytes = sizeof(uint32_t);
            return a;
        }

        bool isPulledWord(uint32_t* p) const
        {
            for(uint32_t x = 0; x < W; x++)
            {
                uint32_t* lo = tail(x) + VIEW_TAIL_PULLED_OFFSET / 4;
                if(p >= lo && p < lo + W)
                    return true;
            }
            return false;
        }

        uint32_t& word(uint32_t r, uint32_t q, uint64_t dwordPos) const
        {
            return ring[r][q][dwordPos & (sc.ringDwords - 1)];
        }

        static uint32_t packetDwords(uint32_t header)
        {
            switch(header)
            {
            case VIEW_COPY_HEADER:
                return VIEW_COPY_DWORDS;
            case VIEW_ATOMIC_HEADER:
                return VIEW_ATOMIC_DWORDS;
            case VIEW_POLL_HEADER:
                return VIEW_POLL_DWORDS;
            case VIEW_TIMESTAMP_HEADER:
                return VIEW_TIMESTAMP_DWORDS;
            case 0:
                return 1;
            }
            throw std::runtime_error("sim: unknown packet header " + std::to_string(header));
        }

        uint64_t addressAt(uint32_t r, uint32_t q, uint64_t pos) const
        {
            return (uint64_t)word(r, q, pos + 2) << 32 | word(r, q, pos + 1);
        }

        // Mutations of the chain the arm just wrote at dwords [lo, hi).
        void mutateChain(uint32_t r, uint32_t q, uint64_t lo, uint64_t hi, ViewArmArgs const& a)
        {
            struct P
            {
                std::vector<uint32_t> w;
            };
            std::vector<P> p;
            for(uint64_t at = lo; at < hi;)
            {
                const uint32_t n = packetDwords(word(r, q, at));
                if(word(r, q, at) == 0)
                    return;
                P x;
                for(uint32_t i = 0; i < n; i++)
                    x.w.push_back(word(r, q, at + i));
                p.push_back(x);
                at += n;
            }
            auto addr = [](P const& x) { return (uint64_t)x.w[2] << 32 | x.w[1]; };
            const bool head = !p.empty() && p[0].w[0] == VIEW_POLL_HEADER
                              && a.queues[q].ready != nullptr
                              && addr(p[0]) == viewAddress(a.queues[q].ready);
            if(sc.mutant == Mutant::HeadRefE && head)
                p[0].w[3] -= 1;
            if((sc.mutant == Mutant::PulledEarly || sc.mutant == Mutant::PulledLast)
               && a.queues[q].pulled != nullptr)
            {
                for(size_t i = 0; i < p.size(); i++)
                    if(p[i].w[0] == VIEW_ATOMIC_HEADER && addr(p[i]) == viewAddress(a.queues[q].pulled))
                    {
                        P x = p[i];
                        p.erase(p.begin() + i);
                        if(sc.mutant == Mutant::PulledEarly)
                            p.insert(p.begin() + (head ? 1 : 0), x);
                        else
                            p.push_back(x);
                        break;
                    }
            }
            uint64_t at = lo;
            for(auto const& x : p)
                for(uint32_t v : x.w)
                    word(r, q, at++) = v;
        }

        void arm(uint32_t r, uint32_t launch, bool bookOnly)
        {
            ViewArmArgs    a     = args(r, launch);
            const uint32_t epoch = header(r, FUSED_A2A_VIEW_EPOCH_OFFSET);
            const uint32_t error = header(r, FUSED_A2A_VIEW_ERROR_OFFSET);
            if(sc.errorFree() && (!sc.gemmA2A() || sc.blocks != 0))
                for(uint32_t x = 0; x < W; x++)
                    if(x != r && armed[x] < launch)
                        violation(out.skew,
                                  "R-skew: rank " + std::to_string(r) + " arms launch "
                                      + std::to_string(launch) + " before rank "
                                      + std::to_string(x) + " armed launch "
                                      + std::to_string(launch - 1));
            auto add = [&](uint32_t* p, uint32_t v) {
                if(sc.mutant == Mutant::BookWithoutPulled && isPulledWord(p))
                    return;
                *p += v;
            };
            auto store   = [](uint32_t* p, uint32_t v) { *p = v; };
            auto release = [&](uint32_t* p, uint32_t v) {
                if(sc.mutant == Mutant::PublishOnlyUnhalted && error != 0)
                    return;
                *p = v;
            };
            uint32_t bits = 0;
            uint64_t lo[FUSED_A2A_MAX_RANKS];
            for(uint32_t q = 0; q < nq; q++)
                lo[q] = qptr[r][q][1] / 4;
            if(sc.hostArm && !bookOnly)
                bits = viewArmHostQueues(a, epoch, error, add, store);
            else
                for(uint32_t q = 0; q < nq; q++)
                    bits |= viewArmThread(a, q, epoch, error, bookOnly, add, store, release);
            for(uint32_t q = 0; q < nq; q++)
            {
                const uint64_t hi = qptr[r][q][1] / 4;
                if(hi != lo[q])
                {
                    segs[r * nq + q].push_back({lo[q], hi, epoch});
                    mutateChain(r, q, lo[q], hi, a);
                }
            }
            errorWord(r) |= bits | (bookOnly ? VIEW_ERROR_ABORTED : 0u);
            viewAdvance(a, epoch, block[r]);
            armed[r]++;
        }

        bool flagsComplete(uint32_t r, uint32_t launch) const
        {
            ViewArmArgs const a     = args(r, launch);
            const uint32_t    units = viewA2AGemmChainUnits(a);
            for(uint32_t q = 0; q < nq; q++)
                if(block[r][FUSED_A2A_VIEW_FLAG_OFFSET / 4 + q] < units)
                    return false;
            return true;
        }

        bool waitDone(uint32_t r, uint32_t launch) const
        {
            ViewArmArgs const a     = args(r, launch);
            const uint32_t    epoch = header(r, FUSED_A2A_VIEW_EPOCH_OFFSET);
            if(sc.mutant == Mutant::ExitEpochMinus1 && !sc.gemmA2A())
                return viewExitDone(a, epoch - 1u);
            if((sc.mutant == Mutant::ExitIgnoresRelease
                || sc.mutant == Mutant::ReleaseOverwritesPulled)
               && !sc.gemmA2A())
            {
                std::vector<uint32_t> copy(a.tail, a.tail + VIEW_TAIL_BYTES / 4);
                copy[VIEW_TAIL_RELEASED_OFFSET / 4] = 0;
                ViewArmArgs b                       = a;
                b.tail                              = copy.data();
                return viewExitDone(b, epoch);
            }
            return viewWaitDone(a, epoch);
        }

        bool barrierAllows(Op const& op) const
        {
            if(!sc.barrier)
                return true;
            for(uint32_t x = 0; x < W; x++)
            {
                if(dead[x])
                    continue;
                if(op.kind == OpKind::ProduceStart && pc[x] < ops[x].size()
                   && ops[x][pc[x]].launch < op.launch)
                    return false;
                if(op.kind == OpKind::Arm && producedEnd[x] < (int64_t)op.launch
                   && !(pc[x] >= ops[x].size()))
                    return false;
            }
            return true;
        }

        bool streamEnabled(uint32_t r) const
        {
            if(pc[r] >= ops[r].size())
                return false;
            const Op op = ops[r][pc[r]];
            switch(op.kind)
            {
            case OpKind::ProduceStart:
            case OpKind::Arm:
                return barrierAllows(op);
            case OpKind::Gemm:
                return sc.gemmA2A() || flagsComplete(r, op.launch);
            case OpKind::Wait:
                if(phase[r] == 0)
                    return true;
                if(phase[r] == 1)
                    return waitDone(r, op.launch);
                return viewQueuesDrained(args(r, op.launch));
            case OpKind::AbortWait:
                return phase[r] == 0 || viewQueuesDrained(args(r, op.launch));
            default:
                return true;
            }
        }

        void done(uint32_t r)
        {
            phase[r] = 0;
            pc[r]++;
        }

        size_t bufIndex(uint32_t r, uint32_t k, uint32_t seg, uint32_t b, uint32_t h) const
        {
            const uint32_t S = sc.subRounds;
            return (((size_t)(r * sc.slots + k) * W + seg) * sc.blocks + b) * S + h;
        }

        size_t tileIndex(uint32_t s, uint32_t k, uint32_t p, uint32_t j) const
        {
            return ((size_t)(s * sc.slots + k) * W + p) * sc.blocks + j;
        }

        void stream(uint32_t r)
        {
            const Op       op = ops[r][pc[r]];
            const uint32_t e  = op.launch;
            const uint32_t k  = slotOf(e);
            switch(op.kind)
            {
            case OpKind::ProduceStart:
                if(!sc.gemmA2A())
                    bProducing[r * sc.slots + k] = e;
                if(sc.mutant == Mutant::PublishBeforeProducer)
                {
                    ViewArmArgs const a = args(r, e);
                    for(uint32_t q = 0; q < nq; q++)
                        viewPublishEntry(a,
                                         q,
                                         header(r, FUSED_A2A_VIEW_EPOCH_OFFSET),
                                         [](uint32_t* p, uint32_t v) { *p = v; });
                }
                done(r);
                return;
            case OpKind::ProduceEnd:
                bProducing[r * sc.slots + k] = -1;
                producedEnd[r]               = e;
                if(sc.gemmA2A())
                    for(uint32_t p = 0; p < W; p++)
                        for(uint32_t j = 0; j < sc.blocks; j++)
                        {
                            dTag[tileIndex(r, k, p, j)]    = kPoison;
                            recvTag[tileIndex(r, k, p, j)] = kPoison;
                        }
                else
                {
                    bTag[r * sc.slots + k] = e;
                    for(uint32_t seg = 1; seg < W; seg++)
                        for(uint32_t b = 0; b < sc.blocks; b++)
                            for(uint32_t h = 0; h < sc.subRounds; h++)
                                bufTag[bufIndex(r, k, seg, b, h)] = kPoison;
                }
                done(r);
                return;
            case OpKind::Arm:
            case OpKind::BookOnly:
                arm(r, e, op.kind == OpKind::BookOnly);
                done(r);
                return;
            case OpKind::Gemm:
                if(sc.gemmA2A())
                {
                    if(sc.blocks != 0)
                    {
                        const uint32_t p = tile[r] / sc.blocks, j = tile[r] % sc.blocks;
                        dTag[tileIndex(r, k, p, j)] = e;
                        block[r][FUSED_A2A_VIEW_SIG_OFFSET / 4 + p * sc.blocks + j]
                            += sc.tilesPerRank;
                    }
                    if(sc.blocks == 0 || ++tile[r] == W * sc.blocks)
                    {
                        tile[r] = 0;
                        done(r);
                    }
                    return;
                }
                if(viewA2AGemmChainUnits(args(r, e)) != 0)
                {
                    const int64_t want = sc.produceB ? (int64_t)e : -1;
                    for(uint32_t seg = 1; seg < W; seg++)
                        for(uint32_t b = 0; b < sc.blocks; b++)
                            for(uint32_t h = 0; h < sc.subRounds; h++)
                                if(bufTag[bufIndex(r, k, seg, b, h)] != want)
                                {
                                    out.mismatches++;
                                    if(out.first.empty())
                                        out.first = "A2AGemm GEMM of rank " + std::to_string(r)
                                                    + " launch " + std::to_string(e)
                                                    + " reads a stale or poisoned unit";
                                }
                }
                done(r);
                return;
            case OpKind::Wait:
                if(phase[r] == 0)
                {
                    ViewArmArgs const a = args(r, e);
                    if(!sc.gemmA2A() && !viewExitWaits(a))
                    {
                        done(r);
                        return;
                    }
                    if(header(r, FUSED_A2A_VIEW_ERROR_OFFSET) != 0)
                    {
                        done(r);
                        return;
                    }
                    phase[r] = 1;
                    if(sc.spuriousRank == (int)r && e == sc.spuriousLaunch)
                    {
                        fireTimeout(r);
                        out.timeouts--;
                        return;
                    }
                    if(waitDone(r, e))
                        done(r);
                    return;
                }
                done(r);
                return;
            case OpKind::AbortWait:
                if(phase[r] == 0)
                {
                    ViewArmArgs const a     = args(r, e);
                    const uint32_t    epoch = header(r, FUSED_A2A_VIEW_EPOCH_OFFSET);
                    errorWord(r) |= VIEW_ERROR_ABORTED;
                    if(!sc.gemmA2A())
                    {
                        done(r);
                        return;
                    }
                    viewReleaseSig(a, epoch);
                    viewReleaseEntry(a, epoch, viewMax32);
                    phase[r] = 2;
                    return;
                }
                done(r);
                return;
            case OpKind::Consume:
                for(uint32_t s = 0; s < W; s++)
                    for(uint32_t j = 0; j < sc.blocks; j++)
                        if(recvTag[tileIndex(r, k, s, j)] != (int64_t)e)
                        {
                            out.mismatches++;
                            if(out.first.empty())
                                out.first = "GemmA2A consumer of rank " + std::to_string(r)
                                            + " launch " + std::to_string(e)
                                            + " reads a stale or poisoned recv segment "
                                            + std::to_string(s);
                        }
                consumed[r] = e;
                done(r);
                return;
            case OpKind::Die:
                dead[r] = 1;
                if(releaseOnDeath)
                    scheduleHostRelease();
                done(r);
                return;
            }
        }

        bool sdmaEnabled(uint32_t r, uint32_t q) const
        {
            const uint64_t rp = qptr[r][q][0];
            if(rp == qptr[r][q][1])
                return false;
            const uint64_t at = rp / 4;
            if(word(r, q, at) != VIEW_POLL_HEADER)
                return true;
            auto const*    p   = reinterpret_cast<uint32_t const*>(addressAt(r, q, at));
            const uint32_t ref = word(r, q, at + 3);
            return (*p & word(r, q, at + 4)) >= ref;
        }

        uint32_t launchAt(uint32_t r, uint32_t q, uint64_t at)
        {
            auto& s = segs[r * nq + q];
            while(!s.empty() && at >= s.front().hi)
                s.pop_front();
            if(s.empty() || at < s.front().lo)
                throw std::runtime_error("sim: no arm wrote dword " + std::to_string(at));
            return s.front().epoch - sc.epoch0;
        }

        void copy(uint32_t r, uint32_t q, uint64_t at)
        {
            const uint32_t launch = launchAt(r, q, at);
            const Operand  src    = decode(addressAt(r, q, at));
            const Operand  dst    = decode((uint64_t)word(r, q, at + 7) << 32 | word(r, q, at + 6));
            if(src.kind == kKindScratch)
                return;
            const std::string where = " (rank " + std::to_string(r) + " queue " + std::to_string(q)
                                      + " launch " + std::to_string(launch) + ")";
            if(!sc.gemmA2A())
            {
                const size_t  b       = src.rank * sc.slots + src.slot;
                const int64_t making  = bProducing[b];
                const int64_t value   = making >= 0 ? kTorn : bTag[b];
                if(sc.produceB)
                {
                    if(making == (int64_t)launch || (making < 0 && bTag[b] < (int64_t)launch))
                        violation(out.ra, "R-a: COPY reads B of rank " + std::to_string(src.rank)
                                              + " before its producer" + where);
                    else if(making > (int64_t)launch || bTag[b] > (int64_t)launch)
                        violation(out.rb, "R-b: COPY reads B of rank " + std::to_string(src.rank)
                                              + " after its next producer started" + where);
                }
                bufTag[bufIndex(dst.rank, dst.slot, dst.seg, dst.block, dst.h)] = value;
                return;
            }
            const int64_t value = dTag[tileIndex(src.rank, src.slot, src.seg, src.block)];
            if(value != (int64_t)launch)
                violation(out.ra, "GemmA2A COPY reads D tile " + std::to_string(src.block)
                                      + " of another launch" + where);
            if(!dead[dst.rank])
            {
                if(producedEnd[dst.rank] < (int64_t)launch)
                    violation(out.rc, "R-c: COPY lands in recv of rank " + std::to_string(dst.rank)
                                          + " before its consumer of the previous launch and its "
                                            "producer finished"
                                          + where);
                if(consumed[dst.rank] >= (int64_t)launch)
                    violation(out.late, "COPY lands after the receiver's consumer" + where);
            }
            recvTag[tileIndex(dst.rank, dst.slot, dst.seg, dst.block)] = value;
        }

        void sdma(uint32_t r, uint32_t q)
        {
            const uint64_t at     = qptr[r][q][0] / 4;
            const uint32_t header = word(r, q, at);
            const uint32_t n      = packetDwords(header);
            if(header == VIEW_COPY_HEADER)
                copy(r, q, at);
            else if(header == VIEW_ATOMIC_HEADER)
                *reinterpret_cast<uint32_t*>(addressAt(r, q, at)) += word(r, q, at + 3);
            qptr[r][q][0] += n * sizeof(uint32_t);
        }

        void hostRelease()
        {
            releasePending = false;
            auto store     = [](uint32_t* p, uint32_t v) { *p = v; };
            auto none      = [](auto...) { return false; };
            for(uint32_t r = 0; r < W; r++)
                if(!dead[r])
                {
                    viewReleaseTail(tail(r), VIEW_RELEASE_HOST_STORE, store, none, none);
                    if(sc.mutant == Mutant::ReleaseOverwritesPulled)
                        for(uint32_t i = 0; i < VIEW_TAIL_BYTES / 4; i++)
                            tail(r)[i] = VIEW_TAIL_RELEASED;
                }
        }
    };

    enum class Policy
    {
        Uniform,
        StreamsFirst, // SDMA steps 1/20 as likely: ranks run up to the window ahead
        QueuesFirst,
        SlowRank0, // rank 0's stream and queues 1/20 as likely
        FastRank0, // rank 0's stream 20x as likely
        SlowQueue0, // rank 0 queue 0 1/20 as likely
    };

    constexpr Policy kPolicies[] = {Policy::Uniform,
                                    Policy::StreamsFirst,
                                    Policy::QueuesFirst,
                                    Policy::SlowRank0,
                                    Policy::FastRank0,
                                    Policy::SlowQueue0};

    double weight(Policy p, Action const& a)
    {
        const bool sdma = a.kind == ActKind::Sdma;
        switch(p)
        {
        case Policy::Uniform:
            return 1;
        case Policy::StreamsFirst:
            return sdma ? 0.05 : 1;
        case Policy::QueuesFirst:
            return sdma ? 1 : 0.05;
        case Policy::SlowRank0:
            return a.rank == 0 && a.kind != ActKind::HostRelease ? 0.05 : 1;
        case Policy::FastRank0:
            return a.rank == 0 && a.kind == ActKind::Stream ? 20 : 1;
        case Policy::SlowQueue0:
            return sdma && a.rank == 0 && a.queue == 0 ? 0.05 : 1;
        }
        return 1;
    }

    // One schedule: phase 1 with timeouts only when the scenario has an error, then for a
    // throw the host release (dead ranks) and timeouts. releaseAtThrow: for a throw, the host
    // release is due from the first step on, for a dead rank from its death on, and timeouts
    // may fire.
    Outcome runOnce(Scenario const& sc,
                    Policy          policy,
                    uint64_t        seed,
                    bool*           phase1Complete = nullptr,
                    bool            releaseAtThrow = false)
    {
        Sim          sim(sc);
        std::mt19937_64 rng(seed);
        auto         pick = [&](std::vector<Action> const& acts) -> size_t {
            double total = 0;
            for(auto const& a : acts)
                total += weight(policy, a);
            double x = std::uniform_real_distribution<double>(0, total)(rng);
            for(size_t i = 0; i < acts.size(); i++)
            {
                x -= weight(policy, acts[i]);
                if(x <= 0)
                    return i;
            }
            return acts.size() - 1;
        };
        const bool throws  = sc.throwSite >= 0 || (sc.deadRank >= 0 && sc.deadThrows);
        const bool errored = !sc.errorFree() && !throws;
        if(throws && releaseAtThrow)
        {
            if(sc.throwSite >= 0)
                sim.scheduleHostRelease();
            else
                sim.releaseOnDeath = true;
            sim.run(pick, true);
            if(phase1Complete != nullptr)
                *phase1Complete = true;
            return sim.result();
        }
        sim.run(pick, errored);
        if(phase1Complete != nullptr)
            *phase1Complete = sim.finished();
        if(throws && !sim.finished())
        {
            sim.scheduleHostRelease();
            sim.run(pick, true);
        }
        return sim.result();
    }

    struct DfsResult
    {
        uint64_t    states = 0, terminals = 0, bad = 0, badSafety = 0;
        bool        truncated = false;
        std::string first;
    };

    // Every interleaving of an error-free scenario: depth-first over the enabled actions,
    // states deduplicated by Sim::hash. A terminal state is bad unless finished, drained and
    // the counts match; any state with a safety violation or a mismatch is bad.
    DfsResult explore(Scenario const& sc, uint64_t maxStates)
    {
        struct Node
        {
            Sim                  sim;
            std::vector<uint8_t> bytes;
        };
        DfsResult                    res;
        std::unordered_set<uint64_t> seen;
        std::vector<Node>            stack;
        {
            Sim root(sc);
            seen.insert(root.hash());
            stack.push_back({root,
                             std::vector<uint8_t>(arena().base, arena().base + arena().used)});
        }
        while(!stack.empty())
        {
            Node node = std::move(stack.back());
            stack.pop_back();
            res.states++;
            if(res.states > maxStates)
            {
                res.truncated = true;
                break;
            }
            auto restore = [&]() {
                std::memcpy(arena().base, node.bytes.data(), node.bytes.size());
                arena().used = node.bytes.size();
            };
            restore();
            const auto acts = node.sim.enabled();
            if(acts.empty())
            {
                res.terminals++;
                const Outcome o = node.sim.result();
                if(!o.complete || !o.drained || !o.countsOk || o.safety() || o.mismatches)
                {
                    res.bad++;
                    if(res.first.empty())
                        res.first = o.first.empty() ? std::string("terminal: complete ")
                                                          + std::to_string(o.complete)
                                                          + " drained " + std::to_string(o.drained)
                                                          + " counts " + std::to_string(o.countsOk)
                                                    : o.first;
                }
                continue;
            }
            for(auto const& a : acts)
            {
                restore();
                Sim child = node.sim;
                child.apply(a);
                const Outcome o = child.result();
                if(o.safety() || o.mismatches)
                {
                    res.bad++;
                    res.badSafety++;
                    if(res.first.empty())
                        res.first = o.first;
                    continue;
                }
                if(seen.insert(child.hash()).second)
                    stack.push_back({child,
                                     std::vector<uint8_t>(arena().base,
                                                          arena().base + arena().used)});
            }
        }
        return res;
    }

    uint32_t kAllUnits()
    {
        return 1u << 30;
    }

    // Error-free configurations of the random grid.
    std::vector<Scenario> cleanGrid()
    {
        std::vector<Scenario> out;
        auto                  a2a = [&](uint32_t w, uint32_t b, uint32_t f, uint32_t cu, uint32_t s,
                       uint32_t c, uint32_t k) {
            Scenario sc;
            sc.world      = w;
            sc.blocks     = b;
            sc.features   = f;
            sc.numCu      = cu;
            sc.subRounds  = s;
            sc.coalesce   = c;
            sc.peerGroups = k;
            sc.launches   = 4;
            out.push_back(sc);
        };
        for(uint32_t w : {2u, 3u, 4u, 8u})
        {
            a2a(w, 1, 1, 8, 1, 1, 0);
            a2a(w, 3, 2, 4, 2, 1, 0);
            a2a(w, 5, 1, 8, 1, 2, 0);
            a2a(w, 0, 1, 8, 1, 1, 0);
            a2a(w, 4, 0, 8, 1, 1, 0);
        }
        a2a(4, 8, 1, 4, 4, kAllUnits(), 1);
        a2a(8, 16, 2, 8, 2, kAllUnits(), 3);
        a2a(8, 8, 1, 8, 1, 1, 7);
        a2a(4, 7, 1, 7, 1, 1, 0);
        for(uint32_t w : {2u, 3u, 4u, 8u})
            for(uint32_t t : {1u, 2u})
            {
                Scenario sc;
                sc.mode     = VIEW_MODE_GEMM_A2A;
                sc.world    = w;
                sc.blocks   = t;
                sc.launches = 4;
                out.push_back(sc);
            }
        Scenario g0;
        g0.mode     = VIEW_MODE_GEMM_A2A;
        g0.world    = 3;
        g0.blocks   = 0;
        g0.launches = 3;
        out.push_back(g0);
        std::vector<Scenario> more;
        for(auto sc : out)
        {
            more.push_back(sc);
            if(sc.world <= 4)
            {
                Scenario two = sc;
                two.slots    = 2;
                more.push_back(two);
                Scenario       small = sc;
                const uint32_t units
                    = sc.gemmA2A() ? sc.blocks : sc.blocks * sc.subRounds;
                const uint32_t bound = units * (VIEW_COPY_DWORDS + VIEW_ATOMIC_DWORDS)
                                       + 2 * VIEW_ATOMIC_DWORDS + 2 * VIEW_POLL_DWORDS;
                small.ringDwords     = 16;
                while(small.ringDwords < bound + 1)
                    small.ringDwords *= 2;
                small.launches = 6;
                more.push_back(small);
            }
        }
        return more;
    }

    std::string describe(Outcome const& o)
    {
        std::ostringstream s;
        s << "ra " << o.ra << " rb " << o.rb << " rc " << o.rc << " late " << o.late << " skew "
          << o.skew << " mismatches " << o.mismatches << " complete " << o.complete << " drained "
          << o.drained << " counts " << o.countsOk << " timeouts " << o.timeouts << " errors "
          << o.errors << " steps " << o.steps << (o.first.empty() ? "" : " first: ") << o.first;
        return s.str();
    }

    uint64_t seedsPerConfig()
    {
        if(char const* v = std::getenv("VIEW_SIM_SEEDS"))
            return std::strtoull(v, nullptr, 10);
        return 60;
    }

    // Timeouts a scenario may need: any with a GemmA2A dead rank, none otherwise.
    uint32_t allowedTimeouts(Scenario const& sc)
    {
        return sc.deadRank >= 0 && sc.gemmA2A() ? ~0u : 0u;
    }

    bool throws(Scenario const& sc)
    {
        return sc.throwSite >= 0 || (sc.deadRank >= 0 && sc.deadThrows);
    }

    bool cleanOk(Outcome const& o)
    {
        return o.complete && o.drained && o.countsOk && o.safety() == 0 && o.mismatches == 0
               && o.timeouts == 0 && o.errors == 0;
    }
}

// R-a, R-b, R-c, R-skew, R-cnt and liveness without errors, over the grid, every policy and
// seedsPerConfig() seeds each.
TEST(ViewTransportSim, CleanRunsKeepEveryInvariant)
{
    uint64_t runs = 0, steps = 0;
    for(auto const& sc : cleanGrid())
        for(Policy p : kPolicies)
            for(uint64_t seed = 0; seed < seedsPerConfig(); seed++)
            {
                const Outcome o = runOnce(sc, p, seed);
                runs++;
                steps += o.steps;
                ASSERT_TRUE(cleanOk(o)) << sc.name() << " policy " << (int)p << " seed " << seed
                                        << ": " << describe(o);
            }
    std::cout << "SIM_CLEAN runs=" << runs << " steps=" << steps << std::endl;
}

// Every interleaving of small configurations; VIEW_SIM_DFS_LARGE adds the first 3000000 states of
// GemmA2A W3.
TEST(ViewTransportSim, ExhaustiveSmallConfigs)
{
    std::vector<Scenario> grid;
    auto                  add = [&](uint32_t mode, uint32_t b, uint32_t s, uint32_t l, uint32_t slots) {
        Scenario sc;
        sc.mode      = mode;
        sc.blocks    = b;
        sc.subRounds = s;
        sc.launches   = l;
        sc.slots      = slots;
        sc.ringDwords = 256;
        grid.push_back(sc);
    };
    add(VIEW_MODE_A2A_GEMM, 1, 1, 3, 1);
    add(VIEW_MODE_A2A_GEMM, 2, 1, 2, 1);
    add(VIEW_MODE_A2A_GEMM, 1, 2, 2, 1);
    add(VIEW_MODE_A2A_GEMM, 1, 1, 3, 2);
    add(VIEW_MODE_GEMM_A2A, 1, 1, 3, 1);
    add(VIEW_MODE_GEMM_A2A, 1, 1, 3, 2);
    add(VIEW_MODE_A2A_GEMM, 1, 1, 2, 1);
    grid.back().world = 3;
    if(std::getenv("VIEW_SIM_DFS_LARGE") != nullptr)
    {
        add(VIEW_MODE_GEMM_A2A, 1, 1, 2, 1);
        grid.back().world = 3;
    }
    for(auto const& sc : grid)
    {
        const DfsResult r = explore(sc, 3000000);
        std::cout << "SIM_DFS " << sc.name() << " states=" << r.states
                  << " terminals=" << r.terminals << " bad=" << r.bad
                  << " truncated=" << r.truncated << std::endl;
        if(!(sc.gemmA2A() && sc.world == 3))
            EXPECT_FALSE(r.truncated) << sc.name();
        EXPECT_EQ(r.bad, 0u) << sc.name() << ": " << r.first;
        EXPECT_GT(r.terminals, 0u) << sc.name();
    }
}

// R-live: one error at a time (ring full, a wait timeout, a throw at every site of the per-device
// body, a dead rank) leaves no live rank parked: every live stream finishes and every live queue
// drains. A throw needs neither the host release nor a timeout, also with the host release due
// from the first step on, and leaves an error bit; a dead rank's throw needs the host release, and
// timeouts only for GemmA2A.
TEST(ViewTransportSim, ErrorsAndThrowsLeaveNoRankParked)
{
    uint64_t runs = 0;
    auto     check = [&](Scenario const& sc, bool needPhase1) {
        for(Policy p : kPolicies)
            for(uint64_t seed = 0; seed < seedsPerConfig() / 4 + 1; seed++)
                for(bool atThrow : {false, true})
                {
                    if(atThrow && !throws(sc))
                        continue;
                    bool          phase1 = false;
                    const Outcome o      = runOnce(sc, p, seed, &phase1, atThrow);
                    runs++;
                    ASSERT_TRUE(o.complete && o.drained && o.timeouts <= allowedTimeouts(sc))
                        << sc.name() << " policy " << (int)p << " seed " << seed << " release at throw "
                        << atThrow << ": " << describe(o);
                    if(needPhase1)
                        ASSERT_TRUE(phase1) << sc.name() << " policy " << (int)p << " seed " << seed
                                            << ": needed the host release or a timeout";
                    if(sc.throwSite >= 0)
                        ASSERT_NE(o.errors, 0u) << sc.name() << " policy " << (int)p << " seed "
                                                << seed << ": no error bit";
                }
    };
    for(uint32_t mode : {VIEW_MODE_A2A_GEMM, VIEW_MODE_GEMM_A2A})
        for(uint32_t w : {2u, 3u, 4u})
        {
            Scenario base;
            base.mode     = mode;
            base.world    = w;
            base.blocks   = 2;
            base.launches = 4;
            for(uint32_t launch : {0u, 1u, 3u})
            {
                for(uint32_t r : {0u, w - 1})
                {
                    Scenario full       = base;
                    full.ringFullRank   = r;
                    full.ringFullQueue  = 0;
                    full.ringFullLaunch = launch;
                    check(full, false);
                    Scenario late       = base;
                    late.spuriousRank   = r;
                    late.spuriousLaunch = launch;
                    check(late, false);
                }
                for(int site = 0; site < 6; site++)
                    for(uint32_t d : {0u, w / 2, w - 1})
                    {
                        Scenario t    = base;
                        t.throwSite   = site;
                        t.throwDevice = d;
                        t.throwLaunch = launch;
                        check(t, true);
                    }
                for(uint32_t d : {0u, w - 1})
                {
                    Scenario dead   = base;
                    dead.deadRank   = d;
                    dead.deadLaunch = launch;
                    check(dead, false);
                    if(mode == VIEW_MODE_GEMM_A2A)
                    {
                        dead.deadThrows = false;
                        check(dead, false);
                    }
                }
            }
        }
    std::cout << "SIM_ERRORS runs=" << runs << std::endl;
}

// The host arm (viewArmHostQueues): the clean invariants over the clean grid; with a full ring or
// a throw at W2-4, every stream finishes and every queue drains with no timeout and an error bit.
TEST(ViewTransportSim, HostArmKeepsEveryInvariant)
{
    uint64_t runs = 0;
    for(auto sc : cleanGrid())
    {
        sc.hostArm = true;
        for(Policy p : kPolicies)
            for(uint64_t seed = 0; seed < seedsPerConfig() / 4 + 1; seed++)
            {
                const Outcome o = runOnce(sc, p, seed);
                runs++;
                ASSERT_TRUE(cleanOk(o)) << sc.name() << " policy " << (int)p << " seed " << seed
                                        << ": " << describe(o);
            }
    }
    for(uint32_t mode : {VIEW_MODE_A2A_GEMM, VIEW_MODE_GEMM_A2A})
        for(uint32_t w : {2u, 3u, 4u})
            for(uint32_t launch : {0u, 1u, 3u})
            {
                Scenario base;
                base.mode     = mode;
                base.world    = w;
                base.blocks   = 2;
                base.launches = 4;
                base.hostArm  = true;
                Scenario full       = base;
                full.ringFullRank   = w - 1;
                full.ringFullLaunch = launch;
                Scenario t    = base;
                t.throwSite   = (int)ViewThrowSite::Launch;
                t.throwDevice = w - 1;
                t.throwLaunch = launch;
                for(auto const& sc : {full, t})
                    for(Policy p : kPolicies)
                        for(uint64_t seed = 0; seed < seedsPerConfig() / 4 + 1; seed++)
                        {
                            const Outcome o = runOnce(sc, p, seed, nullptr, sc.throwSite >= 0);
                            runs++;
                            ASSERT_TRUE(o.complete && o.drained && o.timeouts == 0 && o.errors != 0)
                                << sc.name() << " policy " << (int)p << " seed " << seed << ": "
                                << describe(o);
                        }
            }
    std::cout << "SIM_HOST_ARM runs=" << runs << std::endl;
}

// A2AGemm throws (W4 check:3:5 and start:3:5, W8 stop:7:5, every site of W2 at launches 0 and
// 1) with the host release due from the first step on, and a dead last rank (W2-4) with it due
// from its death on: every live stream finishes without a timeout; with ReleaseOverwritesPulled
// every W > 2 case needs one.
TEST(ViewTransportSim, HostReleaseAtTheThrowNeedsNoTimeout)
{
    std::vector<Scenario> traces;
    auto                  add = [&](uint32_t w, int site, uint32_t d, uint32_t launch) {
        Scenario sc;
        sc.world    = w;
        sc.blocks   = 2;
        sc.launches = launch + 1;
        if(site >= 0)
        {
            sc.throwSite   = site;
            sc.throwDevice = d;
            sc.throwLaunch = launch;
        }
        else
        {
            sc.deadRank   = (int)d;
            sc.deadLaunch = launch;
        }
        traces.push_back(sc);
    };
    add(4, (int)ViewThrowSite::Check, 3, 5);
    add(4, (int)ViewThrowSite::Start, 3, 5);
    add(8, (int)ViewThrowSite::Stop, 7, 5);
    for(int site = 0; site < 6; site++)
        for(uint32_t d : {0u, 1u})
            for(uint32_t launch : {0u, 1u})
                add(2, site, d, launch);
    for(uint32_t w : {2u, 3u, 4u})
        for(uint32_t launch : {0u, 1u, 3u})
            add(w, -1, w - 1, launch);
    uint64_t runs = 0;
    for(auto const& sc : traces)
    {
        uint64_t old = 0;
        for(Policy p : kPolicies)
            for(uint64_t seed = 0; seed < seedsPerConfig(); seed++)
            {
                const Outcome o = runOnce(sc, p, seed, nullptr, true);
                runs++;
                ASSERT_TRUE(o.complete && o.drained && o.timeouts == 0)
                    << sc.name() << " policy " << (int)p << " seed " << seed << ": " << describe(o);
                Scenario before = sc;
                before.mutant   = Mutant::ReleaseOverwritesPulled;
                old += runOnce(before, p, seed, nullptr, true).timeouts != 0;
            }
        std::cout << "SIM_RELEASE_AT_THROW " << sc.name() << " release-overwrites-pulled runs with a timeout "
                  << old << std::endl;
        if(sc.world > 2)
            EXPECT_GT(old, 0u) << sc.name();
    }
    std::cout << "SIM_RELEASE_AT_THROW runs=" << runs << std::endl;
}

// The epoch bound: clean up to the last launch viewLaunchLimit allows; a run past 2^32 is caught.
TEST(ViewTransportSim, EpochsUpToTheLaunchLimit)
{
    for(uint32_t mode : {VIEW_MODE_A2A_GEMM, VIEW_MODE_GEMM_A2A})
    {
        Scenario sc;
        sc.mode     = mode;
        sc.world    = 3;
        sc.blocks   = 2;
        sc.launches = 4;
        ViewArmArgs a{};
        a.mode         = mode;
        a.tilesPerRank = sc.tilesPerRank;
        sc.epoch0      = (uint32_t)(viewLaunchLimit(a) - sc.launches);
        EXPECT_EQ(viewLaunchLimit(a), mode == VIEW_MODE_GEMM_A2A ? 0xFFFFFFFFull / 3 : 1ull << 31);
        for(Policy p : kPolicies)
            for(uint64_t seed = 0; seed < seedsPerConfig(); seed++)
            {
                const Outcome o = runOnce(sc, p, seed);
                ASSERT_TRUE(cleanOk(o)) << sc.name() << " seed " << seed << ": " << describe(o);
            }
    }
    Scenario wrap;
    wrap.world    = 3;
    wrap.blocks   = 2;
    wrap.launches = 4;
    wrap.epoch0   = 0xFFFFFFFEu;
    uint64_t caught = 0;
    for(Policy p : kPolicies)
        for(uint64_t seed = 0; seed < seedsPerConfig(); seed++)
            caught += runOnce(wrap, p, seed).ra != 0;
    EXPECT_GT(caught, 0u) << "a head reference that wraps to 0 must be caught";
}

namespace
{
    struct Probe
    {
        Scenario    sc;
        std::string expect; // "safety" or "liveness"
    };

    // Whether a schedule of `probes` fails as expected; the failing run in `why`. Safety: a
    // random run with a violation and a consumer mismatch, else an exhaustive one (W = 2) with
    // a violation. Liveness, for a throw also with the host release due from the first step on:
    // a live rank or queue left parked, a timeout the scenario does not need, a throw that needs
    // the host release or leaves no error bit, or an error word in an error-free scenario.
    bool caught(std::vector<Probe> const& probes, std::string& why, uint64_t seeds)
    {
        for(auto const& pr : probes)
            for(Policy p : kPolicies)
                for(uint64_t seed = 0; seed < seeds; seed++)
                    for(bool atThrow : {false, true})
                    {
                        if(atThrow && (pr.expect == "safety" || !throws(pr.sc)))
                            continue;
                        bool          phase1 = true;
                        const Outcome o      = runOnce(pr.sc, p, seed, &phase1, atThrow);
                        const bool    hit
                            = pr.expect == "safety"
                                  ? o.safety() != 0 && o.mismatches != 0
                                  : !o.complete || !o.drained || o.timeouts > allowedTimeouts(pr.sc)
                                        || (pr.sc.throwSite >= 0 && !phase1)
                                        || (pr.sc.throwSite >= 0 && o.errors == 0)
                                        || (pr.sc.errorFree() && o.errors != 0);
                        if(hit)
                        {
                            why = pr.sc.name() + " policy " + std::to_string((int)p) + " seed "
                                  + std::to_string(seed) + (atThrow ? " release at throw" : "")
                                  + ": " + describe(o);
                            return true;
                        }
                    }
        for(auto const& pr : probes)
            if(pr.expect == "safety" && pr.sc.errorFree() && pr.sc.world == 2)
            {
                const DfsResult r = explore(pr.sc, 3000000);
                if(r.badSafety != 0)
                {
                    why = pr.sc.name() + " (exhaustive, violation only): " + r.first;
                    return true;
                }
            }
        return false;
    }

    std::vector<Probe> probesFor(Mutant m)
    {
        auto base = [&](uint32_t mode, uint32_t w) {
            Scenario sc;
            sc.mode     = mode;
            sc.world    = w;
            sc.blocks   = 2;
            sc.launches = 3;
            sc.mutant   = m;
            sc.ringDwords = 1u << 14;
            return sc;
        };
        const Scenario a2 = base(VIEW_MODE_A2A_GEMM, 2), a3 = base(VIEW_MODE_A2A_GEMM, 3);
        const Scenario g2 = base(VIEW_MODE_GEMM_A2A, 2), g3 = base(VIEW_MODE_GEMM_A2A, 3);
        auto           with = [](Scenario sc, auto&& f) {
            f(sc);
            return sc;
        };
        switch(m)
        {
        case Mutant::None:
            return {};
        case Mutant::NoHeadA2AGemm:
        case Mutant::NoExitWait:
        case Mutant::ExitEpochMinus1:
        case Mutant::PulledEarly:
        case Mutant::PublishBeforeProducer:
            return {{a2, "safety"}, {a3, "safety"}};
        case Mutant::NoHeadGemmA2A:
            return {{g2, "safety"}, {g3, "safety"}};
        case Mutant::HeadRefE:
            return {{a2, "safety"}, {g2, "safety"}, {a3, "safety"}, {g3, "safety"}};
        case Mutant::PulledLast:
        case Mutant::NoPulledAtomic:
            return {{a2, "liveness"}, {a3, "liveness"}};
        case Mutant::BookWithoutPulled:
        case Mutant::PublishOnlyUnhalted:
            return {{with(a3,
                          [](Scenario& s) {
                              s.ringFullRank   = 1;
                              s.ringFullLaunch = 1;
                          }),
                     "liveness"}};
        case Mutant::NoUnarmedAbort:
            return {{with(a3,
                          [](Scenario& s) {
                              s.throwSite   = (int)ViewThrowSite::Arm;
                              s.throwDevice = 1;
                              s.throwLaunch = 1;
                          }),
                     "liveness"},
                    {with(g3,
                          [](Scenario& s) {
                              s.throwSite   = (int)ViewThrowSite::Start;
                              s.throwDevice = 0;
                              s.throwLaunch = 1;
                          }),
                     "liveness"}};
        case Mutant::TryArmLaunchOnly:
            return {{with(a3,
                          [](Scenario& s) {
                              s.throwSite   = (int)ViewThrowSite::Wait;
                              s.throwDevice = 0;
                              s.throwLaunch = 1;
                          }),
                     "liveness"}};
        case Mutant::NoHostRelease:
            return {{with(a3,
                          [](Scenario& s) {
                              s.deadRank   = 1;
                              s.deadLaunch = 1;
                          }),
                     "liveness"}};
        case Mutant::NoTimeoutRelease:
            return {{with(g3,
                          [](Scenario& s) {
                              s.deadRank   = 1;
                              s.deadLaunch = 1;
                              s.deadThrows = false;
                          }),
                     "liveness"}};
        case Mutant::ReleaseOverwritesPulled:
        case Mutant::ExitIgnoresRelease:
            return {{with(a3,
                          [](Scenario& s) {
                              s.deadRank   = 1;
                              s.deadLaunch = 1;
                          }),
                     "liveness"},
                    {with(a3,
                          [](Scenario& s) {
                              s.throwSite   = (int)ViewThrowSite::Check;
                              s.throwDevice = 2;
                              s.throwLaunch = 1;
                          }),
                     "liveness"}};
        case Mutant::NoArmedAbortA2AGemm:
            return {{with(a3,
                          [](Scenario& s) {
                              s.throwSite   = (int)ViewThrowSite::Check;
                              s.throwDevice = 2;
                              s.throwLaunch = 1;
                          }),
                     "liveness"}};
        }
        return {};
    }
}

// Calibration: each mutant of the protocol fails the way it must; safety mutants also show up as
// a mismatch at the consumer (the verify detector's replay).
TEST(ViewTransportSim, MutantsAreCaught)
{
    const Mutant all[] = {Mutant::NoHeadA2AGemm,
                          Mutant::NoHeadGemmA2A,
                          Mutant::HeadRefE,
                          Mutant::NoExitWait,
                          Mutant::ExitEpochMinus1,
                          Mutant::PulledEarly,
                          Mutant::PulledLast,
                          Mutant::PublishBeforeProducer,
                          Mutant::NoPulledAtomic,
                          Mutant::BookWithoutPulled,
                          Mutant::PublishOnlyUnhalted,
                          Mutant::NoUnarmedAbort,
                          Mutant::TryArmLaunchOnly,
                          Mutant::NoHostRelease,
                          Mutant::NoTimeoutRelease,
                          Mutant::ReleaseOverwritesPulled,
                          Mutant::ExitIgnoresRelease,
                          Mutant::NoArmedAbortA2AGemm};
    for(Mutant m : all)
    {
        std::string why;
        const bool  hit = caught(probesFor(m), why, std::max<uint64_t>(seedsPerConfig(), 200));
        std::cout << "SIM_MUTANT " << mutantName(m) << (hit ? " caught: " : " NOT CAUGHT") << why
                  << std::endl;
        EXPECT_TRUE(hit) << mutantName(m);
    }
}

// The debug bits as the on-GPU calibration uses them: NO_READY breaks (a)/(c), NO_EXITWAIT
// breaks (b) (also with DELAY_CHAIN); I1's host barrier, DELAY_CHAIN alone, and NO_EXITWAIT
// with a static B pass.
TEST(ViewTransportSim, DebugBitsCalibration)
{
    auto sc = [](uint32_t mode, uint32_t w, int debug) {
        Scenario s;
        s.mode     = mode;
        s.world    = w;
        s.blocks   = 2;
        s.launches = 3;
        s.debug    = debug;
        return s;
    };
    std::string why;
    for(uint32_t mode : {VIEW_MODE_A2A_GEMM, VIEW_MODE_GEMM_A2A})
    {
        EXPECT_TRUE(caught({{sc(mode, 2, VIEW_DEBUG_NO_READY), "safety"},
                            {sc(mode, 3, VIEW_DEBUG_NO_READY), "safety"}},
                           why,
                           200))
            << "NO_READY mode " << mode;
        std::cout << "SIM_DEBUG NO_READY mode " << mode << ": " << why << std::endl;
    }
    for(int debug : {VIEW_DEBUG_NO_EXITWAIT, VIEW_DEBUG_NO_EXITWAIT | VIEW_DEBUG_DELAY_CHAIN})
    {
        Scenario s = sc(VIEW_MODE_A2A_GEMM, 3, debug);
        EXPECT_TRUE(caught({{sc(VIEW_MODE_A2A_GEMM, 2, debug), "safety"}, {s, "safety"}}, why, 200))
            << "debug " << debug;
        std::cout << "SIM_DEBUG debug " << debug << ": " << why << std::endl;
    }
    std::vector<Scenario> pass;
    for(uint32_t mode : {VIEW_MODE_A2A_GEMM, VIEW_MODE_GEMM_A2A})
    {
        Scenario b = sc(mode, 3, VIEW_DEBUG_NO_READY | VIEW_DEBUG_NO_EXITWAIT);
        b.barrier  = true;
        pass.push_back(b);
    }
    pass.push_back(sc(VIEW_MODE_A2A_GEMM, 3, VIEW_DEBUG_DELAY_CHAIN));
    Scenario still = sc(VIEW_MODE_A2A_GEMM, 3, VIEW_DEBUG_NO_EXITWAIT);
    still.produceB = false;
    pass.push_back(still);
    for(auto const& s : pass)
        for(Policy p : kPolicies)
            for(uint64_t seed = 0; seed < seedsPerConfig(); seed++)
            {
                const Outcome o = runOnce(s, p, seed);
                ASSERT_TRUE(cleanOk(o)) << s.name() << " seed " << seed << ": " << describe(o);
            }
    for(auto const& s : pass)
    {
        if(s.world != 3 || s.barrier)
            continue;
        Scenario two = s;
        two.world    = 2;
        const DfsResult r = explore(two, 3000000);
        EXPECT_EQ(r.bad, 0u) << two.name() << ": " << r.first;
        EXPECT_FALSE(r.truncated);
    }
}
