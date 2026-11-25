// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <Tensile/ContractionProblem_fwd.hpp>
#include <Tensile/ContractionSolution_fwd.hpp>
#include <Tensile/FusedA2AKernArg.hpp>
#include <Tensile/Tensile_fwd.hpp>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        struct DeviceContext;

        constexpr uint32_t VIEW_COPY_DWORDS   = 13;
        constexpr uint32_t VIEW_ATOMIC_DWORDS = 8;
        constexpr uint32_t VIEW_POLL_DWORDS   = 6;
        constexpr uint32_t VIEW_TIMESTAMP_DWORDS = 3;

        // COPY_LINEAR_SUB_WIND, 16 B elements.
        constexpr uint32_t VIEW_COPY_HEADER = 0x80000401u;
        // ATOMIC ADD_RTN_32.
        constexpr uint32_t VIEW_ATOMIC_HEADER = 0x1E00000Au;
        // POLL_REGMEM on memory, func 5 (>=).
        constexpr uint32_t VIEW_POLL_HEADER = 0xD0000008u;
        constexpr uint32_t VIEW_POLL_MASK   = 0xFFFFFFFFu;
        // retry_count 0xFFF, interval 10.
        constexpr uint32_t VIEW_POLL_RETRY_INTERVAL = 0x0FFF000Au;
        // TIMESTAMP GET_GLOBAL.
        constexpr uint32_t VIEW_TIMESTAMP_HEADER = 0x0000020Du;

        // Bits of the view block's error word.
        constexpr uint32_t VIEW_ERROR_RING_FULL = 1u;
        constexpr uint32_t VIEW_ERROR_TIMEOUT   = 2u;
        // The GEMM launch after the arm failed.
        constexpr uint32_t VIEW_ERROR_ABORTED = 4u;
        // An own queue still had packets pending after the release.
        constexpr uint32_t VIEW_ERROR_STUCK = 8u;

        constexpr uint32_t VIEW_MODE_A2A_GEMM = 0;
        constexpr uint32_t VIEW_MODE_GEMM_A2A = 1;

        // --fused-a2a-view-arm values.
        constexpr int VIEW_ARM_DEVICE = 0;
        constexpr int VIEW_ARM_HOST   = 1;

        // --fused-a2a-view-debug bits.
        constexpr int VIEW_DEBUG_TIMELINE       = 1;
        constexpr int VIEW_DEBUG_NO_POLL        = 2;
        constexpr int VIEW_DEBUG_INKERNEL_NOOPS = 4;
        // Queue 0 omits its first COPY and keeps every ATOMIC.
        constexpr int VIEW_DEBUG_DROP_COPY = 8;
        // Queue 0's first COPY moves to just before the chain's last ATOMIC.
        constexpr int VIEW_DEBUG_EARLY_RELEASE = 16;
        constexpr int VIEW_DEBUG_COPY_FAULTS   = VIEW_DEBUG_DROP_COPY | VIEW_DEBUG_EARLY_RELEASE;
        // No head POLL and no entry publish.
        constexpr int VIEW_DEBUG_NO_READY = 32;
        // A2AGemm: no pulled ATOMIC, no pulled booking and no exit wait.
        constexpr int VIEW_DEBUG_NO_EXITWAIT = 64;
        // A2AGemm: queue 0 of delayRank copies delayCopies units within delayScratch right
        // before its last COPY.
        constexpr int VIEW_DEBUG_DELAY_CHAIN = 128;
        constexpr int VIEW_DEBUG_ALL         = 255;

        // Client-only tail of a view block, at viewTailOffset bytes: u32 entry[W] at
        // VIEW_TAIL_ENTRY_OFFSET, u32 released at VIEW_TAIL_RELEASED_OFFSET, u32 pulled[W] at
        // VIEW_TAIL_PULLED_OFFSET. entry[x] is rank x's arm count; released is non-zero after the
        // host release; pulled[r] counts rank r's A2AGemm chains for the block's owner that passed
        // their last COPY, plus their bookings.
        constexpr uint32_t VIEW_TAIL_ENTRY_OFFSET    = 0;
        constexpr uint32_t VIEW_TAIL_RELEASED_OFFSET = 32;
        constexpr uint32_t VIEW_TAIL_PULLED_OFFSET   = 64;
        constexpr uint32_t VIEW_TAIL_BYTES           = 128;
        // The host release stores VIEW_TAIL_RELEASED into the tail's first VIEW_TAIL_RELEASE_BYTES.
        constexpr uint32_t VIEW_TAIL_RELEASE_BYTES = VIEW_TAIL_PULLED_OFFSET;
        constexpr uint32_t VIEW_TAIL_RELEASE_WORDS = VIEW_TAIL_RELEASE_BYTES / 4;
        static_assert(VIEW_TAIL_ENTRY_OFFSET + 4 * FUSED_A2A_MAX_RANKS <= VIEW_TAIL_RELEASED_OFFSET);
        static_assert(VIEW_TAIL_RELEASED_OFFSET + 4 <= VIEW_TAIL_RELEASE_BYTES);
        static_assert(VIEW_TAIL_PULLED_OFFSET + 4 * FUSED_A2A_MAX_RANKS <= VIEW_TAIL_BYTES);
        constexpr uint32_t VIEW_TAIL_RELEASED = 0xFFFFFFFFu;
        // A solution's launch count stays below it (viewLaunchLimit).
        constexpr uint64_t VIEW_EPOCH_LIMIT = 1ull << 31;

        // --fused-a2a-view-release values: how the error path writes VIEW_TAIL_RELEASED.
        // CPU stores through the view block pointer.
        constexpr int VIEW_RELEASE_HOST_STORE = 1;
        // hipMemcpyAsync from pinned memory on a non-blocking stream.
        constexpr int VIEW_RELEASE_MEMCPY = 2;
        // A fill kernel on that stream.
        constexpr int VIEW_RELEASE_KERNEL = 3;

        // VIEW_DEBUG_DELAY_CHAIN scratch per device, bytes.
        constexpr size_t VIEW_DELAY_SCRATCH_BYTES = 16ull << 20;

        // VIEW_DEBUG_TIMELINE ring: record (epoch % VIEW_STATS_RECORDS) of VIEW_STATS_FIELDS
        // u64 slots, one slot per VIEW_STATS_SLOT_U64 u64s (32 B). Fields are wall-clock ticks.
        constexpr uint32_t VIEW_STATS_RECORDS  = 64;
        constexpr uint32_t VIEW_STATS_SLOT_U64 = 4;
        constexpr uint32_t VIEW_STATS_ARM_IN   = 0;
        constexpr uint32_t VIEW_STATS_ARM_OUT  = 1;
        constexpr uint32_t VIEW_STATS_GEMM_END = 2;
        constexpr uint32_t VIEW_STATS_WAIT_OUT = 3;
        // Queue q: VIEW_STATS_TS_START + 2q, VIEW_STATS_TS_END + 2q and VIEW_STATS_TS_READY + q.
        constexpr uint32_t VIEW_STATS_TS_START = 4;
        constexpr uint32_t VIEW_STATS_TS_END   = 5;
        constexpr uint32_t VIEW_STATS_TS_READY = 4 + 2 * FUSED_A2A_MAX_RANKS;
        constexpr uint32_t VIEW_STATS_FIELDS   = 4 + 3 * FUSED_A2A_MAX_RANKS;
        constexpr size_t   VIEW_STATS_BYTES
            = (size_t)VIEW_STATS_RECORDS * VIEW_STATS_FIELDS * VIEW_STATS_SLOT_U64 * sizeof(uint64_t);
        constexpr uint32_t VIEW_STATS_TICKS_PER_US = 100;

        constexpr uint64_t kViewWaitTimeoutMs = 5000;
        // viewSyncBounded.
        constexpr uint64_t kViewSyncBoundMs = 2 * kViewWaitTimeoutMs + 1000;

        struct ViewQueueArgs
        {
            uint32_t* ring; // ring base, mapped on the host and the device
            uint64_t  ringBytes; // bytes, a power of two
            uint64_t* rptr; // absolute bytes, written by the engine
            uint64_t* wptr; // absolute bytes
            uint64_t* doorbell; // written with the absolute byte wptr
            uint64_t  srcBase; // byte address of unit (b=0, h=0)'s COPY source
            uint64_t  dstBase; // byte address of unit (b=0, h=0)'s COPY destination
            uint32_t*
                flag; // u32 the unit ATOMICs add to: own flag[i] (A2AGemm), peer p's flag[rank] (GemmA2A)
            uint32_t* ready; // own entry[peer], the head POLL's word; nullptr: no head
            uint32_t* entryOut; // peer's entry[rank]; nullptr: no publish
            uint32_t* pulled; // A2AGemm: source's pulled[rank]; nullptr: no pulled ATOMIC
        };

        // One device's arm for one launch. A2AGemm queue i serves source (rank+1+i) mod W;
        // GemmA2A queue p feeds rank p.
        struct ViewArmArgs
        {
            uint32_t      mode; // VIEW_MODE_*
            uint32_t      rank;
            uint32_t      world; // W
            uint32_t      numQueues; // W-1 (A2AGemm) or W (GemmA2A)
            uint32_t      subRounds; // S; 1 for GemmA2A
            uint32_t      peerGroups; // A2AGemm: the kernel's A2APeerGroups
            uint32_t      tokenBlocks; // B (A2AGemm) or T = NumWorkGroups1 (GemmA2A)
            uint32_t      featureTiles; // A2AGemm: F = NumWorkGroups0
            uint32_t      numCu; // A2AGemm: the kernel's FusedNumCu
            uint32_t      mt1; // MacroTile1, rows
            uint32_t      nTokens; // rows
            uint32_t      srcPitch; // 16 B elements
            uint32_t      dstPitch; // 16 B elements
            uint32_t      slice; // 16 B elements, src and dst
            uint32_t      rectX; // 16 B elements
            uint64_t      srcBlockStride; // bytes from token block b to b+1
            uint64_t      dstBlockStride; // bytes from token block b to b+1
            uint64_t      subRoundStride; // bytes from sub-round h to h+1, src and dst
            uint32_t      coalesce; // A2AGemm: COPY units per ATOMIC
            uint32_t      tilesPerRank; // GemmA2A: work-groups per band, nShard / MT0
            uint32_t*     view; // this device's view block
            uint64_t      waitTimeoutTicks; // wait kernel bound, wall-clock ticks
            uint32_t      debug; // --fused-a2a-view-debug
            uint64_t*     stats; // VIEW_DEBUG_TIMELINE ring, else nullptr
            uint32_t*     tail; // this device's view-block tail
            uint32_t      delayRank; // VIEW_DEBUG_DELAY_CHAIN
            uint32_t      delayCopies; // VIEW_DEBUG_DELAY_CHAIN
            uint64_t      delayScratch; // byte address of 2 * srcBlockStride bytes, or 0
            ViewQueueArgs queues[FUSED_A2A_MAX_RANKS];
        };

        struct ViewRingWriter
        {
            uint32_t* ring; // nullptr: count only
            uint64_t  dwords; // ring size in dwords, a power of two
            uint64_t  pos; // absolute dword position
        };

        __host__ __device__ inline uint32_t viewLoad32(uint32_t* p)
        {
#if defined(__HIP_DEVICE_COMPILE__)
            return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
            return *reinterpret_cast<volatile uint32_t*>(p);
#endif
        }

        __host__ __device__ inline void viewStore32(uint32_t* p, uint32_t v)
        {
#if defined(__HIP_DEVICE_COMPILE__)
            __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
            *reinterpret_cast<volatile uint32_t*>(p) = v;
#endif
        }

        __host__ __device__ inline uint64_t viewLoad64(uint64_t* p)
        {
#if defined(__HIP_DEVICE_COMPILE__)
            return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
            return *reinterpret_cast<volatile uint64_t*>(p);
#endif
        }

        __host__ __device__ inline void viewAdd32(uint32_t* p, uint32_t v)
        {
#if defined(__HIP_DEVICE_COMPILE__)
            __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
            __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST);
#endif
        }

        __host__ __device__ inline void viewMax32(uint32_t* p, uint32_t v)
        {
#if defined(__HIP_DEVICE_COMPILE__)
            __hip_atomic_fetch_max(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
            uint32_t old = __atomic_load_n(p, __ATOMIC_SEQ_CST);
            while(old < v && !__atomic_compare_exchange_n(
                      p, &old, v, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))
                ;
#endif
        }

        // Ordered after every earlier store of the calling thread.
        __host__ __device__ inline void viewRelease32(uint32_t* p, uint32_t v)
        {
#if defined(__HIP_DEVICE_COMPILE__)
            __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
            *reinterpret_cast<volatile uint32_t*>(p) = v;
#endif
        }

        // Ordered after every earlier store of the calling thread.
        __host__ __device__ inline void viewPublish64(uint64_t* p, uint64_t v)
        {
#if defined(__HIP_DEVICE_COMPILE__)
            __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
            *reinterpret_cast<volatile uint64_t*>(p) = v;
#endif
        }

        __host__ __device__ inline void viewPut(ViewRingWriter& w, uint32_t v)
        {
            if(w.ring != nullptr)
                viewStore32(w.ring + (w.pos & (w.dwords - 1)), v);
            w.pos++;
        }

        // Zero dwords up to the ring end when n dwords would straddle it.
        __host__ __device__ inline void viewReserve(ViewRingWriter& w, uint32_t n)
        {
            const uint64_t room = w.dwords - (w.pos & (w.dwords - 1));
            if(n > room)
                for(uint64_t k = 0; k < room; k++)
                    viewPut(w, 0u);
        }

        __host__ __device__ inline void viewPutAddress(ViewRingWriter& w, uint64_t addr)
        {
            viewPut(w, (uint32_t)addr);
            viewPut(w, (uint32_t)(addr >> 32));
        }

        __host__ __device__ inline void viewPutCopy(ViewRingWriter& w,
                                                    uint64_t        src,
                                                    uint64_t        dst,
                                                    uint32_t        srcPitch,
                                                    uint32_t        dstPitch,
                                                    uint32_t        slice,
                                                    uint32_t        rectX,
                                                    uint32_t        rectY)
        {
            viewReserve(w, VIEW_COPY_DWORDS);
            viewPut(w, VIEW_COPY_HEADER);
            viewPutAddress(w, src);
            viewPut(w, 0u);
            viewPut(w, (srcPitch - 1u) << 13);
            viewPut(w, slice - 1u);
            viewPutAddress(w, dst);
            viewPut(w, 0u);
            viewPut(w, (dstPitch - 1u) << 13);
            viewPut(w, slice - 1u);
            viewPut(w, ((rectY << 16) | rectX) - 0x00010001u);
            viewPut(w, 0u);
        }

        __host__ __device__ inline void
            viewPutAtomic(ViewRingWriter& w, uint64_t addr, uint32_t addend)
        {
            viewReserve(w, VIEW_ATOMIC_DWORDS);
            viewPut(w, VIEW_ATOMIC_HEADER);
            viewPutAddress(w, addr);
            viewPut(w, addend);
            for(uint32_t k = 4; k < VIEW_ATOMIC_DWORDS; k++)
                viewPut(w, 0u);
        }

        __host__ __device__ inline void viewPutPoll(ViewRingWriter& w, uint64_t addr, uint32_t ref)
        {
            viewReserve(w, VIEW_POLL_DWORDS);
            viewPut(w, VIEW_POLL_HEADER);
            viewPutAddress(w, addr);
            viewPut(w, ref);
            viewPut(w, VIEW_POLL_MASK);
            viewPut(w, VIEW_POLL_RETRY_INTERVAL);
        }

        __host__ __device__ inline void viewPutTimestamp(ViewRingWriter& w, uint64_t addr)
        {
            viewReserve(w, VIEW_TIMESTAMP_DWORDS);
            viewPut(w, VIEW_TIMESTAMP_HEADER);
            viewPutAddress(w, addr);
        }

        // u64 index of `field` in the VIEW_DEBUG_TIMELINE record of `epoch`.
        __host__ __device__ inline size_t viewStatsIndex(uint32_t epoch, uint32_t field)
        {
            return ((size_t)(epoch % VIEW_STATS_RECORDS) * VIEW_STATS_FIELDS + field)
                   * VIEW_STATS_SLOT_U64;
        }

        __host__ __device__ inline uint32_t viewRectY(ViewArmArgs const& a, uint32_t b)
        {
            const uint32_t left = a.nTokens - b * a.mt1;
            return left < a.mt1 ? left : a.mt1;
        }

        __host__ __device__ inline uint64_t viewAddress(void const* p)
        {
            return (uint64_t)(uintptr_t)p;
        }

        // A2AGemm: F' = min(F, numCu); batch iB holds token blocks
        // [ceil(iB * numCu / F'), min(ceil((iB+1) * numCu / F'), B)). The non-empty batches are
        // InKernel's a2aBatchSpan spans; the indices differ from InKernel's when F > numCu.
        __host__ __device__ inline uint32_t viewA2AGemmFeatures(ViewArmArgs const& a)
        {
            return a.featureTiles < a.numCu ? a.featureTiles : a.numCu;
        }

        __host__ __device__ inline uint32_t viewA2AGemmBatchLo(ViewArmArgs const& a, uint32_t iB)
        {
            const uint32_t f = viewA2AGemmFeatures(a);
            return (iB * a.numCu + f - 1) / f;
        }

        __host__ __device__ inline uint32_t viewA2AGemmBatchHi(ViewArmArgs const& a, uint32_t iB)
        {
            const uint32_t hi = viewA2AGemmBatchLo(a, iB + 1);
            return hi < a.tokenBlocks ? hi : a.tokenBlocks;
        }

        // A2AGemm: nbMax = min(ceil(numCu / F'), B), the size of batch 0 and of the largest one;
        // 0 when F = 0.
        __host__ __device__ inline uint32_t viewA2AGemmBatchBlocksMax(ViewArmArgs const& a)
        {
            if(a.featureTiles == 0)
                return 0;
            const uint32_t nb = viewA2AGemmBatchLo(a, 1);
            return nb < a.tokenBlocks ? nb : a.tokenBlocks;
        }

        // A2AGemm: NB = (B-1) * F' / numCu + 1 dispatch batches; 0 when B = 0 or F = 0.
        __host__ __device__ inline uint32_t viewA2AGemmBatches(ViewArmArgs const& a)
        {
            if(a.tokenBlocks == 0 || a.featureTiles == 0)
                return 0;
            return (a.tokenBlocks - 1) * viewA2AGemmFeatures(a) / a.numCu + 1;
        }

        // A2AGemm: the ng groups of a batch of n blocks and the size gk of the larger ones, as
        // InKernel's a2aBatchSpan with G = numQueues: peerGroups 0: gk = ceil(n / G),
        // ng = ceil(n / gk); else ng = min(G, n, peerGroups), gk = ceil(n / ng). 0, 0 when n = 0.
        __host__ __device__ inline void
            viewA2AGemmShape(ViewArmArgs const& a, uint32_t n, uint32_t& ng, uint32_t& gk)
        {
            ng = gk = 0;
            if(n == 0 || a.numQueues == 0)
                return;
            if(a.peerGroups == 0)
            {
                gk = (n + a.numQueues - 1) / a.numQueues;
                ng = (n + gk - 1) / gk;
                return;
            }
            ng = a.numQueues < n ? a.numQueues : n;
            ng = a.peerGroups < ng ? a.peerGroups : ng;
            gk = (n + ng - 1) / ng;
        }

        // A2AGemm: Qb * S, Qb = gk of a batch of nbMax blocks (viewA2AGemmShape), the largest
        // group of any batch. The ordinal span of one (batch, round) pair.
        __host__ __device__ inline uint32_t viewA2AGemmRoundUnits(ViewArmArgs const& a)
        {
            uint32_t ng = 0, gk = 0;
            viewA2AGemmShape(a, viewA2AGemmBatchBlocksMax(a), ng, gk);
            return gk * a.subRounds;
        }

        // A2AGemm: NB * G * Qb * S, what one launch adds to each flag[i].
        __host__ __device__ inline uint32_t viewA2AGemmChainUnits(ViewArmArgs const& a)
        {
            return viewA2AGemmBatches(a) * a.numQueues * viewA2AGemmRoundUnits(a);
        }

        // A2AGemm: [lo, hi), the token blocks queue q serves in pair t = iB * G + j: group
        // grp = (q - j) mod G of batch iB, whose n blocks split into ng groups, the first
        // n + ng - ng * gk of gk blocks and the rest of gk - 1 (viewA2AGemmShape). Returns
        // whether grp < ng.
        __host__ __device__ inline bool viewA2AGemmGroup(
            ViewArmArgs const& a, uint32_t q, uint32_t t, uint32_t& lo, uint32_t& hi)
        {
            const uint32_t g   = a.numQueues;
            const uint32_t bLo = viewA2AGemmBatchLo(a, t / g);
            const uint32_t bHi = viewA2AGemmBatchHi(a, t / g);
            const uint32_t grp = (q + g - t % g) % g;
            uint32_t       ng = 0, gk = 0;
            viewA2AGemmShape(a, bHi - bLo, ng, gk);
            lo = hi = bLo;
            if(grp >= ng)
                return false;
            const uint32_t big = bHi - bLo + ng - ng * gk;
            lo                 = bLo + grp * gk - (grp > big ? grp - big : 0);
            hi                 = lo + (grp < big ? gk : gk - 1);
            return true;
        }

        // Whether queue q's chain has a COPY of a unit.
        __host__ __device__ inline bool viewChainHasUnit(ViewArmArgs const& a, uint32_t q)
        {
            if(a.mode != VIEW_MODE_A2A_GEMM)
                return a.tokenBlocks != 0;
            const uint32_t pairs = viewA2AGemmBatches(a) * a.numQueues;
            uint32_t       lo = 0, hi = 0;
            for(uint32_t t = 0; t < pairs; t++)
                if(viewA2AGemmGroup(a, q, t, lo, hi))
                    return true;
            return false;
        }

        // The word queue q's head POLL waits on; nullptr under VIEW_DEBUG_NO_READY.
        __host__ __device__ inline uint32_t* viewReady(ViewArmArgs const& a, uint32_t q)
        {
            return (a.debug & VIEW_DEBUG_NO_READY) != 0 ? nullptr : a.queues[q].ready;
        }

        // The word queue q's pulled ATOMIC or booking adds 1 to; nullptr for GemmA2A and under
        // VIEW_DEBUG_NO_EXITWAIT.
        __host__ __device__ inline uint32_t* viewPulled(ViewArmArgs const& a, uint32_t q)
        {
            return a.mode != VIEW_MODE_A2A_GEMM || (a.debug & VIEW_DEBUG_NO_EXITWAIT) != 0
                       ? nullptr
                       : a.queues[q].pulled;
        }

        // The tail of a view block whose tail starts `tailOffset` bytes into it.
        __host__ __device__ inline uint32_t* viewTailOf(void* viewBlock, size_t tailOffset)
        {
            return reinterpret_cast<uint32_t*>(static_cast<char*>(viewBlock) + tailOffset);
        }

        // tails[x] is rank x's tail. Sets a.tail to tails[a.rank] and, for every queue q whose
        // peer p (A2AGemm (rank + 1 + q) mod world, GemmA2A q) is not a.rank: ready to
        // entry[p] of tails[rank], entryOut to entry[rank] of tails[p] and, A2AGemm, pulled to
        // pulled[rank] of tails[p].
        __host__ inline void viewWireTails(ViewArmArgs& a, uint32_t* const* tails)
        {
            a.tail = tails[a.rank];
            for(uint32_t q = 0; q < a.numQueues; q++)
            {
                const uint32_t p
                    = a.mode == VIEW_MODE_A2A_GEMM ? (a.rank + 1 + q) % a.world : q;
                if(p == a.rank)
                    continue;
                a.queues[q].ready    = tails[a.rank] + VIEW_TAIL_ENTRY_OFFSET / sizeof(uint32_t) + p;
                a.queues[q].entryOut = tails[p] + VIEW_TAIL_ENTRY_OFFSET / sizeof(uint32_t) + a.rank;
                if(a.mode == VIEW_MODE_A2A_GEMM)
                    a.queues[q].pulled
                        = tails[p] + VIEW_TAIL_PULLED_OFFSET / sizeof(uint32_t) + a.rank;
            }
        }

        // Queue q's chain for the launch whose view-block epoch is `epoch`, without the
        // VIEW_DEBUG_TIMELINE TIMESTAMPs.
        // A2AGemm: for pair t = 0..NB*G-1 with a non-empty group [lo, hi) (viewA2AGemmGroup), for
        // h = 0..S-1, sub (t, h) copies units (b, h), b = lo..hi-1, with one ATOMIC per
        // `coalesce` units and one after its last unit, which brings the flag to Qb * (S*t + h + 1),
        // or for h = S-1 to Qb * S * t', t' the next non-empty pair (NB * G * Qb * S when none).
        // The first ATOMIC also carries the Qb * S of every pair before the first non-empty one.
        // viewPulled's ATOMIC(+1) directly precedes the last ATOMIC; under VIEW_DEBUG_DELAY_CHAIN
        // queue 0 of a.delayRank puts a.delayCopies COPYs from delayScratch to delayScratch +
        // srcBlockStride, with the last unit's geometry, before the last unit's COPY.
        // GemmA2A: no POLL under VIEW_DEBUG_NO_POLL.
        // The bits of VIEW_DEBUG_COPY_FAULTS in both a.debug and Faults change queue 0's first COPY.
        template <uint32_t Faults = VIEW_DEBUG_COPY_FAULTS>
        __host__ __device__ inline void
            viewBuildUnits(ViewArmArgs const& a, uint32_t q, uint32_t epoch, ViewRingWriter& w)
        {
            ViewQueueArgs const& qa     = a.queues[q];
            const uint64_t       flag   = viewAddress(qa.flag);
            uint32_t* const      pulled = viewPulled(a, q);
            const bool delay = (a.debug & VIEW_DEBUG_DELAY_CHAIN) != 0 && q == 0
                               && a.rank == a.delayRank && a.delayScratch != 0;
            const uint32_t       fault = q == 0 ? a.debug & Faults & VIEW_DEBUG_COPY_FAULTS : 0u;
            bool                 first = true, held = false;
            uint64_t             heldSrc = 0, heldDst = 0;
            uint32_t             heldRectY = 0;
            auto                 copy      = [&](uint64_t src, uint64_t dst, uint32_t rectY) {
                if(first && fault != 0)
                {
                    first = false;
                    if((fault & VIEW_DEBUG_DROP_COPY) == 0)
                    {
                        held      = true;
                        heldSrc   = src;
                        heldDst   = dst;
                        heldRectY = rectY;
                    }
                    return;
                }
                first = false;
                viewPutCopy(w, src, dst, a.srcPitch, a.dstPitch, a.slice, a.rectX, rectY);
            };
            auto putHeld = [&]() {
                if(held)
                    viewPutCopy(
                        w, heldSrc, heldDst, a.srcPitch, a.dstPitch, a.slice, a.rectX, heldRectY);
                held = false;
            };
            if(a.mode == VIEW_MODE_A2A_GEMM)
            {
                const uint32_t round = viewA2AGemmRoundUnits(a);
                const uint32_t qb    = round / a.subRounds;
                const uint32_t pairs = viewA2AGemmBatches(a) * a.numQueues;
                uint32_t       lo = 0, hi = 0, t = 0;
                while(t < pairs && !viewA2AGemmGroup(a, q, t, lo, hi))
                    t++;
                uint32_t added = 0;
                uint32_t lead  = t * round;
                while(t < pairs)
                {
                    uint32_t nLo = 0, nHi = 0, next = t + 1;
                    while(next < pairs && !viewA2AGemmGroup(a, q, next, nLo, nHi))
                        next++;
                    for(uint32_t h = 0; h < a.subRounds; h++)
                    {
                        const uint32_t target
                            = h + 1 < a.subRounds ? qb * (a.subRounds * t + h + 1) : next * round;
                        uint32_t pending = 0;
                        for(uint32_t b = lo; b < hi; b++)
                        {
                            const bool last
                                = b + 1 == hi && h + 1 == a.subRounds && next == pairs;
                            for(uint32_t k = 0; delay && last && k < a.delayCopies; k++)
                                viewPutCopy(w,
                                            a.delayScratch,
                                            a.delayScratch + a.srcBlockStride,
                                            a.srcPitch,
                                            a.dstPitch,
                                            a.slice,
                                            a.rectX,
                                            viewRectY(a, b));
                            copy(qa.srcBase + b * a.srcBlockStride + h * a.subRoundStride,
                                 qa.dstBase + b * a.dstBlockStride + h * a.subRoundStride,
                                 viewRectY(a, b));
                            ++pending;
                            if(b + 1 == hi)
                            {
                                if(last)
                                {
                                    putHeld();
                                    if(pulled != nullptr)
                                        viewPutAtomic(w, viewAddress(pulled), 1u);
                                }
                                viewPutAtomic(w, flag, target - added);
                                added = target;
                                lead  = 0;
                            }
                            else if(pending == a.coalesce)
                            {
                                viewPutAtomic(w, flag, pending + lead);
                                added += pending + lead;
                                lead    = 0;
                                pending = 0;
                            }
                        }
                    }
                    t  = next;
                    lo = nLo;
                    hi = nHi;
                }
                return;
            }
            uint32_t* const sig
                = a.view + FUSED_A2A_VIEW_SIG_OFFSET / sizeof(uint32_t) + q * a.tokenBlocks;
            const uint32_t ref = epoch * a.tilesPerRank + a.tilesPerRank;
            for(uint32_t j = 0; j < a.tokenBlocks; j++)
            {
                if((a.debug & VIEW_DEBUG_NO_POLL) == 0)
                    viewPutPoll(w, viewAddress(sig + j), ref);
                copy(qa.srcBase + j * a.srcBlockStride,
                     qa.dstBase + j * a.dstBlockStride,
                     viewRectY(a, j));
                viewPutAtomic(w, flag, 1u);
            }
            putHeld();
            viewPutAtomic(
                w, viewAddress(a.view + FUSED_A2A_VIEW_OUTBOUND_OFFSET / sizeof(uint32_t)), 1u);
        }

        // viewBuildUnits<Faults> between a leading and a trailing TIMESTAMP when a.stats is set.
        // A chain with a unit and a viewReady word first POLLs that word for >= epoch + 1,
        // followed by a VIEW_STATS_TS_READY TIMESTAMP when a.stats is set.
        template <uint32_t Faults = VIEW_DEBUG_COPY_FAULTS>
        __host__ __device__ inline void
            viewBuildChain(ViewArmArgs const& a, uint32_t q, uint32_t epoch, ViewRingWriter& w)
        {
            if(a.stats != nullptr)
                viewPutTimestamp(
                    w, viewAddress(a.stats + viewStatsIndex(epoch, VIEW_STATS_TS_START + 2 * q)));
            uint32_t* const ready = viewReady(a, q);
            if(ready != nullptr && viewChainHasUnit(a, q))
            {
                viewPutPoll(w, viewAddress(ready), epoch + 1u);
                if(a.stats != nullptr)
                    viewPutTimestamp(
                        w, viewAddress(a.stats + viewStatsIndex(epoch, VIEW_STATS_TS_READY + q)));
            }
            viewBuildUnits<Faults>(a, q, epoch, w);
            // Follows the last ATOMIC of an A2AGemm chain; it touches no counter.
            if(a.stats != nullptr)
                viewPutTimestamp(
                    w, viewAddress(a.stats + viewStatsIndex(epoch, VIEW_STATS_TS_END + 2 * q)));
        }

        // Whether queue q's chain fits between *wptr and *rptr + ringBytes.
        template <uint32_t Faults = VIEW_DEBUG_COPY_FAULTS>
        __host__ __device__ inline bool
            viewQueueFits(ViewArmArgs const& a, uint32_t q, uint32_t epoch)
        {
            ViewQueueArgs const& qa   = a.queues[q];
            const uint64_t       wptr = viewLoad64(qa.wptr);
            ViewRingWriter w{nullptr, qa.ringBytes / sizeof(uint32_t), wptr / sizeof(uint32_t)};
            viewBuildChain<Faults>(a, q, epoch, w);
            return w.pos * sizeof(uint32_t) - viewLoad64(qa.rptr) < qa.ringBytes;
        }

        // Appends queue q's chain after *wptr, then stores *wptr and the doorbell. Returns
        // false, writing nothing, when the chain does not fit in front of *rptr.
        template <uint32_t Faults = VIEW_DEBUG_COPY_FAULTS>
        __host__ __device__ inline bool
            viewArmQueue(ViewArmArgs const& a, uint32_t q, uint32_t epoch)
        {
            if(!viewQueueFits<Faults>(a, q, epoch))
                return false;
            ViewQueueArgs const& qa = a.queues[q];
            ViewRingWriter       w{
                qa.ring, qa.ringBytes / sizeof(uint32_t), viewLoad64(qa.wptr) / sizeof(uint32_t)};
            viewBuildChain<Faults>(a, q, epoch, w);
            const uint64_t end = w.pos * sizeof(uint32_t);
            viewPublish64(qa.wptr, end);
            viewPublish64(qa.doorbell, end);
            return true;
        }

        // No queue is armed once the view block's error word is non-zero.
        __host__ __device__ inline bool viewArmHalted(uint32_t error)
        {
            return error != 0;
        }

        __host__ __device__ inline bool viewQueueDrained(ViewArmArgs const& a, uint32_t q)
        {
            return viewLoad64(a.queues[q].rptr) == viewLoad64(a.queues[q].wptr);
        }

        // A2AGemm: flag[q] := 0 through store(u32*, value), after reading whether queue q is
        // drained; returns VIEW_ERROR_STUCK when it is not. GemmA2A: stores nothing, returns 0.
        template <typename Store>
        __host__ __device__ inline uint32_t
            viewResetFlag(ViewArmArgs const& a, uint32_t q, Store&& store)
        {
            if(a.mode != VIEW_MODE_A2A_GEMM)
                return 0u;
            const uint32_t stuck = viewQueueDrained(a, q) ? 0u : VIEW_ERROR_STUCK;
            store(a.queues[q].flag, 0u);
            return stuck;
        }

        // Adds what queue q's chain would have added, through add(u32*, addend): A2AGemm own
        // flag[q] += NB*G*Qb*S and viewPulled += 1; GemmA2A peer flag[rank] += T and own
        // outbound += 1.
        template <typename Add>
        __host__ __device__ inline void viewBookSkipped(ViewArmArgs const& a, uint32_t q, Add&& add)
        {
            if(a.mode == VIEW_MODE_A2A_GEMM)
            {
                add(a.queues[q].flag, viewA2AGemmChainUnits(a));
                if(uint32_t* const pulled = viewPulled(a, q))
                    add(pulled, 1u);
                return;
            }
            add(a.queues[q].flag, a.tokenBlocks);
            add(a.view + FUSED_A2A_VIEW_OUTBOUND_OFFSET / sizeof(uint32_t), 1u);
        }

        // Armed queue q: adds 1 to viewPulled through add(u32*, addend) when its chain has no
        // unit.
        template <typename Add>
        __host__ __device__ inline void viewBookUnitFree(ViewArmArgs const& a, uint32_t q, Add&& add)
        {
            uint32_t* const pulled = viewPulled(a, q);
            if(pulled != nullptr && !viewChainHasUnit(a, q))
                add(pulled, 1u);
        }

        // viewResetFlag, then arms queue q, or books it through viewBookSkipped when the arm
        // is halted, the queue is stuck or the chain does not fit. An armed queue gets
        // viewBookUnitFree. Returns the error bits to set.
        template <uint32_t Faults = VIEW_DEBUG_COPY_FAULTS, typename Add, typename Store>
        __host__ __device__ inline uint32_t viewArmOrBook(ViewArmArgs const& a,
                                                          uint32_t           q,
                                                          uint32_t           epoch,
                                                          uint32_t           error,
                                                          Add&&              add,
                                                          Store&&            store)
        {
            const bool     halted = viewArmHalted(error);
            const uint32_t stuck  = viewResetFlag(a, q, store);
            if(!halted && stuck == 0 && viewArmQueue<Faults>(a, q, epoch))
            {
                viewBookUnitFree(a, q, add);
                return 0u;
            }
            viewBookSkipped(a, q, add);
            return halted || stuck != 0 ? stuck : VIEW_ERROR_RING_FULL;
        }

        // Stores epoch + 1 into queue q's entryOut through release(u32*, value), unless
        // VIEW_DEBUG_NO_READY.
        template <typename Release>
        __host__ __device__ inline void
            viewPublishEntry(ViewArmArgs const& a, uint32_t q, uint32_t epoch, Release&& release)
        {
            if((a.debug & VIEW_DEBUG_NO_READY) == 0 && a.queues[q].entryOut != nullptr)
                release(a.queues[q].entryOut, epoch + 1u);
        }

        // Queue q's part of the arm kernel for the launch at `epoch`, with `error` read before
        // it: viewPublishEntry, then viewArmOrBook; bookOnly books as a halted arm does.
        template <uint32_t Faults = VIEW_DEBUG_COPY_FAULTS,
                  typename Add,
                  typename Store,
                  typename Release>
        __host__ __device__ inline uint32_t viewArmThread(ViewArmArgs const& a,
                                                          uint32_t           q,
                                                          uint32_t           epoch,
                                                          uint32_t           error,
                                                          bool               bookOnly,
                                                          Add&&              add,
                                                          Store&&            store,
                                                          Release&&          release)
        {
            viewPublishEntry(a, q, epoch, release);
            return viewArmOrBook<Faults>(
                a, q, epoch, bookOnly ? error | VIEW_ERROR_ABORTED : error, add, store);
        }

        // The host arm's queues for the launch at `epoch`, with `error` read before it:
        // viewPublishEntry of every queue through store; then per queue viewResetFlag and
        // viewBookUnitFree when it fits, else viewBookSkipped; then viewArmQueue of every
        // queue that fits. Returns the error bits to set.
        template <uint32_t Faults = VIEW_DEBUG_COPY_FAULTS, typename Add, typename Store>
        __host__ inline uint32_t viewArmHostQueues(
            ViewArmArgs const& a, uint32_t epoch, uint32_t error, Add&& add, Store&& store)
        {
            const bool halted = viewArmHalted(error);
            bool       fits[FUSED_A2A_MAX_RANKS];
            uint32_t   bits = 0;
            for(uint32_t q = 0; q < a.numQueues; q++)
                viewPublishEntry(a, q, epoch, store);
            for(uint32_t q = 0; q < a.numQueues; q++)
            {
                const uint32_t stuck = viewResetFlag(a, q, store);
                fits[q] = !halted && stuck == 0 && viewQueueFits<Faults>(a, q, epoch);
                if(fits[q])
                {
                    viewBookUnitFree(a, q, add);
                    continue;
                }
                bits |= halted || stuck != 0 ? stuck : VIEW_ERROR_RING_FULL;
                viewBookSkipped(a, q, add);
            }
            for(uint32_t q = 0; q < a.numQueues; q++)
                if(fits[q])
                    viewArmQueue<Faults>(a, q, epoch);
            return bits;
        }

        // A solution's launch count stays below it: VIEW_EPOCH_LIMIT, and for GemmA2A
        // 0xFFFFFFFF / tilesPerRank.
        __host__ __device__ inline uint64_t viewLaunchLimit(ViewArmArgs const& a)
        {
            if(a.mode != VIEW_MODE_GEMM_A2A || a.tilesPerRank == 0)
                return VIEW_EPOCH_LIMIT;
            const uint64_t sig = 0xFFFFFFFFull / a.tilesPerRank;
            return sig < VIEW_EPOCH_LIMIT ? sig : VIEW_EPOCH_LIMIT;
        }

        // Whether the launch's A2AGemm exit wait runs.
        __host__ __device__ inline bool viewExitWaits(ViewArmArgs const& a)
        {
            return a.mode == VIEW_MODE_A2A_GEMM && (a.debug & VIEW_DEBUG_NO_EXITWAIT) == 0
                   && a.world > 1;
        }

        // A2AGemm with viewExitWaits, header epoch `epoch`: whether the tail's released word
        // is non-zero or (int32)(pulled[r] - epoch) >= 0 for every r != rank. Otherwise true.
        __host__ __device__ inline bool viewExitDone(ViewArmArgs const& a, uint32_t epoch)
        {
            if(!viewExitWaits(a) || viewLoad32(a.tail + VIEW_TAIL_RELEASED_OFFSET / 4) != 0)
                return true;
            uint32_t* const pulled = a.tail + VIEW_TAIL_PULLED_OFFSET / sizeof(uint32_t);
            for(uint32_t r = 0; r < a.world; r++)
                if(r != a.rank && (int32_t)(viewLoad32(pulled + r) - epoch) < 0)
                    return false;
            return true;
        }

        // GemmA2A: whether every own chain and every peer's chain into this rank finished the
        // launch whose base and outboundBase the header holds.
        __host__ __device__ inline bool viewGemmA2ADone(ViewArmArgs const& a)
        {
            uint32_t* const view = a.view;
            uint32_t* const flag = view + FUSED_A2A_VIEW_FLAG_OFFSET / sizeof(uint32_t);
            uint32_t* const base = view + FUSED_A2A_VIEW_BASE_OFFSET / sizeof(uint32_t);
            bool            done
                = viewLoad32(view + FUSED_A2A_VIEW_OUTBOUND_OFFSET / sizeof(uint32_t))
                      - viewLoad32(view + FUSED_A2A_VIEW_OUTBOUND_BASE_OFFSET / sizeof(uint32_t))
                  >= a.world;
            for(uint32_t s = 0; done && s < a.world; s++)
                done = viewLoad32(flag + s) - viewLoad32(base + s) >= a.tokenBlocks;
            return done;
        }

        // The wait kernel's exit condition at header epoch `epoch`.
        __host__ __device__ inline bool viewWaitDone(ViewArmArgs const& a, uint32_t epoch)
        {
            return a.mode == VIEW_MODE_GEMM_A2A ? viewGemmA2ADone(a) : viewExitDone(a, epoch);
        }

        // Raises own entry[x], x != rank, to `epoch` through max(u32*, value).
        template <typename Max>
        __host__ __device__ inline void
            viewReleaseEntry(ViewArmArgs const& a, uint32_t epoch, Max&& max)
        {
            for(uint32_t x = 0; x < a.world; x++)
                if(x != a.rank)
                    max(a.tail + VIEW_TAIL_ENTRY_OFFSET / sizeof(uint32_t) + x, epoch);
        }

        // Whether a view-block tail of rank `rank` at header epoch `epoch` holds entry[x] ==
        // epoch for every x != rank (checkEntry) and, A2AGemm, pulled[r] == epoch for every
        // r != rank (checkPulled).
        inline bool viewTailCountsMatch(uint32_t        mode,
                                        uint32_t        rank,
                                        uint32_t        world,
                                        uint32_t        epoch,
                                        uint32_t const* tail,
                                        bool            checkEntry,
                                        bool            checkPulled)
        {
            for(uint32_t x = 0; x < world; x++)
            {
                if(x == rank)
                    continue;
                if(checkEntry && tail[VIEW_TAIL_ENTRY_OFFSET / sizeof(uint32_t) + x] != epoch)
                    return false;
                if(checkPulled && mode == VIEW_MODE_A2A_GEMM
                   && tail[VIEW_TAIL_PULLED_OFFSET / sizeof(uint32_t) + x] != epoch)
                    return false;
            }
            return true;
        }

        // The words of the view block a header copy holds.
        constexpr uint32_t kViewHeaderWords = FUSED_A2A_VIEW_SIG_OFFSET / sizeof(uint32_t);

        // The tail words the validator compares with the header epoch.
        struct ViewTailChecks
        {
            bool entry  = false;
            bool pulled = false;
        };

        // entry unless VIEW_DEBUG_NO_READY, pulled unless VIEW_DEBUG_NO_EXITWAIT.
        inline ViewTailChecks viewTailChecks(uint32_t debug)
        {
            return {(debug & VIEW_DEBUG_NO_READY) == 0, (debug & VIEW_DEBUG_NO_EXITWAIT) == 0};
        }

        // The validator's readiness check of `world` ranks: rank d's header copy at
        // headers + d * kViewHeaderWords, its tail at tails + d * VIEW_TAIL_BYTES / 4. Returns
        // false, checking nothing, when an error word is non-zero; otherwise appends to `bad`
        // every rank that fails viewTailCountsMatch at its header epoch, and returns true.
        inline bool viewCheckTails(uint32_t               mode,
                                   uint32_t               world,
                                   uint32_t const*        headers,
                                   uint32_t const*        tails,
                                   ViewTailChecks         checks,
                                   std::vector<uint32_t>& bad)
        {
            for(uint32_t d = 0; d < world; d++)
                if(headers[d * kViewHeaderWords + FUSED_A2A_VIEW_ERROR_OFFSET / sizeof(uint32_t)]
                   != 0)
                    return false;
            for(uint32_t d = 0; d < world; d++)
                if(!viewTailCountsMatch(
                       mode,
                       d,
                       world,
                       headers[d * kViewHeaderWords + FUSED_A2A_VIEW_EPOCH_OFFSET / sizeof(uint32_t)],
                       tails + d * (VIEW_TAIL_BYTES / sizeof(uint32_t)),
                       checks.entry,
                       checks.pulled))
                    bad.push_back(d);
            return true;
        }

        // --fused-a2a-view-throw-at sites, in launchAll's order.
        enum class ViewThrowSite
        {
            Start,
            Arm,
            Launch,
            Wait,
            Stop,
            Check
        };

        struct ViewThrowAt
        {
            bool          set    = false;
            ViewThrowSite site   = ViewThrowSite::Start;
            uint32_t      device = 0;
            uint64_t      launch = 0;
        };

        // "<site>:<device>:<launch>", site one of start, arm, launch, wait, stop, check; empty
        // gives set = false. Throws std::invalid_argument on anything else.
        inline ViewThrowAt viewParseThrowAt(std::string const& text)
        {
            ViewThrowAt t;
            if(text.empty())
                return t;
            static const char* const names[]
                = {"start", "arm", "launch", "wait", "stop", "check"};
            const size_t c1 = text.find(':');
            const size_t c2 = c1 == std::string::npos ? c1 : text.find(':', c1 + 1);
            auto number     = [&](std::string const& s) {
                if(s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
                    throw std::invalid_argument("--fused-a2a-view-throw-at: bad number '" + s
                                                + "' in '" + text + "'");
                return std::stoull(s);
            };
            if(c2 == std::string::npos)
                throw std::invalid_argument(
                    "--fused-a2a-view-throw-at must be <site>:<device>:<launch>; got '" + text
                    + "'");
            const std::string site = text.substr(0, c1);
            for(int i = 0; i < 6; i++)
                if(site == names[i])
                {
                    t.set    = true;
                    t.site   = (ViewThrowSite)i;
                    t.device = (uint32_t)number(text.substr(c1 + 1, c2 - c1 - 1));
                    t.launch = number(text.substr(c2 + 1));
                    return t;
                }
            throw std::invalid_argument("--fused-a2a-view-throw-at: unknown site '" + site + "'");
        }

        // Whether t is set and matches (site, device, launch); a match clears t.set.
        inline bool viewThrowNow(ViewThrowAt& t, ViewThrowSite site, uint32_t device, uint64_t launch)
        {
            if(!t.set || t.site != site || t.device != device || t.launch != launch)
                return false;
            t.set = false;
            return true;
        }

        // Stores epoch * tilesPerRank into sig[0, W*T) of a GemmA2A view block whose header
        // epoch is `epoch`: the threshold of that epoch's POLLs.
        __host__ __device__ inline void viewReleaseSig(ViewArmArgs const& a, uint32_t epoch)
        {
            uint32_t* const sig = a.view + FUSED_A2A_VIEW_SIG_OFFSET / sizeof(uint32_t);
            for(uint32_t i = 0; i < a.world * a.tokenBlocks; i++)
                viewStore32(sig + i, epoch * a.tilesPerRank);
        }

        __host__ __device__ inline bool viewQueuesDrained(ViewArmArgs const& a)
        {
            for(uint32_t q = 0; q < a.numQueues; q++)
                if(!viewQueueDrained(a, q))
                    return false;
            return true;
        }

        // GemmA2A: base/expected and outboundBase for the launch at `epoch`. Then epoch + 1.
        // On the u32 header at `header` (the view block, or a host copy of its first
        // FUSED_A2A_VIEW_SIG_OFFSET bytes).
        __host__ __device__ inline void
            viewAdvance(ViewArmArgs const& a, uint32_t epoch, uint32_t* header)
        {
            if(a.mode == VIEW_MODE_GEMM_A2A)
            {
                uint32_t* const base = header + FUSED_A2A_VIEW_BASE_OFFSET / sizeof(uint32_t);
                uint32_t* const expected
                    = header + FUSED_A2A_VIEW_EXPECTED_OFFSET / sizeof(uint32_t);
                for(uint32_t s = 0; s < a.world; s++)
                {
                    const uint32_t next = viewLoad32(expected + s);
                    viewStore32(base + s, next);
                    viewStore32(expected + s, next + a.tokenBlocks);
                }
                viewStore32(header + FUSED_A2A_VIEW_OUTBOUND_BASE_OFFSET / sizeof(uint32_t),
                            epoch * a.world);
            }
            viewStore32(header + FUSED_A2A_VIEW_EPOCH_OFFSET / sizeof(uint32_t), epoch + 1u);
        }

        // The sign bit of every element of a u32 word of elementBytes-byte floats (1, 2 or 4).
        __host__ __device__ inline uint32_t viewSignMask(uint32_t elementBytes)
        {
            return elementBytes == 1 ? 0x80808080u : elementBytes == 2 ? 0x80008000u : 0x80000000u;
        }

        // Whether `word` equals `golden` per element of sign bit `mask`; an element that is +0 or
        // -0 in both matches.
        __host__ __device__ inline bool
            viewVerifyCompare(uint32_t word, uint32_t golden, uint32_t mask)
        {
            const uint32_t diff = word ^ golden;
            if(diff == 0)
                return true;
            const uint32_t bits = mask == 0x80808080u ? 8u : mask == 0x80008000u ? 16u : 32u;
            for(uint32_t lo = 0; lo < 32; lo += bits)
            {
                const uint32_t lane = (bits == 32 ? 0xFFFFFFFFu : (1u << bits) - 1u) << lo;
                const uint32_t sign = 1u << (lo + bits - 1);
                const uint32_t d    = diff & lane;
                if(d != 0 && (d != sign || (word & lane & ~sign) != 0))
                    return false;
            }
            return true;
        }

        // The host release of a view-block tail by `path`: VIEW_RELEASE_HOST_STORE stores
        // VIEW_TAIL_RELEASED into each of the first VIEW_TAIL_RELEASE_WORDS words through
        // store(u32*, value); VIEW_RELEASE_MEMCPY returns copy(tail, VIEW_TAIL_RELEASE_BYTES) of
        // a viewFillReleaseSource buffer; otherwise returns fill(tail, VIEW_TAIL_RELEASE_WORDS,
        // VIEW_TAIL_RELEASED).
        template <typename Store, typename Copy, typename Fill>
        inline bool
            viewReleaseTail(uint32_t* tail, int path, Store&& store, Copy&& copy, Fill&& fill)
        {
            if(path == VIEW_RELEASE_HOST_STORE)
            {
                for(uint32_t i = 0; i < VIEW_TAIL_RELEASE_WORDS; i++)
                    store(tail + i, VIEW_TAIL_RELEASED);
                return true;
            }
            if(path == VIEW_RELEASE_MEMCPY)
                return copy(tail, (size_t)VIEW_TAIL_RELEASE_BYTES);
            return fill(tail, (size_t)VIEW_TAIL_RELEASE_WORDS, VIEW_TAIL_RELEASED);
        }

        // VIEW_TAIL_RELEASE_WORDS words of VIEW_TAIL_RELEASED at `source`.
        inline void viewFillReleaseSource(void* source)
        {
            for(uint32_t i = 0; i < VIEW_TAIL_RELEASE_WORDS; i++)
                static_cast<uint32_t*>(source)[i] = VIEW_TAIL_RELEASED;
        }

        // Calls query() until it returns other than hipErrorNotReady and returns that, with
        // sleep() between calls; hipErrorNotReady once now() is past `deadline`.
        template <typename Query, typename Now, typename Sleep, typename TimePoint>
        inline hipError_t viewPollIdle(Query&& query, Now&& now, Sleep&& sleep, TimePoint deadline)
        {
            while(true)
            {
                const hipError_t e = query();
                if(e != hipErrorNotReady)
                    return e;
                if(now() > deadline)
                    return hipErrorNotReady;
                sleep();
            }
        }

        // viewPollIdle until now() + bound, then sync() when it returned hipSuccess.
        template <typename Query, typename Sync, typename Now, typename Sleep, typename Duration>
        inline hipError_t
            viewSyncWithin(Query&& query, Sync&& sync, Now&& now, Sleep&& sleep, Duration bound)
        {
            const hipError_t e = viewPollIdle(query, now, sleep, now() + bound);
            return e != hipSuccess ? e : sync();
        }

        // Fills the arm of devices[rank] for launch `launch` (its rotating slot), with
        // today's in-kernel COPY geometry: a2aEnqueue (A2AGemm) and
        // _fusedA2AComputeCopyFields (GemmA2A).
        ViewArmArgs makeViewArmArgs(ContractionProblem const&                          problem,
                                    ContractionSolution const&                         solution,
                                    std::vector<std::shared_ptr<DeviceContext>> const& devices,
                                    size_t                                             rank,
                                    size_t                                             launch,
                                    uint32_t                                           coalesce,
                                    Hardware const&                                    hardware,
                                    int                                                debug       = 0,
                                    uint32_t                                           delayRank   = 1,
                                    uint32_t                                           delayCopies = 32);

        // On `stream` of the current device. arm is VIEW_ARM_DEVICE or VIEW_ARM_HOST; the host
        // arm synchronizes `stream` first.
        void viewArm(ViewArmArgs const& args, hipStream_t stream, int arm);

        // The wait kernel: GemmA2A, or A2AGemm with viewExitWaits or args.stats. With
        // args.stats it stamps the launch's VIEW_STATS_GEMM_END and VIEW_STATS_WAIT_OUT.
        void viewWait(ViewArmArgs const& args, hipStream_t stream);

        // For a launch on `stream` whose per-device body threw. armed: sets VIEW_ERROR_ABORTED;
        // GemmA2A also releases sig[] with viewReleaseSig and own entry[] with viewReleaseEntry
        // and waits, bounded, for the own queues to drain. Not armed: the arm kernel in
        // book-only mode (viewArmThread). Errors are ignored.
        void viewAbort(ViewArmArgs const& args, hipStream_t stream, bool armed) noexcept;

        // When ctx.viewQueuesArmed: viewReleaseTail of ctx's view-block tail by `path`
        // (VIEW_RELEASE_*), not ordered behind ctx.stream. Returns false on any error.
        bool viewReleaseTailHost(DeviceContext& ctx, int path) noexcept;

        // Waits up to kViewSyncBoundMs for ctx.stream and ctx.viewReleaseStream to finish, then
        // synchronizes the device. hipErrorNotReady when either is still busy.
        hipError_t viewSyncBounded(DeviceContext& ctx) noexcept;

        // When ctx.viewQueuesArmed: raises every word of ctx's view block from sig[] on to
        // 0xFFFFFFFF if a queue is not drained, then waits up to kViewWaitTimeoutMs for every
        // ctx.sdmaQueues entry to have *rptr == *wptr and clears ctx.viewQueuesArmed. Returns
        // false when a queue is still not drained or on any error.
        bool viewDrainQueues(DeviceContext& ctx) noexcept;

        // An empty single-thread kernel on `stream`.
        void viewNoop(hipStream_t stream);

        // The VIEWSTAT line of one device's VIEW_STATS_BYTES ring, without a newline.
        std::string viewStatsLine(int device, uint64_t const* ring);
    } // namespace Client
} // namespace TensileLite
