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
        constexpr int VIEW_DEBUG_ALL            = 7;

        // VIEW_DEBUG_TIMELINE ring: record (epoch % VIEW_STATS_RECORDS) of VIEW_STATS_FIELDS
        // u64 slots, one slot per VIEW_STATS_SLOT_U64 u64s (32 B). Fields are wall-clock ticks.
        constexpr uint32_t VIEW_STATS_RECORDS  = 64;
        constexpr uint32_t VIEW_STATS_SLOT_U64 = 4;
        constexpr uint32_t VIEW_STATS_ARM_IN   = 0;
        constexpr uint32_t VIEW_STATS_ARM_OUT  = 1;
        constexpr uint32_t VIEW_STATS_GEMM_END = 2;
        constexpr uint32_t VIEW_STATS_WAIT_OUT = 3;
        // Queue q: VIEW_STATS_TS_START + 2q and VIEW_STATS_TS_END + 2q.
        constexpr uint32_t VIEW_STATS_TS_START = 4;
        constexpr uint32_t VIEW_STATS_TS_END   = 5;
        constexpr uint32_t VIEW_STATS_FIELDS   = 4 + 2 * FUSED_A2A_MAX_RANKS;
        constexpr size_t   VIEW_STATS_BYTES
            = (size_t)VIEW_STATS_RECORDS * VIEW_STATS_FIELDS * VIEW_STATS_SLOT_U64 * sizeof(uint64_t);
        constexpr uint32_t VIEW_STATS_TICKS_PER_US = 100;

        constexpr uint64_t kViewWaitTimeoutMs = 5000;

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

        // A2AGemm: nb = clamp(numCu / F, 1, B) token blocks per dispatch batch.
        __host__ __device__ inline uint32_t viewA2AGemmBatchBlocks(ViewArmArgs const& a)
        {
            const uint32_t nb = a.numCu / a.featureTiles;
            const uint32_t lo = nb < a.tokenBlocks ? nb : a.tokenBlocks;
            return lo > 1 ? lo : 1;
        }

        // A2AGemm: NB = ceil(B / nb) dispatch batches.
        __host__ __device__ inline uint32_t viewA2AGemmBatches(ViewArmArgs const& a)
        {
            const uint32_t nb = viewA2AGemmBatchBlocks(a);
            return (a.tokenBlocks + nb - 1) / nb;
        }

        // A2AGemm: Qb * S, Qb = ceil(nb / G), G = numQueues; 0 when G = 0. Each of the NB * G
        // (batch, round) pairs of a queue's chain adds this much to its flag.
        __host__ __device__ inline uint32_t viewA2AGemmRoundUnits(ViewArmArgs const& a)
        {
            if(a.numQueues == 0)
                return 0;
            return (viewA2AGemmBatchBlocks(a) + a.numQueues - 1) / a.numQueues * a.subRounds;
        }

        // A2AGemm: NB * G * Qb * S, what one launch adds to each flag[i].
        __host__ __device__ inline uint32_t viewA2AGemmChainUnits(ViewArmArgs const& a)
        {
            return viewA2AGemmBatches(a) * a.numQueues * viewA2AGemmRoundUnits(a);
        }

        // A2AGemm: the first token block of batch iB served by queue q in round j, the first
        // b in [iB * nb, ...) with b mod G == (q - j) mod G.
        __host__ __device__ inline uint32_t
            viewA2AGemmFirstBlock(ViewArmArgs const& a, uint32_t q, uint32_t iB, uint32_t j)
        {
            const uint32_t g   = a.numQueues;
            const uint32_t bLo = iB * viewA2AGemmBatchBlocks(a);
            return bLo + ((q + g - j) % g + g - bLo % g) % g;
        }

        // Queue q's chain for the launch whose view-block epoch is `epoch`, without the
        // VIEW_DEBUG_TIMELINE TIMESTAMPs.
        // A2AGemm: for batch iB = 0..NB-1 (blocks [iB * nb, min(iB * nb + nb, B))), round
        // j = 0..G-1 copies the batch's token blocks b with b mod G == (q - j) mod G ascending,
        // each for h = 0..S-1, with one ATOMIC per `coalesce` units. The ATOMIC after the last
        // unit of pair t = iB * G + j brings the flag to (t+1) * Qb * S; after the chain's last
        // unit, to NB * G * Qb * S, so no ATOMIC follows it. A pair without a block before that
        // emits one ATOMIC of Qb * S. GemmA2A: no POLL under VIEW_DEBUG_NO_POLL.
        __host__ __device__ inline void
            viewBuildUnits(ViewArmArgs const& a, uint32_t q, uint32_t epoch, ViewRingWriter& w)
        {
            ViewQueueArgs const& qa   = a.queues[q];
            const uint64_t       flag = viewAddress(qa.flag);
            if(a.mode == VIEW_MODE_A2A_GEMM)
            {
                const uint32_t g     = a.numQueues;
                const uint32_t nb    = viewA2AGemmBatchBlocks(a);
                const uint32_t round = viewA2AGemmRoundUnits(a);
                const uint32_t pairs = viewA2AGemmBatches(a) * g;
                if(pairs == 0)
                    return;
                uint32_t last = pairs - 1;
                while(viewA2AGemmFirstBlock(a, q, last / g, last % g) >= a.tokenBlocks)
                    last--;
                uint32_t pending = 0;
                uint32_t added   = 0;
                for(uint32_t t = 0; t <= last; t++)
                {
                    const uint32_t target = t == last ? pairs * round : (t + 1) * round;
                    const uint32_t bEnd   = t / g * nb + nb;
                    const uint32_t bHi    = bEnd < a.tokenBlocks ? bEnd : a.tokenBlocks;
                    for(uint32_t b = viewA2AGemmFirstBlock(a, q, t / g, t % g); b < bHi; b += g)
                        for(uint32_t h = 0; h < a.subRounds; h++)
                        {
                            viewPutCopy(w,
                                        qa.srcBase + b * a.srcBlockStride + h * a.subRoundStride,
                                        qa.dstBase + b * a.dstBlockStride + h * a.subRoundStride,
                                        a.srcPitch,
                                        a.dstPitch,
                                        a.slice,
                                        a.rectX,
                                        viewRectY(a, b));
                            ++pending;
                            if(b + g >= bHi && h + 1 == a.subRounds)
                            {
                                viewPutAtomic(w, flag, target - added);
                                added   = target;
                                pending = 0;
                            }
                            else if(pending == a.coalesce)
                            {
                                viewPutAtomic(w, flag, pending);
                                added += pending;
                                pending = 0;
                            }
                        }
                    if(added != target)
                    {
                        viewPutAtomic(w, flag, target - added);
                        added = target;
                    }
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
                viewPutCopy(w,
                            qa.srcBase + j * a.srcBlockStride,
                            qa.dstBase + j * a.dstBlockStride,
                            a.srcPitch,
                            a.dstPitch,
                            a.slice,
                            a.rectX,
                            viewRectY(a, j));
                viewPutAtomic(w, flag, 1u);
            }
            viewPutAtomic(
                w, viewAddress(a.view + FUSED_A2A_VIEW_OUTBOUND_OFFSET / sizeof(uint32_t)), 1u);
        }

        // viewBuildUnits between a leading and a trailing TIMESTAMP when a.stats is set.
        __host__ __device__ inline void
            viewBuildChain(ViewArmArgs const& a, uint32_t q, uint32_t epoch, ViewRingWriter& w)
        {
            if(a.stats != nullptr)
                viewPutTimestamp(
                    w, viewAddress(a.stats + viewStatsIndex(epoch, VIEW_STATS_TS_START + 2 * q)));
            viewBuildUnits(a, q, epoch, w);
            // Ends an A2AGemm chain in place of its last ATOMIC; it touches no counter.
            if(a.stats != nullptr)
                viewPutTimestamp(
                    w, viewAddress(a.stats + viewStatsIndex(epoch, VIEW_STATS_TS_END + 2 * q)));
        }

        // Whether queue q's chain fits between *wptr and *rptr + ringBytes.
        __host__ __device__ inline bool
            viewQueueFits(ViewArmArgs const& a, uint32_t q, uint32_t epoch)
        {
            ViewQueueArgs const& qa   = a.queues[q];
            const uint64_t       wptr = viewLoad64(qa.wptr);
            ViewRingWriter w{nullptr, qa.ringBytes / sizeof(uint32_t), wptr / sizeof(uint32_t)};
            viewBuildChain(a, q, epoch, w);
            return w.pos * sizeof(uint32_t) - viewLoad64(qa.rptr) < qa.ringBytes;
        }

        // Appends queue q's chain after *wptr, then stores *wptr and the doorbell. Returns
        // false, writing nothing, when the chain does not fit in front of *rptr.
        __host__ __device__ inline bool
            viewArmQueue(ViewArmArgs const& a, uint32_t q, uint32_t epoch)
        {
            if(!viewQueueFits(a, q, epoch))
                return false;
            ViewQueueArgs const& qa = a.queues[q];
            ViewRingWriter       w{
                qa.ring, qa.ringBytes / sizeof(uint32_t), viewLoad64(qa.wptr) / sizeof(uint32_t)};
            viewBuildChain(a, q, epoch, w);
            const uint64_t end = w.pos * sizeof(uint32_t);
            viewPublish64(qa.wptr, end);
            viewPublish64(qa.doorbell, end);
            return true;
        }

        // GemmA2A arms no queue once the view block's error word is non-zero.
        __host__ __device__ inline bool viewArmHalted(ViewArmArgs const& a, uint32_t error)
        {
            return a.mode == VIEW_MODE_GEMM_A2A && error != 0;
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
        // flag[q] += NB*G*Qb*S; GemmA2A peer flag[rank] += T and own outbound += 1.
        template <typename Add>
        __host__ __device__ inline void viewBookSkipped(ViewArmArgs const& a, uint32_t q, Add&& add)
        {
            if(a.mode == VIEW_MODE_A2A_GEMM)
            {
                add(a.queues[q].flag, viewA2AGemmChainUnits(a));
                return;
            }
            add(a.queues[q].flag, a.tokenBlocks);
            add(a.view + FUSED_A2A_VIEW_OUTBOUND_OFFSET / sizeof(uint32_t), 1u);
        }

        // viewResetFlag, then arms queue q, or books it through viewBookSkipped when the arm
        // is halted, the queue is stuck or the chain does not fit. Returns the error bits to set.
        template <typename Add, typename Store>
        __host__ __device__ inline uint32_t viewArmOrBook(ViewArmArgs const& a,
                                                          uint32_t           q,
                                                          uint32_t           epoch,
                                                          uint32_t           error,
                                                          Add&&              add,
                                                          Store&&            store)
        {
            const bool     halted = viewArmHalted(a, error);
            const uint32_t stuck  = viewResetFlag(a, q, store);
            if(!halted && stuck == 0 && viewArmQueue(a, q, epoch))
                return 0u;
            viewBookSkipped(a, q, add);
            return halted || stuck != 0 ? stuck : VIEW_ERROR_RING_FULL;
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
                                    int                                                debug = 0);

        // On `stream` of the current device. arm is VIEW_ARM_DEVICE or VIEW_ARM_HOST; the host
        // arm synchronizes `stream` first.
        void viewArm(ViewArmArgs const& args, hipStream_t stream, int arm);

        // GemmA2A: the wait kernel. A2AGemm: with args.stats, a kernel that stamps the
        // launch's VIEW_STATS_GEMM_END and VIEW_STATS_WAIT_OUT; otherwise a no-op.
        void viewWait(ViewArmArgs const& args, hipStream_t stream);

        // For a launch whose GEMM was not enqueued after viewArm: on `stream`, sets
        // VIEW_ERROR_ABORTED, releases sig[] with viewReleaseSig and waits, bounded, for the
        // own queues to drain. GemmA2A only; errors are ignored.
        void viewAbort(ViewArmArgs const& args, hipStream_t stream) noexcept;

        // When ctx.viewQueuesArmed: raises every sig[] word of ctx's view block to 0xFFFFFFFF
        // if a queue is not drained, then waits up to kViewWaitTimeoutMs for every
        // ctx.sdmaQueues entry to have *rptr == *wptr and clears ctx.viewQueuesArmed. Returns
        // false when a queue is still not drained or on any error.
        bool viewDrainQueues(DeviceContext& ctx) noexcept;

        // An empty single-thread kernel on `stream`.
        void viewNoop(hipStream_t stream);

        // The VIEWSTAT line of one device's VIEW_STATS_BYTES ring, without a newline.
        std::string viewStatsLine(int device, uint64_t const* ring);
    } // namespace Client
} // namespace TensileLite
