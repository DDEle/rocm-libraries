// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ViewTransport.hpp"

#include <Tensile/ContractionProblem_fwd.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        struct DeviceContext;

        // --fused-a2a-verify values.
        constexpr int VERIFY_OFF     = 0;
        constexpr int VERIFY_PRODUCE = 1;
        constexpr int VERIFY_SKEW    = 2;

        constexpr uint32_t VERIFY_POISON_WORD = 0xC7C7C7C7u;

        // u32 words of the per-device check block, VERIFY_BLOCK_BYTES of fine-grained memory.
        constexpr uint32_t VERIFY_D_MISMATCHES    = 0;
        constexpr uint32_t VERIFY_RECV_MISMATCHES = 1;
        constexpr uint32_t VERIFY_FIRST_BAD       = 2;
        constexpr uint32_t VERIFY_CHECKED         = 3;
        constexpr uint32_t VERIFY_BLOCK_WORDS     = 4;
        constexpr size_t   VERIFY_BLOCK_BYTES     = 64;
        constexpr uint32_t VERIFY_NO_BAD_LAUNCH   = 0xFFFFFFFFu;

        // VerifyState::last of a slot with no launch.
        constexpr uint32_t VERIFY_NO_LAUNCH = 0xFFFFFFFFu;

        // The --fused-a2a-view-debug bits that launch 0 of a verify solution arms without.
        constexpr uint32_t VERIFY_LAUNCH0_CLEARED_DEBUG
            = VIEW_DEBUG_NO_POLL | VIEW_DEBUG_COPY_FAULTS;

        struct VerifyOptions
        {
            int      mode    = VERIFY_OFF;
            bool     barrier = true;
            uint32_t skewUs  = 1000;
            bool     fence   = true;
        };

        // The --fused-a2a-verify options checkFusedA2AVerifyArgs reads.
        struct VerifyArgs
        {
            bool fusedA2A              = true;
            int  verify                = VERIFY_OFF;
            int  barrier               = 1;
            int  fence                 = 1;
            int  skewUs                = 0;
            int  numElementsToValidate = -1;
            int  numWarmups            = 1;
            bool cEqualD               = false;
            bool sparse                = false;
            bool boundsCheckNaN        = false;
            bool scales                = false;
            bool swizzled              = false;
            bool gemmA2A               = false;
            bool betaZero              = true;
            bool activation            = false;
            bool bias                  = false;
        };

        // The first option of `a` that verify rejects, as a message; empty when there is none.
        inline std::string verifyArgsError(VerifyArgs const& a)
        {
            if(a.verify != VERIFY_OFF && a.verify != VERIFY_PRODUCE && a.verify != VERIFY_SKEW)
                return "--fused-a2a-verify must be 0, 1 or 2; got " + std::to_string(a.verify)
                       + ".";
            if(a.barrier != 0 && a.barrier != 1)
                return "--fused-a2a-verify-barrier must be 0 or 1.";
            if(a.fence != 0 && a.fence != 1)
                return "--fused-a2a-verify-fence must be 0 or 1.";
            if(a.skewUs < 0)
                return "--fused-a2a-verify-skew-us must not be negative.";
            if(a.verify == VERIFY_OFF)
                return {};
            if(!a.fusedA2A)
                return "--fused-a2a-verify applies only with --fused-a2a.";
            auto unsupported = [](char const* what) {
                return std::string("--fused-a2a-verify does not support ") + what + ".";
            };
            if(a.numElementsToValidate == 0)
                return unsupported("--num-elements-to-validate=0");
            if(a.numWarmups < 1)
                return unsupported("--num-warmups below 1");
            if(a.cEqualD)
                return unsupported("--c-equal-d");
            if(a.sparse)
                return unsupported("--sparse");
            if(a.boundsCheckNaN)
                return unsupported("--bounds-check=1");
            if(a.scales)
                return unsupported("scale tensors");
            if(a.swizzled)
                return unsupported("swizzled inputs");
            if(!a.gemmA2A)
                return {};
            if(!a.betaZero)
                return unsupported("GemmA2A with beta != 0");
            if(a.activation)
                return unsupported("GemmA2A with an activation");
            if(a.bias)
                return unsupported("GemmA2A with a bias");
            return {};
        }

        // Element i of a buffer is negated under pattern bits `bits` when bit c % 32 of `bits`
        // is set, c = ((period != 0 ? i % period : i) / stride) % shard. shard 0: no element.
        struct VerifyPattern
        {
            uint64_t stride       = 1;
            uint64_t period       = 0;
            uint64_t shard        = 0;
            uint32_t elementBytes = 4;
        };

        // The pattern bits of launch e; 0 for e = 0.
        __host__ __device__ inline uint32_t verifyPatternBits(uint32_t launch)
        {
            return launch * 0x9E3779B1u;
        }

        // The sign bits of the elements of u32 word `word` that `p` negates under `bits`.
        __host__ __device__ inline uint32_t
            verifyPatternMask(VerifyPattern const& p, uint32_t bits, size_t word)
        {
            if(p.shard == 0 || bits == 0)
                return 0u;
            const uint32_t width = 8 * p.elementBytes;
            const uint32_t lanes = 4 / p.elementBytes;
            uint32_t       mask  = 0;
            for(uint32_t l = 0; l < lanes; l++)
            {
                const uint64_t i = (uint64_t)word * lanes + l;
                const uint64_t c = ((p.period != 0 ? i % p.period : i) / p.stride) % p.shard;
                if(((bits >> (c % 32)) & 1u) != 0)
                    mask |= 1u << (l * width + width - 1);
            }
            return mask;
        }

        // Whether a checked word holds `golden` with the sign bits of `flip` flipped, per
        // viewVerifyCompare.
        __host__ __device__ inline bool
            verifyCheckWord(uint32_t word, uint32_t golden, uint32_t elementBytes, uint32_t flip)
        {
            return viewVerifyCompare(word, golden ^ flip, viewSignMask(elementBytes));
        }

        __host__ __device__ inline void verifyMin32(uint32_t* p, uint32_t v)
        {
#if defined(__HIP_DEVICE_COMPILE__)
            __hip_atomic_fetch_min(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
            uint32_t cur = __atomic_load_n(p, __ATOMIC_SEQ_CST);
            while(v < cur
                  && !__atomic_compare_exchange_n(
                      p, &cur, v, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))
            {
            }
#endif
        }

        enum class VerifyOpKind : uint32_t
        {
            Fill,
            Pattern,
            Copy,
            Check
        };

        // One pass over `count` u32 words. Fill: dst := VERIFY_POISON_WORD. Pattern: dst := src
        // with the sign bits of verifyPatternMask(pattern, bits) flipped; dst may be src. Copy:
        // dst := src, loaded as Check loads dst. Check: adds the words of dst that do not hold
        // src under verifyPatternMask to block[counter], lowers block[VERIFY_FIRST_BAD] to
        // `launch` when there is one, and with countLaunch adds 1 to block[VERIFY_CHECKED].
        // fence: every thread of the pass ends with __threadfence_system().
        struct VerifyOp
        {
            VerifyOpKind  kind  = VerifyOpKind::Fill;
            uint32_t*     dst   = nullptr;
            uint32_t*     src   = nullptr;
            size_t        count = 0;
            VerifyPattern pattern;
            uint32_t      bits        = 0;
            uint32_t      launch      = 0;
            uint32_t      counter     = VERIFY_D_MISMATCHES;
            bool          countLaunch = false;
            bool          fence       = true;
        };

        // Words first, first + step, ... of `op`; `lead`: the first thread of the pass.
        __host__ __device__ inline void
            verifyRunOp(VerifyOp const& op, size_t first, size_t step, uint32_t* block, bool lead)
        {
            uint32_t bad = 0;
            for(size_t i = first; i < op.count; i += step)
                switch(op.kind)
                {
                case VerifyOpKind::Fill:
                    op.dst[i] = VERIFY_POISON_WORD;
                    break;
                case VerifyOpKind::Pattern:
                    op.dst[i] = op.src[i] ^ verifyPatternMask(op.pattern, op.bits, i);
                    break;
                case VerifyOpKind::Copy:
                    op.dst[i] = viewLoad32(op.src + i);
                    break;
                case VerifyOpKind::Check:
                    bad += !verifyCheckWord(viewLoad32(op.dst + i),
                                            op.src[i],
                                            op.pattern.elementBytes,
                                            verifyPatternMask(op.pattern, op.bits, i));
                    break;
                }
            if(op.kind != VerifyOpKind::Check)
                return;
            if(bad != 0)
            {
                viewAdd32(block + op.counter, bad);
                verifyMin32(block + VERIFY_FIRST_BAD, op.launch);
            }
            if(op.countLaunch && lead)
                viewAdd32(block + VERIFY_CHECKED, 1u);
        }

        constexpr uint32_t VERIFY_THREADS    = 256;
        constexpr size_t   VERIFY_MAX_BLOCKS = 1024;

        // Blocks of VERIFY_THREADS threads of one pass over `count` words.
        inline uint32_t verifyBlocks(size_t count)
        {
            return (uint32_t)std::clamp<size_t>(
                (count + VERIFY_THREADS - 1) / VERIFY_THREADS, 1, VERIFY_MAX_BLOCKS);
        }

        // verifyRunOp's first, step and lead for thread `thread` of block `block` of a grid of
        // `blocks` blocks of `threads` threads.
        struct VerifySpan
        {
            size_t first = 0;
            size_t step  = 1;
            bool   lead  = false;
        };

        __host__ __device__ inline VerifySpan
            verifySpan(uint32_t block, uint32_t thread, uint32_t blocks, uint32_t threads)
        {
            return {(size_t)block * threads + thread,
                    (size_t)blocks * threads,
                    block == 0 && thread == 0};
        }

        // One tensor of a fused problem. stride: the element stride of its pattern dimension;
        // period: the next larger stride of the tensor, 0 when there is none.
        struct VerifyTensor
        {
            uint32_t elementBytes   = 0;
            bool     signMagnitude  = false;
            uint64_t allocatedBytes = 0;
            bool     padded         = false;
            uint64_t stride         = 1;
            uint64_t period         = 0;
        };

        // VerifyTensor::stride and ::period of dimension `dim` of a tensor of `strides`; stride 1
        // and period 0 for a dim past the strides.
        inline std::pair<uint64_t, uint64_t> verifyStridePeriod(std::vector<size_t> const& strides,
                                                                size_t                     dim)
        {
            if(dim >= strides.size())
                return {1, 0};
            const uint64_t stride = strides[dim];
            uint64_t       period = 0;
            for(size_t s : strides)
                if(s > stride && (period == 0 || s < period))
                    period = s;
            return {stride, period};
        }

        // Pattern dimensions: A2AGemm A along K; GemmA2A A and D along M.
        struct VerifyShape
        {
            bool         gemmA2A = false;
            uint32_t     world   = 0;
            uint64_t     extent  = 0;
            uint64_t     tokens  = 0;
            VerifyTensor a, b, d, buf;
        };

        // A2AGemm: launch e's producer writes A and B from their goldens under
        // verifyPatternBits(e) along the shard coordinate of K, which leaves D unchanged, and
        // poisons D and the a2aBuf words [peerLo, peerHi); the check compares D with the golden.
        // GemmA2A: the producer writes A from goldenA under the bits of e along the shard
        // coordinate of M and poisons D and a2aBuf; the check compares D and recv with their
        // goldens under the same bits. Sizes in u32 words.
        struct VerifyLayout
        {
            bool          gemmA2A  = false;
            uint32_t      world    = 0;
            size_t        aWords   = 0;
            size_t        bWords   = 0;
            size_t        dWords   = 0;
            size_t        bufWords = 0;
            size_t        peerLo   = 0;
            size_t        peerHi   = 0;
            VerifyPattern a, b, d, recv;
        };

        // Throws for a shape verify does not support: A, B (A2AGemm) or D (GemmA2A) not an
        // 8/16/32-bit sign-magnitude float, a tensor whose bytes are not a multiple of 4, an
        // extent the world does not divide, an A2AGemm B other than [W][N][extent/W], or a
        // GemmA2A D with padding.
        inline VerifyLayout verifyLayoutOf(VerifyShape const& s)
        {
            auto fail = [](std::string const& what) {
                throw std::runtime_error("[fused-a2a] --fused-a2a-verify " + what);
            };
            auto words = [&](VerifyTensor const& t, char const* name) -> size_t {
                if(t.allocatedBytes % sizeof(uint32_t) != 0)
                    fail(std::string("needs ") + name + " to be a multiple of 4 bytes; got "
                         + std::to_string(t.allocatedBytes) + ".");
                return t.allocatedBytes / sizeof(uint32_t);
            };
            auto pattern = [&](VerifyTensor const& t, char const* name, uint64_t shard) {
                if(!t.signMagnitude)
                    fail(std::string("supports ") + name + " of type f32, f16, bf16, f8 or bf8.");
                return VerifyPattern{t.stride, t.period, shard, t.elementBytes};
            };
            if(s.world == 0 || s.extent == 0 || s.extent % s.world != 0)
                fail("needs an extent the world divides; got " + std::to_string(s.extent) + " for "
                     + std::to_string(s.world) + ".");
            const uint64_t shard = s.extent / s.world;
            VerifyLayout   l;
            l.gemmA2A  = s.gemmA2A;
            l.world    = s.world;
            l.a        = pattern(s.a, "A", shard);
            l.aWords   = words(s.a, "A");
            l.dWords   = words(s.d, "D");
            l.bufWords = words(s.buf, "a2aBuf");
            if(s.gemmA2A)
            {
                l.d = pattern(s.d, "D", shard);
                if(s.d.padded)
                    fail("needs a GemmA2A D without padding.");
                l.recv = {1, 0, shard, s.d.elementBytes};
                return l;
            }
            pattern(s.b, "B", shard);
            l.b      = {1, 0, shard, s.b.elementBytes};
            l.bWords = words(s.b, "B");
            if(s.b.allocatedBytes != s.world * s.tokens * shard * s.b.elementBytes)
                fail("needs an A2AGemm B of [W][N][extent/W] elements.");
            const uint32_t dBytes   = s.d.elementBytes;
            l.d                     = {1, 0, 0, dBytes == 1 || dBytes == 2 ? dBytes : 4u};
            const uint64_t segBytes = shard * s.tokens * s.b.elementBytes;
            if(segBytes % sizeof(uint32_t) != 0
               || segBytes / sizeof(uint32_t) * s.world > l.bufWords)
                fail("cannot place a2aBuf segments of " + std::to_string(segBytes) + " bytes.");
            l.peerLo = segBytes / sizeof(uint32_t);
            l.peerHi = l.peerLo * s.world;
            return l;
        }

        // The buffers of one rotating slot of one device, and the goldens of that device.
        struct VerifySlot
        {
            void const* a   = nullptr;
            void const* b   = nullptr;
            void const* d   = nullptr;
            void const* buf = nullptr;
        };

        struct VerifyGoldens
        {
            void const* a    = nullptr;
            void const* b    = nullptr;
            void const* d    = nullptr;
            void const* recv = nullptr;
        };

        struct VerifyBuffer
        {
            void*  p     = nullptr;
            size_t bytes = 0;
        };

        struct VerifyGoldenBuffers
        {
            VerifyBuffer a, b, d, recv;
        };

        // 0: a golden the layout does not use.
        struct VerifyGoldenBytes
        {
            size_t a = 0, b = 0, d = 0, recv = 0;
        };

        // The bytes of A, A2AGemm B, D and GemmA2A recv of `l`.
        inline VerifyGoldenBytes verifyGoldenBytes(VerifyLayout const& l)
        {
            const size_t w = sizeof(uint32_t);
            return {l.aWords * w,
                    l.gemmA2A ? 0 : l.bWords * w,
                    l.dWords * w,
                    l.gemmA2A ? l.bufWords * w : 0};
        }

        // Makes `b` hold at least `want` bytes through `release(p)` and `allocate(bytes)`; keeps
        // a buffer that already does.
        template <class Release, class Allocate>
        void verifyGrow(VerifyBuffer& b, size_t want, Release&& release, Allocate&& allocate)
        {
            if(b.bytes >= want)
                return;
            if(b.p != nullptr)
                release(b.p);
            b       = {};
            b.p     = allocate(want);
            b.bytes = want;
        }

        // verifyGrow of every golden to verifyGoldenBytes(l).
        template <class Release, class Allocate>
        void verifyGrowGoldens(VerifyGoldenBuffers& g,
                               VerifyLayout const&  l,
                               Release&&            release,
                               Allocate&&           allocate)
        {
            const VerifyGoldenBytes n = verifyGoldenBytes(l);
            verifyGrow(g.a, n.a, release, allocate);
            verifyGrow(g.b, n.b, release, allocate);
            verifyGrow(g.d, n.d, release, allocate);
            verifyGrow(g.recv, n.recv, release, allocate);
        }

        // The goldens of `l` in `g`, nullptr for an unused one; throws for a golden that is
        // missing or smaller than verifyGoldenBytes(l).
        inline VerifyGoldens verifyGoldensOf(VerifyGoldenBuffers const& g, VerifyLayout const& l)
        {
            const VerifyGoldenBytes n = verifyGoldenBytes(l);
            auto need                 = [](VerifyBuffer const& b, size_t bytes, char const* name) {
                if(bytes == 0)
                    return static_cast<void const*>(nullptr);
                if(b.p == nullptr || b.bytes < bytes)
                    throw std::runtime_error(std::string("[fused-a2a] verify ") + name + " holds "
                                             + std::to_string(b.p == nullptr ? 0 : b.bytes)
                                             + " bytes; needs " + std::to_string(bytes) + ".");
                return static_cast<void const*>(b.p);
            };
            return {need(g.a, n.a, "goldenA"),
                    need(g.b, n.b, "goldenB"),
                    need(g.d, n.d, "goldenD"),
                    need(g.recv, n.recv, "goldenRecv")};
        }

        inline VerifyOp verifyOp(VerifyOpKind  kind,
                                 void const*   dst,
                                 void const*   src,
                                 size_t        count,
                                 VerifyPattern pattern = {},
                                 uint32_t      bits    = 0)
        {
            VerifyOp op;
            op.kind    = kind;
            op.dst     = static_cast<uint32_t*>(const_cast<void*>(dst));
            op.src     = static_cast<uint32_t*>(const_cast<void*>(src));
            op.count   = count;
            op.pattern = pattern;
            op.bits    = bits;
            return op;
        }

        // Launch e's producer on slot `s`; `fence` applies to the A2AGemm writes of A and B.
        inline std::vector<VerifyOp> verifyProduceOps(VerifyLayout const&  l,
                                                      VerifySlot const&    s,
                                                      VerifyGoldens const& g,
                                                      uint32_t             launch,
                                                      bool                 fence)
        {
            const uint32_t bits = verifyPatternBits(launch);
            auto           in   = [&](void const*          dst,
                                      void const*          golden,
                                      size_t               count,
                                      VerifyPattern const& p,
                                      bool                 f) {
                VerifyOp op = verifyOp(VerifyOpKind::Pattern, dst, golden, count, p, bits);
                op.fence    = f;
                return op;
            };
            if(!l.gemmA2A)
                return {verifyOp(VerifyOpKind::Fill,
                                 static_cast<uint32_t const*>(s.buf) + l.peerLo,
                                 nullptr,
                                 l.peerHi - l.peerLo),
                        verifyOp(VerifyOpKind::Fill, s.d, nullptr, l.dWords),
                        in(s.a, g.a, l.aWords, l.a, fence),
                        in(s.b, g.b, l.bWords, l.b, fence)};
            return {in(s.a, g.a, l.aWords, l.a, true),
                    verifyOp(VerifyOpKind::Fill, s.d, nullptr, l.dWords),
                    verifyOp(VerifyOpKind::Fill, s.buf, nullptr, l.bufWords)};
        }

        // Launch e's check of slot `s`: GemmA2A recv, then D; only the D pass counts the launch.
        inline std::vector<VerifyOp> verifyCheckOps(VerifyLayout const&  l,
                                                    VerifySlot const&    s,
                                                    VerifyGoldens const& g,
                                                    uint32_t             launch)
        {
            const uint32_t bits  = verifyPatternBits(launch);
            auto           check = [&](void const*          data,
                                       void const*          golden,
                                       size_t               count,
                                       VerifyPattern const& p,
                                       uint32_t             counter,
                                       bool                 countLaunch) {
                VerifyOp op    = verifyOp(VerifyOpKind::Check, data, golden, count, p, bits);
                op.launch      = launch;
                op.counter     = counter;
                op.countLaunch = countLaunch;
                return op;
            };
            std::vector<VerifyOp> ops;
            if(l.gemmA2A)
                ops.push_back(
                    check(s.buf, g.recv, l.bufWords, l.recv, VERIFY_RECV_MISMATCHES, false));
            ops.push_back(check(s.d, g.d, l.dWords, l.d, VERIFY_D_MISMATCHES, true));
            return ops;
        }

        // GemmA2A: D and recv of slot `s`, last written by launch e, back to the bits of
        // launch 0. A2AGemm: none.
        inline std::vector<VerifyOp>
            verifyNormalizeOps(VerifyLayout const& l, VerifySlot const& s, uint32_t launch)
        {
            if(!l.gemmA2A)
                return {};
            const uint32_t bits = verifyPatternBits(launch);
            return {verifyOp(VerifyOpKind::Pattern, s.d, s.d, l.dWords, l.d, bits),
                    verifyOp(VerifyOpKind::Pattern, s.buf, s.buf, l.bufWords, l.recv, bits)};
        }

        // A, and A2AGemm B, of slot `s` from the goldens.
        inline std::vector<VerifyOp>
            verifyRestoreOps(VerifyLayout const& l, VerifySlot const& s, VerifyGoldens const& g)
        {
            std::vector<VerifyOp> ops = {verifyOp(VerifyOpKind::Pattern, s.a, g.a, l.aWords)};
            if(!l.gemmA2A)
                ops.push_back(verifyOp(VerifyOpKind::Pattern, s.b, g.b, l.bWords));
            return ops;
        }

        // goldenA, and A2AGemm goldenB, from slot `s`.
        inline std::vector<VerifyOp> verifyCaptureInputOps(VerifyLayout const&  l,
                                                           VerifySlot const&    s,
                                                           VerifyGoldens const& g)
        {
            std::vector<VerifyOp> ops = {verifyOp(VerifyOpKind::Copy, g.a, s.a, l.aWords)};
            if(!l.gemmA2A)
                ops.push_back(verifyOp(VerifyOpKind::Copy, g.b, s.b, l.bWords));
            return ops;
        }

        // goldenD, and GemmA2A goldenRecv, from slot `s`.
        inline std::vector<VerifyOp> verifyCaptureGoldenOps(VerifyLayout const&  l,
                                                            VerifySlot const&    s,
                                                            VerifyGoldens const& g)
        {
            std::vector<VerifyOp> ops = {verifyOp(VerifyOpKind::Copy, g.d, s.d, l.dWords)};
            if(l.gemmA2A)
                ops.push_back(verifyOp(VerifyOpKind::Copy, g.recv, s.buf, l.bufWords));
            return ops;
        }

        // --fused-a2a-verify 2: whether `rank` spins before its producer of launch e.
        inline bool verifySkewVictim(uint32_t launch, uint32_t rank, uint32_t world)
        {
            return launch != 0 && world != 0 && launch % world == rank;
        }

        // Whether `rank` spins before its producer of launch e.
        inline bool verifySkews(VerifyOptions const& options,
                                uint32_t             launch,
                                uint32_t             rank,
                                uint32_t             world)
        {
            return options.mode == VERIFY_SKEW && verifySkewVictim(launch, rank, world);
        }

        // Wall-clock ticks of a spin of `us` microseconds at `wallClockKHz`.
        inline uint64_t verifySkewTicks(uint32_t us, uint64_t wallClockKHz)
        {
            return wallClockKHz * us / 1000;
        }

        // Whether launch e synchronizes every device's stream before and after its producers.
        inline bool verifyBarrier(VerifyOptions const& options, uint32_t launch)
        {
            return options.barrier || launch == 0;
        }

        // The --fused-a2a-view-debug bits launch e of a verify solution arms with.
        inline uint32_t verifyViewDebug(uint32_t debug, uint32_t launch)
        {
            return launch == 0 ? debug & ~VERIFY_LAUNCH0_CLEARED_DEBUG : debug;
        }

        enum class VerifyGolden
        {
            Pending,
            Captured,
            Disabled
        };

        // The golden state after the solution's first CPU validation: captured when it ran
        // and passed, disabled when it ran and failed.
        inline VerifyGolden
            verifyGoldenAfterValidation(VerifyGolden state, bool validated, bool failed)
        {
            if(state != VerifyGolden::Pending || !validated)
                return state;
            return failed ? VerifyGolden::Disabled : VerifyGolden::Captured;
        }

        // Host state of one solution: the verify launch counter e and, per rotating slot, e of
        // its last launch since the previous takeLaunched (VERIFY_NO_LAUNCH when none); the run
        // iterations; the spins and the ticks of the last one.
        struct VerifyState
        {
            VerifyGolden          golden   = VerifyGolden::Pending;
            uint32_t              launches = 0;
            std::vector<uint32_t> last;
            uint32_t              iterations = 0;
            uint32_t              skewed     = 0;
            uint64_t              skewTicks  = 0;

            void reset(size_t slots)
            {
                golden     = VerifyGolden::Pending;
                launches   = 0;
                iterations = 0;
                skewed     = 0;
                skewTicks  = 0;
                last.assign(slots, VERIFY_NO_LAUNCH);
            }

            // Counts a spin of `ticks`; 0: no spin.
            void addSkew(uint64_t ticks)
            {
                if(ticks == 0)
                    return;
                skewed++;
                skewTicks = ticks;
            }

            // Returns e of the launch on `slot`; throws std::out_of_range for a slot past reset's.
            uint32_t beginLaunch(size_t slot)
            {
                last.at(slot) = launches;
                return launches++;
            }

            bool checks() const
            {
                return golden == VerifyGolden::Captured;
            }

            // (slot, e of its last launch) of every slot launched since the previous call.
            std::vector<std::pair<size_t, uint32_t>> takeLaunched()
            {
                std::vector<std::pair<size_t, uint32_t>> slots;
                for(size_t k = 0; k < last.size(); k++)
                    if(last[k] != VERIFY_NO_LAUNCH)
                    {
                        slots.emplace_back(k, last[k]);
                        last[k] = VERIFY_NO_LAUNCH;
                    }
                return slots;
            }
        };

        struct VerifyCounts
        {
            uint64_t dMismatches    = 0;
            uint64_t recvMismatches = 0;
            uint32_t firstBad       = VERIFY_NO_BAD_LAUNCH;
            uint64_t checked        = 0;
        };

        inline void verifyInitialBlock(uint32_t (&words)[VERIFY_BLOCK_WORDS])
        {
            words[VERIFY_D_MISMATCHES]    = 0;
            words[VERIFY_RECV_MISMATCHES] = 0;
            words[VERIFY_FIRST_BAD]       = VERIFY_NO_BAD_LAUNCH;
            words[VERIFY_CHECKED]         = 0;
        }

        inline VerifyCounts verifyDecodeBlock(uint32_t const (&words)[VERIFY_BLOCK_WORDS])
        {
            VerifyCounts c;
            c.dMismatches    = words[VERIFY_D_MISMATCHES];
            c.recvMismatches = words[VERIFY_RECV_MISMATCHES];
            c.firstBad       = words[VERIFY_FIRST_BAD];
            c.checked        = words[VERIFY_CHECKED];
            return c;
        }

        // Sums the counters of `c` into `total` and keeps the lower first bad launch.
        inline void verifyAccumulate(VerifyCounts& total, VerifyCounts const& c)
        {
            total.dMismatches += c.dMismatches;
            total.recvMismatches += c.recvMismatches;
            total.checked += c.checked;
            total.firstBad = std::min(total.firstBad, c.firstBad);
        }

        // report: the block counted a check or a mismatch; failed: a mismatch.
        struct VerifyBlockVerdict
        {
            bool report = false;
            bool failed = false;
        };

        // verifyAccumulate of one device's block into `total`, and its verdict.
        inline VerifyBlockVerdict verifyTakeBlock(VerifyCounts& total, VerifyCounts const& c)
        {
            verifyAccumulate(total, c);
            const bool bad = c.dMismatches != 0 || c.recvMismatches != 0;
            return {bad || c.checked != 0, bad};
        }

        // Throws as verifyLayoutOf for `problem` on `world` devices.
        VerifyLayout verifyLayout(ContractionProblem const& problem, size_t world);

        // hipStreamSynchronize on every device's stream.
        void syncAll(std::vector<std::shared_ptr<DeviceContext>> const& devices);

        // On ctx.stream: [skew], then verifyProduceOps of launch e on rotating slot `slot`.
        // Returns the ticks of the skew, 0 without one.
        uint64_t verifyProduce(DeviceContext&       ctx,
                               VerifyLayout const&  layout,
                               size_t               slot,
                               uint32_t             launch,
                               VerifyOptions const& options,
                               uint32_t             rank);

        // On ctx.stream: verifyCheckOps of launch e on slot `slot` into ctx.verifyBlock.
        void verifyCheck(DeviceContext&      ctx,
                         VerifyLayout const& layout,
                         size_t              slot,
                         uint32_t            launch);

        // goldenA and A2AGemm goldenB from slot 0.
        void verifyCaptureInputs(DeviceContext& ctx, VerifyLayout const& layout);

        // goldenD and GemmA2A goldenRecv from slot 0.
        void verifyCaptureGolden(DeviceContext& ctx, VerifyLayout const& layout);

        // verifyNormalizeOps of slot `slot`, last written by launch e.
        void verifyNormalize(DeviceContext&      ctx,
                             VerifyLayout const& layout,
                             size_t              slot,
                             uint32_t            launch);

        // verifyRestoreOps of every rotating slot.
        void verifyRestore(DeviceContext& ctx, VerifyLayout const& layout);

        // Reads ctx.verifyBlock and resets it.
        VerifyCounts verifyReadBlock(DeviceContext& ctx);

        // Stores verifyInitialBlock into a check block.
        void verifyResetBlock(void* block);
    } // namespace Client
} // namespace TensileLite
