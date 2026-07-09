// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Single-process 4-GPU orchestration entry point for the fused GEMM.A2A kernel
// (Task 10). This is deliberately independent of the single-GPU benchmark loop
// in main.cpp: main() dispatches here (before the benchmark loop) when
// --fused-a2a is set and returns immediately afterwards. Nothing in the
// single-GPU path is touched.
//
// What this does (spec §3.1 / §3.2):
//   1. For each of W devices: allocate fresh per-device GEMM operands
//      (x=A, w=B, c=C, out=D) plus the fused-A2A buffers recv[] and flag[]
//      (fine-grained, since they are written by remote peers) and a device-scope
//      counter[].
//   2. Enable pairwise P2P access between all device pairs.
//   3. Per launch: zero counter[]/flag[] on every device, then for each device
//      build the host GEMM kernarg via solution->solve(), APPEND the fixed
//      156-byte fused-A2A segment (8 recv_ptr + 8 flag_ptr + counter_ptr +
//      5 u32 scalars) to that same KernelArguments object, and launch on the
//      device's stream. Because the launch reads kernel.args.size(), appending
//      to the host-generated args auto-sizes the launch to include the fused
//      tail — which is what fills the previously-garbage fused kernarg and makes
//      the hipErrorIllegalAddress(700) crash disappear.
//
// Scope: setup + launch smoke only. NO numeric/layout validation (that is
// Task 11-13). Success == no HIP error (other than the benign
// hipErrorPeerAccessAlreadyEnabled) and every kernel exits cleanly.

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include "ClientProblemFactory.hpp"
#include "SolutionIterator.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        // Compile-time fixed slot count for the fused-A2A kernarg segment. MUST
        // match FUSED_A2A_MAX_RANKS in Tensile/Components/Signature.py: the
        // kernel metadata always reserves 8 recv_ptr + 8 flag_ptr slots
        // regardless of the runtime world size, so the host must append exactly
        // 8 of each (unused slots j>=W filled with nullptr).
        static constexpr int    FUSED_A2A_MAX_RANKS      = 8;
        // Expected byte growth of args after appending the fused segment:
        //   (2*8 + 1) pointers * 8B + 5 scalars * 4B = 156B.
        static constexpr size_t FUSED_A2A_SEGMENT_BYTES  = (2 * FUSED_A2A_MAX_RANKS + 1) * 8 + 5 * 4;
        // A2A shards along N; kernel epilogue uses 256-wide N-tiles.
        static constexpr uint32_t FUSED_A2A_N_TILE = 256;
        static constexpr uint32_t FUSED_A2A_M_TILE = 256;

        namespace
        {
            // Append the fixed-size fused-A2A kernarg segment to `args` in the
            // exact emission order of Signature.py fusedA2AKernArgLayout().
            //
            // Alignment: recv_ptr_0 is appendAligned<void*> so it lands on an
            // 8-byte boundary, mirroring how the kernel metadata 8-aligns the
            // first SIG_GLOBALBUFFER arg of the segment. The remaining pointers
            // (8B) and scalars (4B) are appended contiguously with no interior
            // padding, matching the Python layout (off += 8 / off += 4).
            void appendFusedSegment(KernelArguments&           args,
                                    std::vector<void*> const&  recvPtrs,   // size W (device d's view: recv[j])
                                    std::vector<void*> const&  flagPtrs,   // size W
                                    void*                      counterPtr,
                                    uint32_t                   myRank,
                                    uint32_t                   target,
                                    uint32_t                   worldSize,
                                    uint32_t                   nShard,
                                    uint32_t                   drain)
            {
                size_t before = args.size();

                for(int j = 0; j < FUSED_A2A_MAX_RANKS; j++)
                {
                    void* p = (j < (int)recvPtrs.size()) ? recvPtrs[j] : nullptr;
                    if(j == 0)
                        args.appendAligned<void*>("recv_ptr_0", p);
                    else
                        args.append<void*>("recv_ptr_" + std::to_string(j), p);
                }
                for(int j = 0; j < FUSED_A2A_MAX_RANKS; j++)
                {
                    void* p = (j < (int)flagPtrs.size()) ? flagPtrs[j] : nullptr;
                    args.append<void*>("flag_ptr_" + std::to_string(j), p);
                }
                args.append<void*>("counter_ptr", counterPtr);
                args.append<uint32_t>("FusedMyRank", myRank);
                args.append<uint32_t>("FusedTarget", target);
                args.append<uint32_t>("FusedW", worldSize);
                args.append<uint32_t>("FusedNShard", nShard);
                args.append<uint32_t>("FusedDrain", drain);

                size_t grew = args.size() - before;
                if(grew != FUSED_A2A_SEGMENT_BYTES)
                {
                    std::cerr << "[fused-a2a] WARNING: fused segment grew args by " << grew
                              << " bytes, expected " << FUSED_A2A_SEGMENT_BYTES
                              << " (alignment/padding mismatch — epilogue will read wrong offsets)"
                              << std::endl;
                }
            }
        } // namespace

        // Entry point invoked from main() when --fused-a2a is passed. Returns a
        // process exit code (0 == smoke passed).
        int runFusedA2A(po::variables_map const&                                       args,
                        std::shared_ptr<MasterSolutionLibrary<ContractionProblemGemm>> library,
                        std::shared_ptr<Hardware>                                      hardware,
                        ClientProblemFactory&                                          problemFactory)
        {
            const int W     = args["fused-a2a-world"].as<int>();
            const int drain = args["fused-a2a-drain"].as<int>() ? 1 : 0;

            std::cout << "[fused-a2a] single-process " << W << "-GPU setup + launch smoke\n";

            int deviceCount = 0;
            HIP_CHECK_EXC(hipGetDeviceCount(&deviceCount));
            if(deviceCount < W)
            {
                std::cerr << "[fused-a2a] need " << W << " devices, found " << deviceCount
                          << std::endl;
                return 1;
            }

            // Pick the first problem / first solution. Task 10 only needs one
            // fused kernel launched on all W devices to prove the kernarg fill.
            auto problems = problemFactory.problems();
            if(problems.empty())
            {
                std::cerr << "[fused-a2a] no problems in config" << std::endl;
                return 1;
            }
            auto* problem = dynamic_cast<ContractionProblemGemm*>(problems.front().get());
            if(!problem)
            {
                std::cerr << "[fused-a2a] first problem is not a plain GEMM" << std::endl;
                return 1;
            }

            // Resolve the solution via the normal iterator (needs preProblem to
            // set its internal m_problem before getSolution()).
            auto solutionIterator = SolutionIterator::Default(library, hardware, args);
            solutionIterator->preProblem(problem);
            if(!solutionIterator->moreSolutionsInProblem())
            {
                std::cerr << "[fused-a2a] no solution for problem" << std::endl;
                return 1;
            }
            std::shared_ptr<ContractionSolution> solution = solutionIterator->getSolution();
            if(!solution)
            {
                std::cerr << "[fused-a2a] getSolution returned null" << std::endl;
                return 1;
            }
            std::cout << "[fused-a2a] solution: " << solution->name() << std::endl;

            // --- Derive fused shape from the problem (spec §0 relations, but
            //     using THIS problem's real M/N/K, not the big §0 defaults). ---
            const size_t M = problem->freeSizeA(0); // GEMM free dim M
            const size_t N = problem->freeSizeB(0); // GEMM free dim N == A2A columns
            if(N % (size_t)W != 0)
            {
                std::cerr << "[fused-a2a] N(" << N << ") not divisible by W(" << W << ")"
                          << std::endl;
                return 1;
            }
            const uint32_t nShard       = (uint32_t)(N / (size_t)W);
            // tiles_per_rank / M_tiles use ceil so tiny smoke shapes still yield
            // a non-zero target (spec §0 assumes 256-aligned; small yaml may not).
            const uint32_t tilesPerRank = (uint32_t)((nShard + FUSED_A2A_N_TILE - 1) / FUSED_A2A_N_TILE);
            const uint32_t mTiles       = (uint32_t)((M + FUSED_A2A_M_TILE - 1) / FUSED_A2A_M_TILE);
            const uint32_t target       = mTiles * tilesPerRank;

            // Fail-fast on shapes that violate the fused-A2A design constraints
            // (spec section 0). The kernel maps a whole PUSH workgroup to a
            // SINGLE dst_rank (n_col_base_wg = WorkGroup1 * MacroTile1), which is
            // only correct when each rank's shard is an integer number of
            // N-tiles -- i.e. n_shard is a multiple of the 256-wide macro-tile.
            // If n_shard < MacroTile1 (or not a multiple), one workgroup spans
            // several ranks: its lanes are all attributed to one rank, so data
            // is scattered to the wrong recv buffer AND, under DRAIN, ranks with
            // no supplying workgroup poll a flag slot no one ever sets -> the
            // GPU hangs forever. Reject such shapes on the host instead of
            // launching into a deadlock. Supported stage-1 shapes need
            // N % W == 0, n_shard % 256 == 0 (=> n_shard >= 256, all W ranks
            // covered), and M % 256 == 0.
            if(N % (size_t)W != 0 || (nShard % FUSED_A2A_N_TILE) != 0
               || (M % (size_t)FUSED_A2A_M_TILE) != 0)
            {
                std::cerr
                    << "[fused-a2a] ERROR: problem shape violates fused-A2A "
                       "constraints (spec section 0).\n"
                    << "  M=" << M << " N=" << N << " W=" << W
                    << " n_shard=N/W=" << nShard << " MacroTile=" << FUSED_A2A_N_TILE
                    << "\n"
                    << "  require: N % W == 0, (N/W) % " << FUSED_A2A_N_TILE
                    << " == 0 (so n_shard >= " << FUSED_A2A_N_TILE
                    << " and every rank is covered), M % " << FUSED_A2A_M_TILE
                    << " == 0.\n"
                    << "  e.g. W=4 needs N >= " << ((size_t)W * FUSED_A2A_N_TILE)
                    << " (n_shard >= 256). Refusing to launch (would deadlock in "
                       "the DRAIN barrier)."
                    << std::endl;
                return -1;
            }

            // The fused PUSH store writes the FULL macro-tile (MT0 rows), NOT
            // just the logical M rows, and the recv SRD uses no edge clamp
            // (num_records = BufferOOB). So a PUSH WG's lanes address recv rows
            // up to the padded macro-tile height. Size recv to the M rounded up
            // to the 256-wide macro-tile so those padding-row writes stay inside
            // the allocation. (Same reasoning for n_shard rounded to the N-tile.)
            const size_t Mpad      = ((M + FUSED_A2A_M_TILE - 1) / FUSED_A2A_M_TILE) * FUSED_A2A_M_TILE;
            const size_t nShardPad = ((size_t)nShard + FUSED_A2A_N_TILE - 1) / FUSED_A2A_N_TILE * FUSED_A2A_N_TILE;
            const size_t recvBytes    = (size_t)W * Mpad * nShardPad * sizeof(uint16_t); // bf16
            const size_t flagBytes    = (size_t)W * sizeof(uint32_t);
            const size_t counterBytes = (size_t)W * sizeof(uint32_t);
            const size_t aBytes       = problem->a().totalAllocatedBytes();
            const size_t bBytes       = problem->b().totalAllocatedBytes();
            const size_t cBytes       = problem->c().totalAllocatedBytes();
            const size_t dBytes       = problem->d().totalAllocatedBytes();

            std::cout << "[fused-a2a] M=" << M << " N=" << N << " nShard=" << nShard
                      << " tilesPerRank=" << tilesPerRank << " mTiles=" << mTiles
                      << " target=" << target << " drain=" << drain << "\n";

            // --- Phase 1: per-device fresh allocation (spec §3.1). ---
            std::vector<void*> recv(W, nullptr), flag(W, nullptr), counter(W, nullptr);
            std::vector<void*> xA(W, nullptr), wB(W, nullptr), cC(W, nullptr), outD(W, nullptr);

            for(int d = 0; d < W; d++)
            {
                HIP_CHECK_EXC(hipSetDevice(d));
                // Fine-grained: written by remote peers, must bypass stale L2.
                HIP_CHECK_EXC(hipExtMallocWithFlags(&recv[d], recvBytes, hipDeviceMallocFinegrained));
                HIP_CHECK_EXC(hipExtMallocWithFlags(&flag[d], flagBytes, hipDeviceMallocFinegrained));
                // Local (not remotely written): plain device memory.
                HIP_CHECK_EXC(hipMalloc(&counter[d], counterBytes));
                HIP_CHECK_EXC(hipMalloc(&xA[d], aBytes));
                HIP_CHECK_EXC(hipMalloc(&wB[d], bBytes));
                HIP_CHECK_EXC(hipMalloc(&cC[d], cBytes));
                HIP_CHECK_EXC(hipMalloc(&outD[d], dBytes));
                // Give GEMM operands deterministic, valid contents; zero recv/out.
                HIP_CHECK_EXC(hipMemset(xA[d], 0, aBytes));
                HIP_CHECK_EXC(hipMemset(wB[d], 0, bBytes));
                HIP_CHECK_EXC(hipMemset(cC[d], 0, cBytes));
                HIP_CHECK_EXC(hipMemset(outD[d], 0, dBytes));
                HIP_CHECK_EXC(hipMemset(recv[d], 0, recvBytes));
            }

            // --- P2P pairwise enable (spec §3.1). AlreadyEnabled is benign. ---
            for(int s = 0; s < W; s++)
            {
                HIP_CHECK_EXC(hipSetDevice(s));
                for(int t = 0; t < W; t++)
                {
                    if(t == s)
                        continue;
                    int canAccess = 0;
                    HIP_CHECK_EXC(hipDeviceCanAccessPeer(&canAccess, s, t));
                    if(!canAccess)
                    {
                        std::cerr << "[fused-a2a] WARNING: device " << s << " cannot P2P device "
                                  << t << std::endl;
                        continue;
                    }
                    hipError_t pe = hipDeviceEnablePeerAccess(t, 0);
                    if(pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled)
                        HIP_CHECK_EXC(pe);
                }
            }

            // --- Per-device streams + code-object adapters. The main adapter's
            //     modules are bound to device 0; give each device its own adapter
            //     with the fused .co loaded in that device's context so launches
            //     on devices 1..W-1 resolve the kernel correctly. ---
            auto filename = args["library-file"].as<std::string>();
            size_t dirPos = filename.rfind('/');
            std::string libraryDirectory = (dirPos != std::string::npos)
                                               ? filename.substr(0, dirPos + 1)
                                               : std::string(".");

            std::vector<std::shared_ptr<hip::SolutionAdapter>> adapters(W);
            std::vector<hipStream_t>                           streams(W, nullptr);
            auto const& codeObjectFiles = args["code-object"].as<std::vector<std::string>>();

            for(int d = 0; d < W; d++)
            {
                HIP_CHECK_EXC(hipSetDevice(d));
                HIP_CHECK_EXC(hipStreamCreate(&streams[d]));
                adapters[d] = std::make_shared<hip::SolutionAdapter>();
                bool loadedAny = false;
                for(auto const& co : codeObjectFiles)
                {
                    if(adapters[d]->loadCodeObjectFile(co) == hipSuccess)
                        loadedAny = true;
                }
                // Lazy loading discovers the fused .co by kernel name from the
                // TensileLibrary directory (same mechanism as main()).
                (void)adapters[d]->initializeLazyLoading(hardware->archName(), libraryDirectory);
                (void)loadedAny;
            }

            // --- Phase 2: zero counter/flag, then solve + append + launch on
            //     all W devices (spec §3.2). ---
            for(int d = 0; d < W; d++)
            {
                HIP_CHECK_EXC(hipSetDevice(d));
                HIP_CHECK_EXC(hipMemset(counter[d], 0, counterBytes)); // inc from 0
                HIP_CHECK_EXC(hipMemset(flag[d], 0, flagBytes));       // NOT_READY
            }
            for(int d = 0; d < W; d++)
                HIP_CHECK_EXC(hipDeviceSynchronize());

            // Launch each device (single-process, sequential enqueue to W
            // streams). DRAIN=ON: the last WG polls this device's own flag, which
            // is set by peers — so all W must be launched for the barrier to
            // release. We enqueue all first, then synchronize.
            std::vector<std::vector<KernelInvocation>> perDeviceKernels(W);
            for(int d = 0; d < W; d++)
            {
                HIP_CHECK_EXC(hipSetDevice(d));

                // Build this device's GEMM inputs pointing at ITS fresh operands.
                ContractionInputs inputs;
                inputs.a     = xA[d];
                inputs.b     = wB[d];
                inputs.c     = cC[d];
                inputs.d     = outD[d];
                inputs.alpha = static_cast<float>(1);
                inputs.beta  = static_cast<float>(0);
                inputs.gpu   = true;

                auto kernels = solution->solve(*problem, inputs, *hardware, nullptr, 0, streams[d]);
                if(kernels.empty())
                {
                    std::cerr << "[fused-a2a] solve() produced no kernels on device " << d
                              << std::endl;
                    return 1;
                }

                // recv/flag pointer views for device d: slot j = peer j's buffer.
                // recv_ptr_[j] is peer j's recv base (where device d PUSHes its
                // shard to). recv_ptr_[myRank] is this device's own recv base.
                std::vector<void*> recvView(W), flagView(W);
                for(int j = 0; j < W; j++)
                {
                    recvView[j] = recv[j];
                    flagView[j] = flag[j];
                }

                // The fused segment must sit at the very tail; append to the LAST
                // invocation's args (the single-call fused GEMM has one).
                KernelInvocation& last = kernels.back();
                size_t beforeSize = last.args.size();
                appendFusedSegment(last.args,
                                   recvView,
                                   flagView,
                                   counter[d],
                                   (uint32_t)d,     // my_rank
                                   target,
                                   (uint32_t)W,
                                   nShard,
                                   (uint32_t)drain);
                std::cout << "[fused-a2a] dev " << d << " kernarg: host base(before append)="
                          << beforeSize << " size(after)=" << last.args.size() << "\n";

                perDeviceKernels[d] = std::move(kernels);
                HIP_CHECK_EXC(adapters[d]->launchKernels(perDeviceKernels[d], streams[d], nullptr,
                                                         nullptr));
            }

            // Wait for every device and report.
            bool ok = true;
            for(int d = 0; d < W; d++)
            {
                HIP_CHECK_EXC(hipSetDevice(d));
                hipError_t se = hipStreamSynchronize(streams[d]);
                if(se != hipSuccess)
                {
                    std::cerr << "[fused-a2a] device " << d
                              << " kernel FAILED: " << hipGetErrorString(se) << std::endl;
                    ok = false;
                }
                else
                {
                    std::cout << "[fused-a2a] device " << d << " kernel exited cleanly\n";
                }
            }

            // Cleanup.
            for(int d = 0; d < W; d++)
            {
                HIP_CHECK_EXC(hipSetDevice(d));
                if(recv[d])
                    (void)hipFree(recv[d]);
                if(flag[d])
                    (void)hipFree(flag[d]);
                if(counter[d])
                    (void)hipFree(counter[d]);
                if(xA[d])
                    (void)hipFree(xA[d]);
                if(wB[d])
                    (void)hipFree(wB[d]);
                if(cC[d])
                    (void)hipFree(cC[d]);
                if(outD[d])
                    (void)hipFree(outD[d]);
                if(streams[d])
                    (void)hipStreamDestroy(streams[d]);
            }

            std::cout << "[fused-a2a] smoke " << (ok ? "PASSED" : "FAILED") << std::endl;
            return ok ? 0 : 2;
        }

    } // namespace Client
} // namespace TensileLite
