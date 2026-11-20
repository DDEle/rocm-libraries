/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <Tensile/Contractions.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/EmbeddedLibrary.hpp>
#include <Tensile/FusedA2A.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include "BenchmarkTimer.hpp"
#include "ClientConfig.hpp"
#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp"
#include "DeviceContext.hpp"
#include "HardwareMonitorListener.hpp"
#include "MetaRunListener.hpp"
#include "ProgressListener.hpp"
#include "ReferenceValidator.hpp"
#include "SolutionIterator.hpp"
#include "SynchronizerValidator.hpp"
#include "TimingEvents.hpp"
#include "TimingInstrumentation.hpp"
#include "ViewTransport.hpp"
#include "ViewVerify.hpp"

#include "LibraryUpdateReporter.hpp"
#include "LogReporter.hpp"
#include "MetaResultReporter.hpp"
#include "PerformanceReporter.hpp"
#include "ResultFileReporter.hpp"
#include "ResultReporter.hpp"

#include "ProgramOptions.hpp"
#include "Utility.hpp"

#if TENSILELITE_CLIENT_ENABLE_ROCPROFSDK
#include "Profiler.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cstddef>
#if defined(__linux__)
// Used by getMinKernelSizeToGwEnd() to parse .co ELF symbol tables. <elf.h>
// is glibc-only, so the whole ELF code path is Linux-gated.
#include <cstring> // std::memcmp / std::strcmp
#include <elf.h>
#include <limits> // std::numeric_limits<> sentinel
#endif
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>

namespace TensileLite
{
    namespace Client
    {
        __global__ void flush_icache()
        {
            asm __volatile__("s_icache_inv \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t"
                             "s_nop 0 \n\t" ::
                                 :);
        }

        uint32_t flush_grid_size()
        {
            hipDeviceProp_t deviceProps;
            HIP_CHECK_EXC(hipGetDeviceProperties(&deviceProps, 0));
            uint32_t numBlocks = deviceProps.multiProcessorCount * 60;
            return numBlocks;
        }

        float estimate_flush_kernel_time(hipStream_t stream, bool useGPUTimer)
        {
            const int                                                   flushIter = 100000;
            hipEvent_t                                                  start, stop;
            std::chrono::time_point<std::chrono::high_resolution_clock> begTime;

            if(useGPUTimer)
            {
                HIP_CHECK_EXC(hipEventCreate(&start));
                HIP_CHECK_EXC(hipEventCreate(&stop));
            }

            //warmup runs
            for(int i = 0; i < flushIter; i++)
            {
                hipLaunchKernelGGL(flush_icache, flush_grid_size(), 64, 0, stream);
            }

            if(useGPUTimer)
            {
                HIP_CHECK_EXC(hipEventRecord(start, stream));
            }
            else
            {
                HIP_CHECK_EXC(hipStreamSynchronize(stream));
                begTime = std::chrono::high_resolution_clock::now();
            }

            for(int i = 0; i < flushIter; i++)
            {
                hipLaunchKernelGGL(flush_icache, flush_grid_size(), 64, 0, stream);
            }

            float time{};

            if(useGPUTimer)
            {
                HIP_CHECK_EXC(hipEventRecord(stop, stream));
                HIP_CHECK_EXC(hipEventSynchronize(stop));
            }

            HIP_CHECK_EXC(hipStreamSynchronize(stream));

            if(useGPUTimer)
            {
                HIP_CHECK_EXC(hipEventElapsedTime(&time, start, stop));
                HIP_CHECK_EXC(hipEventDestroy(start));
                HIP_CHECK_EXC(hipEventDestroy(stop));
            }
            else
            {
                time = std::chrono::duration<float,
                                             std::milli>{std::chrono::high_resolution_clock::now()
                                                         - begTime}
                           .count();
            }
            return time / flushIter;
        }

        template <typename T>
        po::typed_value<T>* value_default(std::string const& desc)
        {
            return po::value<T>()->default_value(T(), desc);
        }

        template <typename T>
        po::typed_value<T>* value_default()
        {
            return po::value<T>()->default_value(T());
        }

        template <typename T>
        po::typed_value<std::vector<T>>* vector_default_empty()
        {
            return value_default<std::vector<T>>("[]");
        }

        po::options_description all_options()
        {
            po::options_description options("Tensile client options");

            // clang-format off
            options.add_options()
                ("help,h", "Show help message.")

                ("config-file",              vector_default_empty<std::string>(), "INI config file(s) to read.")

                ("library-file,l",           po::value<std::string>(), "Load a (YAML) solution library.  If not specified, we will use "
                                                                       "the embedded library, if available.")
                ("code-object,c",            vector_default_empty<std::string>(), "Code object file with kernel(s).  If none are "
                                                                                  "specified, we will use the embedded code "
                                                                                  "object(s) if available.")

                ("performance-metric",       po::value<PerformanceMetric>()->default_value(PerformanceMetric::DeviceEfficiency), "Metric for benchmarking results")

                ("problem-identifier",       po::value<std::string>(), "Problem identifer (Einstein notation). Either "
                                                                       "this or free/batch/bound must be specified.")

                ("type",                     po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "Data type")
                ("a-type",                   po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "A data type")
                ("b-type",                   po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "B data type")
                ("c-type",                   po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "C data type")
                ("d-type",                   po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "D data type")
                ("e-type",                   po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "E data type")
                ("amaxD-type",               po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "amaxD data type")
                ("alpha-type",               po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "alpha data type")
                ("beta-type",                po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "beta data type")
                ("compute-input-type-A",     po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "compute input data type A")
                ("compute-input-type-B",     po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "compute input data type B")
                ("f32-xdl-math-op",          po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "Use xf32 compute for float input and output matrices.")
                ("mx-a-block",               po::value<int>()->default_value(0), "block of mx datatype input matrix A")
                ("mx-b-block",               po::value<int>()->default_value(0), "block of mx datatype input matrix B")
                ("mx-a-free-block",          po::value<int>()->default_value(1), "free-dimension (M) extent of one mx scaling tile for matrix A; 1 = one scale per row")
                ("mx-b-free-block",          po::value<int>()->default_value(1), "free-dimension (N) extent of one mx scaling tile for matrix B; 1 = one scale per column")
                ("mx-a-type",                po::value<rocisa::DataType>()->default_value(rocisa::DataType::E8), "type of mx datatype input matrix A")
                ("mx-b-type",                po::value<rocisa::DataType>()->default_value(rocisa::DataType::E8), "type of mx datatype input matrix B")
                ("swizzle-tensor-a",         po::value<bool>()->default_value(false), "Swizzle input tensor A.")
                ("swizzle-tensor-b",         po::value<bool>()->default_value(false), "Swizzle input tensor B.")
                ("fused-a2a",                po::value<FusedA2A>()->default_value(FusedA2A::Disabled), "Fused A2A mode: Disabled, GemmA2A (all-to-all PUSH from the GEMM epilogue) or A2AGemm (B's K shards gathered from the peers into a2aBuf one shard at a time).")
                ("mx-scale-format",          po::value<int>()->default_value(0), "MX scale data format (0=none, 1=pre-swizzle for GPU kernel layout)")
                ("activation-compute-type",  po::value<rocisa::DataType>()->default_value(rocisa::DataType::None), "Activation compute type.")
                ("high-precision-accumulate", po::value<bool>()->default_value(false), "Use high-precision accumulate.")
                ("sparse",                   po::value<int>()->default_value(0), "A or B matrix is sparse matrix.")
                ("strided-batched",          po::value<bool>()->default_value(true), "Use strided-batched or general batched")
                ("batch-mode",               po::value<int>()->default_value(0), "Runtime batch ABI: 0=strided, 1=pointer array")
                ("grouped-gemm",             po::value<bool>()->default_value(false), "Use grouped gemm")
                ("kernel-language",          po::value<KernelLanguage>()->default_value(KernelLanguage::Any), "Select kernel language.")
                ("deterministic-mode",       po::value<bool>()->default_value(false), "Enforce deterministic summation patterns"
                                                                                      "by not splitting U among workgroups")

                ("init-seed",                po::value<unsigned int>()->default_value(0), "The seed for srand")
                ("init-a",                   po::value<InitMode>()->default_value(InitMode::Random), "Initialization for A")
                ("init-b",                   po::value<InitMode>()->default_value(InitMode::Random), "Initialization for B")
                ("init-c",                   po::value<InitMode>()->default_value(InitMode::Random), "Initialization for C")
                ("init-d",                   po::value<InitMode>()->default_value(InitMode::Zero), "Initialization for D")
                ("init-e",                   po::value<InitMode>()->default_value(InitMode::Zero), "Initialization for E")
                ("init-gate",                po::value<InitMode>()->default_value(InitMode::Random),  "Initialization for gate residual")
                ("init-alpha",               po::value<InitMode>()->default_value(InitMode::Two), "Initialization for alpha")
                ("init-beta",                po::value<InitMode>()->default_value(InitMode::Two), "Initialization for beta")
                ("init-bias",                po::value<InitMode>()->default_value(InitMode::One), "Initialization for bias")
                ("init-scaleA",              po::value<InitMode>()->default_value(InitMode::Two), "Initialization for scaleA")
                ("init-scaleB",              po::value<InitMode>()->default_value(InitMode::Two), "Initialization for scaleB")
                ("init-scaleC",              po::value<InitMode>()->default_value(InitMode::Two), "Initialization for scaleC")
                ("init-scaleD",              po::value<InitMode>()->default_value(InitMode::Two), "Initialization for scaleD")
                ("init-scaleAlphaVec",       po::value<InitMode>()->default_value(InitMode::One), "Initialization for scaleAlphaVec")
                ("init-mx-a",                po::value<InitMode>()->default_value(InitMode::One), "Initialization for MX Scale for A")
                ("init-mx-b",                po::value<InitMode>()->default_value(InitMode::One), "Initialization for MX Scale for B")
                ("mx-data-generator",        po::value<int>()->default_value(-1), "A/B data source. -1: mxDataGenerator for MX inputs only. 1: mxDataGenerator for A/B, MX or not (non-MX A/B must be FP8/BF8). 0: init-a/init-b like non-MX inputs, MX or not, with MX scales from init-mx-a/init-mx-b (MX FP8/BF8/FP4 data).")
                ("pristine-on-gpu",          po::value<bool>()->default_value(true), "Keep a pristine copy of inputs on GPU for performance")
                ("c-equal-d",                po::value<bool>()->default_value(false), "C equals D")
                ("print-valids",             po::value<bool>()->default_value(false), "Print values that pass validation")
                ("print-max",                po::value<int>()->default_value(-1), "Max number of values to print")
                ("num-elements-to-validate", po::value<int>()->default_value(0), "Number of elements to validate")
                ("check-synchronizer",       po::value<bool>()->default_value(true),
                "Fail the run if a StreamK, GSU MBSK or output-amax kernel leaves the shared Synchronizer buffer nonzero on exit."
                " Solutions known not to use the buffer are skipped.")
                ("validate-after-enqueues",  po::value<bool>()->default_value(false),
                "Validate final output and synchronizer state after each enqueue group; requires c-equal-d=false.")
                ("bounds-check",             po::value<BoundsCheckMode>()->default_value(BoundsCheckMode::Disable),
                "1:Use sentinel values to check memory boundaries."
                "2:Memory bound check by front guard page"
                "3:Memory bound check by back guard page"
                "4:Memory bound check by both side guard page")
                ("prune-mode",               po::value<PruneSparseMode>()->default_value(PruneSparseMode::PruneRandom), "prune Sparse mode")

                ("print-tensor-a",                  po::value<bool>()->default_value(false), "Print tensor A.")
                ("print-tensor-b",                  po::value<bool>()->default_value(false), "Print tensor B.")
                ("print-tensor-c",                  po::value<bool>()->default_value(false), "Print tensor C.")
                ("print-tensor-d",                  po::value<bool>()->default_value(false), "Print tensor D.")
                ("print-tensor-ref",                po::value<bool>()->default_value(false), "Print reference tensor D.")
                ("print-tensor-bias",               po::value<bool>()->default_value(false), "Print tensor Bias.")
                ("print-tensor-gate",               po::value<bool>()->default_value(false), "Print tensor GateResidual.")
                ("print-tensor-scale-alpha-vec",    po::value<bool>()->default_value(false), "Print tensor ScaleAlphaVec.")
                ("print-tensor-amaxd",              po::value<bool>()->default_value(false), "Print tensor AmaxD value from both CPU and GPU.")

                ("dump-tensors",             po::value<bool>()->default_value(false), "Binary dump tensors instead of printing.")

                ("device-idx",               po::value<int>()->default_value(0), "Device index")
                ("world",                    po::value<std::vector<int>>()->default_value(std::vector<int>()), "Fused GEMM+A2A: device count of each problem, one value per problem-size.")
                ("fused-a2a-extent",         po::value<std::vector<int>>()->default_value(std::vector<int>()), "Fused GEMM+A2A: extent that goes all-to-all, one value per problem-size: leading feature (M) count for GemmA2A, reduction (K) count for A2AGemm (defaults to K).")
                ("world-filter",             po::value<std::vector<int>>()->default_value(std::vector<int>()), "Fused GEMM+A2A: run only the problems whose world is listed; empty runs all.")
                ("fused-a2a-drain-recv",     po::value<int>()->default_value(1), "Runtime drainRecv flag passed to the fused kernel (1=on): the kernel exits only once this card's recv buffer is complete.")
                ("fused-a2a-drain-send",     po::value<int>()->default_value(1), "Runtime drainSend flag passed to the fused kernel (1=on): the kernel exits only once this card's engines have finished reading D[0:AM), so D can be reused on stream order alone. Independent of --fused-a2a-drain-recv.")
                ("fused-a2a-view-arm",       po::value<int>()->default_value(VIEW_ARM_DEVICE), "Fused A2A, FusedA2ATransport=View solutions: who writes the SDMA chains before each launch. 0: a kernel on the device's stream; 1: the host thread, after synchronizing that stream.")
                ("fused-a2a-view-coalesce",  po::value<int>()->default_value(1), "Fused A2A, A2AGemm FusedA2ATransport=View solutions: COPY units per arrival ATOMIC on each queue; the last ATOMIC carries the remainder.")
                ("fused-a2a-view-debug",     po::value<int>()->default_value(0), "Fused A2A debug bitmask. 1: View solutions record a per-launch timeline (arm kernel, GEMM end, wait kernel, and a TIMESTAMP at the start and end of each queue's chain) and print VIEWSTAT lines per solution; 2: GemmA2A View chains are built without their POLLs (data is wrong); 4: InKernel solutions launch an empty kernel right before and right after the fused GEMM; 8: View queue 0 omits its first COPY; 16: View queue 0's first COPY moves to just before its chain's last ATOMIC.")
                ("fused-a2a-verify",         po::value<int>()->default_value(VERIFY_OFF), "Fused A2A: 0 off; 1 before every launch e a producer rewrites the inputs from copies taken before the problem's first launch, each element negated by bit (its shard coordinate mod 32) of e * 0x9E3779B1, and poisons D and the a2aBuf peer segments (A2AGemm: A and B along K, so D equals the launch-0 D bitwise; GemmA2A: A along M, all of a2aBuf); after the launch a kernel compares D (GemmA2A: and the received a2aBuf) with the launch-0 result under the same pattern. Launch 0 has no pattern. 2 as 1, and from launch 1 on rank (e mod W) spins --fused-a2a-verify-skew-us before its producer.")
                ("fused-a2a-verify-barrier", po::value<int>()->default_value(1), "Fused A2A, --fused-a2a-verify: 1 synchronizes every device's stream before and after the producers of each launch; 0 does not.")
                ("fused-a2a-verify-skew-us", po::value<int>()->default_value(1000), "Fused A2A, --fused-a2a-verify 2: the spin, in microseconds.")
                ("fused-a2a-verify-fence",   po::value<int>()->default_value(1), "Fused A2A, --fused-a2a-verify: 1 the A2AGemm producer of A and B ends with __threadfence_system(); 0 it does not.")
                ("fused-a2a-sdma-rotate",    po::value<int>(), "Fused A2A, gfx942 and gfx950: move each remote peer SDMA queue this many steps along the even xGMI engines 2,4,...,14 from its KFD recommendation; nonzero requires 14 xGMI SDMA engines. Ignored on other architectures, which spread the remote peer queues over the recommended engines. The loopback queue uses engine 0. Defaults to -1.")
                ("use-default-stream",       po::value<bool>()->default_value(false), "Use default Hip stream to run kernels.")

                ("num-warmups",              po::value<int>()->default_value(0), "Number of warmups to run")
                ("sync-after-warmups",       po::value<bool>()->default_value(true), "Synchronize GPU after warmup kernel runs")
                ("num-benchmarks",           po::value<int>()->default_value(1), "Number of benchmarks to run")
                ("num-enqueues-per-sync",    po::value<int>()->default_value(1), "Enqueues per sync, will affect by min-flops-per-sync")
                ("max-enqueues-per-sync",    po::value<int>()->default_value(-1), "Max Enqueues per sync, will affect by min-flops-per-sync")
                ("num-syncs-per-benchmark",  po::value<int>()->default_value(1), "Syncs per benchmark")
                ("skip-slow-solution-ratio", po::value<float>()->default_value(0.0), "ratio to skip slow solution during warm-up stage")
                ("min-flops-per-sync",       po::value<size_t>()->default_value(0), "Minimum number of flops per sync to increase stability for small problems.")
                ("use-gpu-timer",            po::value<bool>()->default_value(true), "Use GPU timer")
                ("sleep-percent",            po::value<int>()->default_value(0), "Sleep percentage")
                ("hardware-monitor",         po::value<bool>()->default_value(true), "Use hardware monitor.")

                ("perf-l2-read-hits",        po::value<double>()->default_value(0.0), "L2 read hits")
                ("perf-l2-write-hits",       po::value<double>()->default_value(0.5), "L2 write hits")
                ("perf-l2-read-bw-mul",      po::value<double>()->default_value(2.0), "L2 read bandwidth multiplier")
                ("perf-read-efficiency",     po::value<double>()->default_value(0.85), "Read efficiency")
                ("csv-export-extra-cols",    po::value<bool>()->default_value(false), "CSV exports winner information")
                ("csv-merge-same-problems",  po::value<bool>()->default_value(false), "CSV merge rows of same problem id")
                ("PrintWinnersOnly",         po::value<bool>()->default_value(false), "PrintWinnersOnly")

                ("problem-size,p",           vector_default_empty<std::string>(), "Specify a problem size.  Comma-separated list of "
                                                                                  "sizes, in the order of the Einstein notation.")

                ("prob-sol-map",             vector_default_empty<std::string>(), "[probIdx, solIdx]")

                ("a-strides",                vector_default_empty<std::string>(), "Unspecified means default stride "
                                                                                  "(prev_dim_stride*prev_dim_size)"
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")

                ("b-strides",                vector_default_empty<std::string>(), "Unspecified means default stride "
                                                                                  "(prev_dim_stride*prev_dim_size)"
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")

                ("c-strides",                vector_default_empty<std::string>(), "Unspecified means default stride "
                                                                                  "(prev_dim_stride*prev_dim_size)"
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")

                ("d-strides",                vector_default_empty<std::string>(), "Unspecified means default stride "
                                                                                  "(prev_dim_stride*prev_dim_size)"
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")

                ("e-strides",                vector_default_empty<std::string>(), "Unspecified means default stride "
                                                                                  "(prev_dim_stride*prev_dim_size)"
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")
                ("bias-strides",             vector_default_empty<std::string>(), "Unspecified means default stride "
                                                                                  "(prev_dim_stride*prev_dim_size)"
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")
                ("gate-strides",             vector_default_empty<std::string>(), "Unspecified means default stride "
                                                                                  "(prev_dim_stride*prev_dim_size)"
                                                                                  "specifying once applies to all problem sizes, "
                                                                                  "otherwise specify once per problem size.")
                ("problem-start-idx",        po::value<int>()->default_value(0),  "First problem to run")
                ("num-problems",             po::value<int>()->default_value(-1), "Number of problems to run")

                ("solution-start-idx",       po::value<int>()->default_value(-1), "First solution to run")
                ("num-solutions",            po::value<int>()->default_value(-1), "Number of solutions to run")
                ("best-solution",            po::value<bool>()->default_value(false), "Best solution benchmark mode")
                ("require-selected-solution", po::value<bool>()->default_value(false), "Requires --best-solution: a problem for which "
                                                                                       "the library selects no solution is an error "
                                                                                       "(non-zero exit) and is skipped, instead of "
                                                                                       "running library solution 0.")

                ("results-file",             po::value<std::string>()->default_value("results.csv"), "File name to write results.")
                ("log-file",                 po::value<std::string>(),                               "File name for output log.")
                ("log-file-append",          po::value<bool>()->default_value(false),                "Append to log file.")
                ("log-level",                po::value<LogLevel>()->default_value(LogLevel::Debug),  "Log level")

                ("library-update-file",      po::value<std::string>()->default_value(""), "File name for writing indices "
                                                                                          "and speeds suitable for updating "
                                                                                          "an existing library logic file.")
                ("library-update-comment",   po::value<bool>()->default_value(false), "Include solution name as a "
                                                                                      "comment in library update "
                                                                                      "file.")

                ("a-ops",                    vector_default_empty<TensorOp>(), "Operations applied to A.")
                ("b-ops",                    vector_default_empty<TensorOp>(), "Operations applied to B.")
                ("c-ops",                    vector_default_empty<TensorOp>(), "Operations applied to C.")
                ("d-ops",                    vector_default_empty<TensorOp>(), "Operations applied to D.")

                ("exit-on-error",            po::value<bool>()->default_value(false), "Exit run early on failed kernels or other errors.")
                ("selection-only",           po::value<bool>()->default_value(false), "Don't run any solutions, only print kernel selections.")
                ("max-workspace-size",       po::value<size_t>()->default_value(32*1024*1024), "Max workspace for training")
                ("granularity-threshold",    po::value<double>()->default_value(0.0), "Don't run a solution if total granularity is below")
                ("prediction-threshold",     po::value<double>()->default_value(2.0), "Don't run a solution if predicted performance is low")

                ("activation-type",           po::value<ActivationType>()->default_value(ActivationType::None), "An activation type")
                ("activation-no-guard",          po::value<bool>()->default_value(false), "Use activation guard to deall with nan outputs.")
                ("activation-additional-args",vector_default_empty<std::string>(), "Activation additional floating-point number arguments.")
                ("activation-enum-args",      po::value<std::vector<ActivationType>>()->default_value(std::vector<ActivationType>(1, ActivationType::None), "[]"), "Activation enum argument.")
                ("hybrid-assignment-policy", po::value<std::vector<std::string>>(), "Launch policy for a selected WorkAssignment=Hybrid kernel: Default, DynamicWorkQueue, or Auto. Default preserves the existing behavior, including the heuristic when sm_count_target is positive. Each value runs the problem once.")
                ("streamk-hybrid-mode",       po::value<std::vector<int>>(), "Legacy alias for hybrid-assignment-policy: 0=Default, 1=DynamicWorkQueue, 2=Auto. Conflicting explicit aliases are rejected.")
                ("uniform-summation-order",   po::value<int>()->default_value(-1), "Override uniform summation order setting: -1=use default, 0=off, 1=on.")
                ("use-bias",                  po::value<int>()->default_value(0), "Use bias.")
                ("bias-source",               po::value<int>()->default_value(3), "Bias source.")
                ("use-scaleAB",               po::value<std::string>()->default_value(""), "Use scaleAB.")
                ("use-scaleCD",               po::value<bool>()->default_value(false), "Use scaleCD.")
                ("use-scaleAlphaVec",         po::value<int>()->default_value(0), "Alpha support: 0=host, 1/2/3=M/N/MN vector, 5/6/7=vector plus device scalar.")
                ("device-scalar-alpha",       po::value<bool>(), "Override the runtime alpha mode; defaults to the scalar bit in use-scaleAlphaVec.")
                ("bias-type-args",            po::value<std::vector<rocisa::DataType>>()->default_value(std::vector<rocisa::DataType>(1, rocisa::DataType::None), "[]"), "Bias data type args.")
                ("use-gate-residual",         po::value<bool>()->default_value(false), "Use gate residual.")
                ("gate-type-args",            po::value<std::vector<rocisa::DataType>>()->default_value(std::vector<rocisa::DataType>(1, rocisa::DataType::None), "[]"), "Gate residual data type args.")
                ("factor-dim-args",           po::value<std::vector<int>>()->default_value(std::vector<int>(1, 0), "[]"), "factor dimensions args.")
                ("icache-flush-args",         po::value<std::vector<bool>>()->default_value(std::vector<bool>(1, false), "[]"), "ICache flush args.")
                ("icache-rotate-copies",      po::value<int>()->default_value(0),
                    "Number of EXTRA hipModule_t copies of each --code-object file to load and "
                    "rotate per launch (I-cache cold-miss test). "
                    "0 = disabled, no extra copies (default). "
                    "N (>0) = load N extra copies (total = N+1 modules including the original). "
                    "-1 = auto: extras = max(inputArr.size()-1, cache-overflow term). "
                    "First term aligns with DataInit's rotating buffer cycle; "
                    "second term targets overflowing L1 I-cache: "
                    "Linux uses --icache-rotate-size*2*1024 / min(kernel_start->label_GW_End "
                    "across all --code-object); non-Linux uses --icache-rotate-size directly.")
                ("icache-rotate-size",        po::value<int>()->default_value(64),
                    "Cache budget (in KB) for the cache-overflow term of --icache-rotate-copies=-1. "
                    "Linux: effective bytes = N * 2 * 1024 (default 64 -> 128KB, ~2x typical L1 "
                    "I-cache); larger -> more copies loaded for the same per-kernel hot-path size. "
                    "Non-Linux: used directly as the extras count (no ELF parsing).")
                ("use-e",                     po::value<bool>()->default_value(false), "Use E.")
                ("use-gradient",              po::value<bool>()->default_value(false), "Use gradient.")
                ("use-user-args",             po::value<bool>()->default_value(false), "Use user argument structure as kernel input.")
                ("rotating-buffer-size",      po::value<int32_t>()->default_value(0), "Size of rotating buffer in the unit of MB.")
                ("rotating-buffer-mode",      po::value<int32_t>()->default_value(0), "Rotating mode.")
                ("output-amaxD",              po::value<bool>()->default_value(false), "Output AmaxD.")
                ("timing-instrumentation",    po::value<bool>()->default_value(false)->implicit_value(true), "Enable detailed timing instrumentation output to stderr.")
                ("rocprof-counter",           vector_default_empty<std::string>(), "Rocprof counters.")
                ("metadata-layout",           po::value<int32_t>()->default_value(0), "Sparse Metadata Layout")
                ;
            // clang-format on

            return options;
        }

        std::shared_ptr<Hardware> GetHardware(po::variables_map const& args)
        {
            int deviceCount = 0;
            HIP_CHECK_EXC(hipGetDeviceCount(&deviceCount));

            int deviceIdx = args["device-idx"].as<int>();

            if(deviceIdx >= deviceCount)
                throw std::runtime_error(concatenate(
                    "Invalid device index ", deviceIdx, " (", deviceCount, " total found.)"));

            HIP_CHECK_EXC(hipSetDevice(deviceIdx));

            return hip::GetCurrentDevice();
        }

        std::shared_ptr<MasterSolutionLibrary<ContractionProblemGemm>>
            LoadSolutionLibrary(po::variables_map const& args)
        {
            auto filename = args["library-file"];
            if(!filename.empty())
            {
                return std::dynamic_pointer_cast<MasterSolutionLibrary<ContractionProblemGemm>>(
                    LoadLibraryFile<ContractionProblemGemm>(filename.as<std::string>()));
            }

            auto embeddedLibrary
                = std::dynamic_pointer_cast<MasterSolutionLibrary<ContractionProblemGemm>>(
                    EmbeddedLibrary<ContractionProblemGemm>::Get());

            if(embeddedLibrary != nullptr)
                return embeddedLibrary;

            throw std::runtime_error("Client must be linked with an embedded library or "
                                     "a library must be specified at runtime.");
        }

        void LoadCodeObjects(po::variables_map const& args, hip::SolutionAdapter& adapter)
        {
            auto const& filenames = args["code-object"].as<std::vector<std::string>>();
            auto        logLevel  = args["log-level"].as<LogLevel>();

            if(filenames.empty())
            {
                adapter.loadEmbeddedCodeObjects();
            }
            else
            {
                //only trigger exception when failed to load all code objects.
                bool       loaded   = false;
                hipError_t retError = hipSuccess;

                for(auto const& filename : filenames)
                {
                    hipError_t ret;

                    if(logLevel >= LogLevel::Verbose)
                        std::cout << "Loading " << filename << std::endl;
                    ret = adapter.loadCodeObjectFile(filename);

                    if(ret == hipSuccess)
                        loaded = true;
                    else
                        retError = ret;
                }

                if(!loaded)
                    HIP_CHECK_EXC(retError);
            }
        }

        template <typename T>
        T parse_num(std::string const& s)
        {
            T                  t{};
            std::istringstream ss(s);
            ss >> t;
            if(!ss && !ss.eof())
                throw std::runtime_error("Failed to parse number: " + s);
            return t;
        }

        template <typename T>
        std::vector<T> split_nums(std::string const& value)
        {
            std::vector<std::string> parts = po::split_string(value);
            std::vector<T>           rv;
            rv.reserve(parts.size());
            for(auto const& part : parts)
                if(!part.empty())
                    rv.push_back(parse_num<T>(part));
            return rv;
        }

        template <typename T>
        void parse_arg_nums(po::variables_map& args, std::string const& name)
        {
            auto inValue = args[name].as<std::vector<std::string>>();

            std::vector<std::vector<T>> outValue;
            outValue.reserve(inValue.size());
            for(auto const& str : inValue)
                outValue.push_back(split_nums<T>(str));

            args.at(name).value() = std::any(outValue);
        }

        void parse_arg_bools(po::variables_map& args, std::string const& name)
        {
            auto opts             = args[name].as<std::vector<bool>>();
            args.at(name).value() = std::any(opts);
        }

        void parse_arg_ints(po::variables_map& args, std::string const& name)
        {
            parse_arg_nums<size_t>(args, name);
        }

        void parse_arg_double(po::variables_map& args, std::string const& name)
        {
            parse_arg_nums<double>(args, name);
        }

        void parse_bias_type_args(po::variables_map& args, std::string const& name)
        {
            auto type             = args[name].as<std::vector<rocisa::DataType>>();
            args.at(name).value() = std::any(type);
        }

        void parse_activation_enum_args(po::variables_map& args, std::string const& name)
        {
            auto type             = args[name].as<std::vector<ActivationType>>();
            args.at(name).value() = std::any(type);
        }

        void parse_activation_int(po::variables_map& args, std::string const& name)
        {
            auto type             = args[name].as<ActivationType>();
            args.at(name).value() = std::any(type);
        }

        template <typename T>
        void parse_arg_nums_map(po::variables_map& args, std::string const& name)
        {
            auto inValue = args[name].as<std::vector<std::string>>();

            std::map<int, int> outValue;
            for(auto const& str : inValue)
            {
                auto vec         = split_nums<T>(str);
                outValue[vec[0]] = vec[1];
            }

            args.at(name).value() = std::any(outValue);
        }

        void parse_arg_ints_map(po::variables_map& args, std::string const& name)
        {
            parse_arg_nums_map<int>(args, name);
        }

        void fix_data_types(po::variables_map& args)
        {
            auto type = args["type"].as<rocisa::DataType>();

            // These types use the same data type for all inputs/outputs, so we allow
            // using the overarching 'type' parameter.
            if(type == rocisa::DataType::Float || type == rocisa::DataType::Double
               || type == rocisa::DataType::ComplexFloat || type == rocisa::DataType::ComplexDouble
               || type == rocisa::DataType::Int32)
            {
                args.at("a-type").value()     = std::any(type);
                args.at("b-type").value()     = std::any(type);
                args.at("c-type").value()     = std::any(type);
                args.at("d-type").value()     = std::any(type);
                args.at("alpha-type").value() = std::any(type);
                args.at("beta-type").value()  = std::any(type);
            }
        }

        po::variables_map parse_args(int argc, const char* argv[])
        {
            auto options = all_options();

            po::variables_map args;
            po::store(po::parse_command_line(argc, argv, options), args);
            po::notify(args);

            if(args.count("help"))
            {
                std::cout << options << std::endl;
                exit(1);
            }

            if(args.count("config-file"))
            {
                auto configFiles = args["config-file"].as<std::vector<std::string>>();
                for(auto filename : configFiles)
                {
                    std::cout << "loading config file " << filename << std::endl;
                    std::ifstream file(filename.c_str());
                    if(file.bad())
                        throw std::runtime_error(concatenate("Could not open ", filename));
                    po::store(po::parse_config_file(file, options), args);
                }
            }

#if !(TENSILELITE_CLIENT_ENABLE_ROCPROFSDK)
            if(args["rocprof-counter"].as<std::vector<std::string>>().size())
            {
                throw std::runtime_error("rocprof-counter is provided but client is not built with -DTENSILELITE_CLIENT_ENABLE_ROCPROFSDK.");
            }
#endif
            // Only TopSolutionIterator, which --best-solution selects, reads the option; without
            // it the check would silently not run.
            if(args["require-selected-solution"].as<bool>() && !args["best-solution"].as<bool>())
                throw std::runtime_error(
                    "--require-selected-solution applies only with --best-solution");
            fix_data_types(args);

            parse_arg_ints(args, "problem-size");
            parse_arg_ints(args, "a-strides");
            parse_arg_ints(args, "b-strides");
            parse_arg_ints(args, "c-strides");
            parse_arg_ints(args, "d-strides");
            parse_arg_ints(args, "e-strides");
            parse_arg_ints(args, "bias-strides");
            parse_bias_type_args(args, "bias-type-args");
            parse_arg_ints(args, "gate-strides");
            parse_bias_type_args(args, "gate-type-args");
            parse_activation_int(args, "activation-type");
            parse_activation_enum_args(args, "activation-enum-args");
            parse_arg_double(args, "activation-additional-args");
            parse_arg_bools(args, "icache-flush-args");
            // std::cout << "Pasring parse_arg_ints_map()" << std::endl;
            parse_arg_ints_map(args, "prob-sol-map");
            return args;
        }

#if defined(__linux__)
        // Parse the AMDGPU ELF code object at `coPath` and return the smallest
        // distance (in bytes) from any FUNC/GLOBAL kernel-entry symbol to its
        // corresponding `label_GW_End` LOCAL symbol. This is a Tensile-specific
        // proxy for "I-cache hot-path size of the smallest kernel": label_GW_End
        // marks the first global-write-epilogue exit, which roughly bounds the
        // code that a launch executes through during the main compute path.
        //
        // Returns 0 when the file can't be opened, isn't ELF64, parsing fails,
        // or no FUNC+GLOBAL / label_GW_End symbol pairing is found.
        //
        // Linux-only (uses <elf.h>). Non-Linux build paths use the raw
        // --icache-rotate-size value instead.
        std::uintmax_t getMinKernelSizeToGwEnd(std::string const& coPath)
        {
            std::ifstream f(coPath, std::ios::binary);
            if(!f)
                return 0;

            Elf64_Ehdr eh{};
            f.read(reinterpret_cast<char*>(&eh), sizeof(eh));
            if(!f
               || std::memcmp(eh.e_ident, ELFMAG, SELFMAG) != 0
               || eh.e_ident[EI_CLASS] != ELFCLASS64)
                return 0;

            // Read all section headers.
            std::vector<Elf64_Shdr> shdrs(eh.e_shnum);
            f.seekg(eh.e_shoff);
            f.read(reinterpret_cast<char*>(shdrs.data()),
                   static_cast<std::streamsize>(eh.e_shnum * sizeof(Elf64_Shdr)));
            if(!f)
                return 0;

            // Find .symtab and its linked string table (sh_link).
            Elf64_Shdr const* symSh = nullptr;
            for(auto const& sh : shdrs)
                if(sh.sh_type == SHT_SYMTAB)
                {
                    symSh = &sh;
                    break;
                }
            if(!symSh
               || symSh->sh_link >= shdrs.size()
               || symSh->sh_size == 0
               || symSh->sh_size % sizeof(Elf64_Sym) != 0)
                return 0;

            auto const& strSh = shdrs[symSh->sh_link];

            // Read string table.
            std::vector<char> strs(strSh.sh_size);
            f.seekg(strSh.sh_offset);
            f.read(strs.data(), static_cast<std::streamsize>(strSh.sh_size));
            if(!f)
                return 0;

            // Read symbol table.
            auto const             symCount = symSh->sh_size / sizeof(Elf64_Sym);
            std::vector<Elf64_Sym> syms(symCount);
            f.seekg(symSh->sh_offset);
            f.read(reinterpret_cast<char*>(syms.data()),
                   static_cast<std::streamsize>(symSh->sh_size));
            if(!f)
                return 0;

            // Collect kernel entry addresses (FUNC + GLOBAL) and label_GW_End
            // addresses (we match by name; some toolchains emit different binds).
            std::vector<std::uint64_t> kernelStarts;
            std::vector<std::uint64_t> gwEndAddrs;
            for(auto const& s : syms)
            {
                if(s.st_name >= strs.size())
                    continue;
                char const*   name = &strs[s.st_name];
                unsigned char type = ELF64_ST_TYPE(s.st_info);
                unsigned char bind = ELF64_ST_BIND(s.st_info);

                if(type == STT_FUNC && bind == STB_GLOBAL)
                    kernelStarts.push_back(s.st_value);
                else if(std::strcmp(name, "label_GW_End") == 0)
                    gwEndAddrs.push_back(s.st_value);
            }

            if(kernelStarts.empty() || gwEndAddrs.empty())
                return 0;

            std::sort(kernelStarts.begin(), kernelStarts.end());

            // For each label_GW_End, the owning kernel is the one with the
            // largest start address <= this end address.
            std::uintmax_t minSize = std::numeric_limits<std::uintmax_t>::max();
            for(auto end : gwEndAddrs)
            {
                auto it = std::upper_bound(kernelStarts.begin(),
                                           kernelStarts.end(),
                                           end);
                if(it == kernelStarts.begin())
                    continue;
                std::uint64_t start = *(it - 1);
                if(end <= start)
                    continue;
                auto sz = static_cast<std::uintmax_t>(end - start);
                if(sz < minSize)
                    minSize = sz;
            }

            return (minSize == std::numeric_limits<std::uintmax_t>::max())
                       ? std::uintmax_t{0}
                       : minSize;
        }
#endif // defined(__linux__)

        void checkFusedA2AVerifyArgs(po::variables_map const& args)
        {
            VerifyArgs v;
            v.verify                = args["fused-a2a-verify"].as<int>();
            v.barrier               = args["fused-a2a-verify-barrier"].as<int>();
            v.fence                 = args["fused-a2a-verify-fence"].as<int>();
            v.skewUs                = args["fused-a2a-verify-skew-us"].as<int>();
            v.numElementsToValidate = args["num-elements-to-validate"].as<int>();
            v.numWarmups            = args["num-warmups"].as<int>();
            v.cEqualD               = args["c-equal-d"].as<bool>();
            v.sparse                = args["sparse"].as<int>() != 0;
            v.boundsCheckNaN = args["bounds-check"].as<BoundsCheckMode>() == BoundsCheckMode::NaN;
            v.scales = !args["use-scaleAB"].as<std::string>().empty()
                       || args["use-scaleCD"].as<bool>() || args["use-scaleAlphaVec"].as<int>() != 0
                       || args["mx-a-block"].as<int>() != 0 || args["mx-b-block"].as<int>() != 0;
            v.swizzled = args["swizzle-tensor-a"].as<bool>() || args["swizzle-tensor-b"].as<bool>();
            v.gemmA2A  = args["fused-a2a"].as<FusedA2A>() == FusedA2A::GemmA2A;
            v.betaZero = args["init-beta"].as<InitMode>() == InitMode::Zero;
            v.activation = args["activation-type"].as<ActivationType>() != ActivationType::None;
            v.bias       = args["use-bias"].as<int>() != 0;
            v.fusedA2A   = args["fused-a2a"].as<FusedA2A>() != FusedA2A::Disabled;
            const std::string error = verifyArgsError(v);
            if(!error.empty())
                throw std::runtime_error("[fused-a2a] " + error);
        }

        void checkFusedA2AArgs(po::variables_map const& args)
        {
            const int toValidate = args["num-elements-to-validate"].as<int>();
            if(toValidate > 0)
                throw std::runtime_error(
                    concatenate("[fused-a2a] --num-elements-to-validate=",
                                toValidate,
                                " is not supported. Use -1 for all, 0 for none."));
            for(auto name : {"init-a", "init-b"})
                if(!IsRandomLike(args[name].as<InitMode>()))
                    throw std::runtime_error(
                        concatenate("[fused-a2a] --",
                                    name,
                                    " must be Random, RandomNarrow, RandomNegPosLimited or "
                                    "UniformLowPrecision; got ",
                                    args[name].as<InitMode>()));
            if(args["device-idx"].as<int>() != 0)
                throw std::runtime_error(
                    "[fused-a2a] only --device-idx=0 is supported. Use HIP_VISIBLE_DEVICES to "
                    "pick the set.");
            auto const& flush = args["icache-flush-args"].as<std::vector<bool>>();
            if(std::find(flush.begin(), flush.end(), true) != flush.end())
                throw std::runtime_error(
                    "[fused-a2a] --icache-flush-args is not supported; leave it at false.");
            if(args["fused-a2a-drain-recv"].as<int>() == 0)
                throw std::runtime_error(
                    "[fused-a2a] --fused-a2a-drain-recv=0 is not supported; leave it at 1.");
            if(args["fused-a2a-drain-send"].as<int>() == 0)
                throw std::runtime_error(
                    "[fused-a2a] --fused-a2a-drain-send=0 is not supported; leave it at 1.");
            const int arm = args["fused-a2a-view-arm"].as<int>();
            if(arm != VIEW_ARM_DEVICE && arm != VIEW_ARM_HOST)
                throw std::runtime_error(
                    concatenate("[fused-a2a] --fused-a2a-view-arm must be 0 or 1; got ", arm, "."));
            if(args["fused-a2a-view-coalesce"].as<int>() < 1)
                throw std::runtime_error(
                    "[fused-a2a] --fused-a2a-view-coalesce must be at least 1.");
            const int viewDebug = args["fused-a2a-view-debug"].as<int>();
            if(viewDebug < 0 || viewDebug > VIEW_DEBUG_ALL)
                throw std::runtime_error(concatenate(
                    "[fused-a2a] --fused-a2a-view-debug must be in [0, 31]; got ", viewDebug, "."));
            if(args["use-user-args"].as<bool>())
                throw std::runtime_error("[fused-a2a] --use-user-args is not supported.");
            if(!args["library-update-file"].as<std::string>().empty())
                throw std::runtime_error("[fused-a2a] --library-update-file is not supported.");
            if(args["fused-a2a"].as<FusedA2A>() == FusedA2A::A2AGemm
               && args["batch-mode"].as<int>() != 0)
                throw std::runtime_error(
                    "[fused-a2a] A2AGemm supports only --batch-mode=0 (strided).");
            if(args["rotating-buffer-size"].as<int32_t>() > 0
               && args["rotating-buffer-mode"].as<int32_t>() == 1)
                throw std::runtime_error(
                    "[fused-a2a] --rotating-buffer-mode=1 is not supported; use mode 0.");
            if(!args["world-filter"].as<std::vector<int>>().empty()
               && !args["prob-sol-map"].as<std::map<int, int>>().empty())
                throw std::runtime_error(
                    "[fused-a2a] --world-filter cannot be combined with --prob-sol-map.");
        }

    } // namespace Client
} // namespace TensileLite

int main(int argc, const char* argv[])
{
    using namespace TensileLite;
    using namespace TensileLite::Client;

    auto const mainStart     = TimingClock::now();
    auto const mainStartWall = std::chrono::system_clock::now();

    // Declared first => destroyed last among main()'s locals. Stamps the wall clock once
    // every local (dataInit, library, hip buffers, ...) has been destroyed so the Python
    // side can separate "locals destruction" from "static destruction / process exit".
    struct LocalsDestroyedStamp
    {
        ~LocalsDestroyedStamp()
        {
            if(g_timingInstrumentationEnabled)
            {
                auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count();
                std::clog << "TIMING_TS:main_locals_destroyed:" << ns << "\n";
                std::clog.flush();
            }
        }
    } localsDestroyedStamp;

    auto args = parse_args(argc, argv);
    auto const argsParsed = TimingClock::now();

    // Enable timing instrumentation if requested
    g_timingInstrumentationEnabled = args["timing-instrumentation"].as<bool>();

    // Set srand
    unsigned int seed = args["init-seed"].as<unsigned int>();
    if(seed == 0)
    {
        seed = time(NULL);
    }
    std::cout << std::endl << "srand seed is set to " << seed << std::endl << std::endl;
    srand(seed);

    // Timing buffer is initialized before the problem factory so timers inside it work.
    auto const preCalibration = TimingClock::now();
    initTimingBuffer();
    calibrateTimingOverhead();
    auto const calibrated = TimingClock::now();

    ClientProblemFactory problemFactory(args);
    auto const factoryDone = TimingClock::now();
    // These happen before the timing buffer exists; record them now.
    if(g_timingInstrumentationEnabled)
    {
        g_timingBuffer.push_back(TimingRec{
            "main_parse_args",
            std::chrono::duration<double, std::milli>(argsParsed - mainStart).count()});
        g_timingBuffer.push_back(TimingRec{
            "main_timing_calibration",
            std::chrono::duration<double, std::milli>(calibrated - preCalibration).count()});
        g_timingBuffer.push_back(TimingRec{
            "main_problem_factory",
            std::chrono::duration<double, std::milli>(factoryDone - calibrated).count()});
    }

    const FusedA2A fusedMode = args["fused-a2a"].as<FusedA2A>();
    bool           fusedA2A  = fusedMode != FusedA2A::Disabled;
    if(fusedA2A)
        checkFusedA2AArgs(args);
    checkFusedA2AVerifyArgs(args);
    const int      viewArmMode  = args["fused-a2a-view-arm"].as<int>();
    const uint32_t viewCoalesce = (uint32_t)args["fused-a2a-view-coalesce"].as<int>();
    const int      viewDebug    = args["fused-a2a-view-debug"].as<int>();
    VerifyOptions  verifyOptions;
    verifyOptions.mode      = args["fused-a2a-verify"].as<int>();
    verifyOptions.barrier   = args["fused-a2a-verify-barrier"].as<int>() != 0;
    verifyOptions.skewUs    = (uint32_t)args["fused-a2a-verify-skew-us"].as<int>();
    verifyOptions.fence     = args["fused-a2a-verify-fence"].as<int>() != 0;
    auto const& worldFilter = args["world-filter"].as<std::vector<int>>();
    if(!fusedA2A && !worldFilter.empty())
        throw std::runtime_error("--world-filter applies only with --fused-a2a");
    if(!fusedA2A && args.count("fused-a2a-sdma-rotate"))
        throw std::runtime_error("--fused-a2a-sdma-rotate applies only with --fused-a2a");

    auto problems        = problemFactory.problems();
    int  firstProblemIdx = args["problem-start-idx"].as<int>();
    int  numProblems     = args["num-problems"].as<int>();
    if(numProblems < 0)
        numProblems = problems.size();
    int lastProblemIdx = firstProblemIdx + numProblems - 1;

    std::vector<int> problemIdxs;
    for(int i = firstProblemIdx; i <= lastProblemIdx; i++)
        if(worldFilter.empty()
           || std::count(worldFilter.begin(), worldFilter.end(), (int)problems[i]->world()))
            problemIdxs.push_back(i);

    std::shared_ptr<Hardware>                   hardware;
    std::vector<std::shared_ptr<DeviceContext>> devices;
    {
        ScopedTimer timer("hip_initialization");
        hardware     = GetHardware(args);
        size_t world = 1;
        for(int i : problemIdxs)
            world = std::max<size_t>(world, problems[i]->world());
        if(fusedA2A)
        {
            int deviceCount = 0;
            HIP_CHECK_EXC(hipGetDeviceCount(&deviceCount));
            if(world > (size_t)deviceCount)
                throw std::runtime_error(concatenate("[fused-a2a] the selected problems need ",
                                                     world,
                                                     " devices, found ",
                                                     deviceCount,
                                                     ". Narrow them with --world-filter."));
        }

        std::vector<int> deviceIds(world);
        std::iota(deviceIds.begin(), deviceIds.end(), args["device-idx"].as<int>());
        const int sdmaRotate = args.count("fused-a2a-sdma-rotate")
                                   ? args["fused-a2a-sdma-rotate"].as<int>()
                                   : kFusedA2ASdmaRotateDefault;
        for(int id : deviceIds)
            devices.push_back(
                std::make_shared<DeviceContext>(id,
                                                args["use-default-stream"].as<bool>(),
                                                fusedA2A ? kCommFlagsBytes : 0,
                                                fusedA2A ? deviceIds : std::vector<int>{},
                                                sdmaRotate));
    }
    std::shared_ptr<DeviceContext>              device0 = devices[0];
    std::vector<std::shared_ptr<DeviceContext>> active;

    std::shared_ptr<MasterSolutionLibrary<ContractionProblemGemm>> library;
    {
        ScopedTimer timer("library_loading");
        library = LoadSolutionLibrary(args);
        if(!library)
            throw std::runtime_error("Failed to load solution library");
    }

    auto filename = args["library-file"].as<std::string>();

    size_t      directoryPos     = filename.rfind('/');
    std::string libraryDirectory = filename;
    if(directoryPos != std::string::npos)
        libraryDirectory.resize(directoryPos + 1);
    else
        libraryDirectory = '.';

    // A failed primary code-object load may ask the adapter to reload its
    // helper HSACOs. Record that context before the first load can enter the
    // recovery path; actual helper loading remains below.
    forEachDevice(devices, [&](DeviceContext& ctx) {
        ctx.adapter->setLazyLoadingContext(hardware->archName(), libraryDirectory);
    });
#if TENSILELITE_CLIENT_ENABLE_ROCPROFSDK
    RocProfiler::getInstance().start();
#endif
    {
        ScopedTimer timer("code_object_loading");
        forEachDevice(devices, [&](DeviceContext& ctx) { LoadCodeObjects(args, *ctx.adapter); });
    }

    // I-cache rotation: load N extra independent hipModule_t copies of every
    // --code-object so each launch can use a different copy (different device PC).
    // --icache-rotate-copies=0 (default) = no extra copies, no rotation.
    // --icache-rotate-copies=N (N>0)     = load N extras (total = N+1 modules).
    {
        int icacheRotateCopies = args["icache-rotate-copies"].as<int>();
        if(icacheRotateCopies > 0)
        {
            ScopedTimer timer("icache_rotate_extra_copies_loading");
            auto const& filenames = args["code-object"].as<std::vector<std::string>>();
            forEachDevice(devices, [&](DeviceContext& ctx) {
                for(auto const& filename : filenames)
                    HIP_CHECK_EXC(
                        ctx.adapter->loadCodeObjectFileExtraCopies(filename, icacheRotateCopies));
            });
            std::cout << "Loaded " << icacheRotateCopies
                      << " extra rotation copies of each --code-object for I-cache test"
                      << " (total = " << (icacheRotateCopies + 1) << " modules)"
                      << std::endl;
        }
    }
#if TENSILELITE_CLIENT_ENABLE_ROCPROFSDK
    RocProfiler::getInstance().stop();
#endif

    {
        ScopedTimer timer("lazy_loading_init");
        forEachDevice(devices, [&](DeviceContext& ctx) {
            auto result
                = ctx.adapter->initializeLazyLoading(hardware->archName(), libraryDirectory);
            if(result != hipSuccess)
                throw std::runtime_error("Lazy loading failed. (" + std::to_string(int(result))
                                         + ").");
        });
    }

    // I-cache rotation (explicit --icache-rotate-copies=N): the main --code-object
    // rotation copies were loaded above (before lazy-load init), but the helper
    // bundle (Kernels.so) only becomes available in m_modules after
    // initializeLazyLoading. Mirror it into the rotation slots now so helper
    // kernels (PostGSU / betaOnly / reduction / conversion) resolve under rotation.
    {
        int icacheRotateCopies = args["icache-rotate-copies"].as<int>();
        if(icacheRotateCopies > 0)
            forEachDevice(devices, [&](DeviceContext& ctx) {
                HIP_CHECK_EXC(ctx.adapter->loadHelperKernelExtraCopies(icacheRotateCopies));
            });
    }

    bool        gpuTimer         = args["use-gpu-timer"].as<bool>();
    bool        runKernels       = !args["selection-only"].as<bool>();
    bool        exitOnError      = args["exit-on-error"].as<bool>();
    bool        groupedGemm      = args["grouped-gemm"].as<bool>();
    const auto& icacheFlushArgs  = args["icache-flush-args"].as<std::vector<bool>>();

    float skip_slow_solution_ratio = args["skip-slow-solution-ratio"].as<float>();
    if(skip_slow_solution_ratio > 1.0 || skip_slow_solution_ratio < 0.0)
    {
        std::cout << "Invalid Skip Slow Solution Ratio: " << skip_slow_solution_ratio << std::endl;
        std::cout << "Please Set Valid Ratio : (0.0 ~ 1.0)." << std::endl;
        exit(1);
    }

    {
        // Re-seed before data init: HIP runtime init above may consume rand() non-deterministically
        srand(seed);
        ScopedTimer timer("data_init_setup");
        forEachDevice(devices, [&](DeviceContext& ctx) {
            ctx.dataInit = std::make_shared<DataInitialization>(args, problemFactory);
        });
    }

    std::shared_ptr<SolutionIterator> solutionIterator;
    {
        ScopedTimer timer("solution_iterator_setup");
        solutionIterator = SolutionIterator::Default(library, hardware, args);
    }

    MetaRunListener listeners;
    std::shared_ptr<BenchmarkTimer>        benchmarkTimer;
    std::shared_ptr<SynchronizerValidator> synchronizerValidator;
    std::shared_ptr<ReferenceValidator>    referenceValidator;
    float                                  flushTimeMs{};

    {
        ScopedTimer timer("listener_setup");
        for(auto const& ctx : devices)
            listeners.addListener(ctx->dataInit);
        listeners.addListener(solutionIterator);
        listeners.addListener(std::make_shared<ProgressListener>(args));

        if(runKernels)
        {
            bool hasIcacheFlush
                = std::any_of(begin(icacheFlushArgs), end(icacheFlushArgs), [](auto i) { return i; });
            flushTimeMs
                = hasIcacheFlush ? estimate_flush_kernel_time(device0->stream, gpuTimer) : 0.f;
            // Before ReferenceValidator: postSolution runs listeners in
            // reverse, so the dirty-buffer verdict lands after the reference
            // one rather than being overwritten by it.
            synchronizerValidator = std::make_shared<SynchronizerValidator>(args);
            listeners.addListener(synchronizerValidator);
            referenceValidator = std::make_shared<ReferenceValidator>(args);
            listeners.addListener(referenceValidator);
            benchmarkTimer = std::make_shared<BenchmarkTimer>(args, *hardware, flushTimeMs * 1000);
            listeners.addListener(benchmarkTimer);
            listeners.addListener(std::make_shared<HardwareMonitorListener>(args));
#if TENSILELITE_CLIENT_ENABLE_ROCPROFSDK
            if (!args["rocprof-counter"].as<std::vector<std::string>>().empty())
                listeners.addListener(Profiler::Default(args));
#endif
        }
    }

    std::shared_ptr<MetaResultReporter> reporters;
    {
        ScopedTimer timer("reporter_setup");
        reporters = std::make_shared<MetaResultReporter>();
        reporters->addReporter(PerformanceReporter::Default(args));

        // PerformanceReporter needs to be called before these two, or else values
        // will be missing
        reporters->addReporter(LogReporter::Default(args));
        reporters->addReporter(ResultFileReporter::Default(args));
        reporters->addReporter(LibraryUpdateReporter::Default(args));

        if(args.count("log-file"))
        {
            std::string filename = args["log-file"].as<std::string>();
            auto        logFile  = std::make_shared<std::ofstream>(
                filename.c_str(), args["log-file-append"].as<bool>() ? std::ios::app : std::ios::out);

            reporters->addReporter(LogReporter::Default(args, logFile, LogLevel::Normal));
        }

        listeners.setReporter(reporters);
        listeners.setDevices(&active);
    }

    // ReferenceValidator validator(args, dataInit);
    // BenchmarkTimer timer(args);

    reporters->report(ResultKey::ProblemCount, problemFactory.problems().size());

    bool  useUserArgs = args["use-user-args"].as<bool>();
    void* dUA         = nullptr;
    void* dUAHost     = nullptr;

    if(Debug::Instance().getBenchmark())
    {
        std::stringstream ss;
        static_cast<void>(benchmarkAllocation(ss));
        std::cout << ss.str();
    }

    while(listeners.needMoreBenchmarkRuns())
    {
        listeners.preBenchmarkRun();
        const auto flushGridSize = flush_grid_size();
        for(auto icacheFlush : icacheFlushArgs)
        {
            benchmarkTimer->setIFlushTimeUs(icacheFlush ? flushTimeMs * 1000 : 0.f);

            // I-cache rotation counter: rotationLaunchIdx
            // for scope across all problem / iterations, accumulate continuously
            // to avoid starting from copy 0 for each problem
            size_t rotationLaunchIdx = 0;
            for(int problemIdx : problemIdxs)
            {
                // Declared first => destroyed last: problem_total covers the whole body,
                // problem_teardown covers destruction of the body's locals (see mark() below).
                ScopedTimer          problemTotalTimer("problem_total");
                ScopeTeardownTimer   problemTeardownTimer("problem_teardown");
                auto problem = problems[problemIdx].get();
                active.assign(devices.begin(), devices.begin() + problem->world());

                {
                    ScopedTimer timer("problem_report_index");
                    reporters->report(ResultKey::ProblemIndex, problemIdx);
                    reporters->report(ResultKey::ProblemProgress,
                                      concatenate(problemIdx, "/", lastProblemIdx));
                }

                {
                    ScopedTimer timer("pre_problem");
                    listeners.preProblem(problem);
                }
                {
                    ScopedTimer timer("gpu_input_preparation");
                    forEachDevice(active, [&](DeviceContext& ctx) {
                        ctx.inputs = ctx.dataInit->prepareGPUInputs(problem);
                    });
                }

                size_t warmupInvocations    = listeners.numWarmupRuns();
                size_t syncs                = listeners.numSyncs();
                size_t enq                  = listeners.numEnqueuesPerSync();
                size_t maxRotatingBufferNum = std::max(warmupInvocations, syncs * enq);

                {
                    ScopedTimer timer("rotating_buffer_preparation");
                    forEachDevice(active, [&](DeviceContext& ctx) {
                        ctx.inputArr = ctx.dataInit->prepareRotatingGPUOutput(
                            maxRotatingBufferNum, problem, ctx.inputs, ctx.stream);
                        static_cast<void>(hipDeviceSynchronize());
                    });
                }

                // I-cache rotation auto-load: when --icache-rotate-copies=-1,
                // pick the larger of two extra-copy counts:
                //   1) inputArr.size() - 1
                //      Aligns the code rotation cycle with DataInit's actual
                //      data buffer rotation cycle (inputArr.size() = 1 original
                //      + rotatingNum extras).
                //   2) Cache-overflow term targeting ~128KB of L1 I-cache.
                //      On Linux: parse each .co's ELF symbol table to find the
                //      smallest kernel hot-path size K (kernel_start ->
                //      label_GW_End), then load (128KB / K) - 1 extras so the
                //      total rotated code (~128KB) overflows typical L1 I-cache.
                //      On non-Linux: <elf.h> is unavailable; fall back to
                //      args["icache-rotate-size"] as the raw extras count.
                // Use adapter.numRotationModules()==1 as the "not yet loaded"
                // guard so the load only fires on the first problem; subsequent
                // problems reuse the same N.
                {
                    int icacheArg = args["icache-rotate-copies"].as<int>();
                    if(icacheArg == -1 && device0->adapter->numRotationModules() == 1)
                    {
                        auto const& filenames
                            = args["code-object"].as<std::vector<std::string>>();

                        // Term 1: align with data rotation cycle.
                        int extrasFromDataInit = static_cast<int>(device0->inputArr.size()) - 1;

                        // Term 2: cache-overflow term.
#if defined(__linux__)
                        // K = min(kernel_start -> label_GW_End) across all .co.
                        std::uintmax_t K = 0;
                        for(auto const& filename : filenames)
                        {
                            std::uintmax_t sz = getMinKernelSizeToGwEnd(filename);
                            if(sz > 0 && (K == 0 || sz < K))
                                K = sz;
                        }
                        // kCacheBudgetBytes = icache-rotate-size * 2 * 1024.
                        // Default icache-rotate-size=64 -> 128 KB
                        // Loosely targets ~2x typical L1 I-cache.
                        int rotateSizeKB = args["icache-rotate-size"].as<int>();
                        if(rotateSizeKB < 0)
                            rotateSizeKB = 0;
                        std::uintmax_t kCacheBudgetBytes = std::uintmax_t(rotateSizeKB) * 2 * 1024;
                        int extrasFromCache = 0;
                        if(K == 0)
                        {
                            std::cerr << "[icache-rotate] warning: no label_GW_End "
                                    << "found in any --code-object; cache-based "
                                    << "term contributes 0" << std::endl;
                        }
                        else if(kCacheBudgetBytes > K)
                        {
                            extrasFromCache = static_cast<int>(kCacheBudgetBytes / K - 1);
                        }
#else
                        // <elf.h> unavailable on this platform; use the raw
                        // --icache-rotate-size value as the extras count.
                        int extrasFromCache
                            = static_cast<int>(args["icache-rotate-size"].as<int>());
#endif

                        int extras = std::max(extrasFromDataInit, extrasFromCache);
                        if(extras > 0)
                        {
                            ScopedTimer timer("icache_rotate_extra_copies_loading");
                            forEachDevice(devices, [&](DeviceContext& ctx) {
                                for(auto const& filename : filenames)
                                    HIP_CHECK_EXC(ctx.adapter->loadCodeObjectFileExtraCopies(
                                        filename, extras));
                                // Mirror the helper bundle (Kernels.so) into every
                                // rotation slot so helper kernels (PostGSU / betaOnly /
                                // reduction / conversion) resolve under rotation instead
                                // of only from copy 0. The helper is already loaded into
                                // m_modules by initializeLazyLoading at this point.
                                HIP_CHECK_EXC(ctx.adapter->loadHelperKernelExtraCopies(extras));
                            });
                        }
#if defined(__linux__)
                        std::cout << "[icache-rotate] auto extras = max("
                                  << extrasFromDataInit << " from inputArr.size()-1, "
                                  << extrasFromCache << " from "
                                  << kCacheBudgetBytes << "/" << K << ") = "
                                  << extras
                                  << " (total = " << (extras + 1) << " modules)"
                                  << std::endl;
#else
                        std::cout << "[icache-rotate] auto extras = max("
                                  << extrasFromDataInit << " from inputArr.size()-1, "
                                  << extrasFromCache << " from --icache-rotate-size) = "
                                  << extras
                                  << " (total = " << (extras + 1) << " modules)"
                                  << std::endl;
#endif
                    }
                }

                // The first per-solution iteration must re-upload inputs so that
                // the upload happens after preSolution() and can read the picked
                // solution's problemType.mxScaleFormat to pick the correct host
                // upload layout for MX scale tensors. The extra upload is a no-op
                // cost on non-MX problems.
                bool resetInput = true;
                // --fused-a2a-verify: goldenA and goldenB are taken once per problem.
                bool verifyInputsCaptured = false;
                while(solutionIterator->moreSolutionsInProblem())
                {
                    ScopedTimer        solutionTotalTimer("solution_total");
                    ScopeTeardownTimer solutionTeardownTimer("solution_teardown");
                    std::shared_ptr<ContractionSolution> solution;
                    {
                        ScopedTimer timer("solution_selection");
                        solution = solutionIterator->getSolution();
                    }
                    if(solution == nullptr)
                        throw std::runtime_error("Could not find a solution");

                    // WorkgroupMappingXCCCheck admits CU-fallback solutions on the basis that
                    // they launch at WGMXCC=1, as hipBLASLt does; zero keeps the solution's own
                    // mapping.
                    if(auto gemm = dynamic_cast<ContractionProblemGemm*>(problem);
                       gemm && !fusedA2A)
                        gemm->setParams().setWGMXCC(solution->isFallbackForHW(*hardware) ? 1 : 0);

                    {
                        ScopedTimer timer("pre_solution");
                        listeners.preSolution(solution.get());
                    }
                    bool deviceLost = false;
                    bool runThisSolution = false;
                    VerifyLayout verifyLayoutOf;
                    VerifyState  verifyState;
                    // Set by the first producer; the inputs are restored after the solution.
                    bool verifyProduced = false;
                    {
                        ScopedTimer timer("run_current_solution_check");
                        runThisSolution = solutionIterator->runCurrentSolution() && runKernels;
                    }
                    if(runThisSolution)
                    {
                        try
                        {
                            checkFusedA2ASolution(*problem, *solution, *hardware, active.size());
                            const bool verify = fusedA2A && verifyOptions.mode != VERIFY_OFF;
                            if(verify)
                            {
                                verifyLayoutOf = verifyLayout(*problem, active.size());
                                verifyState.reset(device0->inputArr.size());
                            }
                            resetCommBuffers(*problem,
                                             *solution,
                                             active,
                                             viewDebug,
                                             verify ? &verifyLayoutOf : nullptr);
                            const bool viewTransport = fusedA2A && fusedA2AViewTransport(*solution);
                            const bool inKernelNoops
                                = fusedA2A && !viewTransport
                                  && (viewDebug & VIEW_DEBUG_INKERNEL_NOOPS) != 0;
                            while(listeners.needMoreRunsInSolution())
                            {
                                ScopedTimer        runIterTotalTimer("run_iteration_total");
                                ScopeTeardownTimer runIterTeardownTimer("run_iteration_teardown");
                                verifyState.iterations++;
                                if(resetInput)
                                {
                                    if(verify)
                                        syncAll(active);
                                    ScopedTimer timer("gpu_input_reset");
                                    struct ResetScope
                                    {
                                        ResetScope() { g_inGpuInputReset = true; }
                                        ~ResetScope() { g_inGpuInputReset = false; }
                                    } resetScope;
                                    forEachDevice(active, [&](DeviceContext& ctx) {
                                        ctx.inputArr[0] = ctx.dataInit->prepareGPUInputs(problem);
                                    });
                                }
                                resetInput = true;
                                synchronizerValidator->beginRunIteration();

                                std::vector<std::vector<std::vector<KernelInvocation>>> kernels;
                                std::vector<std::vector<ViewArmArgs>>                   viewArgs;
                                {
                                    ScopedTimer  timer("kernel_solving");
                                    const size_t slots    = device0->inputArr.size();
                                    const size_t channels = fusedA2A ? kCommChannels : 1;
                                    kernels.assign(
                                        std::lcm(slots, channels),
                                        std::vector<std::vector<KernelInvocation>>(active.size()));
                                    if(viewTransport)
                                        viewArgs.assign(kernels.size(),
                                                        std::vector<ViewArmArgs>(active.size()));
                                    for(size_t k = 0; k < kernels.size(); k++)
                                        for(size_t d = 0; d < active.size(); d++)
                                        {
                                            auto&                ctx = *active[d];
                                            ScopedDevice         device(ctx.deviceId);
                                            ContractionInputs    scratch;
                                            ProblemInputs const* inputs = rankInputs(
                                                scratch, *problem, active, d, k, viewTransport);
                                            if(viewTransport)
                                                viewArgs[k][d] = makeViewArmArgs(*problem,
                                                                                 *solution,
                                                                                 active,
                                                                                 d,
                                                                                 k,
                                                                                 viewCoalesce,
                                                                                 *hardware,
                                                                                 viewDebug);
                                            kernels[k][d]
                                                = useUserArgs
                                                      ? solution->solveTensileGPU(*problem,
                                                                                  *inputs,
                                                                                  *hardware,
                                                                                  &dUA,
                                                                                  &dUAHost,
                                                                                  nullptr,
                                                                                  0,
                                                                                  ctx.stream)
                                                      : solution->solve(*problem,
                                                                        *inputs,
                                                                        *hardware,
                                                                        nullptr,
                                                                        0,
                                                                        ctx.stream);
                                        }
                                }

                                auto const   warmupSetupStart  = TimingClock::now();
                                size_t       warmupInvocations = listeners.numWarmupRuns();
                                size_t       warmupEventCount  = kernels[0][0].size();
                                TimingEvents warmupStartEvents(
                                    active, warmupInvocations, warmupEventCount);
                                TimingEvents warmupStopEvents(
                                    active, warmupInvocations, warmupEventCount);
                                recordTimingSince("warmup_events_setup", warmupSetupStart);

                                // I-cache rotation counter: increments per launchKernels call
                                // and selects which hipModule_t copy services the next launch.
                                // numRotationModules()==1 → no rotation (always copy 0).
                                auto rotateAndSelect = [&]() {
                                    for(auto const& ctx : devices)
                                        ctx->adapter->selectRotationCopy(
                                            (int)(rotationLaunchIdx
                                                  % (size_t)ctx->adapter->numRotationModules()));
                                    rotationLaunchIdx++;
                                };

                                // One counter across warmups, the profiler run and every sync.
                                size_t launchIdx = 0;
                                auto   launchAll = [&](TimingEvents const* startEvents,
                                                       TimingEvents const* stopEvents,
                                                       size_t              invocation) {
                                    const size_t launchSlot = launchIdx++ % kernels.size();
                                    auto const&  launch     = kernels[launchSlot];
                                    const size_t dataSlot
                                        = verify ? launchSlot % device0->inputArr.size() : 0;
                                    uint32_t verifyLaunch = 0;
                                    if(verify)
                                    {
                                        verifyLaunch = verifyState.beginLaunch(dataSlot);
                                        const bool barrier
                                            = verifyBarrier(verifyOptions, verifyLaunch);
                                        if(barrier)
                                            syncAll(active);
                                        if(!verifyInputsCaptured)
                                        {
                                            forEachDevice(active, [&](DeviceContext& ctx) {
                                                verifyCaptureInputs(ctx, verifyLayoutOf);
                                            });
                                            verifyInputsCaptured = true;
                                        }
                                        verifyProduced = true;
                                        for(size_t d = 0; d < active.size(); d++)
                                        {
                                            ScopedDevice device(active[d]->deviceId);
                                            verifyState.addSkew(verifyProduce(*active[d],
                                                                              verifyLayoutOf,
                                                                              dataSlot,
                                                                              verifyLaunch,
                                                                              verifyOptions,
                                                                              (uint32_t)d));
                                        }
                                        if(barrier)
                                            syncAll(active);
                                    }
                                    for(size_t d = 0; d < active.size(); d++)
                                    {
                                        auto&        ctx = *active[d];
                                        ScopedDevice device(ctx.deviceId);
                                        if(viewTransport)
                                        {
                                            // The events bracket the arm, the GEMM and the
                                            // wait.
                                            ViewArmArgs view = viewArgs[launchSlot][d];
                                            if(verify)
                                                view.debug
                                                    = verifyViewDebug(view.debug, verifyLaunch);
                                            if(startEvents)
                                                HIP_CHECK_EXC(hipEventRecord(
                                                    startEvents->card(d)[invocation].front(),
                                                    ctx.stream));
                                            try
                                            {
                                                viewArm(view, ctx.stream, viewArmMode);
                                                HIP_CHECK_EXC(ctx.adapter->launchKernels(
                                                    launch[d], ctx.stream, nullptr, nullptr));
                                            }
                                            catch(...)
                                            {
                                                viewAbort(view, ctx.stream);
                                                throw;
                                            }
                                            viewWait(view, ctx.stream);
                                            if(stopEvents)
                                                HIP_CHECK_EXC(hipEventRecord(
                                                    stopEvents->card(d)[invocation].back(),
                                                    ctx.stream));
                                        }
                                        else if(inKernelNoops)
                                        {
                                            // The events bracket both empty kernels and the
                                            // GEMM.
                                            if(startEvents)
                                                HIP_CHECK_EXC(hipEventRecord(
                                                    startEvents->card(d)[invocation].front(),
                                                    ctx.stream));
                                            viewNoop(ctx.stream);
                                            HIP_CHECK_EXC(ctx.adapter->launchKernels(
                                                launch[d], ctx.stream, nullptr, nullptr));
                                            viewNoop(ctx.stream);
                                            if(stopEvents)
                                                HIP_CHECK_EXC(hipEventRecord(
                                                    stopEvents->card(d)[invocation].back(),
                                                    ctx.stream));
                                        }
                                        else if(startEvents)
                                            HIP_CHECK_EXC(ctx.adapter->launchKernels(
                                                launch[d],
                                                ctx.stream,
                                                startEvents->card(d)[invocation],
                                                stopEvents->card(d)[invocation]));
                                        else
                                            HIP_CHECK_EXC(ctx.adapter->launchKernels(
                                                launch[d], ctx.stream, nullptr, nullptr));
                                        if(verify && verifyState.checks())
                                            verifyCheck(
                                                ctx, verifyLayoutOf, dataSlot, verifyLaunch);
                                    }
                                };

                                // Whether this run iteration launched anything (for the
                                // end-of-iteration attribution sync below).
                                bool iterationLaunched = false;

                                if(warmupInvocations > 0)
                                {
                                    {
                                        ScopedTimer timer("warmup_runs");
                                        listeners.preWarmup();
                                        launchAll(&warmupStartEvents, &warmupStopEvents, 0);
                                        iterationLaunched = true;
                                    }

                                    // Attribution only: wait for warmup 0 under its own timer, so
                                    // the validators' readbacks do not absorb the kernel's runtime.
                                    // Not done without instrumentation, to keep the launch/host
                                    // overlap of production runs.
                                    if(g_timingInstrumentationEnabled)
                                    {
                                        ScopedTimer timer("warmup_first_sync");
                                        forEachDevice(active, [](DeviceContext& ctx) {
                                            HIP_CHECK_EXC(hipStreamSynchronize(ctx.stream));
                                        });
                                    }

                                    {
                                        ScopedTimer timer("validate_warmups");
                                        listeners.validateWarmups(
                                            {0}, warmupStartEvents, warmupStopEvents);
                                    }
                                    if(verify && verifyState.golden == VerifyGolden::Pending)
                                    {
                                        verifyState.golden = verifyGoldenAfterValidation(
                                            verifyState.golden,
                                            referenceValidator->validatedSolution(),
                                            referenceValidator->errorInSolution());
                                        if(verifyState.golden == VerifyGolden::Captured)
                                            forEachDevice(active, [&](DeviceContext& ctx) {
                                                verifyCaptureGolden(ctx, verifyLayoutOf);
                                            });
                                        else if(verifyState.golden == VerifyGolden::Disabled)
                                            reporters->log(LogLevel::Error,
                                                           "fused-A2A verify: golden not captured, "
                                                           "verify is off for this solution.\n");
                                    }

                                    {
                                        ScopedTimer timer("warmup_runs");
                                        for(int i = 1; i < warmupInvocations; i++)
                                            launchAll(&warmupStartEvents, &warmupStopEvents, i);
                                        listeners.postWarmup(warmupStartEvents, warmupStopEvents);
                                    }
                                }

#if TENSILELITE_CLIENT_ENABLE_ROCPROFSDK
                                TimingEvents ProfilerStartEvents(active, 1, warmupEventCount);
                                TimingEvents ProfilerStopEvents(active, 1, warmupEventCount);
                                listeners.preProfiler();
                                rotateAndSelect();
                                launchAll(&ProfilerStartEvents, &ProfilerStopEvents, 0);
                                iterationLaunched = true;
                                listeners.postProfiler();
#endif

                                size_t syncs      = listeners.numSyncs();
                                size_t enq        = listeners.numEnqueuesPerSync();
                                size_t eventCount = gpuTimer ? warmupEventCount : 0;
                                size_t slotCount  = device0->inputArr.size();

                                {
                                    ScopedTimer timer("benchmark_runs");
                                    {
                                        ScopedTimer preSyncsTimer("benchmark_pre_syncs");
                                        listeners.preSyncs();
                                    }
                                    if(enq)
                                        for(int i = 0; i < syncs; i++)
                                        {
                                            // Declared first => destroyed last: times the event
                                            // destruction at the end of the iteration (mark() below).
                                            ScopeTeardownTimer eventsTeardownTimer(
                                                "benchmark_events_teardown");
                                            auto const eventsSetupStart = TimingClock::now();
                                            TimingEvents startEvents(active, enq, eventCount);
                                            TimingEvents stopEvents(active, enq, eventCount);
                                            std::vector<size_t> slots;
                                            for(size_t j = 0; j < std::min(enq, slotCount); j++)
                                                slots.push_back((launchIdx + j) % slotCount);
                                            recordTimingSince("benchmark_events_setup",
                                                              eventsSetupStart);

                                            {
                                                ScopedTimer kernelTimer("gpu_kernel_execution");
                                                listeners.preEnqueues();

                                                for(int j = 0; j < enq; j++)
                                                {
                                                    rotateAndSelect();
                                                    launchAll(nullptr, nullptr, 0);
                                                    iterationLaunched = true;

                                                    if(icacheFlush)
                                                    {
                                                        hipLaunchKernelGGL(flush_icache,
                                                                           flushGridSize,
                                                                           64,
                                                                           0,
                                                                           device0->stream);
                                                    }
                                                }

                                                listeners.postEnqueues(startEvents, stopEvents);
                                            }
                                            {
                                                ScopedTimer validateEnqueuesTimer(
                                                    "benchmark_validate_enqueues");
                                                if(verify)
                                                {
                                                    syncAll(active);
                                                    for(auto const& [slot, e] :
                                                        verifyState.takeLaunched())
                                                        forEachDevice(
                                                            active, [&](DeviceContext& ctx) {
                                                                verifyNormalize(
                                                                    ctx, verifyLayoutOf, slot, e);
                                                            });
                                                    syncAll(active);
                                                }
                                                listeners.validateEnqueues(
                                                    slots, startEvents, stopEvents);
                                            }
                                            eventsTeardownTimer.mark();
                                        }

                                    {
                                        ScopedTimer postSyncsTimer("benchmark_post_syncs");
                                        listeners.postSyncs();
                                    }
                                }

                                // Attribution only: wait for everything this iteration queued
                                // (e.g. warmups 1..N-1 with SyncsPerBenchmark: 0) under its own
                                // timer. Otherwise the next solution's first blocking call (e.g.
                                // its input reset or GPU readback validation) or an
                                // end-of-iteration check would absorb it. Inside the try: a kernel
                                // fault surfacing here takes the device-lost path below.
                                if(g_timingInstrumentationEnabled && iterationLaunched)
                                {
                                    ScopedTimer timer("run_iteration_sync");
                                    forEachDevice(active, [](DeviceContext& ctx) {
                                        HIP_CHECK_EXC(hipStreamSynchronize(ctx.stream));
                                    });
                                }

                                if(useUserArgs)
                                {
                                    solution->relaseDeviceUserArgs(dUA, dUAHost);
                                }

                                // Check the Synchronizer after all of this iteration's launches,
                                // before the next iteration's input reset (GuardPageAll runs two
                                // iterations). The verdict is reported in postSolution. It can
                                // throw, so it follows the release of the user args.
                                synchronizerValidator->checkRunIteration();
                                runIterTeardownTimer.mark();
                            }
                            if(verify)
                                std::cout
                                    << "VERIFY_SOLUTION world=" << active.size()
                                    << " launches=" << verifyState.launches << " golden="
                                    << (verifyState.golden == VerifyGolden::Captured   ? "captured"
                                        : verifyState.golden == VerifyGolden::Disabled ? "disabled"
                                                                                       : "pending")
                                    << " slots=" << verifyState.last.size()
                                    << " iterations=" << verifyState.iterations
                                    << " skewed=" << verifyState.skewed
                                    << " skew_ticks=" << verifyState.skewTicks
                                    << " name=" << solution->name() << std::endl;
                            forEachDevice(active, [](DeviceContext&) {
                                HIP_CHECK_EXC(hipDeviceSynchronize());
                            });
                        }
                        catch(std::runtime_error const& err)
                        {
                            reporters->report(ResultKey::Validation, "INVALID");
                            reporters->log(LogLevel::Error,
                                           concatenate("Exception occurred: ", err.what(), "\n"));
                            // A sticky error (e.g. a kernel memory fault) leaves the device unusable.
                            for(auto const& ctx : devices)
                            {
                                hipError_t syncErr = hipSetDevice(ctx->deviceId);
                                if(syncErr == hipSuccess)
                                    syncErr = hipDeviceSynchronize();
                                if(syncErr != hipSuccess)
                                {
                                    deviceLost = true;
                                    reporters->log(LogLevel::Error,
                                                   concatenate("Device unusable (",
                                                               hipGetErrorName(syncErr),
                                                               "), aborting.\n"));
                                }
                            }
                            static_cast<void>(hipSetDevice(device0->deviceId));
                        }
                        for(auto const& ctx : devices)
                            if(!viewDrainQueues(*ctx))
                            {
                                deviceLost = true;
                                reporters->log(
                                    LogLevel::Error,
                                    concatenate("SDMA queues of device ",
                                                ctx->deviceId,
                                                " did not drain after a FusedA2ATransport=View "
                                                "solution, aborting.\n"));
                            }
                        if(verifyProduced && !deviceLost)
                        {
                            try
                            {
                                syncAll(active);
                                forEachDevice(active, [&](DeviceContext& ctx) {
                                    verifyRestore(ctx, verifyLayoutOf);
                                });
                                syncAll(active);
                            }
                            catch(std::runtime_error const& err)
                            {
                                reporters->log(
                                    LogLevel::Error,
                                    concatenate("fused-A2A verify: inputs not restored: ",
                                                err.what(),
                                                "\n"));
                            }
                        }
                        static_cast<void>(hipSetDevice(device0->deviceId));
                    }

                    {
                        ScopedTimer timer("post_solution");
                        listeners.postSolution();
                    }
                    solutionTeardownTimer.mark();

                    if(deviceLost || (exitOnError && listeners.error() > 0))
                    {
                        flushTimingBuffer();
                        // error range in shell is [0-255]
                        return std::clamp(listeners.error(), 1, 255);
                    }
                }

                {
                    ScopedTimer timer("post_problem");
                    listeners.postProblem();
                }
                problemTeardownTimer.mark();
            }
        }

        listeners.postBenchmarkRun();
    }

    {
        ScopedTimer timer("finalize_report");
        listeners.finalizeReport();
    }

    // Flush all buffered timing records to stderr.
    // Timed with raw chrono instead of ScopedTimer because ScopedTimer pushes
    // its record into the buffer on destruction — after flushTimingBuffer()
    // has already drained and cleared it, so the record would be lost.
    {
        auto flushStart = TimingClock::now();
        recordTimingSince("main_total", mainStart);
        flushTimingBuffer();
        auto flushMs = std::chrono::duration<double, std::milli>(
            TimingClock::now() - flushStart).count();
        std::clog << "TIMING:flush_timing_buffer:" << flushMs << "\n";
        if(g_timingInstrumentationEnabled)
        {
            // Wall-clock stamps so the Python side can split process startup / teardown.
            auto toNs = [](std::chrono::system_clock::time_point t) {
                return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch())
                    .count();
            };
            std::clog << "TIMING_TS:main_start:" << toNs(mainStartWall) << "\n"
                      << "TIMING_TS:main_end:" << toNs(std::chrono::system_clock::now()) << "\n";
            std::clog.flush();
        }
    }

    // error range in shell is [0-255]
    return std::min(listeners.error(), 255);
}
