// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Benchmark logic for the fused GEMM + all-to-all epilogue. One rank per
// process, started by an external launcher; identity and rendezvous come from
// the environment variables torchrun sets.

#include "collective_rendezvous.hpp"
#include "hipblaslt_arguments.hpp"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <SdmaQueue.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace hipblaslt_bench
{
    constexpr int    kRendezvousTimeoutSec = 60;
    constexpr size_t kWorkspaceSize        = 128ull * 1024 * 1024;

    struct RankResources
    {
        hipblasLtHandle_t                  handle    = nullptr;
        hipblasLtFusedEpilogueDescriptor_t fused     = nullptr;
        hipblasLtMatmulDesc_t              mm        = nullptr;
        hipblasLtMatrixLayout_t            lay[4]    = {};
        hipblasLtMatmulPreference_t        pref      = nullptr;
        hipStream_t                        stream    = nullptr;
        void*                              dA        = nullptr;
        void*                              dB        = nullptr;
        void*                              dC        = nullptr;
        void*                              dD        = nullptr;
        void*                              dRecv     = nullptr;
        void*                              workspace = nullptr;
        void*                recvPtrs[HIPBLASLT_DEVICE_COMM_MAX_WORLD] = {};
        hipblasLtSdmaQueue_t queues[HIPBLASLT_DEVICE_COMM_MAX_WORLD]   = {};
        std::vector<std::unique_ptr<TensileLite::Client::SdmaQueue>> ownedQueues;
    };

    inline void print_usage(const char* program)
    {
        std::printf(
            "Usage: %s <options>\n"
            "\t-h, --help\t\tShow this help message\n"
            "\t-m, --m\t\t\tFeature extent (free0), default 18432\n"
            "\t-n, --n\t\t\tToken extent (free1), default 2048\n"
            "\t-k, --k\t\t\tBound extent, default 8192\n"
            "\t--a2a_extent\t\tFeatures taking the A2A path, default 10240\n"
            "\t--a2a_channels\t\tFlag regions to alternate over, default 2\n"
            "\t--a2a_world\t\tChecked against WORLD_SIZE; env wins\n"
            "\t--timing\t\t1 to measure, default 0\n"
            "\t--iters\t\t\tEnqueues per sample, default 10\n"
            "\t--verify\t\t1 to check every iteration against the closed form\n"
            "Rank identity comes from RANK / WORLD_SIZE / LOCAL_RANK /\n"
            "MASTER_ADDR / MASTER_PORT. With none set the run is single-rank.\n",
            program);
    }

    inline bool match(const char* arg, const char* shortName, const char* longName)
    {
        return (shortName && std::strcmp(arg, shortName) == 0)
               || std::strcmp(arg, longName) == 0;
    }

    inline bool parse_a2a_args(int argc, char** argv, Arguments& arg, std::string& error)
    {
        for(int i = 1; i < argc; ++i)
        {
            const char* opt = argv[i];
            if(match(opt, "-h", "--help"))
            {
                error = "help";
                return false;
            }
            if(i + 1 >= argc)
            {
                error = std::string("missing value for ") + opt;
                return false;
            }
            const char* value = argv[++i];

            if(match(opt, "-m", "--m"))
                arg.M[0] = std::strtoll(value, nullptr, 10);
            else if(match(opt, "-n", "--n"))
                arg.N[0] = std::strtoll(value, nullptr, 10);
            else if(match(opt, "-k", "--k"))
                arg.K[0] = std::strtoll(value, nullptr, 10);
            else if(match(opt, nullptr, "--a2a_extent"))
                arg.a2a_extent = std::strtoll(value, nullptr, 10);
            else if(match(opt, nullptr, "--a2a_channels"))
                arg.a2a_channels = uint8_t(std::strtoul(value, nullptr, 10));
            else if(match(opt, nullptr, "--a2a_world"))
                arg.a2a_world = uint8_t(std::strtoul(value, nullptr, 10));
            else if(match(opt, nullptr, "--timing"))
                arg.timing = int8_t(std::strtol(value, nullptr, 10));
            else if(match(opt, nullptr, "--iters"))
                arg.iters = int32_t(std::strtol(value, nullptr, 10));
            else if(match(opt, nullptr, "--verify"))
                arg.unit_check = int8_t(std::strtol(value, nullptr, 10));
            else
            {
                error = std::string("unknown option ") + opt;
                return false;
            }
        }
        return true;
    }
} // namespace hipblaslt_bench
