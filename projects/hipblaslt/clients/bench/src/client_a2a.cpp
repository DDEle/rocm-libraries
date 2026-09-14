// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "a2a_bench.hpp"

using namespace hipblaslt_bench;

int main(int argc, char* argv[])
{
    const hipblaslt_bench::LauncherEnv env = hipblaslt_bench::read_launcher_env();

    Arguments arg;
    arg.init();
    arg.M[0]       = 18432;
    arg.N[0]       = 2048;
    arg.K[0]       = 8192;
    arg.a2a_extent = 10240;

    // The launcher owns the group size; a command-line value only asserts it.
    // 0 marks "not given on the command line".
    arg.a2a_world = 0;

    std::string error;
    if(!parse_a2a_args(argc, argv, arg, error))
    {
        print_usage(argv[0]);
        return error == "help" ? 0 : 1;
    }
    if(arg.a2a_world != 0 && arg.a2a_world != uint8_t(env.world))
    {
        std::printf("error: --a2a_world %u disagrees with WORLD_SIZE %u\n",
                    unsigned(arg.a2a_world),
                    env.world);
        return 1;
    }
    arg.a2a_world = uint8_t(env.world);

    if(env.world > HIPBLASLT_DEVICE_COMM_MAX_WORLD)
    {
        std::printf("skipped: WORLD_SIZE %u exceeds %d\n",
                    env.world,
                    HIPBLASLT_DEVICE_COMM_MAX_WORLD);
        return 0;
    }

    hipblaslt_bench::TcpRendezvous rendezvous(env, kRendezvousTimeoutSec);
    if(!rendezvous.same_host_group())
    {
        std::printf("skipped: ranks span hosts\n");
        return 0;
    }

    if(env.rank == 0)
        std::printf("a2a_world,a2a_extent,a2a_channels,M,N,K\n%u,%lld,%u,%lld,%lld,%lld\n",
                    unsigned(arg.a2a_world),
                    static_cast<long long>(arg.a2a_extent),
                    unsigned(arg.a2a_channels),
                    static_cast<long long>(arg.M[0]),
                    static_cast<long long>(arg.N[0]),
                    static_cast<long long>(arg.K[0]));

    const uint8_t reachable = peers_reachable(env, arg) ? 1 : 0;

    // Every rank's peer pre-check is gathered before any rank decides to stop.
    std::vector<uint8_t> allReachable(env.world);
    if(rendezvous.allgather(&reachable, allReachable.data(), sizeof(reachable))
       != HIPBLAS_STATUS_SUCCESS)
    {
        std::printf("error: allgather for peer pre-check failed\n");
        return 1;
    }
    bool groupReachable = true;
    for(uint32_t j = 0; j < env.world; ++j)
        groupReachable = groupReachable && allReachable[j] != 0;
    if(!groupReachable)
    {
        std::printf("skipped: peer access unavailable on at least one rank\n");
        return 0;
    }

    RankResources res;
    res.rendezvous = &rendezvous;
    if(!setup_rank(env, arg, res))
        return 1;

    hipblasLtMatmulHeuristicResult_t heur{};
    const uint8_t                    found = select_algo(arg, res, heur) ? 1 : 0;

    // Every rank's result is gathered before any rank branches on it.
    std::vector<uint8_t> allFound(env.world);
    if(rendezvous.allgather(&found, allFound.data(), sizeof(found)) != HIPBLAS_STATUS_SUCCESS)
    {
        std::printf("error: allgather for algo selection failed\n");
        return 1;
    }
    bool groupFound = true;
    for(uint32_t j = 0; j < env.world; ++j)
        groupFound = groupFound && allFound[j] != 0;
    if(!groupFound)
    {
        std::printf("skipped: no fused GEMM+A2A solution in the loaded library\n");
        return 0;
    }

    uint32_t        launchCount = 0;
    hipblasStatus_t lastStatus  = HIPBLAS_STATUS_SUCCESS;
    auto            launch      = make_launch(arg, res, heur, launchCount, lastStatus);

    launch(0);
    if(hipStreamSynchronize(res.stream) != hipSuccess
       || lastStatus != HIPBLAS_STATUS_SUCCESS)
    {
        std::printf("error: matmul -> %d\n", int(lastStatus));
        return 1;
    }
    return 0;
}
