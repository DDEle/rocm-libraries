// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Cross-process peer resolution in hipblasLtSetDeviceComm, run as a mixed deployment:
// four ranks over two processes, two ranks each, so one communicator carries a self
// peer, a same-process peer and two cross-process peers at once.
//
// Run with four peer-capable devices:
//   HIP_VISIBLE_DEVICES=0,1,2,3 ./sample_a2a_multiprocess
// Exits 0 and prints a reason when the machine cannot host the topology.

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "../../../library/src/amd_detail/rocblaslt/src/include/handle.h"

namespace
{
    constexpr uint32_t kWorld            = 4;
    constexpr uint32_t kRanksPerProcess  = 2;
    constexpr uint32_t kChannels         = kWorld;
    constexpr uint32_t kSentinelBase     = 0xC0DE0000u;
    constexpr int      kSocketTimeoutSec = 30;

    constexpr int kStatusOk      = 0;
    constexpr int kStatusFailed  = 1;
    constexpr int kStatusSkipped = 2;

    size_t channelOffset(uint32_t channel)
    {
        return size_t(channel) * TensileLite::FUSED_A2A_FLAG_BLOCK_BYTES;
    }

    bool writeAll(int fd, const void* data, size_t bytes)
    {
        const char* p = static_cast<const char*>(data);
        while(bytes > 0)
        {
            const ssize_t n = ::write(fd, p, bytes);
            if(n <= 0)
                return false;
            p += n;
            bytes -= size_t(n);
        }
        return true;
    }

    bool readAll(int fd, void* data, size_t bytes)
    {
        char* p = static_cast<char*>(data);
        while(bytes > 0)
        {
            const ssize_t n = ::read(fd, p, bytes);
            if(n <= 0)
                return false;
            p += n;
            bytes -= size_t(n);
        }
        return true;
    }

    struct ProcessContext
    {
        int      sock     = -1;
        uint32_t rankBase = 0;

        std::mutex              mutex;
        std::condition_variable cv;
        uint32_t                arrived   = 0;
        bool                    exchanged = false;
        bool                    failed    = false;

        RocblasltFusedA2APeerRecord records[kWorld] = {};
    };

    uint32_t peerRankBase(uint32_t rankBase)
    {
        return rankBase == 0 ? kRanksPerProcess : 0;
    }

    bool swapRecordsWithPeerProcess(ProcessContext& ctx)
    {
        const size_t chunk = sizeof(RocblasltFusedA2APeerRecord) * kRanksPerProcess;
        return writeAll(ctx.sock, &ctx.records[ctx.rankBase], chunk)
               && readAll(ctx.sock, &ctx.records[peerRankBase(ctx.rankBase)], chunk);
    }

    bool crossProcessBarrier(int sock)
    {
        char token = 1;
        return writeAll(sock, &token, 1) && readAll(sock, &token, 1);
    }

    // The last rank of this process to arrive runs the cross-process swap; the others wait
    // for it and then read the completed set.
    hipblasStatus_t mixedAllgather(void*       userData,
                                   const void* sendbuf,
                                   void*       recvbuf,
                                   size_t      bytesPerRank)
    {
        auto* ctx = static_cast<ProcessContext*>(userData);
        if(bytesPerRank != sizeof(RocblasltFusedA2APeerRecord))
            return HIPBLAS_STATUS_INVALID_VALUE;

        RocblasltFusedA2APeerRecord record{};
        std::memcpy(&record, sendbuf, sizeof(record));
        if(record.rank >= kWorld)
            return HIPBLAS_STATUS_INVALID_VALUE;

        {
            std::unique_lock<std::mutex> lock(ctx->mutex);
            ctx->records[record.rank] = record;
            if(++ctx->arrived == kRanksPerProcess)
            {
                ctx->failed    = !swapRecordsWithPeerProcess(*ctx);
                ctx->exchanged = true;
                ctx->cv.notify_all();
            }
            else if(!ctx->cv.wait_for(lock,
                                      std::chrono::seconds(kSocketTimeoutSec),
                                      [ctx] { return ctx->exchanged; }))
            {
                return HIPBLAS_STATUS_INTERNAL_ERROR;
            }

            if(ctx->failed)
                return HIPBLAS_STATUS_INTERNAL_ERROR;
        }

        std::memcpy(recvbuf, ctx->records, bytesPerRank * kWorld);
        return HIPBLAS_STATUS_SUCCESS;
    }

    rocblaslt_comm_peer_kind expectedKind(uint32_t rank, uint32_t peer)
    {
        if(peer == rank)
            return rocblaslt_comm_peer_self;
        if(peer / kRanksPerProcess == rank / kRanksPerProcess)
            return rocblaslt_comm_peer_local;
        return rocblaslt_comm_peer_ipc;
    }

    const char* kindName(rocblaslt_comm_peer_kind kind)
    {
        switch(kind)
        {
        case rocblaslt_comm_peer_self:
            return "self";
        case rocblaslt_comm_peer_local:
            return "local";
        case rocblaslt_comm_peer_ipc:
            return "ipc";
        default:
            return "none";
        }
    }
}

__global__ void a2aStoreSentinel(uint32_t* dst, uint32_t value)
{
    *dst = value;
}

namespace
{
    // Four peer-capable devices. Runs after fork. Only the reporting process prints.
    bool topologyAvailable(bool report)
    {
        int deviceCount = 0;
        if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount < int(kWorld))
        {
            if(report)
                std::printf("skipped: needs %u devices, found %d\n", kWorld, deviceCount);
            return false;
        }
        for(uint32_t a = 0; a < kWorld; ++a)
        {
            for(uint32_t b = 0; b < kWorld; ++b)
            {
                if(a == b)
                    continue;
                int canAccess = 0;
                if(hipDeviceCanAccessPeer(&canAccess, int(a), int(b)) != hipSuccess
                   || canAccess == 0)
                {
                    if(report)
                        std::printf("skipped: device %u cannot reach device %u\n", a, b);
                    return false;
                }
            }
        }
        return true;
    }

    bool registerRanks(ProcessContext& ctx, std::vector<hipblasLtHandle_t>& handles)
    {
        std::vector<hipblasStatus_t> created(kRanksPerProcess, HIPBLAS_STATUS_NOT_INITIALIZED);
        std::vector<hipblasStatus_t> registered(kRanksPerProcess, HIPBLAS_STATUS_NOT_INITIALIZED);

        std::vector<std::thread> threads;
        for(uint32_t slot = 0; slot < kRanksPerProcess; ++slot)
        {
            threads.emplace_back([&, slot] {
                const uint32_t rank = ctx.rankBase + slot;
                if(hipSetDevice(int(rank)) != hipSuccess)
                    return;
                created[slot] = hipblasLtCreate(&handles[slot]);
                if(created[slot] != HIPBLAS_STATUS_SUCCESS)
                    return;
                registered[slot] = hipblasLtSetDeviceComm(
                    handles[slot], rank, kWorld, kChannels, mixedAllgather, &ctx);
            });
        }
        for(auto& thread : threads)
            thread.join();

        for(uint32_t slot = 0; slot < kRanksPerProcess; ++slot)
        {
            if(created[slot] != HIPBLAS_STATUS_SUCCESS)
            {
                std::printf("rank %u: hipblasLtCreate failed (%d)\n",
                            ctx.rankBase + slot,
                            int(created[slot]));
                return false;
            }
            if(registered[slot] != HIPBLAS_STATUS_SUCCESS)
            {
                std::printf("rank %u: hipblasLtSetDeviceComm failed (%d)\n",
                            ctx.rankBase + slot,
                            int(registered[slot]));
                return false;
            }
        }
        return true;
    }

    bool checkDispatch(const ProcessContext& ctx, const std::vector<hipblasLtHandle_t>& handles)
    {
        bool ok = true;
        for(uint32_t slot = 0; slot < kRanksPerProcess; ++slot)
        {
            const uint32_t rank   = ctx.rankBase + slot;
            const auto*    handle = reinterpret_cast<const _rocblaslt_handle*>(handles[slot]);
            for(uint32_t peer = 0; peer < kWorld; ++peer)
            {
                const auto want = expectedKind(rank, peer);
                const auto got  = handle->comm_peer_kind[peer];
                if(got != want)
                {
                    std::printf("rank %u peer %u: resolved as %s, expected %s\n",
                                rank,
                                peer,
                                kindName(got),
                                kindName(want));
                    ok = false;
                }
                if(handle->comm_peer_flag[peer] == nullptr)
                {
                    std::printf("rank %u peer %u: null flag pointer\n", rank, peer);
                    ok = false;
                }
            }
        }
        return ok;
    }

    bool writeSentinels(const ProcessContext& ctx, const std::vector<hipblasLtHandle_t>& handles)
    {
        for(uint32_t slot = 0; slot < kRanksPerProcess; ++slot)
        {
            const uint32_t rank   = ctx.rankBase + slot;
            const auto*    handle = reinterpret_cast<const _rocblaslt_handle*>(handles[slot]);
            if(hipSetDevice(int(rank)) != hipSuccess)
                return false;

            for(uint32_t peer = 0; peer < kWorld; ++peer)
            {
                auto* slotPtr
                    = static_cast<char*>(handle->comm_peer_flag[peer]) + channelOffset(rank);
                a2aStoreSentinel<<<1, 1>>>(reinterpret_cast<uint32_t*>(slotPtr),
                                           kSentinelBase | rank);
            }
            if(hipDeviceSynchronize() != hipSuccess)
            {
                std::printf("rank %u: sentinel store failed\n", rank);
                return false;
            }
        }
        return true;
    }

    bool checkSentinels(const ProcessContext& ctx, const std::vector<hipblasLtHandle_t>& handles)
    {
        bool ok = true;
        for(uint32_t slot = 0; slot < kRanksPerProcess; ++slot)
        {
            const uint32_t rank   = ctx.rankBase + slot;
            const auto*    handle = reinterpret_cast<const _rocblaslt_handle*>(handles[slot]);
            if(hipSetDevice(int(rank)) != hipSuccess)
                return false;

            for(uint32_t writer = 0; writer < kWorld; ++writer)
            {
                uint32_t    value   = 0;
                const auto* slotPtr
                    = static_cast<const char*>(handle->comm_flag_base) + channelOffset(writer);
                if(hipMemcpy(&value, slotPtr, sizeof(value), hipMemcpyDeviceToHost) != hipSuccess)
                {
                    std::printf("rank %u: readback of channel %u failed\n", rank, writer);
                    return false;
                }
                if(value != (kSentinelBase | writer))
                {
                    std::printf("rank %u channel %u: got 0x%08x, expected 0x%08x\n",
                                rank,
                                writer,
                                value,
                                kSentinelBase | writer);
                    ok = false;
                }
            }
        }
        return ok;
    }

    int runProcess(ProcessContext& ctx)
    {
        if(!topologyAvailable(ctx.rankBase == 0))
            return kStatusSkipped;

        std::vector<hipblasLtHandle_t> handles(kRanksPerProcess, nullptr);
        int                            status = kStatusOk;

        if(!registerRanks(ctx, handles))
            status = kStatusFailed;
        else if(!checkDispatch(ctx, handles))
            status = kStatusFailed;
        else if(!writeSentinels(ctx, handles))
            status = kStatusFailed;
        else if(!crossProcessBarrier(ctx.sock))
        {
            std::printf("cross-process barrier failed\n");
            status = kStatusFailed;
        }
        else if(!checkSentinels(ctx, handles))
            status = kStatusFailed;

        for(auto& handle : handles)
            if(handle != nullptr)
                hipblasLtDestroy(handle);
        return status;
    }
}

int main()
{
    int sockets[2] = {-1, -1};
    if(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0)
    {
        std::printf("socketpair failed\n");
        return 1;
    }

    timeval timeout{};
    timeout.tv_sec = kSocketTimeoutSec;
    for(int fd : sockets)
    {
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    }

    const pid_t child = ::fork();
    if(child < 0)
    {
        std::printf("fork failed\n");
        return 1;
    }

    ProcessContext ctx;
    if(child == 0)
    {
        ::close(sockets[0]);
        ctx.sock     = sockets[1];
        ctx.rankBase = kRanksPerProcess;
        const int status = runProcess(ctx);
        ::close(ctx.sock);
        std::fflush(stdout);
        ::_exit(status);
    }

    ::close(sockets[1]);
    ctx.sock     = sockets[0];
    ctx.rankBase = 0;
    const int parentStatus = runProcess(ctx);
    ::close(ctx.sock);

    int childRaw = 0;
    ::waitpid(child, &childRaw, 0);
    const int childStatus = WIFEXITED(childRaw) ? WEXITSTATUS(childRaw) : 1;

    if(parentStatus == kStatusFailed || childStatus == kStatusFailed)
        return 1;
    if(parentStatus == kStatusSkipped || childStatus == kStatusSkipped)
        return 0;

    std::printf("A2A mixed-deployment peer resolution verified over %u ranks\n", kWorld);
    return 0;
}
