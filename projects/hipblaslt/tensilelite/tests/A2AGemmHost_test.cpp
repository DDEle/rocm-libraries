// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "A2AGemmHost.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

using namespace TensileLite::Client;

TEST(A2AGemmHost, CommResidue)
{
    using Words                = std::vector<uint32_t>;
    constexpr size_t seq       = FUSED_A2A_MODE1_SEQ_OFFSET / sizeof(uint32_t);
    constexpr size_t done      = FUSED_A2A_MODE1_DONE_OFFSET / sizeof(uint32_t);
    constexpr size_t maxRanks  = TensileLite::FUSED_A2A_MAX_RANKS;
    const size_t     wordCount = TensileLite::FUSED_A2A_FLAG_BLOCK_BYTES / sizeof(uint32_t);
    struct Case
    {
        std::string                 name;
        size_t                      rank;
        size_t                      world;
        size_t                      launches;
        std::function<void(Words&)> mutate;
        bool                        clean;
    };
    std::vector<Case> cases{
        {"clean", 1, 3, 5, [](Words&) {}, true},
        {"stale seq", 1, 3, 6, [](Words&) {}, false},
        {"missing done", 1, 3, 5, [](Words& w) { w[done + 2] = 4; }, false},
        {"own done", 1, 3, 5, [](Words& w) { w[done + 1] = 5; }, false},
        {"stray past done", 1, 3, 5, [](Words& w) { w[done + maxRanks] = 1; }, false},
        {"stray before seq", 1, 3, 5, [](Words& w) { w[seq - 1] = 1; }, false},
        {"stray after seq", 1, 3, 5, [](Words& w) { w[seq + 1] = 1; }, false},
        {"stray last word", 1, 3, 5, [](Words& w) { w.back() = 1; }, false},
        {"world 1 clean", 0, 1, 5, [](Words& w) { w.assign(w.size(), 0); }, true},
    };
    for(size_t s = 3; s < maxRanks; s++)
        cases.push_back({"done beyond world " + std::to_string(s),
                         1,
                         3,
                         5,
                         [s](Words& w) { w[done + s] = 5; },
                         false});
    for(size_t stray : {seq, done, done + 1, size_t(0), wordCount - 1})
        cases.push_back({"world 1 word " + std::to_string(stray),
                         0,
                         1,
                         5,
                         [stray](Words& w) {
                             w.assign(w.size(), 0);
                             w[stray] = 5;
                         },
                         false});

    for(auto const& c : cases)
    {
        Words words(wordCount, 0);
        words[seq]      = 5;
        words[done + 0] = 5;
        words[done + 2] = 5;
        c.mutate(words);
        EXPECT_EQ(a2aGemmCommResidue(words.data(), c.rank, c.world, c.launches).empty(), c.clean)
            << c.name;
    }
}
