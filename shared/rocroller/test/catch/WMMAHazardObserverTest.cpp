/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/KernelOptions_detail.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "common/Utilities.hpp"

#include "TestContext.hpp"

using namespace rocRoller;

namespace WMMAObserverTests
{
    class WMMAObserverTest : public TestContext
    {
    public:
        WMMAObserverTest(GPUArchitectureGFX gfx)
            : TestContext(TestContext::ForTarget({gfx})){};
        WMMAObserverTest(GPUArchitectureGFX gfx, KernelOptions kernelOptions)
            : TestContext(TestContext::ForTarget({gfx}, kernelOptions)){};
        void peekAndSchedule(Instruction& inst, uint expectedNops = 0)
        {
            auto peeked = m_context->observer()->peek(inst);
            CHECK(peeked.nops == expectedNops);
            m_context->schedule(inst);
        }
    };

    TEST_CASE("RAW hazards on GFX1200/GFX1201", "[codegen]")
    {
        auto target = GENERATE(GPUArchitectureGFX::GFX1200, GPUArchitectureGFX::GFX1201);
        WMMAObserverTest t{target};

        { // A/B is previous D
            const auto v0 = t.createRegisters(Register::Type::Vector, DataType::Half, 5);
            const auto v1 = t.createRegisters(Register::Type::Vector, DataType::Half, 1, 2);
            SECTION("Expect one V_NOP If the second WMMA's A is the first WMMA's D")
            {
                std::vector<Instruction> insts = {Instruction("v_wmma_f16_16x16x16_f16",
                                                              {v1[0]->subset({0, 1})},
                                                              {v0[0], v0[1], v0[2]},
                                                              {},
                                                              ""),
                                                  Instruction("v_wmma_f16_16x16x16_f16",
                                                              {v0[3]},
                                                              {v1[0]->subset({0, 1}), v0[4], v0[3]},
                                                              {},
                                                              ""),
                                                  Instruction("s_endpgm", {}, {}, {}, "")};

                t.peekAndSchedule(insts[0]);
                t.peekAndSchedule(insts[1], 1);

                CHECK(1 == countSubstring(t.output(), "v_nop"));
                t.output().clear();
            }

            SECTION("Expect one V_NOP If the second WMMA's B is the first WMMA's D")
            {
                std::vector<Instruction> insts = {Instruction("v_wmma_f16_16x16x16_f16",
                                                              {v1[0]->subset({0, 1})},
                                                              {v0[0], v0[1], v0[2]},
                                                              {},
                                                              ""),
                                                  Instruction("v_wmma_f16_16x16x16_f16",
                                                              {v0[3]},
                                                              {v1[0]->subset({0, 1}), v0[4], v0[3]},
                                                              {},
                                                              ""),
                                                  Instruction("s_endpgm", {}, {}, {}, "")};

                t.peekAndSchedule(insts[0]);
                t.peekAndSchedule(insts[1], 1);

                CHECK(1 == countSubstring(t.output(), "v_nop"));
                t.output().clear();
            }
        }

        { // A/B overlap with D
            const auto v0 = t.createRegisters(Register::Type::Vector, DataType::Half, 6);
            const auto v1 = t.createRegisters(Register::Type::Vector, DataType::Half, 1, 3);
            SECTION("Expect one V_NOP If the second WMMA's A overlaps with the first WMMA's D")
            {
                std::vector<Instruction> insts = {Instruction("v_wmma_f16_16x16x16_f16",
                                                              {v1[0]->subset({0, 1})},
                                                              {v0[0], v0[1], v0[2]},
                                                              {},
                                                              ""),
                                                  Instruction("v_wmma_f16_16x16x16_f16",
                                                              {v0[3]},
                                                              {v1[0]->subset({1, 2}), v0[4], v0[5]},
                                                              {},
                                                              ""),
                                                  Instruction("s_endpgm", {}, {}, {}, "")};

                t.peekAndSchedule(insts[0]);
                t.peekAndSchedule(insts[1], 1);

                CHECK(1 == countSubstring(t.output(), "v_nop"));
                t.output().clear();
            }

            SECTION("Expect one V_NOP If the second WMMA's B overlaps with the first WMMA's D")
            {
                // If the second WMMA's B overlaps the first WMMA's D then one V_NOP is expected
                std::vector<Instruction> insts = {Instruction("v_wmma_f16_16x16x16_f16",
                                                              {v1[0]->subset({0, 1})},
                                                              {v0[0], v0[1], v0[2]},
                                                              {},
                                                              ""),
                                                  Instruction("v_wmma_f16_16x16x16_f16",
                                                              {v0[3]},
                                                              {v0[4], v1[0]->subset({1, 2}), v0[5]},
                                                              {},
                                                              ""),
                                                  Instruction("s_endpgm", {}, {}, {}, "")};

                t.peekAndSchedule(insts[0]);
                t.peekAndSchedule(insts[1], 1);

                CHECK(1 == countSubstring(t.output(), "v_nop"));
                t.output().clear();
            }
        }

        SECTION("No register overlap should no be a hazard")
        {
            const auto v0 = t.createRegisters(Register::Type::Vector, DataType::Half, 4);
            const auto v1 = t.createRegisters(Register::Type::Vector, DataType::Half, 1, 4);
            std::vector<Instruction> insts = {Instruction("v_wmma_f16_16x16x16_f16",
                                                          {v1[0]->subset({0, 1})},
                                                          {v0[0], v0[1], v1[0]->subset({0, 1})},
                                                          {},
                                                          ""),
                                              Instruction("v_wmma_f16_16x16x16_f16",
                                                          {v1[0]->subset({2, 3})},
                                                          {v0[2], v0[3], v1[0]->subset({2, 3})},
                                                          {},
                                                          ""),
                                              Instruction("s_endpgm", {}, {}, {}, "")};

            t.peekAndSchedule(insts[0]);
            t.peekAndSchedule(insts[1]);

            CHECK(0 == countSubstring(t.output(), "v_nop"));
            t.output().clear();
        }
    }

    TEST_CASE("WAR/WAW hazards on GFX1200/GFX1201", "[codegen]")
    {
        auto target = GENERATE(GPUArchitectureGFX::GFX1200, GPUArchitectureGFX::GFX1201);
        WMMAObserverTest t{target};

        const auto v0 = t.createRegisters(Register::Type::Vector, DataType::Half, 4);
        const auto v1 = t.createRegisters(Register::Type::Vector, DataType::Float, 1, 4);
        const auto v2 = t.createRegisters(Register::Type::Vector, DataType::Float, 1);

        SECTION("WMMA writes D then VALU reads D")
        {
            std::vector<Instruction> insts = {
                Instruction("v_wmma_f32_16x16x16_f16",
                            {v1[0]->subset({0, 1})},
                            {v0[0], v0[1], v1[0]->subset({0, 1})},
                            {},
                            ""),
                Instruction("v_add_f32", {v2[0]}, {v2[0], v1[0]->subset({0})}, {}, ""),
                Instruction("s_endpgm", {}, {}, {}, ""),
            };

            t.peekAndSchedule(insts[0]);
            t.peekAndSchedule(insts[1], 8);

            CHECK(8 == countSubstring(t.output(), "v_nop"));
            t.output().clear();
        }

        SECTION("WMMA writes D then VALU writes D")
        {
            std::vector<Instruction> insts = {
                Instruction("v_wmma_f32_16x16x16_f16",
                            {v1[0]->subset({0, 1})},
                            {v0[0], v0[1], v1[0]->subset({0, 1})},
                            {},
                            ""),
                Instruction("v_add_f32", {v1[0]->subset({0})}, {v2[0], v2[0]}, {}, ""),
                Instruction("s_endpgm", {}, {}, {}, ""),
            };

            t.peekAndSchedule(insts[0]);
            t.peekAndSchedule(insts[1], 8);

            CHECK(8 == countSubstring(t.output(), "v_nop"));
            t.output().clear();
        }
    }

    TEST_CASE("RAW hazards on GFX1250", "[codegen]")
    {
        const auto target             = GPUArchitectureGFX::GFX1250;
        const auto coexecutionEnabled = GENERATE(true, false);

        KernelOptions kernelOptions{};
        kernelOptions->coexecutionEnabled = coexecutionEnabled;
        WMMAObserverTest t{target, kernelOptions};

        const auto FC = Register::AllocationOptions::FullyContiguous();

        SECTION("Expect 1 + coexecSlots V_NOP(s) if second WMMA's A is first WMMA's D and A && B "
                "are *NOT* F8")
        {

            const auto v = t.createRegisters(Register::Type::Vector, DataType::Half, 5, 8, FC);
            // coexecSlots = 4, if first WMMA is not:
            //  - v_wmma*_iu8; and
            //  - v_wmma*_iu4; and
            //  - v_wmma*f8f6f4 && A and B are not F8
            const auto               expectedVNops = 1 + (coexecutionEnabled ? 4 : 0);
            std::vector<Instruction> insts
                = {Instruction("v_wmma_f16_16x16x32_f16", {v[0]}, {v[1], v[2], v[3]}, {}, ""),
                   Instruction("v_wmma_f16_16x16x32_f16", {v[4]}, {v[0], v[2], v[3]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            t.peekAndSchedule(insts[0]);
            t.peekAndSchedule(insts[1], expectedVNops);

            CHECK(expectedVNops == countSubstring(t.output(), "v_nop"));
            t.output().clear();
        }

        SECTION("Expect 1 + coexecSlots V_NOP(s) if second WMMA's A is first WMMA's D and A"
                "*IS* F8")
        {
            const auto v0 = t.createRegisters(Register::Type::Vector, DataType::Float, 3, 8, FC);
            const auto vA = t.createRegisters(Register::Type::Vector, DataType::FP8x4, 2, 16, FC);
            const auto vB = t.createRegisters(Register::Type::Vector, DataType::FP4x8, 2, 16, FC);
            // coexecSlots = 8, if first WMMA is:
            //  - v_wmma*_iu8; or
            //  - v_wmma*_iu4; or
            //  - v_wmma*f8f6f4 && A or B is F8
            const auto               expectedVNops = 1 + (coexecutionEnabled ? 8 : 0);
            std::vector<Instruction> insts         = {
                Instruction("v_wmma_f32_16x16x128_f8f6f4", {v0[0]}, {vA[0], vB[0], v0[1]}, {}, ""),
                Instruction("v_wmma_f32_16x16x128_f8f6f4", {v0[2]}, {vA[1], vB[1], v0[0]}, {}, ""),
                Instruction("s_endpgm", {}, {}, {}, "")};

            t.peekAndSchedule(insts[0]);
            t.peekAndSchedule(insts[1], expectedVNops);

            CHECK(expectedVNops == countSubstring(t.output(), "v_nop"));
            t.output().clear();
        }

        SECTION(
            "Expect 0 + coexecSlots V_NOP(s) if VALU reads D after WMMA and A && B are *NOT* F8")
        {
            const auto v = t.createRegisters(Register::Type::Vector, DataType::Half, 5, 8, FC);
            // coexecSlots = 4, if first WMMA is not:
            //  - v_wmma*_iu8; and
            //  - v_wmma*_iu4; and
            //  - v_wmma*f8f6f4 && A and B are not F8
            const auto               expectedVNops = 0 + (coexecutionEnabled ? 4 : 0);
            std::vector<Instruction> insts
                = {Instruction("v_wmma_f16_16x16x32_f16", {v[0]}, {v[1], v[2], v[3]}, {}, ""),
                   Instruction("v_add_f16",
                               {v[4]->subset({0})},
                               {v[0]->subset({1}), v[4]->subset({1})},
                               {},
                               ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            t.peekAndSchedule(insts[0]);
            t.peekAndSchedule(insts[1], expectedVNops);

            CHECK(expectedVNops == countSubstring(t.output(), "v_nop"));
            t.output().clear();
        }

        SECTION("Expect 0 + coexecSlots V_NOP(s) if VALU reads D after WMMA and A *IS* F8")
        {
            const auto v0 = t.createRegisters(Register::Type::Vector, DataType::Float, 3, 8, FC);
            const auto vA = t.createRegisters(Register::Type::Vector, DataType::FP8x4, 2, 16, FC);
            const auto vB = t.createRegisters(Register::Type::Vector, DataType::FP4x8, 2, 16, FC);
            // coexecSlots = 8, if first WMMA is:
            //  - v_wmma*_iu8; or
            //  - v_wmma*_iu4; or
            //  - v_wmma*f8f6f4 && A or B is F8
            const auto               expectedVNops = 0 + (coexecutionEnabled ? 8 : 0);
            std::vector<Instruction> insts         = {
                Instruction("v_wmma_f32_16x16x128_f8f6f4", {v0[0]}, {vA[0], vB[0], v0[1]}, {}, ""),
                Instruction("v_add_f16",
                            {v0[2]->subset({0})},
                            {v0[0]->subset({1}), v0[2]->subset({1})},
                            {},
                            ""),
                Instruction("s_endpgm", {}, {}, {}, "")};

            t.peekAndSchedule(insts[0]);
            t.peekAndSchedule(insts[1], expectedVNops);

            CHECK(expectedVNops == countSubstring(t.output(), "v_nop"));
            t.output().clear();
        }
    }

    TEST_CASE("WAR/WAW hazards on GFX1250", "[codegen]")
    {
        const auto target             = GPUArchitectureGFX::GFX1250;
        const auto coexecutionEnabled = GENERATE(true, false);

        KernelOptions kernelOptions{};
        kernelOptions->coexecutionEnabled = coexecutionEnabled;
        WMMAObserverTest t{target, kernelOptions};

        const auto FC = Register::AllocationOptions::FullyContiguous();

        SECTION(
            "Expect 0 + coexecSlots V_NOP(s) if VALU writes A after WMMA and A && B are *NOT* F8")
        {
            const auto v = t.createRegisters(Register::Type::Vector, DataType::Half, 5, 8, FC);
            // coexecSlots = 4, if first WMMA is not:
            //  - v_wmma*_iu8; and
            //  - v_wmma*_iu4; and
            //  - v_wmma*f8f6f4 && A and B are not F8
            const auto               expectedVNops = 0 + (coexecutionEnabled ? 4 : 0);
            std::vector<Instruction> insts
                = {Instruction("v_wmma_f16_16x16x32_f16", {v[0]}, {v[1], v[2], v[3]}, {}, ""),
                   Instruction("v_add_f16",
                               {v[1]->subset({0})},
                               {v[4]->subset({1}), v[4]->subset({1})},
                               {},
                               ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            t.peekAndSchedule(insts[0]);
            t.peekAndSchedule(insts[1], expectedVNops);

            CHECK(expectedVNops == countSubstring(t.output(), "v_nop"));
            t.output().clear();
        }

        SECTION("Expect 0 + coexecSlots V_NOP(s) if VALU writes A after WMMA and A *IS* F8")
        {
            const auto v0 = t.createRegisters(Register::Type::Vector, DataType::Float, 3, 8, FC);
            const auto vA = t.createRegisters(Register::Type::Vector, DataType::FP8x4, 2, 16, FC);
            const auto vB = t.createRegisters(Register::Type::Vector, DataType::FP4x8, 2, 16, FC);
            // coexecSlots = 8, if first WMMA is:
            //  - v_wmma*_iu8; or
            //  - v_wmma*_iu4; or
            //  - v_wmma*f8f6f4 && A or B is F8
            const auto               expectedVNops = 0 + (coexecutionEnabled ? 8 : 0);
            std::vector<Instruction> insts         = {
                Instruction("v_wmma_f32_16x16x128_f8f6f4", {v0[0]}, {vA[0], vB[0], v0[1]}, {}, ""),
                Instruction("v_add_f16",
                            {vA[0]->subset({1})},
                            {v0[2]->subset({1}), v0[2]->subset({1})},
                            {},
                            ""),
                Instruction("s_endpgm", {}, {}, {}, "")};

            t.peekAndSchedule(insts[0]);
            t.peekAndSchedule(insts[1], expectedVNops);

            CHECK(expectedVNops == countSubstring(t.output(), "v_nop"));
            t.output().clear();
        }

        SECTION("Expect 0 + coexecSlots V_NOP(s) if VALU writes D after WMMA and A && B "
                "are *NOT* F8")
        {
            const auto v = t.createRegisters(Register::Type::Vector, DataType::Half, 5, 8, FC);
            // coexecSlots = 4, if first WMMA is not:
            //  - v_wmma*_iu8; and
            //  - v_wmma*_iu4; and
            //  - v_wmma*f8f6f4 && A and B are not F8
            const auto               expectedVNops = 0 + (coexecutionEnabled ? 4 : 0);
            std::vector<Instruction> insts
                = {Instruction("v_wmma_f16_16x16x32_f16", {v[0]}, {v[1], v[2], v[3]}, {}, ""),
                   Instruction("v_add_f16",
                               {v[0]->subset({0})},
                               {v[4]->subset({1}), v[4]->subset({1})},
                               {},
                               ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            t.peekAndSchedule(insts[0]);
            t.peekAndSchedule(insts[1], expectedVNops);

            CHECK(expectedVNops == countSubstring(t.output(), "v_nop"));
            t.output().clear();
        }

        SECTION("Expect 0 + coexecSlots V_NOP(s) if VALU writes D after WMMA and A *IS* F8")
        {
            const auto v0 = t.createRegisters(Register::Type::Vector, DataType::Float, 3, 8, FC);
            const auto vA = t.createRegisters(Register::Type::Vector, DataType::FP8x4, 2, 16, FC);
            const auto vB = t.createRegisters(Register::Type::Vector, DataType::FP4x8, 2, 16, FC);
            // coexecSlots = 8, if first WMMA is:
            //  - v_wmma*_iu8; or
            //  - v_wmma*_iu4; or
            //  - v_wmma*f8f6f4 && A or B is F8
            const auto               expectedVNops = 0 + (coexecutionEnabled ? 8 : 0);
            std::vector<Instruction> insts         = {
                Instruction("v_wmma_f32_16x16x128_f8f6f4", {v0[0]}, {vA[0], vB[0], v0[1]}, {}, ""),
                Instruction("v_add_f16",
                            {v0[0]->subset({0})},
                            {v0[2]->subset({1}), v0[2]->subset({1})},
                            {},
                            ""),
                Instruction("s_endpgm", {}, {}, {}, "")};

            t.peekAndSchedule(insts[0]);
            t.peekAndSchedule(insts[1], expectedVNops);

            CHECK(expectedVNops == countSubstring(t.output(), "v_nop"));
            t.output().clear();
        }
    }
}
