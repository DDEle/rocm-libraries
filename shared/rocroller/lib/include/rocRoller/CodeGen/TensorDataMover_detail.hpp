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

#pragma once

#include <rocRoller/Expression.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

namespace rocRoller
{
    namespace TensorDataMover
    {
        inline auto Literal(const auto& x)
        {
            return Register::Value::Literal(x);
        }

        inline auto L(const auto& x)
        {
            return Expression::literal(x);
        }

        struct BitfieldValue
        {
            static constexpr size_t NUM_DWORD_BITS = 32;

            BitfieldValue(size_t             bitOffset,
                          size_t             bitWidth,
                          Register::ValuePtr value,
                          std::string        name = "");

            void operator=(const BitfieldValue& v);
            void operator=(Register::ValuePtr v);

            size_t getBitOffset() const;

            size_t getBitWidth() const;

            void setValue(Register::ValuePtr value);

            Register::ValuePtr getValue() const;

            std::string getName() const;

            std::string toString() const;

            bool                operator==(const BitfieldValue& bf);
            bool                operator!=(const BitfieldValue& bf);
            std::vector<size_t> sgprIndexInGroup() const;

            static std::vector<BitfieldValue>
                combineLiteralBitFields(std::vector<BitfieldValue> bitfields);
            static std::vector<BitfieldValue> splitBitfield(BitfieldValue bitfield);
            static std::vector<BitfieldValue> splitBitFields(std::vector<BitfieldValue> bitfields);

        private:
            const size_t       m_bitOffset;
            const size_t       m_bitWidth;
            Register::ValuePtr m_value;
            std::string        m_name;
        };

        struct TDMGroup0Fields
        {
            std::vector<BitfieldValue> getBitfields()
            {
                return {reserved0,
                        gatherIndexSize,
                        gatherMode,
                        ldsAddress,
                        globalAddress,
                        reserved1,
                        type};
            }

            BitfieldValue reserved0{0, 30, Literal(1), "reserved"};
            BitfieldValue gatherIndexSize{30, 1, Literal(0), "gatherIndexSize"};
            BitfieldValue gatherMode{31, 1, Literal(0), "gatherMode"};
            BitfieldValue ldsAddress{32, 32, Literal(0), "ldsAddress"};
            BitfieldValue globalAddress{64, 57, Literal(0), "globalAddress"};
            BitfieldValue reserved1{121, 5, Literal(0), "reserved"};
            BitfieldValue type{126, 2, Literal(2), "type"};
        };

        struct TDMGroup1Fields
        {
            std::vector<BitfieldValue> getBitfields()
            {
                return {
                    workgroupMask,
                    dataSize,
                    sendAtomicBarrierOption,
                    tensorIterateMode,
                    paddingMode,
                    earlyTimeoutOption,
                    padInterval,
                    padAmount,
                    atomicBarrierAddress,
                    tensorDim0,
                    tensorDim1,
                    tileDim0,
                    tileDim1,
                    tileDim2,
                    tensorDim0Stride,
                    tensorDim1Stride,
                };
            }

            BitfieldValue workgroupMask{0, 16, Literal(0), "workgroupMask"};
            BitfieldValue dataSize{16, 2, Literal(0), "dataSize"};
            BitfieldValue sendAtomicBarrierOption{18, 1, Literal(0), "sendAtomicBarrierOption"};
            BitfieldValue tensorIterateMode{19, 1, Literal(0), "tensorIterateMode"};
            BitfieldValue paddingMode{20, 1, Literal(0), "paddingMode"};
            BitfieldValue earlyTimeoutOption{21, 1, Literal(0), "earlyTimeoutOption"};
            BitfieldValue padInterval{22, 3, Literal(0), "padInterval"};
            BitfieldValue padAmount{25, 7, Literal(0), "padAmount"};
            BitfieldValue atomicBarrierAddress{32, 16, Literal(0), "atomicBarrierAddress"};
            BitfieldValue tensorDim0{48, 32, Literal(0), "tensorDim0"};
            BitfieldValue tensorDim1{80, 32, Literal(0), "tensorDim1"};
            BitfieldValue tileDim0{112, 16, Literal(0), "tileDim0"};
            BitfieldValue tileDim1{128, 16, Literal(0), "tileDim1"};
            BitfieldValue tileDim2{144, 16, Literal(0), "tileDim2"};
            BitfieldValue tensorDim0Stride{160, 48, Literal(0), "tensorDim0Stride"};
            BitfieldValue tensorDim1Stride{208, 48, Literal(0), "tensorDim1Stride"};
        };
    }
}
