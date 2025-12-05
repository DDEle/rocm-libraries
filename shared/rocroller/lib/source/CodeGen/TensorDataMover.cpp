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

#include <ranges>

#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/TensorDataMover.hpp>
#include <rocRoller/CodeGen/TensorDataMover_detail.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{
    namespace TensorDataMover
    {
        constexpr size_t MAX_PAD_AMOUNT_DWORDS   = 128;
        constexpr size_t MAX_PAD_INTERVAL_DWORDS = 256;

        BitfieldValue::BitfieldValue(size_t             bitOffset,
                                     size_t             bitWidth,
                                     Register::ValuePtr value,
                                     std::string        name)
            : m_bitOffset(bitOffset)
            , m_bitWidth(bitWidth)
            , m_value(value)
            , m_name(name)
        {
            if(value->regType() == Register::Type::Literal)
            {
                const auto valueAsUInt64 = getUInt64(value->getLiteralValue());
                const auto valueBitwidth = std::bit_width(valueAsUInt64);
                AssertFatal(
                    valueBitwidth <= bitWidth,
                    fmt::format("Value {} needs {} bits but specified bit width of {} bits.",
                                valueAsUInt64,
                                valueBitwidth,
                                bitWidth));
            }
        }

        void BitfieldValue::operator=(const BitfieldValue& v)
        {
            AssertFatal(m_bitOffset == v.m_bitOffset,
                        "Cannot copy-assign bitfield with different bit offset.",
                        ShowValue(*this),
                        ShowValue(v));
            AssertFatal(m_bitWidth == v.m_bitWidth,
                        "Cannot copy-assign bitfield with different bitWidth",
                        ShowValue(*this),
                        ShowValue(v));
            m_value = v.m_value;
        }

        void BitfieldValue::operator=(Register::ValuePtr v)
        {
            setValue(v);
        }

        size_t BitfieldValue::getBitOffset() const
        {
            return m_bitOffset;
        }

        size_t BitfieldValue::getBitWidth() const
        {
            return m_bitWidth;
        }

        void BitfieldValue::setValue(Register::ValuePtr value)
        {
            if(value->regType() == Register::Type::Literal)
            {
                const auto valueAsUInt64 = getUInt64(value->getLiteralValue());
                const auto valueBitwidth = std::bit_width(valueAsUInt64);
                AssertFatal(value->regType() != Register::Type::Literal
                                || valueBitwidth <= m_bitWidth,
                            fmt::format("Bitfield has {} bits but tried to set value {} "
                                        "that needs {} bits.",
                                        m_bitWidth,
                                        valueAsUInt64,
                                        valueBitwidth));
            }
            m_value = value;
        }

        Register::ValuePtr BitfieldValue::getValue() const
        {
            return m_value;
        }

        std::string BitfieldValue::getName() const
        {
            return m_name;
        }

        std::string BitfieldValue::toString() const
        {
            return fmt::format("{}[{}:{}] = {}",
                               m_name.size() > 0 ? m_name : "unamed_bitfield",
                               m_bitOffset + m_bitWidth - 1,
                               m_bitOffset,
                               m_value ? m_value->toString() : "nullptr");
        }

        bool BitfieldValue::operator==(const BitfieldValue& bf)
        {
            return m_bitOffset == bf.m_bitOffset && m_bitWidth == bf.m_bitWidth
                   && ((m_value->regType() == Register::Type::Literal
                        && getUInt64(m_value->getLiteralValue())
                               == getUInt64(bf.m_value->getLiteralValue()))
                       || m_value == bf.m_value);
        }

        bool BitfieldValue::operator!=(const BitfieldValue& bf)
        {
            return not(*this == bf);
        }

        std::vector<size_t> BitfieldValue::sgprIndexInGroup() const
        {
            const size_t startIndex = m_bitOffset / 32;
            const size_t endIndex   = (m_bitOffset + m_bitWidth) / 32;
            const auto   r
                = std::views::iota(startIndex, endIndex + (startIndex == endIndex ? 1 : 0));
            return std::vector(r.begin(), r.end());
        }

        std::vector<BitfieldValue>
            BitfieldValue::combineLiteralBitFields(std::vector<BitfieldValue> bitfields)
        {
            std::vector<BitfieldValue> combinedBitFields{};

            const size_t COMBINED_BITFIELD_BIT_WIDTH = NUM_DWORD_BITS;

            uint64_t currentDwordValue          = 0;
            size_t   startBitOffset             = 0;
            size_t   remainingBitsInDword       = COMBINED_BITFIELD_BIT_WIDTH;
            size_t   numBitsInCombinedBitFields = 0;

            auto appendCombinedBitfield = [&]() {
                const size_t numBits          = COMBINED_BITFIELD_BIT_WIDTH - remainingBitsInDword;
                auto         combinedBitfield = BitfieldValue(
                    startBitOffset,
                    numBits,
                    Register::Value::Literal(static_cast<uint32_t>(currentDwordValue)));
                combinedBitFields.push_back(combinedBitfield);

                currentDwordValue    = 0;
                remainingBitsInDword = COMBINED_BITFIELD_BIT_WIDTH;
                startBitOffset += numBits;
            };

            for(const auto& bitfield : bitfields)
            {
                const auto bitfieldRegType = bitfield.getValue()->regType();
                const auto isLiteral       = bitfieldRegType == Register::Type::Literal;
                const auto isScalar        = bitfieldRegType == Register::Type::Scalar;
                AssertFatal(
                    isLiteral || isScalar,
                    "Only BitFieldValue of Literal or Scalar Register::Type can be combined",
                    ShowValue(bitfieldRegType));

                size_t bfBitWidth = bitfield.getBitWidth();
                numBitsInCombinedBitFields += bfBitWidth;

                if(isLiteral)
                {
                    size_t   remainingBitsInValue = bfBitWidth;
                    uint64_t value = getUInt64(bitfield.getValue()->getLiteralValue());

                    while(remainingBitsInValue > 0)
                    {
                        const size_t bitOffsetInDword
                            = COMBINED_BITFIELD_BIT_WIDTH - remainingBitsInDword;
                        const size_t numInsertedBits
                            = (remainingBitsInValue <= remainingBitsInDword) ? remainingBitsInValue
                                                                             : remainingBitsInDword;

                        const uint64_t selectBitsMask = (1UL << numInsertedBits) - 1UL;
                        const uint64_t bitsToInsert   = value & selectBitsMask;

                        currentDwordValue &= ~(selectBitsMask << bitOffsetInDword);
                        currentDwordValue |= (bitsToInsert << bitOffsetInDword);

                        remainingBitsInDword -= numInsertedBits;
                        remainingBitsInValue -= numInsertedBits;

                        value >>= numInsertedBits;

                        if(remainingBitsInDword == 0)
                        {
                            appendCombinedBitfield();
                        }
                    }
                }
                else
                {
                    // Append current combined bitfield of partial dword if any
                    if(remainingBitsInDword < COMBINED_BITFIELD_BIT_WIDTH)
                    {
                        appendCombinedBitfield();
                    }

                    // Copy bitfield in Scalar registers
                    combinedBitFields.push_back(bitfield);
                    startBitOffset += bfBitWidth;
                }
            }

            // Append last partial dword if any
            if(remainingBitsInDword < COMBINED_BITFIELD_BIT_WIDTH)
            {
                appendCombinedBitfield();
            }

            AssertFatal(startBitOffset == numBitsInCombinedBitFields,
                        "Failed to split all bits in bitfield");

            return combinedBitFields;
        }

        std::vector<BitfieldValue> BitfieldValue::splitBitfield(BitfieldValue bitfield)
        {
            if(bitfield.getBitWidth() <= NUM_DWORD_BITS)
            {
                return {bitfield};
            }
            const size_t numSGPRs = (bitfield.getBitWidth() + NUM_DWORD_BITS - 1) / NUM_DWORD_BITS;

            const auto& value        = bitfield.getValue();
            const auto  isLiteral    = value->regType() == Register::Type::Literal;
            uint64_t    literalValue = isLiteral ? getUInt64(value->getLiteralValue()) : 0;

            std::vector<BitfieldValue> subBitfields;

            size_t remainingBitsInValue = bitfield.getBitWidth();
            size_t currentBitOffset     = bitfield.getBitOffset();
            for(size_t sourceSGPRIndex = 0; sourceSGPRIndex < numSGPRs; ++sourceSGPRIndex)
            {
                const size_t sgprIndex            = currentBitOffset / NUM_DWORD_BITS;
                const size_t bitOffsetInSGPR      = currentBitOffset - sgprIndex * NUM_DWORD_BITS;
                const size_t remainingBitsInDword = NUM_DWORD_BITS - bitOffsetInSGPR;
                const size_t numBits              = remainingBitsInValue <= remainingBitsInDword
                                                        ? remainingBitsInValue
                                                        : remainingBitsInDword;

                if(isLiteral)
                {
                    const uint64_t selectBitsMask = (1UL << numBits) - 1UL;
                    const uint32_t valueBits = static_cast<uint32_t>(literalValue & selectBitsMask);
                    subBitfields.push_back(
                        BitfieldValue{currentBitOffset, numBits, Literal(valueBits)});

                    literalValue >>= numBits;
                }
                else
                {
                    subBitfields.push_back(
                        BitfieldValue{currentBitOffset, numBits, value->subset({sourceSGPRIndex})});
                }

                currentBitOffset += numBits;
                remainingBitsInValue -= numBits;
            }
            AssertFatal(remainingBitsInValue == 0, "Failed to split all bits in bitfield");

            return subBitfields;
        }

        std::vector<BitfieldValue>
            BitfieldValue::splitBitFields(std::vector<BitfieldValue> bitfields)
        {
            std::vector<BitfieldValue> splitFields;
            for(const auto& bf : bitfields)
            {
                auto sbfs = splitBitfield(bf);
                splitFields.insert(splitFields.end(), sbfs.begin(), sbfs.end());
            }
            return splitFields;
        }

        struct BitfieldSplitter
        {
            BitfieldSplitter() = default;

            template <std::ranges::range Range>
            Range operator()(Range& x) const
            {
                return BitfieldValue::splitBitFields(x);
            }
        };
        template <std::ranges::range Range>
        auto operator|(Range&& x, const BitfieldSplitter& splitter)
        {
            return splitter(x);
        }

        struct BitfieldCombiner
        {
            BitfieldCombiner() = default;
            template <std::ranges::range Range>
            Range operator()(Range& x) const
            {
                return BitfieldValue::combineLiteralBitFields(x);
            }
        };
        template <std::ranges::range Range>
        Range operator|(Range&& x, const BitfieldCombiner& combiner)
        {
            return combiner(x);
        }

        std::ostream& operator<<(std::ostream& os, BitfieldValue const& bf)
        {
            os << bf.toString();
            return os;
        }

        std::string toString(GatherIndexSizeOption option)
        {
            switch(option)
            {
            case GatherIndexSizeOption::RowAddress16Bits:
                return "RowAddress16Bits";
            case GatherIndexSizeOption::RowAddress32Bits:
                return "RowAddress32Bits";
            default:
                Throw<FatalError>(
                    fmt::format("Invalid GatherIndexSizeOption {}", static_cast<int>(option)));
            }
        }

        std::string toString(DataSizeOption option)
        {
            switch(option)
            {
            case DataSizeOption::OneByte:
                return "OneByte";
            case DataSizeOption::TwoBytes:
                return "TwoBytes";
            case DataSizeOption::FourBytes:
                return "FourBytes";
            case DataSizeOption::EightBytes:
                return "EightBytes";
            default:
                Throw<FatalError>(
                    fmt::format("Invalid DataSizeOption {}", static_cast<int>(option)));
            }
        }

        std::string toString(SwitchOption opt)
        {
            return fmt::format("{}::{}", opt.getName(), (opt.isEnabled() ? "ENABLED" : "DISABLED"));
        }

        TDMDescriptor::TDMDescriptor(ContextPtr ctx)
        {
            m_context = ctx;

            m_sgprsGroup0 = std::make_shared<Register::Value>(
                ctx,
                Register::Type::Scalar,
                VariableType{DataType::None, PointerType::TDMDescGroup0},
                1);
            m_sgprsGroup1 = std::make_shared<Register::Value>(
                ctx,
                Register::Type::Scalar,
                VariableType{DataType::None, PointerType::TDMDescGroup1},
                1);

            // TODO: support 3D, 4D, and 5D tensors and gather mode
            m_sgprsGroup2 = nullptr;
            m_sgprsGroup3 = nullptr;

            m_group0Bitfields = std::make_shared<TDMGroup0Fields>();
            m_group1Bitfields = std::make_shared<TDMGroup1Fields>();
        }

        Generator<Instruction>
            updateBitfield(ContextPtr ctx, BitfieldValue bitfield, Register::ValuePtr sgprGroup)
        {
            // TODO: explicitly indicate which TDM bitfields are affected.
            for(const auto bf : BitfieldValue::splitBitfield(bitfield))
            {
                const auto sgprIndices = bf.sgprIndexInGroup();
                if(bf.getBitWidth() == BitfieldValue::NUM_DWORD_BITS)
                {
                    co_yield ctx->copier()->copy(sgprGroup->subset(sgprIndices),
                                                 bf.getValue(),
                                                 fmt::format("Updating TDM descriptor"));
                }
                else
                {
                    const uint32_t numBits = bf.getBitWidth();
                    const uint32_t bitOffsetInSgpr
                        = bf.getBitOffset() - sgprIndices[0] * BitfieldValue::NUM_DWORD_BITS;
                    const uint32_t selectBitsMask = (1 << numBits) - 1;
                    const uint32_t clearBitsMask  = ~(selectBitsMask << bitOffsetInSgpr);

                    auto targetSGPR     = sgprGroup->subset(sgprIndices);
                    auto targetSGPRExpr = targetSGPR->expression();
                    auto sourceSGPRExpr = bf.getValue()->expression();

                    auto expr = (targetSGPRExpr & L(clearBitsMask))
                                | ((sourceSGPRExpr & L(selectBitsMask)) << L(bitOffsetInSgpr));
                    co_yield Expression::generate(targetSGPR, expr, ctx);
                }
            }
        }

        Generator<Instruction> TDMDescriptor::update()
        {
            for(const auto& bf :
                m_group0Bitfields->getBitfields() | BitfieldSplitter() | BitfieldCombiner())
            {
                co_yield updateBitfield(m_context, bf, m_sgprsGroup0);
            }

            for(const auto& bf :
                m_group1Bitfields->getBitfields() | BitfieldSplitter() | BitfieldCombiner())
            {
                co_yield updateBitfield(m_context, bf, m_sgprsGroup1);
            }
        }

        void TDMDescriptor::setGatherIndexSizeValue(GatherIndexSizeOption option)
        {
            switch(option)
            {
            case GatherIndexSizeOption::RowAddress16Bits:
                m_group0Bitfields->gatherIndexSize = Register::Value::Literal(0);
                break;
            case GatherIndexSizeOption::RowAddress32Bits:
                m_group0Bitfields->gatherIndexSize = Register::Value::Literal(1);
                break;
            default:
                Throw<FatalError>(
                    fmt::format("Invalid GatherIndexSizeOption {}", static_cast<int>(option)));
            }
        }

        void TDMDescriptor::setDataSizeValue(DataSizeOption option)
        {
            switch(option)
            {
            case DataSizeOption::OneByte:
                m_group1Bitfields->dataSize = Literal(0);
                break;
            case DataSizeOption::TwoBytes:
                m_group1Bitfields->dataSize = Literal(1);
                break;
            case DataSizeOption::FourBytes:
                m_group1Bitfields->dataSize = Literal(2);
                break;
            case DataSizeOption::EightBytes:
                m_group1Bitfields->dataSize = Literal(3);
                break;
            default:
                Throw<FatalError>(
                    fmt::format("Invalid DataSizeOption {}", static_cast<int>(option)));
            }
        }

        Generator<Instruction> TDMDescriptor::updateDataSize()
        {
            co_yield updateBitfield(m_context, m_group1Bitfields->dataSize, m_sgprsGroup1);
        }

        void TDMDescriptor::setPadIntervalValue(uint32_t numDwords)
        {
            AssertFatal(
                numDwords > 0,
                "Padding interval needs to be greated than zero. Otherwise, padding is not needed",
                ShowValue(numDwords));
            AssertFatal((numDwords & (numDwords - 1)) == 0,
                        "Padding interval must a power of two.",
                        ShowValue(numDwords));
            AssertFatal(numDwords <= MAX_PAD_INTERVAL_DWORDS,
                        "Padding interval must be at most 256 dwords",
                        ShowValue(numDwords),
                        ShowValue(MAX_PAD_INTERVAL_DWORDS));
            m_group1Bitfields->padInterval = Literal(numDwords);
        }

        Generator<Instruction> TDMDescriptor::updatePadInterval()
        {
            co_yield updateBitfield(m_context, m_group1Bitfields->padInterval, m_sgprsGroup1);
        }

        void TDMDescriptor::setPadAmountValue(uint32_t numDwords)
        {
            AssertFatal(
                numDwords > 0,
                "Padding amount needs to be greated than zero. Otherwise, padding is not needed.",
                ShowValue(numDwords));
            AssertFatal(numDwords <= MAX_PAD_AMOUNT_DWORDS,
                        "Padding amount must be at most 128 dwords",
                        ShowValue(numDwords),
                        ShowValue(MAX_PAD_AMOUNT_DWORDS));
            m_group1Bitfields->padAmount = Literal(numDwords - 1);
        }

        Generator<Instruction> TDMDescriptor::updatePadAmount()
        {
            co_yield updateBitfield(m_context, m_group1Bitfields->padAmount, m_sgprsGroup1);
        }

#define DEFINE_SWITCH_BITFIELD_METHODS(groupBitfields, groupSgprs, OptionType, bitfield) \
    void TDMDescriptor::enable##OptionType()                                             \
    {                                                                                    \
        groupBitfields->bitfield = Literal(1);                                           \
    }                                                                                    \
    void TDMDescriptor::disable##OptionType()                                            \
    {                                                                                    \
        groupBitfields->bitfield = Literal(0);                                           \
    }                                                                                    \
    Generator<Instruction> TDMDescriptor::update##OptionType()                           \
    {                                                                                    \
        co_yield updateBitfield(m_context, groupBitfields->bitfield, groupSgprs);        \
    }

#define DEFINE_BITFIELD_METHODS(groupBitfields, groupSgpr, method, bitfield)     \
    void TDMDescriptor::set##method(Register::ValuePtr value)                    \
    {                                                                            \
        groupBitfields->bitfield = value;                                        \
    }                                                                            \
    Generator<Instruction> TDMDescriptor::update##method()                       \
    {                                                                            \
        co_yield updateBitfield(m_context, groupBitfields->bitfield, groupSgpr); \
    }

        // TDM Resource Descriptor Group 0
        DEFINE_SWITCH_BITFIELD_METHODS(m_group0Bitfields, m_sgprsGroup0, GatherMode, gatherMode)
        DEFINE_BITFIELD_METHODS(m_group0Bitfields, m_sgprsGroup0, LdsAddress, ldsAddress)
        DEFINE_BITFIELD_METHODS(m_group0Bitfields, m_sgprsGroup0, GlobalAddress, globalAddress)

        // TDM Resource Descriptor Group 1
        DEFINE_BITFIELD_METHODS(m_group1Bitfields, m_sgprsGroup1, WorkgroupMask, workgroupMask)
        DEFINE_SWITCH_BITFIELD_METHODS(m_group1Bitfields,
                                       m_sgprsGroup1,
                                       SendAtomicBarrierOption,
                                       sendAtomicBarrierOption)
        DEFINE_SWITCH_BITFIELD_METHODS(m_group1Bitfields,
                                       m_sgprsGroup1,
                                       TensorIterateMode,
                                       tensorIterateMode)
        DEFINE_SWITCH_BITFIELD_METHODS(m_group1Bitfields, m_sgprsGroup1, PaddingOption, paddingMode)
        DEFINE_SWITCH_BITFIELD_METHODS(m_group1Bitfields,
                                       m_sgprsGroup1,
                                       EarlyTimeOutOption,
                                       earlyTimeoutOption)
        DEFINE_BITFIELD_METHODS(m_group1Bitfields,
                                m_sgprsGroup1,
                                AtomicBarrierAddress,
                                atomicBarrierAddress)
        DEFINE_BITFIELD_METHODS(m_group1Bitfields, m_sgprsGroup1, TensorDim0, tensorDim0)
        DEFINE_BITFIELD_METHODS(m_group1Bitfields, m_sgprsGroup1, TensorDim1, tensorDim1)
        DEFINE_BITFIELD_METHODS(m_group1Bitfields, m_sgprsGroup1, TileDim0, tileDim0)
        DEFINE_BITFIELD_METHODS(m_group1Bitfields, m_sgprsGroup1, TileDim1, tileDim1)
        DEFINE_BITFIELD_METHODS(m_group1Bitfields, m_sgprsGroup1, TileDim2, tileDim2)
        DEFINE_BITFIELD_METHODS(m_group1Bitfields,
                                m_sgprsGroup1,
                                TensorDim0Stride,
                                tensorDim0Stride)
        DEFINE_BITFIELD_METHODS(m_group1Bitfields,
                                m_sgprsGroup1,
                                TensorDim1Stride,
                                tensorDim1Stride)
#undef DEFINE_BITFIELD_METHODS
#undef DEFINE_SWITCH_BITFIELD_METHODS
    }
}
