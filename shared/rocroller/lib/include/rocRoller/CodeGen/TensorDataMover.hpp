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

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context_fwd.hpp>

namespace rocRoller
{
    namespace TensorDataMover
    {
        struct TDMGroup0Fields;
        struct TDMGroup1Fields;

        enum class GatherIndexSizeOption : int
        {
            RowAddress16Bits = 0,
            RowAddress32Bits,
            Count
        };

        enum class DataSizeOption : int
        {
            OneByte = 0,
            TwoBytes,
            FourBytes,
            EightBytes,
            Count
        };

        struct SwitchOption
        {
            SwitchOption(std::string name, bool value = false)
                : name(name)
                , enabled(value)
            {
            }

            std::string toString()
            {
                return fmt::format("{}::{}", name, (enabled ? "ENABLED" : "DISABLED"));
            }

            std::string getName()
            {
                return name;
            }
            bool isEnabled()
            {
                return enabled;
            }

        private:
            std::string name;
            bool        enabled;
        };

        std::string toString(SwitchOption opt);

        struct GatherMode : SwitchOption
        {
            GatherMode(std ::string name, bool value)
                : SwitchOption(name, value)
            {
            }
        };

        struct SendAtomicBarrierOption : SwitchOption
        {
            SendAtomicBarrierOption(std ::string name, bool value)
                : SwitchOption(name, value)
            {
            }
        };

        struct TensorIterateMode : SwitchOption
        {
            TensorIterateMode(std ::string name, bool value)
                : SwitchOption(name, value)
            {
            }
        };

        struct PaddingOption : SwitchOption
        {
            PaddingOption(std ::string name, bool value)
                : SwitchOption(name, value)
            {
            }
        };

        struct EarlyTimeOutOption : SwitchOption
        {
            EarlyTimeOutOption(std ::string name, bool value)
                : SwitchOption(name, value)
            {
            }
        };

        std::string toString(GatherIndexSizeOption opt);

        class TDMDescriptor
        {
        public:
            TDMDescriptor(ContextPtr ctx);

            Generator<Instruction> update();

            std::tuple<Register::ValuePtr,
                       Register::ValuePtr,
                       Register::ValuePtr,
                       Register::ValuePtr>
                getAllRegisters()
            {
                return {m_sgprsGroup0, m_sgprsGroup1, m_sgprsGroup2, m_sgprsGroup3};
            }

            // TDM Resource Descriptor Group 0
            void                   setGatherIndexSizeValue(GatherIndexSizeOption indexSizeOpt);
            Generator<Instruction> updateGatherIndexSize();

            void                   enableGatherMode();
            void                   disableGatherMode();
            Generator<Instruction> updateGatherMode();

            void                   setLdsAddress(Register::ValuePtr value);
            Generator<Instruction> updateLdsAddress();

            void                   setGlobalAddress(Register::ValuePtr value);
            Generator<Instruction> updateGlobalAddress();

            // TDM Resource Descriptor Group 1
            void                   setWorkgroupMask(Register::ValuePtr value);
            Generator<Instruction> updateWorkgroupMask();

            void                   setDataSizeValue(DataSizeOption dataSize);
            Generator<Instruction> updateDataSize();

            void                   enableSendAtomicBarrierOption();
            void                   disableSendAtomicBarrierOption();
            Generator<Instruction> updateSendAtomicBarrierOption();

            void                   enableTensorIterateMode();
            void                   disableTensorIterateMode();
            Generator<Instruction> updateTensorIterateMode();

            void                   enablePaddingOption();
            void                   disablePaddingOption();
            Generator<Instruction> updatePaddingOption();

            void                   enableEarlyTimeOutOption();
            void                   disableEarlyTimeOutOption();
            Generator<Instruction> updateEarlyTimeOutOption();

            void                   setPadIntervalValue(uint32_t padInterval);
            Generator<Instruction> updatePadInterval();

            void                   setPadAmountValue(uint32_t padAmount);
            Generator<Instruction> updatePadAmount();

            void                   setAtomicBarrierAddress(Register::ValuePtr value);
            Generator<Instruction> updateAtomicBarrierAddress();

            void                   setTensorDim0(Register::ValuePtr value);
            Generator<Instruction> updateTensorDim0();

            void                   setTensorDim1(Register::ValuePtr value);
            Generator<Instruction> updateTensorDim1();

            void                   setTileDim0(Register::ValuePtr value);
            Generator<Instruction> updateTileDim0();

            void                   setTileDim1(Register::ValuePtr value);
            Generator<Instruction> updateTileDim1();

            void                   setTileDim2(Register::ValuePtr value);
            Generator<Instruction> updateTileDim2();

            void                   setTensorDim0Stride(Register::ValuePtr value);
            Generator<Instruction> updateTensorDim0Stride();

            void                   setTensorDim1Stride(Register::ValuePtr value);
            Generator<Instruction> updateTensorDim1Stride();

        private:
            ContextPtr m_context;

            std::shared_ptr<TDMGroup0Fields> m_group0Bitfields;
            std::shared_ptr<TDMGroup1Fields> m_group1Bitfields;

            Register::ValuePtr m_sgprsGroup0;
            Register::ValuePtr m_sgprsGroup1;
            Register::ValuePtr m_sgprsGroup2;
            Register::ValuePtr m_sgprsGroup3;
        };
    }
}
