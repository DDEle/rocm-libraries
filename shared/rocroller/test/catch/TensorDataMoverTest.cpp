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

#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CodeGen/TensorDataMover.hpp>
#include <rocRoller/CodeGen/TensorDataMover_detail.hpp>
#include <rocRoller/Utilities/Error.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <common/SourceMatcher.hpp>

#include "CustomMatchers.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"

using namespace rocRoller;
using namespace TensorDataMover;

namespace TensorDataMoverTest
{
    TEST_CASE("BitfieldValue basic behavior", "[utility]")
    {
        auto context = TestContext::ForDefaultTarget();
        auto ctx     = context.get();

        SECTION("Stringification of named and unamed BitfieldValues")
        {
            BitfieldValue unamedBf{42, 6, Literal(33)};
            CHECK(unamedBf.toString() == "unamed_bitfield[47:42] = 33");

            BitfieldValue namedBf{24, 8, Literal(22), "testBitfield"};
            CHECK(namedBf.toString() == "testBitfield[31:24] = 22");
        }

        SECTION("BitFieldValues are *EQUAL* if their bit offset, bit width, and value are equal")
        {
            BitfieldValue bf0{42, 5, Literal(11)};
            BitfieldValue bf1{bf0.getBitOffset(), bf0.getBitWidth(), bf0.getValue()};
            CHECK(bf0 == bf1);

            // Equal even if ValuePtr is different as long as literal values are the same.
            BitfieldValue bf2{bf0.getBitOffset(), bf0.getBitWidth(), Literal(11)};
            CHECK(bf0 == bf2);

            auto sReg0
                = Register::Value::Placeholder(ctx, Register::Type::Scalar, DataType::UInt64, 1);
            BitfieldValue bf3{22, 5, sReg0};
            BitfieldValue bf4{bf3.getBitOffset(), bf3.getBitWidth(), sReg0};
            CHECK(bf3 == bf4);
        }

        SECTION("BitFieldValues are *DIFFERENT* if any of its bit offset, bit width, or value is "
                "different")
        {
            BitfieldValue bf0{42, 5, Literal(11)};
            BitfieldValue bf1{bf0.getBitOffset() + 1, bf0.getBitWidth(), bf0.getValue()};
            CHECK(bf0 != bf1);

            BitfieldValue bf2{bf0.getBitOffset(), bf0.getBitWidth() + 1, Literal(11)};
            CHECK(bf0 != bf2);

            BitfieldValue bf3{bf0.getBitOffset(), bf0.getBitWidth(), Literal(22)};
            CHECK(bf0 != bf3);

            auto scalarReg0
                = Register::Value::Placeholder(ctx, Register::Type::Scalar, DataType::UInt64, 1);
            auto scalarReg1
                = Register::Value::Placeholder(ctx, Register::Type::Scalar, DataType::UInt64, 1);
            BitfieldValue bf4{42, 5, scalarReg0};
            BitfieldValue bf5{bf4.getBitOffset(), bf4.getBitWidth(), scalarReg1};
            CHECK(bf4 != bf5);
        }

        SECTION(
            "Equivalent BitfiledValues can only be copy-assigned to change underlaying ValuePtr")
        {
            auto sReg0
                = Register::Value::Placeholder(ctx, Register::Type::Scalar, DataType::UInt64, 1);
            BitfieldValue bf0{42, 5, sReg0};
            BitfieldValue bf1{bf0.getBitOffset(), bf0.getBitWidth(), Literal(0)};
            // OK since bf0 & bf1 have the same bit offset and bit width
            CHECK_NOTHROW(bf1 = bf0);
            CHECK_NOTHROW(bf0 = bf1);

            BitfieldValue bf3{bf0.getBitOffset() + 1, bf0.getBitWidth(), Literal(0)};
            CHECK_THROWS_AS(bf3 = bf0, FatalError);
            CHECK_THROWS_AS(bf0 = bf3, FatalError);
        }
    }

    TEST_CASE("Packing and codegen of TensorDataMover descriptors", "[codegen][utility]")
    {

        SECTION("A descriptor with all bitfields as literals should be fully packed")
        {
            auto context = TestContext::ForDefaultTarget();
            auto ctx     = context.get();

            TensorDataMover::TDMDescriptor desc{ctx};
            desc.setLdsAddress(Literal(0x8BADF00D));
            desc.setGlobalAddress(Literal(0x01600DBEEFDEADBD));
            auto kb = [&]() -> Generator<Instruction> { co_yield desc.update(); };
            ctx->schedule(kb());

            auto expectedCode = R"(
                s_mov_b32 s0, 1
                s_mov_b32 s1, 2343432205
                s_mov_b32 s2, 4024348093
                s_mov_b32 s3, 2170555838
                s_mov_b32 s4, 0
                s_mov_b32 s5, 0
                s_mov_b32 s6, 0
                s_mov_b32 s7, 0
                s_mov_b32 s8, 0
                s_mov_b32 s9, 0
                s_mov_b32 s10, 0
                s_mov_b32 s11, 0
            )";
            CHECK(NormalizedSource(context.output()) == NormalizedSource(expectedCode));
        }

        SECTION("Update ldsAddress should emit only one additional instruction")
        {
            auto context = TestContext::ForDefaultTarget();
            auto ctx     = context.get();

            TensorDataMover::TDMDescriptor desc{ctx};
            auto                           kb = [&]() -> Generator<Instruction> {
                desc.setLdsAddress(Literal(0x8BADF00D));
                co_yield desc.update();

                desc.setLdsAddress(Literal(0xDECAFBAD));
                co_yield desc.updateLdsAddress();

                auto ldsAddress = Register::Value::Placeholder(
                    ctx, Register::Type::Scalar, DataType::UInt32, 1);
                ldsAddress->allocateNow();

                desc.setLdsAddress(ldsAddress);
                co_yield desc.updateLdsAddress();
            };
            ctx->schedule(kb());

            const auto expectedCode = R"(
                s_mov_b32 s0, 1
                s_mov_b32 s1, 2343432205
                s_mov_b32 s2, 0
                s_mov_b32 s3, 2147483648
                s_mov_b32 s4, 0
                s_mov_b32 s5, 0
                s_mov_b32 s6, 0
                s_mov_b32 s7, 0
                s_mov_b32 s8, 0
                s_mov_b32 s9, 0
                s_mov_b32 s10, 0
                s_mov_b32 s11, 0
                // update to ldsAddress field with literal
                s_mov_b32 s1, 3737844653
                // update to ldsAddress field with SGPR
                s_mov_b32 s1, s12
            )";
            CHECK(NormalizedSource(context.output()) == NormalizedSource(expectedCode));
        }

        SECTION("Update globalAddress (57 bits) without touching reserved bits")
        {
            auto context = TestContext::ForDefaultTarget();
            auto ctx     = context.get();

            TensorDataMover::TDMDescriptor desc{ctx};
            auto                           kb = [&]() -> Generator<Instruction> {
                desc.setGlobalAddress(Literal(0x01600DBEEFDEADBD));
                co_yield desc.update();

                auto globalAddress = Register::Value::Placeholder(
                    ctx, Register::Type::Scalar, DataType::UInt64, 1);
                globalAddress->allocateNow();

                desc.setGlobalAddress(globalAddress);
                co_yield desc.updateGlobalAddress();
            };
            ctx->schedule(kb());

            const auto expectedCode = R"(
                s_mov_b32 s0, 1
                s_mov_b32 s1, 0
                s_mov_b32 s2, 4024348093
                s_mov_b32 s3, 2170555838
                s_mov_b32 s4, 0
                s_mov_b32 s5, 0
                s_mov_b32 s6, 0
                s_mov_b32 s7, 0
                s_mov_b32 s8, 0
                s_mov_b32 s9, 0
                s_mov_b32 s10, 0
                s_mov_b32 s11, 0
                s_mov_b32 s2, s12
                // using bit masks to avoid touching reserved bits
                s_and_b32 s14, s3, 4261412864
                s_and_b32 s15, s13, 33554431
                s_or_b32 s3, s14, s15
            )";
            CHECK(NormalizedSource(context.output()) == NormalizedSource(expectedCode));
        }
    }

    class TensorDataMoverKernel : public AssemblyTestKernel
    {
    public:
        TensorDataMoverKernel(ContextPtr context,
                              DataType   datatype,
                              uint32_t   tensorX,
                              uint32_t   tensorY,
                              uint32_t   tensorDim0,
                              uint32_t   tensorDim1,
                              uint32_t   tileDim0,
                              uint32_t   tileDim1)
            : AssemblyTestKernel(context)
            , m_datatype(datatype)
            , m_tensorX(tensorX)
            , m_tensorY(tensorY)
            , m_tensorDim0(tensorDim0)
            , m_tensorDim1(tensorDim1)
            , m_tileDim0(tileDim0)
            , m_tileDim1(tileDim1)
        {
        }

        void generate() override
        {
            const uint workitemCountX = 32u;

            auto k = m_context->kernel();

            auto one                = Expression::literal(1);
            auto workitemCountXExpr = Expression::literal(32);

            k->setKernelDimensions(1);
            k->setWorkgroupSize({workitemCountX, 1, 1});
            k->setWorkitemCount({workitemCountXExpr, one, one});

            k->addArgument(
                {"outputTile", {m_datatype, PointerType::PointerGlobal}, DataDirection::WriteOnly});
            k->addArgument(
                {"inputTensor", {m_datatype, PointerType::PointerGlobal}, DataDirection::ReadOnly});

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            const auto dataSizeOption = [](auto datatype) -> DataSizeOption {
                const size_t size = DataTypeInfo::Get(datatype).elementBytes;
                switch(size)
                {
                case 1:
                    return DataSizeOption::OneByte;
                case 2:
                    return DataSizeOption::TwoBytes;
                case 4:
                    return DataSizeOption::FourBytes;
                case 8:
                    return DataSizeOption::EightBytes;
                default:
                    Throw<FatalError>(fmt::format(
                        "Invalid datatype {}. TDM does not support moving data with {} bytes.",
                        toString(datatype),
                        size));
                }
            }(m_datatype);

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr sOutputTile, sInputTensor;
                co_yield m_context->argLoader()->getValue("outputTile", sOutputTile);
                co_yield m_context->argLoader()->getValue("inputTensor", sInputTensor);

                auto sLDSAddress = Register::Value::Placeholder(
                    m_context, Register::Type::Scalar, DataType::UInt32, 1);

                auto lds
                    = Register::Value::AllocateLDS(m_context, m_datatype, m_tileDim0 * m_tileDim1);

                co_yield m_context->copier()->copy(
                    sLDSAddress, Register::Value::Literal(lds->getLDSAllocation()->offset()));

                auto desc = std::make_shared<TDMDescriptor>(m_context);
                desc->setLdsAddress(sLDSAddress);
                desc->setGlobalAddress(sInputTensor);
                desc->setDataSizeValue(dataSizeOption);
                desc->setTensorDim0(Literal(m_tensorDim0));
                desc->setTensorDim1(Literal(m_tensorDim1));
                desc->setTensorDim0Stride(Literal(m_tensorDim1));
                desc->setTensorDim1Stride(Literal(1));
                // tileDim0 is tile's fast-moving dimension
                desc->setTileDim0(Literal(m_tileDim1));
                desc->setTileDim1(Literal(m_tileDim0));

                co_yield desc->update();

                co_yield m_context->mem()->loadTensorToLDS(desc).map(
                    MemoryInstructions::addExtraDst(lds));

                desc->setGlobalAddress(sOutputTile);
                // tileDim0 is tile's fast-moving dimension
                desc->setTensorDim0(Literal(m_tileDim1));
                desc->setTensorDim1(Literal(m_tileDim0));
                desc->setTensorDim0Stride(Literal(m_tileDim1));
                co_yield desc->update();

                co_yield m_context->mem()->storeTensorFromLDS(desc).map(
                    MemoryInstructions::addExtraSrc(lds));
            };
            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }

        template <typename StorageType>
        void run()
        {
            std::vector<StorageType> inputTensor(m_tensorDim0 * m_tensorDim1);

            constexpr StorageType deadValue = []() -> StorageType {
                constexpr size_t numBytes = sizeof(StorageType);
                switch(numBytes)
                {
                case 1:
                    return static_cast<uint8_t>(0xFF);
                case 2:
                    return static_cast<uint16_t>(0xDEAD);
                case 4:
                    return static_cast<uint32_t>(0xDEADBEEF);
                case 8:
                    return static_cast<uint64_t>(0xDEADBEEF8BADF00DULL);
                default:
                    Throw<FatalError>(
                        fmt::format("TDM does not support moving data with {} bytes.", numBytes));
                }
            }();

            for(size_t i = 0; i < m_tensorDim0; ++i)
            {
                for(size_t j = 0; j < m_tensorDim1; ++j)
                {
                    if((m_tensorX <= i && i < (m_tensorX + m_tileDim0))
                       && (m_tensorY <= j && j < (m_tensorY + m_tileDim1)))
                    {
                        inputTensor[i * m_tensorDim1 + j]
                            = static_cast<StorageType>(i * m_tileDim1 + j);
                    }
                    else
                    {
                        inputTensor[i * m_tensorDim1 + j] = deadValue;
                    }
                }
            }

            const auto debugTensor = false;
            auto       printTensor = [deadValue](auto v, size_t M, size_t N) {
                if(!debugTensor)
                {
                    return;
                }
                for(size_t i = 0; i < M; ++i)
                {
                    for(size_t j = 0; j < N; ++j)
                    {
                        const auto x = static_cast<uint32_t>(v[i * N + j]);
                        if(x == deadValue)
                        {
                            std::cout << fmt::format("{:4} ", "xxxx");
                        }
                        else
                        {
                            std::cout << fmt::format("{:4} ", x);
                        }
                    }
                    std::cout << "\n";
                }
            };

            printTensor(inputTensor, m_tensorDim0, m_tensorDim1);

            std::vector<StorageType> outputTile(m_tileDim0 * m_tileDim1);

            auto d_outputTile  = make_shared_device<StorageType>(outputTile.size());
            auto d_inputTensor = make_shared_device(inputTensor);

            this->operator()(
                KernelInvocation{.workitemCount = {32, 1, 1}, .workgroupSize = {32, 1, 1}},
                d_outputTile.get(),
                d_inputTensor.get() + (m_tensorX * m_tensorDim1 + m_tensorY));

            CHECK_THAT(hipMemcpy(outputTile.data(),
                                 d_outputTile.get(),
                                 sizeof(StorageType) * outputTile.size(),
                                 hipMemcpyDefault),
                       HasHipSuccess(0));

            for(size_t i = 0; i < m_tileDim0; ++i)
            {
                for(size_t j = 0; j < m_tileDim1; ++j)
                {
                    CHECK(inputTensor[(i + m_tensorX) * m_tensorDim1 + (j + m_tensorY)]
                          == outputTile[i * m_tileDim1 + j]);
                }
            }

            printTensor(outputTile, m_tileDim0, m_tileDim1);
        }

        template <typename TestType>
        static void testBody(bool run = false);

    private:
        DataType m_datatype;
        uint32_t m_tensorX, m_tensorY;
        uint32_t m_tensorDim0, m_tensorDim1;
        uint32_t m_tileDim0, m_tileDim1;
    };

    template <typename TestType>
    void TensorDataMoverKernel::testBody(bool runOnGPU)
    {
        constexpr auto datatype = [numBytes = sizeof(TestType)]() {
            switch(numBytes)
            {
            case 1:
                return DataType::UInt8;
            case 2:
                return DataType::UInt16;
            case 4:
                return DataType::UInt32;
            case 8:
                return DataType::UInt64;
            default:
                Throw<FatalError>(
                    fmt::format("Test does not support datatypes with {} bytes.", numBytes));
            }
        }();
        // clang-format off
        auto tdmParams = GENERATE(
            std::tuple( 0,  0, 32, 32, 16, 16),
            std::tuple( 0, 16, 32, 32, 16, 16),
            std::tuple(16,  0, 32, 32, 16, 16),
            std::tuple(16, 16, 32, 32, 16, 16),

            std::tuple(  0,   0, 512, 512,  16, 128),
            std::tuple(128,   0, 512, 512, 128,  16),
            std::tuple(  0, 256, 512, 512,  32, 256),
            std::tuple(256,   0, 512, 512, 256,  32),

            std::tuple(  0,   0, 1024, 512,  32, 256),
            std::tuple(  0, 256, 1024, 512, 256,  32),
            std::tuple(512, 256, 1024, 512,  32, 256),
            std::tuple(512, 256, 1024, 512, 256,  32)
        );

        auto kernelTarget = GENERATE(GPUArchTargetGFX1250Rev0, GPUArchTargetGFX1250Rev1);

        // clang-format on
        DYNAMIC_SECTION(
            "Test the following params: " << fmt::format("{} {}", toString(datatype), tdmParams))
        {
            const auto [tensorX, tensorY, tensorDim0, tensorDim1, tileDim0, tileDim1] = tdmParams;

            AssertFatal(tensorX < tensorDim0 && tensorY < tensorDim1 && tileDim0 <= tensorDim0
                            && tileDim1 <= tensorDim1,
                        "Invalid params.",
                        ShowValue(datatype),
                        ShowValue(tensorX),
                        ShowValue(tensorY),
                        ShowValue(tensorDim0),
                        ShowValue(tensorDim1),
                        ShowValue(tileDim0),
                        ShowValue(tileDim1));

            auto context = TestContext::ForTarget(kernelTarget,
                                                  KernelOptions{},
                                                  datatype,
                                                  tensorX,
                                                  tensorY,
                                                  tensorDim0,
                                                  tensorDim1,
                                                  tileDim0,
                                                  tileDim1);
            auto ctx     = context.get();

            TensorDataMoverKernel tdmKernel(
                ctx, datatype, tensorX, tensorY, tensorDim0, tensorDim1, tileDim0, tileDim1);

            if(runOnGPU)
            {
                auto        currDeviceCtx  = TestContext::ForTestDevice();
                const auto& currDeviceArch = currDeviceCtx->targetArchitecture();
                if(!currDeviceArch.HasCapability(GPUCapability::HasTDM))
                {
                    SKIP(fmt::format("Target {} does not support TDM",
                                     toString(currDeviceArch.target())));
                }
                if(kernelTarget != currDeviceCtx->targetArchitecture().target())
                {
                    SKIP(fmt::format("Cannot run kernel for {} on {} device",
                                     toString(kernelTarget),
                                     toString(currDeviceArch.target())));
                }

                tdmKernel.run<TestType>();
            }
            else
            {
                tdmKernel.generate();
            }

            const auto code = context.output();
            CHECK(countSubstring(code, "tensor_load_to_lds ") == 1);
            CHECK(countSubstring(code, "tensor_store_from_lds ") == 1);
        }
    }

    TEMPLATE_TEST_CASE("Run TDM kernels that copy tiles of a tensor",
                       "[memory-instructions][gpu]",
                       uint64_t,
                       uint32_t,
                       uint16_t,
                       uint8_t)
    {
        TensorDataMoverKernel::testBody<TestType>(/*runOnGPU*/ true);
    }

    TEMPLATE_TEST_CASE("Assemble TDM kernels that copy tiles of a tensor",
                       "[memory-instructions][codegen]",
                       uint64_t,
                       uint32_t,
                       uint16_t,
                       uint8_t)
    {
        TensorDataMoverKernel::testBody<TestType>();
    }
}
