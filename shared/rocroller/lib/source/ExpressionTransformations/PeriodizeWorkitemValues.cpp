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

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph_fwd.hpp>
#include <variant>

namespace rocRoller
{
    namespace Expression
    {
        struct PeriodizeWorkitemValuesVisitor
        {
            PeriodizeWorkitemValuesVisitor(ContextPtr                  ctx,
                                           KernelGraph::KernelGraphPtr graph,
                                           const uint                  period)
                : m_context(ctx)
                , m_graph(graph)
                , m_period{period} {};

            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.arg)
                {
                    cpy.arg = call(expr.arg);
                }
                return std::make_shared<Expression>(cpy);
            }

            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.lhs)
                {
                    cpy.lhs = call(expr.lhs);
                }
                if(expr.rhs)
                {
                    cpy.rhs = call(expr.rhs);
                }
                return std::make_shared<Expression>(cpy);
            }

            template <CTernary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.lhs)
                {
                    cpy.lhs = call(expr.lhs);
                }
                if(expr.r1hs)
                {
                    cpy.r1hs = call(expr.r1hs);
                }
                if(expr.r2hs)
                {
                    cpy.r2hs = call(expr.r2hs);
                }
                return std::make_shared<Expression>(cpy);
            }

            template <CNary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                auto cpy = expr;
                std::ranges::for_each(cpy.operands, [this](auto& op) { op = call(op); });
                return std::make_shared<Expression>(std::move(cpy));
            }

            ExpressionPtr operator()(ScaledMatrixMultiply const& expr) const
            {
                ScaledMatrixMultiply cpy = expr;
                if(expr.matA)
                {
                    cpy.matA = call(expr.matA);
                }
                if(expr.matB)
                {
                    cpy.matB = call(expr.matB);
                }
                if(expr.matC)
                {
                    cpy.matC = call(expr.matC);
                }
                if(expr.scaleA)
                {
                    cpy.scaleA = call(expr.scaleA);
                }
                if(expr.scaleB)
                {
                    cpy.scaleB = call(expr.scaleB);
                }
                return std::make_shared<Expression>(cpy);
            }

            template <CValue Value>
            ExpressionPtr operator()(Value const& expr) const
            {
                // TODO represent this expression directly in the CoordinateGraph
                if constexpr(std::same_as<Value, DataFlowTag>)
                {
                    auto node = m_graph->coordinates.getNode(expr.tag);
                    if(std::holds_alternative<KernelGraph::CoordinateGraph::Workitem>(node))
                    {
                        auto workitem = std::get<KernelGraph::CoordinateGraph::Workitem>(node);
                        if(workitem.dim == 0)
                        {
                            DataFlowTag df{expr};
                            auto const  wavefrontSize
                                = m_context->targetArchitecture().GetCapability(
                                    GPUCapability::DefaultWavefrontSize);
                            AssertFatal(m_period < wavefrontSize && m_period != 0
                                            && (m_period & (m_period - 1)) == 0,
                                        "period must be a non-zero power of 2 less than the "
                                        "wavefront size: ",
                                        ShowValue(m_period),
                                        ShowValue(wavefrontSize));

                            auto const logPeriod        = static_cast<uint>(log2(m_period));
                            auto const logWavefrontSize = static_cast<uint>(log2(wavefrontSize));

                            auto workitemXExpr        = std::make_shared<Expression>(df);
                            auto logPeriodExpr        = literal(logPeriod);
                            auto periodMask           = literal(m_period - 1);
                            auto logWavefrontSizeExpr = literal(logWavefrontSize);

                            auto nWave = std::make_shared<Expression>(
                                ArithmeticShiftR({workitemXExpr, logWavefrontSizeExpr}));

                            auto nWaveOffset
                                = std::make_shared<Expression>(ShiftL({nWave, logPeriodExpr}));

                            auto maskedPeriodInWave = std::make_shared<Expression>(
                                BitwiseAnd({workitemXExpr, periodMask}));

                            auto newExpr = std::make_shared<Expression>(
                                Add({maskedPeriodInWave, nWaveOffset}));

                            return newExpr;
                        }
                    }
                }
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr call(ExpressionPtr expr) const
            {
                if(!expr)
                    return expr;

                return std::visit(*this, *expr);
            }

        private:
            ContextPtr                  m_context;
            KernelGraph::KernelGraphPtr m_graph;
            const uint                  m_period;
        };

        ExpressionPtr periodizeWorkitemValues(ExpressionPtr               expr,
                                              ContextPtr                  ctx,
                                              KernelGraph::KernelGraphPtr graph,
                                              const uint                  period)
        {
            auto visitor = PeriodizeWorkitemValuesVisitor(ctx, graph, period);
            return visitor.call(expr);
        }

    }
}
