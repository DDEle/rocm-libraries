// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/Expression_fwd.hpp>

#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelGraph/KernelGraph_fwd.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager_fwd.hpp>

namespace rocRoller
{
    namespace Expression
    {
        using ExpressionTransformType = std::function<ExpressionPtr(ExpressionPtr)>;

        ExpressionPtr identity(ExpressionPtr expr);

        /**
         * Transform sub-expressions of `expr` into new kernel arguments
         *
         * Return value should be Translate time or KernelExecute time evaluable.
         */
        ExpressionPtr launchTimeSubExpressions(ExpressionPtr expr, ContextPtr context);

        /**
         * Restore any command arguments that have been cleaned (transformed from command
         * arguments into kernel arguments.)
         *
         * Return value should be Translate time or KernelLaunch time evaluable.
         */
        ExpressionPtr restoreCommandArguments(ExpressionPtr expr);

        /**
         * @brief Attempt to replace division operations found within an expression with faster operations.
         *
         * @param expr Input expression
         * @param context
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr fastDivision(ExpressionPtr expr, ContextPtr context);

        /**
         * Ensures that the kernel arguments will include the magic constants required to divide/modulo
         * by `expr`.
         * Requires `expr` to have a type of either Int32 or Int64, and to be evaluable at kernel launch time.
         */
        void enableDivideBy(ExpressionPtr expr, ContextPtr context);

        /**
         * Gets expressions which can be used to compute magic division of denominator.
         *
         * Returns [magicMultiple, magicShift, magicSign, magicShiftMSB]
         *
         * If denominator is unsigned, magicSign will be nullptr.
         */
        std::tuple<ExpressionPtr, ExpressionPtr, ExpressionPtr, ExpressionPtr>
            getMagicDivisionParams(ExpressionPtr denominator, ContextPtr context);

        /**
         * @brief Attempt to replace multiplication operations found within an expression with faster operations.
         *
         * @param expr Input expression
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr fastMultiplication(ExpressionPtr expr);

        /**
         * Attempt to combine multiple shifts:
         * - Opposite shifts by same amount: mask off bits that would be zeroed out.
         */
        ExpressionPtr combineShifts(ExpressionPtr expr);

        /**
         * Splits BitfieldCombine expressions that target more than 32 bits into a Concatenate of 32 bit sub-expressions.
         */
        ExpressionPtr splitBitfieldCombine(ExpressionPtr expr);

        /**
         * Splits uint64_t literal operands in a Concatenate expression into two Raw32 operands.
         */
        Concatenate splitConcatenate(Concatenate const& expr);

        /**
         * @brief Simplify expressions
         *
         * @param expr Input expression
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr simplify(ExpressionPtr expr);

        /**
         * @brief Fuse binary expressions into ternaries.
         *
         * @param expr Input expression
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr fuseTernary(ExpressionPtr expr);

        /**
         * @brief Fuse binary expressions if one combination is able to be condensed by association
         *
         * @param expr Input expression
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr fuseAssociative(ExpressionPtr expr);

        ExpressionPtr dataFlowTagPropagation(ExpressionPtr             expr,
                                             RegisterTagManager const& tagManager);

        /**
         * Resolve all DataFlowTags in the given expression.
         * Each DataFlowTag is transformed into either an expression or a register value.
         *
         * @param expr Input expression
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr dataFlowTagPropagation(ExpressionPtr expr, ContextPtr context);

        /**
         * Resolve all PositionalArguments in the given expression.
         *
         * Each PositionalArgument is transformed into the expression
         * in position `slot` of `arguments`.
         */
        ExpressionPtr positionalArgumentPropagation(ExpressionPtr              expr,
                                                    std::vector<ExpressionPtr> arguments);

        /**
         * @brief Attempt to compute e^x operations by using exp2(x * log2(e)).
         *
         * @param expr Input expression
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr lowerExponential(ExpressionPtr expr);

        /**
         * @brief Propagate converts to input values
         *
         * @param expr Input expression
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr convertPropagation(ExpressionPtr expr);

        ExpressionPtr makeScalar(ExpressionPtr expr);

        /**
         * @brief Replace unsigned ArithmeticShiftR with LogicalShiftR
         *
         */
        ExpressionPtr lowerUnsignedArithmeticShiftR(ExpressionPtr expr);

        /**
         * Helper (lambda/transducer) for applying all fast arithmetic transformations.
         *
         * Usage:
         *
         *   FastArithmetic transformer(context);
         *   auto fast_expr = transformer(expr);
         *
         * Can also be passed as an ExpressionTransducer.
         */
        struct FastArithmetic
        {
            FastArithmetic() = delete;
            explicit FastArithmetic(ContextPtr);

            ExpressionPtr                        operator()(ExpressionPtr) const;
            std::vector<ExpressionTransformType> getTransforms() const;
            ExpressionPtr                        applyTransforms(ExpressionPtr,
                                                                 const std::vector<ExpressionTransformType>&) const;

        private:
            ContextPtr m_context;
        };

        /**
         * @brief Transform RandomNumber expression into a set of expressions that
         *        implement the PRNG algorithm when PRNG instruction is unavailable.
         *
         * @param expr Input expression
         * @param context
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr lowerPRNG(ExpressionPtr exp, ContextPtr context);

        /**
         * @brief Resolve all ValuePtr expressions that are bitfields into
         * BitFieldExtract expressions.
         *
         * @param expr Input expression
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr lowerBitfieldValues(ExpressionPtr expr);

        /**
         * @brief Attempt to replace a BitfieldCombine expr with
         * a composite expression consisting of shift and bitwise
         * AND/OR
         *
         * @param expr Input expression
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr lowerBitfieldCombine(ExpressionPtr expr);

        /**
         * @brief Periodize the WorkitemX index space within and across waves.
         *
         * This transform modifies all occurances of the WorkitemX
         * index within an expression.  Workitem indices are repeated
         * with a given period, p, within each wave; and are strided,
         * by p, across waves.
         *
         * The transformation is defined as follows:
         *
         *   WI -> (WI & (p - 1)) + floor(WI / wavefrontSize) * p
         *
         * For example, consider a Wave-32 architecture with a kernel
         * launched with 256 workitems per workgroup.
         *
         * Originally, the workitem X indices are:
         *
         *      wave 0: [  0   1   2   3   4   5   6   7 ...  31]
         *      wave 1: [ 32  33  34  35  36  37  38  39 ...  63]
         *      ...
         *      wave 7: [224 225 226 227 228 229 230 231 ... 255]
         *
         * With p=4, after periodization:
         *
         *      wave 0: [  0   1   2   3   0   1   2   3 ...   3]
         *      wave 1: [  4   5   6   7   4   5   6   7 ...   7]
         *      ...
         *      wave 7: [ 28  29  30  31  28  29  30  31 ...  31]
         *
         * @param expr Input expression
         * @param context
         * @param graph The kernel graph containing the workitem coordinates.
         * @param period The number of unique indicies to repeat in a wave,
         *               must be a power of 2.
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr periodizeWorkitemValues(ExpressionPtr               expr,
                                              ContextPtr                  ctx,
                                              KernelGraph::KernelGraphPtr graph,
                                              const uint                  period);
    }
}