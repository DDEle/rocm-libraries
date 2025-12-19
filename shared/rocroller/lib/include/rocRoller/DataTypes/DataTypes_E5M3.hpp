/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2025 AMD ROCm(TM) Software
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

#include <limits>
#ifdef ROCROLLER_USE_HIP
#include <hip/hip_runtime.h>
#endif

#include <rocRoller/DataTypes/DataTypes_F8_Utils.hpp>
#include <rocRoller/DataTypes/DataTypes_Scale_Utils.hpp>
#include <rocRoller/DataTypes/DistinctType.hpp>

namespace rocRoller
{
    /**
     * \ingroup DataTypes
     */
    struct E5M3
    {
        E5M3()
            : scale(0)
        {
        }

        explicit E5M3(uint8_t scale)
            : scale(scale)
        {
        }

        explicit E5M3(float scale)
        {
            this->scale = floatToE5M3(scale);
        }

        uint8_t scale;

        inline operator float() const
        {
            return E5M3ToFloat(this->scale);
        }

        explicit inline operator uint8_t() const
        {
            return this->scale;
        }
    };
    static_assert(sizeof(E5M3) == 1, "E5M3 must be 1 byte.");

    inline E5M3 operator-(E5M3 const& a)
    {
        return static_cast<E5M3>(-static_cast<float>(a));
    }

    inline std::ostream& operator<<(std::ostream& os, const E5M3 val)
    {
        os << val.scale;
        return os;
    }
} // namespace rocRoller

namespace std
{

    template <typename T>
    requires(std::is_convertible_v<T, uint8_t>&& std::is_integral_v<T>) inline bool
        operator==(rocRoller::E5M3 const& a, T const& b)
    {
        return a.scale == static_cast<uint8_t>(b);
    }

    template <typename T>
    requires(std::is_convertible_v<T, uint8_t>&& std::is_integral_v<T>) inline bool
        operator!=(rocRoller::E5M3 const& a, T const& b)
    {
        return a.scale != static_cast<uint8_t>(b);
    }

    template <>
    struct is_floating_point<rocRoller::E5M3> : true_type
    {
    };

    template <>
    struct hash<rocRoller::E5M3>
    {
        size_t operator()(const rocRoller::E5M3& a) const
        {
            return hash<uint8_t>()(a.scale);
        }
    };
} // namespace std
