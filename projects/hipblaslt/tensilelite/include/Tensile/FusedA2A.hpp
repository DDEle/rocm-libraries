// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#include <stdexcept>
#include <string>

namespace TensileLite
{
    enum class FusedA2A : int
    {
        None = 0,
        GemmA2A,
        A2AGemm,
        Count
    };

    inline std::string ToString(FusedA2A mode)
    {
        switch(mode)
        {
        case FusedA2A::None:
            return "None";
        case FusedA2A::GemmA2A:
            return "GemmA2A";
        case FusedA2A::A2AGemm:
            return "A2AGemm";
        default:
            return "Invalid";
        }
    }

    inline FusedA2A fusedA2AFromString(std::string const& name)
    {
        for(int i = 0; i < static_cast<int>(FusedA2A::Count); i++)
            if(ToString(static_cast<FusedA2A>(i)) == name)
                return static_cast<FusedA2A>(i);
        throw std::invalid_argument("FusedA2A '" + name
                                    + "' is not one of None, GemmA2A, A2AGemm");
    }

    inline std::ostream& operator<<(std::ostream& stream, FusedA2A mode)
    {
        return stream << ToString(mode);
    }
} // namespace TensileLite
