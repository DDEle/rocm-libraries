// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>
#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    inline std::string toString(GPUArchitectureGFX const& gfx)
    {
        switch(gfx)
        {
        case GPUArchitectureGFX::GFX908:
            return "gfx908";
        case GPUArchitectureGFX::GFX90A:
            return "gfx90a";
        case GPUArchitectureGFX::GFX942:
            return "gfx942";
        case GPUArchitectureGFX::GFX950:
            return "gfx950";
        case GPUArchitectureGFX::GFX1010:
            return "gfx1010";
        case GPUArchitectureGFX::GFX1011:
            return "gfx1011";
        case GPUArchitectureGFX::GFX1012:
            return "gfx1012";
        case GPUArchitectureGFX::GFX1030:
            return "gfx1030";
        case GPUArchitectureGFX::GFX1200:
            return "gfx1200";
        case GPUArchitectureGFX::GFX1201:
            return "gfx1201";
        case GPUArchitectureGFX::GFX1250:
            return "gfx1250";
        default:
            return "gfxunknown";
        }
    }

    inline std::ostream& operator<<(std::ostream& stream, GPUArchitectureGFX const& gfx)
    {
        return stream << toString(gfx);
    }

    inline std::string name(GPUArchitectureGFX const& gfx)
    {
        switch(gfx)
        {
        case GPUArchitectureGFX::GFX908:
            return "AMD CDNA 1";
        case GPUArchitectureGFX::GFX90A:
            return "AMD CDNA 2";
        case GPUArchitectureGFX::GFX942:
            return "AMD CDNA 3";
        case GPUArchitectureGFX::GFX950:
            return "AMD CDNA 4";
        case GPUArchitectureGFX::GFX1012:
            return "AMD RDNA 1";
        case GPUArchitectureGFX::GFX1030:
            return "AMD RDNA 2";
        case GPUArchitectureGFX::GFX1200:
        case GPUArchitectureGFX::GFX1201:
            return "AMD RDNA 4";
        case GPUArchitectureGFX::GFX1250:
            return "AMD CDNA 5";
        default:
            return "unknown";
        }
    }

    inline std::string GPUArchitectureFeatures::toString() const
    {
        std::string rv = "";
        if(sramecc)
        {
            rv = concatenate(rv, "sramecc+");
        }
        if(xnack)
        {
            if(!rv.empty())
            {
                rv = concatenate(rv, ":");
            }
            rv = concatenate(rv, "xnack+");
        }
        return rv;
    }

    inline std::string GPUArchitectureFeatures::toLLVMString() const
    {
        std::string rv = "";
        if(xnack)
        {
            rv = concatenate(rv, "+xnack");
        }
        if(sramecc)
        {
            if(xnack)
                rv = concatenate(rv, ",");
            rv = concatenate(rv, "+sramecc");
        }
        return rv;
    }

    inline std::string GPUArchitectureTarget::toString() const
    {
        std::string rv{rocRoller::toString(gfx)};
        if(asicRevisionId >= 0)
            rv = concatenate(rv, "rev", asicRevisionId);

        if(features.sramecc || features.xnack)
            rv = concatenate(rv, ":", features.toString());
        return rv;
    }

    inline std::string GPUArchitectureTarget::toAssemblerString() const
    {
        if(features.sramecc || features.xnack)
            return concatenate(gfx, ":", features.toString());
        else
            return rocRoller::toString(gfx);
    }

    inline std::string GPUArchitectureTarget::name() const
    {
        return rocRoller::name(gfx);
    }

    inline GPUArchitectureTarget GPUArchitectureTarget::fromString(std::string const& archStr,
                                                                   int asicRevisionId)
    {
        GPUArchitectureTarget rv;

        int         start = 0;
        size_t      end   = archStr.find(":");
        std::string arch  = archStr.substr(start, end - start);

        std::regex  pattern(R"(rev(\d+))");
        std::smatch revIdMatch;
        if(std::regex_search(arch, revIdMatch, pattern))
        {
            rv.asicRevisionId = std::stoi(revIdMatch[1]);
            arch              = arch.substr(0, revIdMatch.position(0));
        }
        else if(arch == "gfx1250")
        {
            if(asicRevisionId == 2)
            {
                int newRevId = Settings::Get(Settings::GFX1250AsicRevisionId);
                Log::warn("Overriding current device asic revision id from {} to {}",
                          asicRevisionId,
                          newRevId);
                asicRevisionId = newRevId;
            }
            rv.asicRevisionId = asicRevisionId;
        }

        rv.gfx = rocRoller::fromString<GPUArchitectureGFX>(arch);

        while(end != std::string::npos)
        {
            start               = end + 1;
            end                 = archStr.find(":", start);
            std::string feature = archStr.substr(start, end - start);
            if(feature == "xnack+")
            {
                rv.features.xnack = true;
            }
            else if(feature == "sramecc+")
            {
                rv.features.sramecc = true;
            }
        }
        return rv;
    }
}
