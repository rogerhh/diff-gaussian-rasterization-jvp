/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include "forward.h"
#include <cuda_runtime_api.h>
#include <cstdint>
#include <tuple>
#include "float_grad.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <fstream>
#include <iomanip>

namespace CudaRasterizer
{
    template <typename T>
    static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
    {
        std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
        ptr = reinterpret_cast<T*>(offset);
        chunk = reinterpret_cast<char*>(ptr + count);
    }

    template<typename T> 
    size_t required(size_t P)
    {
        char* size = nullptr;
        T::fromChunk(size, P);
        return ((size_t)size) + 128;
    }
};

