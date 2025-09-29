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
#include "forward.h"
#include "backward.h"
#include <cuda_runtime_api.h>
#include <cstdint>
#include <tuple>
#include "float_grad.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <fstream>
#include <iomanip>
#include "rasterizer_impl.h"

namespace CudaRasterizer
{

// Helper function to find the next-highest bit of the MSB
// on the CPU.
static inline uint32_t getHigherMsb(uint32_t n)
{
    uint32_t msb = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1)
    {
        step /= 2;
        if (n >> msb)
            msb += step;
        else
            msb -= step;
    }
    if (n >> msb)
        msb++;
    return msb;
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ 
inline void duplicateWithKeysJvp(
    int P,
    const FloatGradArray<float2> points_xy,
    const FloatGradArray<float> depths,
    const uint32_t* offsets,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted,
    int* radii,
    dim3 grid)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    // Generate no key/value pair for invisible Gaussians
    if (radii[idx] > 0)
    {
        // Find this Gaussian's offset in buffer for writing keys/values.
        uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
        uint2 rect_min, rect_max;

        getRect(get_data(points_xy[idx]), radii[idx], rect_min, rect_max, grid);

        // For each tile that the bounding rect overlaps, emit a 
        // key/value pair. The key is |  tile ID  |      depth      |,
        // and the value is the ID of the Gaussian. Sorting the values 
        // with this key yields Gaussian IDs in a list, such that they
        // are first sorted by tile and then by depth. 
        for (int y = rect_min.y; y < rect_max.y; y++)
        {
            for (int x = rect_min.x; x < rect_max.x; x++)
            {
                uint64_t key = y * grid.x + x;
                key <<= 32;
                key |= *((uint32_t*)&depths.data_ptr()[idx]);
                gaussian_keys_unsorted[off] = key;
                gaussian_values_unsorted[off] = idx;
                off++;
            }
        }
    }
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ 
inline void identifyTileRangesJvp(int L, uint64_t* point_list_keys, uint2* ranges)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= L)
        return;

    // Read tile ID from key. Update start/end of tile range if at limit.
    uint64_t key = point_list_keys[idx];
    uint32_t currtile = key >> 32;
    if (idx == 0)
        ranges[currtile].x = 0;
    else
    {
        uint32_t prevtile = point_list_keys[idx - 1] >> 32;
        if (currtile != prevtile)
        {
            ranges[prevtile].y = idx;
            ranges[currtile].x = idx;
        }
    }
    if (idx == L - 1)
        ranges[currtile].y = L;
}
        
// Forward rendering procedure for differentiable rasterization
// of Gaussians.
template <typename... JvpArgs>
int Rasterizer::forwardJvp(JvpArgs&&... jvp_args)
    // std::function<char* (size_t)> geometryBuffer,
    // std::function<char* (size_t)> binningBuffer,
    // std::function<char* (size_t)> imageBuffer,
    // const int P, int D, int M,
    // const float* background,
    // const int width, int height,
    // const float* means3D,
    // const float* shs,
    // const float* colors_precomp,
    // const float* opacities,
    // const float* scales,
    // const float scale_modifier,
    // const float* rotations,
    // const float* cov3D_precomp,
    // const float* viewmatrix,
    // const float* projmatrix,
    // const float* cam_pos,
    // const float tan_fovx, float tan_fovy,
    // const bool prefiltered,
    // float* out_color,
    // float* depth,
    // bool antialiasing,
    // int* radii = nullptr,
    // bool debug = false);
{
    auto jvp_args_tuple = std::forward_as_tuple(std::forward<JvpArgs>(jvp_args)...);
    std::function<char* (size_t)> geometryBuffer = std::get<0>(jvp_args_tuple);
    std::function<char* (size_t)> binningBuffer = std::get<1>(jvp_args_tuple);
    std::function<char* (size_t)> imageBuffer = std::get<2>(jvp_args_tuple);
    int P = std::get<3>(jvp_args_tuple);
    int D = std::get<4>(jvp_args_tuple);
    int M = std::get<5>(jvp_args_tuple);
    auto background = std::get<6>(jvp_args_tuple);
    int width = std::get<7>(jvp_args_tuple);
    int height = std::get<8>(jvp_args_tuple);
    auto means3D = std::get<9>(jvp_args_tuple);
    auto shs = std::get<10>(jvp_args_tuple);
    auto colors_precomp = std::get<11>(jvp_args_tuple);
    auto opacities = std::get<12>(jvp_args_tuple);
    auto scales = std::get<13>(jvp_args_tuple);
    auto scale_modifier = std::get<14>(jvp_args_tuple);
    auto rotations = std::get<15>(jvp_args_tuple);
    auto cov3D_precomp = std::get<16>(jvp_args_tuple);
    auto viewmatrix = std::get<17>(jvp_args_tuple);
    auto projmatrix = std::get<18>(jvp_args_tuple);
    auto cam_pos = std::get<19>(jvp_args_tuple);
    auto tan_fovx = std::get<20>(jvp_args_tuple);
    auto tan_fovy = std::get<21>(jvp_args_tuple);
    const bool prefiltered = std::get<22>(jvp_args_tuple);
    auto out_color = std::get<23>(jvp_args_tuple);
    auto depth = std::get<24>(jvp_args_tuple);
    bool antialiasing = std::get<25>(jvp_args_tuple);
    int* radii = std::get<26>(jvp_args_tuple);
    bool debug = std::get<27>(jvp_args_tuple);

    auto focal_y = height / (2.0f * tan_fovy);
    auto focal_x = width / (2.0f * tan_fovx);

    size_t chunk_size = required<GeometryStateJvp>(P);
    char* geom_chunkptr = geometryBuffer(chunk_size);
    char* geom_chunkptr_start = geom_chunkptr;
    GeometryStateJvp geomState = GeometryStateJvp::fromChunk(geom_chunkptr, P);

    if (radii == nullptr)
    {
        radii = geomState.internal_radii;
    }

    constexpr int NUM_CHANNELS = CudaRasterizer::NUM_CHANNELS;
    constexpr int BLOCK_X = CudaRasterizer::BLOCK_X;
    constexpr int BLOCK_Y = CudaRasterizer::BLOCK_Y;

    dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Dynamically resize image-based auxiliary buffers during training
    size_t img_chunk_size = required<ImageStateJvp>(width * height);
    char* img_chunkptr = imageBuffer(img_chunk_size);
    char* img_chunkptr_start = img_chunkptr;
    ImageStateJvp imgState = ImageStateJvp::fromChunk(img_chunkptr, width * height);

    if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
    {
        throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
    }

    FORWARD::preprocessJvp(
        P, D, M,
        means3D,
        cast<glm::vec3>(scales),
        scale_modifier,
        cast<glm::vec4>(rotations),
        opacities,
        shs,
        geomState.clamped,
        cov3D_precomp,
        colors_precomp,
        viewmatrix, 
        projmatrix,
        cast<glm::vec3>(cam_pos),
        width, height,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        radii,
        geomState.means2D,
        geomState.depths,
        geomState.cov3D,
        geomState.rgb,
        geomState.conic_opacity,
        tile_grid,
        geomState.tiles_touched,
        prefiltered,
        antialiasing);

    // Compute prefix sum over full list of touched tile counts by Gaussians
    // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

    // Retrieve total number of Gaussian instances to launch and resize aux buffers
    int num_rendered;
    CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

    size_t binning_chunk_size = required<BinningState>(num_rendered);
    char* binning_chunkptr = binningBuffer(binning_chunk_size);
    char* binning_chunkptr_start = binning_chunkptr;
    BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

    // For each instance to be rendered, produce adequate [ tile | depth ] key 
    // and corresponding dublicated Gaussian indices to be sorted
    duplicateWithKeysJvp << <(P + 255) / 256, 256 >> > (
        P,
        geomState.means2D,
        geomState.depths,
        geomState.point_offsets,
        binningState.point_list_keys_unsorted,
        binningState.point_list_unsorted,
        radii,
        tile_grid)
    CHECK_CUDA(, debug)

    int bit = getHigherMsb(tile_grid.x * tile_grid.y);

    // Sort complete list of (duplicated) Gaussian indices by keys
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        binningState.list_sorting_space,
        binningState.sorting_size,
        binningState.point_list_keys_unsorted, binningState.point_list_keys,
        binningState.point_list_unsorted, binningState.point_list,
        num_rendered, 0, 32 + bit), debug)

    CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

    // Identify start and end of per-tile workloads in sorted list
    if (num_rendered > 0)
        identifyTileRangesJvp << <(num_rendered + 255) / 256, 256 >> > (
            num_rendered,
            binningState.point_list_keys,
            imgState.ranges);
    CHECK_CUDA(, debug)

    static_assert(is_float_grad<decltype(colors_precomp)>::value == 
                  is_float_grad<decltype(geomState.rgb)>::value,
                  "Colors precomputed and RGB must be of the same type (float or FloatGradArray).");

    // Let each tile blend its range of Gaussians independently in parallel
    auto feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
    CHECK_CUDA(FORWARD::renderJvp(
        tile_grid, block,
        imgState.ranges,
        binningState.point_list,
        width, height,
        geomState.means2D,
        feature_ptr,
        geomState.conic_opacity,
        imgState.accum_alpha,
        imgState.n_contrib,
        background,
        out_color,
        geomState.depths,
        depth), debug)

    // Free up information not used by buffers
    // We need to keep the grads because they are used in hessian backward
    char* binning_chunkptr_end = binning_chunkptr_start;
    ReducedBinningState reduced_binning_state = ReducedBinningState::fromChunk(binning_chunkptr_end, num_rendered);
    size_t final_binning_size = binning_chunkptr_end - binning_chunkptr_start;
    binningBuffer(final_binning_size);

    return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
template <typename... JvpArgs>
void Rasterizer::backwardJvp(JvpArgs&&... jvp_args)
    // const int P, int D, int M, int R,
    // const float* background,
    // const int width, int height,
    // const float* means3D,
    // const float* shs,
    // const float* colors_precomp,
    // const float* opacities,
    // const float* scales,
    // const float scale_modifier,
    // const float* rotations,
    // const float* cov3D_precomp,
    // const float* viewmatrix,
    // const float* projmatrix,
    // const float* campos,
    // const float tan_fovx, float tan_fovy,
    // const int* radii,
    // char* geom_buffer,
    // char* binning_buffer,
    // char* img_buffer,
    // const float* dL_dpix,
    // const float* dL_invdepths,
    // float* dL_dmean2D,
    // float* dL_dconic,
    // float* dL_dopacity,
    // float* dL_dcolor,
    // float* dL_dinvdepth,
    // float* dL_dmean3D,
    // float* dL_dcov3D,
    // float* dL_dsh,
    // float* dL_dscale,
    // float* dL_drot,
    // bool antialiasing,
    // bool debug)
{
    auto jvp_args_tuple = std::forward_as_tuple(std::forward<JvpArgs>(jvp_args)...);
    const int P = std::get<0>(jvp_args_tuple);
    const int D = std::get<1>(jvp_args_tuple);
    const int M = std::get<2>(jvp_args_tuple);
    const int R = std::get<3>(jvp_args_tuple);
    auto background = std::get<4>(jvp_args_tuple);
    const int width = std::get<5>(jvp_args_tuple);
    const int height = std::get<6>(jvp_args_tuple);
    auto means3D = std::get<7>(jvp_args_tuple);
    auto shs = std::get<8>(jvp_args_tuple);
    auto colors_precomp = std::get<9>(jvp_args_tuple);
    auto opacities = std::get<10>(jvp_args_tuple);
    auto scales = std::get<11>(jvp_args_tuple);
    auto scale_modifier = std::get<12>(jvp_args_tuple);
    auto rotations = std::get<13>(jvp_args_tuple);
    auto cov3D_precomp = std::get<14>(jvp_args_tuple);
    auto viewmatrix = std::get<15>(jvp_args_tuple);
    auto projmatrix = std::get<16>(jvp_args_tuple);
    auto campos = std::get<17>(jvp_args_tuple);
    auto tan_fovx = std::get<18>(jvp_args_tuple);
    auto tan_fovy = std::get<19>(jvp_args_tuple);
    const int* radii = std::get<20>(jvp_args_tuple);
    char* geom_buffer = std::get<21>(jvp_args_tuple);
    char* binning_buffer = std::get<22>(jvp_args_tuple);
    char* img_buffer = std::get<23>(jvp_args_tuple);
    auto dL_dpix = std::get<24>(jvp_args_tuple);
    auto dL_invdepths = std::get<25>(jvp_args_tuple);
    auto dL_dmean2D = std::get<26>(jvp_args_tuple);
    auto dL_dconic = std::get<27>(jvp_args_tuple);
    auto dL_dopacity = std::get<28>(jvp_args_tuple);
    auto dL_dcolor = std::get<29>(jvp_args_tuple);
    auto dL_dinvdepth = std::get<30>(jvp_args_tuple);
    auto dL_dmean3D = std::get<31>(jvp_args_tuple);
    auto dL_dcov3D = std::get<32>(jvp_args_tuple);
    auto dL_dsh = std::get<33>(jvp_args_tuple);
    auto dL_dscale = std::get<34>(jvp_args_tuple);
    auto dL_drot = std::get<35>(jvp_args_tuple);
    bool antialiasing = std::get<36>(jvp_args_tuple);
    bool debug = std::get<37>(jvp_args_tuple);

    GeometryStateJvp geomState = GeometryStateJvp::fromChunk(geom_buffer, P);
    ReducedBinningState binningState = ReducedBinningState::fromChunk(binning_buffer, R);
    ImageStateJvp imgState = ImageStateJvp::fromChunk(img_buffer, width * height);

    if (radii == nullptr)
    {
        radii = geomState.internal_radii;
    }

    auto focal_y = height / (2.0f * tan_fovy);
    auto focal_x = width / (2.0f * tan_fovx);

    const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Compute loss gradients w.r.t. 2D mean position, conic matrix,
    // opacity and RGB of Gaussians from per-pixel loss gradients.
    // If we were given precomputed colors and not SHs, use them.
    auto color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;

    CHECK_CUDA(BACKWARD::renderJvp(
        tile_grid,
        block,
        imgState.ranges,
        binningState.point_list,
        width, height,
        background,
        geomState.means2D,
        geomState.conic_opacity,
        color_ptr,
        geomState.depths,
        imgState.accum_alpha,
        imgState.n_contrib,
        dL_dpix,
        dL_invdepths,
        cast<float3>(dL_dmean2D),
        cast<float4>(dL_dconic),
        dL_dopacity,
        dL_dcolor,
        dL_dinvdepth), debug);

    // Take care of the rest of preprocessing. Was the precomputed covariance
    // given to us or a scales/rot pair? If precomputed, pass that. If not,
    // use the one we computed ourselves.
    auto cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
    CHECK_CUDA(BACKWARD::preprocessJvp(P, D, M,
        cast<float3>(means3D),
        radii,
        shs,
        geomState.clamped,
        opacities,
        cast<glm::vec3>(scales),
        cast<glm::vec4>(rotations),
        scale_modifier,
        cov3D_ptr,
        viewmatrix,
        projmatrix,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        cast<glm::vec3>(campos),
        cast<float3>(dL_dmean2D),
        dL_dconic,
        dL_dinvdepth,
        dL_dopacity,
        cast<glm::vec3>(dL_dmean3D),
        dL_dcolor,
        dL_dcov3D,
        dL_dsh,
        cast<glm::vec3>(dL_dscale),
        cast<glm::vec4>(dL_drot),
        antialiasing), debug);
}

};  // namespace CudaRasterizer
