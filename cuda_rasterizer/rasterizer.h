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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <cstdint>
#include <vector>
#include <functional>
#include "float_grad.h"

namespace CudaRasterizer
{
    class Rasterizer
    {
    public:

        static void markVisible(
            int P,
            float* means3D,
            float* viewmatrix,
            float* projmatrix,
            bool* present);

        static int forward(
            std::function<char* (size_t)> geometryBuffer,
            std::function<char* (size_t)> binningBuffer,
            std::function<char* (size_t)> imageBuffer,
            const int P, int D, int M,
            const float* background,
            const int width, int height,
            const float* means3D,
            const float* shs,
            const float* colors_precomp,
            const float* opacities,
            const float* scales,
            const float scale_modifier,
            const float* rotations,
            const float* cov3D_precomp,
            const float* viewmatrix,
            const float* projmatrix,
            const float* cam_pos,
            const float tan_fovx, float tan_fovy,
            const bool prefiltered,
            float* out_color,
            float* depth,
            bool antialiasing,
            int* radii = nullptr,
            bool debug = false,
            bool track_weights = false,
            float* weights = nullptr);

        static int testfunc() { return 0; }

        template <typename... JvpArgs>
        static int forwardJvp(JvpArgs&&... jvp_args);
        //     std::function<char* (size_t)> geometryBuffer,
        //     std::function<char* (size_t)> binningBuffer,
        //     std::function<char* (size_t)> imageBuffer,
        //     const int P, int D, int M,
        //     const float* background,
        //     const int width, int height,
        //     const float* means3D,
        //     const float* shs,
        //     const float* colors_precomp,
        //     const float* opacities,
        //     const float* scales,
        //     const float scale_modifier,
        //     const float* rotations,
        //     const float* cov3D_precomp,
        //     const float* viewmatrix,
        //     const float* projmatrix,
        //     const float* cam_pos,
        //     const float tan_fovx, float tan_fovy,
        //     const bool prefiltered,
        //     float* out_color,
        //     float* depth,
        //     bool antialiasing,
        //     int* radii = nullptr,
        //     bool debug = false);

        static void backward(
            const int P, int D, int M, int R,
            const float* background,
            const int width, int height,
            const float* means3D,
            const float* shs,
            const float* colors_precomp,
            const float* opacities,
            const float* scales,
            const float scale_modifier,
            const float* rotations,
            const float* cov3D_precomp,
            const float* viewmatrix,
            const float* projmatrix,
            const float* campos,
            const float tan_fovx, float tan_fovy,
            const int* radii,
            char* geom_buffer,
            char* binning_buffer,
            char* image_buffer,
            const float* dL_dpix,
            const float* dL_invdepths,
            float* dL_dmean2D,
            float* dL_dconic,
            float* dL_dopacity,
            float* dL_dcolor,
            float* dL_dinvdepth,
            float* dL_dmean3D,
            float* dL_dcov3D,
            float* dL_dsh,
            float* dL_dscale,
            float* dL_drot,
            bool antialiasing,
            bool debug);

        static void preprocessBackward(
            const int P, int D, int M, int R,
            const float* background,
            const int width, int height,
            const float* means3D,
            const float* shs,
            const float* colors_precomp,
            const float* opacities,
            const float* scales,
            const float scale_modifier,
            const float* rotations,
            const float* cov3D_precomp,
            const float* viewmatrix,
            const float* projmatrix,
            const float* campos,
            const float tan_fovx, float tan_fovy,
            const int* radii,
            char* geom_buffer,
            char* binning_buffer,
            char* img_buffer,
            const float* dL_dmean2D,
            const float* dL_dconic,
            const float* dL_dinvdepth,
            float* dL_dcolor,
            float* dL_dopacity,
            float* dL_dmean3D,
            float* dL_dcov3D,
            float* dL_dsh,
            float* dL_dscale,
            float* dL_drot,
            bool antialiasing,
            bool debug);

        template <typename... JvpArgs>
        static void backwardJvp(JvpArgs&&... jvp_args);
        //     const int P, int D, int M, int R,
        //     const float* background,
        //     const int width, int height,
        //     const float* means3D,
        //     const float* shs,
        //     const float* colors_precomp,
        //     const float* opacities,
        //     const float* scales,
        //     const float scale_modifier,
        //     const float* rotations,
        //     const float* cov3D_precomp,
        //     const float* viewmatrix,
        //     const float* projmatrix,
        //     const float* campos,
        //     const float tan_fovx, float tan_fovy,
        //     const int* radii,
        //     char* geom_buffer,
        //     char* binning_buffer,
        //     char* image_buffer,
        //     const float* dL_dpix,
        //     const float* dL_invdepths,
        //     float* dL_dmean2D,
        //     float* dL_dconic,
        //     float* dL_dopacity,
        //     float* dL_dcolor,
        //     float* dL_dinvdepth,
        //     float* dL_dmean3D,
        //     float* dL_dcov3D,
        //     float* dL_dsh,
        //     float* dL_dscale,
        //     float* dL_drot,
        //     bool antialiasing,
        //     bool debug);
    };

    template <typename T>
    static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment);

    struct GeometryState
    {
        size_t scan_size;
        float* depths;
        char* scanning_space;
        bool* clamped;
        int* internal_radii;
        float2* means2D;
        float* cov3D;
        float4* conic_opacity;
        float* rgb;
        uint32_t* point_offsets;
        uint32_t* tiles_touched;

        static GeometryState fromChunk(char*& chunk, size_t P);
    };

    struct GeometryStateJvp
    {
        // Note: The JVP version of GeometryState needs to have 
        // the tangent vectors at the end of the structure
        // Because it may be re-interpreted as a GeometryState

        size_t scan_size;
        FloatGradArray<float> depths;
        char* scanning_space;
        bool* clamped;
        int* internal_radii;
        FloatGradArray<float2> means2D;
        FloatGradArray<float> cov3D;
        FloatGradArray<float4> conic_opacity;
        FloatGradArray<float> rgb;
        uint32_t* point_offsets;
        uint32_t* tiles_touched;

        static GeometryStateJvp fromChunk(char*& chunk, size_t P);
    };

    struct ImageState
    {
        uint2* ranges;
        uint32_t* n_contrib;
        float* accum_alpha;

        static ImageState fromChunk(char*& chunk, size_t N);
    };

    struct ImageStateJvp
    {
        // Note: The JVP version of GeometryState needs to have 
        // the tangent vectors at the end of the structure
        // Because it may be re-interpreted as a ImageState
        
        uint2* ranges;
        uint32_t* n_contrib;
        FloatGradArray<float> accum_alpha;

        static ImageStateJvp fromChunk(char*& chunk, size_t N);
    };

    struct BinningState
    {
        size_t sorting_size;
        uint64_t* point_list_keys_unsorted;
        uint64_t* point_list_keys;
        uint32_t* point_list_unsorted;
        uint32_t* point_list;
        char* list_sorting_space;

        static BinningState fromChunk(char*& chunk, size_t P);
    };

    struct ReducedBinningState
    {
        uint32_t* point_list;
        static ReducedBinningState fromChunk(char*& chunk, size_t P);
    };

    template<typename T> 
    size_t required(size_t P);
};

#include "rasterizer_impl.h"
#include "rasterizer_jvp_impl.h"

#endif
