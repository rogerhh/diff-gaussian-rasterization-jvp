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
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
    
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx, 
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool antialiasing,
    const bool debug);
    
std::tuple<int, 
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDAJvp(
    const torch::Tensor& background,
    const torch::Tensor& background_grad,
    const torch::Tensor& means3D,
    const torch::Tensor& means3D_grad,
    const torch::Tensor& colors,
    const torch::Tensor& colors_grad,
    const torch::Tensor& opacity,
    const torch::Tensor& opacity_grad,
    const torch::Tensor& scales,
    const torch::Tensor& scales_grad,
    const torch::Tensor& rotations,
    const torch::Tensor& rotations_grad,
    const float scale_modifier,
    const float scale_modifier_grad,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& cov3D_precomp_grad,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& viewmatrix_grad,
    const torch::Tensor& projmatrix,
    const torch::Tensor& projmatrix_grad,
    const float tan_fovx, 
    const float tan_fovx_grad,
    const float tan_fovy,
    const float tan_fovy_grad,
    const int image_height,
    const int image_width,
    const torch::Tensor& sh,
    const torch::Tensor& sh_grad,
    const int degree,
    const torch::Tensor& campos,
    const torch::Tensor& campos_grad,
    const bool prefiltered,
    const bool antialiasing,
    const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& radii,
    const torch::Tensor& colors,
    const torch::Tensor& scales,
    const torch::Tensor& opacities,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx, 
    const float tan_fovy,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dout_invdepth,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const torch::Tensor& geomBuffer,
    const int R,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const bool antialiasing,
    const bool debug);
        
torch::Tensor markVisible(
        torch::Tensor& means3D,
        torch::Tensor& viewmatrix,
        torch::Tensor& projmatrix);
