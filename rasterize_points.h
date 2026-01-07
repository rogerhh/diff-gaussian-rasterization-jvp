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
    
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
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
    const bool debug,
    const bool track_weights);
    
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 PreprocessBackwardCUDA(
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
    const int image_height,
    const int image_width,
    const torch::Tensor& dL_dout_means2D,
    const torch::Tensor& dL_dout_conic,
    const torch::Tensor& dL_dout_invdepth,
    const torch::Tensor& dL_dout_colors,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const torch::Tensor& geomBuffer,
    const int R,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const bool antialiasing,
    const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDAJvp(
    const torch::Tensor& background,
    const torch::Tensor& background_grad,
    const torch::Tensor& means3D,
    const torch::Tensor& means3D_grad,
    const torch::Tensor& radii,
    const torch::Tensor& colors,
    const torch::Tensor& colors_grad,
    const torch::Tensor& scales,
    const torch::Tensor& scales_grad,
    const torch::Tensor& opacities,
    const torch::Tensor& opacities_grad,
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
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dout_color_grad,
    const torch::Tensor& dL_dout_invdepth,
    const torch::Tensor& dL_dout_invdepth_grad,
    const torch::Tensor& sh,
    const torch::Tensor& sh_grad,
    const int degree,
    const torch::Tensor& campos,
    const torch::Tensor& campos_grad,
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

std::tuple<torch::Tensor, torch::Tensor> ComputeRelocationCUDA(
		torch::Tensor& opacity_old,
		torch::Tensor& scale_old,
		torch::Tensor& N,
		torch::Tensor& binoms,
		const int n_max);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ComputeTrustRegionStepCUDA(
        const torch::Tensor& xyz_params,
        const torch::Tensor& scaling_params,
        const torch::Tensor& quat_params,
        const torch::Tensor& opacity_params,
        const torch::Tensor& shs_params,
        torch::Tensor& xyz_params_step,
        torch::Tensor& scaling_params_step,
        torch::Tensor& quat_params_step,
        torch::Tensor& opacity_params_step,
        torch::Tensor& shs_params_step,
        const float trust_radius,
        const float min_mass_scaling,
        const float max_mass_scaling,
        const float scale_modifier,
        const float quat_norm_tr);
