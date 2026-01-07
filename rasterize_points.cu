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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/utils.h"
#include "cuda_rasterizer/trust_region.h"
#include <fstream>
#include <string>
#include <functional>
#include "float_grad.h"

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

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
    const bool track_weights)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  constexpr int NUM_CHANNELS = CudaRasterizer::NUM_CHANNELS;

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_invdepth = torch::full({0, H, W}, 0.0, float_opts);
  float* out_invdepthptr = nullptr;

  out_invdepth = torch::full({1, H, W}, 0.0, float_opts).contiguous();
  out_invdepthptr = out_invdepth.data<float>();

  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor squared_weights;
  if(track_weights) {
        squared_weights = torch::full({P}, 0.0, float_opts);
  }
  else {
        squared_weights = torch::empty({0}, float_opts);
  }
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  auto background_contiguous = background.contiguous();
  auto means3D_contiguous = means3D.contiguous();
  auto sh_contiguous = sh.contiguous();
  auto colors_contiguous = colors.contiguous();
  auto opacity_contiguous = opacity.contiguous();
  auto scales_contiguous = scales.contiguous();
  auto rotations_contiguous = rotations.contiguous();
  auto cov3D_precomp_contiguous = cov3D_precomp.contiguous();
  auto viewmatrix_contiguous = viewmatrix.contiguous();
  auto projmatrix_contiguous = projmatrix.contiguous();
  auto campos_contiguous = campos.contiguous();
  auto out_color_contiguous = out_color.contiguous();
  auto radii_contiguous = radii.contiguous();
  
  int rendered = 0;
  if(P != 0)
  {
      int M = 0;
      if(sh.size(0) != 0)
      {
        M = sh.size(1);
      }

      rendered = CudaRasterizer::Rasterizer::forward(
        geomFunc,
        binningFunc,
        imgFunc,
        P, degree, M,
        background_contiguous.data<float>(),
        W, H,
        means3D_contiguous.data<float>(),
        sh_contiguous.data<float>(),
        colors_contiguous.data<float>(),
        opacity_contiguous.data<float>(),
        scales_contiguous.data<float>(),
        scale_modifier,
        rotations_contiguous.data<float>(),
        cov3D_precomp_contiguous.data<float>(),
        viewmatrix_contiguous.data<float>(),
        projmatrix_contiguous.data<float>(),
        campos_contiguous.data<float>(),
        tan_fovx,
        tan_fovy,
        prefiltered,
        out_color_contiguous.data<float>(),
        out_invdepthptr,
        antialiasing,
        radii_contiguous.data<int>(),
        debug,
        track_weights,
        squared_weights.data<float>());
  }
  return std::make_tuple(rendered, out_color_contiguous, radii_contiguous, geomBuffer, binningBuffer, imgBuffer, out_invdepth, squared_weights);
}

std::tuple<int, 
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
    torch::Tensor, torch::Tensor, torch::Tensor>
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
    const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  constexpr int NUM_CHANNELS = CudaRasterizer::NUM_CHANNELS;

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_color_grad = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_invdepth = torch::full({0, H, W}, 0.0, float_opts);
  torch::Tensor out_invdepth_grad = torch::full({0, H, W}, 0.0, float_opts);
  float* out_invdepth_ptr = nullptr;
  float* out_invdepth_grad_ptr = nullptr;

  out_invdepth = torch::full({1, H, W}, 0.0, float_opts).contiguous();
  out_invdepth_grad = torch::full({1, H, W}, 0.0, float_opts).contiguous();
  out_invdepth_ptr = out_invdepth.data<float>();
  out_invdepth_grad_ptr = out_invdepth_grad.data<float>();

  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBufferJvp = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBufferJvp = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBufferJvp);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBufferJvp);

  auto background_contiguous = background.contiguous();
  auto background_grad_contiguous = background_grad.contiguous();
  auto means3D_contiguous = means3D.contiguous();
  auto means3D_grad_contiguous = means3D_grad.contiguous();
  auto sh_contiguous = sh.contiguous();
  auto sh_grad_contiguous = sh_grad.contiguous();
  auto colors_contiguous = colors.contiguous();
  auto colors_grad_contiguous = colors_grad.contiguous();
  auto opacity_contiguous = opacity.contiguous();
  auto opacity_grad_contiguous = opacity_grad.contiguous();
  auto scales_contiguous = scales.contiguous();
  auto scales_grad_contiguous = scales_grad.contiguous();
  auto rotations_contiguous = rotations.contiguous();
  auto rotations_grad_contiguous = rotations_grad.contiguous();
  auto cov3D_precomp_contiguous = cov3D_precomp.contiguous();
  auto cov3D_precomp_grad_contiguous = cov3D_precomp_grad.contiguous();
  auto viewmatrix_contiguous = viewmatrix.contiguous();
  auto viewmatrix_grad_contiguous = viewmatrix_grad.contiguous();
  auto projmatrix_contiguous = projmatrix.contiguous();
  auto projmatrix_grad_contiguous = projmatrix_grad.contiguous();
  auto campos_contiguous = campos.contiguous();
  auto campos_grad_contiguous = campos_grad.contiguous();
  auto out_color_contiguous = out_color.contiguous();
  auto out_color_grad_contiguous = out_color_grad.contiguous();
  auto radii_contiguous = radii.contiguous();

  FloatGradArray<float> background_floatgrad(background_contiguous.data<float>(), background_grad_contiguous.data<float>());
  FloatGradArray<float> means3D_floatgrad(means3D_contiguous.data<float>(), means3D_grad_contiguous.data<float>());
  FloatGradArray<float> colors_floatgrad(colors_contiguous.data<float>(), colors_grad_contiguous.data<float>());
  FloatGradArray<float> opacity_floatgrad(opacity_contiguous.data<float>(), opacity_grad_contiguous.data<float>());
  FloatGradArray<float> scales_floatgrad(scales_contiguous.data<float>(), scales_grad_contiguous.data<float>());
  FloatGradArray<float> rotations_floatgrad(rotations_contiguous.data<float>(), rotations_grad_contiguous.data<float>());
  FloatGrad<float> scale_modifier_floatgrad(scale_modifier, scale_modifier_grad);
  FloatGradArray<float> cov3D_precomp_floatgrad(cov3D_precomp_contiguous.data<float>(), cov3D_precomp_grad_contiguous.data<float>());
  FloatGradArray<float> viewmatrix_floatgrad(viewmatrix_contiguous.data<float>(), viewmatrix_grad_contiguous.data<float>());
  FloatGradArray<float> projmatrix_floatgrad(projmatrix_contiguous.data<float>(), projmatrix_grad_contiguous.data<float>());
  FloatGrad<float> tan_fovx_floatgrad(tan_fovx, tan_fovx_grad);
  FloatGrad<float> tan_fovy_floatgrad(tan_fovy, tan_fovy_grad);
  FloatGradArray<float> sh_floatgrad(sh_contiguous.data<float>(), sh_grad_contiguous.data<float>());
  FloatGradArray<float> campos_floatgrad(campos_contiguous.data<float>(), campos_grad_contiguous.data<float>());
  FloatGradArray<float> out_color_floatgrad(out_color_contiguous.data<float>(), out_color_grad_contiguous.data<float>());
  FloatGradArray<float> out_invdepth_floatgrad(out_invdepth_ptr, out_invdepth_grad_ptr);
  
  int rendered = 0;
  if(P != 0)
  {
      int M = 0;
      if(sh.size(0) != 0)
      {
        M = sh.size(1);
      }

      rendered = CudaRasterizer::Rasterizer::forwardJvp(
        geomFunc,
        binningFunc,
        imgFunc,
        P, degree, M,
        background_floatgrad,
        W, H,
        means3D_floatgrad,
        sh_floatgrad,
        colors_floatgrad,
        opacity_floatgrad,
        scales_floatgrad,
        scale_modifier_floatgrad,
        rotations_floatgrad,
        cov3D_precomp_floatgrad,
        viewmatrix_floatgrad,
        projmatrix_floatgrad,
        campos_floatgrad,
        tan_fovx_floatgrad,
        tan_fovy_floatgrad,
        prefiltered,
        out_color_floatgrad,
        out_invdepth_floatgrad,
        antialiasing,
        radii_contiguous.data<int>(),
        debug);
  }
  return std::make_tuple(rendered, out_color_contiguous, radii_contiguous, geomBufferJvp, binningBuffer, imgBufferJvp, out_invdepth, out_color_grad_contiguous, out_invdepth_grad);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& radii,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
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
    const bool debug)
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {    
    M = sh.size(1);
  }

  constexpr int NUM_CHANNELS = CudaRasterizer::NUM_CHANNELS;

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dinvdepths = torch::zeros({0, 1}, means3D.options());
  
  float* dL_dinvdepthsptr = nullptr;
  float* dL_dout_invdepthptr = nullptr;
  if(dL_dout_invdepth.size(0) != 0)
  {
    dL_dinvdepths = torch::zeros({P, 1}, means3D.options()).contiguous();
    dL_dinvdepthsptr = dL_dinvdepths.data<float>();
    dL_dout_invdepthptr = dL_dout_invdepth.data<float>();
  }

  auto background_contiguous = background.contiguous();
  auto means3D_contiguous = means3D.contiguous();
  auto sh_contiguous = sh.contiguous();
  auto colors_contiguous = colors.contiguous();
  auto opacities_contiguous = opacities.contiguous();
  auto scales_contiguous = scales.contiguous();
  auto rotations_contiguous = rotations.contiguous();
  auto cov3D_precomp_contiguous = cov3D_precomp.contiguous();
  auto viewmatrix_contiguous = viewmatrix.contiguous();
  auto projmatrix_contiguous = projmatrix.contiguous();
  auto campos_contiguous = campos.contiguous();
  auto radii_contiguous = radii.contiguous();
  auto geomBuffer_contiguous = geomBuffer.contiguous();
  auto binningBuffer_contiguous = binningBuffer.contiguous();
  auto imageBuffer_contiguous = imageBuffer.contiguous();
  auto dL_dout_color_contiguous = dL_dout_color.contiguous();
  auto dL_dmeans2D_contiguous = dL_dmeans2D.contiguous();
  auto dL_dconic_contiguous = dL_dconic.contiguous();
  auto dL_dopacity_contiguous = dL_dopacity.contiguous();
  auto dL_dcolors_contiguous = dL_dcolors.contiguous();
  auto dL_dmeans3D_contiguous = dL_dmeans3D.contiguous();
  auto dL_dcov3D_contiguous = dL_dcov3D.contiguous();
  auto dL_dsh_contiguous = dL_dsh.contiguous();
  auto dL_dscales_contiguous = dL_dscales.contiguous();
  auto dL_drotations_contiguous = dL_drotations.contiguous();

  if(P != 0)
  {  
      CudaRasterizer::Rasterizer::backward(P, degree, M, R,
      background_contiguous.data<float>(),
      W, H, 
      means3D_contiguous.data<float>(),
      sh_contiguous.data<float>(),
      colors_contiguous.data<float>(),
      opacities_contiguous.data<float>(),
      scales.data_ptr<float>(),
      scale_modifier,
      rotations.data_ptr<float>(),
      cov3D_precomp_contiguous.data<float>(),
      viewmatrix_contiguous.data<float>(),
      projmatrix_contiguous.data<float>(),
      campos_contiguous.data<float>(),
      tan_fovx,
      tan_fovy,
      radii_contiguous.data<int>(),
      reinterpret_cast<char*>(geomBuffer_contiguous.data_ptr()),
      reinterpret_cast<char*>(binningBuffer_contiguous.data_ptr()),
      reinterpret_cast<char*>(imageBuffer_contiguous.data_ptr()),
      dL_dout_color_contiguous.data<float>(),
      dL_dout_invdepthptr,
      dL_dmeans2D_contiguous.data<float>(),
      dL_dconic_contiguous.data<float>(),  
      dL_dopacity_contiguous.data<float>(),
      dL_dcolors_contiguous.data<float>(),
      dL_dinvdepthsptr,
      dL_dmeans3D_contiguous.data<float>(),
      dL_dcov3D_contiguous.data<float>(),
      dL_dsh_contiguous.data<float>(),
      dL_dscales_contiguous.data<float>(),
      dL_drotations_contiguous.data<float>(),
      antialiasing,
      debug);
  }

  return std::make_tuple(dL_dmeans2D_contiguous, dL_dcolors_contiguous, dL_dopacity_contiguous, dL_dmeans3D_contiguous, dL_dcov3D_contiguous, dL_dsh_contiguous, dL_dscales_contiguous, dL_drotations_contiguous);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 PreprocessBackwardCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& radii,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
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
    const torch::Tensor& dL_dout_means2D,
    const torch::Tensor& dL_dout_dconic,
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
    const bool debug)
{
  const int P = means3D.size(0);
  
  int M = 0;
  if(sh.size(0) != 0)
  {    
    M = sh.size(1);
  }

  constexpr int NUM_CHANNELS = CudaRasterizer::NUM_CHANNELS;

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dinvdepths = torch::zeros({0, 1}, means3D.options());
  
  float* dL_dinvdepthsptr = nullptr;
  float* dL_dout_invdepthptr = nullptr;
  if(dL_dout_invdepth.size(0) != 0)
  {
    dL_dinvdepths = torch::zeros({P, 1}, means3D.options()).contiguous();
    dL_dinvdepthsptr = dL_dinvdepths.data<float>();
    dL_dout_invdepthptr = dL_dout_invdepth.data<float>();
  }

  auto background_contiguous = background.contiguous();
  auto means3D_contiguous = means3D.contiguous();
  auto sh_contiguous = sh.contiguous();
  auto colors_contiguous = colors.contiguous();
  auto opacities_contiguous = opacities.contiguous();
  auto scales_contiguous = scales.contiguous();
  auto rotations_contiguous = rotations.contiguous();
  auto cov3D_precomp_contiguous = cov3D_precomp.contiguous();
  auto viewmatrix_contiguous = viewmatrix.contiguous();
  auto projmatrix_contiguous = projmatrix.contiguous();
  auto campos_contiguous = campos.contiguous();
  auto radii_contiguous = radii.contiguous();
  auto geomBuffer_contiguous = geomBuffer.contiguous();
  auto binningBuffer_contiguous = binningBuffer.contiguous();
  auto imageBuffer_contiguous = imageBuffer.contiguous();
  auto dL_dout_means2D_contiguous = dL_dout_means2D.contiguous();
  auto dL_dout_dconic_contiguous = dL_dout_dconic.contiguous();
  auto dL_dout_colors_contiguous = dL_dout_colors.contiguous();
  auto dL_dconic_contiguous = dL_dconic.contiguous();
  auto dL_dopacity_contiguous = dL_dopacity.contiguous();
  auto dL_dmeans3D_contiguous = dL_dmeans3D.contiguous();
  auto dL_dcov3D_contiguous = dL_dcov3D.contiguous();
  auto dL_dsh_contiguous = dL_dsh.contiguous();
  auto dL_dscales_contiguous = dL_dscales.contiguous();
  auto dL_drotations_contiguous = dL_drotations.contiguous();

  if(P != 0)
  {  
      CudaRasterizer::Rasterizer::preprocessBackward(P, degree, M, R,
      background_contiguous.data<float>(),
      image_width, image_height,
      means3D_contiguous.data<float>(),
      sh_contiguous.data<float>(),
      colors_contiguous.data<float>(),
      opacities_contiguous.data<float>(),
      scales.data_ptr<float>(),
      scale_modifier,
      rotations.data_ptr<float>(),
      cov3D_precomp_contiguous.data<float>(),
      viewmatrix_contiguous.data<float>(),
      projmatrix_contiguous.data<float>(),
      campos_contiguous.data<float>(),
      tan_fovx,
      tan_fovy,
      radii_contiguous.data<int>(),
      reinterpret_cast<char*>(geomBuffer_contiguous.data_ptr()),
      reinterpret_cast<char*>(binningBuffer_contiguous.data_ptr()),
      reinterpret_cast<char*>(imageBuffer_contiguous.data_ptr()),
      dL_dout_means2D_contiguous.data<float>(),
      dL_dout_dconic_contiguous.data<float>(),
      dL_dout_invdepthptr,
      dL_dout_colors_contiguous.data<float>(),
      dL_dopacity_contiguous.data<float>(),
      dL_dmeans3D_contiguous.data<float>(),
      dL_dcov3D_contiguous.data<float>(),
      dL_dsh_contiguous.data<float>(),
      dL_dscales_contiguous.data<float>(),
      dL_drotations_contiguous.data<float>(),
      antialiasing,
      debug);
  }

  return std::make_tuple(dL_dopacity_contiguous, dL_dmeans3D_contiguous, dL_dcov3D_contiguous, dL_dsh_contiguous, dL_dscales_contiguous, dL_drotations_contiguous);
}

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
    const torch::Tensor& opacities,
    const torch::Tensor& opacities_grad,
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
    const bool debug)
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {    
    M = sh.size(1);
  }

  constexpr int NUM_CHANNELS = CudaRasterizer::NUM_CHANNELS;

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans3D_grad = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D_grad = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dcolors_grad = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dconic_grad = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dopacity_grad = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dcov3D_grad = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dsh_grad = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dscales_grad = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_drotations_grad = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dinvdepths = torch::zeros({0, 1}, means3D.options());
  torch::Tensor dL_dinvdepths_grad = torch::zeros({0, 1}, means3D.options());
  
  float* dL_dinvdepthsptr = nullptr;
  float* dL_dinvdepths_grad_ptr = nullptr;
  float* dL_dout_invdepthptr = nullptr;
  float* dL_dout_invdepth_grad_ptr = nullptr;
  if(dL_dout_invdepth.size(0) != 0)
  {
    dL_dinvdepths = torch::zeros({P, 1}, means3D.options()).contiguous();
    dL_dinvdepths_grad = torch::zeros({P, 1}, means3D.options()).contiguous();
    dL_dinvdepthsptr = dL_dinvdepths.data<float>();
    dL_dinvdepths_grad_ptr = dL_dinvdepths_grad.data<float>();
    dL_dout_invdepthptr = dL_dout_invdepth.data<float>();
    dL_dout_invdepth_grad_ptr = dL_dout_invdepth_grad.data<float>();
  }

  auto background_contiguous = background.contiguous();
  auto background_grad_contiguous = background_grad.contiguous();
  auto means3D_contiguous = means3D.contiguous();
  auto means3D_grad_contiguous = means3D_grad.contiguous();
  auto sh_contiguous = sh.contiguous();
  auto sh_grad_contiguous = sh_grad.contiguous();
  auto colors_contiguous = colors.contiguous();
  auto colors_grad_contiguous = colors_grad.contiguous();
  auto opacities_contiguous = opacities.contiguous();
  auto opacities_grad_contiguous = opacities_grad.contiguous();
  auto scales_contiguous = scales.contiguous();
  auto scales_grad_contiguous = scales_grad.contiguous();
  auto rotations_contiguous = rotations.contiguous();
  auto rotations_grad_contiguous = rotations_grad.contiguous();
  auto cov3D_precomp_contiguous = cov3D_precomp.contiguous();
  auto cov3D_precomp_grad_contiguous = cov3D_precomp_grad.contiguous();
  auto viewmatrix_contiguous = viewmatrix.contiguous();
  auto viewmatrix_grad_contiguous = viewmatrix_grad.contiguous();
  auto projmatrix_contiguous = projmatrix.contiguous();
  auto projmatrix_grad_contiguous = projmatrix_grad.contiguous();
  auto campos_contiguous = campos.contiguous();
  auto campos_grad_contiguous = campos_grad.contiguous();
  auto radii_contiguous = radii.contiguous();
  auto geomBuffer_contiguous = geomBuffer.contiguous();
  auto binningBuffer_contiguous = binningBuffer.contiguous();
  auto imageBuffer_contiguous = imageBuffer.contiguous();
  auto dL_dout_color_contiguous = dL_dout_color.contiguous();
  auto dL_dout_color_grad_contiguous = dL_dout_color_grad.contiguous();
  auto dL_dmeans2D_contiguous = dL_dmeans2D.contiguous();
  auto dL_dmeans2D_grad_contiguous = dL_dmeans2D_grad.contiguous();
  auto dL_dconic_contiguous = dL_dconic.contiguous();
  auto dL_dconic_grad_contiguous = dL_dconic_grad.contiguous();
  auto dL_dopacity_contiguous = dL_dopacity.contiguous();
  auto dL_dopacity_grad_contiguous = dL_dopacity_grad.contiguous();
  auto dL_dcolors_contiguous = dL_dcolors.contiguous();
  auto dL_dcolors_grad_contiguous = dL_dcolors_grad.contiguous();

  FloatGradArray<float> background_floatgrad(background_contiguous.data<float>(), background_grad_contiguous.data<float>());
  FloatGradArray<float> means3D_floatgrad(means3D_contiguous.data<float>(), means3D_grad_contiguous.data<float>());
  FloatGradArray<float> sh_floatgrad(sh_contiguous.data<float>(), sh_grad_contiguous.data<float>());
  FloatGradArray<float> colors_floatgrad(colors_contiguous.data<float>(), colors_grad_contiguous.data<float>());
  FloatGradArray<float> opacity_floatgrad(opacities_contiguous.data<float>(), opacities_grad_contiguous.data<float>());
  FloatGradArray<float> scales_floatgrad(scales_contiguous.data<float>(), scales_grad_contiguous.data<float>());
  FloatGrad<float> scale_modifier_floatgrad(scale_modifier, scale_modifier_grad);
  FloatGradArray<float> rotations_floatgrad(rotations_contiguous.data<float>(), rotations_grad_contiguous.data<float>());
  FloatGradArray<float> cov3D_precomp_floatgrad(cov3D_precomp_contiguous.data<float>(), cov3D_precomp_grad_contiguous.data<float>());
  FloatGradArray<float> viewmatrix_floatgrad(viewmatrix_contiguous.data<float>(), viewmatrix_grad_contiguous.data<float>());
  FloatGradArray<float> projmatrix_floatgrad(projmatrix_contiguous.data<float>(), projmatrix_grad_contiguous.data<float>());
  FloatGradArray<float> campos_floatgrad(campos_contiguous.data<float>(), campos_grad_contiguous.data<float>());
  FloatGrad<float> tan_fovx_floatgrad(tan_fovx, tan_fovx_grad);
  FloatGrad<float> tan_fovy_floatgrad(tan_fovy, tan_fovy_grad);
  FloatGradArray<float> dL_dout_color_floatgrad(dL_dout_color_contiguous.data<float>(), dL_dout_color_grad_contiguous.data<float>());
  FloatGradArray<float> dL_dout_invdepth_floatgrad(dL_dout_invdepthptr, dL_dout_invdepth_grad_ptr);
  FloatGradArray<float> dL_dmeans2D_floatgrad(dL_dmeans2D.data<float>(), dL_dmeans2D_grad.data<float>());
  FloatGradArray<float> dL_dconic_floatgrad(dL_dconic.data<float>(), dL_dconic_grad.data<float>());
  FloatGradArray<float> dL_dopacity_floatgrad(dL_dopacity.data<float>(), dL_dopacity_grad.data<float>());
  FloatGradArray<float> dL_dcolors_floatgrad(dL_dcolors.data<float>(), dL_dcolors_grad.data<float>());
  FloatGradArray<float> dL_dinvdepths_floatgrad(dL_dinvdepthsptr, dL_dinvdepths_grad_ptr);
  FloatGradArray<float> dL_dmeans3D_floatgrad(dL_dmeans3D.data<float>(), dL_dmeans3D_grad.data<float>());
  FloatGradArray<float> dL_dcov3D_floatgrad(dL_dcov3D.data<float>(), dL_dcov3D_grad.data<float>());
  FloatGradArray<float> dL_dsh_floatgrad(dL_dsh.data<float>(), dL_dsh_grad.data<float>());
  FloatGradArray<float> dL_dscales_floatgrad(dL_dscales.data<float>(), dL_dscales_grad.data<float>());
  FloatGradArray<float> dL_drotations_floatgrad(dL_drotations.data<float>(), dL_drotations_grad.data<float>());


  if(P != 0)
  {  
      CudaRasterizer::Rasterizer::backwardJvp(P, degree, M, R,
        background_floatgrad,
        W, H, 
        means3D_floatgrad,
        sh_floatgrad,
        colors_floatgrad,
        opacity_floatgrad,
        scales_floatgrad,
        scale_modifier,
        rotations_floatgrad,
        cov3D_precomp_floatgrad,
        viewmatrix_floatgrad,
        projmatrix_floatgrad,
        campos_floatgrad,
        tan_fovx_floatgrad,
        tan_fovy_floatgrad,
        radii_contiguous.data<int>(),
        reinterpret_cast<char*>(geomBuffer_contiguous.data_ptr()),
        reinterpret_cast<char*>(binningBuffer_contiguous.data_ptr()),
        reinterpret_cast<char*>(imageBuffer_contiguous.data_ptr()),
        dL_dout_color_floatgrad,
        dL_dout_invdepth_floatgrad,
        dL_dmeans2D_floatgrad,
        dL_dconic_floatgrad,
        dL_dopacity_floatgrad,
        dL_dcolors_floatgrad,
        dL_dinvdepths_floatgrad,
        dL_dmeans3D_floatgrad,
        dL_dcov3D_floatgrad,
        dL_dsh_floatgrad,
        dL_dscales_floatgrad,
        dL_drotations_floatgrad,
        antialiasing,
        debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, dL_dmeans2D_grad, dL_dcolors_grad, dL_dopacity_grad, dL_dmeans3D_grad, dL_dcov3D_grad, dL_dsh_grad, dL_dscales_grad, dL_drotations_grad);
}

torch::Tensor markVisible(
        torch::Tensor& means3D,
        torch::Tensor& viewmatrix,
        torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
    CudaRasterizer::Rasterizer::markVisible(P,
        means3D.contiguous().data<float>(),
        viewmatrix.contiguous().data<float>(),
        projmatrix.contiguous().data<float>(),
        present.contiguous().data<bool>());
  }
  
  return present;
}

std::tuple<torch::Tensor, torch::Tensor> ComputeRelocationCUDA(
	torch::Tensor& opacity_old,
	torch::Tensor& scale_old,
	torch::Tensor& N,
	torch::Tensor& binoms,
	const int n_max)
{
	const int P = opacity_old.size(0);
  
	torch::Tensor final_opacity = torch::full({P}, 0, opacity_old.options().dtype(torch::kFloat32));
	torch::Tensor final_scale = torch::full({3 * P}, 0, scale_old.options().dtype(torch::kFloat32));

	if(P != 0)
	{
		UTILS::ComputeRelocation(P,
			opacity_old.contiguous().data<float>(),
			scale_old.contiguous().data<float>(),
			N.contiguous().data<int>(),
			binoms.contiguous().data<float>(),
			n_max,
			final_opacity.contiguous().data<float>(),
			final_scale.contiguous().data<float>());
	}

	return std::make_tuple(final_opacity, final_scale);

}

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
        const float quat_norm_tr)
{

    const int P = xyz_params.size(0);

    int M = 0;
    if(shs_params.size(0) != 0)
    {    
      M = shs_params.size(1);
    }

    auto xyz_params_contiguous = xyz_params.contiguous();
    auto scaling_params_contiguous = scaling_params.contiguous();
    auto quat_params_contiguous = quat_params.contiguous();
    auto opacity_params_contiguous = opacity_params.contiguous();
    auto shs_params_contiguous = shs_params.contiguous();

    if(!xyz_params_step.is_contiguous()) { xyz_params_step = xyz_params_step.contiguous(); }
    if(!scaling_params_step.is_contiguous()) { scaling_params_step = scaling_params_step.contiguous(); }
    if(!quat_params_step.is_contiguous()) { quat_params_step = quat_params_step.contiguous(); }
    if(!opacity_params_step.is_contiguous()) { opacity_params_step = opacity_params_step.contiguous(); }
    if(!shs_params_step.is_contiguous()) { shs_params_step = shs_params_step.contiguous(); }

    if(P != 0)
    {
        TRUST_REGION::ComputeTrustRegionStep(
            P, M,
            trust_radius,
            min_mass_scaling,
            max_mass_scaling,
            quat_norm_tr,
            xyz_params_contiguous.data<float>(),
            (const glm::vec3*) scaling_params_contiguous.data<float>(),
            scale_modifier,
            (const glm::vec4*) quat_params_contiguous.data<float>(),
            opacity_params_contiguous.data<float>(),
            shs_params_contiguous.data<float>(),
            xyz_params_step.data<float>(),
            (glm::vec3*) scaling_params_step.data<float>(),
            (glm::vec4*) quat_params_step.data<float>(),
            opacity_params_step.data<float>(),
            shs_params_step.data<float>());
    }

    return std::make_tuple(xyz_params_step, scaling_params_step, quat_params_step, opacity_params_step, shs_params_step);
}
