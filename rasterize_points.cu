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
  torch::Tensor out_invdepth = torch::full({0, H, W}, 0.0, float_opts);
  float* out_invdepthptr = nullptr;

  out_invdepth = torch::full({1, H, W}, 0.0, float_opts).contiguous();
  out_invdepthptr = out_invdepth.data<float>();

  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
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
        background.contiguous().data<float>(),
        W, H,
        means3D.contiguous().data<float>(),
        sh.contiguous().data_ptr<float>(),
        colors.contiguous().data<float>(), 
        opacity.contiguous().data<float>(), 
        scales.contiguous().data_ptr<float>(),
        scale_modifier,
        rotations.contiguous().data_ptr<float>(),
        cov3D_precomp.contiguous().data<float>(), 
        viewmatrix.contiguous().data<float>(), 
        projmatrix.contiguous().data<float>(),
        campos.contiguous().data<float>(),
        tan_fovx,
        tan_fovy,
        prefiltered,
        out_color.contiguous().data<float>(),
        out_invdepthptr,
        antialiasing,
        radii.contiguous().data<int>(),
        debug);
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_invdepth);
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

  FloatGradArray<float> background_floatgrad(background.contiguous().data<float>(), background_grad.contiguous().data<float>());
  FloatGradArray<float> means3D_floatgrad(means3D.contiguous().data<float>(), means3D_grad.contiguous().data<float>());
  FloatGradArray<float> colors_floatgrad(colors.contiguous().data<float>(), colors_grad.contiguous().data<float>());
  FloatGradArray<float> opacity_floatgrad(opacity.contiguous().data<float>(), opacity_grad.contiguous().data<float>());
  FloatGradArray<float> scales_floatgrad(scales.contiguous().data<float>(), scales_grad.contiguous().data<float>());
  FloatGradArray<float> rotations_floatgrad(rotations.contiguous().data<float>(), rotations_grad.contiguous().data<float>());
  FloatGrad<float> scale_modifier_floatgrad(scale_modifier, scale_modifier_grad);
  FloatGradArray<float> cov3D_precomp_floatgrad(cov3D_precomp.contiguous().data<float>(), cov3D_precomp_grad.contiguous().data<float>());
  FloatGradArray<float> viewmatrix_floatgrad(viewmatrix.contiguous().data<float>(), viewmatrix_grad.contiguous().data<float>());
  FloatGradArray<float> projmatrix_floatgrad(projmatrix.contiguous().data<float>(), projmatrix_grad.contiguous().data<float>());
  FloatGrad<float> tan_fovx_floatgrad(tan_fovx, tan_fovx_grad);
  FloatGrad<float> tan_fovy_floatgrad(tan_fovy, tan_fovy_grad);
  FloatGradArray<float> sh_floatgrad(sh.contiguous().data<float>(), sh_grad.contiguous().data<float>());
  FloatGradArray<float> campos_floatgrad(campos.contiguous().data<float>(), campos_grad.contiguous().data<float>());
  FloatGradArray<float> out_color_floatgrad(out_color.contiguous().data<float>(), out_color_grad.contiguous().data<float>());
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
        radii.contiguous().data<int>(),
        debug);
  }
  return std::make_tuple(rendered, out_color, radii, geomBufferJvp, binningBuffer, imgBufferJvp, out_invdepth, out_color_grad, out_invdepth_grad);
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
    dL_dinvdepths = torch::zeros({P, 1}, means3D.options());
    dL_dinvdepths = dL_dinvdepths.contiguous();
    dL_dinvdepthsptr = dL_dinvdepths.data<float>();
    dL_dout_invdepthptr = dL_dout_invdepth.data<float>();
  }

  if(P != 0)
  {  
      CudaRasterizer::Rasterizer::backward(P, degree, M, R,
      background.contiguous().data<float>(),
      W, H, 
      means3D.contiguous().data<float>(),
      sh.contiguous().data<float>(),
      colors.contiguous().data<float>(),
      opacities.contiguous().data<float>(),
      scales.data_ptr<float>(),
      scale_modifier,
      rotations.data_ptr<float>(),
      cov3D_precomp.contiguous().data<float>(),
      viewmatrix.contiguous().data<float>(),
      projmatrix.contiguous().data<float>(),
      campos.contiguous().data<float>(),
      tan_fovx,
      tan_fovy,
      radii.contiguous().data<int>(),
      reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
      reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
      reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
      dL_dout_color.contiguous().data<float>(),
      dL_dout_invdepthptr,
      dL_dmeans2D.contiguous().data<float>(),
      dL_dconic.contiguous().data<float>(),  
      dL_dopacity.contiguous().data<float>(),
      dL_dcolors.contiguous().data<float>(),
      dL_dinvdepthsptr,
      dL_dmeans3D.contiguous().data<float>(),
      dL_dcov3D.contiguous().data<float>(),
      dL_dsh.contiguous().data<float>(),
      dL_dscales.contiguous().data<float>(),
      dL_drotations.contiguous().data<float>(),
      antialiasing,
      debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
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
