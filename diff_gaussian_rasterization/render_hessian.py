#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
import torch.autograd.forward_ad as fwAD
from . import _C
import time

from diff_gaussian_rasterization.utils import has_tangent, get_primal, get_tangent, unpack_dual

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):

    means3D_primal, means3D_tangent = unpack_dual(means3D)
    means2D_primal, means2D_tangent = unpack_dual(means2D)
    sh_primal, sh_tangent = unpack_dual(sh)
    colors_precomp_primal, colors_precomp_tangent = unpack_dual(colors_precomp)
    opacities_primal, opacities_tangent = unpack_dual(opacities)
    scales_primal, scales_tangent = unpack_dual(scales)
    rotations_primal, rotations_tangent = unpack_dual(rotations)
    cov3Ds_precomp_primal, cov3Ds_precomp_tangent = unpack_dual(cov3Ds_precomp)
    raster_settings_primal = GaussianRasterizationSettingsHessian(
            image_height=raster_settings.image_height,
            image_width=raster_settings.image_width,
            tanfovx=get_primal(raster_settings.tanfovx),
            tanfovy=get_primal(raster_settings.tanfovy),
            bg=get_primal(raster_settings.bg),
            scale_modifier=get_primal(raster_settings.scale_modifier),
            viewmatrix=get_primal(raster_settings.viewmatrix),
            projmatrix=get_primal(raster_settings.projmatrix),
            sh_degree=raster_settings.sh_degree,
            campos=get_primal(raster_settings.campos),
            prefiltered=raster_settings.prefiltered,
            debug=raster_settings.debug,
            antialiasing=raster_settings.antialiasing)
    raster_settings_tangent = GaussianRasterizationSettingsHessian(
            image_height=raster_settings.image_height,
            image_width=raster_settings.image_width,
            tanfovx=get_tangent(raster_settings.tanfovx),
            tanfovy=get_tangent(raster_settings.tanfovy),
            bg=get_tangent(raster_settings.bg),
            scale_modifier=get_tangent(raster_settings.scale_modifier),
            viewmatrix=get_tangent(raster_settings.viewmatrix),
            projmatrix=get_tangent(raster_settings.projmatrix),
            sh_degree=raster_settings.sh_degree,
            campos=get_tangent(raster_settings.campos),
            prefiltered=raster_settings.prefiltered,
            debug=raster_settings.debug,
            antialiasing=raster_settings.antialiasing)

    color_primal, radii, invdepths_primal, color_tangent, invdepths_tangent = _RasterizeGaussians.apply(
        means3D_primal,
        means3D_tangent,
        means2D_primal,
        means2D_tangent,
        sh_primal,
        sh_tangent,
        colors_precomp_primal,
        colors_precomp_tangent,
        opacities_primal,
        opacities_tangent,
        scales_primal,
        scales_tangent,
        rotations_primal,
        rotations_tangent,
        cov3Ds_precomp_primal,
        cov3Ds_precomp_tangent,
        raster_settings_primal,
        raster_settings_tangent,)

    color_dual = fwAD.make_dual(color_primal, color_tangent)
    invdepths_dual = fwAD.make_dual(invdepths_primal, invdepths_tangent)

    return color_dual, radii, invdepths_dual


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means3D_tangent,
        means2D,
        means2D_tangent,
        sh,
        sh_tangent,
        colors_precomp,
        colors_precomp_tangent,
        opacities,
        opacities_tangent,
        scales,
        scales_tangent,
        rotations,
        rotations_tangent,
        cov3Ds_precomp,
        cov3Ds_precomp_tangent,
        raster_settings,
        raster_settings_tangent,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            raster_settings_tangent.bg,
            means3D,
            means3D_tangent,
            colors_precomp,
            colors_precomp_tangent,
            opacities,
            opacities_tangent,
            scales,
            scales_tangent,
            rotations,
            rotations_tangent,
            raster_settings.scale_modifier,
            raster_settings_tangent.scale_modifier,
            cov3Ds_precomp,
            cov3Ds_precomp_tangent,
            raster_settings.viewmatrix,
            raster_settings_tangent.viewmatrix,
            raster_settings.projmatrix,
            raster_settings_tangent.projmatrix,
            raster_settings.tanfovx,
            raster_settings_tangent.tanfovx,
            raster_settings.tanfovy,
            raster_settings_tangent.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            sh_tangent,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings_tangent.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths, color_grad, invdepths_grad = _C.rasterize_gaussians_jvp(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.raster_settings_tangent = raster_settings_tangent
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp,
                              colors_precomp_tangent,
                              means3D, 
                              means3D_tangent,
                              scales, 
                              scales_tangent,
                              rotations, 
                              rotations_tangent,
                              cov3Ds_precomp, 
                              cov3Ds_precomp_tangent,
                              radii, 
                              sh, 
                              sh_tangent,
                              opacities, 
                              opacities_tangent,
                              geomBuffer, 
                              binningBuffer, 
                              imgBuffer)

        return color, radii, invdepths, color_grad, invdepths_grad

    @staticmethod
    def backward(ctx, hvp_out_color, _, hvp_out_depth, grad_out_color, grad_out_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        raster_settings_tangent = ctx.raster_settings_tangent
        (colors_precomp, colors_precomp_tangent,
         means3D, means3D_tangent,
         scales, scales_tangent,
         rotations, rotations_tangent,
         cov3Ds_precomp, cov3Ds_precomp_tangent,
         radii,
         sh, sh_tangent,
         opacities, opacities_tangent,
         geomBuffer, binningBuffer, imgBuffer) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                raster_settings_tangent.bg,
                means3D, 
                means3D_tangent,
                radii, 
                colors_precomp, 
                colors_precomp_tangent,
                opacities,
                opacities_tangent,
                scales, 
                scales_tangent,
                rotations, 
                rotations_tangent,
                raster_settings.scale_modifier, 
                raster_settings_tangent.scale_modifier,
                cov3Ds_precomp, 
                cov3Ds_precomp_tangent,
                raster_settings.viewmatrix, 
                raster_settings_tangent.viewmatrix,
                raster_settings.projmatrix, 
                raster_settings_tangent.projmatrix,
                raster_settings.tanfovx, 
                raster_settings_tangent.tanfovx,
                raster_settings.tanfovy, 
                raster_settings_tangent.tanfovy,
                grad_out_color,
                hvp_out_color,
                grad_out_depth, 
                hvp_out_depth,
                sh, 
                sh_tangent,
                raster_settings.sh_degree, 
                raster_settings.campos,
                raster_settings_tangent.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.antialiasing,
                raster_settings.debug)

        # torch.set_printoptions(precision=10)
        # print(f"in render_hessian, menas3D = {means3D}, grad_out_color = {grad_out_color}")

        # Compute gradients for relevant tensors by invoking backward method
        (grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations,
         hvp_means2D, hvp_colors_precomp, hvp_opacities, hvp_means3D, hvp_cov3Ds_precomp, hvp_sh, hvp_scales, hvp_rotations) = _C.rasterize_gaussians_backward_jvp(*args)        

        # Restructure args as C++ method expects them
        args1 = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                opacities,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_out_depth, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.antialiasing,
                raster_settings.debug)
        (grad_means2D1, grad_colors_precomp1, grad_opacities1, grad_means3D1, grad_cov3Ds_precomp1, grad_sh1, grad_scales1, grad_rotations1) = _C.rasterize_gaussians_backward(*args1)

        # Important: the ordering of the output should be hvp, grad
        grads = (
            hvp_means3D,
            grad_means3D,
            hvp_means2D,
            grad_means2D,
            hvp_sh,
            grad_sh,
            hvp_colors_precomp,
            grad_colors_precomp,
            hvp_opacities,
            grad_opacities,
            hvp_scales,
            grad_scales,
            hvp_rotations,
            grad_rotations,
            hvp_cov3Ds_precomp,
            grad_cov3Ds_precomp,
            None,
            None,
        )

        return grads

class GaussianRasterizationSettingsHessian(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool

class GaussianRasterizerHessian:
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

