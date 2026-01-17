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
from . import _C
import time

from diff_gaussian_rasterization.utils import has_tangent, get_tangent

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
    tensor_args = (means3D, means2D, sh, colors_precomp, opacities, scales, rotations, cov3Ds_precomp, raster_settings.bg, raster_settings.viewmatrix, raster_settings.projmatrix, raster_settings.campos)

    jvp = any(has_tangent(x) for x in tensor_args)

    if not jvp:
        return _RasterizeGaussians.apply(
            means3D,
            means2D,
            sh,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            raster_settings,
            jvp,
            None, None, None, None, None, None, None, None, None,
        )
    else:

        means3D_tangent = get_tangent(means3D)
        means2D_tangent = get_tangent(means2D)
        sh_tangent = get_tangent(sh)
        colors_precomp_tangent = get_tangent(colors_precomp)
        opacities_tangent = get_tangent(opacities)
        scales_tangent = get_tangent(scales)
        rotations_tangent = get_tangent(rotations)
        cov3Ds_precomp_tangent = get_tangent(cov3Ds_precomp)
        raster_settings_tangent = GaussianRasterizationSettings(
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
                antialiasing=raster_settings.antialiasing,
                track_weights=raster_settings.track_weights,)

        res = _RasterizeGaussians.apply(
            means3D,
            means2D,
            sh,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            raster_settings,
            jvp,
            means3D_tangent,
            means2D_tangent,
            sh_tangent,
            colors_precomp_tangent,
            opacities_tangent,
            scales_tangent,
            rotations_tangent,
            cov3Ds_precomp_tangent,
            raster_settings_tangent,)

        return res


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        jvp,
        means3D_tangent,
        means2D_tangent,
        sh_tangent,
        colors_precomp_tangent,
        opacities_tangent,
        scales_tangent,
        rotations_tangent,
        cov3Ds_precomp_tangent,
        raster_settings_tangent,
    ):

        assert not (jvp and raster_settings.track_weights), "JVP mode not supported with track_weights=True"

        if not jvp:

            # Restructure arguments the way that the C++ lib expects them
            args = (
                raster_settings.bg, 
                means3D,
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
                raster_settings.image_height,
                raster_settings.image_width,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                raster_settings.prefiltered,
                raster_settings.antialiasing,
                raster_settings.debug,
                raster_settings.track_weights,
            )

            # Invoke C++/CUDA rasterizer
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths, squared_weights = _C.rasterize_gaussians(*args)

        else:
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
            squared_weights = None

            ctx.save_for_forward(color_grad, invdepths_grad)


        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer)

        return color, radii, invdepths, squared_weights

    @staticmethod
    def backward(ctx, grad_out_color, _grad_radii, grad_out_depth, _grad_squared_weights):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
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

        # Compute gradients for relevant tensors by invoking backward method
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)        

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            None, None, None, None, None, None, None, None, None, None
        )

        return grads

    @staticmethod
    def jvp(
        ctx,
        grad_means3D,
        grad_means2D,
        grad_sh,
        grad_colors_precomp,
        grad_opacities,
        grad_scales,
        grad_rotations,
        grad_cov3Ds_precomp,
        grad_raster_settings,
        grad_jvp,
        grad_means3D_tangent,
        grad_means2D_tangent,
        grad_sh_tangent,
        grad_colors_precomp_tangent,
        grad_opacities_tangent,
        grad_scales_tangent,
        grad_rotations_tangent,
        grad_cov3Ds_precomp_tangent,
        grad_raster_settings_tangent,):

        color_grad, invdepths_grad = ctx.saved_tensors
        return color_grad, None, invdepths_grad


class GaussianRasterizationSettings(NamedTuple):
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
    track_weights: bool = False

class GaussianRasterizer(nn.Module):
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

def compute_relocation(opacity_old, scale_old, N, binoms, n_max):
    new_opacity, new_scale = _C.compute_relocation(opacity_old, scale_old, N.int(), binoms, n_max)
    return new_opacity, new_scale 

def compute_trust_region_step(xyz_params, scaling_params, quat_params, opacity_params, shs_params,
                              xyz_params_step, scaling_params_step, quat_params_step, opacity_params_step, shs_params_step,
                              trust_radius, opacity_trust_radius,
                              min_mass_scaling, max_mass_scaling, 
                              scale_modifier, quat_norm_tr):
    xyz_params_step, scaling_params_step, quat_params_step, opacity_params_step, shs_params_step = _C.compute_trust_region_step(
            xyz_params, scaling_params, quat_params, opacity_params, shs_params,
            xyz_params_step, scaling_params_step, quat_params_step, opacity_params_step, shs_params_step,
            trust_radius, opacity_trust_radius,
            min_mass_scaling, max_mass_scaling, scale_modifier, quat_norm_tr)

    return xyz_params_step, scaling_params_step, quat_params_step, opacity_params_step, shs_params_step
