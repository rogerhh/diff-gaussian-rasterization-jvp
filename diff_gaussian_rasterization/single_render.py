from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C
import time


from diff_gaussian_rasterization.utils import has_tangent, get_tangent

def batch_rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    batch_raster_settings,
):
    tensor_args = (means3D, means2D, sh, colors_precomp, opacities, scales, rotations, cov3Ds_precomp, batch_raster_settings.bg, batch_raster_settings.viewmatrices, batch_raster_settings.projmatrices, batch_raster_settings.camposes)

    jvp = any(has_tangent(x) for x in tensor_args)

    if not jvp:
        return _BatchRasterizeGaussians.apply(
            means3D,
            means2D,
            sh,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            batch_raster_settings,
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
        batch_raster_settings_tangent = BatchGaussianRasterizationSettings(
                batch_size=batch_raster_settings.batch_size,
                image_heights=batch_raster_settings.image_heights,
                image_widths=batch_raster_settings.image_widths,
                tanfovxs=get_tangent(batch_raster_settings.tanfovxs),
                tanfovys=get_tangent(batch_raster_settings.tanfovys),
                bg=get_tangent(batch_raster_settings.bg),
                scale_modifier=get_tangent(batch_raster_settings.scale_modifier),
                viewmatrices=get_tangent(batch_raster_settings.viewmatrices),
                projmatrices=get_tangent(batch_raster_settings.projmatrices),
                sh_degree=batch_raster_settings.sh_degree,
                camposes=get_tangent(batch_raster_settings.camposes),
                prefiltered=batch_raster_settings.prefiltered,
                debug=batch_raster_settings.debug,
                antialiasing=batch_raster_settings.antialiasing,
                track_weights=batch_raster_settings.track_weights,)

        res = _BatchRasterizeGaussians.apply(
            means3D,
            means2D,
            sh,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            batch_raster_settings,
            jvp,
            means3D_tangent,
            means2D_tangent,
            sh_tangent,
            colors_precomp_tangent,
            opacities_tangent,
            scales_tangent,
            rotations_tangent,
            cov3Ds_precomp_tangent,
            batch_raster_settings_tangent,)

        return res

class _BatchRasterizeGaussians(torch.autograd.Function):
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
        batch_raster_settings,
        jvp,
        means3D_tangent,
        means2D_tangent,
        sh_tangent,
        colors_precomp_tangent,
        opacities_tangent,
        scales_tangent,
        rotations_tangent,
        cov3Ds_precomp_tangent,
        batch_raster_settings_tangent,
    ):

        B = batch_raster_settings.batch_size
        P = means3D.shape[0]

        assert B == 1, "Currently only batch size of 1 is supported in batch rasterization."

        max_height = max(batch_raster_settings.image_heights)
        max_width = max(batch_raster_settings.image_widths)

        num_rendereds = []
        geomBuffers = []
        binningBuffers = []
        imgBuffers = []

        assert not (batch_raster_settings.track_weights and jvp), "Tracking weights is not supported in JVP mode."

        if batch_raster_settings.track_weights:
            squared_weightss = {"means3D": torch.zeros_like(means3D), 
                                "sh": torch.zeros_like(sh),
                                "opacities": torch.zeros_like(opacities),
                                "scales": torch.zeros_like(scales),
                                "rotations": torch.zeros_like(rotations)}

            # Run one partial backward pass for weight tracking
            NUM_CHANNELS = 3
            grad_out_means2D = torch.ones((P, 3), device=means3D.device, dtype=means3D.dtype)
            grad_out_conic = torch.ones((P, 2, 2), device=means3D.device, dtype=means3D.dtype)
            grad_out_invdepth = torch.ones((P, 1), device=means3D.device, dtype=means3D.dtype)
            grad_out_colors = torch.ones((P, NUM_CHANNELS), device=means3D.device, dtype=means3D.dtype)

        else:
            squared_weightss = None

        if not jvp:

            for i in range(B):
                # Restructure arguments the way that the C++ lib expects them
                args = (
                    batch_raster_settings.bg, 
                    means3D,
                    colors_precomp,
                    opacities,
                    scales,
                    rotations,
                    batch_raster_settings.scale_modifier,
                    cov3Ds_precomp,
                    batch_raster_settings.viewmatrices[i],
                    batch_raster_settings.projmatrices[i],
                    batch_raster_settings.tanfovxs[i],
                    batch_raster_settings.tanfovys[i],
                    batch_raster_settings.image_heights[i],
                    batch_raster_settings.image_widths[i],
                    sh,
                    batch_raster_settings.sh_degree,
                    batch_raster_settings.camposes[i],
                    batch_raster_settings.prefiltered,
                    batch_raster_settings.antialiasing,
                    batch_raster_settings.debug,
                    batch_raster_settings.track_weights
                )

                # Invoke C++/CUDA rasterizer
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths, squared_weights = _C.rasterize_gaussians(*args)

                num_rendereds.append(num_rendered)
                colors = color.unsqueeze(0)
                radiis = radii.unsqueeze(0)
                invdepthss = invdepths.unsqueeze(0)
                geomBuffers.append(geomBuffer)
                binningBuffers.append(binningBuffer)
                imgBuffers.append(imgBuffer)

                if batch_raster_settings.track_weights:

                    # DEBUG: If tracking weights, only run backward preprocessing
                    # Restructure args as C++ method expects them
                    args = (batch_raster_settings.bg,
                            means3D, 
                            radiis[i], 
                            colors_precomp, 
                            opacities,
                            scales, 
                            rotations, 
                            batch_raster_settings.scale_modifier, 
                            cov3Ds_precomp, 
                            batch_raster_settings.viewmatrices[i], 
                            batch_raster_settings.projmatrices[i], 
                            batch_raster_settings.tanfovxs[i], 
                            batch_raster_settings.tanfovys[i], 
                            batch_raster_settings.image_heights[i],
                            batch_raster_settings.image_widths[i],
                            grad_out_means2D,
                            grad_out_conic,
                            grad_out_invdepth,
                            grad_out_colors,
                            sh, 
                            batch_raster_settings.sh_degree, 
                            batch_raster_settings.camposes[i],
                            geomBuffers[i],
                            num_rendereds[i],
                            binningBuffers[i],
                            imgBuffers[i],
                            batch_raster_settings.antialiasing,
                            batch_raster_settings.debug,)

                    grad_opacities_i, grad_means3D_i, grad_cov3Ds_precomp_i, grad_sh_i, grad_scales_i, grad_rotations_i = _C.preprocess_backward(*args)        

                    squared_weightss["means3D"] += squared_weights[:,None] * (grad_means3D_i ** 2)
                    squared_weightss["sh"] += squared_weights[:,None,None] * (grad_sh_i ** 2)
                    squared_weightss["opacities"] += opacities ** 2     # TODO: find a better approximation for this
                    squared_weightss["scales"] += squared_weights[:,None] * (grad_scales_i ** 2)
                    squared_weightss["rotations"] += squared_weights[:,None] * (grad_rotations_i ** 2)

        else:
            colors_grad = torch.zeros((B, 3, max_height, max_width), device=means3D.device, dtype=means3D.dtype)
            invdepthss_grad = torch.zeros((B, 1, max_height, max_width), device=means3D.device, dtype=means3D.dtype)

            for i in range(B):
                # Restructure arguments the way that the C++ lib expects them
                args = (
                    batch_raster_settings.bg, 
                    batch_raster_settings_tangent.bg,
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
                    batch_raster_settings.scale_modifier,
                    batch_raster_settings_tangent.scale_modifier,
                    cov3Ds_precomp,
                    cov3Ds_precomp_tangent,
                    batch_raster_settings.viewmatrices[i],
                    batch_raster_settings_tangent.viewmatrices[i],
                    batch_raster_settings.projmatrices[i],
                    batch_raster_settings_tangent.projmatrices[i],
                    batch_raster_settings.tanfovxs[i],
                    batch_raster_settings_tangent.tanfovxs[i],
                    batch_raster_settings.tanfovys[i],
                    batch_raster_settings_tangent.tanfovys[i],
                    batch_raster_settings.image_heights[i],
                    batch_raster_settings.image_widths[i],
                    sh,
                    sh_tangent,
                    batch_raster_settings.sh_degree,
                    batch_raster_settings.camposes[i],
                    batch_raster_settings_tangent.camposes[i],
                    batch_raster_settings.prefiltered,
                    batch_raster_settings.antialiasing,
                    batch_raster_settings.debug
                )

                # Invoke C++/CUDA rasterizer
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths, color_grad, invdepths_grad = _C.rasterize_gaussians_jvp(*args)

                num_rendereds.append(num_rendered)
                colors = color.unsqueeze(0)
                radiis = radii.unsqueeze(0)
                invdepthss = invdepths.unsqueeze(0)
                geomBuffers.append(geomBuffer)
                binningBuffers.append(binningBuffer)
                imgBuffers.append(imgBuffer)

            ctx.save_for_forward(colors_grad, invdepthss_grad)

        # Keep relevant tensors for backward
        ctx.batch_raster_settings = batch_raster_settings
        ctx.num_rendereds = num_rendereds
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radiis, sh, opacities, geomBuffers[0], binningBuffers[0], imgBuffers[0])

        return colors, radiis, invdepthss, squared_weightss

    @staticmethod
    def backward(ctx, grad_out_color, _grad_radiis, grad_out_depth, _grad_squared_weights):

        # Restore necessary values from context
        num_rendereds = ctx.num_rendereds
        batch_raster_settings = ctx.batch_raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radiis, sh, opacities, geomBuffers, binningBuffers, imgBuffers = ctx.saved_tensors

        geomBuffers = [geomBuffers]
        binningBuffers = [binningBuffers]
        imgBuffers = [imgBuffers]

        B = batch_raster_settings.batch_size

        assert B == 1, "Currently only batch size of 1 is supported in batch rasterization."

        grad_means3D = torch.zeros_like(means3D)
        grad_means2D = None
        grad_sh = torch.zeros_like(sh)
        grad_colors_precomp = None
        grad_opacities = torch.zeros_like(opacities)
        grad_scales = torch.zeros_like(scales)
        grad_rotations = torch.zeros_like(rotations)
        grad_cov3Ds_precomp = None

        for i in range(B):

            # Restructure args as C++ method expects them
            args = (batch_raster_settings.bg,
                    means3D, 
                    radiis[i], 
                    colors_precomp, 
                    opacities,
                    scales, 
                    rotations, 
                    batch_raster_settings.scale_modifier, 
                    cov3Ds_precomp, 
                    batch_raster_settings.viewmatrices[i], 
                    batch_raster_settings.projmatrices[i], 
                    batch_raster_settings.tanfovxs[i], 
                    batch_raster_settings.tanfovys[i], 
                    grad_out_color[i],
                    grad_out_depth[i], 
                    sh, 
                    batch_raster_settings.sh_degree, 
                    batch_raster_settings.camposes[i],
                    geomBuffers[i],
                    num_rendereds[i],
                    binningBuffers[i],
                    imgBuffers[i],
                    batch_raster_settings.antialiasing,
                    batch_raster_settings.debug,)

            # Compute gradients for relevant tensors by invoking backward method
            grad_means2D_i, grad_colors_precomp_i, grad_opacities_i, grad_means3D_i, grad_cov3Ds_precomp_i, grad_sh_i, grad_scales_i, grad_rotations_i = _C.rasterize_gaussians_backward(*args)        
            grad_means3D += grad_means3D_i
            if grad_means2D is None:
                grad_means2D = grad_means2D_i
            else:
                grad_means2D += grad_means2D_i
            grad_sh += grad_sh_i
            if grad_colors_precomp is None:
                grad_colors_precomp = grad_colors_precomp_i
            else:
                grad_colors_precomp += grad_colors_precomp_i
            grad_opacities += grad_opacities_i
            grad_scales += grad_scales_i
            grad_rotations += grad_rotations_i
            if grad_cov3Ds_precomp is None:
                grad_cov3Ds_precomp = grad_cov3Ds_precomp_i
            else:
                grad_cov3Ds_precomp += grad_cov3Ds_precomp_i

            del grad_means2D_i, grad_colors_precomp_i, grad_opacities_i, grad_means3D_i, grad_cov3Ds_precomp_i, grad_sh_i, grad_scales_i, grad_rotations_i


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
        grad_batch_raster_settings,
        grad_jvp,
        grad_means3D_tangent,
        grad_means2D_tangent,
        grad_sh_tangent,
        grad_colors_precomp_tangent,
        grad_opacities_tangent,
        grad_scales_tangent,
        grad_rotations_tangent,
        grad_cov3Ds_precomp_tangent,
        grad_batch_raster_settings_tangent,):

        colors_grad, invdepthss_grad = ctx.saved_tensors
        return colors_grad, None, invdepthss_grad, None


class BatchGaussianRasterizationSettings(NamedTuple):
    batch_size : int
    image_heights : list[int]
    image_widths : list[int]
    tanfovxs : list[float]
    tanfovys : list[float]
    bg : torch.Tensor
    scale_modifier : float
    viewmatrices : list[torch.Tensor]
    projmatrices : list[torch.Tensor]
    sh_degree : int
    camposes : list[torch.Tensor]
    prefiltered : bool
    debug : bool
    antialiasing : bool
    track_weights : bool = False

class BatchGaussianRasterizer(nn.Module):
    def __init__(self, batch_raster_settings):
        super().__init__()
        self.batch_raster_settings = batch_raster_settings

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        batch_raster_settings = self.batch_raster_settings

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
        return batch_rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            batch_raster_settings, 
        )

