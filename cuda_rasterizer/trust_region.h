#ifndef CUDA_RASTERIZER_TRUST_REGION_H_INCLUDED
#define CUDA_RASTERIZER_TRUST_REGION_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace TRUST_REGION
{
    void ComputeTrustRegionStep(
        const int P, const int M,
        const float trust_radius,
        const float min_mass_scaling,
        const float max_mass_scaling,
        const float quat_norm_tr,
        const float* xyz_params,
        const glm::vec3* scaling_params,
        const float scale_modifier,
        const glm::vec4* quat_params,
        const float* opacity_params,
        const float* shs_params,
        float* xyz_params_step,
        glm::vec3* scaling_params_step,
        glm::vec4* quat_params_step,
        float* opacity_params_step,
        float* shs_params_step);


}

#endif
