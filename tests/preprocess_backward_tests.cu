#include "auxiliary.h"

#include <gtest/gtest.h>
#include <iostream>
#include <tuple>
#include <utility>
#include "test_utils.h"
#include "float_grad.h"
#include "helper_math.h"
#include "backward_impl.h"

#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

void call_preprocess_backward_jvp(
    int P, int D, int M,
    const float3* means3D,
    const int* radii,
    const float* shs,
    const bool* clamped,
    const float* opacities,
    const glm::vec3* scales,
    const glm::vec4* rotations,
    const float scale_modifier,
    const float* cov3Ds,
    const float* viewmatrix,
    const float* projmatrix,
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    const glm::vec3* campos,
    const float3* dL_dmean2D,
    const float* dL_dconic,
    const float* dL_dinvdepth,
    float* dL_dopacity,
    glm::vec3* dL_dmean3D,
    float* dL_dcolor,
    float* dL_dcov3D,
    float* dL_dsh,
    glm::vec3* dL_dscale,
    glm::vec4* dL_drot,
    bool antialiasing) {

    BACKWARD::preprocess(
        P, D, M,
        means3D, radii, shs, clamped, opacities,
        scales, rotations, scale_modifier, cov3Ds, 
        viewmatrix, projmatrix,
        focal_x, focal_y, tan_fovx, tan_fovy,
        campos, dL_dmean2D, dL_dconic, dL_dinvdepth,
        dL_dopacity, dL_dmean3D, dL_dcolor,
        dL_dcov3D, dL_dsh, dL_dscale,
        dL_drot, antialiasing
    );
}

void call_preprocess_backward_floatgrad(
    int P, int D, int M,
    FloatGradArray<float3> means3D,
    const int* radii,
    FloatGradArray<float> shs,
    const bool* clamped,
    FloatGradArray<float> opacities,
    FloatGradArray<glm::vec3> scales,
    FloatGradArray<glm::vec4> rotations,
    FloatGrad<float> scale_modifier,
    FloatGradArray<float> cov3Ds,
    FloatGradArray<float> viewmatrix,
    FloatGradArray<float> projmatrix,
    FloatGrad<float> focal_x, FloatGrad<float> focal_y,
    FloatGrad<float> tan_fovx, FloatGrad<float> tan_fovy,
    FloatGradArray<glm::vec3> campos,
    FloatGradArray<float3> dL_dmean2D,
    FloatGradArray<float> dL_dconic,
    FloatGradArray<float> dL_dinvdepth,
    FloatGradArray<float> dL_dopacity,
    FloatGradArray<glm::vec3> dL_dmean3D,
    FloatGradArray<float> dL_dcolor,
    FloatGradArray<float> dL_dcov3D,
    FloatGradArray<float> dL_dsh,
    FloatGradArray<glm::vec3> dL_dscale,
    FloatGradArray<glm::vec4> dL_drot,
    bool antialiasing) {
    BACKWARD::preprocessJvp(
            P, D, M,
            means3D, radii, shs, clamped, opacities,
            scales, rotations, scale_modifier, cov3Ds,
            viewmatrix, projmatrix,
            focal_x, focal_y, tan_fovx, tan_fovy,
            campos, dL_dmean2D, dL_dconic, dL_dinvdepth,
            dL_dopacity, dL_dmean3D, dL_dcolor,
            dL_dcov3D, dL_dsh, dL_dscale,
            dL_drot, antialiasing
    );

}

TEST(BackwardJvpTest, PreprocessTest) {
    std::vector<int> pdm_vec;
    int pdm_rows, pdm_cols;
    read_csv("backward_render_data/pdm.txt", pdm_vec, pdm_rows, pdm_cols);
    int P = pdm_vec[0];
    int D = pdm_vec[1];
    int M = pdm_vec[2];

    std::vector<float> means3D_vec;
    int means3D_rows, means3D_cols;
    read_csv("backward_render_data/means3D.txt", means3D_vec, means3D_rows, means3D_cols);
    float3* means3D_host = (float3*) means3D_vec.data();

    std::vector<int> radii_vec;
    int radii_rows, radii_cols;
    read_csv("backward_render_data/radii.txt", radii_vec, radii_rows, radii_cols);
    int* radii_host = radii_vec.data();

    std::vector<char> clamped_vec;
    int clamped_rows, clamped_cols;
    read_csv("backward_render_data/clamped.txt", clamped_vec, clamped_rows, clamped_cols);
    bool* clamped_host = (bool*) clamped_vec.data();

    std::vector<float> shs_vec;
    int shs_rows, shs_cols;
    read_csv("backward_render_data/shs.txt", shs_vec, shs_rows, shs_cols);
    float* shs_host = shs_vec.data();

    std::vector<float> opacities_vec;
    int opacities_rows, opacities_cols;
    read_csv("backward_render_data/opacities.txt", opacities_vec, opacities_rows, opacities_cols);
    float* opacities_host = opacities_vec.data();

    std::vector<float> scales_vec;
    int scales_rows, scales_cols;
    read_csv("backward_render_data/scales.txt", scales_vec, scales_rows, scales_cols);
    glm::vec3* scales_host = (glm::vec3*) scales_vec.data();

    std::vector<float> rotations_vec;
    int rotations_rows, rotations_cols;
    read_csv("backward_render_data/rotations.txt", rotations_vec, rotations_rows, rotations_cols);
    glm::vec4* rotations_host = (glm::vec4*) rotations_vec.data();

    std::vector<float> scale_modifier_vec;
    int scale_modifier_rows, scale_modifier_cols;
    read_csv("backward_render_data/scale_modifier.txt", scale_modifier_vec, scale_modifier_rows, scale_modifier_cols);
    float scale_modifier = scale_modifier_vec[0];

    std::vector<float> cov3Ds_vec;
    int cov3Ds_rows, cov3Ds_cols;
    read_csv("backward_render_data/cov3D.txt", cov3Ds_vec, cov3Ds_rows, cov3Ds_cols);
    float* cov3Ds_host = cov3Ds_vec.data();

    std::vector<float> viewmatrix_vec;
    int viewmatrix_rows, viewmatrix_cols;
    read_csv("backward_render_data/viewmatrix.txt", viewmatrix_vec, viewmatrix_rows, viewmatrix_cols);
    float* viewmatrix_host = viewmatrix_vec.data();

    std::vector<float> projmatrix_vec;
    int projmatrix_rows, projmatrix_cols;
    read_csv("backward_render_data/projmatrix.txt", projmatrix_vec, projmatrix_rows, projmatrix_cols);
    float* projmatrix_host = projmatrix_vec.data();

    std::vector<float> focal_fov_vec;
    int focal_fov_rows, focal_fov_cols;
    read_csv("backward_render_data/focal_fov.txt", focal_fov_vec, focal_fov_rows, focal_fov_cols);
    float focal_x = focal_fov_vec[0];
    float focal_y = focal_fov_vec[1];
    float tan_fovx = focal_fov_vec[2];
    float tan_fovy = focal_fov_vec[3];

    std::vector<float> campos_vec;
    int campos_rows, campos_cols;
    read_csv("backward_render_data/campos.txt", campos_vec, campos_rows, campos_cols);
    glm::vec3* campos_host = (glm::vec3*) campos_vec.data();

    std::vector<float> dL_dmean2D_vec;
    int dL_dmean2D_rows, dL_dmean2D_cols;
    read_csv("backward_render_data/dL_dmean2D.txt", dL_dmean2D_vec, dL_dmean2D_rows, dL_dmean2D_cols);
    float3* dL_dmean2D_host = (float3*) dL_dmean2D_vec.data();

    std::vector<float> dL_dconic_vec;
    int dL_dconic_rows, dL_dconic_cols;
    read_csv("backward_render_data/dL_dconic.txt", dL_dconic_vec, dL_dconic_rows, dL_dconic_cols);
    float* dL_dconic_host = dL_dconic_vec.data();

    std::vector<float> dL_dinvdepth_vec;
    int dL_dinvdepth_rows, dL_dinvdepth_cols;
    read_csv("backward_render_data/dL_dinvdepth.txt", dL_dinvdepth_vec, dL_dinvdepth_rows, dL_dinvdepth_cols);
    float* dL_dinvdepth_host = dL_dinvdepth_vec.data();

    // These 2 are both inputs and outputs
    std::vector<float> dL_dopacity_input_vec;
    int dL_dopacity_rows, dL_dopacity_cols;
    read_csv("backward_render_data/dL_dopacity.txt", dL_dopacity_input_vec, dL_dopacity_rows, dL_dopacity_cols);

    std::vector<float> dL_dcolor_input_vec;
    int dL_dcolor_rows, dL_dcolor_cols;
    read_csv("backward_render_data/dL_dcolor.txt", dL_dcolor_input_vec, dL_dcolor_rows, dL_dcolor_cols);
    float* dL_dcolor_host = dL_dcolor_input_vec.data();

    std::vector<float> dL_dopacity_ref_vec = dL_dopacity_input_vec; // Initialize with input
    std::vector<float3> dL_dmean3D_ref_vec(P);
    std::vector<float> dL_dcolor_ref_vec = dL_dcolor_input_vec; // Initialize with input
    std::vector<float> dL_dcov3D_ref_vec(cov3Ds_rows * cov3Ds_cols);
    std::vector<float> dL_dsh_ref_vec(shs_rows * shs_cols);
    std::vector<float3> dL_dscale_ref_vec(scales_rows);
    std::vector<float4> dL_drot_ref_vec(rotations_rows);


    // Transfer all to device
    float3* means3D_device = host_to_device(means3D_host, means3D_rows);
    int* radii_device = host_to_device(radii_host, radii_vec.size());
    bool* clamped_device = host_to_device(clamped_host, clamped_vec.size());
    float* shs_device = host_to_device(shs_host, shs_vec.size());
    float* opacities_device = host_to_device(opacities_host, opacities_vec.size());
    glm::vec3* scales_device = host_to_device(scales_host, scales_rows);
    glm::vec4* rotations_device = host_to_device(rotations_host, rotations_rows);
    float* cov3Ds_device = host_to_device(cov3Ds_host, cov3Ds_vec.size());
    float* viewmatrix_device = host_to_device(viewmatrix_host, viewmatrix_vec.size());
    float* projmatrix_device = host_to_device(projmatrix_host, projmatrix_vec.size());
    glm::vec3* campos_device = host_to_device(campos_host, campos_rows);
    float3* dL_dmean2D_device = host_to_device(dL_dmean2D_host, dL_dmean2D_rows);
    float* dL_dconic_device = host_to_device(dL_dconic_host, dL_dconic_vec.size());
    float* dL_dinvdepth_device = host_to_device(dL_dinvdepth_host, dL_dinvdepth_vec.size());
    float* dL_dopacity_ref_device = host_to_device(dL_dopacity_ref_vec.data(), dL_dopacity_ref_vec.size());
    float* dL_dcolor_ref_device = host_to_device(dL_dcolor_host, dL_dcolor_ref_vec.size());
    float3* dL_dmean3D_ref_device = host_to_device(dL_dmean3D_ref_vec.data(), P);
    float* dL_dcov3D_ref_device = host_to_device(dL_dcov3D_ref_vec.data(), dL_dcov3D_ref_vec.size());
    float* dL_dsh_ref_device = host_to_device(dL_dsh_ref_vec.data(), dL_dsh_ref_vec.size());
    float3* dL_dscale_ref_device = host_to_device(dL_dscale_ref_vec.data(), scales_rows);
    float4* dL_drot_ref_device = host_to_device(dL_drot_ref_vec.data(), rotations_rows);

    bool antialiasing = false;

    cudaDeviceSynchronize();

    int i = 1090550;
    printf("before jvp color %.4e\n", dL_dcolor_ref_vec[3 * i + 0]);
    printf("before jvp color %.4e\n", dL_dcolor_ref_vec[3 * i + 1]);
    printf("before jvp color %.4e\n", dL_dcolor_ref_vec[3 * i + 2]);

    // Call the kernel
    call_preprocess_backward_jvp(
        P, D, M,
        means3D_device,
        radii_device,
        shs_device,
        clamped_device,
        opacities_device,
        scales_device,
        rotations_device,
        scale_modifier,
        cov3Ds_device,
        viewmatrix_device,
        projmatrix_device,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        campos_device,
        dL_dmean2D_device,
        dL_dconic_device,
        dL_dinvdepth_device,
        dL_dopacity_ref_device,
        (glm::vec3*) dL_dmean3D_ref_device,
        dL_dcolor_ref_device,
        dL_dcov3D_ref_device,
        dL_dsh_ref_device,
        (glm::vec3*) dL_dscale_ref_device,
        (glm::vec4*) dL_drot_ref_device,
        antialiasing
    );

    cudaDeviceSynchronize();

    // Transfer results back to host
    device_to_host(dL_dopacity_ref_vec.data(), dL_dopacity_ref_device, dL_dopacity_rows);
    device_to_host(dL_dmean3D_ref_vec.data(), dL_dmean3D_ref_device, P);
    device_to_host(dL_dcolor_ref_vec.data(), dL_dcolor_ref_device, dL_dcolor_rows);
    device_to_host(dL_dcov3D_ref_vec.data(), dL_dcov3D_ref_device, cov3Ds_rows);
    device_to_host(dL_dsh_ref_vec.data(), dL_dsh_ref_device, shs_rows);
    device_to_host(dL_dscale_ref_vec.data(), dL_dscale_ref_device, scales_rows);
    device_to_host(dL_drot_ref_vec.data(), dL_drot_ref_device, rotations_rows);

    cudaDeviceSynchronize();

    // Initialize tangent vectors
    std::vector<float> means3D_grad_vec(means3D_vec.size(), 0.1f);
    std::vector<float> shs_grad_vec(shs_vec.size(), 0.2f);
    std::vector<float> opacities_grad_vec(opacities_vec.size(), 0.3f);
    std::vector<float> scales_grad_vec(scales_vec.size(), 0.1f);
    std::vector<float> rotations_grad_vec(rotations_vec.size(), 0.2f);
    float scale_modifier_grad = 0.3f;
    std::vector<float> cov3Ds_grad_vec(cov3Ds_vec.size(), 0.3f);
    std::vector<float> viewmatrix_grad_vec(viewmatrix_vec.size(), 0.1f);
    std::vector<float> projmatrix_grad_vec(projmatrix_vec.size(), 0.2f);
    float focal_x_grad = 0.1f, focal_y_grad = 0.2f;
    float tan_fovx_grad = 0.1f, tan_fovy_grad = 0.2f;
    std::vector<float> campos_grad_vec(campos_vec.size(), 0.3f);
    std::vector<float> dL_dmean2D_grad_vec(dL_dmean2D_vec.size(), 0.1f);
    std::vector<float> dL_dconic_grad_vec(dL_dconic_vec.size(), 0.2f);
    std::vector<float> dL_dinvdepth_grad_vec(dL_dinvdepth_vec.size(), 0.3f);

    std::vector<float> dL_dopacity_vec = dL_dopacity_input_vec; // Initialize with input
    std::vector<float> dL_dopacity_grad_vec(dL_dopacity_vec.size(), 0.0f);
    std::vector<glm::vec3> dL_dmean3D_vec(P);
    std::vector<glm::vec3> dL_dmean3D_grad_vec(dL_dmean3D_vec.size());
    std::vector<float> dL_dcolor_vec = dL_dcolor_input_vec; // Initialize with input
    std::vector<float> dL_dcolor_grad_vec(dL_dcolor_vec.size(), 0.0f);
    std::vector<float> dL_dcov3D_vec(cov3Ds_rows * cov3Ds_cols);
    std::vector<float> dL_dcov3D_grad_vec(dL_dcov3D_vec.size(), 0.0f);
    std::vector<float> dL_dsh_vec(shs_rows * shs_cols);
    std::vector<float> dL_dsh_grad_vec(dL_dsh_vec.size(), 0.0f);
    std::vector<glm::vec3> dL_dscale_vec(scales_rows);
    std::vector<glm::vec3> dL_dscale_grad_vec(dL_dscale_vec.size());
    std::vector<glm::vec4> dL_drot_vec(rotations_rows);
    std::vector<glm::vec4> dL_drot_grad_vec(dL_drot_vec.size());

    // Transfer all to device
    float3* means3D_grad_device = host_to_device((float3*)means3D_grad_vec.data(), means3D_rows);
    float* shs_grad_device = host_to_device(shs_grad_vec.data(), shs_grad_vec.size());
    float* opacities_grad_device = host_to_device(opacities_grad_vec.data(), opacities_grad_vec.size());
    glm::vec3* scales_grad_device = host_to_device((glm::vec3*)scales_grad_vec.data(), scales_rows);
    glm::vec4* rotations_grad_device = host_to_device((glm::vec4*)rotations_grad_vec.data(), rotations_rows);
    float* cov3Ds_grad_device = host_to_device(cov3Ds_grad_vec.data(), cov3Ds_grad_vec.size());
    float* viewmatrix_grad_device = host_to_device(viewmatrix_grad_vec.data(), viewmatrix_grad_vec.size());
    float* projmatrix_grad_device = host_to_device(projmatrix_grad_vec.data(), projmatrix_grad_vec.size());
    glm::vec3* campos_grad_device = host_to_device((glm::vec3*)campos_grad_vec.data(), campos_rows);
    float3* dL_dmean2D_grad_device = host_to_device((float3*)dL_dmean2D_grad_vec.data(), dL_dmean2D_rows);
    float* dL_dconic_grad_device = host_to_device(dL_dconic_grad_vec.data(), dL_dconic_grad_vec.size());
    float* dL_dinvdepth_grad_device = host_to_device(dL_dinvdepth_grad_vec.data(), dL_dinvdepth_grad_vec.size());

    float* dL_dopacity_device = host_to_device(dL_dopacity_vec.data(), dL_dopacity_vec.size());
    float* dL_dopacity_grad_device = host_to_device(dL_dopacity_grad_vec.data(), dL_dopacity_grad_vec.size());
    glm::vec3* dL_dmean3D_device = host_to_device(dL_dmean3D_vec.data(), dL_dmean3D_vec.size());
    glm::vec3* dL_dmean3D_grad_device = host_to_device(dL_dmean3D_grad_vec.data(), dL_dmean3D_grad_vec.size());
    float* dL_dcolor_device = host_to_device(dL_dcolor_vec.data(), dL_dcolor_vec.size());
    float* dL_dcolor_grad_device = host_to_device(dL_dcolor_grad_vec.data(), dL_dcolor_grad_vec.size());
    float* dL_dcov3D_device = host_to_device(dL_dcov3D_vec.data(), dL_dcov3D_vec.size());
    float* dL_dcov3D_grad_device = host_to_device(dL_dcov3D_grad_vec.data(), dL_dcov3D_grad_vec.size());
    float* dL_dsh_device = host_to_device(dL_dsh_vec.data(), dL_dsh_vec.size());
    float* dL_dsh_grad_device = host_to_device(dL_dsh_grad_vec.data(), dL_dsh_grad_vec.size());
    glm::vec3* dL_dscale_device = host_to_device(dL_dscale_vec.data(), dL_dscale_vec.size());
    glm::vec3* dL_dscale_grad_device = host_to_device(dL_dscale_grad_vec.data(), dL_dscale_grad_vec.size());
    glm::vec4* dL_drot_device = host_to_device(dL_drot_vec.data(), dL_drot_vec.size());
    glm::vec4* dL_drot_grad_device = host_to_device(dL_drot_grad_vec.data(), dL_drot_grad_vec.size());

    // Wrap inputs in FloatGrad
    FloatGradArray<float3> means3D_floatgrad(means3D_device, means3D_grad_device);
    FloatGradArray<float> shs_floatgrad(shs_device, shs_grad_device);
    FloatGradArray<float> opacities_floatgrad(opacities_device, opacities_grad_device);
    FloatGradArray<glm::vec3> scales_floatgrad(scales_device, scales_grad_device);
    FloatGradArray<glm::vec4> rotations_floatgrad(rotations_device, rotations_grad_device);
    FloatGrad<float> scale_modifier_floatgrad(scale_modifier, scale_modifier_grad);
    FloatGradArray<float> cov3Ds_floatgrad(cov3Ds_device, cov3Ds_grad_device);
    FloatGradArray<float> viewmatrix_floatgrad(viewmatrix_device, viewmatrix_grad_device);
    FloatGradArray<float> projmatrix_floatgrad(projmatrix_device, projmatrix_grad_device);
    FloatGrad<float> focal_x_floatgrad(focal_x, focal_x_grad);
    FloatGrad<float> focal_y_floatgrad(focal_y, focal_y_grad);
    FloatGrad<float> tan_fovx_floatgrad(tan_fovx, tan_fovx_grad);
    FloatGrad<float> tan_fovy_floatgrad(tan_fovy, tan_fovy_grad);
    FloatGradArray<glm::vec3> campos_floatgrad(campos_device, campos_grad_device);
    FloatGradArray<float3> dL_dmean2D_floatgrad(dL_dmean2D_device, dL_dmean2D_grad_device);
    FloatGradArray<float> dL_dconic_floatgrad(dL_dconic_device, dL_dconic_grad_device);
    FloatGradArray<float> dL_dinvdepth_floatgrad(dL_dinvdepth_device, dL_dinvdepth_grad_device);
    FloatGradArray<float> dL_dopacity_floatgrad(dL_dopacity_device, dL_dopacity_grad_device);
    FloatGradArray<glm::vec3> dL_dmean3D_floatgrad(dL_dmean3D_device, dL_dmean3D_grad_device);
    FloatGradArray<float> dL_dcolor_floatgrad(dL_dcolor_device, dL_dcolor_grad_device);
    FloatGradArray<float> dL_dcov3D_floatgrad(dL_dcov3D_device, dL_dcov3D_grad_device);
    FloatGradArray<float> dL_dsh_floatgrad(dL_dsh_device, dL_dsh_grad_device);
    FloatGradArray<glm::vec3> dL_dscale_floatgrad(dL_dscale_device, dL_dscale_grad_device);
    FloatGradArray<glm::vec4> dL_drot_floatgrad(dL_drot_device, dL_drot_grad_device);

    printf("before floatgrad color %.4e\n", dL_dcolor_vec[3 * i + 0]);
    printf("before floatgrad color %.4e\n", dL_dcolor_vec[3 * i + 1]);
    printf("before floatgrad color %.4e\n", dL_dcolor_vec[3 * i + 2]);
    
    call_preprocess_backward_floatgrad(
        P, D, M,
        means3D_floatgrad,
        radii_device,
        shs_floatgrad,
        clamped_device,
        opacities_floatgrad,
        scales_floatgrad,
        rotations_floatgrad,
        scale_modifier_floatgrad,
        cov3Ds_floatgrad,
        viewmatrix_floatgrad,
        projmatrix_floatgrad,
        focal_x_floatgrad, focal_y_floatgrad,
        tan_fovx_floatgrad, tan_fovy_floatgrad,
        campos_floatgrad,
        dL_dmean2D_floatgrad,
        dL_dconic_floatgrad,
        dL_dinvdepth_floatgrad,
        dL_dopacity_floatgrad,
        dL_dmean3D_floatgrad,
        dL_dcolor_floatgrad,
        dL_dcov3D_floatgrad,
        dL_dsh_floatgrad,
        dL_dscale_floatgrad,
        dL_drot_floatgrad,
        antialiasing);

    cudaDeviceSynchronize();

    // Transfer results back to host
    device_to_host(dL_dopacity_vec.data(), dL_dopacity_device, dL_dopacity_vec.size());
    device_to_host(dL_dopacity_grad_vec.data(), dL_dopacity_grad_device, dL_dopacity_grad_vec.size());
    device_to_host(dL_dmean3D_vec.data(), dL_dmean3D_device, dL_dmean3D_vec.size());
    device_to_host(dL_dmean3D_grad_vec.data(), dL_dmean3D_grad_device, dL_dmean3D_grad_vec.size());
    device_to_host(dL_dcolor_vec.data(), dL_dcolor_device, dL_dcolor_vec.size());
    device_to_host(dL_dcolor_grad_vec.data(), dL_dcolor_grad_device, dL_dcolor_grad_vec.size());
    device_to_host(dL_dcov3D_vec.data(), dL_dcov3D_device, dL_dcov3D_vec.size());
    device_to_host(dL_dcov3D_grad_vec.data(), dL_dcov3D_grad_device, dL_dcov3D_grad_vec.size());
    device_to_host(dL_dsh_vec.data(), dL_dsh_device, dL_dsh_vec.size());
    device_to_host(dL_dsh_grad_vec.data(), dL_dsh_grad_device, dL_dsh_grad_vec.size());
    device_to_host(dL_dscale_vec.data(), dL_dscale_device, dL_dscale_vec.size());
    device_to_host(dL_dscale_grad_vec.data(), dL_dscale_grad_device, dL_dscale_grad_vec.size());
    device_to_host(dL_drot_vec.data(), dL_drot_device, dL_drot_vec.size());
    device_to_host(dL_drot_grad_vec.data(), dL_drot_grad_device, dL_drot_grad_vec.size());

    cudaDeviceSynchronize();



    for(int i = 0; i < P; i++) {
    // for(int i = 1090026; i < 1090046; i++) {
        EXPECT_NEAR(dL_dopacity_ref_vec[i], dL_dopacity_vec[i], 1e-5);
        EXPECT_NEAR(dL_dmean3D_ref_vec[i].x, dL_dmean3D_vec[i].x, 1e-5);
        EXPECT_NEAR(dL_dmean3D_ref_vec[i].y, dL_dmean3D_vec[i].y, 1e-5);
        EXPECT_NEAR(dL_dmean3D_ref_vec[i].z, dL_dmean3D_vec[i].z, 1e-5);
        EXPECT_NEAR(dL_dcolor_ref_vec[3 * i + 0], dL_dcolor_vec[3 * i + 0], 1e-5) << "index " << i;
        EXPECT_NEAR(dL_dcolor_ref_vec[3 * i + 1], dL_dcolor_vec[3 * i + 1], 1e-5) << "index " << i;
        EXPECT_NEAR(dL_dcolor_ref_vec[3 * i + 2], dL_dcolor_vec[3 * i + 2], 1e-5) << "index " << i;
        EXPECT_NEAR(dL_dcov3D_ref_vec[6 * i + 0], dL_dcov3D_vec[6 * i + 0], 1e-5) << "index " << i;
        EXPECT_NEAR(dL_dcov3D_ref_vec[6 * i + 1], dL_dcov3D_vec[6 * i + 1], 1e-5) << "index " << i;
        EXPECT_NEAR(dL_dcov3D_ref_vec[6 * i + 2], dL_dcov3D_vec[6 * i + 2], 1e-5);
        EXPECT_NEAR(dL_dcov3D_ref_vec[6 * i + 3], dL_dcov3D_vec[6 * i + 3], 1e-5);
        EXPECT_NEAR(dL_dcov3D_ref_vec[6 * i + 4], dL_dcov3D_vec[6 * i + 4], 1e-5);
        EXPECT_NEAR(dL_dcov3D_ref_vec[6 * i + 5], dL_dcov3D_vec[6 * i + 5], 1e-5);

        for (int j = 0; j < D + 1; j++) {
            EXPECT_NEAR(dL_dsh_ref_vec[(D + 1) * i + j], dL_dsh_vec[(D + 1) * i + j], 1e-5);
        }
        EXPECT_NEAR(dL_dscale_ref_vec[i].x, dL_dscale_vec[i].x, 1e-5);
        EXPECT_NEAR(dL_dscale_ref_vec[i].y, dL_dscale_vec[i].y, 1e-5);
        EXPECT_NEAR(dL_dscale_ref_vec[i].z, dL_dscale_vec[i].z, 1e-5);

        EXPECT_NEAR(dL_drot_ref_vec[i].x, dL_drot_vec[i].x, 1e-5);
        EXPECT_NEAR(dL_drot_ref_vec[i].y, dL_drot_vec[i].y, 1e-5);
        EXPECT_NEAR(dL_drot_ref_vec[i].z, dL_drot_vec[i].z, 1e-5);
        EXPECT_NEAR(dL_drot_ref_vec[i].w, dL_drot_vec[i].w, 1e-5);



    }


}
