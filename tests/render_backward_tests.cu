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

void call_render_backward_jvp(
    const dim3 grid, const dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    int W, int H,
    const float* bg_color,
    const float2* means2D,
    const float4* conic_opacity,
    const float* colors,
    const float* depths,
    const float* final_Ts,
    const uint32_t* n_contrib,
    const float* dL_dpixels,
    const float* dL_invdepths,
    float3* dL_dmean2D,
    float4* dL_dconic2D,
    float* dL_dopacity,
    float* dL_dcolors,
    float* dL_dinvdepths) {

    BACKWARD::render(grid, block,
                     ranges,
                     point_list,
                     W, H,
                     bg_color,
                     means2D,
                     conic_opacity,
                     colors,
                     depths,
                     final_Ts,
                     n_contrib,
                     dL_dpixels,
                     dL_invdepths,
                     dL_dmean2D,
                     dL_dconic2D,
                     dL_dopacity,
                     dL_dcolors,
                     dL_dinvdepths);
}

void call_render_backward_floatgrad(
    const dim3 grid, const dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    int W, int H,
    FloatGradArray<float> bg_color,
    FloatGradArray<float2> means2D,
    FloatGradArray<float4> conic_opacity,
    FloatGradArray<float> colors,
    FloatGradArray<float> depths,
    FloatGradArray<float> final_Ts,
    const uint32_t* n_contrib,
    FloatGradArray<float> dL_dpixels,
    FloatGradArray<float> dL_invdepths,
    FloatGradArray<float3> dL_dmean2D,
    FloatGradArray<float4> dL_dconic2D,
    FloatGradArray<float> dL_dopacity,
    FloatGradArray<float> dL_dcolors,
    FloatGradArray<float> dL_dinvdepths) {

    BACKWARD::renderJvp(grid, block,
                        ranges,
                        point_list,
                        W, H,
                        bg_color,
                        means2D,
                        conic_opacity,
                        colors,
                        depths,
                        final_Ts,
                        n_contrib,
                        dL_dpixels,
                        dL_invdepths,
                        dL_dmean2D,
                        dL_dconic2D,
                        dL_dopacity,
                        dL_dcolors,
                        dL_dinvdepths);

}

TEST(BackwardJvpTest, RenderTest) {
    std::vector<int> tile_grid_vec;
    int tile_grid_rows, tile_grid_cols;
    read_csv("backward_render_data/tile_grid.txt", tile_grid_vec, tile_grid_rows, tile_grid_cols);
    dim3 tile_grid(tile_grid_vec[0], tile_grid_vec[1], tile_grid_vec[2]);

    std::vector<int> block_vec;
    int block_rows, block_cols;
    read_csv("backward_render_data/block.txt", block_vec, block_rows, block_cols);
    dim3 block(block_vec[0], block_vec[1], block_vec[2]);

    std::vector<int> ranges_vec;
    int ranges_rows, ranges_cols;
    read_csv("backward_render_data/ranges.txt", ranges_vec, ranges_rows, ranges_cols);
    uint2* ranges_host = (uint2*) ranges_vec.data();

    std::vector<int> point_list_vec;
    int point_list_rows, point_list_cols;
    read_csv("backward_render_data/point_list.txt", point_list_vec, point_list_rows, point_list_cols);
    uint32_t* point_list_host = (uint32_t*) point_list_vec.data();

    std::vector<int> wh_vec;
    int wh_rows, wh_cols;
    read_csv("backward_render_data/wh.txt", wh_vec, wh_rows, wh_cols);
    int W = wh_vec[0];
    int H = wh_vec[1];

    std::vector<float> bg_vec;
    int bg_rows, bg_cols;
    read_csv("backward_render_data/bg.txt", bg_vec, bg_rows, bg_cols);
    float* bg_host = bg_vec.data();

    std::vector<float> means2D_vec;
    int means2D_rows, means2D_cols;
    read_csv("backward_render_data/means2D.txt", means2D_vec, means2D_rows, means2D_cols);
    float2* means2D_host = (float2*) means2D_vec.data();

    std::vector<float> conic_opacity_vec;
    int conic_opacity_rows, conic_opacity_cols;
    read_csv("backward_render_data/conic_opacity.txt", conic_opacity_vec, conic_opacity_rows, conic_opacity_cols);
    float4* conic_opacity_host = (float4*) conic_opacity_vec.data();

    std::vector<float> colors_vec;
    int colors_rows, colors_cols;
    read_csv("backward_render_data/color_ptr.txt", colors_vec, colors_rows, colors_cols);
    float* colors_host = colors_vec.data();

    std::vector<float> depths_vec;
    int depths_rows, depths_cols;
    read_csv("backward_render_data/depths.txt", depths_vec, depths_rows, depths_cols);
    float* depths_host = depths_vec.data();

    std::vector<float> final_Ts_vec;
    int final_Ts_rows, final_Ts_cols;
    read_csv("backward_render_data/accum_alpha.txt", final_Ts_vec, final_Ts_rows, final_Ts_cols);
    float* final_Ts_host = final_Ts_vec.data();

    std::vector<int> n_contrib_vec;
    int n_contrib_rows, n_contrib_cols;
    read_csv("backward_render_data/n_contrib.txt", n_contrib_vec, n_contrib_rows, n_contrib_cols);
    uint32_t* n_contrib_host = (uint32_t*) n_contrib_vec.data();

    std::vector<float> dL_dpixels_vec;
    int dL_dpixels_rows, dL_dpixels_cols;
    read_csv("backward_render_data/dL_dpix.txt", dL_dpixels_vec, dL_dpixels_rows, dL_dpixels_cols);
    float* dL_dpixels_host = dL_dpixels_vec.data();

    std::vector<float> dL_invdepths_vec;
    int dL_invdepths_rows, dL_invdepths_cols;
    read_csv("backward_render_data/dL_invdepths.txt", dL_invdepths_vec, dL_invdepths_rows, dL_invdepths_cols);
    float* dL_invdepths_host = dL_invdepths_vec.data();

    int P = means2D_rows;

    std::vector<float3> dL_dmean2D_ref_vec(P);
    std::vector<float4> dL_dconic2D_ref_vec(P);
    std::vector<float> dL_dopacity_ref_vec(P);
    std::vector<float> dL_dcolors_ref_vec(P * 3);
    std::vector<float> dL_dinvdepths_ref_vec(P);

    // Transfer all to device
    uint2* ranges_device = host_to_device(ranges_host, ranges_rows);
    uint32_t* point_list_device = host_to_device(point_list_host, point_list_rows);
    float* bg_device = host_to_device(bg_host, bg_rows * bg_cols);
    float2* means2D_device = host_to_device(means2D_host, means2D_rows);
    float4* conic_opacity_device = host_to_device(conic_opacity_host, conic_opacity_rows);
    float* colors_device = host_to_device(colors_host, colors_rows * colors_cols);
    float* depths_device = host_to_device(depths_host, depths_rows * depths_cols);
    float* final_Ts_device = host_to_device(final_Ts_host, final_Ts_rows * final_Ts_cols);
    uint32_t* n_contrib_device = host_to_device(n_contrib_host, n_contrib_rows * n_contrib_cols);
    float* dL_dpixels_device = host_to_device(dL_dpixels_host, dL_dpixels_rows * dL_dpixels_cols);
    float* dL_invdepths_device = host_to_device(dL_invdepths_host, dL_invdepths_rows * dL_invdepths_cols);
    float3* dL_dmean2D_ref_device = host_to_device(dL_dmean2D_ref_vec.data(), P);
    float4* dL_dconic2D_ref_device = host_to_device(dL_dconic2D_ref_vec.data(), P);
    float* dL_dopacity_ref_device = host_to_device(dL_dopacity_ref_vec.data(), P);
    float* dL_dcolors_ref_device = host_to_device(dL_dcolors_ref_vec.data(), P * 3);
    float* dL_dinvdepths_ref_device = host_to_device(dL_dinvdepths_ref_vec.data(), P);

    // Call the kernel
    call_render_backward_jvp(
        tile_grid, block,
        ranges_device,
        point_list_device,
        W, H,
        bg_device,
        means2D_device,
        conic_opacity_device,
        colors_device,
        depths_device,
        final_Ts_device,
        n_contrib_device,
        dL_dpixels_device,
        dL_invdepths_device,
        dL_dmean2D_ref_device,
        dL_dconic2D_ref_device,
        dL_dopacity_ref_device,
        dL_dcolors_ref_device,
        dL_dinvdepths_ref_device
    );

    cudaDeviceSynchronize();

    // Transfer results back to host
    device_to_host(dL_dmean2D_ref_vec.data(), dL_dmean2D_ref_device, P);
    device_to_host(dL_dconic2D_ref_vec.data(), dL_dconic2D_ref_device, P);
    device_to_host(dL_dopacity_ref_vec.data(), dL_dopacity_ref_device, P);
    device_to_host(dL_dcolors_ref_vec.data(), dL_dcolors_ref_device, P * 3);
    device_to_host(dL_dinvdepths_ref_vec.data(), dL_dinvdepths_ref_device, P);

    cudaDeviceSynchronize();


    // Initialize the gradients

    std::vector<float> bg_grad_vec(bg_vec.size(), 0.1f);
    std::vector<float> means2D_grad_vec(means2D_vec.size(), 0.2f);
    std::vector<float> conic_opacity_grad_vec(conic_opacity_vec.size(), 0.3f);
    std::vector<float> colors_grad_vec(colors_vec.size(), 0.1f);
    std::vector<float> depths_grad_vec(depths_vec.size(), 0.2f);
    std::vector<float> final_Ts_grad_vec(final_Ts_vec.size(), 0.3f);
    std::vector<float> n_contrib_grad_vec(n_contrib_vec.size(), 0.1f);
    std::vector<float> dL_dpixels_grad_vec(dL_dpixels_vec.size(), 0.2f);
    std::vector<float> dL_invdepths_grad_vec(dL_invdepths_vec.size(), 0.3f);

    std::vector<float3> dL_dmean2D_vec(P, {0.0f, 0.0f, 0.0f});
    std::vector<float3> dL_dmean2D_grad_vec(P, {0.0f, 0.0f, 0.0f});
    std::vector<float4> dL_dconic2D_vec(P, {0.0f, 0.0f, 0.0f, 0.0f});
    std::vector<float4> dL_dconic2D_grad_vec(P, {0.0f, 0.0f, 0.0f, 0.0f});
    std::vector<float> dL_dopacity_vec(P, 0.0f);
    std::vector<float> dL_dopacity_grad_vec(P, 0.0f);
    std::vector<float> dL_dcolors_vec(P * 3, 0.0f);
    std::vector<float> dL_dcolors_grad_vec(P * 3, 0.0f);
    std::vector<float> dL_dinvdepths_vec(P, 0.0f);
    std::vector<float> dL_dinvdepths_grad_vec(P, 0.0f);

    float* bg_grad_device = host_to_device(bg_grad_vec.data(), bg_grad_vec.size());
    float2* means2D_grad_device = host_to_device((float2*) means2D_grad_vec.data(), means2D_grad_vec.size() / 2);
    float4* conic_opacity_grad_device = host_to_device((float4*) conic_opacity_grad_vec.data(), conic_opacity_grad_vec.size() / 4);
    float* colors_grad_device = host_to_device(colors_grad_vec.data(), colors_grad_vec.size());
    float* depths_grad_device = host_to_device(depths_grad_vec.data(), depths_grad_vec.size());
    float* final_Ts_grad_device = host_to_device(final_Ts_grad_vec.data(), final_Ts_grad_vec.size());
    float* dL_dpixels_grad_device = host_to_device(dL_dpixels_grad_vec.data(), dL_dpixels_grad_vec.size());
    float* dL_invdepths_grad_device = host_to_device(dL_invdepths_grad_vec.data(), dL_invdepths_grad_vec.size());
    float3* dL_dmean2D_device = host_to_device(dL_dmean2D_vec.data(), P);
    float3* dL_dmean2D_grad_device = host_to_device(dL_dmean2D_grad_vec.data(), P);
    float4* dL_dconic2D_device = host_to_device(dL_dconic2D_vec.data(), P);
    float4* dL_dconic2D_grad_device = host_to_device(dL_dconic2D_grad_vec.data(), P);
    float* dL_dopacity_device = host_to_device(dL_dopacity_vec.data(), P);
    float* dL_dopacity_grad_device = host_to_device(dL_dopacity_grad_vec.data(), P);
    float* dL_dcolors_device = host_to_device(dL_dcolors_vec.data(), P * 3);
    float* dL_dcolors_grad_device = host_to_device(dL_dcolors_grad_vec.data(), P * 3);
    float* dL_dinvdepths_device = host_to_device(dL_dinvdepths_vec.data(), P);
    float* dL_dinvdepths_grad_device = host_to_device(dL_dinvdepths_grad_vec.data(), P);

    FloatGradArray<float> bg_floatgrad(bg_device, bg_grad_device);
    FloatGradArray<float2> means2D_floatgrad(means2D_device, means2D_grad_device);
    FloatGradArray<float4> conic_opacity_floatgrad(conic_opacity_device, conic_opacity_grad_device);
    FloatGradArray<float> colors_floatgrad(colors_device, colors_grad_device);
    FloatGradArray<float> depths_floatgrad(depths_device, depths_grad_device);
    FloatGradArray<float> final_Ts_floatgrad(final_Ts_device, final_Ts_grad_device);
    FloatGradArray<float> dL_dpixels_floatgrad(dL_dpixels_device, dL_dpixels_grad_device);
    FloatGradArray<float> dL_invdepths_floatgrad(dL_invdepths_device, dL_invdepths_grad_device);
    FloatGradArray<float3> dL_dmean2D_floatgrad(dL_dmean2D_device, dL_dmean2D_grad_device);
    FloatGradArray<float4> dL_dconic2D_floatgrad(dL_dconic2D_device, dL_dconic2D_grad_device);
    FloatGradArray<float> dL_dopacity_floatgrad(dL_dopacity_device, dL_dopacity_grad_device);
    FloatGradArray<float> dL_dcolors_floatgrad(dL_dcolors_device, dL_dcolors_grad_device);
    FloatGradArray<float> dL_dinvdepths_floatgrad(dL_dinvdepths_device, dL_dinvdepths_grad_device);

    // Call the kernel
    call_render_backward_floatgrad(
        tile_grid, block,
        ranges_device,
        point_list_device,
        W, H,
        bg_floatgrad,
        means2D_floatgrad,
        conic_opacity_floatgrad,
        colors_floatgrad,
        depths_floatgrad,
        final_Ts_floatgrad,
        n_contrib_device,
        dL_dpixels_floatgrad,
        dL_invdepths_floatgrad,
        dL_dmean2D_floatgrad,
        dL_dconic2D_floatgrad,
        dL_dopacity_floatgrad,
        dL_dcolors_floatgrad,
        dL_dinvdepths_floatgrad
    );

    cudaDeviceSynchronize();

    // Transfer results back to host
    device_to_host(dL_dmean2D_vec.data(), dL_dmean2D_device, P);
    device_to_host(dL_dmean2D_grad_vec.data(), dL_dmean2D_grad_device, P);
    device_to_host(dL_dconic2D_vec.data(), dL_dconic2D_device, P);
    device_to_host(dL_dconic2D_grad_vec.data(), dL_dconic2D_grad_device, P);
    device_to_host(dL_dopacity_vec.data(), dL_dopacity_device, P);
    device_to_host(dL_dopacity_grad_vec.data(), dL_dopacity_grad_device, P);
    device_to_host(dL_dcolors_vec.data(), dL_dcolors_device, P * 3);
    device_to_host(dL_dcolors_grad_vec.data(), dL_dcolors_grad_device, P * 3);
    device_to_host(dL_dinvdepths_vec.data(), dL_dinvdepths_device, P);
    device_to_host(dL_dinvdepths_grad_vec.data(), dL_dinvdepths_grad_device, P);

    cudaDeviceSynchronize();

    for(int i = 0; i < 10; i++) {

        // dL_dmean2D
        EXPECT_NEAR(dL_dmean2D_vec[i].x, dL_dmean2D_ref_vec[i].x, 1e-8);
        EXPECT_NEAR(dL_dmean2D_vec[i].y, dL_dmean2D_ref_vec[i].y, 1e-8);
        EXPECT_NEAR(dL_dmean2D_vec[i].z, dL_dmean2D_ref_vec[i].z, 1e-8);

        // dL_dmean2D grad
        EXPECT_TRUE(!((dL_dmean2D_vec[i].x == 0.0f) ^ (dL_dmean2D_grad_vec[i].x == 0.0f)));
        EXPECT_TRUE(!((dL_dmean2D_vec[i].y == 0.0f) ^ (dL_dmean2D_grad_vec[i].y == 0.0f)));
        EXPECT_TRUE(!((dL_dmean2D_vec[i].z == 0.0f) ^ (dL_dmean2D_grad_vec[i].z == 0.0f)));

        // dL_dconic2D
        EXPECT_NEAR(dL_dconic2D_vec[i].x, dL_dconic2D_ref_vec[i].x, 1e-8);
        EXPECT_NEAR(dL_dconic2D_vec[i].y, dL_dconic2D_ref_vec[i].y, 1e-8);
        EXPECT_NEAR(dL_dconic2D_vec[i].z, dL_dconic2D_ref_vec[i].z, 1e-8);
        EXPECT_NEAR(dL_dconic2D_vec[i].w, dL_dconic2D_ref_vec[i].w, 1e-8);

        // dL_dconic2D grad
        EXPECT_TRUE(!((dL_dconic2D_vec[i].x == 0.0f) ^ (dL_dconic2D_grad_vec[i].x == 0.0f)));
        EXPECT_TRUE(!((dL_dconic2D_vec[i].y == 0.0f) ^ (dL_dconic2D_grad_vec[i].y == 0.0f)));
        EXPECT_TRUE(!((dL_dconic2D_vec[i].z == 0.0f) ^ (dL_dconic2D_grad_vec[i].z == 0.0f)));
        EXPECT_TRUE(!((dL_dconic2D_vec[i].w == 0.0f) ^ (dL_dconic2D_grad_vec[i].w == 0.0f)));

        // dL_dopacity
        EXPECT_NEAR(dL_dopacity_vec[i], dL_dopacity_ref_vec[i], 1e-8);

        // dL_dopacity grad
        EXPECT_TRUE(!((dL_dopacity_vec[i] == 0.0f) ^ (dL_dopacity_grad_vec[i] == 0.0f)));

        // dL_dcolors
        EXPECT_NEAR(dL_dcolors_vec[i * 3 + 0], dL_dcolors_ref_vec[i * 3 + 0], 1e-8);
        EXPECT_NEAR(dL_dcolors_vec[i * 3 + 1], dL_dcolors_ref_vec[i * 3 + 1], 1e-8);
        EXPECT_NEAR(dL_dcolors_vec[i * 3 + 2], dL_dcolors_ref_vec[i * 3 + 2], 1e-8);
        
        // dL_dcolors grad
        EXPECT_TRUE(!((dL_dcolors_vec[i * 3 + 0] == 0.0f) ^ (dL_dcolors_grad_vec[i * 3 + 0] == 0.0f)));

        // dL_dinvdepths
        EXPECT_NEAR(dL_dinvdepths_vec[i], dL_dinvdepths_ref_vec[i], 1e-8);

        // dL_dinvdepths grad
        // This one is special, because dL_dinvdepths is often zero but the grad is not zero
        EXPECT_TRUE(!((dL_dinvdepths_vec[i] != 0.0f) && (dL_dinvdepths_grad_vec[i] == 0.0f)));

    }

}
