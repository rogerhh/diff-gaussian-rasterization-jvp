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

__global__
void call_computeColorFromSH_jvp(int P, int D, int M, 
                                 float3* orig_points, 
                                 glm::vec3* cam_pos, 
                                 float* shs, 
                                 bool* clamped,
                                 glm::vec3* dL_dcolor,
                                 glm::vec3* dL_dmeans,
                                 glm::vec3* dL_dshs) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    BACKWARD::computeColorFromSH(idx, D, M, cast<glm::vec3>(orig_points), 
                                 *cam_pos, shs, clamped,
                                 dL_dcolor, dL_dmeans, dL_dshs);
}

__global__
void call_computeColorFromSH_floatgrad(int P, int D, int M, 
                                       float3* orig_points, 
                                       FloatGradArray<glm::vec3> cam_pos, 
                                       FloatGradArray<float> shs, 
                                       bool* clamped,
                                       glm::vec3* dL_dcolor,
                                       FloatGradArray<glm::vec3> dL_dmeans,
                                       FloatGradArray<glm::vec3> dL_dshs) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    BACKWARD::computeColorFromSH(idx, D, M, cast<glm::vec3>(orig_points), 
                                 cam_pos[0], shs, clamped,
                                 dL_dcolor, dL_dmeans, dL_dshs);
}

TEST(BackwardJvpTest, ComputeColorFromSHTest) {
    std::vector<float> orig_points_host;
    int orig_points_rows, orig_points_cols;
    read_csv("data/means3D.csv", orig_points_host, orig_points_rows, orig_points_cols);
    EXPECT_EQ(orig_points_cols, 3);

    std::vector<int> sh_degree_host;
    int sh_degree_rows, sh_degree_cols;
    read_csv("data/sh_degree.csv", sh_degree_host, sh_degree_rows, sh_degree_cols);

    std::vector<float> shs_host;
    int shs_rows, shs_cols;
    read_csv("data/sh.csv", shs_host, shs_rows, shs_cols);

    std::vector<float> cam_pos_host;
    int cam_pos_rows, cam_pos_cols;
    read_csv("data/campos.csv", cam_pos_host, cam_pos_rows, cam_pos_cols);

    std::vector<float> cam_pos_grad_host;
    for (int i = 0; i < cam_pos_rows * cam_pos_cols; i++) {
        cam_pos_grad_host.push_back((i * 0.2f) + 0.5f);
    }

    std::vector<float> shs_grad_host;
    for (int i = 0; i < shs_rows * shs_cols; i++) {
        shs_grad_host.push_back(((i % 13) * 0.1f) + 0.3f);
    }

    bool* clamped_host = new bool[orig_points_rows * 3]();

    std::vector<glm::vec3> dL_dcolor_host(orig_points_rows, glm::vec3(0.0f));
    for (int i = 0; i < orig_points_rows; i++) {
        dL_dcolor_host[i] = glm::vec3(((i % 4) * 0.1f) + 0.3f, 
                                      ((i % 5) * 0.09f) + 0.5f, 
                                      ((i % 6) * 0.11f) + 0.7f);
    }

    float3* orig_points_device = host_to_device((float3*) orig_points_host.data(), orig_points_rows * orig_points_cols);
    int sh_degree = sh_degree_host[0];
    glm::vec3* cam_pos_device = host_to_device((glm::vec3*) cam_pos_host.data(), cam_pos_rows);
    float* shs_device = host_to_device(shs_host.data(), shs_rows * shs_cols);
    bool* clamped_device = host_to_device(clamped_host, orig_points_rows * 3);
    glm::vec3* cam_pos_grad_device = host_to_device((glm::vec3*) cam_pos_grad_host.data(), cam_pos_rows);
    float* shs_grad_device = host_to_device(shs_grad_host.data(), shs_rows * shs_cols);
    glm::vec3* dL_dcolor_device = host_to_device(dL_dcolor_host.data(), orig_points_rows);

    std::vector<glm::vec3> dL_dmeans_ref_host(orig_points_rows, glm::vec3(0.0f));
    std::vector<glm::vec3> dL_dshs_ref_host(orig_points_rows * shs_cols, glm::vec3(0.0f));

    glm::vec3* dL_dmeans_ref_device = host_to_device(dL_dmeans_ref_host.data(), orig_points_rows);
    glm::vec3* dL_dshs_ref_device = host_to_device(dL_dshs_ref_host.data(), orig_points_rows * shs_cols);

    int P = orig_points_rows;
    int num_blocks = (P + 16) / 16;
    int threads_per_block = 16;

    sh_degree = 2;

    call_computeColorFromSH_jvp<<<num_blocks, threads_per_block>>>(
        P, sh_degree, shs_cols,
        orig_points_device,
        cam_pos_device, 
        shs_device, 
        clamped_device,
        dL_dcolor_device,
        dL_dmeans_ref_device,
        dL_dshs_ref_device
    );

    cudaDeviceSynchronize();

    device_to_host(dL_dmeans_ref_host.data(), dL_dmeans_ref_device, orig_points_rows);
    device_to_host(dL_dshs_ref_host.data(), dL_dshs_ref_device, orig_points_rows * shs_cols);

    FloatGradArray<glm::vec3> cam_pos_floatgrad(cam_pos_device, cam_pos_grad_device);
    FloatGradArray<float> shs_floatgrad(shs_device, shs_grad_device);

    std::vector<glm::vec3> dL_dmeans_host(orig_points_rows, glm::vec3(0.0f));
    std::vector<glm::vec3> dL_dmeans_grad_host(orig_points_rows, glm::vec3(0.0f));
    std::vector<glm::vec3> dL_dshs_host(orig_points_rows * shs_cols, glm::vec3(0.0f));
    std::vector<glm::vec3> dL_dshs_grad_host(orig_points_rows * shs_cols, glm::vec3(0.0f));

    glm::vec3* dL_dmeans_device = host_to_device(dL_dmeans_host.data(), orig_points_rows);
    glm::vec3* dL_dmeans_grad_device = host_to_device(dL_dmeans_grad_host.data(), orig_points_rows);
    glm::vec3* dL_dshs_device = host_to_device(dL_dshs_host.data(), orig_points_rows * shs_cols);
    glm::vec3* dL_dshs_grad_device = host_to_device(dL_dshs_grad_host.data(), orig_points_rows * shs_cols);

    FloatGradArray<glm::vec3> dL_dmeans_floatgrad(dL_dmeans_device, dL_dmeans_grad_device);
    FloatGradArray<glm::vec3> dL_dshs_floatgrad(dL_dshs_device, dL_dshs_grad_device);

    call_computeColorFromSH_floatgrad<<<num_blocks, threads_per_block>>>(
        P, sh_degree, shs_cols,
        orig_points_device,
        cam_pos_floatgrad,
        shs_floatgrad,
        clamped_device,
        dL_dcolor_device,
        dL_dmeans_floatgrad,
        dL_dshs_floatgrad
    );

    cudaDeviceSynchronize();

    device_to_host(dL_dmeans_host.data(), dL_dmeans_device, orig_points_rows);
    device_to_host(dL_dmeans_grad_host.data(), dL_dmeans_grad_device, orig_points_rows);
    device_to_host(dL_dshs_host.data(), dL_dshs_device, orig_points_rows * shs_cols);
    device_to_host(dL_dshs_grad_host.data(), dL_dshs_grad_device, orig_points_rows * shs_cols);

    // for (int i = 0; i < P; i++) {
    for (int i = 17045; i < 17046; i++) {
        EXPECT_NEAR(dL_dmeans_ref_host[i].x, dL_dmeans_host[i].x, 1e-5f) << " at index " << i;
        EXPECT_NEAR(dL_dmeans_ref_host[i].y, dL_dmeans_host[i].y, 1e-5f) << " at index " << i;
        EXPECT_NEAR(dL_dmeans_ref_host[i].z, dL_dmeans_host[i].z, 1e-5f) << " at index " << i;
        EXPECT_NEAR(dL_dshs_ref_host[i].x, dL_dshs_host[i].x, 1e-5f) << " at index " << i;
        EXPECT_NEAR(dL_dshs_ref_host[i].y, dL_dshs_host[i].y, 1e-5f) << " at index " << i;
        EXPECT_NEAR(dL_dshs_ref_host[i].z, dL_dshs_host[i].z, 1e-5f) << " at index " << i;

        if (i < 10) {
            std::cout << "idx = " << i << " dL_dmeans_ref = " 
                      << dL_dmeans_ref_host[i].x << ", " 
                      << dL_dmeans_ref_host[i].y << ", " 
                      << dL_dmeans_ref_host[i].z << " | dL_dmeans = "
                      << dL_dmeans_host[i].x << ", "
                      << dL_dmeans_host[i].y << ", "
                      << dL_dmeans_host[i].z << " | dL_dmeans_grad = "
                      << dL_dmeans_grad_host[i].x << ", "
                      << dL_dmeans_grad_host[i].y << ", "
                      << dL_dmeans_grad_host[i].z << std::endl;
        }
    }

    free(clamped_host);
    free_device(orig_points_device);
    free_device(cam_pos_device);
    free_device(shs_device);
    free_device(clamped_device);
    free_device(cam_pos_grad_device);
    free_device(shs_grad_device);
    free_device(dL_dcolor_device);
    free_device(dL_dmeans_ref_device);
    free_device(dL_dshs_ref_device);
    free_device(dL_dmeans_device);
    free_device(dL_dmeans_grad_device);
    free_device(dL_dshs_device);
    free_device(dL_dshs_grad_device);

}
