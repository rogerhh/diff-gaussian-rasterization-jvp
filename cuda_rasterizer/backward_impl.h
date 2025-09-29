#ifndef BACKWARD_IMPL_H
#define BACKWARD_IMPL_H

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <float_grad.h>
#include "float_grad_helper_math.h"

namespace cg = cooperative_groups;

namespace BACKWARD {

template <typename T>
__device__ __forceinline__ auto sq(T x) { return x * x; }

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
template <typename... JvpArgs>
__device__ void computeColorFromSH(JvpArgs&&... jvp_args)
    // int idx, int deg, int max_coeffs, const glm::vec3* means, 
    // glm::vec3 campos, const float* shs, const bool* clamped, 
    // const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs
{
    auto jvp_args_tuple = std::forward_as_tuple(std::forward<JvpArgs>(jvp_args)...);
    int idx = std::get<0>(jvp_args_tuple);
    int deg = std::get<1>(jvp_args_tuple);
    int max_coeffs = std::get<2>(jvp_args_tuple);
    auto means = std::get<3>(jvp_args_tuple);
    auto campos = std::get<4>(jvp_args_tuple);
    auto shs = std::get<5>(jvp_args_tuple);
    const bool* clamped = std::get<6>(jvp_args_tuple);
    auto dL_dcolor = std::get<7>(jvp_args_tuple);
    auto dL_dmeans = std::get<8>(jvp_args_tuple);
    auto dL_dshs = std::get<9>(jvp_args_tuple);


    // Compute intermediate values, as it is done during forward
    auto pos = means[idx];
    auto dir_orig = pos - campos;
    auto dir = dir_orig / FLOATGRAD::length(dir_orig);

    using Vec3Type = std::conditional_t<is_float_grad<decltype(dir)>::value
                                          || is_float_grad<decltype(shs)>::value,
                                          FloatGrad<glm::vec3>, glm::vec3>;
    auto sh = cast<glm::vec3>(shs) + idx * max_coeffs;

    // Use PyTorch rule for clamping: if clamping was applied,
    // gradient becomes 0.
    Vec3Type dL_dRGB = dL_dcolor[idx];
    dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
    dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
    dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

    Vec3Type dRGBdx(0.0f, 0.0f, 0.0f);
    Vec3Type dRGBdy(0.0f, 0.0f, 0.0f);
    Vec3Type dRGBdz(0.0f, 0.0f, 0.0f);
    auto x = dir.x;
    auto y = dir.y;
    auto z = dir.z;

    // Target location for this Gaussian to write SH gradients to
    auto dL_dsh = dL_dshs + idx * max_coeffs;

    // No tricks here, just high school-level calculus.
    auto dRGBdsh0 = SH_C0;
    dL_dsh[0] = dRGBdsh0 * dL_dRGB;

    if (deg > 0)
    {
        auto dRGBdsh1 = -SH_C1 * y;
        auto dRGBdsh2 = SH_C1 * z;
        auto dRGBdsh3 = -SH_C1 * x;
        dL_dsh[1] = dRGBdsh1 * dL_dRGB;
        dL_dsh[2] = dRGBdsh2 * dL_dRGB;
        dL_dsh[3] = dRGBdsh3 * dL_dRGB;

        dRGBdx = -SH_C1 * sh[3];
        dRGBdy = -SH_C1 * sh[1];
        dRGBdz = SH_C1 * sh[2];

        if (deg > 1)
        {
            auto xx = x * x, yy = y * y, zz = z * z;
            auto xy = x * y, yz = y * z, xz = x * z;

            auto dRGBdsh4 = SH_C2[0] * xy;
            auto dRGBdsh5 = SH_C2[1] * yz;
            auto dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
            auto dRGBdsh7 = SH_C2[3] * xz;
            auto dRGBdsh8 = SH_C2[4] * (xx - yy);
            dL_dsh[4] = dRGBdsh4 * dL_dRGB;
            dL_dsh[5] = dRGBdsh5 * dL_dRGB;
            dL_dsh[6] = dRGBdsh6 * dL_dRGB;
            dL_dsh[7] = dRGBdsh7 * dL_dRGB;
            dL_dsh[8] = dRGBdsh8 * dL_dRGB;

            dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
            dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
            dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

            if (deg > 2)
            {
                auto dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
                auto dRGBdsh10 = SH_C3[1] * xy * z;
                auto dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
                auto dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
                auto dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
                auto dRGBdsh14 = SH_C3[5] * z * (xx - yy);
                auto dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
                dL_dsh[9] = dRGBdsh9 * dL_dRGB;
                dL_dsh[10] = dRGBdsh10 * dL_dRGB;
                dL_dsh[11] = dRGBdsh11 * dL_dRGB;
                dL_dsh[12] = dRGBdsh12 * dL_dRGB;
                dL_dsh[13] = dRGBdsh13 * dL_dRGB;
                dL_dsh[14] = dRGBdsh14 * dL_dRGB;
                dL_dsh[15] = dRGBdsh15 * dL_dRGB;

                dRGBdx += (
                    SH_C3[0] * sh[9] * 3.f * 2.f * xy +
                    SH_C3[1] * sh[10] * yz +
                    SH_C3[2] * sh[11] * -2.f * xy +
                    SH_C3[3] * sh[12] * -3.f * 2.f * xz +
                    SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
                    SH_C3[5] * sh[14] * 2.f * xz +
                    SH_C3[6] * sh[15] * 3.f * (xx - yy));

                dRGBdy += (
                    SH_C3[0] * sh[9] * 3.f * (xx - yy) +
                    SH_C3[1] * sh[10] * xz +
                    SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
                    SH_C3[3] * sh[12] * -3.f * 2.f * yz +
                    SH_C3[4] * sh[13] * -2.f * xy +
                    SH_C3[5] * sh[14] * -2.f * yz +
                    SH_C3[6] * sh[15] * -3.f * 2.f * xy);

                dRGBdz += (
                    SH_C3[1] * sh[10] * xy +
                    SH_C3[2] * sh[11] * 4.f * 2.f * yz +
                    SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
                    SH_C3[4] * sh[13] * 4.f * 2.f * xz +
                    SH_C3[5] * sh[14] * (xx - yy));
            }
        }
    }

    // The view direction is an input to the computation. View direction
    // is influenced by the Gaussian's mean, so SHs gradients
    // must propagate back into 3D position.
    Vec3Type dL_ddir(FLOATGRAD::dot(dRGBdx, dL_dRGB), FLOATGRAD::dot(dRGBdy, dL_dRGB), FLOATGRAD::dot(dRGBdz, dL_dRGB));

    using Float3Type = std::conditional_t<is_float_grad<decltype(dir_orig)>::value
                                          || is_float_grad<decltype(dL_ddir)>::value,
                                          FloatGrad<float3>, float3>;

    // Account for normalization of direction
    Float3Type dL_dmean = dnormvdv(make_float3( dir_orig.x, dir_orig.y, dir_orig.z ), make_float3( dL_ddir.x, dL_ddir.y, dL_ddir.z ));

    // Gradients of loss w.r.t. Gaussian means, but only the portion 
    // that is caused because the mean affects the view-dependent color.
    // Additional mean gradient is accumulated in below methods.
    dL_dmeans[idx] += Vec3Type(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
template <typename TupleType>
__global__ void computeCov2DCUDA(TupleType jvp_args_tuple)
    // int P, const float3* means, const int* radii, const float* cov3Ds,
    // const float h_x, float h_y, const float tan_fovx, float tan_fovy,
    // const float* view_matrix, const float* opacities, const float* dL_dconics,
    // float* dL_dopacity, const float* dL_dinvdepth, float3* dL_dmeans,
    // float* dL_dcov, bool antialiasing
{
    int P = std::get<0>(jvp_args_tuple);
    auto means = std::get<1>(jvp_args_tuple);
    const int* radii = std::get<2>(jvp_args_tuple);
    auto cov3Ds = std::get<3>(jvp_args_tuple);
    auto h_x = std::get<4>(jvp_args_tuple);
    auto h_y = std::get<5>(jvp_args_tuple);
    auto tan_fovx = std::get<6>(jvp_args_tuple);
    auto tan_fovy = std::get<7>(jvp_args_tuple);
    auto view_matrix = std::get<8>(jvp_args_tuple);
    auto opacities = std::get<9>(jvp_args_tuple);
    auto dL_dconics = std::get<10>(jvp_args_tuple);
    auto dL_dopacity = std::get<11>(jvp_args_tuple);
    auto dL_dinvdepth = std::get<12>(jvp_args_tuple);
    auto dL_dmeans = std::get<13>(jvp_args_tuple);
    auto dL_dcov = std::get<14>(jvp_args_tuple);
    bool antialiasing = std::get<15>(jvp_args_tuple);

    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !(radii[idx] > 0))
        return;

    // Reading location of 3D covariance for this Gaussian
    auto cov3D = cov3Ds + 6 * idx;

    // Fetch gradients, recompute 2D covariance and relevant 
    // intermediate forward results needed in the backward.
    auto mean = means[idx];
    auto dL_dconic = make_float3(dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3]);
    auto t = transformPoint4x3(mean, view_matrix);
    
    auto limx = 1.3f * tan_fovx;
    auto limy = 1.3f * tan_fovy;
    auto txtz = t.x / t.z;
    auto tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;
    
    auto x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
    auto y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

    using Mat3Type = std::conditional_t<is_float_grad<decltype(t)>::value
                                        || is_float_grad<decltype(h_x)>::value
                                        || is_float_grad<decltype(h_y)>::value
                                        || is_float_grad<decltype(view_matrix)>::value
                                        || is_float_grad<decltype(cov3D)>::value,
                                        FloatGrad<glm::mat3>, glm::mat3>;

    Mat3Type J = Mat3Type(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
        0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
        0, 0, 0);

    Mat3Type W = Mat3Type(
        view_matrix[0], view_matrix[4], view_matrix[8],
        view_matrix[1], view_matrix[5], view_matrix[9],
        view_matrix[2], view_matrix[6], view_matrix[10]);

    Mat3Type Vrk = Mat3Type(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]);

    Mat3Type T = W * J;

    Mat3Type cov2D = FLOATGRAD::transpose(T) * FLOATGRAD::transpose(Vrk) * T;

    // Use helper variables for 2D covariance entries. More compact.
    auto c_xx = cov2D[0][0];
    auto c_xy = cov2D[0][1];
    auto c_yy = cov2D[1][1];
    
    constexpr float h_var = 0.3f;
    FloatGrad<float> d_inside_root = 0.f;
    if(antialiasing)
    {
        auto det_cov = c_xx * c_yy - c_xy * c_xy;
        c_xx += h_var;
        c_yy += h_var;
        auto det_cov_plus_h_cov = c_xx * c_yy - c_xy * c_xy;
        auto h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability
        auto dL_dopacity_v = dL_dopacity[idx];
        auto d_h_convolution_scaling = dL_dopacity_v * opacities[idx];
        dL_dopacity[idx] = dL_dopacity_v * h_convolution_scaling;
        d_inside_root = (det_cov / det_cov_plus_h_cov) <= 0.000025f ? 0.f : d_h_convolution_scaling / (2 * h_convolution_scaling);
    } 
    else
    {
        c_xx += h_var;
        c_yy += h_var;
    }
    
    FloatGrad<float> dL_dc_xx = 0.0f;
    FloatGrad<float> dL_dc_xy = 0.0f;
    FloatGrad<float> dL_dc_yy = 0.0f;
    if(antialiasing)
    {
        // https://www.wolframalpha.com/input?i=d+%28%28x*y+-+z%5E2%29%2F%28%28x%2Bw%29*%28y%2Bw%29+-+z%5E2%29%29+%2Fdx
        // https://www.wolframalpha.com/input?i=d+%28%28x*y+-+z%5E2%29%2F%28%28x%2Bw%29*%28y%2Bw%29+-+z%5E2%29%29+%2Fdz
        auto x = c_xx;
        auto y = c_yy;
        auto z = c_xy;
        auto w = h_var;
        auto denom_f = d_inside_root / sq(w * w + w * (x + y) + x * y - z * z);
        auto dL_dx = w * (w * y + y * y + z * z) * denom_f;
        auto dL_dy = w * (w * x + x * x + z * z) * denom_f;
        auto dL_dz = -2.f * w * z * (w + x + y) * denom_f;
        dL_dc_xx = dL_dx;
        dL_dc_yy = dL_dy;
        dL_dc_xy = dL_dz;
    }
    
    auto denom = c_xx * c_yy - c_xy * c_xy;

    auto denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

    if (denom2inv != 0)
    {
        // Gradients of loss w.r.t. entries of 2D covariance matrix,
        // given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
        // e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
        
        // dL_dc_xx += denom2inv * ((tmp1 + tmp2) + tmp3);
        dL_dc_xx += denom2inv * (-c_yy * c_yy * dL_dconic.x + 2 * c_xy * c_yy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.z);
        dL_dc_yy += denom2inv * (-c_xx * c_xx * dL_dconic.z + 2 * c_xx * c_xy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.x);
        dL_dc_xy += denom2inv * 2 * (c_xy * c_yy * dL_dconic.x - (denom + 2 * c_xy * c_xy) * dL_dconic.y + c_xx * c_xy * dL_dconic.z);
        
        // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
        // given gradients w.r.t. 2D covariance matrix (diagonal).
        // cov2D = transpose(T) * transpose(Vrk) * T;
        dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_dc_xx + T[0][0] * T[1][0] * dL_dc_xy + T[1][0] * T[1][0] * dL_dc_yy);
        dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_dc_xx + T[0][1] * T[1][1] * dL_dc_xy + T[1][1] * T[1][1] * dL_dc_yy);
        dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_dc_xx + T[0][2] * T[1][2] * dL_dc_xy + T[1][2] * T[1][2] * dL_dc_yy);
        
        // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
        // given gradients w.r.t. 2D covariance matrix (off-diagonal).
        // Off-diagonal elements appear twice --> double the gradient.
        // cov2D = transpose(T) * transpose(Vrk) * T;
        dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_dc_xx + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_dc_xy + 2 * T[1][0] * T[1][1] * dL_dc_yy;
        dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_dc_xx + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_dc_xy + 2 * T[1][0] * T[1][2] * dL_dc_yy;
        dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_dc_xx + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_dc_xy + 2 * T[1][1] * T[1][2] * dL_dc_yy;
    }
    else
    {
        for (int i = 0; i < 6; i++)
            dL_dcov[6 * idx + i] = 0;
    }

    // Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
    // cov2D = transpose(T) * transpose(Vrk) * T;
    auto dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_dc_xx +
    (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc_xy;
    auto dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_dc_xx +
    (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc_xy;
    auto dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_dc_xx +
    (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc_xy;
    auto dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc_yy +
    (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_dc_xy;
    auto dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc_yy +
    (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_dc_xy;
    auto dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc_yy +
    (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_dc_xy;

    // Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
    // T = W * J
    auto dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
    auto dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
    auto dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
    auto dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

    auto tz = 1.f / t.z;
    auto tz2 = tz * tz;
    auto tz3 = tz2 * tz;

    // Gradients of loss w.r.t. transformed Gaussian mean t
    auto dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
    auto dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
    auto dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;
    // Account for inverse depth gradients
    if (dL_dinvdepth.data_ptr())
        dL_dtz -= dL_dinvdepth[idx] / (t.z * t.z);


    // Account for transformation of mean to t
    // t = transformPoint4x3(mean, view_matrix);
    auto dL_dmean = transformVec4x3Transpose(make_float3(dL_dtx, dL_dty, dL_dtz), view_matrix);

    // Gradients of loss w.r.t. Gaussian means, but only the portion 
    // that is caused because the mean affects the covariance matrix.
    // Additional mean gradient is accumulated in BACKWARD::preprocess.
    cast<float3>(dL_dmeans)[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
template <typename... JvpArgs>
__device__ void computeCov3D(JvpArgs&&... jvp_args)
    // int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, 
    // const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots
{
    auto jvp_args_tuple = std::forward_as_tuple(std::forward<JvpArgs>(jvp_args)...);
    int idx = std::get<0>(jvp_args_tuple);
    auto scale = std::get<1>(jvp_args_tuple);
    auto mod = std::get<2>(jvp_args_tuple);
    auto rot = std::get<3>(jvp_args_tuple);
    auto dL_dcov3Ds = std::get<4>(jvp_args_tuple);
    auto dL_dscales = std::get<5>(jvp_args_tuple);
    auto dL_drots = std::get<6>(jvp_args_tuple);

    // Recompute (intermediate) results for the 3D covariance computation.
    auto q = rot;// / glm::length(rot);
    auto r = q.x;
    auto x = q.y;
    auto y = q.z;
    auto z = q.w;

    constexpr bool use_floatgrad = is_float_grad<decltype(scale)>::value
                                   || is_float_grad<decltype(mod)>::value
                                   || is_float_grad<decltype(rot)>::value
                                   || is_float_grad<decltype(dL_dcov3Ds)>::value;

    using Mat3Type = std::conditional_t<use_floatgrad,
                                        FloatGrad<glm::mat3>, glm::mat3>;
    using Vec3Type = std::conditional_t<use_floatgrad,
                                        FloatGrad<glm::vec3>, glm::vec3>;
    using Vec4Type = std::conditional_t<use_floatgrad,
                                        FloatGrad<glm::vec4>, glm::vec4>;
    using Float4Type = std::conditional_t<use_floatgrad,
                                          FloatGrad<float4>, float4>;

    Mat3Type R = Mat3Type(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    Mat3Type S = glm::mat3(1.0f);

    Vec3Type s = mod * scale;
    S[0][0] = s.x;
    S[1][1] = s.y;
    S[2][2] = s.z;

    Mat3Type M = S * R;

    auto dL_dcov3D = dL_dcov3Ds + 6 * idx;

    Vec3Type dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
    Vec3Type ounc = 0.5f * Vec3Type(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

    // Convert per-element covariance loss gradients to matrix form
    Mat3Type dL_dSigma = Mat3Type(
        dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
        0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
        0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
    );

    // Compute loss gradient w.r.t. matrix M
    // dSigma_dM = 2 * M
    Mat3Type dL_dM = 2.0f * M * dL_dSigma;

    Mat3Type Rt = FLOATGRAD::transpose(R);
    Mat3Type dL_dMt = FLOATGRAD::transpose(dL_dM);

    // Gradients of loss w.r.t. scale
    auto dL_dscale = dL_dscales[idx];
    dL_dscale.x = FLOATGRAD::dot(Rt[0], dL_dMt[0]);
    dL_dscale.y = FLOATGRAD::dot(Rt[1], dL_dMt[1]);
    dL_dscale.z = FLOATGRAD::dot(Rt[2], dL_dMt[2]);

    dL_dMt[0] *= s.x;
    dL_dMt[1] *= s.y;
    dL_dMt[2] *= s.z;

    // Gradients of loss w.r.t. normalized quaternion
    Vec4Type dL_dq;
    dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
    dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
    dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
    dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

    // Gradients of loss w.r.t. unnormalized quaternion
    auto dL_drot = cast<float4>(dL_drots + idx);
    *dL_drot = make_float4( dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w );//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C, typename TupleType>
__global__ void preprocessCUDAJvp(TupleType jvp_args_tuple)
    // int P, int D, int M, const float3* means, const int* radii,
    // const float* shs, const bool* clamped, const glm::vec3* scales,
    // const glm::vec4* rotations, const float scale_modifier,
    // const float* proj, const glm::vec3* campos, const float3* dL_dmean2D,
    // glm::vec3* dL_dmeans, float* dL_dcolor, float* dL_dcov3D, float* dL_dsh, 
    // glm::vec3* dL_dscale, glm::vec4* dL_drot, float* dL_dopacity
{
    auto P = std::get<0>(jvp_args_tuple);
    auto D = std::get<1>(jvp_args_tuple);
    auto M = std::get<2>(jvp_args_tuple);
    auto means = std::get<3>(jvp_args_tuple);
    const int* radii = std::get<4>(jvp_args_tuple);
    auto shs = std::get<5>(jvp_args_tuple);
    const bool* clamped = std::get<6>(jvp_args_tuple);
    auto scales = std::get<7>(jvp_args_tuple);
    auto rotations = std::get<8>(jvp_args_tuple);
    auto scale_modifier = std::get<9>(jvp_args_tuple);
    auto proj = std::get<10>(jvp_args_tuple);
    auto campos = std::get<11>(jvp_args_tuple);
    auto dL_dmean2D = std::get<12>(jvp_args_tuple);
    auto dL_dmeans = std::get<13>(jvp_args_tuple);
    auto dL_dcolor = std::get<14>(jvp_args_tuple);
    auto dL_dcov3D = std::get<15>(jvp_args_tuple);
    auto dL_dsh = std::get<16>(jvp_args_tuple);
    auto dL_dscale = std::get<17>(jvp_args_tuple);
    auto dL_drot = std::get<18>(jvp_args_tuple);
    auto dL_dopacity = std::get<19>(jvp_args_tuple);

    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !(radii[idx] > 0))
        return;

    constexpr bool use_floatgrad = is_float_grad<decltype(means)>::value
                                   || is_float_grad<decltype(shs)>::value
                                   || is_float_grad<decltype(scales)>::value
                                   || is_float_grad<decltype(rotations)>::value
                                   || is_float_grad<decltype(dL_dmean2D)>::value
                                   || is_float_grad<decltype(dL_dmeans)>::value
                                   || is_float_grad<decltype(dL_dcolor)>::value
                                   || is_float_grad<decltype(dL_dcov3D)>::value
                                   || is_float_grad<decltype(dL_dsh)>::value
                                   || is_float_grad<decltype(dL_dscale)>::value
                                   || is_float_grad<decltype(dL_drot)>::value;

    using Vec3Type = std::conditional_t<use_floatgrad,
                                        FloatGrad<glm::vec3>, glm::vec3>;

    auto m = means[idx];

    // Taking care of gradients from the screenspace points
    auto m_hom = transformPoint4x4(m, proj);
    auto m_w = 1.0f / (m_hom.w + 0.0000001f);

    // Compute loss gradient w.r.t. 3D means due to gradients of 2D means
    // from rendering procedure
    Vec3Type dL_dmean;
    auto mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
    auto mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
    dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
    dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
    dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

    // That's the second part of the mean gradient. Previous computation
    // of cov2D and following SH conversion also affects it.
    dL_dmeans[idx] += dL_dmean;

    // Compute gradient updates due to computing colors from SHs
    if (shs.data_ptr())
        computeColorFromSH(idx, D, M, cast<glm::vec3>(means), *campos, shs, clamped, cast<glm::vec3>(dL_dcolor), cast<glm::vec3>(dL_dmeans), cast<glm::vec3>(dL_dsh));

    // Compute gradient updates due to computing covariance from scale/rotation
    if (scales.data_ptr())
        computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(CudaRasterizer::BLOCK_X * CudaRasterizer::BLOCK_Y)
renderCUDAJvpSplit(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float* __restrict__ bg_color_data,
    const float* __restrict__ bg_color_grad,
    const float2* __restrict__ points_xy_image_data,
    const float2* __restrict__ points_xy_image_grad,
    const float4* __restrict__ conic_opacity_data,
    const float4* __restrict__ conic_opacity_grad,
    const float* __restrict__ colors_data,
    const float* __restrict__ colors_grad,
    const float* __restrict__ depths_data,
    const float* __restrict__ depths_grad,
    const float* __restrict__ final_Ts_data,
    const float* __restrict__ final_Ts_grad,
    const uint32_t* __restrict__ n_contrib,
    const float* __restrict__ dL_dpixels_data,
    const float* __restrict__ dL_dpixels_grad,
    const float* __restrict__ dL_invdepths_data,
    const float* __restrict__ dL_invdepths_grad,
    float3* __restrict__ dL_dmean2D_data,
    float3* __restrict__ dL_dmean2D_grad,
    float4* __restrict__ dL_dconic2D_data,
    float4* __restrict__ dL_dconic2D_grad,
    float* __restrict__ dL_dopacity_data,
    float* __restrict__ dL_dopacity_grad,
    float* __restrict__ dL_dcolors_data,
    float* __restrict__ dL_dcolors_grad,
    float* __restrict__ dL_dinvdepths_data,
    float* __restrict__ dL_dinvdepths_grad
)
{
    // We rasterize again. Compute necessary block info.
    auto block = cg::this_thread_block();
    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
    const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    const uint32_t pix_id = W * pix.y + pix.x;
    const float2 pixf = { (float)pix.x, (float)pix.y };

    const bool inside = pix.x < W&& pix.y < H;
    const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

    bool done = !inside;
    int toDo = range.y - range.x;

    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy_data[BLOCK_SIZE];
    __shared__ float2 collected_xy_grad[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity_data[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity_grad[BLOCK_SIZE];
    __shared__ float collected_colors_data[C * BLOCK_SIZE];
    __shared__ float collected_colors_grad[C * BLOCK_SIZE];
    __shared__ float collected_depths_data[BLOCK_SIZE];
    __shared__ float collected_depths_grad[BLOCK_SIZE];


    // In the forward, we stored the final value for T, the
    // product of all (1 - alpha) factors. 
    auto T_final = inside ? FloatGrad<float>(final_Ts_data[pix_id], final_Ts_grad[pix_id]) : 0.0f;
    auto T = T_final;

    // We start from the back. The ID of the last contributing
    // Gaussian is known from each pixel from the forward.
    uint32_t contributor = toDo;
    const int last_contributor = inside ? n_contrib[pix_id] : 0;

    float accum_rec_data[C] = { 0 };
    float accum_rec_grad[C] = { 0 };
    FloatGradArray<float> accum_rec(accum_rec_data, accum_rec_grad);
    float dL_dpixel_data[C];
    float dL_dpixel_grad[C];
    FloatGradArray<float> dL_dpixel(dL_dpixel_data, dL_dpixel_grad);
    FloatGrad<float> dL_invdepth = 0.0f;
    FloatGrad<float> accum_invdepth_rec = 0.0f;
    if (inside)
    {
        for (int i = 0; i < C; i++) {
            dL_dpixel[i] = FloatGrad<float>(dL_dpixels_data[i * H * W + pix_id], dL_dpixels_grad[i * H * W + pix_id]);
        }
        if(dL_invdepths_data) {
            dL_invdepth = FloatGrad<float>(dL_invdepths_data[pix_id], dL_invdepths_grad[pix_id]);
        }
    }

    FloatGrad<float> last_alpha = 0.0f;
    float last_color_data[C] = { 0 };
    float last_color_grad[C] = { 0 };
    FloatGradArray<float> last_color(last_color_data, last_color_grad);
    FloatGrad<float> last_invdepth = 0.0f;

    // Gradient of pixel coordinate w.r.t. normalized 
    // screen-space viewport corrdinates (-1 to 1)
    const float ddelx_dx = 0.5 * W;
    const float ddely_dy = 0.5 * H;

    // Traverse all Gaussians
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        // Load auxiliary data into shared memory, start in the BACK
        // and load them in revers order.
        block.sync();
        const int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y)
        {
            const int coll_id = point_list[range.y - progress - 1];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy_data[block.thread_rank()] = points_xy_image_data[coll_id];
            collected_xy_grad[block.thread_rank()] = points_xy_image_grad[coll_id];
            collected_conic_opacity_data[block.thread_rank()] = conic_opacity_data[coll_id];
            collected_conic_opacity_grad[block.thread_rank()] = conic_opacity_grad[coll_id];
            for (int i = 0; i < C; i++) {
                collected_colors_data[i * BLOCK_SIZE + block.thread_rank()] = colors_data[coll_id * C + i];
                collected_colors_grad[i * BLOCK_SIZE + block.thread_rank()] = colors_grad[coll_id * C + i];
            }

            if(dL_invdepths_data) {
                collected_depths_data[block.thread_rank()] = depths_data[coll_id];
                collected_depths_grad[block.thread_rank()] = depths_grad[coll_id];
            }
        }
        block.sync();

        // Iterate over Gaussians
        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
        {
            // Keep track of current Gaussian ID. Skip, if this one
            // is behind the last contributor for this pixel.
            contributor--;
            if (contributor >= last_contributor)
                continue;

            // Compute blending values, as before.
            FloatGrad<const float2> xy(collected_xy_data[j], collected_xy_grad[j]);
            FloatGrad<const float2> d = make_float2(xy.x - pixf.x, xy.y - pixf.y);
            FloatGrad<const float4> con_o(collected_conic_opacity_data[j], collected_conic_opacity_grad[j]);
            auto power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;

            if (power > 0.0f)
                continue;

            FloatGrad<float> G = exp(power);
            FloatGrad<float> alpha = min(0.99f, con_o.w * G);

            if (alpha < 1.0f / 255.0f)
                continue;

            T = T / (1.f - alpha);
            FloatGrad<float> dchannel_dcolor = alpha * T;

            // Propagate gradients to per-Gaussian colors and keep
            // gradients w.r.t. alpha (blending factor for a Gaussian/pixel
            // pair).
            FloatGrad<float> dL_dalpha = 0.0f;
            const int global_id = collected_id[j];
            for (int ch = 0; ch < C; ch++)
            {
                FloatGrad<const float> c(collected_colors_data[ch * BLOCK_SIZE + j], collected_colors_grad[ch * BLOCK_SIZE + j]);
                // Update last color (to be used in the next iteration)
                accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
                last_color[ch] = c;

                FloatGrad<const float> dL_dchannel = dL_dpixel[ch];
                dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
                // Update the gradients w.r.t. color of the Gaussian. 
                // Atomic, since this pixel is just one of potentially
                // many that were affected by this Gaussian.
                auto dL_dcolor = dchannel_dcolor * dL_dchannel;
                atomicAdd(&(dL_dcolors_data[global_id * C + ch]), get_data(dL_dcolor));
                atomicAdd(&(dL_dcolors_grad[global_id * C + ch]), get_grad(dL_dcolor));
            }
            // Propagate gradients from inverse depth to alphaas and
            // per Gaussian inverse depths
            if (dL_dinvdepths_data)
            {
                FloatGrad<const float> collected_depths_j(collected_depths_data[j], collected_depths_grad[j]);
                auto invd = 1.f / collected_depths_j;
                accum_invdepth_rec = last_alpha * last_invdepth + (1.f - last_alpha) * accum_invdepth_rec;
                last_invdepth = invd;
                dL_dalpha += (invd - accum_invdepth_rec) * dL_invdepth;
                auto dL_dinvdepths = dchannel_dcolor * dL_invdepth;

                atomicAdd(&(dL_dinvdepths_data[global_id]), get_data(dL_dinvdepths));
                atomicAdd(&(dL_dinvdepths_grad[global_id]), get_grad(dL_dinvdepths));
            }

            dL_dalpha *= T;
            // Update last alpha (to be used in the next iteration)
            last_alpha = alpha;

            // Account for fact that alpha also influences how much of
            // the background color is added if nothing left to blend
            FloatGrad<float> bg_dot_dpixel = 0.0f;
            for (int i = 0; i < C; i++) {
                FloatGrad<float> bg_color_i(bg_color_data[i], bg_color_grad[i]);
                bg_dot_dpixel += bg_color_i * dL_dpixel[i];
            }
            dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


            // Helpful reusable temporary variables
            auto dL_dG = con_o.w * dL_dalpha;
            auto gdx = G * d.x;
            auto gdy = G * d.y;
            auto dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
            auto dG_ddely = -gdy * con_o.z - gdx * con_o.y;

            // Update gradients w.r.t. 2D mean position of the Gaussian
            auto dL_dmean2D_x = dL_dG * dG_ddelx * ddelx_dx;
            auto dL_dmean2D_y = dL_dG * dG_ddely * ddely_dy;
            atomicAdd(&dL_dmean2D_data[global_id].x, get_data(dL_dmean2D_x));
            atomicAdd(&dL_dmean2D_grad[global_id].x, get_grad(dL_dmean2D_x));
            atomicAdd(&dL_dmean2D_data[global_id].y, get_data(dL_dmean2D_y));
            atomicAdd(&dL_dmean2D_grad[global_id].y, get_grad(dL_dmean2D_y));

            // Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
            auto dL_dconic2D_x = -0.5f * gdx * d.x * dL_dG;
            auto dL_dconic2D_y = -0.5f * gdx * d.y * dL_dG;
            auto dL_dconic2D_w = -0.5f * gdy * d.y * dL_dG;
            atomicAdd(&dL_dconic2D_data[global_id].x, get_data(dL_dconic2D_x));
            atomicAdd(&dL_dconic2D_grad[global_id].x, get_grad(dL_dconic2D_x));
            atomicAdd(&dL_dconic2D_data[global_id].y, get_data(dL_dconic2D_y));
            atomicAdd(&dL_dconic2D_grad[global_id].y, get_grad(dL_dconic2D_y));
            atomicAdd(&dL_dconic2D_data[global_id].w, get_data(dL_dconic2D_w));
            atomicAdd(&dL_dconic2D_grad[global_id].w, get_grad(dL_dconic2D_w));

            // Update gradients w.r.t. opacity of the Gaussian
            auto dL_dopacity = G * dL_dalpha;
            atomicAdd(&(dL_dopacity_data[global_id]), get_data(dL_dopacity));
            atomicAdd(&(dL_dopacity_grad[global_id]), get_grad(dL_dopacity));
        }
    }
}

template <typename... JvpArgs>
void preprocessJvp(JvpArgs&&... jvp_args)
    // int P, int D, int M, const float3* means3D, const int* radii,
    // const float* shs, const bool* clamped, const float* opacities,
    // const glm::vec3* scales, const glm::vec4* rotations,
    // const float scale_modifier, const float* cov3Ds,
    // const float* viewmatrix, const float* projmatrix,
    // const float focal_x, float focal_y,
    // const float tan_fovx, float tan_fovy,
    // const glm::vec3* campos, const float3* dL_dmean2D,
    // const float* dL_dconic, const float* dL_dinvdepth,
    // float* dL_dopacity, glm::vec3* dL_dmean3D,
    // float* dL_dcolor, float* dL_dcov3D, float* dL_dsh,
    // glm::vec3* dL_dscale, glm::vec4* dL_drot, bool antialiasing)
{
    auto jvp_args_tuple = std::make_tuple(std::forward<JvpArgs>(jvp_args)...);
    int P = std::get<0>(jvp_args_tuple);
    int D = std::get<1>(jvp_args_tuple);
    int M = std::get<2>(jvp_args_tuple);
    auto means3D = std::get<3>(jvp_args_tuple);
    const int* radii = std::get<4>(jvp_args_tuple);
    auto shs = std::get<5>(jvp_args_tuple);
    const bool* clamped = std::get<6>(jvp_args_tuple);
    auto opacities = std::get<7>(jvp_args_tuple);
    auto scales = std::get<8>(jvp_args_tuple);
    auto rotations = std::get<9>(jvp_args_tuple);
    auto scale_modifier = std::get<10>(jvp_args_tuple);
    auto cov3Ds = std::get<11>(jvp_args_tuple);
    auto viewmatrix = std::get<12>(jvp_args_tuple);
    auto projmatrix = std::get<13>(jvp_args_tuple);
    auto focal_x = std::get<14>(jvp_args_tuple);
    auto focal_y = std::get<15>(jvp_args_tuple);
    auto tan_fovx = std::get<16>(jvp_args_tuple);
    auto tan_fovy = std::get<17>(jvp_args_tuple);
    auto campos = std::get<18>(jvp_args_tuple);
    auto dL_dmean2D = std::get<19>(jvp_args_tuple);
    auto dL_dconic = std::get<20>(jvp_args_tuple);
    auto dL_dinvdepth = std::get<21>(jvp_args_tuple);
    auto dL_dopacity = std::get<22>(jvp_args_tuple);
    auto dL_dmean3D = std::get<23>(jvp_args_tuple);
    auto dL_dcolor = std::get<24>(jvp_args_tuple);
    auto dL_dcov3D = std::get<25>(jvp_args_tuple);
    auto dL_dsh = std::get<26>(jvp_args_tuple);
    auto dL_dscale = std::get<27>(jvp_args_tuple);
    auto dL_drot = std::get<28>(jvp_args_tuple);
    bool antialiasing = std::get<29>(jvp_args_tuple);

    // int P, const float3* means, const int* radii, const float* cov3Ds,
    // const float h_x, float h_y, const float tan_fovx, float tan_fovy,
    // const float* view_matrix, const float* opacities, const float* dL_dconics,
    // float* dL_dopacity, const float* dL_dinvdepth, float3* dL_dmeans,
    // float* dL_dcov, bool antialiasing
    auto compute_cov2D_args_tuple = std::make_tuple(
        P,
        means3D,
        radii,
        cov3Ds,
        focal_x,
        focal_y,
        tan_fovx,
        tan_fovy,
        viewmatrix,
        opacities,
        dL_dconic,
        dL_dopacity,
        dL_dinvdepth,
        dL_dmean3D,
        dL_dcov3D,
        antialiasing);

    // Propagate gradients for the path of 2D conic matrix computation. 
    // Somewhat long, thus it is its own kernel rather than being part of 
    // "preprocess". When done, loss gradient w.r.t. 3D means has been
    // modified and gradient w.r.t. 3D covariance matrix has been computed.    
    computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
        compute_cov2D_args_tuple
    );

    auto preprocess_args_tuple = std::make_tuple(
        P, D, M,
        means3D,
        radii,
        shs,
        clamped,
        scales,
        rotations,
        scale_modifier,
        projmatrix,
        campos,
        dL_dmean2D,
        dL_dmean3D,
        dL_dcolor,
        dL_dcov3D,
        dL_dsh,
        dL_dscale,
        dL_drot,
        dL_dopacity);

    // Propagate gradients for remaining steps: finish 3D mean gradients,
    // propagate color gradients to SH (if desireD), propagate 3D covariance
    // matrix gradients to scale and rotation.
    preprocessCUDAJvp<CudaRasterizer::NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
        preprocess_args_tuple
    );
}

template <typename... JvpArgs>
void renderJvp(JvpArgs&&... jvp_args) 
//     const dim3 grid, dim3 block,
//     const uint2* ranges,
//     const uint32_t* point_list,
//     int W, int H,
//     const float* bg_color,
//     const float2* means2D,
//     const float4* conic_opacity,
//     const float* colors,
//     const float* depths,
//     const float* final_Ts,
//     const uint32_t* n_contrib,
//     const float* dL_dpixels,
//     const float* dL_invdepths,
//     float3* dL_dmean2D,
//     float4* dL_dconic2D,
//     float* dL_dopacity,
//     float* dL_dcolors,
//     float* dL_dinvdepths);
{
    auto jvp_args_tuple = std::make_tuple(std::forward<JvpArgs>(jvp_args)...);
    const dim3 grid = std::get<0>(jvp_args_tuple);
    dim3 block = std::get<1>(jvp_args_tuple);
    const uint2* ranges = std::get<2>(jvp_args_tuple);
    const uint32_t* point_list = std::get<3>(jvp_args_tuple);
    int W = std::get<4>(jvp_args_tuple);
    int H = std::get<5>(jvp_args_tuple);
    auto bg_color = std::get<6>(jvp_args_tuple);
    auto means2D = std::get<7>(jvp_args_tuple);
    auto conic_opacity = std::get<8>(jvp_args_tuple);
    auto colors = std::get<9>(jvp_args_tuple);
    auto depths = std::get<10>(jvp_args_tuple);
    auto final_Ts = std::get<11>(jvp_args_tuple);
    const uint32_t* n_contrib = std::get<12>(jvp_args_tuple);
    auto dL_dpixels = std::get<13>(jvp_args_tuple);
    auto dL_invdepths = std::get<14>(jvp_args_tuple);
    auto dL_dmean2D = std::get<15>(jvp_args_tuple);
    auto dL_dconic2D = std::get<16>(jvp_args_tuple);
    auto dL_dopacity = std::get<17>(jvp_args_tuple);
    auto dL_dcolors = std::get<18>(jvp_args_tuple);
    auto dL_dinvdepths = std::get<19>(jvp_args_tuple);

    renderCUDAJvpSplit<CudaRasterizer::NUM_CHANNELS> << <grid, block >> >(
        ranges,
        point_list,
        W, H,
        bg_color.data_ptr(),
        bg_color.grad_ptr(),
        means2D.data_ptr(),
        means2D.grad_ptr(),
        conic_opacity.data_ptr(),
        conic_opacity.grad_ptr(),
        colors.data_ptr(),
        colors.grad_ptr(),
        depths.data_ptr(),
        depths.grad_ptr(),
        final_Ts.data_ptr(),
        final_Ts.grad_ptr(),
        n_contrib,
        dL_dpixels.data_ptr(),
        dL_dpixels.grad_ptr(),
        dL_invdepths.data_ptr(),
        dL_invdepths.grad_ptr(),
        dL_dmean2D.data_ptr(),
        dL_dmean2D.grad_ptr(),
        dL_dconic2D.data_ptr(),
        dL_dconic2D.grad_ptr(),
        dL_dopacity.data_ptr(),
        dL_dopacity.grad_ptr(),
        dL_dcolors.data_ptr(),
        dL_dcolors.grad_ptr(),
        dL_dinvdepths.data_ptr(),
        dL_dinvdepths.grad_ptr()
    );
}

} // namespace BACKWARD

#endif // BACKWARD_IMPL_H
