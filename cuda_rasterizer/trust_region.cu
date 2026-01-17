#include "trust_region.h"
#include "auxiliary.h"

__forceinline__ __device__ glm::vec4 normalize_quat(
    const glm::vec4& quat_tilde)
{
    float norm = glm::length(quat_tilde);
    return quat_tilde / norm;
}

__forceinline__ __device__ glm::mat3 quat_to_rotation(
    const glm::vec4& quat)
{
    float w = quat[0];
    float x = quat[1];
    float y = quat[2];
    float z = quat[3];

    glm::mat3 R(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y + w * z), 2.f * (x * z - w * y),
        2.f * (x * y - w * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + w * x),
        2.f * (x * z + w * y), 2.f * (y * z - w * x), 1.f - 2.f * (x * x + y * y)
    );

    // glm::mat3 R(
    //     1.f - 2.f * (y * y + z * z), 2.f * (x * y - w * z), 2.f * (x * z + w * y),
    //     2.f * (x * y + w * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - w * x),
    //     2.f * (x * z - w * y), 2.f * (y * z + w * x), 1.f - 2.f * (x * x + y * y)
    // );

    return R;
}

__forceinline__ __device__ glm::mat3 quat_to_rot_unnormalized(
    const float w, const float x, const float y, const float z, const float q_sq)
{
    glm::mat3 R(
        q_sq - 2.f * (y * y + z * z), 2.f * (x * y + w * z), 2.f * (x * z - w * y),
        2.f * (x * y - w * z), q_sq - 2.f * (x * x + z * z), 2.f * (y * z + w * x),
        2.f * (x * z + w * y), 2.f * (y * z - w * x), q_sq - 2.f * (x * x + y * y)
    );

    // glm::mat3 R(
    //     q * q - 2.f * (y * y + z * z), 2.f * (x * y - w * z), 2.f * (x * z + w * y),
    //     2.f * (x * y + w * z), q * q - 2.f * (x * x + z * z), 2.f * (y * z - w * x),
    //     2.f * (x * z - w * y), 2.f * (y * z + w * x), q * q - 2.f * (x * x + y * y)
    // );

    return R;
}

__forceinline__ __device__ glm::mat3 quat_to_drot_unnormalized(
    const float w, const float x, const float y, const float z, const char wrt)
{
    glm::mat3 dR;

    switch(wrt) {
        case 'w':
            dR = glm::mat3(
                2 * w, 2 * z, -2 * y,
                -2 * z, 2 * w, 2 * x,
                2 * y, -2 * x, 2 * w
            );
            // dR = glm::mat3(
            //     2 * w, -2 * z, 2 * y,
            //     2 * z, 2 * w, -2 * x,
            //     -2 * y, 2 * x, 2 * w
            // );
            break;
        case 'x':
            dR = glm::mat3(
                2 * x, 2 * y, 2 * z,
                2 * y, -2 * x, 2 * w,
                2 * z, -2 * w, -2 * x
            );
            // dR = glm::mat3(
            //     2 * x, 2 * y, 2 * z,
            //     2 * y, -2 * x, -2 * w,
            //     2 * z, 2 * w, -2 * x
            // );
            break;
        case 'y':
            dR = glm::mat3(
                -2 * y, 2 * x, -2 * w,
                2 * x, 2 * y, 2 * z,
                2 * w, 2 * z, -2 * y
            );
            // dR = glm::mat3(
            //     -2 * y, 2 * x, 2 * w,
            //     2 * x, 2 * y, 2 * z,
            //     -2 * w, 2 * z, -2 * y
            // );
            break;
        case 'z':
            dR = glm::mat3(
                -2 * z, 2 * w, 2 * x,
                -2 * w, -2 * z, 2 * y,
                2 * x, 2 * y, 2 * z
            );
            // dR = glm::mat3(
            //     -2 * z, -2 * w, 2 * x,
            //     2 * w, -2 * z, 2 * y,
            //     2 * x, 2 * y, 2 * z
            // );
            break;
        default:
            dR = glm::mat3(0.0f);
    }
    return dR;
}

__forceinline__ __device__ glm::mat3 quat_to_d2rot_unnormalized(
    const float w, const float x, const float y, const float z, const char wrt)
{
    glm::mat3 d2R;

    switch(wrt) {
        case 'w':
            d2R = glm::mat3(
                2.0f, 0.0f, 0.0f,
                0.0f, 2.0f, 0.0f,
                0.0f, 0.0f, 2.0f
            );
            break;
        case 'x':
            d2R = glm::mat3(
                2.0f, 0.0f, 0.0f,
                0.0f, -2.0f, 0.0f,
                0.0f, 0.0f, -2.0f
            );
            break;
        case 'y':
            d2R = glm::mat3(
                -2.0f, 0.0f, 0.0f,
                0.0f, 2.0f, 0.0f,
                0.0f, 0.0f, -2.0f
            );
            break;
        case 'z':
            d2R = glm::mat3(
                -2.0f, 0.0f, 0.0f,
                0.0f, -2.0f, 0.0f,
                0.0f, 0.0f, 2.0f
            );
            break;
        default:
            d2R = glm::mat3(0.0f);
    }
    return d2R;
}

__forceinline__ __device__ glm::mat3 SMSinv(
    const glm::vec3& S,
    const glm::mat3& M)
{
    glm::mat3 SMSinv = M;

    // Do S * M * S^-1
    for(int r = 0; r < 3; r++) {
        for(int c = 0; c < 3; c++) {
            SMSinv[c][r] *= S[r] / S[c];
        }
    }
    return SMSinv;
}

__forceinline__ __device__ float frob_norm_squared(
    const glm::mat3& M)
{
    float norm_sq = 0.0f;
    for(int r = 0; r < 3; r++) {
        for(int c = 0; c < 3; c++) {
            norm_sq += M[c][r] * M[c][r];
        }
    }
    return norm_sq;
}

__forceinline__ __device__ float trace(
    const glm::mat3& M)
{
    return M[0][0] + M[1][1] + M[2][2];
}

__forceinline__ __device__ glm::mat3 build_covariance_from_scaling_rotation(
    const glm::vec3& scaling,
    const float scale_modifier,
    const glm::vec4& quat)
{
    glm::mat3 R = quat_to_rotation(quat);
    float sx = scaling.x * scale_modifier;
    float sy = scaling.y * scale_modifier;
    float sz = scaling.z * scale_modifier;
    R[0] *= sx;
    R[1] *= sy;
    R[2] *= sz;
    glm::mat3 covar = R * glm::transpose(R);
    return covar;
}


__forceinline__ __device__ glm::mat3 build_inverse_covariance_from_scaling_rotation(
    const glm::vec3& scaling,
    const float scale_modifier,
    const glm::vec4& quat)
{
    glm::mat3 R = quat_to_rotation(quat);
    float sx = scaling.x * scale_modifier;
    float sy = scaling.y * scale_modifier;
    float sz = scaling.z * scale_modifier;
    R[0] /= sx;
    R[1] /= sy;
    R[2] /= sz;
    glm::mat3 covar_inv = R * glm::transpose(R);
    return covar_inv;
}

__forceinline__ __device__ glm::vec4 compute_quat_to_trace_coefficient(
    const glm::vec4& quat_tilde, 
    const glm::vec3& scaling,
    bool debug=false)
{
    float w = quat_tilde[0];
    float x = quat_tilde[1];
    float y = quat_tilde[2];
    float z = quat_tilde[3];
    float q_sq = x * x + y * y + z * z + w * w;
    float q_4 = q_sq * q_sq;
    float q_6 = q_4 * q_sq;

    glm::mat3 R_tilde = glm::transpose(quat_to_rot_unnormalized(w, x, y, z, q_sq));
    glm::mat3 R = R_tilde / q_sq;

    glm::vec4 coeffs(0.0f);

    char wrts[4] = {'w', 'x', 'y', 'z'};
    float ts[4] = {w, x, y, z};

    for(int i = 0; i < 4; i++) {
        char wrt = wrts[i];
        float t = ts[i];
        float t2 = t * t;

        glm::mat3 dR_tilde = glm::transpose(quat_to_drot_unnormalized(w, x, y, z, wrt));
        glm::mat3 d2R_tilde = glm::transpose(quat_to_d2rot_unnormalized(w, x, y, z, wrt));

        glm::mat3 dG = glm::transpose(R) * (dR_tilde / q_sq - 2.0f * t * R_tilde / q_4);
        glm::mat3 d2G = glm::transpose(R) * (-2.0f * t * dR_tilde / q_4 + d2R_tilde / q_sq + 8.0f * t2 * R_tilde / q_6 - 2.0f * t * dR_tilde / q_4 - 2.0f * R_tilde / q_4);

        glm::mat3 SdGSinv = SMSinv(scaling, dG);

        float coeff = 2 * frob_norm_squared(SdGSinv) + 2 * trace(d2G);
        coeffs[i] = coeff;

        // // DEBUG
        // if(debug && i == 0) {
        //     printf("wrt = %c, coeff = %f\n", wrt, coeff);
        //     printf("w: %f, x: %f, y: %f, z: %f\n", w, x, y, z);
        //     printf("q_sq: %f, q_4: %f, q_6: %f\n", q_sq, q_4, q_6);
        //     printf("R:\n");
        //     for(int r = 0; r < 3; r++) {
        //         for(int c = 0; c < 3; c++) {
        //             printf("%f ", R[c][r]);
        //         }
        //         printf("\n");
        //     }
        //     printf("R_tilde:\n");
        //     for(int r = 0; r < 3; r++) {
        //         for(int c = 0; c < 3; c++) {
        //             printf("%f ", R_tilde[c][r]);
        //         }
        //         printf("\n");
        //     }
        //     printf("dR_tilde:\n");
        //     for(int r = 0; r < 3; r++) {
        //         for(int c = 0; c < 3; c++) {
        //             printf("%f ", dR_tilde[c][r]);
        //         }
        //         printf("\n");
        //     }
        //     printf("d2R_tilde:\n");
        //     for(int r = 0; r < 3; r++) {
        //         for(int c = 0; c < 3; c++) {
        //             printf("%f ", d2R_tilde[c][r]);
        //         }
        //         printf("\n");
        //     }
        //     printf("dG:\n");
        //     for(int r = 0; r < 3; r++) {
        //         for(int c = 0; c < 3; c++) {
        //             printf("%f ", dG[c][r]);
        //         }
        //         printf("\n");
        //     }
        //     printf("SdGSinv:\n");
        //     for(int r = 0; r < 3; r++) {
        //         for(int c = 0; c < 3; c++) {
        //             printf("%f ", SdGSinv[c][r]);
        //         }
        //         printf("\n");
        //     }
        //     printf("d2G:\n");
        //     for(int r = 0; r < 3; r++) {
        //         for(int c = 0; c < 3; c++) {
        //             printf("%f ", d2G[c][r]);
        //         }
        //         printf("\n");
        //     }
        //     glm::mat3 temp1 = dR_tilde / q_sq;
        //     printf("(dR_tilde / q_sq):\n");
        //     for(int r = 0; r < 3; r++) {
        //         for(int c = 0; c < 3; c++) {
        //             printf("%f ", temp1[c][r]);
        //         }
        //         printf("\n");
        //     }
        //     temp1 =  t * R_tilde / q_4;
        //     printf("(t * R_tilde / q_4):\n");
        //     for(int r = 0; r < 3; r++) {
        //         for(int c = 0; c < 3; c++) {
        //             printf("%f ", temp1[c][r]);
        //         }
        //         printf("\n");
        //     }
        //     printf("frob_norm_squared(SdGSinv): %f\n", frob_norm_squared(SdGSinv));
        //     printf("trace(d2G): %f\n", trace(d2G));
        // }
        
    }

    coeffs = max(coeffs, glm::vec4(1e-20f));

    return coeffs;
}

__global__ void compute_trust_region_step(
    const int P, const int max_coeffs,
    const float trust_radius,
    const float opacity_trust_radius,
    const float min_opacity_scaling,
    const float max_opacity_scaling,
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
    float* shs_params_step)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= P) return;

    const float SH0 = 0.282;
    const float SH_rest = 1.0;
    const float features_dc_min = -0.5;
    glm::vec3 scaling_param = scaling_params[idx];
    glm::vec3 scaling = exp(scaling_param);
    glm::vec4 quat_tilde = quat_params[idx];
    glm::vec4 quat = normalize_quat(quat_tilde);
    float opacity_param = opacity_params[idx];
    float opacity = sigmoid(opacity_param);
    float opacity_scaling = max(min_opacity_scaling, min(max_opacity_scaling, opacity));

    glm::vec3* sh = ((glm::vec3*)shs_params) + idx * max_coeffs;
    glm::vec3 color = SH0 * sh[0] - features_dc_min;
    color = glm::max(color, glm::vec3(0.0f));
    glm::mat3 covar = build_covariance_from_scaling_rotation(scaling, scale_modifier, quat);

    // Position clip
    float tr_pos = trust_radius / 3.0 / opacity_scaling;
    glm::mat3 covar_inv = build_inverse_covariance_from_scaling_rotation(scaling, scale_modifier, quat);

    for(int i = 0; i < 3; i++) {
        float diag_inv = covar_inv[i][i];
        float log_thresh = min(log(max(1.0 - tr_pos, 1e-20)), 0.0);
        float pos_thresh = sqrt(-8.0 / diag_inv * log_thresh);
        float pos_step = xyz_params_step[idx * 3 + i];
        xyz_params_step[idx * 3 + i] = min(max(pos_step, -pos_thresh), pos_thresh);
    }


    // Scale clip
    float tr_scaling = max(trust_radius / 3.0 / opacity_scaling, 0.0);
    glm::vec3 scaling_thresh = sqrt((4.0f / 3.0f) * (scaling * scaling) * tr_scaling);
    glm::vec3 scaling_param_step = scaling_params_step[idx];
    glm::vec3 scaling_new = exp(scaling_param + scaling_param_step);
    scaling_new = min(max(scaling_new, scaling - scaling_thresh), scaling + scaling_thresh);
    scaling_params_step[idx] = log(scaling_new) - scaling_param;

    // Rotation threshold
    float tr_quat = trust_radius / 4.0 / opacity_scaling;
    glm::vec4 quat_coeffs = compute_quat_to_trace_coefficient(quat_tilde, scaling, idx == 0);
    glm::vec4 quat_thresh_hellinger = sqrt((-8.0f / quat_coeffs) * logf(max(1.0 - tr_quat, 1e-20)));
    float quat_thresh_norm = glm::length(quat_tilde) * quat_norm_tr;
    glm::vec4 quat_thresh = min(quat_thresh_hellinger, glm::vec4(quat_thresh_norm));
    glm::vec4 quat_param_step = quat_params_step[idx];
    quat_params_step[idx] = min(max(quat_param_step, -quat_thresh), quat_thresh);

    // Opacity threshold
    float opacity_thresh = sqrt(4.0 * opacity * opacity_trust_radius);
    float opacity_new = sigmoid(opacity_param + opacity_params_step[idx]);
    opacity_new = min(max(opacity_new, opacity - opacity_thresh), opacity + opacity_thresh);
    opacity_new = min(max(opacity_new, 1e-5f), 1.0f - 1e-5f);
    opacity_params_step[idx] = opacity_params_step[idx] == 0.0? 0.0 : inverse_sigmoid(opacity_new) - opacity_param; // Keep opacity step if 0.0

    // DC color threshold
    float tr_color_dc = trust_radius / 3.0 / opacity_scaling;
    glm::vec3 color_dc_thresh_hellinger = sqrt(4.0f * color * tr_color_dc) / SH0;
    // glm::vec3 color_dc_thresh_opacity = opacity >= 0.99? color_dc_thresh_hellinger
    //                                     : max((1.0f - color) / SH0, glm::vec3(0.0f));
    glm::vec3 upper_color_thresh = max((1.0f - color) / SH0, glm::vec3(0.0f));
    glm::vec3 lower_color_thresh = max((color - 0.0f) / SH0, glm::vec3(0.0f));
    glm::vec3 color_dc_thresh_upper = min(color_dc_thresh_hellinger, upper_color_thresh);
    glm::vec3 color_dc_thresh_lower = min(color_dc_thresh_hellinger, lower_color_thresh);
    glm::vec3* sh_param_step = ((glm::vec3*)shs_params_step) + idx * max_coeffs;
    glm::vec3 sh0_step = sh_param_step[0];

    sh_param_step[0] = min(max(sh0_step, -color_dc_thresh_lower), color_dc_thresh_upper);
    // Rest color threshold
    glm::vec3 color_rest_thresh = min(color_dc_thresh_upper, color_dc_thresh_lower) / 20.0f;
    for(int c = 1; c < 16; c++) {
        const float max_rest_coeff = 0.5f;
        glm::vec3 shc_params = sh[c];
        glm::vec3 upper_color_rest_thresh = max((max_rest_coeff - shc_params), glm::vec3(0.0f));
        glm::vec3 lower_color_rest_thresh = max((shc_params - (-max_rest_coeff)), glm::vec3(0.0f));
        glm::vec3 color_rest_thresh_upper = min(color_rest_thresh, upper_color_rest_thresh);
        glm::vec3 color_rest_thresh_lower = min(color_rest_thresh, lower_color_rest_thresh);

        glm::vec3 shc_step = sh_param_step[c];
        sh_param_step[c] = min(max(shc_step, -color_rest_thresh_lower), color_rest_thresh_upper);
    }
}

void TRUST_REGION::ComputeTrustRegionStep(
    const int P, const int M,
    const float trust_radius,
    const float opacity_trust_radius,
    const float min_opacity_scaling,
    const float max_opacity_scaling,
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
    float* shs_params_step)
{
    int num_blocks = (P + 255) / 256;
    dim3 block(256, 1, 1);
    dim3 grid(num_blocks, 1, 1);
    compute_trust_region_step<<<grid, block>>> (
         P, M,
         trust_radius, 
         opacity_trust_radius, 
         min_opacity_scaling, 
         max_opacity_scaling,
         quat_norm_tr,
         xyz_params, 
         scaling_params, 
         scale_modifier, 
         quat_params,
         opacity_params, 
         shs_params,
         xyz_params_step, 
         scaling_params_step,
         quat_params_step,
         opacity_params_step, 
         shs_params_step);
}
