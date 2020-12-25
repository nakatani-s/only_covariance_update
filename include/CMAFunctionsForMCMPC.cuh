/*

*/
#include <stdio.h>
#include<cuda.h>
#include<math.h>
#include "params.cuh"
#include "DataStructure.cuh"

float CMA_mu_weight(Data1 *h_dFB, const int index_max );
float CMA_update_variance( float *Us, const float c_sig, const float d_sig, const float func_Xi, float var);
void CMA_evolutional_path_Ps(float *out, float *inv_C, float *dy, float mu, const float c_s);
void CMA_evolutional_path_Pc(float *out, float *dy, float mu_w, const float c_s);
void CMA_estimate_mean(float *Us_hst, float *Dy_hst, float var);
float CMA_mat_weight(Data1 *vec_P, const int Num, int No);

void CMA_weighted_mean(Data1 *h_Data, const int top, float *Dy_host);

__global__ void CMA_check_symmetric(float *out, float *in);
__global__ void CMA_path_sigma_tensor(float *d_cov, Input_vec *vector_P, int index_vec);
__global__ void CMA_zeros_matrix(float *Mat);
__global__ void CMA_matrix_sum(float *out, float *tensor, float *prev_cma, Input_vec *vector_P, const int el, int No);
__global__ void CMA_path_pc_tensor(float *out, float *vector);
__global__ void CMA_matrix_Difference(float *out, float *left, float *right);
__global__ void CMA_update_covariance_matrix(float *out, float *third, float *second, float c1, float c_mu);
__global__ void CMA_matrix_sum_renew(float *out, float *tensor, float *prev_cma, float weight);
