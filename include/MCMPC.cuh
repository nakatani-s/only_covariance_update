/* 
    MCMPC.cuh
*/
#include<cuda.h>
#include<curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include "params.cuh"
#include "DataStructure.cuh"
#include "dynamics.cuh"


__constant__ float  d_constraints[NUM_CONST], d_matrix[dim_weight_matrix]/*, d_hat_Q[dim_hat_Q], d_param[dim_param]*/;
__shared__ float W_comp[THREAD_PER_BLOCKS], L_comp[THREAD_PER_BLOCKS], values[HORIZON];
__shared__ int best_thread_id_this_block;



// #ifndef MCMPC_CUH
// #define MCMPC_CUH
__global__ void setup_kernel(curandState *state,int seed);
__global__ void setup_init_Covariance(float *Mat);
unsigned int countBlocks(unsigned int a, unsigned int b);
void weighted_mean(Data1 *h_Data, int Blocks, float *Us_host);
//__global__ void MCMPC_GPU_Linear_Example(float *state, curandState *devs, Data1 *d_Datas, float var, int Blocks, float *d_cov);
//__global__ void MCMPC_GPU_Linear_Example(float x, float y, float w, curandState *devs, Data1 *d_Datas, float var, int Blocks, float *d_cov);
__global__ void MCMPC_GPU_Linear_Example(float x, float y, float w, curandState *devs, Data1 *d_Datas, float var, int Blocks, float *d_cov, float *d_param, float *d_matrix);
__global__ void shift_Input_vec(Input_vec *dst, float *dev_Us);


#ifdef USING_THRUST

//thrust::device_vector<int> indices_device_vec;
//thrust::device_vector<float> cost_device_vec_for_sorting;
__global__ void Using_Thrust_MCMPC_Linear(float x, float y, float w, curandState *devs, Input_vec *d_Datas, float var, int Blocks, float *d_cov, float *d_param, float *d_matrix, float *cost_vec);
__global__ void Using_Thrust_MCMPC_Pendulum(float x, float th, float dx, float dth, curandState *devs, Input_vec *d_Datas, float var, int Blocks, float *d_cov, float *d_param, float *d_const, float *d_matrix, float *cost_vec);
__global__ void set_Input_vec(Input_vec *d_Input_vec, float init);
__global__ void reset_Input_vec(Input_vec *d_Input_vec, float *opt);
__global__ void callback_elite_sample(Data1 *d_Datas, Input_vec *dst, int *elite_indices);
#endif
//__global__ void Using_Thrust_MCMPC_Linear(float x, float y, float w, curandState *devs, Data1 *d_Datas, float var, int Blocks, float *d_cov, float *d_param, float *d_matrix, float *cost_vec);
// #endif
