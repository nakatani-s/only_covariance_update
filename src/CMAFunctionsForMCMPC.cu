/*

*/
#include "../include/CMAFunctionsForMCMPC.cuh"

float CMA_mu_weight(Data1 *h_dFB, const int index_max )
{
    float regularation = 0.0f;
    float sum_weight = 0.0f;
    for(int index = 0; index < index_max; index++)
    {
        if(isnan(h_dFB[index].W))
        {
            regularation += 0.0f;
        }else{
            regularation += h_dFB[index].W; 
        }
    }
    float temp = 0.0f;
    for(int index = 0; index < index_max; index++)
    {
        if(isnan(h_dFB[index].W))
        {
            sum_weight += 0.0f;
        }else{
            temp = (h_dFB[index]).W / regularation;
            sum_weight += powf(temp,2); 
        }
    }

    return (1/sum_weight);

}

__global__ void CMA_check_symmetric(float *out, float *in)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if( blockIdx.x < threadIdx.x)
    {
        if(!(out[id] == in[id])){
            out[id] = in[id];
        }
    }
}

void CMA_evolutional_path_Ps(float *out, float *inv_C, float *dy, float mu, const float c_s)
{
    float temp[HORIZON] = { };
    int mat_index = 0;
    float constatnt;
    if(isnan(c_s*(2-c_s)*mu)){
        constatnt = 1.0f;
    }else{
        constatnt = sqrtf(c_s*(2-c_s)*mu);
    }
    for(int index = 0; index < HORIZON; index++){
        mat_index = index * HORIZON;
        for(int columun = 0; columun < HORIZON; columun++){
            if(isnan(inv_C[mat_index+columun] + dy[columun])){
                temp[index] += 0.0f;
            }else{
                temp[index] += inv_C[mat_index+columun] * dy[columun];
            }
        }
        out[index] = (1 - c_s) * out[index] + constatnt * temp[index];
        //printf("Cc == %f  const == %f  temp[%d] == %f\n", (1 - c_s), constatnt, index,temp[index]);
    }
}

void CMA_evolutional_path_Pc(float *out, float *dy, float mu_w, const float c_s)
{
    float temp[HORIZON] = { };
    //int mat_index = 0;
    float constatnt;
    if(isnan(c_s*(2-c_s)*mu_w)){
        constatnt = 1.0f;
    }else{
        constatnt = sqrtf(c_s*(2-c_s)*mu_w);
    }
    for(int index = 0; index < HORIZON; index++){
        for(int columun = 0; columun < HORIZON; columun++){
            if(isnan(dy[columun])){
                temp[index] += 0.0f;
            }else{
                temp[index] += dy[columun];
            }
        }
        out[index] = (1 - c_s) * out[index] + constatnt * temp[index];
    }
}

void CMA_weighted_mean(Data1 *h_Data, const int top, float *Dy_host)
{
    float total_weight = 0.0f;
    float temp[HORIZON] = {};
    for(int i = 0; i < top; i++){
        if(isnan(h_Data[i].W))
        {
            total_weight += 0.0f;
        }else{
            total_weight += h_Data[i].W;
        }
    }

    for(int i = 0; i < HORIZON; i++)
    {
        for(int k = 0; k < top; k++)
        {
            if(isnan(h_Data[k].W))
            {
                temp[i] += 0.0f;
            }else{
                temp[i] += h_Data[k].W * h_Data[k].dy[i] / total_weight;
            }
            if(isnan(temp[i]))
            {
               Dy_host[i] = 0.0f;
            }else{
               Dy_host[i] = temp[i];
            } 
        }
    }
}

float CMA_update_variance( float *Us, const float c_sig, const float d_sig, const float func_Xi, float var)
{
    float a[8] = { };
    a[0] = c_sigma / d_sigma;
    for(int index = 0; index < HORIZON; index++){
        a[1] += powf( Us[index], 2);
    }
    a[2] = sqrtf(a[1]);
    a[3] = a[2] / func_Xi;
    a[4] = a[3] - 1.0f;
    a[5] = a[0] * a[4];
    a[6] = exp(a[5]);
    a[7] = var * a[6];

    return a[7];
}

__global__ void CMA_path_sigma_tensor(float *d_cov, Input_vec *vector_P, int index_vec)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    //float temp;
    if(isnan(vector_P[index_vec].dy[threadIdx.x] + vector_P[index_vec].dy[blockIdx.x])){
        d_cov[id] = 0.0f;
    }else{
        d_cov[id] = vector_P[index_vec].dy[threadIdx.x] * vector_P[index_vec].dy[blockIdx.x];
    }
}

__global__ void CMA_path_pc_tensor(float *out, float *vector)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    //float temp;
    if(isnan(vector[threadIdx.x] + vector[blockIdx.x])){
        out[id] = 0.0f;
    }else{
        out[id] = vector[threadIdx.x] * vector[blockIdx.x];
    }
}

__global__ void CMA_zeros_matrix(float *Mat)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    Mat[id] = 0.0f;
}

float CMA_mat_weight(Data1 *vec_P, const int Num, int No)
{
    float ret = 0.0f;
    float total_here = 0.0f;
    for(int i = 0; i < Num; i++){
        if(isnan(vec_P[i].W))
        {
            total_here += 0.0f;
        }else{
            total_here += vec_P[i].W;
        }
    }
    ret = vec_P[No].W / total_here;
    return ret;
}


__global__ void CMA_matrix_sum(float *out, float *tensor, float *prev_cma, Input_vec *vector_P, const int el, int No)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    float weight_here;
    for(int i = 0; i < el; i++){
        if(isnan(vector_P[i].W))
        {
            weight_here += 0.0f;
        }else{
            weight_here += vector_P[i].W;
        }
    }
    float here_w;
    here_w = vector_P[No].W / weight_here;
    __syncthreads();

    out[id] += here_w * (tensor[id] - prev_cma[id]);
}

__global__ void CMA_matrix_sum_renew(float *out, float *tensor, float *prev_cma, float weight)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    out[id] += weight * (tensor[id] - prev_cma[id]);
}
__global__ void CMA_matrix_Difference(float *out, float *left, float *right)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    out[id] = left[id] - right[id];
}

__global__ void CMA_update_covariance_matrix(float *out, float *third, float *second, float c1, float c_mu)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    out[id] = out[id] + c1 * second[id] +  c_mu * third[id];
}

void CMA_estimate_mean(float *Us_hst, float *Dy_hst, float var)
{
    for(int t = 0; t < HORIZON; t++)
    {
        Us_hst[t] = Us_hst[t] + var * Dy_hst[t];
        if(Us_hst[t] < -1.0f){
           Us_hst[t] = -1.0f;
        }
        if(Us_hst[t] > 1.0f){
           Us_hst[t] = 1.0f;
        }
    }
}
