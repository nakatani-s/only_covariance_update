/*
include "MCMPC.cuh"
*/
#include<stdio.h>
#include "../include/MCMPC.cuh" 

__global__ void setup_kernel(curandState *state,int seed) {
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence number,
     no offset */
    curand_init(seed, id, 0, &state[id]);
}

unsigned int countBlocks(unsigned int a, unsigned int b) {
	unsigned int num;
	num = a / b;
	if (a < b || a % b > 0)
		num++;
	return num;
}

void weighted_mean(Data1 *h_Data, int Blocks, float *Us_host)
{
    float total_weight = 0.0f;
    float temp[HORIZON] = {};
    for(int i = 0; i < Blocks; i++){
        if(isnan(h_Data[i].W))
        {
            total_weight += 0.0f;
        }else{
            total_weight += h_Data[i].W;
        }
    }

    for(int i = 0; i < HORIZON; i++)
    {
        for(int k = 0; k < Blocks; k++)
        {
            if(isnan(h_Data[k].W))
            {
                temp[i] += 0.0f;
            }else{
                temp[i] += h_Data[k].W * h_Data[k].Input[i] / total_weight;
            }
            if(isnan(temp[i]))
            {
               Us_host[i] = 0.0f;
            }else{
               Us_host[i] = temp[i];
            } 
        }
    }
}

__device__ float generate_u(int t, float mean, float var, float *d_cov, float *z)
{
    int count_index;
    count_index = t * HORIZON;
    float ret, sec_term;
    sec_term = 0; 
    for(int k = 0; k < HORIZON; k++)
    {
        
        sec_term += d_cov[count_index+k]*z[k];
        /*if(t == 0 && k == 0){
             sec_term += d_cov[t]*z[k];
        }else{
             sec_term += d_cov[t + k*HORIZON -1]*z[k];
        }*/
    }
    ret = mean + var * sec_term;
    return ret;
}

__device__ float gen_u(unsigned int id, curandState *state, float ave, float vr) {
    float u;
    curandState localState = state[id];
    u = curand_normal(&localState) * vr + ave;
    return u;
}

__global__ void setup_init_Covariance(float *Mat)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    //float values;
    /*if(threadIdx.x == 0 && blockIdx.x ==0)
    {
        values[threadIdx.x] = 1.0f;
    }*/
    if(threadIdx.x == blockIdx.x)
    {
        Mat[id] = 1.0f;
        //values[threadIdx.x] = 1.0f;
    }else{
        Mat[id] = 0.0f;
        //values[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    /*if(threadIdx.x == 0)
    {
       for(int i =0; i < blockDim.x; i++)
           Mat[id] = values[i];
    }  */   
    
}

__global__ void MCMPC_GPU_Linear_Example(float x, float y, float w, curandState *devs, Data1 *d_Datas, float var, int Blocks, float *d_cov, float *d_param, float *d_matrix)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int seq;
    seq = id;
    float qx = 0.0f;
    float total_cost = 0.0f;
    float u[HORIZON]= { };
    //float block_var;
    // int Powers;
    //printf("hoge id=%d\n", id);
    //float d_state_here[dim_state] = {x,y,w};
    float get_state[dim_state] = {};
    float z[HORIZON] = { };

    for(int t = 0; t < HORIZON; t++)
    {
        //block_var = var;
        for(int t_x = 0; t_x < HORIZON; t_x++)
        {
            z[t_x] = gen_u(seq, devs, 0, 1.0f);
            //z[t_x] = gen_u(seq, devs, d_Datas[0].Input[t_x], var);
            seq += HORIZON;
        }
        u[t] = generate_u(t, d_Datas[0].Input[t] /*ここが影響している可能性*/, var, d_cov, z); //ここが影響している可能性
        if(isnan(u[t])){
            u[t] = d_Datas[0].Input[t];
        }
        //u[t] = z[t];
        /*if(u[t]<-4.0f){
           u[t] = -4.0f;
        }
        if(u[t] > 4.0f){
           u[t] = 4.0f;
        }*/
        //printf("hoge = %d id=%d @ %f %f\n",t, id, u[t], z[t]);
        //calc_Linear_example(d_state_here, u[t], d_param, get_state);
        get_state[0] = d_param[0]*x + d_param[1]*y + d_param[2]*w + d_param[9]*u[t];
        get_state[1] = d_param[3]*x + d_param[4]*y + d_param[5]*w + d_param[10]*u[t];
        get_state[2] = d_param[6]*x + d_param[7]*y + d_param[8]*w + d_param[11]*u[t];
        x = get_state[0];
        y = get_state[1];
        w = get_state[2];
        //printf("hoge id=%d @ %f %f %f\n", id, u[t], d_param[0], get_state[1]);
        //qx += d_matrix[0] * get_state[0] * get_state[0] + d_matrix[4] * get_state[1] * get_state[1] +d_matrix[5] * u[t] * u[t];
         qx = x * x * d_matrix[0] + y * y * d_matrix[1] + w * w * d_matrix[2] + d_matrix[3]*u[t]*u[t]; 
        //qx += d_matrix[1] * get_state[0] * get_state[1];
        //qx += d_matrix[3] * get_state[0] * get_state[1];
        //qx += d_matrix[4] * get_state[1] * get_state[1];
        //qx += d_matrix[5] * u[t] * u[t];
        /*for(int h = 0; h < dim_state; h++){
           d_state_here[h] = get_state[h];
        }*/
        
        total_cost += qx;

        qx = 0.0f;
    }

    float KL_COST, S, lambda;
    lambda = HORIZON * dim_state;
    S = total_cost / lambda;
    KL_COST = exp(-S);
    W_comp[threadIdx.x] = KL_COST;
    L_comp[threadIdx.x] = total_cost;
    __syncthreads();
    if(threadIdx.x == 0)
    {
        best_thread_id_this_block = 0;
        for(int y = 1; y < blockDim.x; y++){
            if(L_comp[y] < L_comp[best_thread_id_this_block])
            {
                best_thread_id_this_block = y;
            }
        }
    }
    __syncthreads();
    if(threadIdx.x == best_thread_id_this_block)
    {
        Data1 block_best;
        block_best.L = L_comp[best_thread_id_this_block];
        block_best.W = W_comp[best_thread_id_this_block];
        for(int z = 0; z < HORIZON; z++)
        {
            block_best.Input[z] = u[z];
        }
        d_Datas[blockIdx.x] = block_best;

    } 
}

__global__ void shift_Input_vec(Input_vec *dst, float *dev_Us)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    for(int k = 0; k < HORIZON - 1; k++){
        dst[id].Input[k] = dev_Us[k+1];
    }
    dst[id].Input[HORIZON-1] = dev_Us[HORIZON - 1];
    __syncthreads();
}


#ifdef USING_THRUST
__global__ void Using_Thrust_MCMPC_Linear(float x, float y, float w, curandState *devs,Input_vec *d_Datas, float var, int Blocks, 
                                          float *d_cov, float *d_param, float *d_matrix, float *cost_vec){

    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int seq;
    seq = id;
    float qx = 0.0f;
    float total_cost = 0.0f;
    float u[HORIZON]= { };
    //float block_var;
    // int Powers;
    //printf("hoge id=%d\n", id);
    //float d_state_here[dim_state] = {x,y,w};
    float get_state[dim_state] = {};
    float z[HORIZON] = { };
    cost_vec[id] = 0.0f;

    for(int t_x = 0; t_x < HORIZON; t_x++)
    {
            z[t_x] = gen_u(seq, devs, 0, 1.0f);
            //z[t_x] = gen_u(seq, devs, d_Datas[0].Input[t_x], var);
            seq += N_OF_SAMPLES;
            __syncthreads();
    }
    for(int t = 0; t < HORIZON; t++)
    {
        //block_var = var;
        __syncthreads();
        //printf("id == %d -> z[%d]==%f\n",id, t, z[t]);
        u[t] = generate_u(t, d_Datas[0].Input[t] /*ここが影響している可能性*/, var, d_cov, z); //ここが影響している可能性
        if(isnan(u[t])){
            u[t] = d_Datas[0].Input[t];
        }
        get_state[0] = d_param[0]*x + d_param[1]*y + d_param[2]*w + d_param[9]*u[t];
        get_state[1] = d_param[3]*x + d_param[4]*y + d_param[5]*w + d_param[10]*u[t];
        get_state[2] = d_param[6]*x + d_param[7]*y + d_param[8]*w + d_param[11]*u[t];
        x = get_state[0];
        y = get_state[1];
        w = get_state[2];
        qx = x * x * d_matrix[0] + y * y * d_matrix[1] + w * w * d_matrix[2] + d_matrix[3]*u[t]*u[t];       
        total_cost += qx;

        qx = 0.0f;
    }
    if(isnan(total_cost))
    {
       total_cost = 100000;
    }
    float KL_COST, S, lambda;
    lambda = HORIZON * dim_state;
    //lambda = 10.0f;
    S = total_cost / lambda;
    KL_COST = exp(-S);
    /*W_comp[threadIdx.x] = KL_COST;
    L_comp[threadIdx.x] = total_cost;*/

    __syncthreads();
    d_Datas[id].W = KL_COST;
    //d_Datas[id].L = total_cost;
    for(int index = 0; index < HORIZON; index++)
    {
        d_Datas[id].Input[index] = u[index];
        d_Datas[id].dy[index] = z[index];
    }
    cost_vec[id] = total_cost;
    __syncthreads();
}

__global__ void Using_Thrust_MCMPC_Pendulum(float x, float th, float dx, float dth, curandState *devs, Input_vec *d_Datas, 
                                            float var, int Blocks, float *d_cov, float *d_param, float *d_const, float *d_matrix, float *cost_vec)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int seq;
    seq = id;
    float qx = 0.0f;
    float total_cost = 0.0f;
    float u[HORIZON]= { };
    //float block_var;
    // int Powers;
    //printf("hoge id=%d\n", id);
    //float d_state_here[dim_state] = {x,y,w};
    //float get_state[dim_state] = {};
    float z[HORIZON] = { };
    cost_vec[id] = 0.0f;
    float ddx, ddtheta;
    for(int t_x = 0; t_x < HORIZON; t_x++)
    {
            z[t_x] = gen_u(seq, devs, 0, 1.0f);
            //z[t_x] = gen_u(seq, devs, d_Datas[0].Input[t_x], var);
            seq += HORIZON;
    }
    __syncthreads();
    for(int t = 0; t < HORIZON; t++)
    {
        //block_var = var;
        //__syncthreads();
        //printf("id == %d -> z[%d]==%f\n",id, t, z[t]);
        u[t] = generate_u(t, d_Datas[0].Input[t] /*ここが影響している可能性*/, var, d_cov, z); //ここが影響している可能性
        if(isnan(u[t])){
            u[t] = d_Datas[0].Input[t];
        }

        if(u[t] < d_const[0]){
            u[t] = d_const[0];
        }
        if(u[t] > d_const[1]){
            u[t] = d_const[1];
        }
        ddx = Cart_type_Pendulum_ddx(u[t], x, th, dx, dth, d_param);
        ddtheta = Cart_type_Pendulum_ddtheta(u[t], x, th, dx, dth, d_param); 
        dx = dx + (ddx * interval);
        dth =  dth + (ddtheta * interval);
        x = x + (dx * interval);
        th = th + (dth * interval);

        while (th > M_PI)
            th -= (2 * M_PI);
        while (th < -M_PI)
            th += (2 * M_PI);
        /*if(id == 1000 || id == 1001){
          printf("id = %d :: u[%d] = %f x = %f th = %f\n", id, t, u[t], x, th);
        }*/
        qx = x * x * d_matrix[0] + th * th * d_matrix[1] + dx * dx * d_matrix[2] + dth * dth * d_matrix[3] + d_matrix[4] * u[t] * u[t];
        /*qx = x * x * d_matrix[0] + y * y * d_matrix[1] + w * w * d_matrix[2] + d_matrix[3]*u[t]*u[t];*/
        
        if( x <= 0){
            qx += 1 / pow(9.0*(x - d_const[2]),2);
            if(x < d_const[2]){
               qx += 10000000;
            }
        }else{
            qx += 1 / pow(9.0*(d_const[3] - x),2);
            if(x > d_const[3]){
               qx += 10000000;
            }
        }

        total_cost += qx;

        qx = 0.0f;
    }


    if(isnan(total_cost))
    {
       total_cost = 100000;
    }
    float KL_COST, S, lambda;
    // lambda = HORIZON * dim_state;
    lambda = 4 * HORIZON;
    //lambda = 10.0f;
    S = total_cost / lambda;
    KL_COST = exp(-S);
    /*W_comp[threadIdx.x] = KL_COST;
    L_comp[threadIdx.x] = total_cost;*/

    __syncthreads();
    d_Datas[id].W = KL_COST;
    d_Datas[id].L = total_cost;
    //d_Datas[id].L = total_cost;
    for(int index = 0; index < HORIZON; index++)
    {
        d_Datas[id].Input[index] = u[index];
        d_Datas[id].dy[index] = z[index];
    }
    cost_vec[id] = total_cost;
    __syncthreads();

}

__global__ void set_Input_vec(Input_vec *d_Input_vec, float init)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    for(int i=0; i < HORIZON; i++){
        d_Input_vec[id].Input[i] = init;
    }
    __syncthreads();
}

__global__ void callback_elite_sample(Data1 *d_Datas, Input_vec *dst, int *elite_indices)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    d_Datas[id].W =  dst[elite_indices[id]].W;
    d_Datas[id].L =  dst[elite_indices[id]].L;
    for(int i = 0; i < HORIZON; i++){
        d_Datas[id].Input[i] = dst[elite_indices[id]].Input[i];
        d_Datas[id].dy[i] = dst[elite_indices[id]].dy[i];
    }
}

__global__ void reset_Input_vec(Input_vec *d_Input_vec, float *opt){
   unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
   for(int i = 0; i < HORIZON; i++){
        d_Input_vec[id].Input[i] = opt[i];
    }
}
#endif
