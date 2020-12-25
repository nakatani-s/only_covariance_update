/*
data exchange between host and device
*/
#include <curand_kernel.h>
#ifndef DATASTRUCTURE_CUH
#define DATASTRUCTURE_CUH
typedef struct{
    float L;
    float W;
    float dy[HORIZON];
    float Input[HORIZON];
}Data1;

typedef struct{
    float L;
    float W;
    float dy[HORIZON];
    float Input[HORIZON];
}Input_vec;
#endif