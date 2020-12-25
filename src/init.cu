/*
#include "../include/~.cuh"
*/ 
#include <math.h>

// include header files described by editter
#include "../include/init.cuh"

void Mat_sys_A(float *a)
{
#ifdef Pendulum
    a[0] = 0.1f;
    a[1] = 0.024f;
    a[2] = 0.2f;
    a[3] = a[1] * powf(a[2],2) /3;
    a[4] = 1.265f;
    a[5] = 0.0000001;
    a[6] = 9.81f;
#else
    a[0] = 0.0f;
    a[1] = 1.0f;
    a[2] = 0.0f;
    a[3] = 0.0f;
    a[4] = -1.1364f;
    a[5] = 0.2273f;
    a[6] = 0.0f;
    a[7] = -0.1339f;
    a[8] = -0.1071f;

    a[9] = 0.0f;
    a[10] = 0.0f;
    a[11] = 0.0893f;
#endif
 }

void init_state(float *st)
{
#ifdef Pendulum
    st[0] = 0.0f; //x
    st[1] = M_PI; //theta
    st[2] = 0.0f; //dx
    st[3] = 0.0f; //dth
#else
    st[0] = 2.98f;
    st[1] = 0.7f;
    st[3] = 0.0f;
#endif
}

void init_Weight_matrix(float * matrix)
{
#ifdef Pendulum
    matrix[0] = 1.75f;
    matrix[1] = 1.75f;
    matrix[2] = 0.04f;
    matrix[3] = 0.05f;
    matrix[4] = 1.0f;
#else
    matrix[0] = 2.0f;
    matrix[1] = 1.0f;
    matrix[2] = 0.1f;
    matrix[3] = 1.0f;
#endif
}

void init_opt( float *opt )
{
    opt[0] = -2.69f;
    opt[1] = 2.3787f;
    opt[2] = -2.0953f;
    opt[3] = 1.8364f;
    opt[4] = -1.5989f;
    opt[5] = 1.38f;
    opt[6] = -1.1772f;
    opt[7] = 0.9881f;
    opt[8] = -0.8106f;
    opt[9] = 0.6424f;
    opt[10] = -0.4818f;
    opt[11] = 0.327f;
    opt[12] = -0.1778f;
    opt[13] = 0.0428f;
    opt[14] = 0.0027f;
}

void init_constraint( float *constraint )
{
    constraint[0] = -1.0f;
    constraint[1] = 1.0f;
    constraint[2] = -1.0f;
    constraint[3] = 1.0f;
}
