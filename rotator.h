#ifndef ROTATOR_H
#define ROTATOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>

// #define Haar_Random_Orthogonal_Matrix
#define Householder_Random_Orthogonal_Matrix

#ifdef Haar_Random_Orthogonal_Matrix
/**
 * 正交矩阵工具:
 * 生成 Haar 随机正交矩阵，并使用该矩阵旋转向量，对比于 odalib/utils/rotator.hpp 
 * 中的 Rotator 类中的 HouseholderQR 方法，该方法生成的正交矩阵服从 Haar 分布，更加均匀
 * 但 L2 范数误差稍高，一般在 1e-5 级别
 */
inline float frandNormal()
{
    // Box-Muller transform: normal distribution N(0,1)
    float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

// 生成一个 Haar random orthogonal matrix P of size D x D
void randomOrthogonalMatrix(float *P, int D)
{
    // ---- Step 1: Build random Gaussian matrix A ----
    float *A = (float *)malloc(sizeof(float) * D * D);
    for (int i = 0; i < D * D; i++)
    {
        A[i] = frandNormal();
    }

    // ---- Step 2: QR decomposition using Gram-Schmidt ----
    float *Q = (float *)malloc(sizeof(float) * D * D);
    float *v = (float *)malloc(sizeof(float) * D);

    for (int j = 0; j < D; j++)
    {
        // Copy column j into v
        for (int i = 0; i < D; i++)
            v[i] = A[i * D + j];

        // Subtract projection onto earlier columns
        for (int k = 0; k < j; k++)
        {
            float dot = 0.0f;
            for (int i = 0; i < D; i++)
                dot += Q[i * D + k] * v[i];

            for (int i = 0; i < D; i++)
                v[i] -= dot * Q[i * D + k];
        }

        // Normalize v → Q[:,j]
        float norm = 0.0f;
        for (int i = 0; i < D; i++)
            norm += v[i] * v[i];
        norm = sqrtf(norm);

        for (int i = 0; i < D; i++)
            Q[i * D + j] = v[i] / norm;
    }

    // ---- Step 3: Fix signs (Haar measure correction) ----
    // Flip each column randomly with 50% chance
    for (int j = 0; j < D; j++)
    {
        if (rand() & 1)
        {
            for (int i = 0; i < D; i++)
                Q[i * D + j] = -Q[i * D + j];
        }
    }

    // ---- Step 4: P = Qᵀ ----
    // Because rotation applied as P * vector, transpose is useful sometimes
    for (int i = 0; i < D; i++)
    {
        for (int j = 0; j < D; j++)
        {
            P[i * D + j] = Q[j * D + i];
        }
    }

    free(A);
    free(Q);
    free(v);
}

#endif

#ifdef Householder_Random_Orthogonal_Matrix
/**
 * Householder QR 生成随机正交矩阵的 L2 范数误差较低，可达到 1e-6 级别
 * 但该方法生成的正交矩阵并不服从 Haar 分布，均匀性较差，且旋转幅度非常小
 * 可根据实际需求选择使用
 */
inline float frandUniform() {
    // [-1,1] uniform random, 模拟 Eigen::Random()
    return (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

float vector_norm(const float* v, int len) {
    float s = 0.0f;
    for (int i = 0; i < len; i++) s += v[i]*v[i];
    return sqrtf(s);
}

// Householder QR: generate Q from random matrix A
void randomOrthogonalMatrix(float* P, int D) {
    // Step 1: generate random matrix A ~ [-1,1]
    float* A = (float*)malloc(sizeof(float) * D * D);
    for (int i = 0; i < D*D; i++) A[i] = frandUniform();

    // Step 2: Householder QR decomposition (in-place on A)
    float* R = (float*)malloc(sizeof(float) * D * D); // for reconstruct Q
    for (int i=0;i<D*D;i++) R[i] = (i%D==i/D) ? 1.0f : 0.0f; // Q init = I

    for (int k=0;k<D;k++) {
        // Compute Householder vector v for column k
        float norm_x = 0.0f;
        for (int i=k;i<D;i++) norm_x += A[i*D+k]*A[i*D+k];
        norm_x = sqrtf(norm_x);
        if (norm_x < 1e-8f) continue;

        float vk_sign = (A[k*D+k]>=0) ? 1.0f : -1.0f;
        float beta = A[k*D+k] + vk_sign*norm_x;

        float* v = (float*)malloc(sizeof(float)*(D-k));
        for(int i=0;i<D-k;i++) v[i] = A[(i+k)*D+k];
        v[0] = beta;

        float v_norm = vector_norm(v,D-k);
        for(int i=0;i<D-k;i++) v[i] /= v_norm;

        // Apply Householder to remaining columns of A
        for(int j=k;j<D;j++){
            float dot = 0.0f;
            for(int i=0;i<D-k;i++) dot += v[i]*A[(i+k)*D+j];
            for(int i=0;i<D-k;i++) A[(i+k)*D+j] -= 2.0f*v[i]*dot;
        }

        // Apply same reflection to R (reconstruct Q)
        for(int j=0;j<D;j++){
            float dot = 0.0f;
            for(int i=0;i<D-k;i++) dot += v[i]*R[(i+k)*D+j];
            for(int i=0;i<D-k;i++) R[(i+k)*D+j] -= 2.0f*v[i]*dot;
        }

        free(v);
    }

    // Step 3: transpose Q to match Eigen's householderQ().transpose()
    for(int i=0;i<D;i++)
        for(int j=0;j<D;j++)
            P[i*D+j] = R[j*D+i];

    free(A);
    free(R);
}
#endif

// 生成一个正交矩阵
void createRotatorMatrix(float **P_out, size_t D)
{
    float *P = (float *)malloc(sizeof(float) * D * D);
    randomOrthogonalMatrix(P, D);
    *P_out = P; // 现在 P 是一个随机旋转矩阵，可以用于向量旋转等操作
}

void destroyRotatorMatrix(float **P)
{
    if (P && *P)
    {
        free(*P);
        *P = NULL;
    }
}

// 使用矩阵 P 旋转向量 vec_in，结果存入 vec_out
void rotateVector(const float* P, const float* vec_in, float* vec_out, size_t D) {
    for (size_t i = 0; i < D; i++) {
        float s = 0.0f;
        for (size_t j = 0; j < D; j++)
            s += P[i * D + j] * vec_in[j];
        vec_out[i] = s;
    }
}

#endif // ROTATOR_H