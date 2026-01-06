#ifndef ESTIMATOR_EASY_H
#define ESTIMATOR_EASY_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>

#include "native_check.h"
#include "encoder.h"
#include "rotator.h"
#include "data_quantizer.h"
#include "query_quantizer.h"

/**
 * 为了简化距离估算，我们提供了一个更易用的距离估算器上下文封装：
 * 该封装不对数据库向量的量化编码进行拆分，而是将完整的 CAQ 量化编码存储在一个结构体中
 * 这样可以简化距离估算器的使用，但可能会牺牲部分性能（大部分向量距离 query 都较远，没有必要使用全部精度进行计算）
 */
typedef struct {
    size_t dim;                             // 维度
    size_t numBits;                         // 数据库向量的每个维度量化为 numBits 位
    QueryQuantizerCtxT *queryQuantCtx;      // query 量化上下文，存储经过旋转和残差化后的 query 向量
    QueryQuantCodeT *queryQuantCode;        // query 量化编码
} FullBitL2EstimatorCtxT;

void CreateFullBitL2EstimatorCtx(
    FullBitL2EstimatorCtxT **ctx,
    size_t dim,
    size_t numBits,
    QueryQuantizerCtxT *queryQuantCtx,
    QueryQuantCodeT *queryQuantCode
) {
    *ctx = (FullBitL2EstimatorCtxT *)malloc(sizeof(FullBitL2EstimatorCtxT));
    if (*ctx == NULL) {
        return;
    }
    (*ctx)->dim = dim;
    (*ctx)->numBits = numBits;
    (*ctx)->queryQuantCtx = queryQuantCtx;
    (*ctx)->queryQuantCode = queryQuantCode;
}
void DestroyFullBitL2EstimatorCtx(FullBitL2EstimatorCtxT **ctx) {
    if (ctx == NULL || *ctx == NULL) {
        return;
    }
    free(*ctx);
    *ctx = NULL;
}

/**
 * 注意：这里传入的 dataCaqCode 必须是通过 CaqQuantizerInit(&quantizerCtx, dim, centroid, numBits, false);
 * 创建的量化器得到的量化编码，即完整存储 numBits 位编码的结构体，不进行缩放
 */
void CaqEstimateDistance(
    const FullBitL2EstimatorCtxT *queryCtx,
    const CaqQuantCodeT *dataCaqCode,
    float *distanceOut
) {
    const float o_l2sqr = dataCaqCode->oriVecL2Norm * dataCaqCode->oriVecL2Norm;
    float sum_q  = queryCtx->queryQuantCode->residualQuerySum;
    float ipQQ = queryCtx->queryQuantCode->residualQueryL2Sqr;
    float ipScale = -2.0f / dataCaqCode->oriVecQuantVecIp * o_l2sqr;
    
    float *queryVec = queryCtx->queryQuantCtx->rotatedVector;
    const uint8_t *dataCodes = dataCaqCode->storedCodes;

    // 1. 计算 float * uint8_t解码 的内积
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= queryCtx->dim; i += 8) {
        __m256 qv = _mm256_loadu_ps(queryVec + i);
        __m128i bytes8 = _mm_loadl_epi64((const __m128i *)(dataCodes + i));
        __m256 codes_ps = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes8));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(qv, codes_ps));
    }
    __m128 low = _mm256_castps256_ps128(acc);
    __m128 high = _mm256_extractf128_ps(acc, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float sum_qd = _mm_cvtss_f32(sum128);
    for (; i < queryCtx->dim; ++i) {
        sum_qd += queryVec[i] * (float)dataCodes[i];
    }
#else
    float sum_qd = 0.0f;
    for (size_t i = 0; i < queryCtx->dim; ++i) {
        float q = queryVec[i];
        sum_qd += q * (float)dataCodes[i];
    }
#endif
    float ipOQ = dataCaqCode->delta * (sum_qd + 0.5f * sum_q) + dataCaqCode->min * sum_q;

    float L2Distance = o_l2sqr + ipQQ + ipScale * ipOQ;
    if (L2Distance < 0) L2Distance = 0.0f;
    *distanceOut = L2Distance;
}

#endif // ESTIMATOR_EASY_H