#ifndef QUANTIZER_H
#define QUANTIZER_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>

#include "encoder.h"
#include "rotator.h"

#define QUERY_QUANTIZER_NUM_BITS 8
/**
 * 在计算数据库向量与 query 向量距离时，SAQ 需要对 query 进行单独的量化处理，以加速距离计算
 * 其量化过程为一个简单的 LVQ 量化，主要包括：
 * 1. 遍历确定值域 [min, max]
 * 2. 计算量化步长 delta = (max - min) / 2^b
 * 3. 对每个维度进行量化编码
 */
typedef struct {
    size_t dim;                     // 维度
    float *centroid;                // 质心指针，用于残差化
    float *rotatorMatrix;           // 随机正交矩阵指针，用于旋转向量
    float *residualVector;          // 用于存储残差向量的缓冲区指针，避免每次量化都分配内存
    float *rotatedVector;           // 用于存储旋转后向量的缓冲区指针，避免每次量化都分配内存
} QueryQuantizerCtxT;

void QueryQuantizerCtxInit(QueryQuantizerCtxT **ctx,
                         size_t dim,
                         float *centroid,
                         float *rotatorMatrix) {
    *ctx = (QueryQuantizerCtxT *)malloc(sizeof(QueryQuantizerCtxT));
    (*ctx)->dim = dim;
    (*ctx)->centroid = centroid;
    (*ctx)->rotatorMatrix = rotatorMatrix;
    (*ctx)->residualVector = (float *)malloc(sizeof(float) * dim);  // 分配残差向量缓冲区
    (*ctx)->rotatedVector = (float *)malloc(sizeof(float) * dim);   // 分配旋转后向量缓冲区
}

void QueryQuantizerCtxDestroy(QueryQuantizerCtxT **ctx) {
    if (ctx == NULL || *ctx == NULL) {
        return;
    }
    if ((*ctx)->residualVector) {
        free((*ctx)->residualVector);
        (*ctx)->residualVector = NULL;
    }
    if ((*ctx)->rotatedVector) {
        free((*ctx)->rotatedVector);
        (*ctx)->rotatedVector = NULL;
    }
    free(*ctx);
    *ctx = NULL;
}


typedef struct {
    uint8_t *quantizedResidualQueryCodes;   // 量化编码结果指针
    float residualQueryMin;                 // 残差查询向量的最小值
    float residualQueryMax;                 // 残差查询向量的最大值
    float delta;                            // 量化步长
    float residualQueryL2Sqr;               // 残差查询向量的平方 L2 范数
    float residualQueryL2Norm;              // 残差查询向量的 L2 范数
    float residualQuerySum;                 // 残差查询向量的元素和
} QueryQuantCodeT;

void CreateQueryQuantCode(QueryQuantCodeT **code, size_t dim) {
    *code = (QueryQuantCodeT *)malloc(sizeof(QueryQuantCodeT));
    (*code)->quantizedResidualQueryCodes = (uint8_t *)malloc(sizeof(uint8_t) * dim);
    (*code)->residualQueryMin = 0.0f;
    (*code)->residualQueryMax = 0.0f;
    (*code)->delta = 0.0f;
    (*code)->residualQueryL2Sqr = 0.0f;
    (*code)->residualQueryL2Norm = 0.0f;
    (*code)->residualQuerySum = 0.0f;
}

void DestroyQueryQuantCode(QueryQuantCodeT **code) {
    if (code == NULL || *code == NULL) {
        return;
    }
    if ((*code)->quantizedResidualQueryCodes) {
        free((*code)->quantizedResidualQueryCodes);
        (*code)->quantizedResidualQueryCodes = NULL;
    }
    free(*code);
    *code = NULL;
}

void QueryQuantizeVector(const QueryQuantizerCtxT *ctx,
                       const float *inputVector,
                       QueryQuantCodeT **outputCode) {
    size_t D = ctx->dim;
    float *centroid = ctx->centroid;
    float *rotatorMatrix = ctx->rotatorMatrix;
    float *residualVector = ctx->residualVector;
    float *rotatedVector = ctx->rotatedVector;
    
    CreateQueryQuantCode(outputCode, D);

    // Step 1: 计算残差向量
    for (size_t i = 0; i < D; ++i) {
        residualVector[i] = inputVector[i] - centroid[i];
    }

    // Step 2: 旋转向量
    rotateVector(ctx->rotatorMatrix, residualVector, rotatedVector, D);

    // Step 3: 量化编码
    // 3.1 确定值域 [min, max]
    float minVal = rotatedVector[0];
    float maxVal = rotatedVector[0];
    for (size_t i = 1; i < D; ++i) {
        if (rotatedVector[i] < minVal) {
            minVal = rotatedVector[i];
        }
        if (rotatedVector[i] > maxVal) {
            maxVal = rotatedVector[i];
        }
    }
    (*outputCode)->residualQueryMin = minVal;
    (*outputCode)->residualQueryMax = maxVal;

    // 3.2 计算量化步长 delta
    float delta = (maxVal - minVal) / ((1 << QUERY_QUANTIZER_NUM_BITS) - 0.01);
    (*outputCode)->delta = delta;

    // 3.3 量化编码
    float l2Sqr = 0.0f;
    float sum = 0.0f;
    uint8_t maxCode = (1u << QUERY_QUANTIZER_NUM_BITS) - 1;
    for (size_t i = 0; i < D; ++i) {
        float val = rotatedVector[i];
        float q = (val - minVal) / delta;
        if (q < 0.0f) q = 0.0f;                     // 下限
        if (q > (float)maxCode) q = (float)maxCode; // 上限
        uint8_t code = (uint8_t)floorf(q);          // cast<uint8_t>()
        (*outputCode)->quantizedResidualQueryCodes[i] = code;
        l2Sqr += val * val;                         
        sum += val;                               
    }
    (*outputCode)->residualQueryL2Sqr = l2Sqr;
    (*outputCode)->residualQueryL2Norm = sqrtf(l2Sqr);
    (*outputCode)->residualQuerySum = sum;
}



#endif // QUANTIZER_H