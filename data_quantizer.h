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

/**
 * 一个标准的 CAQ 量化流程主要由以下任务构成:
 * 1. 根据质心将所有向量残差化
 * 2. 使用随机正交矩阵对向量进行旋转，以打散向量的分布，提升量化效果
 * 3. 使用编码器对旋转后的向量进行 CAQ 量化编码
 * 4. 存储量化编码结果及相关统计量
 * 
 * 本模块主要实现步骤 1、2、3 ，提供相应的接口函数，步骤 4 由使用者自行实现。
 */
typedef struct {
    size_t dim;                     // 维度
    float *rotatorMatrix;           // 随机正交矩阵指针，同量化上下文的向量都使用一个相同的旋转矩阵进行旋转
    float *centroid;                // 质心指针，同量化上下文的向量都使用一个相同的质心进行残差化
    CaqEncodeConfig *encodeConfig;  // 量化配置指针，同量化上下文的向量的编码配置相同

    float *residualVector;          // 用于存储残差向量的缓冲区指针，避免每次量化都分配内存
    float *rotatedVector;           // 用于存储旋转后向量的缓冲区指针，避免每次量化都分配内存
} L2CaqQuantizerCtxT;

void CaqQuantizerInit(L2CaqQuantizerCtxT **ctx,
                         size_t dim,
                         float *centroid,
                         size_t numBits) {
    *ctx = (L2CaqQuantizerCtxT *)malloc(sizeof(L2CaqQuantizerCtxT));
    if (*ctx == NULL) {
        return;
    }
    (*ctx)->dim = dim;
    (*ctx)->centroid = centroid;                                    // 质心数据指针由外部传入管理
    (*ctx)->residualVector = (float *)malloc(sizeof(float) * dim);  // 分配残差向量缓冲区
    (*ctx)->rotatedVector = (float *)malloc(sizeof(float) * dim);   // 分配旋转后向量缓冲区
    createRotatorMatrix(&(*ctx)->rotatorMatrix, dim);
    createCaqQuantConfig(dim, numBits, &(*ctx)->encodeConfig);
}

void CaqQuantizationDestroy(L2CaqQuantizerCtxT **ctx) {
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
    destroyRotatorMatrix(&(*ctx)->rotatorMatrix);
    destroyCaqQuantConfig(&(*ctx)->encodeConfig);
    free(*ctx);
    *ctx = NULL;
}

void CaqQuantizeVector(const L2CaqQuantizerCtxT *ctx,
                       const float *inputVector,
                       CaqQuantCodeT **outputCode) {
    size_t D = ctx->dim;
    float *residualVector = ctx->residualVector;
    float *rotatedVector = ctx->rotatedVector;

    createCaqQuantCode(outputCode, D);

    // Step 1: 计算残差向量
    for (size_t i = 0; i < D; ++i) {
        residualVector[i] = inputVector[i] - ctx->centroid[i];
    }

    // Step 2: 旋转向量
    rotateVector(ctx->rotatorMatrix, residualVector, rotatedVector, D);

    // Step 3: 量化编码
    encode(rotatedVector, *outputCode, ctx->encodeConfig);
}



#endif // QUANTIZER_H