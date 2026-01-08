#ifndef DATA_QUANTIZER_H
#define DATA_QUANTIZER_H

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

    bool ownsRotatorMatrix;         // 是否由本上下文负责释放 rotatorMatrix
    bool ownsEncodeConfig;          // 是否由本上下文负责释放 encodeConfig

    float *residualVector;          // 用于存储残差向量的缓冲区指针，避免每次量化都分配内存
    float *rotatedVector;           // 用于存储旋转后向量的缓冲区指针，避免每次量化都分配内存
} CaqQuantizerCtxT;

void CaqQuantizerInitShared(
    CaqQuantizerCtxT **ctx,
    size_t dim,
    float *centroid,
    size_t numBits,
    bool useSeparateStorage,
    float *sharedRotatorMatrix,         // 若传入非 NULL，则共享该旋转矩阵
    CaqEncodeConfig *sharedEncodeConfig // 若传入非 NULL，则共享该编码配置
) {
    *ctx = (CaqQuantizerCtxT *)malloc(sizeof(CaqQuantizerCtxT));
    if (*ctx == NULL) {
        return;
    }
    (*ctx)->dim = dim;
    (*ctx)->centroid = centroid;                                    // 质心数据指针由外部传入管理
    (*ctx)->residualVector = (float *)malloc(sizeof(float) * dim);  // 分配残差向量缓冲区
    (*ctx)->rotatedVector = (float *)malloc(sizeof(float) * dim);   // 分配旋转后向量缓冲区

    if (sharedRotatorMatrix) {
        (*ctx)->rotatorMatrix = sharedRotatorMatrix;
        (*ctx)->ownsRotatorMatrix = false;
    } else {
        createRotatorMatrix(&(*ctx)->rotatorMatrix, dim);
        (*ctx)->ownsRotatorMatrix = true;
    }

    if (sharedEncodeConfig) {
        (*ctx)->encodeConfig = sharedEncodeConfig;
        (*ctx)->ownsEncodeConfig = false;
    } else {
        CreateCaqEncodeConfig(dim, numBits, useSeparateStorage, &(*ctx)->encodeConfig);
        (*ctx)->ownsEncodeConfig = true;
    }
}

void CaqQuantizerInit(CaqQuantizerCtxT **ctx,
                         size_t dim,
                         float *centroid,
                         size_t numBits,
                         bool useSeparateStorage
) {
    // 默认创建自有的旋转矩阵与编码配置
    CaqQuantizerInitShared(ctx, dim, centroid, numBits, useSeparateStorage, NULL, NULL);
}

void CaqQuantizerDestroy(CaqQuantizerCtxT **ctx) {
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
    if ((*ctx)->ownsRotatorMatrix) {
        destroyRotatorMatrix(&(*ctx)->rotatorMatrix);
    }
    if ((*ctx)->ownsEncodeConfig) {
        DestroyCaqQuantConfig(&(*ctx)->encodeConfig);
    }
    free(*ctx);
    *ctx = NULL;
}

void CaqQuantizeVector(const CaqQuantizerCtxT *ctx,
                       const float *inputVector,
                       CaqQuantCodeT **outputCode) {
    size_t D = ctx->dim;
    float *residualVector = ctx->residualVector;
    float *rotatedVector = ctx->rotatedVector;

    CreateCaqQuantCode(outputCode, D);

    // Step 1: 计算残差向量
    for (size_t i = 0; i < D; ++i) {
        residualVector[i] = inputVector[i] - ctx->centroid[i];
    }

    // Step 2: 旋转向量
    rotateVector(ctx->rotatorMatrix, residualVector, rotatedVector, D);

    // Step 3: 量化编码
    Encode(rotatedVector, *outputCode, ctx->encodeConfig);
}

void CaqSeparateStoredCodes(
    const CaqEncodeConfig *cfg, 
    const CaqQuantCodeT *caqCode,
    CaqOneBitQuantCodeT **oneBitCode,
    CaqResBitQuantCodeT **resBitCode
) {
    SeparateCode(
        cfg,
        caqCode,
        oneBitCode,
        resBitCode
    );
}

void CaqMergeStoredCodes(
    const CaqEncodeConfig *cfg,
    CaqQuantCodeT *caqCode
) {
    StoreCode(
        cfg,
        caqCode
    );
}

#endif // DATA_QUANTIZER_H