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
 * 一个标准的 CAQ 的 L2 距离估算流程主要由以下任务构成:
 * 1. 上下文准备（质心相同的向量属于相同上下文，因为质心相同，量化时使用的正交矩阵也相同）
 *  1. 根据质心将 query 向量残差化，得到 q'
 *  2. 使用与量化设置相同的随机正交矩阵对 q' 向量进行旋转，以保障 L2 距离一致
 *  3. 对 q' 进行量化编码，以使用整型计算代替浮点运算，节约算力资源
 * 2. 计算 1bit 编码的距离
 * 
 * 
 * 若距离为 IP ，则需要：
 * 1. 上下文准备（质心相同的向量属于相同上下文，因为质心相同，量化时使用的正交矩阵也相同）
 *  1. 无需残差化，但需计算 query 与 质心的内积 <q, c>
 *  2. 使用与量化设置相同的随机正交矩阵对 q 向量进行旋转，以保障 IP 距离一致 
 *  3. 对 q 进行量化编码，以使用整型计算代替浮点运算，节约算力资源
 * 
 * 本模块输入为 query 向量，
 */

typedef struct {
    // TODO：向量 1bit 内积缓存 MAP，避免 1bit 内积重复计算
} CaqScannerCtxT;

typedef struct {
    size_t dim;                     // 维度
    size_t numBits;                 // 每个维度量化为 numBits 位
    float *rotatorMatrix;           // 随机正交矩阵指针，同计算上下文的向量都使用一个相同的旋转矩阵进行旋转
    float *centroid;                // 质心指针，同计算上下文的向量都使用一个相同的质心进行残差化


    CaqScannerCtxT *scannerCtx;     // 扫描上下文，用于缓存 1bit 内积等中间结果，避免重复计算
} L2CaqEstimatorCtxT;

void CaqEstimatorInit(L2CaqEstimatorCtxT **ctx,
                         size_t dim,
                         size_t numBits,
                         float *rotatorMatrix,
                         float *centroid,
                         CaqScannerCtxT *scannerCtx
                         ) {
    *ctx = (L2CaqEstimatorCtxT *)malloc(sizeof(L2CaqEstimatorCtxT));
    if (*ctx == NULL) {
        return;
    }
    (*ctx)->dim = dim;
    (*ctx)->numBits = numBits;
    (*ctx)->centroid = centroid;                                    // 质心数据指针由外部传入管理
    (*ctx)->rotatorMatrix = rotatorMatrix;                          // 随机正交矩阵指针由外部传入管理
    (*ctx)->scannerCtx = scannerCtx;                                // 扫描上下文由外部传入管理
}

void CaqQuantizationDestroy(L2CaqEstimatorCtxT **ctx) {
    if (ctx == NULL || *ctx == NULL) {
        return;
    }
    free(*ctx);
    *ctx = NULL;
}

#endif // QUANTIZER_H