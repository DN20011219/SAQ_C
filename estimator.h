#ifndef ESTIMATOR_H
#define ESTIMATOR_H

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
 * 一次 CAQ 批量扫描任务即：对一个 query 向量，计算其与大量数据库向量的距离
 * 计算过程可分为两个阶段：
 * 1. 1bit 距离估算阶段
 * 2. 精排阶段
 * 为了提升扫描效率，避免重复计算 1bit 距离，使用扫描上下文缓存中间结果
 */
typedef struct {
    // TODO：向量 1bit 内积缓存 MAP，避免 1bit 内积重复计算
} CaqScannerCtxT;

void createCaqScannerCtx(CaqScannerCtxT **ctx) {
    *ctx = (CaqScannerCtxT *)malloc(sizeof(CaqScannerCtxT));
    if (*ctx == NULL) {
        return;
    }
}
void destroyCaqScannerCtx(CaqScannerCtxT **ctx) {
    if (ctx == NULL || *ctx == NULL) {
        return;
    }
    free(*ctx);
    *ctx = NULL;
}
void findCachedOneBitIp(
    CaqScannerCtxT *ctx,
    double *oneBitIpOut
) {
    // TODO: 实现 1bit 内积缓存逻辑
}
void cacheOneBitIp(
    CaqScannerCtxT *ctx,
    double oneBitIp
) {
    // TODO: 实现 1bit 内积缓存逻辑
}
/**
 * 一个标准的 CAQ 的 L2 距离估算流程主要由以下任务构成:
 * 1. 上下文准备（质心相同的向量属于相同上下文，因为质心相同，量化时使用的正交矩阵也相同）
 *  1. 根据质心将 query 向量残差化，得到 q'
 *  2. 使用与量化设置相同的随机正交矩阵对 q' 向量进行旋转，以保障 L2 距离一致
 *  3. 对 q' 进行量化编码，以使用整型计算代替浮点运算，节约算力资源。
 *     对于 query ，我们将其每个维度固定量化为 QUERY_QUANTIZER_NUM_BITS 位
 * 2. 计算 1bit 编码的距离
 * 
 * 
 * 若距离为 IP ，则需要：
 * 1. 上下文准备（质心相同的向量属于相同上下文，因为质心相同，量化时使用的正交矩阵也相同）
 *  1. 无需残差化，但需计算 query 与 质心的内积 <q, c>
 *  2. 使用与量化设置相同的随机正交矩阵对 q 向量进行旋转，以保障 IP 距离一致 
 *  3. 对 q 进行量化编码，以使用整型计算代替浮点运算，节约算力资源
 * 
 * 本模块输入为 query 向量 + 数据库向量的量化编码，输出为距离估算结果
 */
typedef struct {
    size_t dim;                     // 维度
    size_t numBits;                 // 数据库向量的每个维度量化为 numBits 位
    float oneOverSqrtD;             // 1 / sqrt(D)
    float *rotatorMatrix;           
    float *centroid;                
    QueryQuantizerCtxT *queryQuantCtx;      // query 量化上下文
    QueryQuantCodeT *queryQuantCode;        // 量化后的 query 向量
    CaqScannerCtxT *scannerCtx;             // 扫描上下文，用于缓存 1bit 内积等中间结果，避免重复计算
} OneBitL2CaqEstimatorCtxT;

void OneBitCaqEstimatorInit(OneBitL2CaqEstimatorCtxT **ctx,
                         size_t dim,
                         size_t numBits,
                         float *rotatorMatrix,  // 随机正交矩阵指针，同计算上下文的向量都使用一个相同的旋转矩阵进行旋转
                         float *centroid,       // 质心指针，同计算上下文的向量都使用一个相同的质心进行残差化
                         float *query,
                         CaqScannerCtxT *scannerCtx
                         ) {
    *ctx = (OneBitL2CaqEstimatorCtxT *)malloc(sizeof(OneBitL2CaqEstimatorCtxT));
    if (*ctx == NULL) {
        return;
    }
    (*ctx)->dim = dim;
    (*ctx)->numBits = numBits;
    (*ctx)->oneOverSqrtD = 1.0f / sqrtf((float)dim);
    (*ctx)->centroid = centroid;                                    // 质心数据指针由外部传入管理
    (*ctx)->rotatorMatrix = rotatorMatrix;                          // 随机正交矩阵指针由外部传入管理
    (*ctx)->scannerCtx = scannerCtx;                                // 扫描上下文由外部传入管理

    // 1.残差化并同向旋转
    QueryQuantizerCtxT *queryQuantCtx = NULL;
    QueryQuantizerCtxInit(&queryQuantCtx, dim, centroid, rotatorMatrix);
    (*ctx)->queryQuantCtx = queryQuantCtx;
    
    // 2.量化编码
    QuantizeQueryVector(queryQuantCtx, query, &((*ctx)->queryQuantCode));
}

void OneBitCaqEstimatorDestroy(OneBitL2CaqEstimatorCtxT **ctx) {
    if (ctx == NULL || *ctx == NULL) {
        return;
    }
    if ((*ctx)->queryQuantCtx) {
        QueryQuantizerCtxDestroy(&((*ctx)->queryQuantCtx));
        (*ctx)->queryQuantCtx = NULL;
    }
    if ((*ctx)->queryQuantCode) {
        DestroyQueryQuantCode(&((*ctx)->queryQuantCode));
        (*ctx)->queryQuantCode = NULL;
    }
    free(*ctx);
    *ctx = NULL;
}

/**
 * 距离计算算子，我们提供了：
 * 1. 1 * N bit 单向量内积距离计算接口的模板实现，即：数据库向量每个维度量化为 1 bit，query 向量每个维度量化为 N bit
 * 2. 8 * N bit 单向量内积距离计算接口的模板实现
 * 
 * 对于 SIMD 指令，我们提供了：
 * 1. x86 AVX 指令集的实现
 * 2. ARM NEON 指令集的实现
 * 3. 若均不支持，则使用普通 C 语言实现，该实现无需 QueryQuantizerLayoutT::LAYOUT_SEPARATED 布局支持
 */
#if (defined(__x86_64__) && defined(__AVX__))

uint64_t estimateOneBitIp(
    const OneBitL2CaqEstimatorCtxT *ctx,
    const CaqOneBitQuantCodeT *dataCaqCode
) {
    size_t BlockNum = GetOneBitCodeSimdBlockNum(ctx->dim);
    size_t bytesPerBlock = GetBytesPerSimdBlock();
    size_t bytesPerPlane = bytesPerBlock * BlockNum; // 一整个 bitplane 的字节跨度
    const QueryQuantCodeT *queryCode = ctx->queryQuantCode;
    uint64_t ipEstimate = 0;
    uint64_t partialIp = 0;
    uint64_t temp[4];

    // 以 block num 为单位处理，load 一次 database 向量的 SIMD_MAX_CAPACITY 维度后，
    // 完成所有 bit lane 的计算，减少内存访问次数
    for (int j = 0; j < BlockNum; ++j) {
        const uint8_t *dBlock = dataCaqCode->storedCodes + j * bytesPerBlock;

        for (int lane = 0; lane < QUERY_QUANTIZER_NUM_BITS; ++lane) {
            partialIp = 0;
        
            const uint8_t *qBlock =
                ctx->queryQuantCode->quantizedResidualQueryCodes + lane * bytesPerPlane +
                j * bytesPerBlock;
            
            // 1. 将 query 向量第 j 个 block 的 256 维度的第 B bit 加载到 AVX 寄存器
            // 2. 将 data 向量第 j 个 block 的 256 维度的 1bit 编码加载到 AVX 寄存器
            // 处理 32 bytes (256 bits)
            __m256i q_vec = _mm256_loadu_si256((__m256i *)qBlock);
            __m256i d_vec = _mm256_loadu_si256((__m256i *)dBlock);

            // 3. 使用 _mm256_and_si256 进行按位与运算，得到当前 lane 的内积结果
            __m256i and_vec = _mm256_and_si256(q_vec, d_vec);

            // 4. 使用 __builtin_popcountll 统计内积结果中 1 的个数
            _mm256_storeu_si256((__m256i *)temp, and_vec);
            for (int k = 0; k < 4; ++k) {
                partialIp += __builtin_popcountll(temp[k]); // 或者 _mm_popcnt_u64(temp[k])
            }

            // 5. 将结果累加到 ipEstimate 中，第 lane bit 的权重为 2^(b - lane - 1)
            // 具体权重示例: lane0 → 128, lane1 → 64, lane2 → 32, lane3 → 16
            // lane4 → 8, lane5 → 4, lane6 → 2, lane7 → 1
            ipEstimate += partialIp * (1ULL << (QUERY_QUANTIZER_NUM_BITS - lane - 1));
        }
    }
    return ipEstimate;
}

#elif (defined(__aarch64__) && defined(__ARM_NEON))

uint64_t estimateOneBitIp(
    const OneBitL2CaqEstimatorCtxT *ctx,
    const CaqOneBitQuantCodeT *dataCaqCode
) {
    // TODO: 实现 ARM NEON 版本的 1 * N bit 内积计算
    return 0;
}

#else

uint64_t estimateOneBitIp(
    const OneBitL2CaqEstimatorCtxT *ctx,
    const CaqOneBitQuantCodeT *dataCaqCode
) {
    size_t D = ctx->dim;
    const QueryQuantCodeT *queryCode = ctx->queryQuantCode;
    const uint8_t *dataCodes = dataCaqCode->storedCodes;
    uint64_t ipEstimate = 0;

    size_t i = 0;

    // 按 64 位（8 字节）处理
    for (; i + 64 <= D; i += 64) {
        uint64_t dataWord = *(const uint64_t*)(dataCodes + i / 8);
        // 展开 64 次，无分支
        for (int b = 0; b < 64; ++b) {
            uint32_t qval;
            GetQueryCodeValue(ctx->queryQuantCtx, queryCode, i + b, &qval);
            ipEstimate += qval * ((dataWord >> b) & 1ULL);
        }
    }

    // 剩余部分（最多 63 维）
    for (; i < D; ++i) {
        uint32_t qval;
        GetQueryCodeValue(ctx->queryQuantCtx, queryCode, i, &qval);
        ipEstimate += qval * ((dataCodes[i / 8] >> (i % 8)) & 1u);
    }

    return ipEstimate;
}

#endif


float OneBitCaqEstimateDistance(
    const OneBitL2CaqEstimatorCtxT *ctx,
    const CaqOneBitQuantCodeT *dataCaqCode
) {
    // 1. 计算 1bit 内积距离
    uint64_t ipEstimate = estimateOneBitIp(ctx, dataCaqCode);

    // 2. 根据公式计算最终距离估算值
    float const_bound = 0.58;
    float est_error = 0.8;
    float ip_oa1_qq = (
        ipEstimate - 
        (0.5 * ctx->queryQuantCode->residualQuerySum - const_bound * ctx->queryQuantCode->residualQueryL2Norm)
    ) * (4 / est_error * ctx->oneOverSqrtD) * dataCaqCode->oriVecL2Norm;

    float L2Distance = ctx->queryQuantCode->residualQueryL2Sqr + 
                        dataCaqCode->oriVecL2Norm * dataCaqCode->oriVecL2Norm -
                        ip_oa1_qq;
    return L2Distance;

    // TODO: 若距离为 IP，return ip_oa1_qq * 0.5;
}

float FloatL2(
    const float *a,
    const float *b,
    size_t dim
) {
    float l2 = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        l2 += diff * diff;
    }
    return l2;
}

#endif // ESTIMATOR_H