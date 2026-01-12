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
    // TODO: 向量 L2 范数缓存 MAP，避免重复计算
    float IpDummy;     // 占位符
    float L2NormDummy; // 占位符
} CaqScannerCtxT;

void CreateCaqScannerCtx(CaqScannerCtxT **ctx) {
    *ctx = (CaqScannerCtxT *)malloc(sizeof(CaqScannerCtxT));
    if (*ctx == NULL) {
        return;
    }
}
void DestroyCaqScannerCtx(CaqScannerCtxT **ctx) {
    if (ctx == NULL || *ctx == NULL) {
        return;
    }
    free(*ctx);
    *ctx = NULL;
}
void InsertIntoCache(
    CaqScannerCtxT *ctx,
    uint32_t nodeId,
    float oneBitIp,
    float oL2Norm
) {
    // TODO: 实现 1bit 内积缓存逻辑
    ctx->IpDummy = oneBitIp;
    ctx->L2NormDummy = oL2Norm;
}
void FindInCache(
    CaqScannerCtxT *ctx,
    uint32_t nodeId,
    float *oneBitIpOut,
    float *oL2NormOut
) {
    // TODO: 实现 1bit 内积缓存逻辑
    *oneBitIpOut = ctx->IpDummy;
    *oL2NormOut = ctx->L2NormDummy;
}

/**
 * 一个标准的 CAQ 的 L2 距离估算流程主要由以下任务构成:
 * 1. 上下文准备（质心相同的向量属于相同上下文，因为质心相同，量化时使用的正交矩阵也相同）
 *  1. 根据质心将 query 向量残差化，得到 q'
 *  2. 使用与量化设置相同的随机正交矩阵对 q' 向量进行旋转，以保障 L2 距离一致
 *  3. 对 q' 进行量化编码，以使用整型计算代替浮点运算，节约算力资源。
 *     对于 query ，我们将其每个维度固定量化为 QUERY_QUANTIZER_NUM_BITS 位
 * 2. 计算 1bit 编码的距离
 * 3. 精排阶段（可选）：
 *  1. 使用剩余位数的量化编码对距离进行精排
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

// 已预旋转质心 + 传入已旋转 query 的初始化版本，避免重复旋转
void OneBitCaqEstimatorInitRotated(OneBitL2CaqEstimatorCtxT **ctx,
                         size_t dim,
                         size_t numBits,
                         float *rotatorMatrix,
                         float *centroidRotated,
                         float *rotatedQuery,
                         CaqScannerCtxT *scannerCtx
                         ) {
    *ctx = (OneBitL2CaqEstimatorCtxT *)malloc(sizeof(OneBitL2CaqEstimatorCtxT));
    if (*ctx == NULL) {
        return;
    }
    (*ctx)->dim = dim;
    (*ctx)->numBits = numBits;
    (*ctx)->oneOverSqrtD = 1.0f / sqrtf((float)dim);
    (*ctx)->centroid = NULL;
    (*ctx)->rotatorMatrix = rotatorMatrix;
    (*ctx)->scannerCtx = scannerCtx;

    QueryQuantizerCtxT *queryQuantCtx = NULL;
    QueryQuantizerCtxInitRotated(&queryQuantCtx, dim, rotatorMatrix, centroidRotated);
    (*ctx)->queryQuantCtx = queryQuantCtx;

    QuantizeQueryVector(queryQuantCtx, rotatedQuery, &((*ctx)->queryQuantCode));
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
 * 精确距离估算器上下文，用于存储 CAQ 精排阶段所需的上下文信息
 */
typedef struct {
    size_t dim;                     // 维度
    size_t numBits;                 // 数据库向量的每个维度量化为 numBits 位
    float caqDelta;                 // CAQ 量化步长，固定为 2.0 / (1 << numBits)
    QueryQuantizerCtxT *queryQuantCtx;      // query 量化上下文
    QueryQuantCodeT *queryQuantCode;// 量化后的 query 向量，与 OneBitL2CaqEstimatorCtxT 相同
    CaqScannerCtxT *scannerCtx;     // 扫描上下文，用于缓存 1bit 内积等中间结果，避免重复计算
} RestBitL2EstimatorCtxT;

void CreateRestBitL2EstimatorCtx(
    RestBitL2EstimatorCtxT **ctx,
    size_t dim,
    size_t numBits,
    QueryQuantizerCtxT *queryQuantCtx,
    QueryQuantCodeT *queryQuantCode,
    CaqScannerCtxT *scannerCtx
) {
    *ctx = (RestBitL2EstimatorCtxT *)malloc(sizeof(RestBitL2EstimatorCtxT));
    if (*ctx == NULL) {
        return;
    }
    (*ctx)->dim = dim;
    (*ctx)->numBits = numBits;
    (*ctx)->caqDelta = 2.0f / (float)(1 << numBits);
    (*ctx)->queryQuantCtx = queryQuantCtx;
    (*ctx)->queryQuantCode = queryQuantCode;  // query 量化编码由外部传入管理
    (*ctx)->scannerCtx = scannerCtx;          // 扫描上下文由外部传入管理
}
void DestroyRestBitL2EstimatorCtx(RestBitL2EstimatorCtxT **ctx) {
    if (ctx == NULL || *ctx == NULL) {
        return;
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
#if (defined(SIMD_AVX_ENABLED))

// 是否需要对 query 量化编码使用分离式布局以提升 SIMD 计算效率，开启需要同步预处理 query 编码
#define USE_SEPARATED_QUERY_LAYOUT
// 预先计算分离式布局的 query block 指针，避免每次估计距离时重复计算指针偏移
#define PRE_COMPUTED_QUERY_BLOCK_PTRS

#ifdef USE_SEPARATED_QUERY_LAYOUT

#ifndef PRE_COMPUTED_QUERY_BLOCK_PTRS

/**
 * 1bit IP 估计（纯 AVX 优化版，分离式 query layout，需要对 query 做转置）
 */
float estimateOneBitIp( 
    const OneBitL2CaqEstimatorCtxT *ctx,
    const CaqOneBitQuantCodeT *dataCaqCode
) {
    // 纯 AVX ：使用 128-bit SSE2 对 32B block 分两半处理
    size_t BlockNum = GetOneBitCodeSimdBlockNum(ctx->dim);
    size_t bytesPerBlock = GetBytesPerSimdBlock(); // 32 bytes in AVX, but 16 bytes for NEON
    const QueryQuantCodeT *queryCode = ctx->queryQuantCode;
    uint64_t ipEstimate = 0;
    uint64_t ppcScalar = dataCaqCode->totalPopcount; // 1bit popcount 在量化时已统计
    uint64_t tmp64[2]; // 存放 16B AND 结果，便于 popcount
    uint8_t *qBlock;

    for (int j = 0; j < (int)BlockNum; ++j) {
        const uint8_t *dBlock = dataCaqCode->storedCodes + j * bytesPerBlock;
        // ppcScalar 已预存，无需在查询时重复 popcount
        for (int plane = 0; plane < QUERY_QUANTIZER_NUM_BITS; ++plane) {
            uint64_t partialIp = 0;

            GetSepQueryCodeValue(ctx->queryQuantCtx, ctx->queryQuantCode, plane, j, &qBlock);

            // 低 16 字节
            __m128i q_lo = _mm_loadu_si128((const __m128i *)(qBlock));
            __m128i d_lo = _mm_loadu_si128((const __m128i *)(dBlock));
            __m128i a_lo = _mm_and_si128(q_lo, d_lo);
            _mm_storeu_si128((__m128i *)tmp64, a_lo);
            partialIp += (uint64_t)__builtin_popcountll(tmp64[0]) + (uint64_t)__builtin_popcountll(tmp64[1]);

            // 高 16 字节
            __m128i q_hi = _mm_loadu_si128((const __m128i *)(qBlock + 16));
            __m128i d_hi = _mm_loadu_si128((const __m128i *)(dBlock + 16));
            __m128i a_hi = _mm_and_si128(q_hi, d_hi);
            _mm_storeu_si128((__m128i *)tmp64, a_hi);
            partialIp += (uint64_t)__builtin_popcountll(tmp64[0]) + (uint64_t)__builtin_popcountll(tmp64[1]);

            ipEstimate += partialIp * (1ULL << (QUERY_QUANTIZER_NUM_BITS - plane - 1));
        }
    }

    return queryCode->delta * (float)ipEstimate +
           (queryCode->residualQueryMin + 0.5f * queryCode->delta) * (float)ppcScalar;
}
#else
/**
 * 1bit IP 估计（SSE2 优化版，分离式 query layout，需要对 query 做转置）
 */
float estimateOneBitIp(
    const OneBitL2CaqEstimatorCtxT *ctx,
    const CaqOneBitQuantCodeT *dataCaqCode
) {
    const size_t BlockNum       = GetOneBitCodeSimdBlockNum(ctx->dim);
    const size_t bytesPerBlock  = GetBytesPerSimdBlock(); // 32 bytes
    const QueryQuantCodeT *queryCode = ctx->queryQuantCode;

    uint64_t ipEstimate = 0;
    uint64_t ppcScalar  = dataCaqCode->totalPopcount; // 1bit popcount 在量化时已统计

    // popcount buffer
    uint64_t tmp64[2];

    // plane 权重（QUERY_QUANTIZER_NUM_BITS 是编译期常量，理论上可编译器计算）
    uint64_t planeWeight[QUERY_QUANTIZER_NUM_BITS];
    for (int p = 0; p < QUERY_QUANTIZER_NUM_BITS; ++p) {
        planeWeight[p] = 1ULL << (QUERY_QUANTIZER_NUM_BITS - p - 1);
    }

    for (size_t j = 0; j < BlockNum; ++j) {
        const uint8_t *dBlock =
            dataCaqCode->storedCodes + j * bytesPerBlock;

        /* 1bit popcount 已预计算，查询时不再统计；data block 只 load 一次 */
        __m128i d_lo = _mm_loadu_si128((const __m128i *)(dBlock));
        __m128i d_hi = _mm_loadu_si128((const __m128i *)(dBlock + 16));

        /* --------------------------------------------------
         * 3. 预先计算所有 plane 的 query block 指针
         * -------------------------------------------------- */
        uint8_t *qPlanePtr[QUERY_QUANTIZER_NUM_BITS];
        for (int plane = 0; plane < QUERY_QUANTIZER_NUM_BITS; ++plane) {
            GetSepQueryCodeValue(
                ctx->queryQuantCtx,
                queryCode,
                (size_t)plane,
                j,
                &qPlanePtr[plane]
            );
        }

        /* --------------------------------------------------
         * 4. plane 内积累 IP
         * -------------------------------------------------- */
        uint64_t blockIp = 0;

        for (int plane = 0; plane < QUERY_QUANTIZER_NUM_BITS; ++plane) {
            const uint8_t *qBlock = qPlanePtr[plane];

            // 低 16B
            __m128i q_lo = _mm_loadu_si128((const __m128i *)(qBlock));
            __m128i a_lo = _mm_and_si128(q_lo, d_lo);
            _mm_storeu_si128((__m128i *)tmp64, a_lo);
            uint64_t pc =
                (uint64_t)__builtin_popcountll(tmp64[0]) +
                (uint64_t)__builtin_popcountll(tmp64[1]);

            // 高 16B
            __m128i q_hi = _mm_loadu_si128((const __m128i *)(qBlock + 16));
            __m128i a_hi = _mm_and_si128(q_hi, d_hi);
            _mm_storeu_si128((__m128i *)tmp64, a_hi);
            pc +=
                (uint64_t)__builtin_popcountll(tmp64[0]) +
                (uint64_t)__builtin_popcountll(tmp64[1]);

            blockIp += pc * planeWeight[plane];
        }

        ipEstimate += blockIp;
    }

    /* --------------------------------------------------
     * 5. 最终标量修正
     * -------------------------------------------------- */
    return queryCode->delta * (float)ipEstimate +
           (queryCode->residualQueryMin + 0.5f * queryCode->delta) *
               (float)ppcScalar;
}
#endif
#else
/**
 * AVX2 优化版本，要求 D 是 128 的倍数
 * 无需分离式布局支持，减少pack代价，即不需要预先转置 query 编码
 */
float estimateOneBitIp(
    const OneBitL2CaqEstimatorCtxT *ctx,
    const CaqOneBitQuantCodeT *dataCaqCode
) {
    size_t D = ctx->dim;
    const uint8_t *dataCodes = dataCaqCode->storedCodes;
    const uint8_t *queryCodes = ctx->queryQuantCode->quantizedQueryOriCodes;
    const float queryDelta = ctx->queryQuantCode->delta;
    const float queryMin = ctx->queryQuantCode->residualQueryMin + 0.5f * queryDelta;

    __m256i ipVec = _mm256_setzero_si256();

    for (size_t i = 0; i < D; i += 32) {
        uint32_t bits32 = *(uint32_t *)(dataCodes + i / 8);

        for (int k = 0; k < 4; ++k) {  // 每 8 bit 展开一次
            uint8_t bytek = (bits32 >> (k * 8)) & 0xFF;

            // 使用 _mm256_setr_epi32 展开 8-bit -> 8 个 32-bit
            __m256i bv = _mm256_setr_epi32(
                (bytek >> 0) & 1, (bytek >> 1) & 1, (bytek >> 2) & 1, (bytek >> 3) & 1,
                (bytek >> 4) & 1, (bytek >> 5) & 1, (bytek >> 6) & 1, (bytek >> 7) & 1
            );

            // load 8 query bytes，扩展到 32-bit
            __m128i q8 = _mm_loadl_epi64((__m128i*)(queryCodes + i + k * 8));
            __m256i qv = _mm256_cvtepu8_epi32(q8);

            ipVec = _mm256_add_epi32(ipVec, _mm256_mullo_epi32(qv, bv));
        }
    }

    // horizontal add
    uint32_t ipArr[8];
    _mm256_storeu_si256((__m256i*)ipArr, ipVec);

    uint64_t ipSum = 0;
    for (int k = 0; k < 8; ++k) {
        ipSum += ipArr[k];
    }
    uint64_t ppcSum = dataCaqCode->totalPopcount; // 1bit popcount 在量化时已统计

    return queryDelta * (float)ipSum + queryMin * (float)ppcSum;
}
#endif

/**
 * AVX 加速版本：计算低位编码与解码后的 query 的内积
 * ip = sum_i [ ((q_i + 0.5) * qDelta + qMin) * dlow_i ]
 */
void InnerProductRestBit(
    const RestBitL2EstimatorCtxT *ctx,
    const CaqResBitQuantCodeT *dataCaqCode,
    float *resultOut
) {
    // 使用 SSE2 128-bit 整数路径在循环内计算 sum(q*d) 与 sum(d)
    // 再在循环外一次性解码并组合
    size_t D = ctx->dim;
    const QueryQuantCodeT *queryCode = ctx->queryQuantCode;
    const uint8_t *qCodes = queryCode->quantizedQueryOriCodes;
    const uint8_t *dCodes = dataCaqCode->storedCodes;
    const uint64_t precomputedSumD = dataCaqCode->sumCodes; // sum(d_i) 预计算，免去查询时累加

    __m128i accQD = _mm_setzero_si128(); // 4 x int32 局部累加器
    const __m128i zero = _mm_setzero_si128();

    size_t i = 0;
    for (; i + 16 <= D; i += 16) {
        __m128i q8 = _mm_loadu_si128((const __m128i *)(qCodes + i));
        __m128i d8 = _mm_loadu_si128((const __m128i *)(dCodes + i));

        // 拆分为 16-bit
        __m128i q16_lo = _mm_unpacklo_epi8(q8, zero);
        __m128i q16_hi = _mm_unpackhi_epi8(q8, zero);
        __m128i d16_lo = _mm_unpacklo_epi8(d8, zero);
        __m128i d16_hi = _mm_unpackhi_epi8(d8, zero);

        // sum(q*d)：成对 16-bit 乘加到 32-bit
        __m128i sum32_lo = _mm_madd_epi16(q16_lo, d16_lo);
        __m128i sum32_hi = _mm_madd_epi16(q16_hi, d16_hi);
        accQD = _mm_add_epi32(accQD, sum32_lo);
        accQD = _mm_add_epi32(accQD, sum32_hi);
    }

    // 将向量累加器归约到 64-bit 标量
    uint64_t sumQD = 0;
    uint64_t sumD  = precomputedSumD;
    {
        // 水平求和 accQD 4 个 32-bit
        __m128i tmp1 = _mm_hadd_epi32(accQD, accQD); // [a0+a1, a2+a3, a0+a1, a2+a3]
        __m128i tmp2 = _mm_hadd_epi32(tmp1, tmp1);   // [sum, sum, sum, sum]
        sumQD = (uint32_t)_mm_cvtsi128_si32(tmp2);
    }

    // 处理尾部
    for (; i < D; ++i) {
        sumQD += (uint64_t)qCodes[i] * (uint64_t)dCodes[i];
    }

    // 提取公因式一次解码
    double qDelta = (double)queryCode->delta;
    double qMin   = (double)queryCode->residualQueryMin;
    double ipReal = qDelta * (double)sumQD + (qMin + 0.5 * qDelta) * (double)sumD;
    *resultOut = (float)ipReal;
}

#elif (defined(SIMD_NEON_ENABLED))

/**
 * 1bit IP 估计（NEON 优化版，分离式 query layout，需要对 query 做转置）
 */
float estimateOneBitIp(
    const OneBitL2CaqEstimatorCtxT *ctx,
    const CaqOneBitQuantCodeT *dataCaqCode
) {
    size_t BlockNum = GetOneBitCodeSimdBlockNum(ctx->dim);
    size_t bytesPerBlock = GetBytesPerSimdBlock(); // 32 bytes in AVX, but 16 bytes for NEON
    const QueryQuantCodeT *queryCode = ctx->queryQuantCode;
    
    uint64_t ipEstimate = 0;
    uint64_t ppcScalar = dataCaqCode->totalPopcount; // 1bit popcount 在量化时已统计

    uint64_t tmp64[2];   // 用于 popcount
    uint8_t *qBlock;

    for (size_t j = 0; j < BlockNum; ++j) {
        const uint8_t *dBlock = dataCaqCode->storedCodes + j * bytesPerBlock;

        // 1bit popcount 已预存，无需查询时重复统计

        for (int plane = 0; plane < QUERY_QUANTIZER_NUM_BITS; ++plane) {
            uint64_t partialIp = 0;

            GetSepQueryCodeValue(
                ctx->queryQuantCtx,
                ctx->queryQuantCode,
                plane,
                j,
                &qBlock
            );

            // 单个 16B 块
            uint8x16_t q_lo = vld1q_u8(qBlock);
            uint8x16_t d_lo = vld1q_u8(dBlock);
            uint8x16_t a_lo = vandq_u8(q_lo, d_lo);
            vst1q_u8((uint8_t *)tmp64, a_lo);

            partialIp += __builtin_popcountll(tmp64[0]) + __builtin_popcountll(tmp64[1]);

            ipEstimate += partialIp
                * (1ULL << (QUERY_QUANTIZER_NUM_BITS - plane - 1));
        }
    }

    return queryCode->delta * (float)ipEstimate +
           (queryCode->residualQueryMin + 0.5f * queryCode->delta)
               * (float)ppcScalar;
}

void InnerProductRestBit(
    const RestBitL2EstimatorCtxT *ctx,
    const CaqResBitQuantCodeT *dataCaqCode,
    float *resultOut
) {
    size_t D = ctx->dim;
    const QueryQuantCodeT *queryCode = ctx->queryQuantCode;
    const uint8_t *qCodes = queryCode->quantizedQueryOriCodes;
    const uint8_t *dCodes = dataCaqCode->storedCodes;
    const uint64_t precomputedSumD = dataCaqCode->sumCodes;

    uint32x4_t accQD = vdupq_n_u32(0);  // sum(q * d)

    size_t i = 0;
    for (; i + 16 <= D; i += 16) {
        uint8x16_t q8 = vld1q_u8(qCodes + i);
        uint8x16_t d8 = vld1q_u8(dCodes + i);

        // 解码为 16 位
        uint16x8_t q16_lo = vmovl_u8(vget_low_u8(q8));
        uint16x8_t q16_hi = vmovl_u8(vget_high_u8(q8));
        uint16x8_t d16_lo = vmovl_u8(vget_low_u8(d8));
        uint16x8_t d16_hi = vmovl_u8(vget_high_u8(d8));

        // sum(q * d)
        accQD = vaddq_u32(accQD, vmull_u16(vget_low_u16(q16_lo), vget_low_u16(d16_lo)));
        accQD = vaddq_u32(accQD, vmull_u16(vget_high_u16(q16_lo), vget_high_u16(d16_lo)));
        accQD = vaddq_u32(accQD, vmull_u16(vget_low_u16(q16_hi), vget_low_u16(d16_hi)));
        accQD = vaddq_u32(accQD, vmull_u16(vget_high_u16(q16_hi), vget_high_u16(d16_hi)));
    }

    //  水平规约
    uint64_t sumQD =
        (uint64_t)vgetq_lane_u32(accQD, 0) +
        (uint64_t)vgetq_lane_u32(accQD, 1) +
        (uint64_t)vgetq_lane_u32(accQD, 2) +
        (uint64_t)vgetq_lane_u32(accQD, 3);

    uint64_t sumD = precomputedSumD;

    // 处理尾部
    for (; i < D; ++i) {
        sumQD += (uint64_t)qCodes[i] * (uint64_t)dCodes[i];
    }

    // 解码
    double qDelta = (double)queryCode->delta;
    double qMin   = (double)queryCode->residualQueryMin;
    double ipReal = qDelta * (double)sumQD +
                    (qMin + 0.5 * qDelta) * (double)sumD;

    *resultOut = (float)ipReal;
}

#else
    
/**
 * 计算单个数据库向量与 query 的 1 * N bit 内积距离
 */
float estimateOneBitIp(
    const OneBitL2CaqEstimatorCtxT *ctx,
    const CaqOneBitQuantCodeT *dataCaqCode
) {
    size_t D = ctx->dim;
    const QueryQuantCodeT *queryCode = ctx->queryQuantCode;
    const uint8_t *dataCodes = dataCaqCode->storedCodes;
    const float queryDelta = ctx->queryQuantCode->delta;
    const float queryMin = ctx->queryQuantCode->residualQueryMin + 0.5f * queryDelta;
    uint64_t ipEstimate = 0;
    uint64_t ppcScalar = dataCaqCode->totalPopcount; // 1bit popcount 在量化时已统计

    for (size_t i = 0; i < D; ++i) {
        // Extract i-th bit from dataCodes
        uint8_t bit = (dataCodes[i / 8] >> (i % 8)) & 1u;

        uint32_t qval;
        GetOriQueryCodeValue(ctx->queryQuantCtx, queryCode, i, &qval);

        ipEstimate += qval * bit;
    }

    return (queryDelta * (float)ipEstimate) + (queryMin * (float)ppcScalar);
}

/**
 * 解码 N bit 的 query 量化编码到实数域，然后
 * 计算单个数据库向量编码与 query 的内积距离
 */
void InnerProductRestBit(
    const RestBitL2EstimatorCtxT *ctx,
    const CaqResBitQuantCodeT *dataCaqCode,
    float *resultOut
) {
    size_t D = ctx->dim;

    // 1. 获取数据库向量和 query 向量的量化编码（原始 U8 编码）
    const QueryQuantCodeT *queryCode = ctx->queryQuantCode;
    const uint8_t *qCodes = queryCode->quantizedQueryOriCodes;
    const uint8_t *dCodes = dataCaqCode->storedCodes;
    const uint64_t precomputedSumD = dataCaqCode->sumCodes;

    // 2. 循环内仅做整数域计算：累计 sum(q_i * d_i) 与 sum(d_i)
    //    将解码的公共因子在循环外一次性应用，减少浮点运算
    uint64_t sumQD = 0; // 累计 q_i * d_i
    uint64_t sumD  = precomputedSumD; // d_i 总和已在编码阶段预计算
    for (size_t i = 0; i < D; ++i) {
        sumQD += (uint64_t)qCodes[i] * (uint64_t)dCodes[i];
    }

    // 3. 提取公因式解码：
    //    qReal_i = (q_i + 0.5) * qDelta + qMin
    //    sum(qReal_i * d_i) = qDelta * sum(q_i * d_i) + (qMin + 0.5*qDelta) * sum(d_i)
    double qDelta = (double)queryCode->delta;
    double qMin   = (double)queryCode->residualQueryMin;
    double ipReal = qDelta * (double)sumQD + (qMin + 0.5 * qDelta) * (double)sumD;

    *resultOut = (float)ipReal;
}
// void InnerProductRestBit(
//     const RestBitL2EstimatorCtxT *ctx,
//     const CaqResBitQuantCodeT *dataCaqCode,
//     float *resultOut
// ) {
//     size_t D = ctx->dim;
//     const uint32_t resBits = (uint32_t)(ctx->numBits - 1);

//     // 1.获取数据库向量和 query 向量的量化编码
//     const QueryQuantCodeT *queryCode = ctx->queryQuantCode;
//     const uint8_t *dataCodes = dataCaqCode->storedCodes;

//     float ip = 0.0;
//     // 2.解码 query 到实数域并计算内积，该写法性能较差，但为了便于读者理解还是保留
//     // uint32_t qval;
//     // float qValReal;
//     // for (size_t i = 0; i < D; ++i) {
//     //     GetOriQueryCodeValue(ctx->queryQuantCtx, queryCode, i, &qval);
//     //     // 
//     //     qValReal = ((float)qval + 0.5) * (float)ctx->queryQuantCode->delta + (float)ctx->queryQuantCode->residualQueryMin;
//     //     ip += qValReal * (float)dataCodes[i];
//     // }
//     *resultOut = (float)ip;
// }
#endif

void OneBitCaqEstimateDistance(
    const OneBitL2CaqEstimatorCtxT *ctx,
    const CaqOneBitQuantCodeT *dataCaqCode,
    float *distanceOut
) {
    // 1. 计算 1bit 内积距离
    float ipEstimate = estimateOneBitIp(ctx, dataCaqCode);

    // 2. 根据公式计算最终距离估算值
    float const_bound = 0.58;
    float est_error = 0.8;
    float ip_oa1_qq = (ipEstimate - (0.5 * ctx->queryQuantCode->residualQuerySum - const_bound * ctx->queryQuantCode->residualQueryL2Norm)) * 
                        (4 / est_error * ctx->oneOverSqrtD) * dataCaqCode->oriVecL2Norm;
    
    float L2Distance = ctx->queryQuantCode->residualQueryL2Sqr + 
                        dataCaqCode->oriVecL2Norm * dataCaqCode->oriVecL2Norm -
                        ip_oa1_qq;
    if (L2Distance < 0) L2Distance = 0.0f;
    *distanceOut = L2Distance;  // TODO: 若距离为 IP，return ip_oa1_qq * 0.5;

    // 3. 缓存 1bit 内积结果 和 原始向量 L2 范数，避免重复计算及IO
    InsertIntoCache(
        ctx->scannerCtx,
        0,
        ipEstimate,                 // 缓存原始 1bit 内积估计（未做 OA1 校正）
        dataCaqCode->oriVecL2Norm   // 缓存 原始向量 L2 范数
    );
}

void ResBitCaqEstimateDistance(
    const RestBitL2EstimatorCtxT *ctx,
    const CaqResBitQuantCodeT *dataCaqCode,
    float *distanceOut
) { 
    // 1. 从缓存中获取 1bit 内积结果 和 原始向量 L2 范数
    float oneBitRawIp, oL2Norm;
    FindInCache(ctx->scannerCtx, 0, &oneBitRawIp, &oL2Norm);    // TODO: 将 0 替换为 nodeId
    const float oL2Sqr = oL2Norm * oL2Norm;

    // 2. 计算低位贡献：将低位整数编码与 query 的实数解码进行内积，
    // 并用 CAQ 低位步长映射到实数域
    // [该计算方法是为了便于读者理解的写法，计算效率较低，实际应用中可优化为整数域内积后再映射到实数域]
    // float qDelta = ctx->queryQuantCode->delta;
    // float qMin = ctx->queryQuantCode->residualQueryMin;
    // const uint32_t resBits = (uint32_t)(ctx->numBits - 1);
    // const uint8_t *resCodes = dataCaqCode->storedCodes;
    // float lowIpReal = 0.0;
    // uint32_t qv;
    // for (size_t i = 0; i < ctx->dim; ++i) {
    //     GetOriQueryCodeValue(ctx->queryQuantCtx, ctx->queryQuantCode, i, &qv);
    //     float qReal = ((float)qv + 0.5f) * qDelta + qMin;
    //     uint32_t dlow = (uint32_t)(resCodes[i]);
    //     float dRealLow = ((float)dlow + 0.5f) * ctx->caqDelta + (-1.0f);
    //     lowIpReal += (float)qReal * (float)dRealLow;
    // }
    // // 3. 总内积（实数域）：oneBitRawIp 已经是实数域内积，直接相加
    // float ipOQScaled = (float)((float)oneBitRawIp + lowIpReal);
    // float ipOQ = dataCaqCode->rescaleFactor * ipOQScaled;

    // 2. 提取低位贡献：使用专门的函数计算低位整数编码与 query 的实数解码进行内积
    // 该写法便于后续对低位内积计算进行 SIMD 优化
    float realQueryIpF;
    InnerProductRestBit(ctx, dataCaqCode, &realQueryIpF);
    float caqDelta = ctx->caqDelta;
    float qDelta = ctx->queryQuantCode->delta;
    float qMin = ctx->queryQuantCode->residualQueryMin;
    float sumCodes = ctx->queryQuantCode->quantizedQuerySum;
    float dim = (float)ctx->dim;
    float sumQReal = qDelta * sumCodes + dim * qMin + 0.5 * qDelta * dim;
    float ipOQScaled = oneBitRawIp + realQueryIpF * caqDelta + (-1.0 + caqDelta / 2.0) * sumQReal;
    float ipOQ = dataCaqCode->rescaleFactor * ipOQScaled;

    float L2Distance = oL2Sqr + ctx->queryQuantCode->residualQueryL2Sqr - 2.0f * ipOQ;
    if (L2Distance < 0) L2Distance = 0.0f;

    *distanceOut = L2Distance;
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