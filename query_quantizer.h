#ifndef QUERY_QUANTIZER_H
#define QUERY_QUANTIZER_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>

// #define QUERY_QUANTIZER_PROFILE
#ifdef QUERY_QUANTIZER_PROFILE
#include <time.h>
#endif

#include "native_check.h"
#include "encoder.h"
#include "rotator.h"

// 默认 query 量化使用 8 bit 以保障精度，可能会影响计算速度
#define QUERY_QUANTIZER_NUM_BITS 4
// #define QUERY_QUANTIZER_NUM_BITS 8

#ifdef QUERY_QUANTIZER_PROFILE
static inline int64_t QueryQuantizerDiffNs(const struct timespec *start,
                         const struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000000000ll +
        (end->tv_nsec - start->tv_nsec);
}
#endif

/**
 * 使用 bitwise 进行 1bit 距离计算要求向量的量化编码布局必须是分离式布局（LAYOUT_SEPARATED）
 * 即：所有维度的第 0 bit 存储在一起，所有维度的第 1 bit 存储在一起，依此类推
 * 这样可以方便地使用位运算对所有维度的同一 bit进行并行计算
 * 示例：
 * [D 个 uint8_t，每个是 b bit] -> [b 个 bit-plane，每个 bit-plane 覆盖 D 个维度]
 */
// typedef enum {
//     LAYOUT_SEPARATED = 1,
//     LAYOUT_INTERLEAVED = 2
// } QueryQuantizerLayoutT;
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
    // float rotatedL2Norm;            // 旋转后向量的 L2 范数
    // QueryQuantizerLayoutT layout;   // 量化编码布局
} QueryQuantizerCtxT;

void GetCodesBytesSize(
    size_t dim,
    size_t *sepCodesSize,
    size_t *oriCodesSize
) {
    *sepCodesSize = 0;
#ifdef SIMD_MAX_CAPACITY
    const size_t BLOCK = SIMD_MAX_CAPACITY;      // 256 或 128
    const size_t BYTES_PER_BLOCK = BLOCK / 8;    // 32 或 16
    size_t numBlocks = (dim + BLOCK - 1) / BLOCK;
    *sepCodesSize = QUERY_QUANTIZER_NUM_BITS * numBlocks * BYTES_PER_BLOCK;     // 分离式布局，每个 bit plane 与 SIMD 寄存器大小对齐
#endif
    static_assert(QUERY_QUANTIZER_NUM_BITS <= 8, "QUERY_QUANTIZER_NUM_BITS must be <= 8");
    *oriCodesSize = dim * sizeof(uint8_t);                      // 交错式布局，每个维度一个 uint8_t
}

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
    uint8_t *quantizedQuerySepCodes;        // 量化编码结果指针，使用分离式布局存储，即每个 bit-plane 存储所有维度的对应 bit
    uint8_t *quantizedQueryOriCodes;        // 量化编码结果指针，使用原始交错式布局存储，即每个维度的所有 bit 存储在一起
    float residualQueryMin;                 // 残差查询向量的最小值
    float residualQueryMax;                 // 残差查询向量的最大值
    float delta;                            // 量化步长
    float residualQueryL2Sqr;               // 残差查询向量的平方 L2 范数
    float residualQueryL2Norm;              // 残差查询向量的 L2 范数
    float residualQuerySum;                 // 残差查询向量的元素和
    uint64_t quantizedQuerySum;             // 量化后整型编码 qi 的元素和（缓存 sum_qi）
} QueryQuantCodeT;

void CreateQueryQuantCode(QueryQuantCodeT **code, size_t dim) {
    *code = (QueryQuantCodeT *)malloc(sizeof(QueryQuantCodeT));
    size_t sepCodesLen, oriCodesLen;
    GetCodesBytesSize(dim, &sepCodesLen, &oriCodesLen);
#ifdef SIMD_MAX_CAPACITY
    (*code)->quantizedQuerySepCodes = (uint8_t *)malloc(sizeof(uint8_t) * sepCodesLen); // 分离式布局仅在支持 SIMD 时使用
#else
    (*code)->quantizedQuerySepCodes = NULL;
#endif
    (*code)->quantizedQueryOriCodes = (uint8_t *)malloc(sizeof(uint8_t) * oriCodesLen);
    (*code)->residualQueryMin = 0.0f;
    (*code)->residualQueryMax = 0.0f;
    (*code)->delta = 0.0f;
    (*code)->residualQueryL2Sqr = 0.0f;
    (*code)->residualQueryL2Norm = 0.0f;
    (*code)->residualQuerySum = 0.0f;
    (*code)->quantizedQuerySum = 0ull;
}

void DestroyQueryQuantCode(QueryQuantCodeT **code) {
    if (code == NULL || *code == NULL) {
        return;
    }
    if ((*code)->quantizedQuerySepCodes) {
        free((*code)->quantizedQuerySepCodes);
        (*code)->quantizedQuerySepCodes = NULL;
    }
    if ((*code)->quantizedQueryOriCodes) {
        free((*code)->quantizedQueryOriCodes);
        (*code)->quantizedQueryOriCodes = NULL;
    }
    free(*code);
    *code = NULL;
}

#ifdef SIMD_MAX_CAPACITY
void PrintCodesByDim(
    const uint8_t *codes,
    size_t dim,
    size_t b  // 通常 = QUERY_QUANTIZER_NUM_BITS
) {
    for (size_t i = 0; i < dim; ++i) {
        uint8_t val = codes[i];
        printf("dim[%3zu]: ", i);
        for (size_t bit = 0; bit < b; ++bit) {
            // MSB -> LSB 打印
            uint8_t v = (val >> (b - 1 - bit)) & 1u;
            printf("%u", v);
        }
        printf("\n");
    }
}

void PrintBitplanesLinearBits(
    const uint8_t *bitplanes,
    size_t b,
    size_t numBlocks
) {
    const size_t BLOCK = SIMD_MAX_CAPACITY;          // 256
    const size_t BYTES_PER_BLOCK = BLOCK / 8;        // 32
    const size_t TOTAL_BITS = numBlocks * BLOCK;     // 总共多少 bits per plane

    for (size_t p = 0; p < b; ++p) {
        printf("Plane %zu (logical bit %zu):\n", p, p);

        const uint8_t *plane =
            bitplanes + p * (numBlocks * BYTES_PER_BLOCK);

        // 按 bit 索引 0 到 TOTAL_BITS-1 顺序打印
        for (size_t bit_idx = 0; bit_idx < TOTAL_BITS; ++bit_idx) {
            size_t byte_id = bit_idx / 8;
            size_t bit_in_byte = bit_idx % 8;   // 0 = LSB, 7 = MSB

            uint8_t bit_val = (plane[byte_id] >> bit_in_byte) & 1u;
            printf("%u", bit_val);

            // 每 8 位加空格便于阅读
            if ((bit_idx + 1) % 8 == 0) {
                printf(" ");
            }
        }
        printf("\n\n");
    }
}

/**
 * 在转置后，bitplanes 的存储格式为：
 * [b 个 bit-plane，每个 bit-plane 覆盖 D 个维度]
 * 每个 bit-plane 内部按 block 存储，每个 block 包含 SIMD_MAX_CAPACITY 个维度的 bit 数据
 * 例如，对于 D=8，b=4 的情况，假设 SIMD_MAX_CAPACITY=8，则存储格式为：
 * Plane 0: [dim0_bit0, dim1_bit0, ..., dim7_bit0]
 * Plane 1: [dim0_bit1, dim1_bit1, ..., dim7_bit1]
 * Plane 2: [dim0_bit2, dim1_bit2, ..., dim7_bit2]
 * Plane 3: [dim0_bit3, dim1_bit3, ..., dim7_bit3]
 * 这里的 bit0 是最高有效位 (MSB)，bit3 是最低有效位 (LSB)
 * 例如，若一个维度的数字为 13 (二进制 1101)，则在 bit-plane 中的存储为：
 * dimX_bit0 = 1, dimX_bit1 = 1, dimX_bit2 = 0, dimX_bit3 = 1
 */
void TransposeU8ToBitplanes(
    const uint8_t *codes,
    size_t dim,
    size_t b,
    size_t *outNumBlocks,
    uint8_t **bitplanes     // 输出
) {
    const size_t BLOCK = SIMD_MAX_CAPACITY;      // 256
    const size_t BYTES_PER_BLOCK = BLOCK / 8;    // 32

    size_t numBlocks = (dim + BLOCK - 1) / BLOCK;
    if (outNumBlocks) {
        *outNumBlocks = numBlocks;
    }

    size_t totalBytes = b * numBlocks * BYTES_PER_BLOCK;

    /* 分配一整块连续内存，并清零（保证自动补 0） */
    uint8_t *buf = (uint8_t *)calloc(totalBytes, 1);
    if (!buf) {
        *bitplanes = NULL;
        return;
    }

    /* SWAR fast path when b == 4 (pure integer bit-gather per 8 codes). */
    if (b == 4) {
        const uint64_t PACK_MASK = 0x0101010101010101ull;
        const uint64_t PACK_MULT = 0x0102040810204080ull; // gathers bits from 8 bytes into one byte
        const size_t planeStride = numBlocks * BYTES_PER_BLOCK;

        for (size_t blk = 0; blk < numBlocks; ++blk) {
            size_t base = blk * BLOCK;
            size_t remain = dim > base ? dim - base : 0;
            if (remain == 0) break;
            if (remain > BLOCK) remain = BLOCK;

            uint8_t *planes[4];
            for (size_t bit = 0; bit < 4; ++bit) {
                planes[bit] = buf + (4 - 1 - bit) * planeStride + blk * BYTES_PER_BLOCK;
            }

            for (size_t offset = 0; offset < remain; offset += 8) {
                size_t chunk = remain - offset;
                if (chunk > 8) chunk = 8;

                uint64_t lane = 0;
                memcpy(&lane, codes + base + offset, chunk); // zero-padded tail chunk
                size_t byteIdx = offset >> 3;

                planes[0][byteIdx] = (uint8_t)(((lane >> 0) & PACK_MASK) * PACK_MULT >> 56);
                planes[1][byteIdx] = (uint8_t)(((lane >> 1) & PACK_MASK) * PACK_MULT >> 56);
                planes[2][byteIdx] = (uint8_t)(((lane >> 2) & PACK_MASK) * PACK_MULT >> 56);
                planes[3][byteIdx] = (uint8_t)(((lane >> 3) & PACK_MASK) * PACK_MULT >> 56);
            }
        }

        *bitplanes = buf;
        return;
    }

    /* 填充 bitplanes */
    for (size_t blk = 0; blk < numBlocks; ++blk) {
        size_t base = blk * BLOCK;

        for (size_t bit = 0; bit < b; ++bit) {
            /* plane-major + block */
            uint8_t *dst =
                buf + (b - 1 - bit) * (numBlocks * BYTES_PER_BLOCK)
                    + blk * BYTES_PER_BLOCK;

            for (size_t i = 0; i < BLOCK; ++i) {
                size_t idx = base + i;
                if (idx >= dim) break;

                uint8_t v = (codes[idx] >> bit) & 1u;
                if (v) {
                    /* i=0 -> bit63, i=63 -> bit0 */
                    size_t bitpos  = i;                 // 第几个维度就放在第几个 bit 上
                    size_t byte_id = bitpos >> 3;       // i / 8
                    size_t bit_id  = bitpos & 7;        // i % 8
                    dst[byte_id] |= (uint8_t)(1u << bit_id);
                }
            }
        }
    }

    *bitplanes = buf;
}

void FreeBitplanes(uint8_t **bitplanes, size_t b) {
    if (!bitplanes) return;
    for (size_t p = 0; p < b; ++p) {
        free(bitplanes[p]);
    }
    free(bitplanes);
}
#endif // SIMD_MAX_CAPACITY

/**
 * TODO: QuantizeQueryVector 是查询时开销，因此 SIMD 加速应该是必须的
 * QuantizeQueryVector 需要与 estimator 配合使用
 */
void QuantizeQueryVector(const QueryQuantizerCtxT *ctx,
                       const float *inputVector,
                       QueryQuantCodeT **outputCode) {
    size_t D = ctx->dim;
    float *centroid = ctx->centroid;
    float *rotatorMatrix = ctx->rotatorMatrix;
    float *residualVector = ctx->residualVector;
    float *rotatedVector = ctx->rotatedVector;
#ifdef QUERY_QUANTIZER_PROFILE
    struct timespec t_start, t_after_init, t_after_residual, t_after_rotate,
        t_after_range, t_after_quant, t_after_layout;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
#endif
    
    CreateQueryQuantCode(outputCode, D);
#ifdef QUERY_QUANTIZER_PROFILE
    clock_gettime(CLOCK_MONOTONIC, &t_after_init);
#endif

    // Step 1: 计算残差向量
    for (size_t i = 0; i < D; ++i) {
        residualVector[i] = inputVector[i] - centroid[i];
    }
#ifdef QUERY_QUANTIZER_PROFILE
    clock_gettime(CLOCK_MONOTONIC, &t_after_residual);
#endif

    // Step 2: 旋转向量
    rotateVector(ctx->rotatorMatrix, residualVector, rotatedVector, D);
#ifdef QUERY_QUANTIZER_PROFILE
    clock_gettime(CLOCK_MONOTONIC, &t_after_rotate);
#endif

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
#ifdef QUERY_QUANTIZER_PROFILE
    clock_gettime(CLOCK_MONOTONIC, &t_after_range);
#endif

    // 3.3 量化编码
    float l2Sqr = 0.0f;
    float sum = 0.0f;
    uint64_t codeSum = 0ull;
    uint8_t maxCode = (1u << QUERY_QUANTIZER_NUM_BITS) - 1;
    for (size_t i = 0; i < D; ++i) {
        float val = rotatedVector[i];
        float q = (val - minVal) / delta;
        if (q < 0.0f) q = 0.0f;                     // 下限
        if (q > (float)maxCode) q = (float)maxCode; // 上限
        uint8_t code = (uint8_t)floorf(q);          // cast<uint8_t>()
        // 非 SIMD 情况下仅使用按维度交错存储的编码；SIMD 情况稍后转置为按 plane 存储
        (*outputCode)->quantizedQueryOriCodes[i] = code;
        l2Sqr += val * val;                         
        sum += val;
        codeSum += (uint64_t)code;                 // 累加量化整型编码的元素和
    }
    (*outputCode)->residualQueryL2Sqr = l2Sqr;
    (*outputCode)->residualQueryL2Norm = sqrtf(l2Sqr);
    (*outputCode)->residualQuerySum = sum;
    (*outputCode)->quantizedQuerySum = codeSum;
#ifdef QUERY_QUANTIZER_PROFILE
    clock_gettime(CLOCK_MONOTONIC, &t_after_quant);
#endif

    // Step 4: 根据布局调整量化编码存储方式
#if (defined(SIMD_AVX_ENABLED) || defined(SIMD_NEON_ENABLED))
        // 分离式布局，需要将量化编码重新排列
    size_t numBlocks = 0;
    TransposeU8ToBitplanes(
        (*outputCode)->quantizedQueryOriCodes,
        D,
        QUERY_QUANTIZER_NUM_BITS,
        &numBlocks,
        &((*outputCode)->quantizedQuerySepCodes)
    );
#ifdef QUERY_QUANTIZER_PROFILE
    clock_gettime(CLOCK_MONOTONIC, &t_after_layout);
#endif
#endif
#ifdef QUERY_QUANTIZER_PROFILE
#ifndef (defined(SIMD_AVX_ENABLED) || defined(SIMD_NEON_ENABLED))
    t_after_layout = t_after_quant;
#endif
    int64_t total_ns = QueryQuantizerDiffNs(&t_start, &t_after_layout);
    if (total_ns > 0) {
        int64_t init_ns = QueryQuantizerDiffNs(&t_start, &t_after_init);
        int64_t residual_ns = QueryQuantizerDiffNs(&t_after_init, &t_after_residual);
        int64_t rotate_ns = QueryQuantizerDiffNs(&t_after_residual, &t_after_rotate);
        int64_t range_ns = QueryQuantizerDiffNs(&t_after_rotate, &t_after_range);
        int64_t quant_ns = QueryQuantizerDiffNs(&t_after_range, &t_after_quant);
        int64_t layout_ns = QueryQuantizerDiffNs(&t_after_quant, &t_after_layout);

        printf("QuantizeQueryVector timing: total=%.3f ms | init %.2f%% (%.3f ms) | residual %.2f%% (%.3f ms) | rotate %.2f%% (%.3f ms) | range %.2f%% (%.3f ms) | quantize %.2f%% (%.3f ms) | layout %.2f%% (%.3f ms)\n",
               (double)total_ns / 1e6,
               init_ns * 100.0 / total_ns, (double)init_ns / 1e6,
               residual_ns * 100.0 / total_ns, (double)residual_ns / 1e6,
               rotate_ns * 100.0 / total_ns, (double)rotate_ns / 1e6,
               range_ns * 100.0 / total_ns, (double)range_ns / 1e6,
               quant_ns * 100.0 / total_ns, (double)quant_ns / 1e6,
               layout_ns * 100.0 / total_ns, (double)layout_ns / 1e6);
    }
#endif // QUERY_QUANTIZER_PROFILE
}

void EasyEstimatorCtxPrepare(
    const QueryQuantizerCtxT *ctx,
    const float *inputVector,
    QueryQuantCodeT **outputCode
) {
    size_t D = ctx->dim;
    float *centroid = ctx->centroid;
    float *residualVector = ctx->residualVector;
    float *rotatedVector = ctx->rotatedVector;

    // 计算残差向量
    for (size_t i = 0; i < D; ++i) {
        residualVector[i] = inputVector[i] - centroid[i];
    }

    // 旋转向量
    rotateVector(ctx->rotatorMatrix, residualVector, rotatedVector, D);

    // 计算 L2 范数平方
    float l2Sqr = 0.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < D; ++i) {
        float val = rotatedVector[i];
        l2Sqr += val * val;
        sum += val;
    }

    // 缓存
    CreateQueryQuantCode(outputCode, D);
    (*outputCode)->residualQueryL2Sqr = l2Sqr;
    (*outputCode)->residualQueryL2Norm = sqrtf(l2Sqr);
     (*outputCode)->residualQuerySum = sum;
}

/**
 * 获取 query 量化编码在指定维度的原始布局的值
 */
void GetOriQueryCodeValue(
    const QueryQuantizerCtxT *outputCode,
    const QueryQuantCodeT *queryCode,
    size_t dimIndex,
    uint32_t *value
) {
    uint8_t val = queryCode->quantizedQueryOriCodes[dimIndex];
    *value = (uint32_t)(val);
}

#if (defined(SIMD_AVX_ENABLED) || defined(SIMD_NEON_ENABLED))
/**
 * 获取 query 量化编码在指定维度块和 bit 平面的指针（分离式布局）
 */
void GetSepQueryCodeValue(
    const QueryQuantizerCtxT *quantizerCtx,
    const QueryQuantCodeT *queryCode,
    size_t planeIndex,
    size_t blockIndex,
    uint8_t **value
) {
    size_t BlockNum = GetOneBitCodeSimdBlockNum(quantizerCtx->dim);
    size_t bytesPerBlock = GetBytesPerSimdBlock();
    size_t bytesPerPlane = bytesPerBlock * BlockNum; // 一整个 bitplane 的字节跨度
    *value = queryCode->quantizedQuerySepCodes + planeIndex * bytesPerPlane +
                blockIndex * GetBytesPerSimdBlock();
}
#endif

#endif // QUERY_QUANTIZER_H