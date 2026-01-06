#ifndef ENCODER_H
#define ENCODER_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>

#define ADJUST_ROUND_LIMIT 6
#define ADJUST_EPSILON 1e-8f

/**
 * 量化最终结果为下面两个结构体，分别存储 1bit 编码和剩余 N-1 bit 编码
 */
typedef struct {
    uint8_t* storedCodes;           // 每个维度的最高位 1bit 编码
    float oriVecL2Norm;             // |o|, 放缩后的原始向量的 L2 范数
    uint64_t totalPopcount;         // 预计算的 1bit popcount，总计所有 block
} CaqOneBitQuantCodeT;
typedef struct {
    uint8_t* storedCodes;           // 除了最高位 1bit 外，其余低位 N-1 bit 编码
    float rescaleFactor;            // 放缩因子，用于将量化向量还原到与原始向量相似的尺度
} CaqResBitQuantCodeT;

size_t getCaqOneBitQuantCodeSize(size_t dim) {
#ifdef SIMD_MAX_CAPACITY
    return GetBytesPerSimdBlock() * GetOneBitCodeSimdBlockNum(dim); // 按 SIMD 寄存器对齐分配存储空间
#else
    return (sizeof(uint8_t) * dim + 7) / 8;    // 若无需 SIMD 支持，按字节对齐分配存储空间
#endif
}

size_t getCaqResBitQuantCodeSize(size_t dim, uint32_t resBits) {
    assert(resBits > 0 && resBits <= 8 && "resBits must be in (0, 8]");
    return sizeof(uint8_t) * dim; // 默认一个维度分配一个字节的存储空间
}

void CreateCaqOneBitQuantCode(CaqOneBitQuantCodeT **res, size_t dim) {
    *res = (CaqOneBitQuantCodeT*)malloc(sizeof(CaqOneBitQuantCodeT));
    if (*res == NULL) {
        return;
    }
    size_t mallocSize = getCaqOneBitQuantCodeSize(dim);
    (*res)->storedCodes = (uint8_t*)malloc(mallocSize);
    (*res)->oriVecL2Norm = 0.0f;
    (*res)->totalPopcount = 0ull;
    memset((*res)->storedCodes, 0, mallocSize);
}

void CreateCaqResBitQuantCode(CaqResBitQuantCodeT **res, size_t dim, uint32_t resBits) {
    *res = (CaqResBitQuantCodeT*)malloc(sizeof(CaqResBitQuantCodeT));
    if (*res == NULL) {
        return;
    }
    size_t mallocSize = getCaqResBitQuantCodeSize(dim, resBits);
    (*res)->storedCodes = (uint8_t*)malloc(mallocSize);
    (*res)->rescaleFactor = 0.0f;
    memset((*res)->storedCodes, 0, mallocSize);
}

void DestroyCaqOneBitQuantCode(CaqOneBitQuantCodeT **code) {
    if (code == NULL || *code == NULL) {
        return;
    }
    if ((*code)->storedCodes) {
        free((*code)->storedCodes);
        (*code)->storedCodes = NULL;
    }
    (*code)->totalPopcount = 0ull;
    free(*code);
    *code = NULL;
}

void DestroyCaqResBitQuantCode(CaqResBitQuantCodeT **code) {
    if (code == NULL || *code == NULL) {
        return;
    }
    if ((*code)->storedCodes) {
        free((*code)->storedCodes);
        (*code)->storedCodes = NULL;
    }
    free(*code);
    *code = NULL;
}


/**
 * CaqQuantCodeT 是完整 N bit 量化编码结果的数据结构，仅用于量化过程中间结果存储
 * 最终需要存储的量化编码请使用 CaqOneBitQuantCodeT 和 CaqResBitQuantCodeT 结构体
 */
typedef struct {
    float max;
    float min;
    double delta;                   // (max - min) / 2^b
    uint32_t* codes;                // 量化过程给 32bit 用于编码，拆分模块做裁剪
    uint8_t* storedCodes;           // 最终存储的量化编码，按实际 numBits 位数存储
    float *floatCodes;              // 用于存储浮点型编码，便于后续调整
    double quantizedVectorL2Sqr;    // |o_a|^2, 量化后向量的平方 L2 范数
    float quantizedVectorSum;       // 量化后向量各维度编码之和
    double oriVecQuantVecIp;        // <o, o_a>, 原始向量与量化后向量的内积
    float oriVecL2Sqr;              // |o|^2, 原始向量的平方 L2 范数
    float oriVecL2Norm;             // |o|, 原始向量的 L2 范数
    float rescaleFactor;            // |o|^2 / <o, o_a>
} CaqQuantCodeT;

void CreateCaqQuantCode(CaqQuantCodeT **res, size_t dim) {
    *res = (CaqQuantCodeT*)malloc(sizeof(CaqQuantCodeT));
    if (*res == NULL) {
        return;
    }
    (*res)->max = 0.0f;
    (*res)->min = 0.0f;
    (*res)->delta = 0.0;
    (*res)->codes = (uint32_t*)malloc(sizeof(uint32_t) * dim);
    (*res)->storedCodes = (uint8_t*)malloc(sizeof(uint8_t) * dim);
    (*res)->floatCodes = (float*)malloc(sizeof(float) * dim);
    (*res)->quantizedVectorL2Sqr = 0.0;
    (*res)->oriVecQuantVecIp = 0.0;
    (*res)->oriVecL2Sqr = 0.0f;
    (*res)->oriVecL2Norm = 0.0f;
    (*res)->rescaleFactor = 0.0f;
}

void DestroyCaqQuantCode(CaqQuantCodeT **caqCode) {
    if (caqCode == NULL || *caqCode == NULL) {
        return;
    }
    if ((*caqCode)->codes) {
        free((*caqCode)->codes);
        (*caqCode)->codes = NULL;
    }
    if ((*caqCode)->storedCodes) {
        free((*caqCode)->storedCodes);
        (*caqCode)->storedCodes = NULL;
    }
    if ((*caqCode)->floatCodes) {
        free((*caqCode)->floatCodes);
        (*caqCode)->floatCodes = NULL;
    }
    free(*caqCode);
    *caqCode = NULL;
}

void RescaleMaxToOne(CaqQuantCodeT *caqCode) {
    if (caqCode == NULL) return;
    if (!caqCode->max) return;
    const double rescaleRate = 1.0f / (double)caqCode->max;
    caqCode->min = (float)(caqCode->min * rescaleRate);
    caqCode->max = (float)(caqCode->max * rescaleRate);
    caqCode->delta = caqCode->delta * rescaleRate;
    caqCode->oriVecQuantVecIp = caqCode->oriVecQuantVecIp * rescaleRate;
    caqCode->quantizedVectorL2Sqr = caqCode->quantizedVectorL2Sqr * rescaleRate * rescaleRate;
}

/**
 * 一批向量使用相同的量化配置对象进行量化
 */
typedef struct {
    size_t dimPadded;            // 对齐后的维度
    size_t numBits;              // 每个维度量化为 numBits 位
    int caqAdjustRoundLimit;     // 最大调整轮数
    float caqAdjustEpsilon;      // 调整阈值
    uint32_t codeMax;            // 量化编码的最大值
    uint32_t codeMin;            // 量化编码的最小值
    bool useSeparateStorage;     // 是否使用分离式存储布局，若为 true ，则最高位 1bit 与剩余 N-1 bit 分开存储，且需要进行放缩
} CaqEncodeConfig;

void CreateCaqEncodeConfig(
    size_t dim, 
    size_t numBits,
    bool useSeparateStorage,
    CaqEncodeConfig **cfg_out
) {
    *cfg_out = (CaqEncodeConfig *)malloc(sizeof(CaqEncodeConfig));
    (*cfg_out)->dimPadded = dim;
    (*cfg_out)->numBits = numBits;
    (*cfg_out)->caqAdjustRoundLimit = (int)ADJUST_ROUND_LIMIT;
    (*cfg_out)->caqAdjustEpsilon = ADJUST_EPSILON;
    (*cfg_out)->codeMax = (uint32_t)((1u << numBits) - 1u);
    (*cfg_out)->codeMin = 0u;
    (*cfg_out)->useSeparateStorage = useSeparateStorage;
}

void DestroyCaqQuantConfig(CaqEncodeConfig **cfg) {
    if (cfg == NULL || *cfg == NULL) {
        return;
    }
    free(*cfg);
    *cfg = NULL;
}

// 调整量化编码以优化量化结果
void CodeAdjustment(const float *originalVector, CaqQuantCodeT *caq, const CaqEncodeConfig *cfg) {
    float min = caq->min;
    double delta = caq->delta;
    uint32_t *codes = caq->codes;
    double *OriginalQuantizedIpPtr = &caq->oriVecQuantVecIp;
    double *QuantizedL2SqrPtr = &caq->quantizedVectorL2Sqr;

    double re_eps = cfg->caqAdjustEpsilon * (*QuantizedL2SqrPtr);
    int adj_cnt = 1;
    int round = 1;
    while (adj_cnt) {
        adj_cnt = 0;
        for (size_t j = 0; j < cfg->dimPadded; j++) {
            int curr_adj_cnt = 0;
            float o = originalVector[j];
            double oa = ((double)codes[j] + 0.5) * delta + (double)min;
            uint32_t c = codes[j];
            double oa_l2sqr_tmp = (*QuantizedL2SqrPtr) - oa * oa;
            double ip_delta = delta * (double)o;

            while (c < cfg->codeMax) {
                double new_q = oa + delta;
                double new_length = oa_l2sqr_tmp + new_q * new_q;
                double new_ip = (*OriginalQuantizedIpPtr) + ip_delta;
                if (((*OriginalQuantizedIpPtr) * (*OriginalQuantizedIpPtr) + re_eps) * new_length >=
                    new_ip * new_ip * (*QuantizedL2SqrPtr))
                    break;
                c++;
                *OriginalQuantizedIpPtr = new_ip;
                oa = new_q;
                *QuantizedL2SqrPtr = new_length;
                curr_adj_cnt++;
            }

            while (c > cfg->codeMin) {
                double new_q = oa - delta;
                double new_length = oa_l2sqr_tmp + new_q * new_q;
                double new_ip = (*OriginalQuantizedIpPtr) - ip_delta;
                if (((*OriginalQuantizedIpPtr) * (*OriginalQuantizedIpPtr) + re_eps) * new_length >=
                    new_ip * new_ip * (*QuantizedL2SqrPtr))
                    break;
                c--;
                *OriginalQuantizedIpPtr = new_ip;
                oa = new_q;
                *QuantizedL2SqrPtr = new_length;
                curr_adj_cnt++;
            }
            if (codes[j] != c) {
                codes[j] = c;
                adj_cnt += curr_adj_cnt;
            }
        }
        if (cfg->caqAdjustRoundLimit && round >= cfg->caqAdjustRoundLimit) {
            break;
        }
        round++;

        {
            double check_oa_l2sqr = 0.0;
            double check_ip = 0.0;
            for (size_t j = 0; j < cfg->dimPadded; ++j) {
                float o = originalVector[j];
                uint32_t qc = codes[j];
                double q = ((double)qc + 0.5) * delta + (double)min;
                check_ip += q * (double)o;
                check_oa_l2sqr += q * q;
            }

            *QuantizedL2SqrPtr = check_oa_l2sqr;
            *OriginalQuantizedIpPtr = check_ip;
        }
    }
}

void Encode(const float *originalVector, CaqQuantCodeT *caqCode, const CaqEncodeConfig *cfg) {
    // 清零状态，避免 caqCode 复用导致的数据遗留
    caqCode->max = 0.0f;
    caqCode->min = 0.0f;
    caqCode->delta = 0.0f;
    caqCode->oriVecL2Sqr = 0.0f;
    caqCode->oriVecL2Norm = 0.0f;
    caqCode->rescaleFactor = 0.0f;
    
    if (cfg->numBits == 0) {
        return;
    }

    // Step 1: 初始化量化参数
    for (size_t i = 0; i < cfg->dimPadded; ++i) {
        float absVal = (float)fabs((double)originalVector[i]);
        if (absVal > caqCode->max) {
            caqCode->max = absVal;
        }
    }
    caqCode->min = -caqCode->max;   // 使用对称值域
    caqCode->delta = (double)(caqCode->max - caqCode->min) / (double)(cfg->codeMax + 1u);

    double originalVectorSum = 0.0;
    float sum = 0.0f;
    for (size_t i = 0; i < cfg->dimPadded; ++i) {
        sum += originalVector[i];
    }
    originalVectorSum = (double)(sum);   // 这里由于浮点加法不可交换，而 CPP 版本 double vec_sum = o.sum(); 使用了 Eigen 的 SIMD 加法，可能会有微小差异

    // Step 2: 初步计算量化编码
    if (caqCode->min != caqCode->max) {
        // 非全零向量的处理
        for (size_t i = 0; i < cfg->dimPadded; ++i) {
            float codes = (float)floor(((double)originalVector[i] - (double)caqCode->min) / (caqCode->delta));
            if (codes < 0) codes = 0;
            if (codes > cfg->codeMax) { 
                codes = cfg->codeMax; 
            }
            caqCode->floatCodes[i] = codes;
        }
    } else {
        // 全零向量的特殊处理
        for (size_t i = 0; i < cfg->dimPadded; ++i) {
            caqCode->floatCodes[i] = 0.0f;
        } 
    }

    // 遍历计算量化结果的统计量
    double oriQuantCodeIp = 0;
    uint64_t quantCodeL2Sqr = 0;
    uint64_t quantCodeSum = 0;
    caqCode->oriVecL2Sqr = 0.0f;
    for (size_t i = 0; i < cfg->dimPadded; ++i) {
        float codeFloat = caqCode->floatCodes[i];
        float originalValue = originalVector[i];
        caqCode->codes[i] = (uint32_t)codeFloat; // 转存为整数编码
        int codes = (int)caqCode->codes[i];

        oriQuantCodeIp += codeFloat * originalValue;
        quantCodeL2Sqr += (uint64_t)(codes * codes);
        quantCodeSum += (uint64_t)(codes);
        caqCode->oriVecL2Sqr += originalValue * originalValue;
    }

    caqCode->oriVecQuantVecIp = oriQuantCodeIp * caqCode->delta + (caqCode->min + 0.5 * caqCode->delta) * originalVectorSum;
    // 使用论文中公式 (12) 解码 quantizedValue ，合并同类项后即可获得下面的表达式，如果需要复原单维量化结果，可以使用该公式  float quantizedValue = (codes + 0.5)* caqCode->delta + caqCode->min
    caqCode->quantizedVectorL2Sqr = caqCode->delta * caqCode->delta * quantCodeL2Sqr + 
                                (caqCode->delta * caqCode->delta + 2 * caqCode->delta * caqCode->min) * quantCodeSum +
                                (0.25 * caqCode->delta * caqCode->delta + caqCode->delta * caqCode->min + caqCode->min * caqCode->min) * cfg->dimPadded;

    // 只有在非全零向量的情况下，才进行后续的编码调整
    if (caqCode->quantizedVectorL2Sqr) {
        CodeAdjustment(originalVector, caqCode, cfg);
    }

    if (cfg->useSeparateStorage) {
        // 如果使用分离式存储布局，将值域放缩到 [-1, 1] 范围内，便于后续计算，并设置 rescaleFactor
        RescaleMaxToOne(caqCode);
        if (caqCode->oriVecQuantVecIp) {
            caqCode->rescaleFactor = caqCode->oriVecL2Sqr / caqCode->oriVecQuantVecIp; // rescaleFactor = ||o||^2 / <o, o_a> * v_mx
        } else {
            caqCode->rescaleFactor = 0;
        }
    } else {
        // 更新 caqCode->oriVecQuantVecIp 为新编码解码的内积
        double new_oriQuantIp = 0.0;
        for (size_t i = 0; i < cfg->dimPadded; ++i) {
            float originalValue = originalVector[i];
            uint32_t codes = caqCode->codes[i];
            double q = ((double)codes + 0.5) * caqCode->delta + (double)caqCode->min;
            new_oriQuantIp += q * (double)originalValue;
        }
        caqCode->oriVecQuantVecIp = new_oriQuantIp;
    }

    caqCode->oriVecL2Norm = (float)sqrt((double)caqCode->oriVecL2Sqr);
}

// 将完整的每个维度 N bit 量化编码拆分为每个维度 1 bit 和 N-1 bit 两部分进行存储
void SeparateCode(
    const CaqEncodeConfig *cfg,
    const CaqQuantCodeT *caqCode,
    CaqOneBitQuantCodeT **oneBitCode,
    CaqResBitQuantCodeT **resBitCode
) {
    CreateCaqOneBitQuantCode(oneBitCode, cfg->dimPadded);
    CreateCaqResBitQuantCode(resBitCode, cfg->dimPadded, cfg->numBits - 1);
    CaqOneBitQuantCodeT *oneBit = *oneBitCode;
    CaqResBitQuantCodeT *resBit = *resBitCode;
    oneBit->oriVecL2Norm = caqCode->oriVecL2Norm;
    resBit->rescaleFactor = caqCode->rescaleFactor;
    uint32_t numBits = (uint32_t)cfg->numBits;
    uint32_t resBits = numBits - 1;
    oneBit->totalPopcount = 0ull;

    for (size_t i = 0; i < cfg->dimPadded; ++i) {
        uint32_t code = caqCode->codes[i];
        uint8_t highBit = (uint8_t)((code >> resBits) & 1);       // 取最高位
        uint32_t lowBits = code & ((1u << resBits) - 1u);         // 取低 N-1 位

        // 存储最高位 1bit
        size_t byteIdx1 = i / 8;
        size_t bitIdx1 = i % 8;
        oneBit->storedCodes[byteIdx1] |= (highBit << bitIdx1);
        oneBit->totalPopcount += (uint64_t)highBit;

        // 存储低位 N-1 bit
        resBit->storedCodes[i] = lowBits;
    }
}

void StoreCode(
    const CaqEncodeConfig *cfg,
    CaqQuantCodeT *caqCode
) {
    assert(cfg->numBits <= 8 && "Only support numBits <= 8 in StoreCode");
    for (size_t i = 0; i < cfg->dimPadded; ++i) {
        uint32_t code = caqCode->codes[i];
        caqCode->storedCodes[i] = (uint8_t)(code & 0xFF);
    }
}

void GetResBitQuantCode(
    const CaqResBitQuantCodeT *resBitCode,
    size_t dim,
    uint32_t resBits,
    uint32_t *outCodes
) {
    for (size_t i = 0; i < dim; ++i) {
        outCodes[i] = (uint32_t)(resBitCode->storedCodes[i]);
    }
}

#endif // ENCODER_H