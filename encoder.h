#ifndef ENCODER_H
#define ENCODER_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define ADJUST_ROUND_LIMIT 6
#define ADJUST_EPSILON 1e-8f

typedef struct {
    float max;
    float min;
    double delta;               // (max - min) / 2^b
    uint32_t* codes;             // 量化过程给 32bit 用于编码，拆分模块做裁剪
    float *floatCodes;          // 用于存储浮点型编码，便于后续调整
    double quantizedVectorL2Sqr;      // |o_a|^2, 量化后向量的平方 L2 范数
    double oriVecQuantVecIp; // <o, o_a>, 原始向量与量化后向量的内积
    float oriVecL2Sqr;            // |o|^2, 原始向量的平方 L2 范数
    float oriVecL2Norm;           // |o|, 原始向量的 L2 范数
    float rescaleFactor;        // |o|^2 / <o, o_a>
} CaqQuantCodeT;

void createCaqQuantCode(CaqQuantCodeT **res, size_t dim) {
    *res = (CaqQuantCodeT*)malloc(sizeof(CaqQuantCodeT));
    if (*res == NULL) {
        return;
    }
    (*res)->max = 0.0f;
    (*res)->min = 0.0f;
    (*res)->delta = 0.0;
    (*res)->codes = (uint32_t*)malloc(sizeof(uint32_t) * dim);
    (*res)->floatCodes = (float*)malloc(sizeof(float) * dim);
    (*res)->quantizedVectorL2Sqr = 0.0;
    (*res)->oriVecQuantVecIp = 0.0;
    (*res)->oriVecL2Sqr = 0.0f;
    (*res)->oriVecL2Norm = 0.0f;
    (*res)->rescaleFactor = 0.0f;
}

void destroyCaqQuantCode(CaqQuantCodeT **caqCode) {
    if (caqCode == NULL || *caqCode == NULL) {
        return;
    }
    if ((*caqCode)->codes) {
        free((*caqCode)->codes);
        (*caqCode)->codes = NULL;
    }
    if ((*caqCode)->floatCodes) {
        free((*caqCode)->floatCodes);
        (*caqCode)->floatCodes = NULL;
    }
    free(*caqCode);
    *caqCode = NULL;
}

void rescaleMaxToOne(CaqQuantCodeT *caqCode) {
    if (caqCode == NULL) return;
    if (!caqCode->max) return;
    const double rescaleRate = 1.0f / (double)caqCode->max;
    caqCode->min = (float)(caqCode->min * rescaleRate);
    caqCode->max = (float)(caqCode->max * rescaleRate);
    caqCode->delta = caqCode->delta * rescaleRate;
    caqCode->oriVecQuantVecIp = caqCode->oriVecQuantVecIp * rescaleRate;
    caqCode->quantizedVectorL2Sqr = caqCode->quantizedVectorL2Sqr * rescaleRate * rescaleRate;
}

typedef struct {
    size_t dimPadded;            // 对齐后的维度
    size_t numBits;              // 每个维度量化为 numBits 位
    int caqAdjustRoundLimit;     // 最大调整轮数
    float caqAdjustEpsilon;      // 调整阈值
    uint32_t codeMax;            // 量化编码的最大值
    uint32_t codeMin;            // 量化编码的最小值
} CaqEncodeConfig;

// 为一批向量创建一个公共的量化配置对象
void createCaqQuantConfig(
    size_t dim, 
    size_t numBits,
    CaqEncodeConfig **cfg_out
) {
    *cfg_out = (CaqEncodeConfig *)malloc(sizeof(CaqEncodeConfig));
    (*cfg_out)->dimPadded = dim;
    (*cfg_out)->numBits = numBits;
    (*cfg_out)->caqAdjustRoundLimit = (int)ADJUST_ROUND_LIMIT;
    (*cfg_out)->caqAdjustEpsilon = ADJUST_EPSILON;
    (*cfg_out)->codeMax = (uint32_t)((1u << numBits) - 1u);
    (*cfg_out)->codeMin = 0u;
}

void destroyCaqQuantConfig(CaqEncodeConfig **cfg) {
    if (cfg == NULL || *cfg == NULL) {
        return;
    }
    free(*cfg);
    *cfg = NULL;
}

// 调整量化编码以优化量化结果
void codeAdjustment(const float *originalVector, CaqQuantCodeT *caq, const CaqEncodeConfig *cfg) {
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

void encode(const float *originalVector, CaqQuantCodeT *caqCode, const CaqEncodeConfig *cfg) {
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
        codeAdjustment(originalVector, caqCode, cfg);
    }

    // 将值域放缩到 [-1, 1] 范围内，便于后续计算
    rescaleMaxToOne(caqCode);

    caqCode->oriVecL2Norm = (float)sqrt((double)caqCode->oriVecL2Sqr);
    if (caqCode->oriVecQuantVecIp) {
        caqCode->rescaleFactor = caqCode->oriVecL2Sqr / caqCode->oriVecQuantVecIp; // rescaleFactor = ||o||^2 / <o, o_a> * v_mx
    } else {
        caqCode->rescaleFactor = 0;
    }
}

#endif // ENCODER_H