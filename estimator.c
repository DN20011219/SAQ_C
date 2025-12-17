
#define _POSIX_C_SOURCE 200809L
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "data_quantizer.h"
#include "query_quantizer.h"
#include "rotator.h"
#include "estimator.h"

static float frand_unit(void) {
    return (float)rand() / (float)RAND_MAX;          // [0, 1]
}

static float frand_signed(void) {
    return (frand_unit() * 2.0f) - 1.0f;             // [-1, 1]
}

static void fill_vector(float *v, size_t dim) {
    for (size_t i = 0; i < dim; ++i) {
        v[i] = frand_signed();
    }
}

static void compute_centroid_from_data(const float *data, float *centroid, size_t dim) {
    // 简单质心：使用数据的均值作为每个维度的质心估计（可替换为更复杂方式）
    float mean = 0.0f;
    for (size_t i = 0; i < dim; ++i) mean += data[i];
    mean /= (float)dim;
    for (size_t i = 0; i < dim; ++i) centroid[i] = mean;
}

static double elapsed_ms(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 +
           (end.tv_nsec - start.tv_nsec) / 1e6;
}

typedef struct {
    float *data;
    CaqQuantCodeT *dataCaqCode;
    CaqOneBitQuantCodeT *oneBitCode;
    CaqResBitQuantCodeT *resBitCode;
} DataItem;

typedef struct {
    float *query;
} QueryItem;

int main(int argc, char **argv)
{
    const size_t numBits = 9;// 固定为 9 bit 量化
    size_t dim = 512;
    int data_count = 100000; // 10W级别
    int query_count = 100;   // 100-1000
    int loop_count = 1;      // LOOP 测试次数（每对 query-data 重复估算次数）
    unsigned seed = (unsigned)time(NULL);
    float centroid_eps = 0.0f; // 关闭“接近 query”的质心模式，改为从 data 计算
    if (argc > 1) {
        data_count = atoi(argv[1]);
        if (data_count <= 0) data_count = 100000;
    }
    if (argc > 2) {
        query_count = atoi(argv[2]);
        if (query_count <= 0) query_count = 100;
    }
    if (argc > 3) {
        dim = (size_t)atoi(argv[3]);
        if (dim == 0) dim = 512;
    }
    if (argc > 4) {
        seed = (unsigned)atoi(argv[4]);
        if (seed == 0) seed = 1u;
    }
    if (argc > 5) {
        loop_count = atoi(argv[5]);
        if (loop_count <= 0) loop_count = 1;
    }

    srand(seed);

    float *centroid = (float *)malloc(sizeof(float) * dim);
    float *data = (float *)malloc(sizeof(float) * dim);
    float *query = (float *)malloc(sizeof(float) * dim);

    L2CaqQuantizerCtxT *quantizerCtx = NULL;
    CaqEncodeConfig *caqCfg = NULL;
    CaqQuantizerInit(&quantizerCtx, dim, centroid, numBits);
    CreateCaqQuantConfig(dim, numBits, &caqCfg);

    CaqScannerCtxT *scannerCtx = NULL;
    CreateCaqScannerCtx(&scannerCtx);

    double absErrSum = 0.0;
    double relErrSum = 0.0;
    double maxAbsErr = 0.0;
    double maxRelErr = 0.0;
    double restAbsErrSum = 0.0;
    double restRelErrSum = 0.0;
    double restMaxAbsErr = 0.0;
    double restMaxRelErr = 0.0;

    DataItem *data_items = (DataItem *)calloc((size_t)data_count, sizeof(DataItem));
    if (!data_items) {
        fprintf(stderr, "alloc cache failed\n");
        return 1;
    }
    QueryItem *query_items = (QueryItem *)calloc((size_t)query_count, sizeof(QueryItem));
    if (!query_items) {
        fprintf(stderr, "alloc query cache failed\n");
        return 1;
    }

    struct timespec t0, t1;
    double init_time_ms = 0.0;
    double estimate_time_ms = 0.0;
    double rest_estimate_time_ms = 0.0;
    size_t loop_unstable_pairs = 0;   // 有非零差异的 pair 数
    float loop_max_abs_diff = 0.0f;   // 最大差异

    // Phase 1: 生成所有 data 和 query；计算全局质心；用全局质心量化所有 data
    for (int i = 0; i < data_count; ++i) {
        fill_vector(data, dim);
        data_items[i].data = (float *)malloc(sizeof(float) * dim);
        memcpy(data_items[i].data, data, sizeof(float) * dim);
    }
    for (int q = 0; q < query_count; ++q) {
        fill_vector(query, dim);
        query_items[q].query = (float *)malloc(sizeof(float) * dim);
        memcpy(query_items[q].query, query, sizeof(float) * dim);
    }
    // 计算全局质心：按所有 data 的均值
    memset(centroid, 0, sizeof(float) * dim);
    for (int i = 0; i < data_count; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            centroid[d] += data_items[i].data[d];
        }
    }
    for (size_t d = 0; d < dim; ++d) centroid[d] /= (float)data_count;

    // 用全局质心量化所有 data
    // 更新量化器上下文中的质心指针（已指向 centroid 缓冲区，内容已更新即可）
    for (int i = 0; i < data_count; ++i) {
        CaqQuantizeVector(quantizerCtx, data_items[i].data, &data_items[i].dataCaqCode);
        CaqSeparateStoredCodes(caqCfg, data_items[i].dataCaqCode, &data_items[i].oneBitCode, &data_items[i].resBitCode);
    }
    // Phase 2: 对每个 query 初始化一次估算器（使用全局质心），并遍历所有 data 的 oneBit 码进行估算
    for (int q = 0; q < query_count; ++q) {
        OneBitL2CaqEstimatorCtxT *estimatorCtx = NULL;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        OneBitCaqEstimatorInit(&estimatorCtx,
                      dim,
                      QUERY_QUANTIZER_NUM_BITS,
                      quantizerCtx->rotatorMatrix,
                      centroid,
                      query_items[q].query,
                      scannerCtx);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        init_time_ms += elapsed_ms(t0, t1);

        RestBitL2EstimatorCtxT *restCtx = NULL;
        CreateRestBitL2EstimatorCtx(&restCtx,
                                    dim,
                                    numBits,
                                    estimatorCtx->queryQuantCtx,
                                    estimatorCtx->queryQuantCode,
                                    scannerCtx);

        for (int i = 0; i < data_count; ++i) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            float baselineEstimate;
            OneBitCaqEstimateDistance(estimatorCtx, data_items[i].oneBitCode, &baselineEstimate);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            estimate_time_ms += elapsed_ms(t0, t1);
            float trueDistance = FloatL2(data_items[i].data, query_items[q].query, dim);

            double absErr = fabs((double)baselineEstimate - (double)trueDistance);
            double relErr = trueDistance ? absErr / (double)trueDistance : 0.0;
            absErrSum += absErr;
            relErrSum += relErr;
            if (absErr > maxAbsErr) maxAbsErr = absErr;
            if (relErr > maxRelErr) maxRelErr = relErr;

            // RestBit 精排距离测试
            clock_gettime(CLOCK_MONOTONIC, &t0);
            float restDistance = 0.0f;
            ResBitCaqEstimateDistance(restCtx, data_items[i].resBitCode, &restDistance);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            rest_estimate_time_ms += elapsed_ms(t0, t1);

            double restAbsErr = fabs((double)restDistance - (double)trueDistance);
            double restRelErr = trueDistance ? restAbsErr / (double)trueDistance : 0.0;
            restAbsErrSum += restAbsErr;
            restRelErrSum += restRelErr;
            if (restAbsErr > restMaxAbsErr) restMaxAbsErr = restAbsErr;
            if (restRelErr > restMaxRelErr) restMaxRelErr = restRelErr;

            // LOOP 稳定性测试：多次重复估算并比较与 baseline 的差异
            if (loop_count > 1) {
                int unstable = 0;
                for (int l = 1; l < loop_count; ++l) {
                    clock_gettime(CLOCK_MONOTONIC, &t0);
                    float e2;
                    OneBitCaqEstimateDistance(estimatorCtx, data_items[i].oneBitCode, &e2);
                    clock_gettime(CLOCK_MONOTONIC, &t1);
                    estimate_time_ms += elapsed_ms(t0, t1);
                    float diff = fabsf(e2 - baselineEstimate);
                    if (diff > 0.0f) unstable = 1;
                    if (diff > loop_max_abs_diff) loop_max_abs_diff = diff;
                }
                if (unstable) loop_unstable_pairs++;
            }
        }
        DestroyRestBitL2EstimatorCtx(&restCtx);
        OneBitCaqEstimatorDestroy(&estimatorCtx);
    }
    double denom = (double)data_count * (double)query_count;
    double meanAbsErr = absErrSum / denom;
    double meanRelErr = relErrSum / denom;
    double restMeanAbsErr = restAbsErrSum / denom;
    double restMeanRelErr = restRelErrSum / denom;

    printf("Data count: %d\n", data_count);
    printf("Query count: %d\n", query_count);
    printf("Loop count: %d\n", loop_count);
    printf("Dim: %zu, numBits: %zu\n", dim, numBits);
    printf("Seed: %u\n", seed);
    printf("Init time (sum): %.3f ms\n", init_time_ms);
    printf("Estimate time (sum): %.3f ms\n", estimate_time_ms);
    printf("Rest estimate time (sum): %.3f ms\n", rest_estimate_time_ms);
    if (loop_count > 1) {
        printf("Loop unstable pairs: %zu\n", loop_unstable_pairs);
        printf("Loop max abs diff: %.6f\n", loop_max_abs_diff);
    }
    printf("Rest mean abs error: %.6f\n", (float)restMeanAbsErr);
    printf("Rest mean rel error: %.6f\n", (float)restMeanRelErr);
    printf("Rest max abs error: %.6f\n", (float)restMaxAbsErr);
    printf("Rest max rel error: %.6f\n", (float)restMaxRelErr);
    printf("Mean abs error: %.6f\n", (float)meanAbsErr);
    printf("Mean rel error: %.6f\n", (float)meanRelErr);
    printf("Max abs error: %.6f\n", (float)maxAbsErr);
    printf("Max rel error: %.6f\n", (float)maxRelErr);

    // Cleanup
    for (int i = 0; i < data_count; ++i) {
        DestroyCaqQuantCode(&data_items[i].dataCaqCode);
        DestroyCaqOneBitQuantCode(&data_items[i].oneBitCode);
        DestroyCaqResBitQuantCode(&data_items[i].resBitCode);
        free(data_items[i].data);
    }
    free(data_items);
    for (int q = 0; q < query_count; ++q) {
        free(query_items[q].query);
    }
    free(query_items);
    CaqQuantizerDestroy(&quantizerCtx);
    DestroyCaqQuantConfig(&caqCfg);
    DestroyCaqScannerCtx(&scannerCtx);
    free(centroid);
    free(data);
    free(query);
    return 0;
}