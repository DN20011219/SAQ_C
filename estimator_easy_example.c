#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "data_quantizer.h"
#include "query_quantizer.h"
#include "rotator.h"
#include "encoder.h"
#include "estimator.h"      // 提供 FloatL2 计算真值距离
#include "estimator_easy.h"

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

static double elapsed_ms(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 +
           (end.tv_nsec - start.tv_nsec) / 1e6;
}

typedef struct {
    float *data;
    CaqQuantCodeT *dataCaqCode;
} DataItem;

typedef struct {
    float *query;
} QueryItem;

int main(int argc, char **argv)
{
    size_t numBits = 8;            // estimator_easy 需要完整存储，受 StoreCode 限制 numBits<=8
    size_t dim = 512;
    int data_count = 100000;       // 10W 级别数据
    int query_count = 100;         // 100-1000 queries
    int loop_count = 1;            // 每对 query-data 重复估算次数
    unsigned seed = (unsigned)time(NULL);
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
    if (argc > 6) {
        size_t nb = (size_t)atoi(argv[6]);
        if (nb == 0 || nb > 8) nb = 8;  // StoreCode 仅支持 numBits<=8
        numBits = nb;
    }

    srand(seed);

    float *centroid = (float *)malloc(sizeof(float) * dim);
    float *data = (float *)malloc(sizeof(float) * dim);
    float *query = (float *)malloc(sizeof(float) * dim);

    CaqQuantizerCtxT *quantizerCtx = NULL;
    // useSeparateCodes=false: 保留完整 numBits 编码，便于 estimator_easy 直接使用
    CaqQuantizerInit(&quantizerCtx, dim, centroid, numBits, false);
    CaqEncodeConfig *caqCfg = quantizerCtx->encodeConfig;

    double absErrSum = 0.0;
    double relErrSum = 0.0;
    double maxAbsErr = 0.0;
    double maxRelErr = 0.0;

    DataItem *data_items = (DataItem *)calloc((size_t)data_count, sizeof(DataItem));
    if (!data_items) {
        fprintf(stderr, "alloc cache failed\n");
        return 1;
    }
    QueryItem *query_items = (QueryItem *)calloc((size_t)query_count, sizeof(QueryItem));
    if (!query_items) {
        fprintf(stderr, "alloc query cache failed\n");
        free(data_items);
        return 1;
    }

    struct timespec t0, t1;
    double init_time_ms = 0.0;
    double estimate_time_ms = 0.0;
    size_t loop_unstable_pairs = 0;
    float loop_max_abs_diff = 0.0f;

    // Phase 1: 生成 data/query；计算全局质心；用质心量化 data
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

    memset(centroid, 0, sizeof(float) * dim);
    for (int i = 0; i < data_count; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            centroid[d] += data_items[i].data[d];
        }
    }
    for (size_t d = 0; d < dim; ++d) centroid[d] /= (float)data_count;

    for (int i = 0; i < data_count; ++i) {
        CaqQuantizeVector(quantizerCtx, data_items[i].data, &data_items[i].dataCaqCode);
        CaqMergeStoredCodes(caqCfg, data_items[i].dataCaqCode);
    }

    // Phase 2: 每个 query 构建 estimator_easy 上下文，直接用完整编码估算距离
    for (int q = 0; q < query_count; ++q) {
        QueryQuantizerCtxT *queryQuantCtx = NULL;
        QueryQuantCodeT *queryQuantCode = NULL;
        FullBitL2EstimatorCtxT *estimatorCtx = NULL;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        QueryQuantizerCtxInit(&queryQuantCtx, dim, centroid, quantizerCtx->rotatorMatrix);
        EasyEstimatorCtxPrepare(queryQuantCtx, query_items[q].query, &queryQuantCode);
        CreateFullBitL2EstimatorCtx(&estimatorCtx, dim, numBits, queryQuantCtx, queryQuantCode);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        init_time_ms += elapsed_ms(t0, t1);

        for (int i = 0; i < data_count; ++i) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            float estimate = 0.0f;
            CaqEstimateDistance(estimatorCtx, data_items[i].dataCaqCode, &estimate);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            estimate_time_ms += elapsed_ms(t0, t1);

            float trueDistance = FloatL2(data_items[i].data, query_items[q].query, dim);
            double absErr = fabs((double)estimate - (double)trueDistance);
            double relErr = trueDistance ? absErr / (double)trueDistance : 0.0;
            absErrSum += absErr;
            relErrSum += relErr;
            if (absErr > maxAbsErr) maxAbsErr = absErr;
            if (relErr > maxRelErr) maxRelErr = relErr;

            if (loop_count > 1) {
                int unstable = 0;
                for (int l = 1; l < loop_count; ++l) {
                    clock_gettime(CLOCK_MONOTONIC, &t0);
                    float e2 = 0.0f;
                    CaqEstimateDistance(estimatorCtx, data_items[i].dataCaqCode, &e2);
                    clock_gettime(CLOCK_MONOTONIC, &t1);
                    estimate_time_ms += elapsed_ms(t0, t1);
                    float diff = fabsf(e2 - estimate);
                    if (diff > 0.0f) unstable = 1;
                    if (diff > loop_max_abs_diff) loop_max_abs_diff = diff;
                }
                if (unstable) loop_unstable_pairs++;
            }
        }

        DestroyFullBitL2EstimatorCtx(&estimatorCtx);
        DestroyQueryQuantCode(&queryQuantCode);
        QueryQuantizerCtxDestroy(&queryQuantCtx);
    }

    double denom = (double)data_count * (double)query_count;
    double meanAbsErr = absErrSum / denom;
    double meanRelErr = relErrSum / denom;

    printf("Data count: %d\n", data_count);
    printf("Query count: %d\n", query_count);
    printf("Loop count: %d\n", loop_count);
    printf("Dim: %zu, numBits: %zu\n", dim, numBits);
    printf("Seed: %u\n", seed);
    printf("Init time (sum): %.3f ms\n", init_time_ms);
    printf("Estimate time (sum): %.3f ms\n", estimate_time_ms);
    if (loop_count > 1) {
        printf("Loop unstable pairs: %zu\n", loop_unstable_pairs);
        printf("Loop max abs diff: %.6f\n", loop_max_abs_diff);
    }
    printf("Mean abs error: %.6f\n", (float)meanAbsErr);
    printf("Mean rel error: %.6f\n", (float)meanRelErr);
    printf("Max abs error: %.6f\n", (float)maxAbsErr);
    printf("Max rel error: %.6f\n", (float)maxRelErr);

    for (int i = 0; i < data_count; ++i) {
        if (data_items[i].dataCaqCode && data_items[i].dataCaqCode->storedCodes) {
            free(data_items[i].dataCaqCode->storedCodes);
            data_items[i].dataCaqCode->storedCodes = NULL;
        }
        DestroyCaqQuantCode(&data_items[i].dataCaqCode);
        free(data_items[i].data);
    }
    free(data_items);
    for (int q = 0; q < query_count; ++q) {
        free(query_items[q].query);
    }
    free(query_items);
    CaqQuantizerDestroy(&quantizerCtx);
    free(centroid);
    free(data);
    free(query);
    return 0;
}
