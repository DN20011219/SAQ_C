#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#include "cluster.h"
#include "estimator.h"
// simple throttled progress printer
static inline size_t progress_step(size_t total) {
    // Use finer granularity for very large totals
    size_t step = (total >= 100000) ? (total / 1000) : (total / 100); // ~0.1% or ~1%
    return step ? step : 1;
}
static inline void print_progress(const char *label, size_t done, size_t total) {
    double pct = total ? (100.0 * (double)done / (double)total) : 100.0;
    fprintf(stdout, "\r%s: %zu/%zu (%.1f%%)", label, done, total, pct);
    fflush(stdout);
    if (done >= total) fprintf(stdout, "\n");
}

// Simple fvecs loader: returns N vectors of dimension D (per-vector header format)
typedef struct {
    size_t n; // number of vectors
    size_t d; // dimension
    float *data; // contiguous: n * d
} FVecs;

static int read_fvecs(const char *path, FVecs *out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Failed to open %s\n", path); return 0; }

    // First pass: count vectors and dimension
    long start = ftell(f);
    int32_t d = 0;
    size_t n = 0;
    while (fread(&d, sizeof(int32_t), 1, f) == 1) {
        if (d <= 0) { fprintf(stderr, "Invalid dim in %s\n", path); fclose(f); return 0; }
        if (fseek(f, (long)d * (long)sizeof(float), SEEK_CUR) != 0) { fclose(f); return 0; }
        ++n;
    }
    if (n == 0) { fprintf(stderr, "Empty fvecs file %s\n", path); fclose(f); return 0; }

    // Allocate and read
    out->n = n;
    out->d = (size_t)d; // assume constant per vector
    out->data = (float *)malloc(sizeof(float) * n * (size_t)d);
    if (!out->data) { fclose(f); return 0; }

    fseek(f, start, SEEK_SET);
    for (size_t i = 0; i < n; ++i) {
        int32_t di;
        if (fread(&di, sizeof(int32_t), 1, f) != 1) { fclose(f); return 0; }
        if ((size_t)di != out->d) { fprintf(stderr, "Dim mismatch at vec %zu in %s\n", i, path); fclose(f); return 0; }
        if (fread(out->data + i * out->d, sizeof(float), out->d, f) != out->d) { fclose(f); return 0; }
    }
    fclose(f);
    return 1;
}

static void free_fvecs(FVecs *fv) {
    if (fv && fv->data) { free(fv->data); fv->data = NULL; }
}

// ivecs loader: integer postings or groundtruth per query
typedef struct {
    size_t n; // number of rows
    size_t d; // ints per row
    int32_t *data; // contiguous: n * d
} IVecs;

static int read_ivecs(const char *path, IVecs *out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Failed to open %s\n", path); return 0; }
    long start = ftell(f);
    int32_t d = 0;
    size_t n = 0;
    while (fread(&d, sizeof(int32_t), 1, f) == 1) {
        if (d <= 0) { fprintf(stderr, "Invalid dim in %s\n", path); fclose(f); return 0; }
        if (fseek(f, (long)d * (long)sizeof(int32_t), SEEK_CUR) != 0) { fclose(f); return 0; }
        ++n;
    }
    if (n == 0) { fprintf(stderr, "Empty ivecs file %s\n", path); fclose(f); return 0; }
    out->n = n;
    out->d = (size_t)d;
    out->data = (int32_t *)malloc(sizeof(int32_t) * n * (size_t)d);
    if (!out->data) { fclose(f); return 0; }
    fseek(f, start, SEEK_SET);
    for (size_t i = 0; i < n; ++i) {
        int32_t di;
        if (fread(&di, sizeof(int32_t), 1, f) != 1) { fclose(f); return 0; }
        if ((size_t)di != out->d) { fprintf(stderr, "Dim mismatch at row %zu in %s\n", i, path); fclose(f); return 0; }
        if (fread(out->data + i * out->d, sizeof(int32_t), out->d, f) != out->d) { fclose(f); return 0; }
    }
    fclose(f);
    return 1;
}

static void free_ivecs(IVecs *iv) {
    if (iv && iv->data) { free(iv->data); iv->data = NULL; }
}

static inline float l2_to_centroid(const float *x, const float *c, size_t d) {
    float s = 0.0f;
    for (size_t i = 0; i < d; ++i) { float diff = x[i] - c[i]; s += diff * diff; }
    return s;
}

// Build IVF postings: assign each base vector to nearest centroid and quantize residual
static int build_ivf(const FVecs *base, const FVecs *centroids, size_t numBits, const IVecs *assignments, IVFIndex *index) {
    if (base->d != centroids->d) { fprintf(stderr, "Dim mismatch base(%zu) != centroids(%zu)\n", base->d, centroids->d); return 0; }
    index->dim = base->d;
    index->K = centroids->n;
    index->clusters = (Cluster *)calloc(index->K, sizeof(Cluster));
    if (!index->clusters) return 0;

    // Init clusters with shared rotator & encode config
    index->sharedRotatorMatrix = NULL;
    index->sharedEncodeConfig = NULL;
    index->ownsSharedRotatorMatrix = true;
    index->ownsSharedEncodeConfig = true;
    createRotatorMatrix(&index->sharedRotatorMatrix, index->dim);
    CreateCaqEncodeConfig(index->dim, numBits, true, &index->sharedEncodeConfig);

    for (size_t k = 0; k < index->K; ++k) {
        ClusterInitShared(&index->clusters[k],
                          (int)k,
                          index->dim,
                          centroids->data + k * index->dim,
                          numBits,
                          true,
                          index->sharedRotatorMatrix,
                          index->sharedEncodeConfig);
        ClusterReserve(&index->clusters[k], 64);
    }

    // Temporary full code per vector (reused per assignment)
    CaqQuantCodeT *fullCode = NULL;

    // Assign and quantize
    size_t pstep = progress_step(base->n);
    for (size_t i = 0; i < base->n; ++i) {
        if ((i == 0) || (i == 1) || ((i + 1) % pstep) == 0 || (i + 1) == base->n) {
            print_progress("Building IVF", i + 1, base->n);
        }
        const float *x = base->data + i * base->d;
        // assign by mapping if provided; else find nearest centroid
        size_t best_k = 0;
        if (assignments) {
            if (assignments->d != 1 || assignments->n != base->n) {
                fprintf(stderr, "Assignments ivecs must be (n=base.n, d=1). Got n=%zu d=%zu\n", assignments->n, assignments->d);
                return 0;
            }
            int32_t cid = assignments->data[i];
            if (cid < 0 || (size_t)cid >= index->K) { fprintf(stderr, "Invalid cid %d at row %zu\n", cid, i); return 0; }
            best_k = (size_t)cid;
        } else {
            float best_d = INFINITY;
            for (size_t k = 0; k < index->K; ++k) {
                float d = l2_to_centroid(x, index->clusters[k].centroid, index->dim);
                if (d < best_d) { best_d = d; best_k = k; }
            }
        }
        Cluster *cl = &index->clusters[best_k];
        // quantize residual with this cluster's context
        CaqQuantizeVector(cl->quantizerCtx, x, &fullCode);
        ClusterAppend(cl, (int)i, fullCode, cl->quantizerCtx->encodeConfig);
        DestroyCaqQuantCode(&fullCode);
    }
    return 1;
}

typedef struct { float dist; int id; } Result;

static int cmp_result(const void *a, const void *b) {
    const Result *ra = (const Result *)a; const Result *rb = (const Result *)b;
    if (ra->dist < rb->dist) return -1; if (ra->dist > rb->dist) return 1; return 0;
}

// Utility helpers for topK maintenance
// Binary-search helpers for sorted gate thresholds
static inline size_t lower_bound_float(const float *arr, size_t sz, float x) {
    size_t lo = 0, hi = sz;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (arr[mid] < x) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo; // first index with arr[idx] >= x
}

static inline float max_gate_value(const float *sorted_gates, size_t sz) {
    if (sz == 0) return -INFINITY;
    // Since array is kept sorted in non-decreasing order, max is the last element.
    // Use a binary search (lower_bound) to conform to the requested approach.
    size_t idx = lower_bound_float(sorted_gates, sz, FLT_MAX);
    return sorted_gates[(idx == 0) ? 0 : (idx - 1)];
}

static inline size_t argmax_dist(const Result *arr, size_t sz) {
    size_t idx = 0;
    float m = arr[0].dist;
    for (size_t i = 1; i < sz; ++i) { if (arr[i].dist > m) { m = arr[i].dist; idx = i; } }
    return idx;
}

// Select top-nprobe clusters by centroid distance
static void select_nprobe(const IVFIndex *index, const float *q, size_t nprobe, size_t *probe_ids) {
    Result *tmp = (Result *)malloc(sizeof(Result) * index->K);
    for (size_t k = 0; k < index->K; ++k) {
        tmp[k].id = (int)k;
        tmp[k].dist = l2_to_centroid(q, index->clusters[k].centroid, index->dim);
    }
    qsort(tmp, index->K, sizeof(Result), cmp_result);
    for (size_t i = 0; i < nprobe && i < index->K; ++i) probe_ids[i] = (size_t)tmp[i].id;
    free(tmp);
}

// Perform IVF search: for each selected cluster, compute CAQ distances and keep topK
static void ivf_search_one(const IVFIndex *index,
                           const float *query,
                           size_t numBits,
                           size_t nprobe,
                           size_t topK,
                           Result *out,
                           size_t *onebit_calls_out,
                           size_t *restbit_calls_out,
                           double *onebit_time_ms_out,
                           double *restbit_time_ms_out,
                           double *select_time_ms_out,
                           double *sort_time_ms_out,
                           double *init_onebit_time_ms_out,
                           double *init_restbit_time_ms_out,
                           double *gating_time_ms_out,
                           double *reservoir_time_ms_out,
                           double *destroy_onebit_time_ms_out,
                           double *destroy_restbit_time_ms_out,
                           double *scanner_time_ms_out) {
    size_t *probe_ids = (size_t *)malloc(sizeof(size_t) * nprobe);
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    select_nprobe(index, query, nprobe, probe_ids);
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double select_ms = (ts1.tv_sec - ts0.tv_sec) * 1000.0 + (ts1.tv_nsec - ts0.tv_nsec) / 1e6;

    // scanner context shared across estimators
    CaqScannerCtxT *scannerCtx = NULL;
    struct timespec sc0, sc1;
    clock_gettime(CLOCK_MONOTONIC, &sc0);
    CreateCaqScannerCtx(&scannerCtx);
    clock_gettime(CLOCK_MONOTONIC, &sc1);
    double scanner_time_ms = (sc1.tv_sec - sc0.tv_sec) * 1000.0 + (sc1.tv_nsec - sc0.tv_nsec) / 1e6;

    // keep only current topK; gate with 1-bit distance before computing rest-bit
    size_t sz = 0; // current number of kept candidates
    Result *cands = (Result *)malloc(sizeof(Result) * topK);
    // Maintain gates in two forms:
    // - gates_by_idx: 1-bit distance aligned with `cands` index (for replacement updates)
    // - sorted_gates: 1-bit distances kept sorted ascending for binary-search threshold
    float *gates_by_idx = (float *)malloc(sizeof(float) * topK);
    float *sorted_gates = (float *)malloc(sizeof(float) * topK);

    size_t onebit_calls = 0;
    size_t restbit_calls = 0;
    double onebit_time_ms = 0.0;
    double restbit_time_ms = 0.0;
    double init_onebit_time_ms = 0.0;
    double init_restbit_time_ms = 0.0;
    double gating_time_ms = 0.0;
    double reservoir_time_ms = 0.0;
    double destroy_onebit_time_ms = 0.0;
    double destroy_restbit_time_ms = 0.0;

    for (size_t pi = 0; pi < nprobe; ++pi) {
        Cluster const *cl = &index->clusters[probe_ids[pi]];
        // create estimator context for this centroid
        OneBitL2CaqEstimatorCtxT *estCtx = NULL;
        struct timespec ti0, ti1;
        clock_gettime(CLOCK_MONOTONIC, &ti0);
        OneBitCaqEstimatorInit(&estCtx,
                               index->dim,
                               QUERY_QUANTIZER_NUM_BITS,
                               cl->quantizerCtx->rotatorMatrix,
                               cl->quantizerCtx->centroid,
                               (float *)query,
                               scannerCtx);
        clock_gettime(CLOCK_MONOTONIC, &ti1);
        init_onebit_time_ms += (ti1.tv_sec - ti0.tv_sec) * 1000.0 + (ti1.tv_nsec - ti0.tv_nsec) / 1e6;
        RestBitL2EstimatorCtxT *restCtx = NULL;
        clock_gettime(CLOCK_MONOTONIC, &ti0);
        CreateRestBitL2EstimatorCtx(&restCtx,
                                    index->dim,
                                    numBits,
                                    estCtx->queryQuantCtx,
                                    estCtx->queryQuantCode,
                                    scannerCtx);
        clock_gettime(CLOCK_MONOTONIC, &ti1);
        init_restbit_time_ms += (ti1.tv_sec - ti0.tv_sec) * 1000.0 + (ti1.tv_nsec - ti0.tv_nsec) / 1e6;

        // scan postings
        for (size_t j = 0; j < cl->size; ++j) {
            float d1 = 0.0f, d2 = 0.0f;
            struct timespec tb0, tb1;
            clock_gettime(CLOCK_MONOTONIC, &tb0);
            OneBitCaqEstimateDistance(estCtx, cl->oneBitCodes[j], &d1);
            clock_gettime(CLOCK_MONOTONIC, &tb1);
            onebit_time_ms += (tb1.tv_sec - tb0.tv_sec) * 1000.0 + (tb1.tv_nsec - tb0.tv_nsec) / 1e6;
            ++onebit_calls;

            // Gating: only compute rest-bit distance if 1-bit distance could enter topK
            struct timespec tg0, tg1;
            if (sz >= topK) {
                clock_gettime(CLOCK_MONOTONIC, &tg0);
                float gate_thresh = max_gate_value(sorted_gates, sz);
                int skip = (d1 >= gate_thresh);
                clock_gettime(CLOCK_MONOTONIC, &tg1);
                gating_time_ms += (tg1.tv_sec - tg0.tv_sec) * 1000.0 + (tg1.tv_nsec - tg0.tv_nsec) / 1e6;
                if (skip) { continue; }
            }

            clock_gettime(CLOCK_MONOTONIC, &tb0);
            ResBitCaqEstimateDistance(restCtx, cl->resBitCodes[j], &d2);
            clock_gettime(CLOCK_MONOTONIC, &tb1);
            restbit_time_ms += (tb1.tv_sec - tb0.tv_sec) * 1000.0 + (tb1.tv_nsec - tb0.tv_nsec) / 1e6;
            ++restbit_calls;

            struct timespec tr0, tr1;
            clock_gettime(CLOCK_MONOTONIC, &tr0);
            if (sz < topK) {
                cands[sz].id = cl->ids[j];
                cands[sz].dist = d2;
                // record gate by candidate index
                gates_by_idx[sz] = d1;
                // insert gate into sorted_gates via binary search
                size_t pos = lower_bound_float(sorted_gates, sz, d1);
                memmove(sorted_gates + pos + 1, sorted_gates + pos, (sz - pos) * sizeof(float));
                sorted_gates[pos] = d1;
                ++sz;
            } else {
                // Replace worst current candidate by refined distance
                size_t worst_idx = argmax_dist(cands, sz);
                if (d2 < cands[worst_idx].dist) {
                    cands[worst_idx].id = cl->ids[j];
                    cands[worst_idx].dist = d2;
                    // update gates structures: remove old gate and insert new gate using binary search
                    float old_gate = gates_by_idx[worst_idx];
                    if (old_gate != d1) {
                        // remove one instance of old_gate from sorted_gates
                        size_t pos_old = lower_bound_float(sorted_gates, sz, old_gate);
                        if (pos_old < sz && sorted_gates[pos_old] == old_gate) {
                            memmove(sorted_gates + pos_old, sorted_gates + pos_old + 1, (sz - pos_old - 1) * sizeof(float));
                        } else {
                            // fallback: linear search if duplicates or precision issues
                            size_t k = 0;
                            for (; k < sz; ++k) { if (sorted_gates[k] == old_gate) break; }
                            if (k < sz) {
                                memmove(sorted_gates + k, sorted_gates + k + 1, (sz - k - 1) * sizeof(float));
                            }
                        }
                        // insert new gate
                        size_t pos_new = lower_bound_float(sorted_gates, sz - 1, d1);
                        memmove(sorted_gates + pos_new + 1, sorted_gates + pos_new, ((sz - 1) - pos_new) * sizeof(float));
                        sorted_gates[pos_new] = d1;
                    }
                    gates_by_idx[worst_idx] = d1;
                }
            }
            clock_gettime(CLOCK_MONOTONIC, &tr1);
            reservoir_time_ms += (tr1.tv_sec - tr0.tv_sec) * 1000.0 + (tr1.tv_nsec - tr0.tv_nsec) / 1e6;
        }
        struct timespec td0, td1;
        clock_gettime(CLOCK_MONOTONIC, &td0);
        DestroyRestBitL2EstimatorCtx(&restCtx);
        clock_gettime(CLOCK_MONOTONIC, &td1);
        destroy_restbit_time_ms += (td1.tv_sec - td0.tv_sec) * 1000.0 + (td1.tv_nsec - td0.tv_nsec) / 1e6;

        clock_gettime(CLOCK_MONOTONIC, &td0);
        OneBitCaqEstimatorDestroy(&estCtx);
        clock_gettime(CLOCK_MONOTONIC, &td1);
        destroy_onebit_time_ms += (td1.tv_sec - td0.tv_sec) * 1000.0 + (td1.tv_nsec - td0.tv_nsec) / 1e6;
    }

    // pick topK by distance
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    qsort(cands, sz, sizeof(Result), cmp_result);
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double sort_ms = (ts1.tv_sec - ts0.tv_sec) * 1000.0 + (ts1.tv_nsec - ts0.tv_nsec) / 1e6;
    size_t ret = (sz < topK) ? sz : topK;
    for (size_t i = 0; i < ret; ++i) out[i] = cands[i];
    for (size_t i = ret; i < topK; ++i) { out[i].id = -1; out[i].dist = INFINITY; }

    clock_gettime(CLOCK_MONOTONIC, &sc0);
    DestroyCaqScannerCtx(&scannerCtx);
    clock_gettime(CLOCK_MONOTONIC, &sc1);
    scanner_time_ms += (sc1.tv_sec - sc0.tv_sec) * 1000.0 + (sc1.tv_nsec - sc0.tv_nsec) / 1e6;
    free(gates_by_idx);
    free(sorted_gates);
    free(cands);
    free(probe_ids);

    if (onebit_calls_out) *onebit_calls_out = onebit_calls;
    if (restbit_calls_out) *restbit_calls_out = restbit_calls;
    if (onebit_time_ms_out) *onebit_time_ms_out = onebit_time_ms;
    if (restbit_time_ms_out) *restbit_time_ms_out = restbit_time_ms;
    if (select_time_ms_out) *select_time_ms_out = select_ms;
    if (sort_time_ms_out) *sort_time_ms_out = sort_ms;
    if (init_onebit_time_ms_out) *init_onebit_time_ms_out = init_onebit_time_ms;
    if (init_restbit_time_ms_out) *init_restbit_time_ms_out = init_restbit_time_ms;
    if (gating_time_ms_out) *gating_time_ms_out = gating_time_ms;
    if (reservoir_time_ms_out) *reservoir_time_ms_out = reservoir_time_ms;
    if (destroy_onebit_time_ms_out) *destroy_onebit_time_ms_out = destroy_onebit_time_ms;
    if (destroy_restbit_time_ms_out) *destroy_restbit_time_ms_out = destroy_restbit_time_ms;
    if (scanner_time_ms_out) *scanner_time_ms_out = scanner_time_ms;
}
static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s <dataset> <C>\n", prog);
        fprintf(stderr, "Usage: %s <dataset> <C> [numBits] [nprobe] [topK]\n", prog);
    fprintf(stderr, "  Auto paths: data/<dataset>/<dataset>_base.fvecs\n");
    fprintf(stderr, "              data/<dataset>/<dataset>_query.fvecs\n");
    fprintf(stderr, "              data/<dataset>/<dataset>_groundtruth.ivecs\n");
    fprintf(stderr, "              data/<dataset>/<dataset>_centroid_<C>.fvecs\n");
    fprintf(stderr, "              data/<dataset>/<dataset>_cluster_id_<C>.ivecs\n");
}

static char *join3(const char *a, const char *b, const char *c) {
    size_t la = strlen(a), lb = strlen(b), lc = strlen(c);
    char *s = (char *)malloc(la + lb + lc + 1);
    memcpy(s, a, la);
    memcpy(s + la, b, lb);
    memcpy(s + la + lb, c, lc);
    s[la + lb + lc] = '\0';
    return s;
}

static char *fmt_centroid(const char *ds, int C) {
    // data/<ds>/<ds>_centroid_<C>.fvecs
    char buf[64];
    snprintf(buf, sizeof(buf), "_centroid_%d.fvecs", C);
    char *p1 = join3("data/", ds, "/");
    char *p2 = join3(p1, ds, buf);
    free(p1);
    return p2;
}
static char *fmt_assign(const char *ds, int C) {
    // data/<ds>/<ds>_cluster_id_<C>.ivecs
    char buf[64];
    snprintf(buf, sizeof(buf), "_cluster_id_%d.ivecs", C);
    char *p1 = join3("data/", ds, "/");
    char *p2 = join3(p1, ds, buf);
    free(p1);
    return p2;
}
static char *fmt_simple(const char *ds, const char *suffix) {
    // data/<ds>/<ds>_<suffix>
    char *p1 = join3("data/", ds, "/");
    char *p2 = join3(p1, ds, "_");
    char *p3 = join3(p2, suffix, "");
    free(p1); free(p2);
    return p3;
}

int main(int argc, char **argv) {
    if (argc < 3) { usage(argv[0]); return 1; }
    const char *dataset = argv[1];
    int C = atoi(argv[2]);
    if (C <= 0) { fprintf(stderr, "Invalid C: %s\n", argv[2]); return 1; }
        // Defaults, optionally overridden by CLI
        size_t numBits = 9;
        size_t nprobe = 1;
        size_t topK = 100;
        if (argc >= 4) {
            size_t v = (size_t)atoi(argv[3]);
            if (v > 0) numBits = v;
        }
        if (argc >= 5) {
            size_t v = (size_t)atoi(argv[4]);
            if (v > 0) nprobe = v;
        }
        if (argc >= 6) {
            size_t v = (size_t)atoi(argv[5]);
            if (v > 0) topK = v;
        }
    // Derived paths
    char *basePath = fmt_simple(dataset, "base.fvecs");
    char *queryPath = fmt_simple(dataset, "query.fvecs");
    char *gtPathStr = fmt_simple(dataset, "groundtruth.ivecs");
    char *centPath = fmt_centroid(dataset, C);
    char *assignPathStr = fmt_assign(dataset, C);

    // Print config summary to avoid mis-specified parameters
    fprintf(stdout, "Config:\n");
    fprintf(stdout, "  dataset=%s C=%d numBits=%zu nprobe=%zu topK=%zu\n", dataset, C, numBits, nprobe, topK);
    fprintf(stdout, "Paths:\n");
    fprintf(stdout, "  base=%s\n", basePath);
    fprintf(stdout, "  query=%s\n", queryPath);
    fprintf(stdout, "  groundtruth=%s\n", gtPathStr);
    fprintf(stdout, "  centroids=%s\n", centPath);
    fprintf(stdout, "  assignments=%s\n", assignPathStr);

    FVecs base = {0}, queries = {0}, cents = {0};
    if (!read_fvecs(basePath, &base)) { free(basePath); free(queryPath); free(gtPathStr); free(centPath); free(assignPathStr); return 1; }
    if (!read_fvecs(queryPath, &queries)) { free_fvecs(&base); free(basePath); free(queryPath); free(gtPathStr); free(centPath); free(assignPathStr); return 1; }
    if (!read_fvecs(centPath, &cents)) { free_fvecs(&base); free_fvecs(&queries); free(basePath); free(queryPath); free(gtPathStr); free(centPath); free(assignPathStr); return 1; }
    if (base.d != queries.d || base.d != cents.d) {
        fprintf(stderr, "Dim mismatch: base=%zu queries=%zu cents=%zu\n", base.d, queries.d, cents.d);
        free_fvecs(&base); free_fvecs(&queries); free_fvecs(&cents); return 1;
    }

    // Print loaded dataset info
    fprintf(stdout, "Loaded:\n");
    fprintf(stdout, "  base: N=%zu D=%zu\n", base.n, base.d);
    fprintf(stdout, "  queries: N=%zu D=%zu\n", queries.n, queries.d);
    fprintf(stdout, "  centroids: K=%zu D=%zu\n", cents.n, cents.d);

    IVFIndex index = {0};
    struct timespec t0, t1;
    // Optional assignments mapping
    IVecs assign = {0};
    const char *assignPath = assignPathStr;
    if (assignPath) {
        if (!read_ivecs(assignPath, &assign)) {
            fprintf(stderr, "Warning: failed to read assignments from %s, will compute nearest centroids.\n", assignPath);
            assignPath = NULL;
        } else {
            fprintf(stdout, "Assignments loaded: N=%zu D=%zu\n", assign.n, assign.d);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (!assignPath) {
        fprintf(stdout, "Building IVF by nearest centroids (may be slow: N*K*D).\n");
    } else {
        fprintf(stdout, "Building IVF using assignments mapping.\n");
    }
    if (!build_ivf(&base, &cents, numBits, assignPath ? &assign : NULL, &index)) {
        fprintf(stderr, "IVF build failed\n");
        free_fvecs(&base); free_fvecs(&queries); free_fvecs(&cents); return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double build_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    fprintf(stdout, "IVF built: N=%zu K=%zu D=%zu in %.3f ms\n", base.n, index.K, index.dim, build_ms);

    // Optional groundtruth
    IVecs gt = {0};
    const char *gtPath = gtPathStr;
    if (gtPath) {
        if (!read_ivecs(gtPath, &gt)) {
            fprintf(stderr, "Warning: failed to read groundtruth from %s, skipping recall evaluation.\n", gtPath);
            gtPath = NULL;
        } else if (gt.n != queries.n) {
            fprintf(stderr, "Warning: groundtruth rows (%zu) != queries (%zu), skipping recall evaluation.\n", gt.n, queries.n);
            free_ivecs(&gt);
            gtPath = NULL;
        }
    }

    // Run queries
    size_t ef = 1024 / 2;
    Result *top = (Result *)malloc(sizeof(Result) * ef);
    double recall_sum = 0.0;
    double lat_sum_ms = 0.0;     // running sum of per-query latency (ms)
    double lat_min_ms = INFINITY; // min latency observed
    double lat_max_ms = 0.0;      // max latency observed
    size_t onebit_calls_sum = 0;
    size_t restbit_calls_sum = 0;
    double onebit_time_sum_ms = 0.0;
    double restbit_time_sum_ms = 0.0;
    double select_time_sum_ms = 0.0;
    double sort_time_sum_ms = 0.0;
    double init_onebit_time_sum_ms = 0.0;
    double init_restbit_time_sum_ms = 0.0;
    double gating_time_sum_ms = 0.0;
    double reservoir_time_sum_ms = 0.0;
    double destroy_onebit_time_sum_ms = 0.0;
    double destroy_restbit_time_sum_ms = 0.0;
    double scanner_time_sum_ms = 0.0;
    size_t qstep = progress_step(queries.n);
    for (size_t qi = 0; qi < queries.n; ++qi) {
        size_t onebit_calls = 0;
        size_t restbit_calls = 0;
        double onebit_time_ms = 0.0;
        double restbit_time_ms = 0.0;
        double select_time_ms = 0.0;
        double sort_time_ms = 0.0;
        double init_onebit_time_ms = 0.0;
        double init_restbit_time_ms = 0.0;
        double gating_time_ms = 0.0;
        double reservoir_time_ms = 0.0;
        double destroy_onebit_time_ms = 0.0;
        double destroy_restbit_time_ms = 0.0;
        double scanner_time_ms = 0.0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        ivf_search_one(&index,
                       queries.data + qi * queries.d,
                       numBits,
                       nprobe,
                       ef,
                       top,
                       &onebit_calls,
                       &restbit_calls,
                       &onebit_time_ms,
                       &restbit_time_ms,
                       &select_time_ms,
                       &sort_time_ms,
                       &init_onebit_time_ms,
                       &init_restbit_time_ms,
                       &gating_time_ms,
                       &reservoir_time_ms,
                       &destroy_onebit_time_ms,
                       &destroy_restbit_time_ms,
                       &scanner_time_ms);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double q_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        lat_sum_ms += q_ms;
        if (q_ms < lat_min_ms) lat_min_ms = q_ms;
        if (q_ms > lat_max_ms) lat_max_ms = q_ms;
        onebit_calls_sum += onebit_calls;
        restbit_calls_sum += restbit_calls;
        onebit_time_sum_ms += onebit_time_ms;
        restbit_time_sum_ms += restbit_time_ms;
        select_time_sum_ms += select_time_ms;
        sort_time_sum_ms += sort_time_ms;
        init_onebit_time_sum_ms += init_onebit_time_ms;
        init_restbit_time_sum_ms += init_restbit_time_ms;
        gating_time_sum_ms += gating_time_ms;
        reservoir_time_sum_ms += reservoir_time_ms;
        destroy_onebit_time_sum_ms += destroy_onebit_time_ms;
        destroy_restbit_time_sum_ms += destroy_restbit_time_ms;
        scanner_time_sum_ms += scanner_time_ms;
        // fprintf(stdout, "Query %zu: time=%.3f ms\n", qi, q_ms);
        // for (size_t i = 0; i < topK; ++i) {
        //     fprintf(stdout, "  #%zu id=%d dist=%.6f\n", i, top[i].id, top[i].dist);
        // }

        // Recall@K per query if groundtruth is present
        if (gtPath) {
            size_t kgt = gt.d;
            size_t denom = (topK < kgt) ? topK : kgt;
            size_t hit = 0;
            int32_t *gtrow = gt.data + qi * kgt;
            for (size_t i = 0; i < topK; ++i) {
                int id = top[i].id;
                // linear search is fine for small K
                for (size_t j = 0; j < denom; ++j) {
                    if (gtrow[j] == id) { ++hit; break; }
                }
            }
            double r = denom ? ((double)hit / (double)denom) : 0.0;
            recall_sum += r;
        }
        if (((qi + 1) % qstep) == 0 || (qi + 1) == queries.n) {
            double avg_ms = lat_sum_ms / (double)(qi + 1);
            double avg_onebit = (double)onebit_calls_sum / (double)(qi + 1);
            double avg_restbit = (double)restbit_calls_sum / (double)(qi + 1);
            double avg_onebit_time = onebit_time_sum_ms / (double)(qi + 1);
            double avg_restbit_time = restbit_time_sum_ms / (double)(qi + 1);
            double avg_select_time = select_time_sum_ms / (double)(qi + 1);
            double avg_sort_time = sort_time_sum_ms / (double)(qi + 1);
            double avg_init_onebit_time = init_onebit_time_sum_ms / (double)(qi + 1);
            double avg_init_restbit_time = init_restbit_time_sum_ms / (double)(qi + 1);
            double avg_gating_time = gating_time_sum_ms / (double)(qi + 1);
            double avg_reservoir_time = reservoir_time_sum_ms / (double)(qi + 1);
            double avg_destroy_onebit_time = destroy_onebit_time_sum_ms / (double)(qi + 1);
            double avg_destroy_restbit_time = destroy_restbit_time_sum_ms / (double)(qi + 1);
            double avg_scanner_time = scanner_time_sum_ms / (double)(qi + 1);
            // Other = total - all measured components
            double other_ms = avg_ms - (avg_onebit_time + avg_restbit_time + avg_select_time + avg_sort_time + avg_init_onebit_time + avg_init_restbit_time + avg_gating_time + avg_reservoir_time + avg_destroy_onebit_time + avg_destroy_restbit_time + avg_scanner_time);
            fprintf(stdout, "  recall@%zu = %.4f\n", topK, recall_sum / (double)(qi + 1));
            fprintf(stdout, "  latency(ms): avg=%.3f min=%.3f max=%.3f\n", avg_ms, lat_min_ms, lat_max_ms);
            fprintf(stdout, "  calls(avg): onebit=%.1f restbit=%.1f\n", avg_onebit, avg_restbit);
            fprintf(stdout, "  time(ms avg per query): onebit=%.3f restbit=%.3f\n", avg_onebit_time, avg_restbit_time);
            fprintf(stdout, "  time(ms avg per query): select=%.3f sort=%.3f other=%.3f\n", avg_select_time, avg_sort_time, other_ms);
                fprintf(stdout, "  misc(ms avg per query): gating=%.3f reservoir=%.3f destroy1=%.3f destroyR=%.3f scanner=%.3f\n",
                    avg_gating_time, avg_reservoir_time, avg_destroy_onebit_time, avg_destroy_restbit_time, avg_scanner_time);
            fprintf(stdout, "  init(ms avg per query): onebit=%.3f restbit=%.3f\n", avg_init_onebit_time, avg_init_restbit_time);
            print_progress("Queries", qi + 1, queries.n);
        }
    }

    free(top);
    IVFIndexDestroy(&index);
    free_fvecs(&base);
    free_fvecs(&queries);
    free_fvecs(&cents);
    if (assignPath) free_ivecs(&assign);
    if (gtPath) {
        double mean_recall = recall_sum / (double)queries.n;
        fprintf(stdout, "Mean recall@%zu = %.4f\n", topK, mean_recall);
        free_ivecs(&gt);
    }
    free(basePath); free(queryPath); free(gtPathStr); free(centPath); free(assignPathStr);
    return 0;
}
