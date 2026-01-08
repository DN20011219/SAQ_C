#ifndef CLUSTER_H
#define CLUSTER_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include "data_quantizer.h"

typedef struct {
    int cid;                       // cluster id
    size_t dim;                    // dimension
    float *centroid;               // centroid pointer (size dim)
    int *ids;                      // vector IDs assigned to this cluster
    size_t size;                   // number of vectors in postings
    size_t cap;                    // capacity for postings

    // Quantized codes per vector (aligned with ids)
    CaqOneBitQuantCodeT **oneBitCodes;
    CaqResBitQuantCodeT **resBitCodes;

    // Quantizer context bound to this centroid (owns rotator + encode config)
    CaqQuantizerCtxT *quantizerCtx;
} Cluster;

typedef struct {
    size_t dim;
    size_t K;          // number of clusters
    Cluster *clusters; // array size K
    float *sharedRotatorMatrix;           // optional shared rotator across clusters
    CaqEncodeConfig *sharedEncodeConfig;  // optional shared encode config across clusters
    bool ownsSharedRotatorMatrix;
    bool ownsSharedEncodeConfig;
} IVFIndex;

static inline void ClusterInit(Cluster *c, int cid, size_t dim, float *centroid, size_t numBits, bool useSeparateStorage) {
    c->cid = cid;
    c->dim = dim;
    c->centroid = centroid;
    c->ids = NULL;
    c->size = 0;
    c->cap = 0;
    c->oneBitCodes = NULL;
    c->resBitCodes = NULL;
    c->quantizerCtx = NULL;
    CaqQuantizerInit(&c->quantizerCtx, dim, centroid, numBits, useSeparateStorage);
}

static inline void ClusterInitShared(Cluster *c,
                                     int cid,
                                     size_t dim,
                                     float *centroid,
                                     size_t numBits,
                                     bool useSeparateStorage,
                                     float *sharedRotatorMatrix,
                                     CaqEncodeConfig *sharedEncodeConfig) {
    c->cid = cid;
    c->dim = dim;
    c->centroid = centroid;
    c->ids = NULL;
    c->size = 0;
    c->cap = 0;
    c->oneBitCodes = NULL;
    c->resBitCodes = NULL;
    c->quantizerCtx = NULL;
    CaqQuantizerInitShared(&c->quantizerCtx,
                           dim,
                           centroid,
                           numBits,
                           useSeparateStorage,
                           sharedRotatorMatrix,
                           sharedEncodeConfig);
}

static inline void ClusterReserve(Cluster *c, size_t newCap) {
    if (newCap <= c->cap) return;
    c->cap = newCap;
    c->ids = (int *)realloc(c->ids, sizeof(int) * c->cap);
    c->oneBitCodes = (CaqOneBitQuantCodeT **)realloc(c->oneBitCodes, sizeof(CaqOneBitQuantCodeT *) * c->cap);
    c->resBitCodes = (CaqResBitQuantCodeT **)realloc(c->resBitCodes, sizeof(CaqResBitQuantCodeT *) * c->cap);
}

static inline void ClusterAppend(Cluster *c, int vid, CaqQuantCodeT *fullCode, const CaqEncodeConfig *cfg) {
    if (c->size == c->cap) {
        size_t newCap = c->cap ? (c->cap * 2) : 64;
        ClusterReserve(c, newCap);
    }
    c->ids[c->size] = vid;
    CaqOneBitQuantCodeT *oneBit = NULL;
    CaqResBitQuantCodeT *resBit = NULL;
    CaqSeparateStoredCodes(cfg, fullCode, &oneBit, &resBit);
    c->oneBitCodes[c->size] = oneBit;
    c->resBitCodes[c->size] = resBit;
    c->size += 1;
}

static inline void ClusterDestroy(Cluster *c) {
    if (!c) return;
    for (size_t i = 0; i < c->size; ++i) {
        DestroyCaqOneBitQuantCode(&c->oneBitCodes[i]);
        DestroyCaqResBitQuantCode(&c->resBitCodes[i]);
    }
    free(c->ids);
    free(c->oneBitCodes);
    free(c->resBitCodes);
    c->ids = NULL;
    c->oneBitCodes = NULL;
    c->resBitCodes = NULL;
    c->size = c->cap = 0;
    CaqQuantizerDestroy(&c->quantizerCtx);
}

static inline void IVFIndexDestroy(IVFIndex *index) {
    if (!index || !index->clusters) return;
    for (size_t k = 0; k < index->K; ++k) {
        ClusterDestroy(&index->clusters[k]);
    }
    free(index->clusters);
    index->clusters = NULL;
    if (index->ownsSharedRotatorMatrix && index->sharedRotatorMatrix) {
        destroyRotatorMatrix(&index->sharedRotatorMatrix);
    }
    if (index->ownsSharedEncodeConfig && index->sharedEncodeConfig) {
        DestroyCaqQuantConfig(&index->sharedEncodeConfig);
    }
}

#endif // CLUSTER_H
