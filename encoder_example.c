#include "encoder.h"
#include <stdio.h>

void encodeExample(const float *vector, size_t dim, size_t numBits) {
    CaqEncodeConfig *ccfg;
    createCaqQuantConfig(dim, numBits, &ccfg);
    CaqQuantCodeT *caqCode = NULL;
    createCaqQuantCode(&caqCode, dim);
    if (caqCode != NULL) {
        encode(vector, caqCode, ccfg);
        printf("[encodeExample] dim=%zu bits=%zu max=%.6f min=%.6f delta=%g rescale=%.6f\n",
               dim, numBits, caqCode->max, caqCode->min, caqCode->delta, caqCode->rescaleFactor);
        printf("[encodeExample] orig L2=%.6f quant L2=%g ip=%g\n",
               caqCode->oriVecL2Norm, caqCode->quantizedVectorL2Sqr, caqCode->oriVecQuantVecIp);
        destroyCaqQuantCode(&caqCode);
    }
    destroyCaqQuantConfig(&ccfg);
}

int main() {
    size_t dim = 4;
    float vec[4] = {0.1f, -0.3f, 0.5f, -0.2f};
    encodeExample(vec, dim, 9);
    printf("encodeExample executed.\n");
    return 0;
}