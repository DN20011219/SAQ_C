#include <stdio.h>

#include "encoder.h"

void encodeExample(const float *vector, size_t dim, size_t numBits) {
    CaqEncodeConfig *ccfg;
    CreateCaqQuantConfig(dim, numBits, &ccfg);
    CaqQuantCodeT *caqCode = NULL;
    CreateCaqQuantCode(&caqCode, dim);
    if (caqCode != NULL) {
        Encode(vector, caqCode, ccfg);
        printf("[encodeExample] dim=%zu bits=%zu max=%.6f min=%.6f delta=%g rescale=%.6f\n",
               dim, numBits, caqCode->max, caqCode->min, caqCode->delta, caqCode->rescaleFactor);
        printf("[encodeExample] orig L2=%.6f quant L2=%g ip=%g\n",
               caqCode->oriVecL2Norm, caqCode->quantizedVectorL2Sqr, caqCode->oriVecQuantVecIp);
        DestroyCaqQuantCode(&caqCode);
    }
    
    DestroyCaqQuantConfig(&ccfg);
    DestroyCaqQuantCode(&caqCode);
}

int main() {
    size_t dim = 4;
    float vec[4] = {0.1f, -0.3f, 0.5f, -0.2f};
    encodeExample(vec, dim, 9);
    printf("encodeExample executed.\n");
    return 0;
}