#include "data_quantizer.h"
#include "query_quantizer.h"
#include "rotator.h"

#include <time.h>


int main(void)
{
    QueryQuantizerCtxT *ctx = NULL;
    size_t dim = 16;
    float *centroid = (float *)malloc(sizeof(float) * dim);
    float *rotatorMatrix = NULL;
    createRotatorMatrix(&rotatorMatrix, dim);

    QueryQuantizerCtxInit(&ctx, dim, centroid, rotatorMatrix);

    QueryQuantCodeT *code = NULL;
    float *inputVector = (float *)malloc(sizeof(float) * dim);
    for (size_t i = 0; i < dim; ++i) {
        inputVector[i] = (float)(rand()) / RAND_MAX;
    }

    QuantizeQueryVector(ctx, inputVector, &code);

    DestroyQueryQuantCode(&code);
    QueryQuantizerCtxDestroy(&ctx);
    free(centroid);
    free(rotatorMatrix);
    free(inputVector);

    return 0;
}